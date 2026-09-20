"""Continuous actors, teacher reanalysis, bounded replay, learner and promotion gate.
All model promotion is internal; the public website is not repointed by this module.
"""
import argparse, concurrent.futures, multiprocessing, gc, hashlib, json, math, os, queue, signal, threading, time
from pathlib import Path
import chess, chess.engine, numpy as np, torch
import torch.nn.functional as F
from autoloop.common import *
from core.encoding import encode,orient_move
from core.moves import move_to_index,move_to_promo_index
from data.record import RECORD_DTYPE,record_to_board
from engine.engine import UniChessEngine
from model.net import NetConfig,UniChessNet
from model.train_iteration_fast import Pool,loss_for,save_atomic
from search.mcts import MCTS,MCTSConfig
from autoloop.replay_buffer import ExperienceReplay,REGRESSION,is_holdout,position_key,replay_version,sample_key
STOP=threading.Event()
def _bounded_int(name, default, lower, upper):
    try:
        value = int(os.environ.get(name, default))
    except (TypeError, ValueError):
        value = default
    return max(lower, min(upper, value))

def handle_stop(*_):STOP.set()


def board_of(row):
    b=chess.Board(row['start'])
    for move in row['history']:b.push_uci(move)
    if b.fen()!=row['fen']:raise ValueError('History/FEN mismatch')
    return b

def load_engine(path):
    torch.backends.cudnn.benchmark=False
    engine=UniChessEngine(path,device='cuda',half=True)
    engine.model.to(memory_format=torch.channels_last)
    return engine

_ACTOR_REQUESTS=None
_ACTOR_RESPONSES=None

def actor_init(requests,responses):
    global _ACTOR_REQUESTS,_ACTOR_RESPONSES
    _ACTOR_REQUESTS=requests;_ACTOR_RESPONSES=responses
    torch.set_num_threads(1)
    signal.signal(signal.SIGTERM,handle_stop)

class ActorProxy:
    def __init__(self,index):self.index=index
    def __call__(self,boards):
        _ACTOR_REQUESTS.put((self.index,boards))
        ok,payload=_ACTOR_RESPONSES[self.index].get(timeout=180)
        if not ok:raise RuntimeError(payload)
        return payload

def actor_task(index,seed,version,simulations,max_plies):
    return game(ActorProxy(index),seed,version,simulations,max_plies)

class ProcessBatcher:
    def __init__(self,engine,requests,responses,batch_limit=256,batch_wait=.006):
        self.engine=engine;self.requests=requests;self.responses=responses;self.batch_limit=batch_limit;self.batch_wait=batch_wait;self.closed=False;self.calls=0
        self.batch_limit=max(64,int(batch_limit));self.batch_wait=max(.001,float(batch_wait))
        self.thread=threading.Thread(target=self.run,daemon=True);self.thread.start()
    def run(self):
        while not self.closed:
            try:first=self.requests.get(timeout=.1)
            except queue.Empty:continue
            requests=[first];n=len(first[1]);until=time.monotonic()+self.batch_wait
            while n<self.batch_limit and time.monotonic()<until:
                try:item=self.requests.get(timeout=.001);requests.append(item);n+=len(item[1])
                except queue.Empty:pass
            started=time.perf_counter()
            try:
                arrays=self.engine.evaluate_batch([b for _,bs in requests for b in bs]);offset=0
                for index,bs in requests:
                    self.responses[index].put((True,tuple(a[offset:offset+len(bs)] for a in arrays)));offset+=len(bs)
                self.calls+=1
                if self.calls%100==0:event('inference','batch',positions=n,requests=len(requests),seconds=time.perf_counter()-started,calls=self.calls)
            except Exception as e:
                for index,bs in requests:self.responses[index].put((False,repr(e)))
    def close(self):self.closed=True;self.thread.join(timeout=5)

_START_POOLS={}
def start_board(rng,heldout=False):
    # 25% opening, 50% <=10-piece endgame, 25% advanced-pawn positions.
    selection=rng.random()
    if selection<.25:
        b=chess.Board()
        for _ in range(int(rng.integers(4,13))):
            if b.is_game_over():break
            b.push(list(b.legal_moves)[int(rng.integers(b.legal_moves.count()))])
        return b
    if heldout not in _START_POOLS:
        paths=sorted((ROOT/'data/shards_evals').glob('*.bin'))
        paths=paths[-4:] if heldout else paths[:-4]
        maps=[np.memmap(p,dtype=RECORD_DTYPE,mode='r') for p in paths]
        cache=ROOT/'runs/iteration46_20260909'/('valid_pools.npz' if heldout else 'train_pools.npz')
        with np.load(cache) as z:groups={k:z[k] for k in ['endgame','promotion']}
        _START_POOLS[heldout]=(maps,np.cumsum([0]+[len(m) for m in maps]),groups)
    maps,offsets,groups=_START_POOLS[heldout]
    indices=groups['endgame' if selection<.75 else 'promotion']
    for _ in range(100):
        idx=int(indices[int(rng.integers(len(indices)))]);shard=int(np.searchsorted(offsets,idx,side='right')-1)
        b=record_to_board(maps[shard][idx-offsets[shard]])
        if b.is_valid() and not b.is_game_over(claim_draw=True):return b
    return chess.Board()

def outcome(b):
    result=b.outcome(claim_draw=False)
    if result:return result.result(),result.termination.name
    if b.is_repetition(3):return '1/2-1/2','CLAIMED_THREEFOLD'
    if b.is_fifty_moves():return '1/2-1/2','CLAIMED_FIFTY_MOVES'
    return None,None

def _result_wdl(fen, result):
    if result not in ('1-0', '0-1', '1/2-1/2'):
        return None
    white = 1 if result == '1-0' else (-1 if result == '0-1' else 0)
    turn = chess.Board(fen).turn
    z = white if turn else -white
    return [int(z == 1), int(z == 0), int(z == -1)]


def training_rows(summary, source='small_selfplay', actor_role=None):
    """Convert a completed search trajectory into AlphaZero-style targets."""
    result = summary.get('result')
    label_result = result if result in ('1-0', '0-1', '1/2-1/2') else (
        '1/2-1/2' if summary.get('termination') == 'PLY_LIMIT' else None)
    if label_result is None:
        return []
    rows = []
    for original in summary.get('trajectory', []):
        policy = original.get('policy') or []
        target = _result_wdl(original['fen'], label_result)
        if not policy or target is None:
            continue
        row = dict(original)
        row['policy'] = [[str(move), float(prob)] for move, prob in policy]
        row['teacher_policy'] = list(row['policy'])
        row['teacher_wdl'] = target
        row['result_wdl'] = target
        row['source'] = source
        row['actor_role'] = actor_role or row.get('actor_role') or 'small'
        row['truncated'] = result is None
        row['importance'] = float(row.get('importance', 0.75 if result is None else 1.0)) * (1.0 + 0.35 * bool(row.get('promotion_threat')) + 0.15 * bool(row.get('repetition')))
        row['termination'] = summary.get('termination')
        rows.append(row)
    return rows

def game(evaluator,seed,version,simulations=800,max_plies=256):
    # MCTS consumes a batched evaluator.  ActorProxy already implements
    # __call__, while UniChessEngine exposes the same contract as
    # evaluate_batch; normalize both forms at the boundary.
    evaluator = getattr(evaluator, 'evaluate_batch', evaluator)
    if not callable(evaluator):
        raise TypeError(f'unsupported MCTS evaluator: {type(evaluator).__name__}')
    rng=np.random.default_rng(seed);b=start_board(rng);start=b.fen();b=chess.Board(start)
    search=MCTS(evaluator,MCTSConfig(simulations=simulations,batch_size=64,claim_draw=True),rng=rng)
    history=[];samples=[];trajectory=[];tree=None;t0=time.time();promotions=0;depths=[]
    while len(history)<max_plies and not STOP.is_set():
        result,reason=outcome(b)
        if result:break
        before=time.perf_counter();move,root=search.best_move(b,temperature=1 if len(history)<16 else 0,add_noise=True,root=tree)
        visits=[[m.uci(),float(p)] for m,p in search.visit_policy(root)]
        if not visits:raise RuntimeError('Nonterminal search produced no policy')
        threat=any(chess.square_rank(s) in (1,2,5,6) for s in b.pieces(chess.PAWN,not b.turn))
        row=dict(start=start,history=list(history),fen=b.fen(),policy=visits,q=search.root_value(root),actor=version,
                 root_visits=int(root.N.sum()),search_seconds=time.perf_counter()-before,root_entropy=float(-sum(p*math.log(max(p,1e-12)) for _,p in visits)),
                 pieces=chess.popcount(b.occupied),promotion_threat=threat,repetition=b.is_repetition(2),chosen=move.uci())
        row['priors']=[[m.uci(),float(v)] for m,v in zip(root.moves,root.P)]
        row['action_q']=[[m.uci(),float(w/n) if n else None] for m,w,n in zip(root.moves,root.W,root.N)]
        trajectory.append(row)
        event('search','move',seed=seed,ply=len(history),move=move.uci(),q=row['q'],visits=row['root_visits'],seconds=row['search_seconds'],entropy=row['root_entropy'],pieces=row['pieces'],**getattr(search,'last_metrics',{}))
        # Retain all actual moves in game record; sample expensive teacher positions.
        if len(history)%8==0 or threat or b.is_repetition(2):samples.append(row)
        promotions+=int(move.promotion is not None);tree=search.advance_root(root,move);b.push(move);history.append(move.uci())
    result,reason=outcome(b)
    if result is None:reason='INTERRUPTED' if STOP.is_set() else 'PLY_LIMIT'
    for row in samples:
        row['result_wdl']=_result_wdl(row['fen'],result)
        row['termination']=reason
    summary=dict(seed=seed,actor=version,start=start,moves=history,result=result,termination=reason,plies=len(history),promotions=promotions,seconds=time.time()-t0,samples=len(samples),trajectory=trajectory)
    return samples,summary

def role_selfplay(champion, generation, version, *, games=2, simulations=256, max_plies=256):
    """Give the PRO model its own outcome-labelled self-play stream."""
    engine = load_engine(champion)
    complete = 0
    rows_written = 0
    try:
        game_count = max(0, int(games))
        if not game_count:
            event('big', 'selfplay_training', generation=generation, games=0,
                  complete_games=0, rows=0, source='big_selfplay')
            return

        # Run several CPU-side MCTS actors against one parent-side GPU engine.
        # ActorProxy batches their leaf evaluations in ProcessBatcher, so the
        # large model sees useful batches instead of one board per call.
        ctx = multiprocessing.get_context('spawn')
        actor_count = min(game_count, _bounded_int('UNICHESS_BIG_ACTOR_COUNT', 8, 1, 16))
        batch_limit = _bounded_int('UNICHESS_BIG_INFERENCE_BATCH', 512, 64, 2048)
        try:
            batch_wait = max(.001, min(.020, float(
                os.environ.get('UNICHESS_BIG_INFERENCE_WAIT_MS', '8')) / 1000))
        except (TypeError, ValueError):
            batch_wait = .008
        requests = ctx.Queue(maxsize=max(16, actor_count * 2))
        responses = [ctx.Queue(maxsize=2) for _ in range(actor_count)]
        evaluator = ProcessBatcher(engine, requests, responses,
                                   batch_limit=batch_limit, batch_wait=batch_wait)
        event('big', 'actor_batch_config', actors=actor_count,
              batch_limit=batch_limit, batch_wait_ms=round(batch_wait * 1000, 3))
        try:
            with concurrent.futures.ProcessPoolExecutor(
                    max_workers=actor_count, mp_context=ctx,
                    initializer=actor_init, initargs=(requests, responses)) as pool:
                futures = [pool.submit(actor_task, index,
                                       700000 + generation * 100 + index,
                                       version, simulations, max_plies)
                           for index in range(game_count)]
                for future in concurrent.futures.as_completed(futures):
                    raw, summary = future.result()
                    summary['actor_role'] = 'big'
                    event('big', 'selfplay_game', **{k: v for k, v in summary.items()
                                                     if k not in ['moves', 'start', 'trajectory']})
                    write_bundle(STATE / 'games' / f'big-selfplay-{time.time_ns()}.json.gz', [summary])
                    labelled = training_rows(summary, source='big_selfplay', actor_role='big')
                    if labelled:
                        complete += 1
                        rows_written += len(labelled)
                        write_bundle(STATE / 'population' / f'big-selfplay-{time.time_ns()}.json.gz', labelled)
        finally:
            evaluator.close()
            requests.close()
            for response in responses:
                response.close()
        event('big', 'selfplay_training', generation=generation, games=game_count,
              complete_games=complete, rows=rows_written, source='big_selfplay')
    finally:
        del engine
        gc.collect()
        torch.cuda.empty_cache()

def teacher(rows):
    output=[]
    with chess.engine.SimpleEngine.popen_uci(str(ROOT/'tools/stockfish')) as sf:
        sf.configure({'Threads':2,'Hash':128})
        tb=ROOT/'data/raw/syzygy345'
        if tb.is_dir() and 'SyzygyPath' in sf.options:sf.configure({'SyzygyPath':str(tb)})
        for row in rows:
            if STOP.is_set():break
            b=board_of(row)
            infos=sf.analyse(b,chess.engine.Limit(nodes=200000),multipv=min(5,b.legal_moves.count()))
            if not infos:continue
            scores=np.array([i['score'].pov(b.turn).score(mate_score=10000) for i in infos],dtype=float)
            probs=np.exp(np.clip((scores-scores.max())/100,-30,0));probs/=probs.sum()
            wdl=infos[0]['score'].pov(b.turn).wdl(ply=b.ply());w=np.array([wdl.wins,wdl.draws,wdl.losses],dtype=float)/1000
            row.update(teacher_policy=[[i['pv'][0].uci(),float(p)] for i,p in zip(infos,probs)],teacher_wdl=w.tolist(),teacher_cp=float(scores[0]),teacher_depth=int(infos[0].get('depth',0)),teacher_nodes=int(infos[0].get('nodes',0)),tablebase_hits=int(infos[0].get('tbhits',0)),teacher='stockfish:'+str(sf.id.get('name')),holdout=is_holdout(b.fen()))
            # Score chosen move with a fixed-node verification; log opportunity loss.
            chosen=chess.Move.from_uci(row['chosen'])
            restricted=sf.analyse(b,chess.engine.Limit(nodes=100000),root_moves=[chosen])
            row['chosen_cp']=restricted['score'].pov(b.turn).score(mate_score=10000)
            row['cp_loss_estimate']=max(0,row['teacher_cp']-row['chosen_cp'])
            output.append(row)
    return output

def replay(role='small'):
    return ExperienceReplay(role).refresh()

def targets(rows):
    x=[];policy=np.zeros((len(rows),4096),np.float32);promo=np.zeros((len(rows),4),np.float32);wdl=[]
    for j,r in enumerate(rows):
        b=board_of(r);x.append(encode(b))
        # Search explores; Stockfish protects against mutually reinforcing blind spots.
        distribution={}
        for field,weight in [('policy',.25),('teacher_policy',.75)]:
            for u,p in r[field]:distribution[u]=distribution.get(u,0)+weight*p
        if r.get('big_policy') and r.get('big_verified'):
            distribution={u:p*.9 for u,p in distribution.items()}
            for u,p in r['big_policy']:distribution[u]=distribution.get(u,0)+.1*p
        for u,p in distribution.items():
            m=chess.Move.from_uci(u)
            if m not in b.legal_moves:raise ValueError('Illegal replay target')
            om=orient_move(m,b.turn);policy[j,move_to_index(om)]+=p
            pi=move_to_promo_index(om)
            if pi is not None:promo[j,pi]+=p
        if promo[j].sum()>0:promo[j]/=promo[j].sum()
        w=np.array(r['teacher_wdl'],np.float32)
        if r.get('result_wdl') is not None:w=.75*w+.25*np.array(r['result_wdl'])
        wdl.append(w)
    return [torch.from_numpy(a).pin_memory() for a in [np.array(x,np.float32),policy,promo,np.array(wdl,np.float32)]]

def replay_loss(model,batch):
    x,p,pr,w=[t.to('cuda',non_blocking=True) for t in batch]
    with torch.autocast('cuda',dtype=torch.bfloat16):pl,prl,wl=model(x.contiguous(memory_format=torch.channels_last))
    lp=-(p*F.log_softmax(pl.float(),1)).sum(1).mean();lv=-(w*F.log_softmax(wl.float(),1)).sum(1).mean()
    mask=pr.sum(1)>0;lpr=(-(pr*F.log_softmax(prl.float(),1)).sum(1)*mask).sum()/mask.sum().clamp_min(1)
    pred=wl.float().softmax(1)
    return lp+lv+.1*lpr,dict(policy=lp.detach(),value=lv.detach(),promotion=lpr.detach(),brier=((pred-w)**2).sum(1).mean().detach(),entropy=-(pred*pred.clamp_min(1e-9).log()).sum(1).mean().detach())

def train(role,champion,rows,generation,updates):
    checkpoint=STATE/'models'/f'{role}-recovery.pt';candidate=STATE/'models'/f'{role}-candidate.pt'
    seed_ck=torch.load(champion,map_location='cpu',weights_only=False)
    learner=STATE/'models'/f'{role}-learner.pt'
    if learner.exists():
        prior=torch.load(learner,map_location='cpu',weights_only=False)
        if prior.get('parent')==digest(champion):seed_ck=prior
    cfg=NetConfig(**seed_ck['cfg'])
    model=UniChessNet(cfg).cuda().to(memory_format=torch.channels_last);model.load_state_dict(seed_ck['model']);del seed_ck
    opt=torch.optim.AdamW(model.parameters(),lr=1e-5,weight_decay=1e-4,fused=True);step=0
    parent=digest(champion)
    if checkpoint.exists():
        ck=torch.load(checkpoint,map_location='cpu',weights_only=False)
        if ck.get('parent')==parent and ck.get('generation')==generation:
            model.load_state_dict(ck['model']);opt.load_state_dict(ck['optimizer']);step=ck['step']
            for s in opt.state.values():
                for k,v in s.items():
                    if torch.is_tensor(v):s[k]=v.cuda().contiguous(memory_format=torch.channels_last) if v.ndim==4 else v.cuda()
        del ck
    data_paths=sorted(Path('data/shards_evals').glob('*.bin'))
    pool=Pool(data_paths[:-4],ROOT/'runs/iteration46_20260909/train_pools.npz')
    prepared=time.time();compiled=targets(rows)
    event(role,'replay_compiled',rows=len(rows),seconds=time.time()-prepared,bytes=sum(t.numel()*t.element_size() for t in compiled))
    torch.backends.cudnn.benchmark=True
    model.train()
    # Tiny fresh replay batches must not overwrite the pretrained BN calibration.
    # Affine BN parameters still learn; only running means/variances are frozen.
    for module in model.modules():
        if isinstance(module,torch.nn.BatchNorm2d):module.eval()
    started=time.time();initial_step=step
    # Keep 75% original supervised anchors and 25% freshly verified self-play.
    batch=1024 if role=='big' else 512
    while step<updates and not STOP.is_set():
        rng=np.random.default_rng(generation*100000+step)
        opt.zero_grad(set_to_none=True)
        indices=torch.from_numpy(rng.integers(len(rows),size=batch//4))
        prepared_batch=[t.index_select(0,indices).pin_memory() for t in compiled]
        loss,metrics=replay_loss(model,prepared_batch);(.25*loss).backward()
        old=pool.sample(rng,batch*3//4);anchor,_,_=loss_for(model,old);(.75*anchor).backward()
        norm=torch.nn.utils.clip_grad_norm_(model.parameters(),2)
        if not bool(torch.isfinite(norm)&torch.isfinite(loss)&torch.isfinite(anchor)):raise FloatingPointError('nonfinite learner')
        opt.step();step+=1
        if step==initial_step+1 or step%10==0 or step==updates or STOP.is_set():
            event(role,'train',generation=generation,step=step,replay_loss=float(loss),anchor_loss=float(anchor),gradient_norm=float(norm),lr=opt.param_groups[0]['lr'],samples_per_second=(step-initial_step)*batch/max(.01,time.time()-started),cuda_allocated=torch.cuda.memory_allocated(),cuda_peak=torch.cuda.max_memory_allocated(),**{k:float(v) for k,v in metrics.items()})
        if step%50==0 or step==updates or STOP.is_set():
            save_atomic(dict(model=model.state_dict(),cfg=cfg.__dict__,optimizer=opt.state_dict(),step=step,generation=generation,parent=parent),checkpoint)
    if STOP.is_set():save_atomic(dict(model=model.state_dict(),cfg=cfg.__dict__,optimizer=opt.state_dict(),step=step,generation=generation,parent=parent),checkpoint)
    if step==updates:save_atomic(dict(model=model.state_dict(),cfg=cfg.__dict__,step=step,generation=generation,parent=parent),candidate)
    del model,opt,pool,compiled;gc.collect();torch.cuda.empty_cache()
    return candidate if step==updates else None

def diagnostics(engine):
    b=chess.Board(REGRESSION);mcts=MCTS(engine.evaluate_batch,MCTSConfig(simulations=3200,batch_size=64));move,root=mcts.best_move(b)
    return dict(promotion_block_move=move.uci(),promotion_block_pass=move.uci()=='d1a1',root_q=mcts.root_value(root))

def validation(engine):
    paths=sorted(Path('data/shards_evals').glob('*.bin'))
    pool=Pool(paths[-4:],ROOT/'runs/iteration46_20260909/valid_pools.npz');result={}
    with torch.no_grad():
        for mode in ['general','endgame','promotion']:
            rng=np.random.default_rng(77291);values=[]
            for _ in range(4):
                x,p,pr,w=pool.sample(rng,128,mode)
                with torch.autocast('cuda',dtype=torch.bfloat16):pl,prl,wl=engine.model(x)
                pred=wl.float().softmax(1);policy=pl.float().softmax(1)
                loss=-(p*F.log_softmax(pl.float(),1)).sum(1).mean()-(w*F.log_softmax(wl.float(),1)).sum(1).mean()
                values.append([float(loss),float(((pred-w)**2).sum(1).mean()),float((pred[:,0]-pred[:,2]-(w[:,0]-w[:,2])).abs().mean()),float((policy.argmax(1)==p.argmax(1)).float().mean())])
            result[mode]=dict(zip(['loss','wdl_brier','expected_value_mae','policy_top1'],np.mean(values,axis=0).tolist()))
    return result

def arena(candidate,champion,generation,pairs=24,simulations=400,max_plies=320):
    new=load_engine(candidate);old=load_engine(champion)
    new_diag=diagnostics(new);old_diag=diagnostics(old)
    nv=validation(new);ov=validation(old)
    validation_pass=all(nv[m]['loss']<=ov[m]['loss']*1.05 and nv[m]['wdl_brier']<=ov[m]['wdl_brier']*1.10 and nv[m]['expected_value_mae']<=ov[m]['expected_value_mae']*1.10 for m in nv)
    if not validation_pass:
        report=dict(generation=generation,accepted=False,validation_pass=False,reason='validation_regression',candidate_validation=nv,champion_validation=ov,candidate_diagnostics=new_diag,champion_diagnostics=old_diag)
        del new,old;gc.collect();torch.cuda.empty_cache();return report
    progress_path=STATE/(candidate.stem+'-arena-progress.json')
    identity=digest(candidate)+digest(champion)
    progress=read_json(progress_path,{})
    if progress.get('identity')!=identity:progress=dict(identity=identity,games=[],partial=None)
    games=progress['games']
    for pair in range(pairs):
        start=start_board(np.random.default_rng(880000+generation*1000+pair),heldout=True).fen()
        for color in [chess.WHITE,chess.BLACK]:
            if any(g['pair']==pair and g['candidate_white']==color for g in games):continue
            b=chess.Board(start);moves=[]
            partial=progress.get('partial')
            if partial and partial['pair']==pair and partial['candidate_white']==color:
                moves=list(partial['moves'])
                for u in moves:b.push_uci(u)
            while len(moves)<max_plies and not STOP.is_set():
                result,reason=outcome(b)
                if result:break
                engine=new if b.turn==color else old
                search=MCTS(engine.evaluate_batch,MCTSConfig(simulations=simulations,batch_size=64,claim_draw=True))
                move,_=search.best_move(b);b.push(move);moves.append(move.uci())
                if len(moves)%8==0:
                    progress['partial']=dict(pair=pair,candidate_white=color,moves=moves);atomic_json(progress_path,progress)
            if STOP.is_set():
                progress['partial']=dict(pair=pair,candidate_white=color,moves=moves);atomic_json(progress_path,progress);break
            result,reason=outcome(b)
            games.append(dict(start=start,moves=moves,result=result,termination=reason or 'PLY_LIMIT',candidate_white=color,pair=pair))
            progress['partial']=None;atomic_json(progress_path,progress)
        if STOP.is_set():break
    scores=[];unknown=sum(g['result'] is None for g in games)
    for pair in range(pairs):
        group=[g for g in games if g['pair']==pair]
        if len(group)!=2 or any(g['result'] is None for g in group):continue
        pair_scores=[]
        for g in group:
            white=1 if g['result']=='1-0' else (0 if g['result']=='0-1' else .5)
            pair_scores.append(white if g['candidate_white'] else 1-white)
        scores.append(float(np.mean(pair_scores)))
    mean=float(np.mean(scores)) if scores else 0
    lower=mean-math.sqrt(math.log(20)/(2*len(scores))) if scores else -1
    positive=sum(v>.5 for v in scores);negative=sum(v<.5 for v in scores);decisive=positive+negative
    sign_p=sum(math.comb(decisive,k) for k in range(positive,decisive+1))/(2**decisive) if decisive else 1.0
    accepted=(not STOP.is_set() and len(scores)==pairs and sign_p<.05 and mean>.52 and validation_pass and (new_diag['promotion_block_pass'] or not old_diag['promotion_block_pass']))
    report=dict(generation=generation,pairs=len(scores),unknown_games=unknown,score=mean,score_lower95_hoeffding=lower,paired_sign_p=sign_p,positive_pairs=positive,negative_pairs=negative,accepted=accepted,candidate_diagnostics=new_diag,champion_diagnostics=old_diag,candidate_validation=nv,champion_validation=ov,validation_pass=validation_pass)
    write_bundle(STATE/'games'/f'arena-{generation}-{candidate.stem}.json.gz',games)
    del new,old;gc.collect();torch.cuda.empty_cache()
    return report

def feedback(champion,rows,version):
    engine=load_engine(champion);out=[];created=time.time()
    from search.mcts import priors_from_policy
    max_rows=_bounded_int('UNICHESS_BIG_REANALYSIS_ROWS',2048,256,4096)
    deep_positions=_bounded_int('UNICHESS_BIG_REANALYSIS_POSITIONS',8,1,32)
    search_sims=_bounded_int('UNICHESS_BIG_REANALYSIS_SIMS',1600,128,6400)
    for i in range(0,min(len(rows),max_rows),128):
        if STOP.is_set():break
        part=rows[i:i+128];boards=[board_of(r) for r in part];p,pr,w=engine.evaluate_batch(boards)
        for r,b,pp,rr,ww in zip(part,boards,p,pr,w):
            moves,prob=priors_from_policy(b,pp,rr);teacher_best=r['teacher_policy'][0][0];big_best=moves[int(np.argmax(prob))].uci()
            r=dict(r);r.update(big_policy=[[m.uci(),float(v)] for m,v in zip(moves,prob)],big_wdl=ww.tolist(),big_version=version,big_created=created,big_verified=(big_best==teacher_best and abs(float(ww[0]-ww[2])-(r['teacher_wdl'][0]-r['teacher_wdl'][2]))<.25));out.append(r)
    hard=sorted(out,key=lambda r:r.get('cp_loss_estimate',0),reverse=True)[:deep_positions]
    for r in hard:
        if STOP.is_set():break
        b=board_of(r);search=MCTS(engine.evaluate_batch,MCTSConfig(simulations=search_sims,batch_size=64,claim_draw=True))
        move,root=search.best_move(b)
        r['big_policy']=[[m.uci(),float(p)] for m,p in search.visit_policy(root)]
        r['big_search_q']=search.root_value(root);r['big_search_sims']=search_sims
        r['big_verified']=move.uci()==r['teacher_policy'][0][0] and abs(r['big_search_q']-(r['teacher_wdl'][0]-r['teacher_wdl'][2]))<.25
    event('big','reanalysis',positions=len(out),deep_search_positions=sum('big_search_q' in r for r in out),verified=sum(r['big_verified'] for r in out),search_sims=search_sims,model=version)
    del engine;gc.collect();torch.cuda.empty_cache();return out

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--role',choices=['small','big'],required=True);ap.add_argument('--mode',choices=['combined','actor','learner'],default='combined');ap.add_argument('--once',action='store_true');ap.add_argument('--smoke',action='store_true');ap.add_argument('--smoke-updates',type=int,default=2);args=ap.parse_args();role=args.role
    lease=lock(role+'-'+args.mode);signal.signal(signal.SIGTERM,handle_stop);signal.signal(signal.SIGINT,handle_stop)
    torch.set_num_threads(4);torch.backends.cudnn.benchmark=True
    state_path=STATE/f'{role}-state.json';state=read_json(state_path,{'generation':0,'last_data':None})
    champion=STATE/'models'/f'{role}-champion.pt'
    with (STATE/(role+'-seed.lock')).open('a') as seed_lock:
        fcntl.flock(seed_lock,fcntl.LOCK_EX)
        if not champion.exists():
            seed=ROOT/('runs/stage1/ckpt_00187578.pt' if role=='small' else 'runs/iteration46_20260909/best.pt')
            ck=torch.load(seed,map_location='cpu',weights_only=False);save_atomic({'model':ck['model'],'cfg':ck['cfg'],'seed':str(seed)},champion);del ck
            event(role,'seed',path=str(seed),sha256=digest(champion),note='initial baseline, not claimed superior')
    while not STOP.is_set():
        try:
            storage=maintain_storage()
            if not storage['ok'] or (STATE/'PAUSE').exists():
                event(role,'paused_storage_or_manual',storage=storage);STOP.wait(30);continue
            if args.mode=='actor':state=read_json(state_path,state)
            generation=state['generation'];version=digest(champion)
            atomic_json(STATE/(f'{role}-actor-status.json' if args.mode=='actor' else f'{role}-status.json'),dict(time=time.time(),phase='generate' if role=='small' and args.mode!='learner' else 'await_replay',generation=generation,champion=version))
            if role=='small' and args.mode!='learner':
                import shutil
                frozen=STATE/'models/small-actor.pt';shutil.copy2(champion,frozen);version=digest(frozen)
                engine=load_engine(frozen);ctx=multiprocessing.get_context('spawn')
                actor_count=1 if args.smoke else _bounded_int('UNICHESS_ACTOR_COUNT',8,1,12)
                batch_limit=_bounded_int('UNICHESS_INFERENCE_BATCH',256,64,384)
                try:
                    batch_wait=max(.001,min(.020,float(os.environ.get('UNICHESS_INFERENCE_WAIT_MS','6'))/1000))
                except (TypeError,ValueError):
                    batch_wait=.006
                requests=ctx.Queue(maxsize=max(16,actor_count*2));responses=[ctx.Queue(maxsize=2) for _ in range(actor_count)]
                evaluator=ProcessBatcher(engine,requests,responses,batch_limit=batch_limit,batch_wait=batch_wait)
                event(role,'actor_batch_config',actors=actor_count,batch_limit=batch_limit,batch_wait_ms=round(batch_wait*1000,3))
                try:
                    with concurrent.futures.ProcessPoolExecutor(max_workers=actor_count,mp_context=ctx,initializer=actor_init,initargs=(requests,responses)) as pool:
                        futures=[pool.submit(actor_task,n,int(time.time())+n,version,32 if args.smoke else 800,8 if args.smoke else 256) for n in range(actor_count)]
                        for future in concurrent.futures.as_completed(futures):
                            raw,summary=future.result()
                            event(role,'selfplay_game',**{k:v for k,v in summary.items() if k not in ['moves','start','trajectory']})
                            write_bundle(STATE/'games'/f'selfplay-{time.time_ns()}.json.gz',[summary])
                            labelled=training_rows(summary, source='small_selfplay', actor_role='small')
                            if labelled:
                                write_bundle(STATE/'population'/f'selfplay-{time.time_ns()}.json.gz',labelled)
                                event(role,'selfplay_training',games=1,rows=len(labelled),source='small_selfplay',complete_games=1)
                            reviewed=teacher(raw)
                            if reviewed:
                                write_bundle(STATE/'replay'/f'batch-{time.time_ns()}.json.gz',reviewed)
                                event(role,'teacher_review',positions=len(reviewed),mean_cp_loss=float(np.mean([r['cp_loss_estimate'] for r in reviewed])),blunders_200cp=sum(r['cp_loss_estimate']>=200 for r in reviewed),promotion_threats=sum(r['promotion_threat'] for r in reviewed),endgames=sum(r['pieces']<=10 for r in reviewed),mean_depth=float(np.mean([r['teacher_depth'] for r in reviewed])))
                finally:
                    evaluator.close();requests.close()
                    for q in responses:q.close()
                del engine,evaluator;gc.collect();torch.cuda.empty_cache()
            if STOP.is_set():break
            if role=='big' and args.mode!='actor':
                try:
                    role_selfplay(champion, generation, version,
                                  games=_bounded_int('UNICHESS_BIG_SELFPLAY_GAMES',8,0,16),
                                  simulations=_bounded_int('UNICHESS_BIG_SELFPLAY_SIMS',256,32,1600),
                                  max_plies=_bounded_int('UNICHESS_BIG_SELFPLAY_PLIES',256,32,512))
                except Exception as selfplay_error:
                    # Selfplay is an input stream, not a reason to starve
                    # replay training or the layered worker.  Record the
                    # failure and continue with already available rows.
                    import traceback
                    event(role, 'selfplay_error', error=repr(selfplay_error),
                          traceback=traceback.format_exc())
            if STOP.is_set():break
            if args.mode=='actor':
                if args.once:break
                continue
            rows=replay(role)
            # Feedback is a replacement for matching records, only teacher-verified targets get weight.
            for p in sorted((STATE/'feedback').glob('*.gz'))[-8:]:
                newer={sample_key(r):r for r in read_bundle(p)}
                rows=[newer[sample_key(r)] if sample_key(r) in newer and newer[sample_key(r)].get('big_created',0)>=r.get('big_created',0) else r for r in rows]
            data_version=replay_version(rows)
            snapshot=STATE/'jobs'/f'{role}-active.json.gz'
            resuming=state.get('phase')=='training' and snapshot.exists()
            if not resuming and (len(rows)<(1 if args.smoke else 128) or ((args.mode!='combined' or role=='big') and data_version==state['last_data'])):
                if args.once:break
                STOP.wait(30);continue
            # Freeze the exact generation data to make interrupted PRO updates reproducible.
            snapshot=STATE/'jobs'/f'{role}-active.json.gz'
            if state.get('phase')=='training' and snapshot.exists():rows=read_bundle(snapshot);data_version=state['active_data']
            else:
                write_bundle(snapshot,rows)
                maximum=400 if role=='big' else 200
                updates=min(maximum,max(1,math.ceil(len(rows)*4/(256 if role=='big' else 128))))
                state.update(phase='training',active_data=data_version,active_updates=args.smoke_updates if args.smoke else updates);atomic_json(state_path,state)
            atomic_json(STATE/f'{role}-status.json',dict(time=time.time(),phase='training',generation=generation,rows=len(rows),champion=version))
            candidate=train(role,champion,rows,generation,state.get('active_updates',args.smoke_updates if args.smoke else (400 if role=='big' else 200)))
            if STOP.is_set() or candidate is None:break
            atomic_json(STATE/f'{role}-status.json',dict(time=time.time(),phase='arena',generation=generation,champion=version))
            if args.smoke:report={'accepted':False,'smoke':True,'generation':generation}
            else:report=arena(candidate,champion,generation,
                              pairs=_bounded_int('UNICHESS_BIG_ARENA_PAIRS',24,8,32),
                              simulations=_bounded_int('UNICHESS_BIG_ARENA_SIMS',400,64,800),
                              max_plies=_bounded_int('UNICHESS_BIG_ARENA_PLIES',320,64,512))
            event(role,'arena',**report);atomic_json(STATE/f'{role}-last-arena.json',report)
            if STOP.is_set():break
            if report['accepted']:
                # Fixed two-slot rollback; no unbounded checkpoint history.
                rollback=STATE/'models'/f'{role}-previous.pt'
                import shutil
                shutil.copy2(champion,rollback);os.replace(candidate,champion)
                event(role,'promoted',generation=generation,parent=version,champion=digest(champion))
                (STATE/'models'/f'{role}-learner.pt').unlink(missing_ok=True)
            elif report.get('validation_pass') and report.get('score',0)>=.45:
                # Continue a promising challenger without calling it the champion.
                import shutil
                shutil.copy2(candidate,STATE/'models'/f'{role}-learner.pt')
            if role=='big':
                atomic_json(STATE/f'{role}-status.json',dict(time=time.time(),phase='reanalysis',generation=generation,champion=digest(champion)))
                annotated=feedback(champion,rows,digest(champion))
                # Keep the third model family under the same governed PRO
                # process. It trains five independent piece-count models and
                # performs three-way mirrored cross-play.
                try:
                    from autoloop.layered import run_generation
                    layered_report=run_generation(
                        generation, annotated or rows,
                        small_checkpoint=STATE/'models/small-champion.pt',
                        big_checkpoint=champion,
                        smoke=args.smoke,
                    )
                    event('layered','worker_cycle',generation=generation,
                          phase=layered_report.get('phase'),
                          promoted_stages=sum(r.get('accepted',False)
                                               for r in layered_report.get('stage_reports',[])))
                except Exception as layered_error:
                    import traceback
                    event('layered','error',generation=generation,
                          error=repr(layered_error),traceback=traceback.format_exc())
                    atomic_json(STATE/'layered-status.json',
                                dict(time=time.time(),phase='error',
                                     generation=generation,error=repr(layered_error)))
            if STOP.is_set():break
            state.update(generation=generation+1,last_data=data_version,phase='ready');atomic_json(state_path,state)
            if args.once:break
        except Exception as e:
            import traceback
            event(role,'error',error=repr(e),traceback=traceback.format_exc());atomic_json(STATE/f'{role}-status.json',dict(time=time.time(),phase='error',error=repr(e)))
            if args.once:raise
            STOP.wait(60)
    event(role,'stopped',generation=state['generation'])
if __name__=='__main__':main()
