"""46M supervised iteration with endgame/promotion oversampling and resumable checkpoints.

This is the supervised bootstrap phase, not a self-play training loop.
"""
import argparse
import json
import math
import time
from pathlib import Path
import sys
import signal
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import torch
import torch.nn.functional as F
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from data.record import RECORD_DTYPE
from model.dataset import decode_batch, decode_targets
from model.net import (NetConfig, UniChessNet, count_params,
                       cfg_conflicts, legal_from_to_mask)


def save_atomic(obj, path):
    temp = path.with_suffix(path.suffix + '.tmp')
    torch.save(obj, temp)
    temp.replace(path)


class Pool:
    def __init__(self, paths, cache):
        self.paths = paths
        self.maps = [np.memmap(p, dtype=RECORD_DTYPE, mode='r') for p in paths]
        self.offsets = np.cumsum([0] + [len(m) for m in self.maps])
        self.total = int(self.offsets[-1])
        fingerprint = [(str(p), p.stat().st_size, p.stat().st_mtime_ns) for p in paths]
        manifest = cache.with_suffix('.json')
        expected = json.dumps(fingerprint)
        if cache.exists() and manifest.exists() and manifest.read_text() == expected:
            with np.load(cache) as d:
                self.endgame, self.promotion = d['endgame'], d['promotion']
        else:
            rng = np.random.default_rng(20260909)
            groups = [[], []]
            for shard, data in enumerate(self.maps):
                selected = [[], []]
                for start in range(0, len(data), 262144):
                    r = data[start:start+262144]
                    occ = np.asarray(r['occ_white'] | r['occ_black']).copy()
                    counts = np.unpackbits(occ.view(np.uint8).reshape(-1, 8), axis=1).sum(1)
                    # White pawns on ranks 6/7; black pawns on ranks 2/3.
                    advanced = (((r['pawns'] & r['occ_white']) & np.uint64(0x00FFFF0000000000)) != 0) | (((r['pawns'] & r['occ_black']) & np.uint64(0x0000000000FFFF00)) != 0)
                    for i, mask in enumerate([counts <= 10, advanced]):
                        idx = np.flatnonzero(mask) + start + self.offsets[shard]
                        selected[i].append(idx)
                for i in range(2):
                    idx = np.concatenate(selected[i])
                    if len(idx) > 50000:
                        idx = rng.choice(idx, 50000, replace=False)
                    groups[i].append(idx.astype(np.int64))
                print('indexed',data.filename,flush=True)
            self.endgame, self.promotion = [np.concatenate(g) for g in groups]
            np.savez(cache, endgame=self.endgame, promotion=self.promotion)
            manifest.write_text(expected)
        if not len(self.endgame) or not len(self.promotion):
            raise ValueError('Specialty sample pool empty')
        print(json.dumps({'samples':self.total,'endgame_pool':len(self.endgame),'promotion_pool':len(self.promotion)}),flush=True)

    def sample(self, rng, n, mode='mixed', device='cuda'):
        if mode == 'mixed':
            a, b = n//2, n//4
            idx = np.concatenate([rng.integers(self.total,size=a), rng.choice(self.endgame,size=b), rng.choice(self.promotion,size=n-a-b)])
            rng.shuffle(idx)
        elif mode == 'general': idx = rng.integers(self.total,size=n)
        else: idx = rng.choice(getattr(self,mode),size=n)
        shards = np.searchsorted(self.offsets,idx,side='right')-1
        result = np.empty(n,dtype=RECORD_DTYPE)
        for s in np.unique(shards):
            sel = shards==s
            result[sel] = self.maps[s][idx[sel]-self.offsets[s]]
        x = decode_batch(result)
        p,pr,w = decode_targets(result)
        tensors = [torch.from_numpy(t) for t in [x,p,pr,w]]
        tensors[0] = tensors[0].contiguous(memory_format=torch.channels_last)
        return [t.pin_memory() if device == 'cpu' else t.to(device) for t in tensors]


def loss_for(model, batch, legal_mask=False):
    """legal_mask: 策略 softmax 只在合法着法上归一（见 model/net.py 与 model/train.py）。

    **默认关**，与本文件其它默认值的取舍不同，理由是这个脚本支持 --resume：
    runs/iteration46_20260909 是 2026-09-09 起、可续训的在跑实验，中途换掉
    损失的归一化口径会让 loss 曲线在续训点断档，best.pt 的比较也就跨不过去。
    新 run 想用就显式加 --legal-mask；老 run 续训保持字节级一致。
    """
    x,p,pr,w=batch
    legal=legal_from_to_mask(x) if legal_mask else None
    with torch.autocast('cuda',dtype=torch.bfloat16):
        pl,prl,wl=model(x)
        has=p.sum(1)>0
        if legal is not None:
            pl=pl.float().masked_fill(~legal,-1e4)   # 不可用 -inf：0*-inf=NaN
        lp=(-(p*F.log_softmax(pl.float(),dim=1)).sum(1)*has).sum()/has.sum().clamp_min(1)
        lw=-(w*F.log_softmax(wl.float(),dim=1)).sum(1).mean()
        eligible=pr!=-100
        lpr=(F.cross_entropy(prl.float(),pr.clamp_min(0),reduction='none')*eligible).sum()/eligible.sum().clamp_min(1)
        loss=lp+lw+0.1*lpr
    return loss,lp,lw


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--data',default='data/shards_evals')
    ap.add_argument('--out',required=True)
    ap.add_argument('--steps',type=int,default=125000)
    ap.add_argument('--batch',type=int,default=128)
    ap.add_argument('--accum',type=int,default=8)
    ap.add_argument('--save-every',type=int,default=1000)
    ap.add_argument('--resume',action='store_true')
    ap.add_argument('--legal-mask',dest='legal_mask',action='store_true',
                    help='策略 softmax 只在合法着法上归一（新 run 建议开；默认关是为了不打断 --resume 的损失口径）')
    args=ap.parse_args()
    torch.set_num_threads(4)
    torch.manual_seed(20260909)
    torch.backends.cudnn.benchmark=True
    out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
    paths=sorted(Path(args.data).glob('*.bin'))
    if len(paths)<8:raise ValueError('Need >=8 shards for held-out split')
    train=Pool(paths[:-4],out/'train_pools.npz')
    valid=Pool(paths[-4:],out/'valid_pools.npz')
    cfg=NetConfig(blocks=24,filters=320)
    model=UniChessNet(cfg).cuda().to(memory_format=torch.channels_last)
    opt=torch.optim.AdamW(model.parameters(),lr=3e-4,weight_decay=1e-4,fused=True)
    rng=np.random.default_rng(20260909);step=0;best=float('inf')
    if args.resume:
        ck=torch.load(out/'latest.pt',map_location='cpu',weights_only=False)
        # 不能直接比字典：NetConfig 后加过字段，老 checkpoint 没有这些 key，
        # 直接相等判断会让每个旧 run 的 --resume 都挂掉，而结构其实没变。
        bad = cfg_conflicts(ck['cfg'], cfg)
        assert not bad, '网络结构与 checkpoint 不符：' + '；'.join(bad)
        assert ck['args']['steps']==args.steps and ck['args']['batch']*ck['args']['accum']==args.batch*args.accum
        model.load_state_dict(ck['model']);opt.load_state_dict(ck['optimizer'])
        for group in opt.param_groups:
            group['fused']=True;group['foreach']=None
        for state in opt.state.values():
            for key,value in state.items():
                if torch.is_tensor(value):
                    value=value.to('cuda')
                    if value.ndim==4:value=value.contiguous(memory_format=torch.channels_last)
                    state[key]=value
        step=ck['step'];best=ck['best'];rng.bit_generator.state=ck['rng']
        torch.set_rng_state(ck['torch_rng']);torch.cuda.set_rng_state_all(ck['cuda_rng'])
    elif (out/'latest.pt').exists():raise ValueError('Checkpoint exists; use --resume')
    config={'args':vars(args),'cfg':cfg.__dict__,'params':count_params(model)['TOTAL'],'train_shards':[str(p) for p in paths[:-4]],'heldout_shards':[str(p) for p in paths[-4:]],'performance_version':2,'phase':'supervised bootstrap from random initialization','mix':{'general':0.5,'<=10 pieces':0.25,'advanced pawns':0.25}}
    (out/'config.json').write_text(json.dumps(config,indent=2))
    print(json.dumps(config),flush=True)
    started=time.time();initial_step=step
    stop_requested=False
    def request_stop(signum, frame):
        nonlocal stop_requested
        stop_requested=True
    signal.signal(signal.SIGTERM,request_stop)
    signal.signal(signal.SIGINT,request_stop)
    executor=ThreadPoolExecutor(max_workers=1)
    def prepare(update):
        # Per-update RNG makes prefetch/checkpoint boundaries reproducible.
        local_rng=np.random.default_rng(20260909+update)
        return [train.sample(local_rng,args.batch,device='cpu') for _ in range(args.accum)]
    future=executor.submit(prepare,step)
    model.train()
    while step<args.steps:
        lr=3e-4*min(1,(step+1)/2000)*(0.05+0.95*0.5*(1+math.cos(math.pi*step/args.steps)))
        for group in opt.param_groups:group['lr']=lr
        opt.zero_grad(set_to_none=True);totals=torch.zeros(3,device='cuda')
        batches=future.result()
        if step+1<args.steps:future=executor.submit(prepare,step+1)
        for cpu_batch in batches:
            batch=[t.to('cuda',non_blocking=True) for t in cpu_batch]
            loss,lp,lw=loss_for(model,batch,args.legal_mask)
            (loss/args.accum).backward()
            totals += torch.stack([loss.detach(),lp.detach(),lw.detach()])
        norm=torch.nn.utils.clip_grad_norm_(model.parameters(),2.0)
        if not bool(torch.isfinite(norm) & torch.isfinite(totals).all()):raise FloatingPointError('non-finite gradients or loss')
        opt.step();step+=1
        if step==1 or step%10==0:
            totals=totals.cpu().tolist()
            record={'step':step,'loss':float(totals[0]/args.accum),'policy':float(totals[1]/args.accum),'value':float(totals[2]/args.accum),'lr':lr,'samples_per_sec':round((step-initial_step)*args.batch*args.accum/(time.time()-started),1),'gpu_peak_mib':round(torch.cuda.max_memory_allocated()/2**20)}
            print(json.dumps(record),flush=True)
            with (out/'metrics.jsonl').open('a') as f:f.write(json.dumps(record)+'\n')
        if stop_requested or step == 10 or step%args.save_every==0 or step==args.steps:
            model.eval();scores={}
            with torch.no_grad():
                for mode in ['general','endgame','promotion']:
                    vrng=np.random.default_rng(991)
                    scores[mode]=float(np.mean([loss_for(model,valid.sample(vrng,128,mode),args.legal_mask)[0].item() for _ in range(8)]))
            score=float(np.mean(list(scores.values())))
            improved=score<best;best=min(best,score)
            payload={'model':model.state_dict(),'cfg':cfg.__dict__,'step':step,'optimizer':opt.state_dict(),'best':best,'rng':rng.bit_generator.state,'torch_rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state_all(),'args':vars(args)}
            save_atomic(payload,out/'latest.pt')
            if improved:save_atomic({'model':model.state_dict(),'cfg':cfg.__dict__,'step':step},out/'best.pt')
            print(json.dumps({'checkpoint':step,'validation':scores,'best':best}),flush=True)
            model.train()
            if stop_requested:
                print('STOPPED_AFTER_CHECKPOINT',flush=True)
                break
    executor.shutdown(wait=True)
    if step>=args.steps: print('SUPERVISED_PHASE_COMPLETE',flush=True)

if __name__=='__main__':main()
