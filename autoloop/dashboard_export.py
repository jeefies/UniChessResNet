"""Read-only public telemetry export. Never publishes commands, paths, PIDs or tracebacks."""
import argparse,collections,functools,gzip,hashlib,json,math,time
from pathlib import Path
from autoloop.common import ROOT,STATE,atomic_json,read_json

FIELDS={'time','kind','generation','step','updates','stage','stage_name','stages','ready_stages','params','loss','policy','value','promotion','lr','samples_per_second','samples_per_sec','replay_loss','anchor_loss','gradient_norm','brier','wdl_brier','expected_value_mae','entropy','cuda_allocated','cuda_peak','gpu_peak_mib','positions','deep_search_positions','verified','mean_cp_loss','blunders_200cp','promotion_threats','endgames','mean_depth','result','termination','plies','promotions','seconds','samples','pairs','pairs_per_family','unknown_games','illegal_games','interrupted_or_cutoff','score','score_lower95_hoeffding','wilson_lower95','paired_sign_p','positive_pairs','negative_pairs','decisive_pairs','accepted','improved','validation_pass','reason','candidate_validation','champion_validation','rows','bytes','parent','champion','model','actor','actor_role','source','opponent','seed','visits','q','max_depth','network_positions','network_batches','collisions','requests','calls','games','completed_games','completion_rate','reliable','pairings','pair_count','simulations','promoted_stages','model_versions','strong_positive_signal','baseline_initialized','status','families'}
FIELDS.update({'fresh_rows','reservoir_rows','training_rows','unique_seen','stage_counts','difficult_rows','oldest_days'})
FIELDS.update({'actors','batch_limit','batch_wait_ms'})

def clean(row):return {k:v for k,v in row.items() if k in FIELDS and not (k=='model' and isinstance(v,str) and '/' in v)}
def events(path):
    rows=[]
    for p in sorted(path.parent.glob(path.name+'*')):
        if p.suffix=='.lock':continue
        try:
            with p.open() as f:
                for line in f:
                    try:rows.append(json.loads(line))
                    except (ValueError,TypeError):pass
        except FileNotFoundError:pass
    return sorted(rows,key=lambda r:r.get('time',0))
def sampled(rows,n=1200):
    if len(rows)<=n:return rows
    return [rows[round(i*(len(rows)-1)/(n-1))] for i in range(n)]
@functools.lru_cache(maxsize=512)
def game_file(name,mtime):
    with gzip.open(name,'rt') as f:
        content=f.read(32*1024**2+1)
    if len(content)>32*1024**2:return []
    rows=[json.loads(line) for line in content.splitlines()]
    for g in rows:
        g['trajectory']=[{k:r[k] for k in ['fen','q','root_visits','root_entropy','search_seconds','chosen'] if k in r} for r in g.get('trajectory',[])]
    return rows

def snapshot(role):
    rows=events(STATE/'metrics'/f'{role}.jsonl');games={}; listing=[];bad=0
    files=sorted((STATE/'games').glob('*.gz'),key=lambda p:p.stat().st_mtime,reverse=True)[:400]
    for p in files:
        try:
            for i,g in enumerate(game_file(str(p),p.stat().st_mtime_ns)):
                if not g.get('start') or not isinstance(g.get('moves'),list):continue
                ident=hashlib.sha256((role+p.name+str(i)+g['start']+' '.join(g['moves'])).encode()).hexdigest()[:24]
                detail={k:g[k] for k in ['start','moves','result','termination','candidate_white','pair','pairing','family','white','black','actor','seed','plies','seconds','promotions'] if k in g}
                detail.update(id=ident,role=role,type=('arena' if p.name.startswith('arena-') else 'crossplay' if p.name.startswith('joint-') else 'stockfish' if p.name.startswith('stockfish-') else 'selfplay'),time=p.stat().st_mtime)
                detail['search']=[{k:r[k] for k in ['fen','q','root_visits','root_entropy','search_seconds','chosen'] if k in r} for r in g.get('trajectory',[])]
                games[ident]=detail
                listing.append({k:v for k,v in detail.items() if k not in ['moves','search']})
                listing[-1]['plies']=len(g['moves'])
        except (ValueError,OSError,EOFError):bad+=1
    status=read_json(STATE/f'{role}-status.json',{})
    status={k:v for k,v in status.items() if k in ['time','phase','generation','champion','rows']}
    progress_path=STATE/(f'{role}-candidate-arena-progress.json' if role != 'layered' else 'layered-last-generation.json')
    progress=read_json(progress_path,{})
    progress_current=progress_path.exists() and progress_path.stat().st_mtime>=status.get('time',0)-1
    if not progress_current:progress={}
    partial=progress.get('partial') or {}
    status['arena_progress']={'available':progress_current,'completed_games':len(progress.get('games',[])),'target_games':48,'current_pair':partial.get('pair'),'current_plies':len(partial.get('moves',[]))}
    actors=read_json(STATE/'small-actor-status.json',{}) if role=='small' else {}
    actors={k:v for k,v in actors.items() if k in ['time','phase','generation','champion']}
    governor=read_json(STATE/'governor-status.json',{}) if role=='big' else {}
    gpu={k:v for k,v in governor.get('gpu',{}).items() if k!='processes'}
    if role=='small':
        try:
            import subprocess
            values=subprocess.check_output(['nvidia-smi','--query-gpu=utilization.gpu,memory.used,memory.free,power.draw,temperature.gpu','--format=csv,noheader,nounits'],text=True,timeout=5).splitlines()[0].split(',')
            gpu=dict(zip(['utilization','used_mib','free_mib','power_w','temperature_c'],map(float,values)))
        except Exception:gpu={}
    trains=[clean(r) for r in rows if r.get('kind')=='train']
    replays=[clean(r) for r in rows if r.get('kind')=='replay_buffer']
    arenas=[clean(r) for r in rows if r.get('kind')=='arena']
    outcomes=collections.Counter(r.get('result') or 'unfinished' for r in rows if r.get('kind')=='selfplay_game')
    termination=collections.Counter(r.get('termination','unknown') for r in rows if r.get('kind')=='selfplay_game')
    selfplay_events=[r for r in rows if r.get('kind')=='selfplay_training']
    crossplay_events=[r for r in rows if r.get('kind')=='crossplay_training']
    stockfish_events=[r for r in rows if r.get('kind')=='stockfish_crossplay']
    learning=dict(
        selfplay_games=sum(int(r.get('games', r.get('complete_games', 0))) for r in selfplay_events),
        selfplay_rows=sum(int(r.get('rows', 0)) for r in selfplay_events),
        crossplay_games=sum(int(r.get('games', 0)) for r in crossplay_events),
        crossplay_rows=sum(int(r.get('rows', 0)) for r in crossplay_events),
        stockfish_games=sum(int(r.get('games', 0)) for r in stockfish_events),
        stockfish_rows=sum(int(r.get('training_rows', r.get('rows', 0))) for r in stockfish_events),
        stockfish_enabled=any(r.get('enabled') for r in stockfish_events))
    bootstrap=events(ROOT/'runs/iteration46_20260909/metrics.jsonl')
    bootstrap=[clean(r) for r in bootstrap if 'step' in r]
    store=read_json(STATE/'storage.json',{})
    layered_status=read_json(STATE/'layered-status.json',{}) if role=='layered' else {}
    if role=='layered':
        last=read_json(STATE/'layered-last-generation.json',{})
        status.update({k:v for k,v in layered_status.items() if k in ['time','phase','generation','stage','stage_name','stages','ready_stages','params','crossplay','stockfish']})
        status['last_generation']=last.get('generation')
        status['last_generation_phase']=last.get('phase')
        status['stage_reports']=last.get('stage_reports',[])
        status['crossplay']=last.get('crossplay',status.get('crossplay',{}))
        status['stockfish']=last.get('stockfish',status.get('stockfish',{}))
        status['progress_eval']=last.get('progress',status.get('progress',{}))
        if last.get('generation') is not None:
            status['progress_eval']=dict(status['progress_eval'],baseline_ready=True)
    return dict(schema=2,exported_at=time.time(),role=role,status=status,actors=actors,gpu=gpu,
        governor={k:v for k,v in governor.items() if k in ['time','mode','storage_ok','idle_since']},
        storage={k:v for k,v in store.items() if k in ['ok','managed_bytes','free_bytes','quota_bytes','reserve_bytes']},
        totals=dict(train_points=len(trains),game_results=dict(outcomes),terminations=dict(termination),promotions=sum(r.get('kind')=='promoted' for r in rows),errors=sum(r.get('kind') in ('error','selfplay_error') for r in rows),unreadable_game_files=bad),
        learning=learning,
        train=sampled(trains),bootstrap=sampled(bootstrap),arenas=arenas[-150:],promotions=[clean(r) for r in rows if r.get('kind')=='promoted'][-100:],
        reviews=[clean(r) for r in rows if r.get('kind') in ['teacher_review','reanalysis']][-200:],
        replay=replays[-100:],
        search=sampled([clean(r) for r in events(STATE/'metrics/search.jsonl')],300),
        inference=sampled([clean(r) for r in events(STATE/'metrics/inference.jsonl')],200),
        game_index=listing,game_details=games,retention='指标为当前保留日志范围；棋谱最多展示最近 400 个文件。')

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--role',choices=['big','small','layered'],required=True);args=ap.parse_args()
    atomic_json(STATE/'dashboard-export.json',snapshot(args.role))
if __name__=='__main__':main()
