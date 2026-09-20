"""Bounded, project-owned storage and durable telemetry for autonomous iteration."""
import fcntl, gzip, hashlib, json, os, shutil, time
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
RUN_NAME = os.environ.get('UNICHESS_LOOP_RUN', 'autoloop')
if '/' in RUN_NAME or '\\' in RUN_NAME or not RUN_NAME.startswith('autoloop'): raise ValueError('Invalid run name')
STATE = ROOT / 'runs' / RUN_NAME
GIB = 1024**3

def setup():
    STATE.mkdir(parents=True, exist_ok=True)
    for d in ['metrics','replay','population','inbox','feedback','models','jobs','games']:
        (STATE/d).mkdir(exist_ok=True)

def atomic_json(path, value):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name(path.name+'.tmp')
    with tmp.open('w') as f:
        json.dump(value,f,ensure_ascii=False,allow_nan=False); f.flush(); os.fsync(f.fileno())
    tmp.replace(path)

def read_json(path, default=None):
    try:return json.loads(Path(path).read_text())
    except FileNotFoundError:return default

def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()

def lock(name):
    setup(); f=(STATE/(name+'.lock')).open('w')
    fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
    f.write(str(os.getpid()));f.flush()
    return f

def event(stream, kind, **fields):
    setup()
    with (STATE/'metrics'/(stream+'.lock')).open('a') as lease:
        fcntl.flock(lease,fcntl.LOCK_EX)
        _event(stream,kind,fields)

def _event(stream,kind,fields):
    p=STATE/'metrics'/(stream+'.jsonl')
    # Single writer per stream; bounded 5 x 16 MiB, no unbounded stdout logs.
    if p.exists() and p.stat().st_size>16*1024**2:
        p.with_suffix('.jsonl.4').unlink(missing_ok=True)
        for n in range(3,0,-1):
            old=p.with_suffix('.jsonl.'+str(n))
            if old.exists():old.replace(p.with_suffix('.jsonl.'+str(n+1)))
        p.replace(p.with_suffix('.jsonl.1'))
    with p.open('a') as f:f.write(json.dumps({'time':time.time(),'kind':kind,**fields},allow_nan=False)+'\n')

def write_bundle(path, rows):
    path=Path(path);tmp=path.with_suffix('.tmp')
    with gzip.open(tmp,'wt',encoding='utf8') as f:
        for row in rows:f.write(json.dumps(row,allow_nan=False)+'\n')
    tmp.replace(path)

def read_bundle(path):
    with gzip.open(path,'rt',encoding='utf8') as f:return [json.loads(s) for s in f]

def maintain_storage(quota_gib=24, reserve_gib=20):
    setup()
    with (STATE/'storage.lock').open('a') as lease:
        fcntl.flock(lease,fcntl.LOCK_EX)
        return _maintain_storage(quota_gib,reserve_gib)

def _size(path):
    try:return path.stat().st_size
    except FileNotFoundError:return 0

def _maintain_storage(quota_gib, reserve_gib):
    """Delete ONLY expendable files inside our managed directories, never seed/data."""
    setup(); base=STATE.resolve(); deleted=[]
    caps={'replay':4*GIB,'population':4*GIB,'inbox':4*GIB,'feedback':2*GIB,'games':GIB,'jobs':GIB}
    # Keep recent 7-day replay and 30-day games; no source/checkpoint recursive deletion.
    for name,cap in caps.items():
        files=sorted((base/name).glob('*.gz'),key=lambda p:p.stat().st_mtime,reverse=True)
        total=0
        for p in files:
            if p.name.endswith('-active.json.gz'):continue
            total+=p.stat().st_size
            expired=time.time()-p.stat().st_mtime>(30 if name=='games' else 7)*86400
            if (total>cap or expired) and not p.is_symlink() and p.resolve().parent==base/name:
                try:p.unlink();deleted.append(str(p.relative_to(base)))
                except FileNotFoundError:pass
    for suffix in ['*.tmp','*.part']:
        for p in base.rglob(suffix):
            if p.is_file() and not p.is_symlink() and time.time()-p.stat().st_mtime>86400:
                p.unlink(missing_ok=True);deleted.append(str(p.relative_to(base)))
    size=sum(_size(p) for p in base.rglob('*') if p.is_file() and not p.is_symlink())
    free=shutil.disk_usage(base).free
    ok=free>=reserve_gib*GIB and size<quota_gib*GIB
    result=dict(ok=ok,managed_bytes=size,free_bytes=free,quota_bytes=quota_gib*GIB,reserve_bytes=reserve_gib*GIB,deleted=deleted)
    atomic_json(base/'storage.json',result)
    return result
