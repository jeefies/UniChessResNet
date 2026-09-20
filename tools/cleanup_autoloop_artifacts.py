"""One-time cleanup of named verification/transfer artifacts; source/data are protected."""
import argparse,json,shutil,time
from pathlib import Path
ap=argparse.ArgumentParser();ap.add_argument('--apply',action='store_true');args=ap.parse_args()
root=Path(__file__).resolve().parents[1];runs=(root/'runs').resolve();state=runs/'autoloop';state.mkdir(exist_ok=True)
protected=[runs/'iteration46_20260909/latest.pt',runs/'iteration46_20260909/best.pt']
assert all(p.is_file() and p.stat().st_size>100*1024**2 for p in protected),'Missing protected bootstrap checkpoints'
names=['iteration46_20260909/migration77439parts','iteration46_fast_smoke','iteration46_smoke_20260909','autoloop_smoke','gpu_bench','gpu_smoke','pro_smoke']
items=[]
for name in names:
 p=runs/name
 if not p.exists():continue
 resolved=p.resolve()
 assert resolved.is_relative_to(runs) and resolved!=runs and not p.is_symlink()
 assert not any(t.is_relative_to(resolved) for t in protected)
 size=sum(f.stat().st_size for f in resolved.rglob('*') if f.is_file())
 items.append({'path':str(resolved),'bytes':size})
 if args.apply:
  if name=='autoloop_smoke':
   evidence=state/'verification-evidence';evidence.mkdir(exist_ok=True)
   for f in list((p/'metrics').glob('*.jsonl'))+list(p.glob('*.log'))+list(p.glob('*test.json')):
    if f.stat().st_size<2*1024**2:shutil.copy2(f,evidence/f.name)
  shutil.rmtree(resolved)
report={'time':time.time(),'applied':args.apply,'items':items,'bytes':sum(i['bytes'] for i in items),'protected':[str(p) for p in protected]}
if args.apply:(state/'cleanup.json').write_text(json.dumps(report,indent=2))
print(json.dumps(report,indent=2))
