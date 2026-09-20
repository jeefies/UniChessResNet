import json
from pathlib import Path
root=Path(__file__).resolve().parents[1];state=root/'runs/autoloop';report={}
for name in ['small-status.json','big-status.json','governor-status.json','storage.json','big-last-arena.json','small-last-arena.json']:
 p=state/name
 if p.exists():report[name]=json.loads(p.read_text())
p=state/'metrics/governor.jsonl'
if p.exists():report['governor_events']=[json.loads(s) for s in p.read_text().splitlines() if json.loads(s)['kind']!='resource'][-12:]
for role in ['small','big']:
 p=state/'metrics'/(role+'.jsonl')
 if p.exists():report[role+'_events']=[json.loads(s) for s in p.read_text().splitlines()][-3:]
report['bundles']={d:len(list((state/d).glob('*.gz'))) for d in ['replay','inbox','feedback','games']}
print(json.dumps(report,indent=2))
