from pathlib import Path
import json,hashlib
root=Path('/home/jeefy/UniChess/runs/iteration46_20260909')
out=root/'migration77439parts'
out.mkdir(exist_ok=True)
manifest=[]
for filename in ['latest.pt','best.pt']:
    with (root/filename).open('rb') as f:
        i=0
        while True:
            data=f.read(16*1024*1024)
            if not data:break
            name=filename+'.%03d'%i
            (out/name).write_bytes(data)
            manifest.append({'name':name,'size':len(data),'sha256':hashlib.sha256(data).hexdigest()})
            i+=1
(out/'manifest.json').write_text(json.dumps(manifest))
print('Prepared',len(manifest),'parts')
