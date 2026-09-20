from pathlib import Path
import shutil,torch
base=Path('/home/jeefy/UniChess/runs/iteration46_20260909')
dest=base.with_name('iteration46_fast_smoke')
dest.mkdir(exist_ok=True)
ck=torch.load(base/'latest.pt',map_location='cpu',weights_only=False)
limit=ck['step']+2
ck['args']['steps']=limit
torch.save(ck,dest/'latest.pt')
for name in ['train_pools.npz','train_pools.json','valid_pools.npz','valid_pools.json']:
    shutil.copy2(base/name,dest/name)
(dest/'smoke_steps.txt').write_text(str(limit))
print('Prepared isolated resume check at',ck['step'],'to',limit,flush=True)
