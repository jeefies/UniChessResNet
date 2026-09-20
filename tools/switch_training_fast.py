"""Switch only after the old trainer saves its next scheduled checkpoint."""
import json,shutil,subprocess,time
from pathlib import Path
base=Path('/home/jeefy/UniChess')
out=base/'runs/iteration46_20260909'
log=out/'train.log'
def progress():
    steps=[];checkpoints=[]
    for line in log.read_text().splitlines():
        try:r=json.loads(line)
        except ValueError:continue
        if 'step' in r:steps.append(r['step'])
        if 'checkpoint' in r:checkpoints.append(r['checkpoint'])
    return max(steps,default=0),max(checkpoints,default=0)
step,checkpoint=progress()
target=(step//1000+1)*1000
print('Waiting for safe checkpoint',target,'current',step,flush=True)
deadline=time.monotonic()+600
while True:
    if time.monotonic()>deadline:raise TimeoutError('Checkpoint wait expired; original training left running')
    step,checkpoint=progress()
    if checkpoint>=target:break
    time.sleep(0.1)
subprocess.run(['systemctl','--user','stop','unichess-train46-20260909.service'],check=True)
shutil.copy2(out/'latest.pt',out/'before_speed.pt')
shutil.copy2(out/'config.json',out/'config.before_speed.json')
cmd=['systemd-run','--user','--unit=unichess-train46-fast-20260909','--description=UniChess 46M optimized training',
     '--property=WorkingDirectory='+str(base),'--property=Nice=10','--property=TimeoutStopSec=180',
     '--property=StandardOutput=append:'+str(out/'train-fast.log'),
     '--property=StandardError=append:'+str(out/'train-fast.log'),
     '/home/jeefy/miniconda3/envs/unichess/bin/python','-u','model/train_iteration_fast.py',
     '--out','runs/iteration46_20260909','--steps','125000','--batch','512','--accum','2','--save-every','1000','--resume']
subprocess.run(cmd,check=True)
record={'resume_step':checkpoint,'batch':512,'accum':2,'unit':'unichess-train46-fast-20260909.service','before_speed_checkpoint':str(out/'before_speed.pt')}
(out/'speed_switch.json').write_text(json.dumps(record,indent=2))
print(json.dumps(record),flush=True)
