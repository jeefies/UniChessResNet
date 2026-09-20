import argparse,subprocess
from pathlib import Path
ap=argparse.ArgumentParser();ap.add_argument('role',choices=['small','actors','relay']);args=ap.parse_args()
root=Path(__file__).resolve().parents[1]
python=Path('/home/jeefy/miniconda3/envs/unichess/bin/python') if args.role!='relay' else root/'.venv/bin/python'
assert python.exists()
name='unichess-autoloop-'+args.role
command=f'{python} -u -m autoloop.worker --role small --mode '+('actor' if args.role=='actors' else 'learner') if args.role!='relay' else f'{python} -u -m autoloop.relay'
unit=f'''[Unit]
Description=UniChess autonomous iteration {args.role}
After=network-online.target
StartLimitIntervalSec=0
[Service]
Type=simple
WorkingDirectory={root}
ExecStart={command}
Restart=always
RestartSec=30
TimeoutStopSec=180
KillSignal=SIGTERM
Environment=PYTHONUNBUFFERED=1
Environment=PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
StandardOutput=null
StandardError=journal
[Install]
WantedBy=default.target
'''
path=Path.home()/'.config/systemd/user';path.mkdir(parents=True,exist_ok=True)
(path/(name+'.service')).write_text(unit)
subprocess.run(['systemctl','--user','daemon-reload'],check=True)
subprocess.run(['systemctl','--user','enable','--now',name],check=True)
subprocess.run(['systemctl','--user','is-active',name],check=True)
