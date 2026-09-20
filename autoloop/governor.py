"""Shared-GPU governor. Only our validated trainer is signalled; pause releases VRAM."""
import argparse, os, signal, subprocess, sys, time
from pathlib import Path
from autoloop.common import ROOT,STATE,atomic_json,event,lock,maintain_storage,read_json

def process(pid):
    try:
        raw=Path(f'/proc/{pid}/cmdline').read_bytes().replace(b'\0',b' ').decode()
        cwd=Path(f'/proc/{pid}/cwd').resolve()
        stat=Path(f'/proc/{pid}/stat').read_text().split()
        return raw,cwd,stat[21],stat[2]
    except (FileNotFoundError,ProcessLookupError,PermissionError):return None

def gpu():
    def query(q):
        return subprocess.check_output(['nvidia-smi',q,'--format=csv,noheader,nounits'],text=True,timeout=8).strip()
    row=query('--query-gpu=memory.used,memory.free,utilization.gpu,power.draw,temperature.gpu').splitlines()[0].split(',')
    apps=query('--query-compute-apps=pid,used_memory')
    processes={}
    for line in apps.splitlines():
        pid,mem=line.split(',');processes[int(pid)]=float(mem.strip())
    return dict(used_mib=float(row[0]),free_mib=float(row[1]),utilization=float(row[2]),power_w=float(row[3]),temperature_c=float(row[4]),processes=processes)

def is_owned(p):
    return p and p[1]==ROOT and ('model/train_iteration_fast.py' in p[0] or 'autoloop.worker' in p[0]) and p[3]!='Z'

def busy(g, own_pid):
    foreign={p:m for p,m in g['processes'].items() if p!=own_pid}
    own_memory=g['processes'].get(own_pid,0)
    # Unattributed memory also protects graphics/other containers not in compute-apps.
    return bool(foreign) or g['used_mib']-own_memory>768 or g['free_mib']<8192

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--adopt',type=int);ap.add_argument('--idle-seconds',type=int,default=120);args=ap.parse_args()
    lease=lock('governor'); pid=args.adopt or (read_json(STATE/'governor-status.json',{}).get('pid')); identity=None; child=None; idle_since=None; stopping=None
    if pid:
        p=process(pid)
        if args.adopt and not is_owned(p):raise RuntimeError('Refusing to adopt unverified process')
        if is_owned(p):identity=p[2]
        else:pid=None
    quit_requested=False
    def stop(*_):
        nonlocal quit_requested
        quit_requested=True
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
    while True:
        if child and child.poll() is not None:child=None
        if pid:
            p=process(pid)
            if not is_owned(p) or p[2]!=identity:
                event('governor','process_exit',pid=pid);pid=None;identity=None;stopping=None
        storage=maintain_storage()
        try:g=gpu(); occupied=busy(g,pid);error=None
        except Exception as e:g={};occupied=True;error=repr(e)
        pressure=occupied or not storage['ok'] or quit_requested or (STATE/'PAUSE').exists()
        if pressure:
            idle_since=None
            if pid and stopping is None:
                # Revalidate PID start time immediately before signalling (PID reuse protection).
                p=process(pid)
                if is_owned(p) and p[2]==identity:
                    os.kill(pid,signal.SIGTERM);stopping=time.time()
                    event('governor','graceful_pause_requested',pid=pid,foreign=g.get('processes'),error=error)
            if pid and stopping and time.time()-stopping>180:
                # Only our process, using last atomic recovery checkpoint if shutdown hangs.
                p=process(pid)
                if is_owned(p) and p[2]==identity:
                    os.kill(pid,signal.SIGKILL);event('governor','forced_stop_after_timeout',pid=pid)
        else:
            idle_since=idle_since or time.time()
            if not pid and time.time()-idle_since>=args.idle_seconds:
                bootstrap=ROOT/'runs/iteration46_20260909'
                # The bootstrap log is the completion marker; checkpoints are loaded by trainer.
                tail=(bootstrap/'train-pro.log').read_bytes()[-16384:] if (bootstrap/'train-pro.log').exists() else b''
                if b'SUPERVISED_PHASE_COMPLETE' not in tail:
                    command=[sys.executable,'-u','model/train_iteration_fast.py','--out',str(bootstrap),'--steps','125000','--batch','1024','--accum','1','--save-every','1000','--resume']
                    log=bootstrap/'train-pro.log'
                else:
                    command=[sys.executable,'-u','-m','autoloop.worker','--role','big'];log=STATE/'worker-big.log'
                    if log.exists() and log.stat().st_size>16*1024**2:log.replace(log.with_suffix('.old'))
                with log.open('a') as f:child=subprocess.Popen(command,cwd=ROOT,stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
                pid=child.pid;p=process(pid);identity=p[2] if p else None
                event('governor','started',pid=pid,command=command)
        status=dict(time=time.time(),pid=pid,mode='yielding' if pressure else ('running' if pid else 'idle_hysteresis'),gpu=g,telemetry_error=error,storage_ok=storage['ok'],idle_since=idle_since)
        atomic_json(STATE/'governor-status.json',status)
        event('governor','resource',**status)
        if quit_requested and not pid:break
        time.sleep(5)
if __name__=='__main__':main()
