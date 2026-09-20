"""Credential-preserving WSL relay. Only bounded, checksummed replay bundles cross hosts."""
import concurrent.futures, json, re, shlex, subprocess, time
from autoloop.common import *
HOSTS={'small':('jeefy@172.16.2.12','/home/jeefy/UniChess'), 'big':('pro6000','/root/autodl-tmp/fwj/UniChess')}

def ssh(host,command):
    return subprocess.check_output(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=10',host,command],text=True,timeout=40)

def remote(role,relative):
    host,base=HOSTS[role];return host,base+'/runs/autoloop/'+relative

def inventory(role,folder):
    host,path=remote(role,folder)
    script="import pathlib,json,time; p=pathlib.Path("+repr(path)+"); print(json.dumps([x.name for x in sorted(p.glob('*.gz'),key=lambda p:p.stat().st_mtime)[-64:] if time.time()-x.stat().st_mtime<604800]))"
    command='python3 -c '+shlex.quote(script)
    return [n for n in json.loads(ssh(host,command)) if re.fullmatch(r'[a-zA-Z0-9_.-]+\.gz',n)]

def transfer(source,srcfolder,target,dstfolder,name):
    src_host,src=remote(source,srcfolder+'/'+name);dst_host,dst=remote(target,dstfolder+'/'+name)
    # Successful receipts persist on disk; do not resend expired/retired bundles.
    receipt=STATE/'jobs'/('sent-'+source+'-'+name+'.json')
    if receipt.exists():return 0
    cache=STATE/'inbox'/name;temp=cache.with_suffix('.part')
    expected=ssh(src_host,'sha256sum '+shlex.quote(src)).split()[0]
    subprocess.run(['scp','-q','-o','BatchMode=yes','-o','ConnectTimeout=10',src_host+':'+src,str(temp)],check=True,timeout=180)
    if temp.stat().st_size>64*1024**2 or digest(temp)!=expected:raise ValueError('Bundle size/hash validation failed')
    temp.replace(cache)
    subprocess.run(['scp','-q','-o','BatchMode=yes','-o','ConnectTimeout=10',str(cache),dst_host+':'+dst+'.part'],check=True,timeout=180)
    got=ssh(dst_host,'sha256sum '+shlex.quote(dst+'.part')).split()[0]
    if got!=expected:raise ValueError('Remote transfer hash mismatch')
    ssh(dst_host,'mv '+shlex.quote(dst+'.part')+' '+shlex.quote(dst))
    atomic_json(receipt,dict(time=time.time(),sha256=expected,bytes=cache.stat().st_size))
    event('relay','transferred',source=source,target=target,name=name,bytes=cache.stat().st_size,sha256=expected)
    return cache.stat().st_size

def _sync_dashboard_role(role):
    src_host,src_root=HOSTS['big'];dst_host,dst_root=HOSTS['small']
    ssh(src_host,'cd '+shlex.quote(src_root)+' && bash run_pro6000.sh -m autoloop.dashboard_export --role '+role)
    src=src_root+'/runs/autoloop/dashboard-export.json';dst=dst_root+'/runs/autoloop/dashboard-'+role+'.json'
    local=STATE/('dashboard-'+role+'.json')
    subprocess.run(['scp','-q','-o','BatchMode=yes','-o','ConnectTimeout=10',src_host+':'+src,str(local)],check=True,timeout=120)
    if local.stat().st_size>32*1024**2:raise ValueError('Dashboard snapshot too large')
    expected=digest(local)
    subprocess.run(['scp','-q','-o','BatchMode=yes','-o','ConnectTimeout=10',str(local),dst_host+':'+dst+'.part'],check=True,timeout=120)
    actual=ssh(dst_host,'sha256sum '+shlex.quote(dst+'.part')).split()[0]
    if actual!=expected:raise ValueError('Dashboard checksum mismatch')
    ssh(dst_host,'mv '+shlex.quote(dst+'.part')+' '+shlex.quote(dst))

def sync_dashboard():
    for role in ['big','layered']:
        _sync_dashboard_role(role)


def sync_small_champion():
    """Copy the current small champion to PRO after two hash checks."""
    source_host, source_root = HOSTS['small']
    target_host, target_root = HOSTS['big']
    source = source_root + '/runs/autoloop/models/small-champion.pt'
    target = target_root + '/runs/autoloop/models/small-champion.pt'
    receipt = STATE / 'jobs/model-small-champion.json'
    signature = ssh(source_host, 'stat -c %s:%Y ' + shlex.quote(source)).strip()
    previous = read_json(receipt, {})
    if previous.get('signature') == signature:
        try:
            ssh(target_host, 'test -s ' + shlex.quote(target))
            return 0
        except subprocess.CalledProcessError:
            pass
    expected = ssh(source_host, 'sha256sum ' + shlex.quote(source)).split()[0]
    try:
        current = ssh(target_host, 'sha256sum ' + shlex.quote(target)).split()[0]
        if current == expected:
            atomic_json(receipt, dict(time=time.time(), signature=signature, sha256=expected,
                                      bytes=int(signature.split(':', 1)[0])))
            return 0
    except Exception:
        pass
    size = int(signature.split(':', 1)[0])
    if size > 128 * 1024**2:
        raise ValueError('small champion snapshot exceeds relay bound')
    cache = STATE / 'small-champion.pt'
    temp = cache.with_suffix('.part')
    rsync_ssh = 'ssh -o BatchMode=yes -o ConnectTimeout=10 -o ServerAliveInterval=15 -o ServerAliveCountMax=20'
    subprocess.run(['rsync', '-a', '--partial', '--append-verify', '--timeout=60', '-e', rsync_ssh,
                    source_host + ':' + source, str(temp)], check=True, timeout=900)
    if temp.stat().st_size != size or digest(temp) != expected:
        temp.unlink(missing_ok=True)
        raise ValueError('local small champion checksum or size mismatch')
    temp.replace(cache)
    ssh(target_host, 'mkdir -p ' + shlex.quote(target.rsplit('/', 1)[0]))
    subprocess.run(['rsync', '-a', '--partial', '--append-verify', '--timeout=60', '-e', rsync_ssh,
                    str(cache), target_host + ':' + target + '.part'], check=True, timeout=900)
    actual = ssh(target_host, 'sha256sum ' + shlex.quote(target + '.part')).split()[0]
    if actual != expected:
        ssh(target_host, 'rm -f ' + shlex.quote(target + '.part'))
        raise ValueError('small champion checksum mismatch')
    ssh(target_host, 'mv ' + shlex.quote(target + '.part') + ' ' + shlex.quote(target))
    atomic_json(receipt, dict(time=time.time(), signature=signature, sha256=expected, bytes=size))
    event('relay', 'model_transferred', source='small', target='big',
          name='small-champion.pt', bytes=size, sha256=expected)
    return size

def main():
    lease=lock('relay')
    while True:
        try:
            host,base=HOSTS['big']
            ssh(host,'cd '+shlex.quote(base)+' && python3 tools/ensure_autoloop_pro.py')
            if maintain_storage()['ok']:
                sync_small_champion()
                for source,sfolder,target,tfolder in [('small','replay','big','inbox'),
                    ('small','population','big','inbox'),('big','population','small','inbox'),
                    ('big','feedback','small','feedback')]:
                    names=inventory(source,sfolder)
                    event('relay','queue',source=source,bundles=len(names),pending=sum(not (STATE/'jobs'/('sent-'+source+'-'+n+'.json')).exists() for n in names))
                    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
                        for result in pool.map(lambda n:transfer(source,sfolder,target,tfolder,n),names):pass
            status={'time':time.time()}
            for role in HOSTS:
                host,path=remote(role,role+'-status.json')
                try:
                    status[role]=json.loads(ssh(host,'cat '+shlex.quote(path)))
                    gpu_values=ssh(host,'nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.free,power.draw,temperature.gpu --format=csv,noheader,nounits').strip().splitlines()[0].split(',')
                    status[role]['gpu']=dict(zip(['utilization','used_mib','free_mib','power_w','temperature_c'],map(float,gpu_values)))
                    if role=='small':
                        _,actor_path=remote(role,'small-actor-status.json')
                        status[role]['actors']=json.loads(ssh(host,'cat '+shlex.quote(actor_path)))
                    event('relay','host_status',role=role,status=status[role])
                except Exception as e:status[role]={'error':repr(e)}
            atomic_json(STATE/'cluster-status.json',status)
            # Receipts outlive remote retention, but are themselves bounded.
            for p in (STATE/'jobs').glob('sent-*.json'):
                if time.time()-p.stat().st_mtime>14*86400:p.unlink()
            for p in (STATE/'inbox').glob('*.part'):
                if time.time()-p.stat().st_mtime>86400:p.unlink()
        except Exception as e:event('relay','error',error=repr(e))
        time.sleep(30)
if __name__=='__main__':main()
