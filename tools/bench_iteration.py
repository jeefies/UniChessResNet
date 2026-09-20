import sys,time,json,gc
from pathlib import Path
import numpy as np
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from model.net import UniChessNet,NetConfig
from model import train_iteration as old
from model import train_iteration_fast as fast
torch.set_num_threads(4);torch.backends.cudnn.benchmark=True
paths=sorted(Path('data/shards_evals').glob('*.bin'))[:-4]
pool=fast.Pool(paths,Path('runs/iteration46_20260909/train_pools.npz'))
results=[]
for batch,optimized in [(128,False),(256,True),(512,True),(1024,True)]:
    torch.manual_seed(82)
    model=UniChessNet(NetConfig(blocks=24,filters=320)).cuda()
    if optimized:model=model.to(memory_format=torch.channels_last)
    opt=torch.optim.AdamW(model.parameters(),lr=1e-4,fused=optimized)
    loss_fn=fast.loss_for if optimized else old.loss_for
    rng=np.random.default_rng(123)
    fixed=pool.sample(rng,batch)
    if not optimized:fixed[0]=fixed[0].contiguous()
    accum=1024//batch
    torch.cuda.reset_peak_memory_stats()
    times=[]
    try:
      for update in range(16):
        torch.cuda.synchronize();start=time.perf_counter()
        opt.zero_grad(set_to_none=True)
        for _ in range(accum):
          loss,lp,lw=loss_fn(model,fixed)
          (loss/accum).backward()
          if not optimized: _=loss.item(),lp.item(),lw.item()
        torch.nn.utils.clip_grad_norm_(model.parameters(),2)
        opt.step();torch.cuda.synchronize()
        if update>=5:times.append(time.perf_counter()-start)
      result={'batch':batch,'accum':accum,'optimized':optimized,'gpu_samples_per_sec':round(1024/np.median(times),1),'peak_mib':round(torch.cuda.max_memory_allocated()/2**20),'loss':loss.item()}
    except torch.OutOfMemoryError:
      result={'batch':batch,'oom':True}
    print(json.dumps(result),flush=True);results.append(result)
    del model,opt,fixed;gc.collect();torch.cuda.empty_cache()
Path('runs/iteration46_20260909/benchmark.json').write_text(json.dumps(results,indent=2))
