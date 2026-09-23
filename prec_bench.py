"""对比推理精度：fp32 vs bf16 vs fp16。"""
import sys, time, torch, numpy as np, chess
sys.path.insert(0, '.')
from unichess_r.engine.engine import UniChessEngine
from unichess_r.core.encoding import encode

eng = UniChessEngine(sys.argv[1], device='cuda', syzygy_path='data/raw/syzygy345')
boards=[]
for _ in range(512):
    bb=chess.Board()
    for _ in range(np.random.randint(4,30)):
        ms=list(bb.legal_moves)
        if not ms: break
        bb.push(ms[np.random.randint(len(ms))])
    boards.append(bb)
xs = np.stack([encode(x) for x in boards])
x = torch.from_numpy(xs).cuda()

def run(dtype, label):
    for _ in range(5):
        with torch.no_grad():
            if dtype is None: p,pr,w = eng.model(x)
            else:
                with torch.autocast('cuda', dtype=dtype): p,pr,w = eng.model(x)
            _ = torch.softmax(p.float(),1).cpu().numpy()
    torch.cuda.synchronize()
    t0=time.perf_counter(); n=30
    for _ in range(n):
        with torch.no_grad():
            if dtype is None: p,pr,w = eng.model(x)
            else:
                with torch.autocast('cuda', dtype=dtype): p,pr,w = eng.model(x)
            out = (torch.softmax(p.float(),1).cpu().numpy(),
                   torch.softmax(pr.float(),1).cpu().numpy(),
                   torch.softmax(w.float(),1).cpu().numpy())
    torch.cuda.synchronize()
    dt=(time.perf_counter()-t0)/n
    print(f'  {label:12} {dt*1000:6.1f} ms  -> {512/dt:>9,.0f} 局面/秒')
    return out

o32 = run(None, 'fp32')
obf = run(torch.bfloat16, 'bf16')
o16 = run(torch.float16, 'fp16')
print()
print('  与 fp32 的最大策略差异:')
print(f'    bf16: {np.abs(o32[0]-obf[0]).max():.2e}')
print(f'    fp16: {np.abs(o32[0]-o16[0]).max():.2e}')
print(f'  WDL 最大差异 bf16 {np.abs(o32[2]-obf[2]).max():.2e}  fp16 {np.abs(o32[2]-o16[2]).max():.2e}')
