"""把 evaluate_batch 拆成「局面编码(CPU)」和「网络前向(GPU)」分别计时。"""
import sys, time, torch, numpy as np, chess
sys.path.insert(0, '.')
from unichess_r.engine.engine import UniChessEngine
from unichess_r.core.encoding import encode

eng = UniChessEngine(sys.argv[1], device='cuda', syzygy_path='data/raw/syzygy345')
b = chess.Board()
boards = []
for _ in range(512):
    bb = chess.Board()
    for _ in range(np.random.randint(4, 30)):
        ms = list(bb.legal_moves)
        if not ms: break
        bb.push(ms[np.random.randint(len(ms))])
    boards.append(bb)

for n in (128, 256, 512):
    sub = boards[:n]
    # 预热
    for _ in range(3): eng.evaluate_batch(sub)
    torch.cuda.synchronize()

    t0=time.perf_counter()
    for _ in range(20):
        xs = np.stack([encode(x) for x in sub])
    t_enc = (time.perf_counter()-t0)/20

    x = torch.from_numpy(xs).to(eng.device)
    torch.cuda.synchronize()
    t0=time.perf_counter()
    for _ in range(20):
        with torch.no_grad():
            p,pr,w = eng.model(x)
            _ = (torch.softmax(p.float(),1).cpu().numpy(),
                 torch.softmax(pr.float(),1).cpu().numpy(),
                 torch.softmax(w.float(),1).cpu().numpy())
    torch.cuda.synchronize()
    t_gpu = (time.perf_counter()-t0)/20

    t0=time.perf_counter()
    for _ in range(20): eng.evaluate_batch(sub)
    torch.cuda.synchronize()
    t_all = (time.perf_counter()-t0)/20

    print(f'  批 {n:4}: 编码(CPU) {t_enc*1000:6.1f} ms ({100*t_enc/t_all:4.1f}%)  '
          f'前向(GPU) {t_gpu*1000:6.1f} ms ({100*t_gpu/t_all:4.1f}%)  '
          f'合计 {t_all*1000:6.1f} ms  -> {n/t_all:,.0f} 局面/秒')
