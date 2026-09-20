"""纯计算基准：合成数据，不经过 dataloader，测 GPU 天花板。"""
import sys, time, torch, torch.nn.functional as F
sys.path.insert(0, '.')
from model.net import UniChessNet, PRESETS

dev = torch.device('cuda')
for preset in ('medium', 'small'):
    cfg = PRESETS[preset]
    m = UniChessNet(cfg).to(dev)
    opt = torch.optim.AdamW(m.parameters(), lr=1e-3)
    scaler = torch.amp.GradScaler('cuda', enabled=True)
    for batch in (1024, 2048, 4096):
        try:
            x = torch.randn(batch, 19, 8, 8, device=dev)
            pt = torch.rand(batch, 4096, device=dev); pt /= pt.sum(1, keepdim=True)
            wt = torch.rand(batch, 3, device=dev); wt /= wt.sum(1, keepdim=True)
            prt = torch.full((batch,), -100, dtype=torch.long, device=dev)
            for _ in range(5):   # 预热
                with torch.autocast('cuda', dtype=torch.bfloat16):
                    p, pr, w = m(x)
                    loss = -(pt*F.log_softmax(p,1)).sum(1).mean() - (wt*F.log_softmax(w,1)).sum(1).mean()
                opt.zero_grad(set_to_none=True); scaler.scale(loss).backward()
                scaler.step(opt); scaler.update()
            torch.cuda.synchronize()
            n = 25; t0 = time.time()
            for _ in range(n):
                with torch.autocast('cuda', dtype=torch.bfloat16):
                    p, pr, w = m(x)
                    loss = -(pt*F.log_softmax(p,1)).sum(1).mean() - (wt*F.log_softmax(w,1)).sum(1).mean()
                opt.zero_grad(set_to_none=True); scaler.scale(loss).backward()
                scaler.step(opt); scaler.update()
            torch.cuda.synchronize()
            dt = time.time()-t0
            print(f'  {cfg.name:9} batch {batch:5}: {n/dt:5.1f} steps/s  {n*batch/dt:>9,.0f} 样本/秒  '
                  f'显存 {torch.cuda.max_memory_allocated()/2**30:.1f} GB')
            torch.cuda.reset_peak_memory_stats()
        except torch.cuda.OutOfMemoryError:
            print(f'  {cfg.name:9} batch {batch:5}: 显存不足')
            torch.cuda.empty_cache()
    del m, opt; torch.cuda.empty_cache()
