"""Stage 1 蒸馏训练。

loss = CE(policy, stockfish_top5_soft)      # 软标签，**softmax 只在合法着法上归一**
     + value_weight * CE(wdl, stockfish_wdl)
     + promo_weight * CE(promo, best_promo)  # 仅升变样本参与（ignore_index=-100）
     + weight_decay 由 AdamW 负责

同一份脚本既跑本机 CPU 小样本冒烟测试，也跑 5070 Ti 全量训练，
差别只在 --device / --batch / --amp。Phase A 的纪律是：
GPU 空出来之前，这条链路必须已经在 CPU 上完整跑通过。
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from dataclasses import replace
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from unichess_r.model.dataset import make_loader
from unichess_r.model.net import (PRESETS, NetConfig, UniChessNet, count_params,
                       legal_from_to_mask)

# 掩码位置填的哨兵值。**绝不能用 -inf**：被掩掉的位置上 target 恰好是 0，
# 而 0 * -inf = NaN，整个 loss 会在第一步就变成 NaN。-1e4 减去 logit 最大值后
# exp() 在 fp32 下精确下溢到 0，对分母没有任何贡献，效果与 -inf 相同且安全。
MASK_FILL = -1e4


def soft_cross_entropy(logits: torch.Tensor, target: torch.Tensor,
                       mask: torch.Tensor | None = None,
                       legal: torch.Tensor | None = None) -> torch.Tensor:
    """软标签交叉熵。target 是概率分布，不是类别索引。

    mask  [N] 样本级开关。为 None 时对整批取平均；给了就只在 mask 为真的
          样本上平均。PGN 分片只带价值标签、策略标签留全零，靠这个屏蔽掉
          它们的策略损失——人类实走的 one-hot 会把策略头往人类风格拉，
          与「棋力最强」的目标相悖。
    legal [N, C] 类别级掩码。给了就把 softmax 的归一化限制在这些类别上。

    **为什么要 legal 掩码。** 4096 维里任一局面平均只有 31 个是合法的
    （实测），其余 99.2% 在该局面下根本不是「另一个选项」，而是规则排除掉的
    东西。不掩码的话，softmax 的分母包含这 4000 多项，网络就得花容量去学
    「这些不可能」——那是规则，不是棋理，学它一点棋力都换不来。

    **这不改变推理语义，是让训练与推理对齐。** 推理端两个消费者
    （search/mcts.py:priors_from_policy、engine/engine.py:_from_network）拿到
    4096 维 softmax 之后，都只取合法项再重新归一化。而重归一化的结果只取决于
    合法项**之间**的 logit 差，与分母里还有多少项无关。所以掩码训练优化的
    恰好就是推理真正使用的那个分布，两边严格等价——不需要改推理端任何代码。
    """
    if legal is not None:
        logits = logits.masked_fill(~legal, MASK_FILL)
    per = -(target * F.log_softmax(logits, dim=1)).sum(dim=1)
    if mask is None:
        return per.mean()
    m = mask.float()
    denom = m.sum()
    if denom < 1:
        return torch.zeros((), device=logits.device, dtype=per.dtype)
    return (per * m).sum() / denom


def policy_topk_accuracy(logits: torch.Tensor, target: torch.Tensor, k: int = 1) -> float:
    """预测的 top-1 是否命中软标签里概率最大的那个走法。"""
    pred = logits.topk(k, dim=1).indices
    gold = target.argmax(dim=1, keepdim=True)
    return (pred == gold).any(dim=1).float().mean().item()


def build_model(args) -> UniChessNet:
    if args.preset:
        cfg = PRESETS[args.preset]
    else:
        cfg = NetConfig(blocks=args.blocks, filters=args.filters)
    # 用 replace 而不是重新构造 NetConfig：原先 --buckets>1 时会把 se_ratio /
    # value_channels / value_hidden 悄悄退回默认值（AGENTS.md 记过这个坑）。
    # buckets=1 时两种写法等价，所以这个修正不改变已有行为。
    return UniChessNet(replace(cfg, num_buckets=args.buckets,
                               policy_head=args.policy_head,
                               promo_head=args.promo_head))


def train(args) -> int:
    device = torch.device(args.device)
    torch.manual_seed(args.seed)

    # 用整批解码的 loader：decode_batch 是向量化位运算，
    # 一次解 1024 条和解 1 条耗时相近，逐条取样会浪费掉全部向量化收益
    # （本机实测：数据驻留页缓存时 16k -> 96k 样本/秒，5.9x）。
    ds, loader = make_loader(args.data, args.batch, num_buckets=args.buckets,
                             num_workers=args.workers,
                             pin_memory=(device.type == "cuda"))
    print(f"数据集：{len(ds):,} 条样本，来自 {len(ds.paths)} 个分片")

    model = build_model(args).to(device)
    print(f"网络：{model.cfg.name}，参数量 {count_params(model)['TOTAL']:,}")

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr,
                            weight_decay=args.weight_decay)
    total_steps = args.steps or (len(loader) * args.epochs)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=args.lr, total_steps=total_steps, pct_start=0.05)

    use_amp = args.amp and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    amp_dtype = torch.bfloat16

    out_dir = Path(args.out); out_dir.mkdir(parents=True, exist_ok=True)
    step, t0 = 0, time.time()
    log: list[dict] = []

    model.train()
    done = False
    for epoch in range(args.epochs if not args.steps else 10 ** 9):
        if done:
            break
        for batch in loader:
            if args.buckets > 1:
                x, p_t, pr_t, w_t, bk = [b.to(device, non_blocking=True) for b in batch]
            else:
                x, p_t, pr_t, w_t = [b.to(device, non_blocking=True) for b in batch]
                bk = None

            # 合法着法掩码直接从输入平面推，不需要 FEN，也不需要碰 python-chess
            # （逐着法遍历会把 GPU 饿死）。见 model/net.py:legal_from_to_mask。
            legal = legal_from_to_mask(x) if args.legal_mask else None
            if legal is not None and step == 0:
                # 契约自检，只做一次。掩码的全部正当性都建立在「绝不误杀合法
                # 着法」上，而它是在随机对局上验的——训练数据是 Stockfish 评
                # 估过的局面，分布不同。这里直接在真实数据上确认：凡是标签给了
                # 概率的位置，掩码必须都放行。一次逐位比较，代价可以忽略。
                leaked = ((p_t > 0) & ~legal).sum().item()
                if leaked:
                    raise RuntimeError(
                        f"合法着法掩码误杀了 {leaked} 个带标签的着法——"
                        f"掩码规则与数据的编码约定对不上，此时训练出来的策略头"
                        f"是错的。请先核对 core/encoding.py 的平面布局。")
                d = legal.float().sum(1).mean()
                # 把这条打进日志是必要的：掩码把 softmax 分母从 4096 缩到 ~640，
                # policy loss 因此**直接低 ln(4096/640) ≈ 1.86 nats**，与棋力
                # 无关。开掩码的 run 和不开的 run 的 policy loss 不可直接比较，
                # 看曲线时必须先看这一行。top1 不受影响，跨 run 比较请用它。
                print(f"掩码自检通过：softmax 分母 4096 -> {d:.0f}（本批均值）；"
                      f"policy loss 基线因此下移约 {math.log(4096 / d):.2f} nats，"
                      f"不可与 --no-legal-mask 的 run 直接比较")

            with torch.autocast("cuda", dtype=amp_dtype, enabled=use_amp):
                p_l, pr_l, w_l = model(x, bk)
                # 策略标签全零 = 该样本没有策略标签（PGN 分片），不计入策略损失
                has_policy = p_t.sum(dim=1) > 1e-6
                loss_p = soft_cross_entropy(p_l, p_t, has_policy, legal)
                loss_w = soft_cross_entropy(w_l, w_t)
                loss_pr = F.cross_entropy(pr_l, pr_t, ignore_index=-100)
                if torch.isnan(loss_pr):      # 这一批没有升变样本
                    loss_pr = torch.zeros((), device=device)
                loss = loss_p + args.value_weight * loss_w + args.promo_weight * loss_pr

            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
            scaler.step(opt); scaler.update(); sched.step()
            step += 1

            if step % args.log_every == 0:
                sel = has_policy
                # top1 要在**掩码后**的 logits 上算：非法位置的 logit 从来没被
                # 训练过，是纯噪声，留着它们会让 top1 偏低，和 loss 对不上。
                p_eval = p_l.detach().float()
                if legal is not None:
                    p_eval = p_eval.masked_fill(~legal, MASK_FILL)
                acc = (policy_topk_accuracy(p_eval[sel], p_t[sel])
                       if sel.any() else float("nan"))
                rec = {"step": step, "loss": loss.item(),
                       "policy": loss_p.item(), "value": loss_w.item(),
                       "top1": acc, "lr": sched.get_last_lr()[0],
                       "sec": round(time.time() - t0, 1)}
                log.append(rec)
                print(f"step {step:>7} | loss {rec['loss']:.4f} "
                      f"(p {rec['policy']:.4f} v {rec['value']:.4f}) | "
                      f"top1 {acc:.3f} | lr {rec['lr']:.2e} | "
                      f"{step/(time.time()-t0):.1f} steps/s", flush=True)

            if step % args.save_every == 0 or step >= total_steps:
                ckpt = out_dir / f"ckpt_{step:08d}.pt"
                torch.save({"model": model.state_dict(),
                            "cfg": model.cfg.__dict__, "step": step}, ckpt)
                (out_dir / "log.json").write_text(json.dumps(log, indent=1))
                print(f"  -> 已保存 {ckpt}")
            if step >= total_steps:
                done = True
                break

    print(f"训练结束：{step} steps，用时 {(time.time()-t0)/60:.1f} 分")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="UniChess Stage 1 蒸馏训练")
    ap.add_argument("--data", required=True, help="分片目录")
    ap.add_argument("--out", default="runs/stage1")
    ap.add_argument("--preset", choices=list(PRESETS), default=None)
    ap.add_argument("--blocks", type=int, default=15)
    ap.add_argument("--filters", type=int, default=192)
    ap.add_argument("--buckets", type=int, default=1)
    ap.add_argument("--policy-head", choices=["conv", "bilinear"], default="conv",
                    help="策略头：conv=1x1 卷积（旧，只读落点格）；bilinear=双线性（两端都读）")
    ap.add_argument("--promo-head", choices=["legacy", "bn"], default="legacy",
                    help="升变头：legacy=旧结构（ReLU 作用在未归一化的 conv 输出上）；bn=补 BatchNorm")
    ap.add_argument("--legal-mask", dest="legal_mask", action="store_true", default=True,
                    help="策略 softmax 只在合法着法上归一（默认开）")
    ap.add_argument("--no-legal-mask", dest="legal_mask", action="store_false",
                    help="关掉合法着法掩码，退回在全 4096 维上归一（消融对照用）")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--batch", type=int, default=1024)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--steps", type=int, default=0, help=">0 时按步数而非轮数停止")
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--value-weight", type=float, default=1.0)
    ap.add_argument("--promo-weight", type=float, default=0.1)
    ap.add_argument("--clip", type=float, default=2.0)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--amp", action="store_true", default=True)
    ap.add_argument("--no-amp", dest="amp", action="store_false")
    ap.add_argument("--log-every", type=int, default=50)
    ap.add_argument("--save-every", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=20260908)
    return train(ap.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
