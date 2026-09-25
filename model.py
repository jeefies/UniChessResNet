"""UniChess 策略/价值网络。

结构：ResNet 主干（带 Squeeze-Excitation）+ 三个头
    policy  4096 维，索引 = from*64 + to（两种实现，见 NetConfig.policy_head）
    promo   4 维（后/车/象/马），见 NetConfig.promo_head
    value   1x1 conv C->8 -> FC -> 3 维 WDL

⚠️ 策略头**不能**做成 flatten + FC：C*64=12288 -> 4096 是 5000 万参数，
   比整个主干还大。现有两种实现都远在这条线以下（192 通道时约 12k / 29k）：
     conv      1x1 conv C->64 -> (64, 8, 8) -> 展平（旧，只读落点格）
     bilinear  <Q[from], K[to]> / sqrt(d) + bias（两端都读，见 BilinearPolicyHead）

⚠️ 改 NetConfig 的字段就会改 `cfg.__dict__`，而 checkpoint 存的就是它。
    续训的脚本不可以直接比字典相等，必须走 cfg_conflicts()——理由见该函数。

平面尺寸 / 策略掩码定义在 Kit/planes19（losses.legal_from_to_mask 的掩码是全仓库唯一版本；
    它要在平面上推出合法着法掩码，零误杀的验证记录见该处 docstring）。

规模是配置项。Stage 1 蒸馏用 15x192，Stage 4 自对弈蒸馏到 10x128（吞吐优先）。
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, fields

import torch
import torch.nn as nn
import torch.nn.functional as F

# 平面尺寸与合法着法掩码只有一个定义处：Kit/planes19。它们由 kit 的对局、搜索和训练
# 共享，本仓库再抄一份就是三条会在某天不一致的真相。
from Kit.planes19 import NUM_PLANES, POLICY_SIZE, PROMO_SIZE
from Kit.planes19.losses import legal_from_to_mask

# WDL 是三分类价值头的尺寸，T/R 共用的约定（kit 没有自己的价值头，不放 Kit/planes19）
WDL_SIZE = 3

@dataclass(frozen=True)
class NetConfig:
    blocks: int = 15
    filters: int = 192
    se_ratio: int = 4          # SE 瓶颈压缩比
    value_channels: int = 8
    value_hidden: int = 256
    num_buckets: int = 1       # >1 时启用按子力数分桶的输出头（Stage 2 之后再开）
    policy_head: str = "conv"  # "conv" = 1x1 卷积（旧）；"bilinear" = 双线性（见下）
    policy_dim: int = 64       # 仅 bilinear 用：Q/K 的投影维度
    promo_head: str = "legacy" # "legacy" = 旧结构；"bn" = 补上归一化（见 UniChessNet）

    @property
    def name(self) -> str:
        base = f"{self.blocks}x{self.filters}"
        if self.num_buckets != 1:
            base = f"{base}-b{self.num_buckets}"
        if self.policy_head != "conv":
            base = f"{base}-{self.policy_head}"
        return base if self.promo_head == "legacy" else f"{base}-promo{self.promo_head}"


class SqueezeExcitation(nn.Module):
    def __init__(self, channels: int, ratio: int):
        super().__init__()
        hidden = max(channels // ratio, 8)
        self.fc1 = nn.Linear(channels, hidden)
        self.fc2 = nn.Linear(hidden, channels * 2)  # 一半做缩放，一半做偏置

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        n, c, _, _ = x.shape
        s = x.mean(dim=(2, 3))
        s = F.relu(self.fc1(s), inplace=True)
        s = self.fc2(s)
        scale, bias = s.split(c, dim=1)
        scale = torch.sigmoid(scale).view(n, c, 1, 1)
        bias = bias.view(n, c, 1, 1)
        return x * scale + bias


class ResBlock(nn.Module):
    def __init__(self, channels: int, se_ratio: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)
        self.se = SqueezeExcitation(channels, se_ratio)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = self.bn2(self.conv2(out))
        out = self.se(out)
        return F.relu(x + out, inplace=True)


class BilinearPolicyHead(nn.Module):
    """双线性策略头：logit(from, to) = <Q[from], K[to]> / sqrt(d_p) + bias[from, to]。

    为什么要换掉 1x1 卷积头——把旧头展开就能看出问题：

        旧： logit(from, to) = Σ_k W[from, k] * h[k, to] + b[from]

    它**只读落点格的特征** h[:, to]，起点格的特征根本没进这个式子，起点只贡献
    一个与局面无关的静态权重向量 W[from]。于是网络必须把「谁能走到我这儿」
    偷偷编码进每个落点格的通道里，等于用主干容量去补头的结构缺陷。
    双线性头两端都读，这正是 Transformer 那边 BilinearPolicyHead 的做法。

    索引契约保持不变：输出 reshape 成 4096 后仍是 from*64 + to（见 core/moves.py），
    所以分片、dataset、MCTS、引擎一律不用改，只是换了算 logit 的方式。

    参数量 2*(C*d_p + d_p) + 64*64：192 通道 / d_p=64 时约 28.8k，
    对比旧头的 12.4k，仍远小于主干（10.4M），不会触发文件头说的那个膨胀问题。
    """

    def __init__(self, channels: int, d_p: int, num_buckets: int):
        super().__init__()
        self.d_p = d_p
        self.nb = num_buckets
        self.scale = d_p ** -0.5
        # 用 1x1 conv 而非 Linear：保持 channels_last 友好（训练侧依赖它）
        self.wq = nn.Conv2d(channels, d_p * num_buckets, 1)
        self.wk = nn.Conv2d(channels, d_p * num_buckets, 1)
        # 与局面无关的先验偏置（如马的走位形状），让 Q/K 专注于局面相关的部分
        self.bias_move = nn.Parameter(torch.zeros(num_buckets, 64, 64))
        nn.init.trunc_normal_(self.bias_move, std=0.02)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """h: [N, C, 8, 8] -> [N, nb, 4096]，索引 from*64 + to。"""
        n = h.shape[0]
        # [N, nb*d_p, 8, 8] -> [N, nb, d_p, 64]；通道按 bucket 主序切分，
        # 与 policy_conv 的 64*nb 布局约定一致。空间 64 = row*8+col = square。
        q = self.wq(h).reshape(n, self.nb, self.d_p, 64).transpose(2, 3)  # [N,nb,64(from),d_p]
        k = self.wk(h).reshape(n, self.nb, self.d_p, 64)                  # [N,nb,d_p,64(to)]
        m = torch.matmul(q, k) * self.scale + self.bias_move              # [N,nb,64,64]
        return m.reshape(n, self.nb, POLICY_SIZE)


class UniChessNet(nn.Module):
    def __init__(self, cfg: NetConfig | None = None):
        super().__init__()
        self.cfg = cfg or NetConfig()
        c, nb = self.cfg.filters, self.cfg.num_buckets

        self.stem = nn.Sequential(
            nn.Conv2d(NUM_PLANES, c, 3, padding=1, bias=False),
            nn.BatchNorm2d(c),
            nn.ReLU(inplace=True),
        )
        self.tower = nn.Sequential(
            *[ResBlock(c, self.cfg.se_ratio) for _ in range(self.cfg.blocks)]
        )

        # policy：两种头二选一，输出都是 [N, nb, 4096]，索引同为 from*64 + to。
        # 属性名分开（policy_conv / policy_bilinear）是有意的——旧权重的
        # state_dict 里是 "policy_conv.*"，共用一个名字会让 runs/stage1 的
        # checkpoint 加载失败，而 config.json 四个预设全指着它。
        if self.cfg.policy_head == "bilinear":
            self.policy_bilinear = BilinearPolicyHead(c, self.cfg.policy_dim, nb)
        elif self.cfg.policy_head == "conv":
            # plane=from_square, 空间=to_square
            self.policy_conv = nn.Conv2d(c, 64 * nb, 1)
        else:
            raise ValueError(f"未知的 policy_head: {self.cfg.policy_head!r}")
        # promo：4 类（后/车/象/马）。两种结构，输出都是 [N, nb, 4]。
        #
        # legacy 是 Conv2d(c, 4, 1) -> Flatten -> ReLU -> Linear，问题在于
        # **ReLU 直接作用在没有归一化的原始 conv 输出上**。主干里每一处
        # ReLU 前面都有 BatchNorm（stem、ResBlock、value_conv 都是），唯独
        # 这里漏了。后果不是精度差一点，是这个瓶颈只有 4 个通道：某个通道
        # 的预激活一旦整体落到负半轴，ReLU 恒输出 0、梯度恒为 0，它就永久
        # 死掉，而死一个就是丢掉这个头四分之一的容量。没有 BN 就没有任何
        # 机制把它拉回来。
        #
        # bn 版补上 BatchNorm2d(4)，顺序改成和 value_conv 完全一致
        # （conv -> BN -> ReLU -> flatten -> FC），conv 因此去掉 bias（BN 的
        # beta 承担了这个角色）。
        #
        # 属性名分开（promo_head / promo_norm）与 policy 头同理：旧权重的
        # state_dict 里是 "promo_head.0.weight"/"promo_head.3.*"，加了 BN 之后
        # 层号和形状都变了，共用名字只会让 runs/stage1 的 checkpoint 以一条
        # 难读的 shape mismatch 报错，不如直接让 key 对不上。
        if self.cfg.promo_head == "bn":
            self.promo_norm = nn.Sequential(
                nn.Conv2d(c, 4, 1, bias=False),
                nn.BatchNorm2d(4),
                nn.ReLU(inplace=True),
                nn.Flatten(),
                nn.Linear(4 * 64, PROMO_SIZE * nb),
            )
        elif self.cfg.promo_head == "legacy":
            self.promo_head = nn.Sequential(
                nn.Conv2d(c, 4, 1), nn.Flatten(), nn.ReLU(inplace=True),
                nn.Linear(4 * 64, PROMO_SIZE * nb),
            )
        else:
            raise ValueError(f"未知的 promo_head: {self.cfg.promo_head!r}")
        # value: 1x1 conv 降维 -> FC -> WDL
        self.value_conv = nn.Sequential(
            nn.Conv2d(c, self.cfg.value_channels, 1),
            nn.BatchNorm2d(self.cfg.value_channels),
            nn.ReLU(inplace=True),
            nn.Flatten(),
        )
        self.value_fc = nn.Sequential(
            nn.Linear(self.cfg.value_channels * 64, self.cfg.value_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(self.cfg.value_hidden, WDL_SIZE * nb),
        )

    def forward(self, x: torch.Tensor, bucket: torch.Tensor | None = None):
        """返回 (policy_logits[N,4096], promo_logits[N,4], wdl_logits[N,3])。

        num_buckets > 1 时必须传 bucket（每个样本的桶索引 [N]），
        输出会按桶索引 gather 出对应那一组头的结果。
        """
        n = x.shape[0]
        nb = self.cfg.num_buckets
        h = self.tower(self.stem(x))

        if self.cfg.policy_head == "bilinear":
            policy = self.policy_bilinear(h)
        else:
            policy = self.policy_conv(h).reshape(n, nb, POLICY_SIZE)
        promo_mod = (self.promo_norm if self.cfg.promo_head == "bn"
                     else self.promo_head)
        promo = promo_mod(h).view(n, nb, PROMO_SIZE)
        wdl = self.value_fc(self.value_conv(h)).view(n, nb, WDL_SIZE)

        if nb == 1:
            return policy[:, 0], promo[:, 0], wdl[:, 0]

        if bucket is None:
            raise ValueError("num_buckets > 1 时必须提供 bucket 索引")
        idx = bucket.view(n, 1, 1)
        policy = policy.gather(1, idx.expand(n, 1, POLICY_SIZE)).squeeze(1)
        promo = promo.gather(1, idx.expand(n, 1, PROMO_SIZE)).squeeze(1)
        wdl = wdl.gather(1, idx.expand(n, 1, WDL_SIZE)).squeeze(1)
        return policy, promo, wdl


def cfg_conflicts(saved: dict, cfg: NetConfig) -> list[str]:
    """比对 checkpoint 里存的 cfg 与当前 cfg，返回所有不一致处（空列表 = 可以续训）。

    **不能直接写 `saved == cfg.__dict__`**，两个 --resume 的脚本原先就是这么
    写的。NetConfig 后来加过字段（policy_head / policy_dim / promo_head），
    老 checkpoint 的 dict 里根本没有这些 key，字典相等判断于是对每一个旧 run
    都失败——而它们的网络结构其实一个字节都没变，新字段的默认值复刻的正是
    旧结构。报出来的还是一句没有信息量的 AssertionError。

    规则：
      * 共有字段：必须逐个相等
      * checkpoint 缺、当前有：当前值必须等于该字段的默认值（默认值 = 旧结构；
        不是默认值就说明结构真的改了，那确实不能续训）
      * checkpoint 有、当前没有：是从更新的版本往回退，一律算冲突
    """
    defaults = {f.name: f.default for f in fields(NetConfig)}
    cur = asdict(cfg)
    out: list[str] = []
    for k in sorted(set(saved) | set(cur)):
        if k not in cur:
            out.append(f"{k}: checkpoint 有（={saved[k]!r}），当前 NetConfig 已无此字段")
        elif k not in saved:
            if cur[k] != defaults.get(k):
                out.append(f"{k}: checkpoint 无此字段（等同默认值 "
                           f"{defaults.get(k)!r}），当前为 {cur[k]!r}")
        elif saved[k] != cur[k]:
            out.append(f"{k}: checkpoint={saved[k]!r}，当前={cur[k]!r}")
    return out


def count_params(model: nn.Module) -> dict[str, int]:
    """按模块统计参数量，用于确认策略头没有意外膨胀。"""
    groups: dict[str, int] = {}
    for name, p in model.named_parameters():
        top = name.split(".")[0]
        groups[top] = groups.get(top, 0) + p.numel()
    groups["TOTAL"] = sum(p.numel() for p in model.parameters())
    return groups


PRESETS = {
    "tiny":   NetConfig(blocks=6,  filters=64),    # Phase A 管线验证，CPU 可训
    "small":  NetConfig(blocks=10, filters=128),   # Stage 4 自对弈
    "medium": NetConfig(blocks=15, filters=192),   # Stage 1 蒸馏（默认）
    "large":  NetConfig(blocks=20, filters=256),   # 备选
}
