"""UniChess 策略/价值网络。

结构：ResNet 主干（带 Squeeze-Excitation）+ 三个头
    policy  1x1 conv C->64  -> (64, 8, 8) -> 展平 4096，索引 = from*64 + to
    promo   小头，4 维（后/车/象/马）
    value   1x1 conv C->8 -> FC -> 3 维 WDL

⚠️ 策略头必须是 1x1 conv，参数量仅 C*64 ≈ 12k。
   若改成 flatten + FC（C*64=12288 -> 4096）会是 5000 万参数，比整个主干还大。

规模是配置项。Stage 1 蒸馏用 15x192，Stage 4 自对弈蒸馏到 10x128（吞吐优先）。
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

NUM_PLANES = 19
POLICY_SIZE = 4096
PROMO_SIZE = 4
WDL_SIZE = 3


@dataclass(frozen=True)
class NetConfig:
    blocks: int = 15
    filters: int = 192
    se_ratio: int = 4          # SE 瓶颈压缩比
    value_channels: int = 8
    value_hidden: int = 256
    num_buckets: int = 1       # >1 时启用按子力数分桶的输出头（Stage 2 之后再开）

    @property
    def name(self) -> str:
        base = f"{self.blocks}x{self.filters}"
        return base if self.num_buckets == 1 else f"{base}-b{self.num_buckets}"


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

        # policy: 1x1 conv 到 64*num_buckets 个平面；plane=from_square, 空间=to_square
        self.policy_conv = nn.Conv2d(c, 64 * nb, 1)
        self.promo_head = nn.Sequential(
            nn.Conv2d(c, 4, 1), nn.Flatten(), nn.ReLU(inplace=True),
            nn.Linear(4 * 64, PROMO_SIZE * nb),
        )
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

        policy = self.policy_conv(h).reshape(n, nb, POLICY_SIZE)
        promo = self.promo_head(h).view(n, nb, PROMO_SIZE)
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
