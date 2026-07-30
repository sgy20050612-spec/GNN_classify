"""
================================================================================
  encoder.py — GNN 编码器 + 预训练权重迁移
================================================================================

  这是我整个 GNN 项目最核心的文件。编码器的任务是把"用户特征 + 图结构"
  压缩成一个 128 维的稠密向量（embedding），这个向量要能表达用户在
  游戏偏好空间中的位置。

  我在这里实现了三种编码器：
    GATEncoder  — 图注意力网络（主力，效果最好）
    GCNEncoder  — 图卷积网络（简单快速，作为对比基线）
    SageEncoder — GraphSAGE（适合大规模图，备用方案）

  每种编码器的结构都是相同的模式：
    Input -> InputProjection -> GNN Layer 1 -> GNN Layer 2 -> Output Embedding

  区别只在于中间的 GNN 层用什么：GATConv / GCNConv / SAGEConv。
================================================================================
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, GCNConv, SAGEConv
from common import BaseLayer2D


# ============================================================================
#  GATEncoder — 图注意力网络编码器（主力）
# ============================================================================

class GATEncoder(nn.Module):
    """
    图注意力网络编码器。这是我项目中效果最好的编码器。

    【为什么选 GAT 而不是 GCN？】
      GCN 对所有邻居一视同仁。GAT 通过注意力机制，让模型自己学
      哪些邻居更重要。这对于游戏用户分类很关键：一个 Action 玩家
      的邻居中既有 Action 玩家也有 RPG 玩家，GAT 能学会给 Action
      邻居更高的权重。

    【结构】
      Input(10维) -> InputProjection(128) -> GATConv(128->512,4头) -> GATConv(512->128,1头)
      每层后都跟 BN + ReLU + Dropout

    【多头注意力】
      第 1 层用 4 个头，每头学不同的注意力模式（有的关注 Action 偏好，
      有的关注 RPG 偏好），然后拼接成 512 维。
      第 2 层用 1 个头做融合，把多视角压缩回 128 维统一表示。
      这就像组织一个 4 人专家小组，每人独立打分，最后汇总意见。

    【参数】
      in_dim    : 输入特征维度，现在是 10（10 种游戏类型）
      hidden_dim: 隐藏层维度，默认 128
      out_dim   : 输出嵌入维度，默认 128
      dropout   : Dropout 率
      heads     : 第一层的注意力头数，默认 4
    """

    def __init__(self, in_dim=10, hidden_dim=128, out_dim=128, dropout=0.5, heads=4):
        super().__init__()

        # 输入投影：10 维游戏分布 -> 128 维隐藏空间
        # 这里用 BaseLayer2D 而不是裸 Linear，因为 BN 能让后续 GAT 的输入更稳定
        self.input_proj = BaseLayer2D(in_dim, hidden_dim, dropout=0.0, act="relu")

        # GAT 第 1 层：多头拼接
        # concat=True 意味着 4 个头各自输出 128 维，拼接后 512 维
        self.conv1 = GATConv(hidden_dim, hidden_dim, heads=heads,
                             dropout=dropout, concat=True)
        self.bn1 = nn.BatchNorm1d(hidden_dim * heads)  # BN(512)

        # GAT 第 2 层：单头融合
        # concat=False 意味着取 1 个头的输出，即 128 维
        self.conv2 = GATConv(hidden_dim * heads, hidden_dim, heads=1,
                             dropout=dropout, concat=False)
        self.bn2 = nn.BatchNorm1d(hidden_dim)           # BN(128)

        self.dropout_rate = dropout

    def forward(self, x, edge_index):
        """
        输入: x (N, 10) 标准化后的游戏类型分布
             edge_index (2, E) 图的边

        输出: h (N, 128) 节点嵌入向量

        每一步:
          1. 把 10 维输入投影到 128 维
          2. 第一层 GAT: 每个节点从邻居收集信息，用注意力加权
          3. 第二层 GAT: 融合多头信息，输出最终嵌入
        """
        h = self.input_proj(x)
        h = F.dropout(h, p=self.dropout_rate, training=self.training)

        h = F.relu(self.bn1(self.conv1(h, edge_index)))
        h = F.dropout(h, p=self.dropout_rate, training=self.training)

        h = F.relu(self.bn2(self.conv2(h, edge_index)))
        h = F.dropout(h, p=self.dropout_rate, training=self.training)

        return h


# ============================================================================
#  GCNEncoder — 图卷积网络编码器（对比基线）
# ============================================================================

class GCNEncoder(nn.Module):
    """
    图卷积网络编码器，作为 GAT 的对比基线。

    GCN 的聚合方式：所有邻居等权重相加，权重由图的度决定。
    公式：h' = ReLU(W * sum(h_j / sqrt(d_i * d_j)))
    这个权重是固定的，不可学习。GAT 的权重是可学习的注意力系数。

    GCN 的优点：
      - 参数更少，不容易过拟合（对数据量不足的小众类型可能更友好）
      - 计算更快（没有注意力矩阵乘法和 softmax）
      - 适合图结构本身就很有信息量的情况

    在 main.py 中用 --encoder gcn 可以切换到这个编码器。
    """

    def __init__(self, in_dim=10, hidden_dim=128, out_dim=128, dropout=0.5):
        super().__init__()
        self.input_proj = BaseLayer2D(in_dim, hidden_dim, dropout=0.0, act="relu")

        self.conv1 = GCNConv(hidden_dim, hidden_dim * 2)
        self.bn1 = nn.BatchNorm1d(hidden_dim * 2)

        self.conv2 = GCNConv(hidden_dim * 2, out_dim)
        self.bn2 = nn.BatchNorm1d(out_dim)

        self.dropout_rate = dropout

    def forward(self, x, edge_index):
        h = self.input_proj(x)
        h = F.dropout(h, p=self.dropout_rate, training=self.training)
        h = F.relu(self.bn1(self.conv1(h, edge_index)))
        h = F.dropout(h, p=self.dropout_rate, training=self.training)
        h = F.relu(self.bn2(self.conv2(h, edge_index)))
        h = F.dropout(h, p=self.dropout_rate, training=self.training)
        return h


# ============================================================================
#  SageEncoder — GraphSAGE 编码器（大图备用）
# ============================================================================

class SageEncoder(nn.Module):
    """
    GraphSAGE 编码器，设计用于处理超大规模图。

    和 GAT/GCN 的区别：GraphSAGE 不要求全图参与计算，可以对每个节点
    采样固定数量的邻居。这让它适合百万级用户的大图（全图计算太慢）。

    目前项目中还没用到，但预留了这个选项。以后如果接入百万级 Steam
    用户数据，这个编码器会比 GAT 更实用。
    """

    def __init__(self, in_dim=10, hidden_dim=128, out_dim=128, dropout=0.5):
        super().__init__()
        self.input_proj = BaseLayer2D(in_dim, hidden_dim, dropout=0.0, act="relu")
        self.conv1 = SAGEConv(hidden_dim, hidden_dim * 2)
        self.bn1 = nn.BatchNorm1d(hidden_dim * 2)
        self.conv2 = SAGEConv(hidden_dim * 2, out_dim)
        self.bn2 = nn.BatchNorm1d(out_dim)
        self.dropout_rate = dropout

    def forward(self, x, edge_index):
        h = self.input_proj(x)
        h = F.dropout(h, p=self.dropout_rate, training=self.training)
        h = F.relu(self.bn1(self.conv1(h, edge_index)))
        h = F.dropout(h, p=self.dropout_rate, training=self.training)
        h = F.relu(self.bn2(self.conv2(h, edge_index)))
        h = F.dropout(h, p=self.dropout_rate, training=self.training)
        return h


# ============================================================================
#  build_encoder — 编码器工厂函数
# ============================================================================

def build_encoder(encoder_type="gat", **kwargs):
    """
    按名字创建编码器。在 main.py 中通过 --encoder 参数切换：
      --encoder gat  -> GATEncoder（默认，效果最好）
      --encoder gcn  -> GCNEncoder（轻量对比基线）
      --encoder sage -> SageEncoder（大图备用）
    """
    registry = {
        "gat":  GATEncoder,
        "gcn":  GCNEncoder,
        "sage": SageEncoder,
    }
    if encoder_type not in registry:
        raise ValueError(f"未识别的编码器类型: {encoder_type}，可选: {list(registry.keys())}")
    return registry[encoder_type](**kwargs)


# ============================================================================
#  transfer_pretrained_weights — 预训练权重迁移
# ============================================================================

def transfer_pretrained_weights(encoder, checkpoint_path, device="cpu"):
    """
    把预训练好的编码器权重迁移到当前编码器。

    这个函数是为了支持"先在大规模 Steam 数据上预训练，再在自己的
    分类任务上微调"的流程。类似 NLP 中 BERT 的用法。

    【迁移策略】
      只迁移 GATConv 层的权重（conv1、conv2、bn1、bn2），
      跳过 input_proj 层。原因是：
        - 预训练的 input_proj 处理的是 10 维游戏数据
        - 当前的 input_proj 处理的也是 10 维（如果维度一致）
        - 但可能在跨任务迁移时维度不同，所以用 shape 检查兜底

    【参数】
      encoder        : 当前编码器实例
      checkpoint_path: 预训练权重文件路径（.pt）
      device         : 目标设备

    【返回】
      encoder : 加载了部分权重的编码器
      info    : 迁移统计信息（哪些加载了、哪些跳过了）
    """
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    # 兼容两种保存格式
    if "encoder_state" in checkpoint:
        pretrained_state = checkpoint["encoder_state"]
        config = checkpoint.get("config", {})
    else:
        pretrained_state = checkpoint
        config = {}

    current_state = encoder.state_dict()
    loaded, skipped, shape_mismatch = [], [], []

    for key, pretrained_param in pretrained_state.items():
        if key not in current_state:
            skipped.append(key)
            continue
        if pretrained_param.shape != current_state[key].shape:
            shape_mismatch.append(
                f"{key}: 预训练 {list(pretrained_param.shape)} vs 当前 {list(current_state[key].shape)}"
            )
            skipped.append(key)
            continue
        # 形状匹配，直接复制
        current_state[key] = pretrained_param.clone()
        loaded.append(key)

    encoder.load_state_dict(current_state)

    return encoder, {
        "total_pretrained_keys": len(pretrained_state),
        "loaded_keys": loaded,
        "skipped_keys": skipped,
        "shape_mismatch_keys": shape_mismatch,
        "pretrained_config": config,
    }


def print_transfer_report(info):
    """
    打印权重迁移的详细报告，方便判断哪些层迁移成功、哪些被跳过了。
    在 main.py 的 --pretrained 模式下会自动调用。
    """
    print(f"\n  {'='*60}")
    print(f"  预训练权重迁移报告")
    print(f"  {'='*60}")
    print(f"  预训练总键数: {info['total_pretrained_keys']}")
    print(f"  成功加载:     {len(info['loaded_keys'])}")
    print(f"  跳过:         {len(info['skipped_keys'])}")
    if info.get("pretrained_config"):
        cfg = info["pretrained_config"]
        print(f"  预训练信息:")
        print(f"    输入维度: {cfg.get('game_genre_dim', 'N/A')}（游戏类型）")
        for g in cfg.get("genre_names", []):
            print(f"      . {g}")
    if info["loaded_keys"]:
        print(f"\n  [OK] 已加载的层:")
        for k in info["loaded_keys"]:
            print(f"      {k}")
    if info["skipped_keys"]:
        print(f"\n  [SKIP] 跳过的层:")
        for k in info["skipped_keys"]:
            print(f"      {k}")
    if info["shape_mismatch_keys"]:
        print(f"\n  [WARN] 形状不匹配（已跳过）:")
        for m in info["shape_mismatch_keys"]:
            print(f"      {m}")
    print(f"  {'='*60}\n")
