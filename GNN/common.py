"""
================================================================================
  common.py — 共享基础模块
================================================================================

  这是我整个项目的"工具箱"。所有其他地方需要用到的基础神经网络组件
  都放在这里，避免在 encoder、decoder、main 里重复写同样的代码。

  我在这里放了三个东西：
    1. BaseLayer2D — 一个标准的全连接块，Linear + BN + ReLU + Dropout
    2. build_mlp   — 工厂函数，快速搭建多层 MLP
    3. MLPClassifier — 纯 MLP 分类器，用来和 GNN 对比效果

  设计原则很简单：只放真正被多处复用的代码。如果某个类只在一个地方用，
  就不放这里。这样 common.py 能保持精简。
================================================================================
"""

import torch.nn as nn


# ============================================================================
#  激活函数映射表
#  我把所有支持的激活函数名字和对应的 PyTorch 类放在这个字典里。
#  这样在 build_mlp 里只需要传字符串 "relu" 就能拿到 nn.ReLU 类。
#  以后想加新的激活函数，在这里加一行就行。
# ============================================================================

ACTIVATION_MAP = {
    "relu":        nn.ReLU,
    "gelu":        nn.GELU,
    "leaky_relu":  nn.LeakyReLU,
    "elu":         nn.ELU,
    "selu":        nn.SELU,
    "tanh":        nn.Tanh,
    "sigmoid":     nn.Sigmoid,
    "identity":    nn.Identity,     # 不做任何变换，调试时有用
}


# ============================================================================
#  BaseLayer2D — 标准全连接块
# ============================================================================

class BaseLayer2D(nn.Module):
    """
    这是我在整个项目中使用频率最高的基础组件。

    它的结构是固定的四步流水线：
      Linear(输入维度, 输出维度) -> BatchNorm1d -> 激活函数 -> Dropout

    为什么这样设计？我总结了几点：
      1. Linear 负责维度变换
      2. BatchNorm 稳定训练（减少内部协变量偏移，允许更高学习率）
      3. 激活函数引入非线性（否则多层 Linear 等于单层）
      4. Dropout 防止过拟合（随机丢弃神经元，强制模型不依赖单一路径）

    【参数说明】
      in_dim  : 输入维度，比如 128
      out_dim : 输出维度，比如 256
      dropout : 丢弃率，0.5 表示随机关掉一半神经元。默认 0 表示不丢弃
      act     : 激活函数名字，比如 "relu"、"gelu"。从 ACTIVATION_MAP 查找
      bias    : 线性层是否加偏置，默认 True

    【为什么叫 BaseLayer2D？】
      这个名字是我起的。虽然实际上是 1D 的全连接（处理的是向量而非图像），
      但我用 2D 表示它有"两层变换"的意思（Linear + Activation）。
      实际上叫 MLPBlock 或 FCBlock 更准确，但写习惯了就没改。
    """

    def __init__(self, in_dim, out_dim, dropout=0.0, act="relu", bias=True):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim, bias=bias)
        self.norm = nn.BatchNorm1d(out_dim)
        # 从字典查激活函数，找不到就用 GELU 兜底
        self.act = ACTIVATION_MAP.get(act, nn.GELU)()
        # dropout > 0 才创建 Dropout 层，否则用 Identity 占位（什么都不做）
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x):
        """
        输入 x: (batch_size, in_dim)  比如 (32, 128)
        输出  : (batch_size, out_dim)  比如 (32, 256)
        """
        return self.dropout(self.act(self.norm(self.linear(x))))


# ============================================================================
#  build_mlp — 快速搭建多层 MLP
# ============================================================================

def build_mlp(in_dim, hidden_dims, out_dim, dropout=0.0, act="relu", bias=True):
    """
    这是我写的工厂函数，用来快速搭一个多层感知机。

    为什么不用 nn.Sequential 直接写？因为写多了会发现模式高度重复：
    每次都写 nn.Linear -> nn.BatchNorm1d -> nn.ReLU -> nn.Dropout，很啰嗦。
    于是我封装了 BaseLayer2D，再用这个函数循环生成。

    【参数说明】
      in_dim      : 输入特征维度，比如 10（10 种游戏类型）
      hidden_dims : 隐藏层维度列表，比如 [128, 64] 表示两个隐藏层
      out_dim     : 输出维度，比如 5（5 个玩家类型）
      dropout     : 每一层后的 Dropout 率
      act         : 激活函数
      bias        : 是否用偏置

    【使用示例】
      # 构建 10->128->64->5 的三层 MLP
      mlp = build_mlp(in_dim=10, hidden_dims=[128, 64], out_dim=5, dropout=0.3)

    【注意事项】
      最后一层（输出层）不加激活函数和 Dropout。因为输出是原始 logits，
      后续交给 CrossEntropyLoss 处理，它内部会做 softmax。
      如果在输出层前加 ReLU，会截断负数，导致梯度无法正常回传。
    """
    layers = []
    prev_dim = in_dim

    for h_dim in hidden_dims:
        layers.append(BaseLayer2D(prev_dim, h_dim, dropout=dropout, act=act, bias=bias))
        prev_dim = h_dim

    # 输出层：纯线性，不做激活和 Dropout
    layers.append(nn.Linear(prev_dim, out_dim, bias=bias))

    return nn.Sequential(*layers)


# ============================================================================
#  MLPClassifier — 纯 MLP 分类器（GNN 的对比基线）
# ============================================================================

class MLPClassifier(nn.Module):
    """
    这是我用来和 GNN 对比的纯 MLP 基线。

    它的作用和 GNNClassifier 完全一样：输入用户特征，输出 5 类预测。
    唯一的区别是它完全不看图的边信息，只看单个用户自己的特征。

    【对比意义】
      如果 GNN 的准确率 > MLP 的准确率：
        -> 图结构（用户之间的相似性）确实提供了额外信息，GNN 有价值
      如果 GNN 的准确率 ≈ MLP 的准确率：
        -> 分类信息全在单个用户的特征里，图没帮上忙，需要检查建图方式

    【forward 签名兼容性】
      我故意让 forward(self, x, edge_index=None) 接受 edge_index 参数，
      但实际上完全不用。这是为了和 GNNClassifier 保持相同的调用接口，
      在 main.py 的 train_model 里可以无差别调用 MLP 和 GNN。
    """

    def __init__(self, in_dim=10, hidden_dim=128, out_dim=5, dropout=0.5):
        super().__init__()
        # 两层隐藏层的 MLP：输入 -> 128 -> 128 -> 输出
        self.net = build_mlp(
            in_dim=in_dim,
            hidden_dims=[hidden_dim, hidden_dim],
            out_dim=out_dim,
            dropout=dropout,
            act="relu",
        )

    def forward(self, x, edge_index=None):
        """
        x: (N, in_dim) 节点特征矩阵
        edge_index: 图边索引（不使用，仅为接口兼容）

        返回: (N, out_dim) 各类别 logits
        """
        return self.net(x)
