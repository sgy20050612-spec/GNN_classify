"""
================================================================================
  decoder.py — 分类解码器
================================================================================

  在 GNN 的架构里，解码器负责"决策"：把编码器生成的 128 维嵌入向量
  映射为 5 个类别的预测分数。它和 encoder.py 是搭档关系：
    encoder: 特征 -> 嵌入（理解数据）
    decoder: 嵌入 -> 类别（做出判断）

  我在这里放了两个解码器：
    MLPDecoder  — 两层 MLP，非线性决策边界，效果最好
    LinearDecoder — 单层线性，简单快速，适合嵌入质量已经很高的情况
================================================================================
"""

import torch.nn as nn
from common import build_mlp


# ============================================================================
#  MLPDecoder — 多层 MLP 分类头
# ============================================================================

class MLPDecoder(nn.Module):
    """
    这是默认的解码器，和 encoder 配合使用。

    结构：Embedding(128) -> FC(128) -> FC(128) -> FC(5)
    中间两层 ReLU + BN + Dropout，最后一层纯线性输出 logits。

    为什么用两层而不是一层？
      单层线性意味着各类别之间的决策边界是线性的。
      两层 MLP 可以学出弯曲的边界，对不同类型之间的"模糊地带"
      处理得更好。而且 128->128->5 参数量不大（约 33K），
      不太担心过拟合。

    我参考了典型的 GNN 论文中的做法：编码器负责消息传递（复杂），
    解码器保持简单（最多 2-3 层 MLP），把学习能力集中在编码器上。
    """

    def __init__(self, embed_dim=128, hidden_dim=128, num_classes=5, dropout=0.5):
        super().__init__()
        self.net = build_mlp(
            in_dim=embed_dim,
            hidden_dims=[hidden_dim, hidden_dim],
            out_dim=num_classes,
            dropout=dropout,
            act="relu",
        )

    def forward(self, embeddings):
        """
        输入: (N, 128) 编码器输出的嵌入向量
        输出: (N, 5)   各类别 logits（未经 softmax）
        """
        return self.net(embeddings)


# ============================================================================
#  LinearDecoder — 单层线性解码器
# ============================================================================

class LinearDecoder(nn.Module):
    """
    一个极简的解码器，只做一次线性变换。

    适用场景：
      - 编码器产出的嵌入已经非常干净，各类别在嵌入空间中线性可分
      - 做快速实验，不想调试 MLP 的超参数
      - 作为对比基线：MLPDecoder 比 LinearDecoder 好多少？

    如果 LinearDecoder 的准确率和 MLPDecoder 差不多，说明编码器很强，
    已经学到了线性可分的表示。如果差很多，说明非线性决策边界确实有需要。
    """

    def __init__(self, embed_dim=128, num_classes=5):
        super().__init__()
        self.linear = nn.Linear(embed_dim, num_classes)

    def forward(self, embeddings):
        return self.linear(embeddings)


# ============================================================================
#  build_decoder — 解码器工厂函数
# ============================================================================

def build_decoder(decoder_type="mlp", **kwargs):
    """
    按名字创建解码器。这样在 main.py 里可以通过命令行参数切换：
      --decoder mlp     -> 用 MLPDecoder
      --decoder linear  -> 用 LinearDecoder

    以后想加新的解码器（比如 Transformer Decoder），只需要：
      1. 在这里的 registry 里加一行
      2. 在上面写对应的类
      3. main.py 的 argparse choices 加一个选项
    不需要改任何其他代码。
    """
    registry = {
        "mlp":    MLPDecoder,
        "linear": LinearDecoder,
    }
    if decoder_type not in registry:
        raise ValueError(f"未识别的解码器类型: {decoder_type}，可选: {list(registry.keys())}")
    return registry[decoder_type](**kwargs)
