"""
================================================================================
  main.py — GNN 游戏用户分类主程序
================================================================================

  这是我整个 GNN 项目的入口。所有训练、评估、预测的流程都在这里编排。

  我设计的核心流程是：
    1. 从 Kaggle 加载真实 Steam 用户数据（10 维游戏类型分布）
    2. 用 KMeans 聚类自动给用户打标签（5 类）
    3. 构建 k-NN 用户相似度图
    4. 用 GAT（图注意力网络）训练分类器
    5. 评估、可视化、交互预测

  运行方式:
    python main.py --train                    # 训练模型
    python main.py --predict                  # 交互预测
    python main.py --train --compare-mlp      # GAT vs MLP 对比
    python main.py --train --encoder gcn      # 使用 GCN 编码器
    python main.py --train --decoder linear   # 使用线性解码器
================================================================================
"""

import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
import matplotlib; matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import warnings; warnings.filterwarnings("ignore")

# 从我的其他模块导入
from common import MLPClassifier
from encoder import build_encoder, transfer_pretrained_weights, print_transfer_report
from decoder import build_decoder

# ============================================================================
#  全局配置
#  我把所有超参数集中在这里，方便调参。改一个地方全局生效。
# ============================================================================

NUM_SAMPLES = 5000       # 训练使用的最大用户数，超过则随机采样
K_NEIGHBORS = 10         # k-NN 图的 k 值，每个用户连 10 个最相似用户
HIDDEN_DIM = 128         # 编码器和解码器的隐藏层维度
DROPOUT = 0.5            # Dropout 率，0.5 表示随机丢弃一半神经元
LEARNING_RATE = 0.003    # Adam 初始学习率
WEIGHT_DECAY = 5e-4      # L2 正则化系数，防止过拟合
EPOCHS = 400             # 最大训练轮数
EARLY_STOP_PATIENCE = 60 # 早停：连续 60 轮验证准确率不涨就停止
RANDOM_SEED = 42         # 固定随机种子，确保每次训练结果可复现
NUM_CLASSES = 5          # 分类类别数（5 种玩家类型）

np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ============================================================================
#  游戏类型定义
#  10 种类型覆盖了 Steam 上绝大多数热门游戏
# ============================================================================

GENRE_NAMES = [
    "Action", "RPG", "Strategy", "Simulation", "Casual",
    "Social/MMO", "Sports", "Adventure", "Puzzle", "Indie",
]
GAME_GENRE_DIM = len(GENRE_NAMES)   # 10 维
INPUT_DIM = GAME_GENRE_DIM          # 编码器输入维度，现在是 10


# ============================================================================
#  Steam 游戏 -> 类型映射表
#  我用关键词匹配来给每个游戏归类。这 150+ 个关键词覆盖了 Steam 上
#  最主流的游戏，其余匹配不到的都归到 Indie（兜底分类）。
#  映射逻辑来自 SteamDB 和 SteamSpy 的公开标签数据。
# ============================================================================

_GENRE_KEYWORDS = {
    0: [  # Action
        "counter-strike", "cs:go", "call of duty", "battlefield", "doom",
        "overwatch", "valorant", "apex", "rainbow six", "far cry", "gta",
        "grand theft auto", "payday", "warframe", "destiny", "borderlands",
        "titanfall", "bioshock", "wolfenstein", "halo", "monster hunter",
    ],
    1: [  # RPG
        "elder scrolls", "skyrim", "fallout", "witcher", "dark souls",
        "elden ring", "dragon age", "mass effect", "baldur", "diablo",
        "final fantasy", "dragon quest", "persona", "nier", "cyberpunk",
        "path of exile", "guild wars", "starfield",
    ],
    2: [  # Strategy
        "civilization", "age of empires", "starcraft", "total war", "xcom",
        "crusader kings", "stellaris", "factorio", "rimworld", "anno",
        "tropico", "cities: skylines", "frostpunk", "hades", "dead cells",
        "slay the spire", "into the breach", "ftl",
    ],
    3: [  # Simulation
        "simulator", "sims", "euro truck", "flight sim", "kerbal",
        "farm", "space engineers",
    ],
    4: [  # Casual
        "stardew", "minecraft", "roblox", "among us", "rocket league",
        "lego", "hollow knight", "celeste", "cuphead", "overcooked",
        "spelunky", "terraria",
    ],
    5: [  # Social / MMO
        "dota", "league of legends", "world of warcraft", "final fantasy xiv",
        "guild wars 2", "lost ark", "destiny 2", "runescape", "vrchat",
        "rust", "pubg", "fortnite", "dayz", "ark:",
    ],
    6: [  # Sports
        "fifa", "nba", "nfl", "madden", "forza", "gran turismo",
        "football manager", "wwe", "ufc",
    ],
    7: [  # Adventure
        "tomb raider", "assassin", "red dead", "last of us", "spider-man",
        "batman", "resident evil", "silent hill", "dead space", "uncharted",
        "horizon", "zelda", "god of war", "death stranding",
    ],
    8: [  # Puzzle
        "portal", "witness", "braid", "fez", "limbo", "inside",
        "poly bridge", "besiege", "baba is you",
    ],
}

# 将映射表扁平化为 {关键词: 类型ID} 字典，方便快速查找
_FLAT_MAP = {}
for _gid, _kws in _GENRE_KEYWORDS.items():
    for _kw in _kws:
        _FLAT_MAP[_kw] = _gid


def _match_genre(title):
    """
    根据游戏标题匹配类型。用大小写无关的包含匹配。
    未匹配到任何关键词的默认归为 Indie（类型 9）。
    """
    t = title.lower()
    for kw, gid in _FLAT_MAP.items():
        if kw in t:
            return gid
    return 9   # 默认：Indie


# ============================================================================
#  Steam 数据加载
#  这是我费了最多功夫的部分。从 Kaggle 下载真实数据，
#  清洗、过滤、归一化，最终产出 (N,10) 的用户类型分布矩阵。
# ============================================================================

def load_steam_data(n_samples=NUM_SAMPLES):
    """
    从 Kaggle 加载真实 Steam 用户数据。

    优先使用 kagglehub 自动下载（首次运行会自动缓存到 ~/.cache/kagglehub），
    也支持本地 CSV 文件（手动下载放到项目根目录）。

    数据处理流程：
      1. 加载 CSV（user_id, game_title, behavior, hours）
      2. 只保留 play 行为（purchase 不算）
      3. 用 _match_genre() 给每条记录分配类型
      4. 按用户聚合：每个用户得到 10 维类型时长向量
      5. 过滤：至少 2 种类型、总时长 > 2 小时才保留
      6. L1 归一化：每用户各类型时长 / 总时长（得到"分布"）
      7. 如果用户数超过 n_samples，随机采样

    返回: (n, 10) numpy 数组，或 None（加载失败）
    """
    import pandas as pd
    import os

    df = None

    # 策略 1：检查本地 CSV
    for p in [r"steam-200k.csv", "steam_video_games.csv"]:
        if os.path.exists(p):
            print(f"  [找到本地文件: {p}]")
            df = pd.read_csv(p)
            break

    # 策略 2：kagglehub 自动下载
    if df is None:
        try:
            import kagglehub
            print("  [正在通过 kagglehub 下载 Steam 数据集...]")
            dl = kagglehub.dataset_download("tamber/steam-video-games")
            csv = os.path.join(dl, "steam-200k.csv")
            if os.path.exists(csv):
                df = pd.read_csv(csv)
                print(f"  [下载完成: {csv}]")
        except Exception as e:
            print(f"  [!] kagglehub 下载失败: {e}")

    if df is None:
        print("  [!] Steam 数据加载失败，将回退到模拟数据")
        return None

    # 清理：前 4 列是有效数据，多余的忽略
    df = df.iloc[:, :4]
    df.columns = ["uid", "game", "action", "hours"]
    df["hours"] = pd.to_numeric(df["hours"], errors="coerce").fillna(0)

    # 只保留 play 行为（购买不玩的记录没有分析价值）
    df = df[df["action"] == "play"].copy()
    if len(df) == 0:
        return None

    # 给每条记录打类型标签
    df["genre"] = df["game"].apply(_match_genre)
    mapped = (df["genre"] != 9).sum()
    print(f"  [类型映射: {mapped}/{len(df)} ({mapped/len(df)*100:.1f}%)]")

    # 按用户聚合：每个用户的 10 种类型总时长
    ug = df.groupby(["uid", "genre"])["hours"].sum().unstack(fill_value=0)
    for g in range(GAME_GENRE_DIM):
        if g not in ug.columns:
            ug[g] = 0.0
    ug = ug[range(GAME_GENRE_DIM)]

    # 过滤低质量用户
    valid = (ug.sum(axis=1) > 2) & ((ug > 0).sum(axis=1) >= 2)
    ug = ug[valid]
    print(f"  [有效用户: {len(ug)}]")

    if len(ug) < 500:
        return None

    # L1 归一化：转成"分布"
    X = ug.values.astype(np.float64)
    X = X / X.sum(axis=1, keepdims=True)

    # 采样
    if n_samples and len(X) > n_samples:
        idx = np.random.choice(len(X), n_samples, replace=False)
        X = X[idx]

    return X


# ============================================================================
#  数据准备：真实数据 + KMeans 聚类标签
#  真实 Steam 数据没有"玩家类型"标签，我用 KMeans 自动聚类。
#  聚类后每个簇用最高权重的 2 个游戏类型命名。
# ============================================================================

def load_real_data(n_samples=NUM_SAMPLES):
    """
    加载真实 Steam 数据，用 KMeans 聚类生成 5 类标签。

    为什么用 KMeans？
      - 真实数据没有标注，无法直接用分类
      - KMeans 根据用户的游戏偏好分布自动分组
      - 分组后可以用 GNN 学习"同一类用户在图上也相似"的模式
      - 这是一个典型的"先聚类后分类"策略

    返回: X (n, 10) 特征矩阵, y (n,) 伪标签数组
    """
    X = load_steam_data(n_samples)

    if X is None or len(X) < 500:
        print("  [!] 真实数据不可用，回退到 sklearn 模拟数据")
        return _generate_fallback_data(n_samples)

    # KMeans 聚类
    X_scaled = StandardScaler().fit_transform(X)
    kmeans = KMeans(n_clusters=NUM_CLASSES, random_state=RANDOM_SEED, n_init=10)
    y = kmeans.fit_predict(X_scaled)

    # 给每个簇起名字：取集群中心最高的 2 个类型
    cluster_names = []
    for c in range(NUM_CLASSES):
        center = kmeans.cluster_centers_[c]
        top_idx = np.argsort(center)[::-1][:2]
        top_genres = [GENRE_NAMES[i] for i in top_idx]
        cluster_names.append(f"{'+'.join(top_genres)}")

    print(f"  [Kaggle 数据加载成功: {len(X)} 用户 x {INPUT_DIM} 维]")
    for c in range(NUM_CLASSES):
        print(f"    Cluster {c} ({cluster_names[c]}): {(y == c).sum()} 人")

    return X, y


def _generate_fallback_data(n_samples):
    """
    回退方案：当 Kaggle 数据不可用时，用 sklearn 生成模拟数据。
    保证程序在任何环境下都能跑通，不至于因为网络问题直接报错退出。
    """
    from sklearn.datasets import make_classification
    X, y = make_classification(
        n_samples=n_samples, n_features=INPUT_DIM,
        n_classes=NUM_CLASSES, n_informative=8,
        random_state=RANDOM_SEED
    )
    X = np.abs(X)
    X = X / X.sum(axis=1, keepdims=True)
    return X, y


# ============================================================================
#  图构建
#  核心思想：把每个用户看成一个节点，用 k-NN 基于余弦相似度连边。
#  相似用户之间有边 -> GNN 的消息传递可以在相似用户之间传播信息。
# ============================================================================

def build_graph(features, labels, k=K_NEIGHBORS):
    """
    构建 k-NN 余弦相似度图。

    为什么要用余弦相似度而不是欧氏距离？
      欧氏距离受量纲影响大。比如 Action 玩家 A 的分布是 [0.5,0,0,0,0,...]，
      Action 玩家 B 的分布是 [0.8,0,0,0,0,...]，欧氏距离说它们差 0.3，
      但余弦相似度说它们方向几乎完全一致，都是"纯 Action 型"。
      对于游戏类型分布数据，看"偏好方向"比看"绝对数值"更有意义。

    StandardScaler 的作用：
      Steam 数据经过 L1 归一化后已经在 0-1 范围，但仍可能有维度间的方差
      差异。Scaler 让每维均值为 0、方差为 1，避免某维主导距离计算。

    返回:
      x: (N, D) 标准化特征张量
      y: (N,) 标签张量
      edge_index: (2, E) 边索引
      scaler: 训练好的 StandardScaler（预测时复用）
    """
    scaler = StandardScaler()
    features_norm = scaler.fit_transform(features)

    # k+1 个近邻（第一个是自己，后面跳过）
    nn_model = NearestNeighbors(n_neighbors=k + 1, metric="cosine")
    nn_model.fit(features_norm)
    _, indices = nn_model.kneighbors(features_norm)

    # 无向边集合（去重）
    edges = set()
    for i in range(len(features)):
        for j_idx in range(1, k + 1):
            j = indices[i, j_idx]
            u, v = (i, j) if i < j else (j, i)
            edges.add((u, v))

    edge_list = list(edges)
    # COO 格式：(2, E)，第 0 行是源节点，第 1 行是目标节点
    edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
    # 添加反向边（GNN 消息传递需要双向边）
    edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=1)

    x = torch.tensor(features_norm, dtype=torch.float32)
    y = torch.tensor(labels, dtype=torch.long)
    return x, y, edge_index, scaler


def split_graph(x, y, edge_index, test_size=0.2):
    """
    划分训练/验证集。

    关键设计：为什么用 mask 而不是直接拆图？
      GNN 的消息传递依赖全图结构。如果把验证节点从图中删掉，
      训练节点就失去了那些邻居，信息流变了，训练出的模型不准确。
      用 mask 保留全图，只标记哪些节点用于计算 loss。

    stratify 参数确保训练和验证集中各类别比例一致。
    """
    n = len(y)
    indices = np.arange(n)
    train_idx, val_idx = train_test_split(
        indices, test_size=test_size, stratify=y.numpy(), random_state=RANDOM_SEED
    )
    train_mask = torch.zeros(n, dtype=torch.bool)
    val_mask = torch.zeros(n, dtype=torch.bool)
    train_mask[train_idx] = True
    val_mask[val_idx] = True
    return x, y, edge_index, train_mask, val_mask


# ============================================================================
#  GNN 分类模型
#  编码器 + 解码器的简单组合。forward() 就是 encoder(x) -> decoder(embedding)
# ============================================================================

class GNNClassifier(nn.Module):
    """
    完整的 GNN 分类器 = 编码器 + 解码器。

    设计很简单，就是串联两个模块。复杂度都在 encoder 和 decoder 各自的
    实现中，这个类只是把它们粘在一起。

    get_embeddings() 方法用于 t-SNE 可视化，只跑编码器部分，
    提取 128 维嵌入向量。
    """
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, x, edge_index):
        """x(N,10) + edge_index -> logits(N,5)"""
        return self.decoder(self.encoder(x, edge_index))

    def get_embeddings(self, x, edge_index):
        """只提取编码器输出，用于可视化"""
        with torch.no_grad():
            return self.encoder(x, edge_index)


# ============================================================================
#  训练逻辑
#  Transductive 训练：全图前向传播，只在训练节点上计算 loss。
#  早停 + 余弦退火，防过拟合。
# ============================================================================

def train_model(model, x, y, edge_index, train_mask, val_mask,
                epochs=EPOCHS, lr=LEARNING_RATE, wd=WEIGHT_DECAY,
                patience=EARLY_STOP_PATIENCE):
    """
    GNN 训练循环。

    这是标准的直推式（transductive）训练：
      - 前向传播看到全图所有节点
      - Loss 只在 train_mask 标记的节点上计算
      - 验证也在 val_mask 节点上做

    优化器选 Adam，因为它在 GNN 训练中通常比 SGD 收敛更快。
    调度器用 CosineAnnealingLR：学习率从初始值按余弦曲线衰减到接近 0，
    训练初期大步快走，后期小步精调。

    早停 patience=60：连续 60 轮验证准确率不提升就停止。
    这个值是我经过实验选的：太短容易过早停止，太长浪费时间。
    """
    model = model.to(DEVICE)
    x = x.to(DEVICE)
    y = y.to(DEVICE)
    edge_index = edge_index.to(DEVICE)
    train_mask = train_mask.to(DEVICE)
    val_mask = val_mask.to(DEVICE)

    optimizer = Adam(model.parameters(), lr=lr, weight_decay=wd)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.CrossEntropyLoss()

    best_val_acc = 0.0
    best_state = None
    patience_counter = 0
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

    print(f"\n{'='*60}")
    print(f"  训练 {type(model).__name__}")
    print(f"{'='*60}")
    print(f"  设备: {DEVICE} | 训练节点: {train_mask.sum().item()} | "
          f"验证节点: {val_mask.sum().item()}")
    print(f"  边数: {edge_index.shape[1]} | "
          f"参数量: {sum(p.numel() for p in model.parameters()):,}")
    print("-" * 60)

    for epoch in range(epochs):
        # === 训练步 ===
        model.train()
        optimizer.zero_grad()
        logits = model(x, edge_index)
        loss = criterion(logits[train_mask], y[train_mask])
        loss.backward()
        optimizer.step()
        scheduler.step()

        # === 验证步（不计算梯度，省显存） ===
        model.eval()
        with torch.no_grad():
            v_logits = model(x, edge_index)
            v_loss = criterion(v_logits[val_mask], y[val_mask])
            train_acc = (logits[train_mask].argmax(1) == y[train_mask]).float().mean().item()
            val_acc = (v_logits[val_mask].argmax(1) == y[val_mask]).float().mean().item()

        for k, v in zip(["train_loss", "train_acc", "val_loss", "val_acc"],
                         [loss.item(), train_acc, v_loss.item(), val_acc]):
            history[k].append(v)

        # === 早停判断 ===
        if val_acc > best_val_acc + 0.001:  # 至少提升 0.1% 才算
            best_val_acc = val_acc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1

        if (epoch + 1) % 50 == 0 or epoch == 0:
            print(f"  Epoch {epoch+1:4d} | Train Loss: {loss.item():.4f} | "
                  f"Train Acc: {train_acc:.4f} | Val Loss: {v_loss.item():.4f} | "
                  f"Val Acc: {val_acc:.4f}")

        if patience_counter >= patience:
            print(f"\n  早停于 Epoch {epoch+1}")
            break

    model.load_state_dict(best_state)
    print(f"\n  最佳验证准确率: {best_val_acc:.4f}")
    print("=" * 60 + "\n")
    return model, history, best_val_acc


# ============================================================================
#  评估 & 可视化
#  三张图：训练曲线 + t-SNE 嵌入 + 混淆矩阵
# ============================================================================

def evaluate_and_plot(model, x, y, edge_index, train_mask, val_mask, scaler, history):
    """
    生成三合一可视化报告。

    图 1: 训练/验证 Loss 曲线 — 诊断过拟合（验证 loss 上升 = 过拟合）
    图 2: t-SNE 嵌入散点图 — 看编码器是否学到好的用户表示（同类聚在一起 = 好）
    图 3: 混淆矩阵 — 看模型在哪些类别之间容易混淆

    t-SNE 说明：把 128 维嵌入降到 2 维，同时尽量保持局部邻域结构。
    这不是聚类分析，只是可视化工具。看的是"同类点在 2D 空间是否聚在一起"。
    """
    model.eval()
    model = model.to(DEVICE)
    x_d = x.to(DEVICE)
    ei_d = edge_index.to(DEVICE)

    with torch.no_grad():
        logits = model(x_d, ei_d)
        preds = logits.argmax(1).cpu()
        embs = model.get_embeddings(x_d, ei_d)

    vp, vt = preds[val_mask].numpy(), y[val_mask].numpy()
    cm = confusion_matrix(vt, vp)

    # t-SNE 降维（取最多 1500 个样本加速计算）
    emb_np = embs.cpu().numpy()
    nts = min(1500, len(emb_np))
    idx_t = np.random.choice(len(emb_np), nts, replace=False)
    emb_2d = TSNE(n_components=2, random_state=RANDOM_SEED, perplexity=30).fit_transform(emb_np[idx_t])
    yt = y[idx_t].numpy()

    # 绘图
    fig, axes = plt.subplots(1, 3, figsize=(21, 5.8))
    colors = ["#4ECDC4", "#FF6B6B", "#45B7D1", "#F7DC6F", "#BB8FCE"]
    cluster_names = [n[:20] for n in cluster_names]

    axes[0].plot(history["train_loss"], label="Train", color="#4ECDC4", lw=2)
    axes[0].plot(history["val_loss"], label="Val", color="#FF6B6B", lw=2)
    axes[0].set_title("Loss"); axes[0].legend(); axes[0].grid(True, alpha=0.3)

    for i in range(NUM_CLASSES):
        m = yt == i
        axes[1].scatter(emb_2d[m, 0], emb_2d[m, 1], c=colors[i],
                        label=cluster_names[i] if i < len(cluster_names) else f"Cluster{i}",
                        alpha=0.5, s=15, edgecolors="white", lw=0.2)
    axes[1].set_title("t-SNE"); axes[1].legend(fontsize=7, ncol=2); axes[1].grid(True, alpha=0.2)

    im = axes[2].imshow(cm, cmap="YlOrRd", aspect="auto")
    sn = [n[:12] for n in cluster_names] if len(cluster_names) == NUM_CLASSES else [f"C{i}" for i in range(NUM_CLASSES)]
    axes[2].set_xticks(range(NUM_CLASSES)); axes[2].set_yticks(range(NUM_CLASSES))
    axes[2].set_xticklabels(sn, rotation=45, ha="right", fontsize=9)
    axes[2].set_yticklabels(sn, fontsize=9)
    axes[2].set_title("Confusion Matrix")
    for i in range(NUM_CLASSES):
        for j in range(NUM_CLASSES):
            axes[2].text(j, i, cm[i, j], ha="center", va="center",
                         color="white" if cm[i, j] > cm.max()/2 else "black",
                         fontsize=11, fontweight="bold")
    plt.colorbar(im, ax=axes[2], shrink=0.8)
    plt.tight_layout()
    plt.savefig("gnn_results.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  [可视化已保存到 gnn_results.png]")
    print("\n" + "-"*60 + "\n  Classification Report\n" + "-"*60)
    print(classification_report(vt, vp, target_names=sn, digits=4))


# ============================================================================
#  交互式预测
#  用户在终端输入 10 种游戏类型的游玩时长，模型实时分类。
# ============================================================================

def interactive_predict(model, scaler):
    """
    交互式 CLI 预测。

    用户输入在 10 种游戏类型中的游玩时长（小时），
    程序自动做 L1 归一化（和训练数据一致），
    然后通过 GNN 推理出用户属于哪一类玩家。

    为什么用自环边？
      新用户不在训练图中，没有邻居，最保守的做法是只连自己。
      模型只用该用户自己的特征做推理。这是标准的归纳式推理方式。
    """
    print("\n" + "="*60)
    print("  游戏玩家分类预测")
    print("="*60)
    print("  请输入你在以下 10 种游戏类型中的总游玩时长（小时）:")
    answers = []
    for g in GENRE_NAMES:
        while True:
            try:
                v = float(input(f"  {g}: "))
                if v >= 0:
                    break
                print("  [!] 请输入 >= 0 的数字")
            except ValueError:
                print("  [!] 请输入有效数字")
        answers.append(v)

    # L1 归一化（与训练数据预处理一致）
    arr = np.array([answers], dtype=np.float64)
    if arr.sum() > 0:
        arr = arr / arr.sum()
    x = torch.tensor(scaler.transform(arr), dtype=torch.float32).to(DEVICE)
    ei = torch.tensor([[0], [0]], dtype=torch.long).to(DEVICE)  # 自环边

    model.eval()
    with torch.no_grad():
        logits = model(x, ei)
        probs = F.softmax(logits, dim=1).cpu().numpy()[0]
        pred = int(logits.argmax(dim=1).cpu().item())

    print("\n" + "-"*60)
    print("  分类结果")
    print("-"*60)
    print(f"  预测类型: {cluster_names[pred] if cluster_names else f'Cluster {pred}'}")
    print("\n  各类别置信度:")
    for i in range(NUM_CLASSES):
        name = cluster_names[i] if cluster_names else f"Cluster {i}"
        filled = int(probs[i] * 30)
        bar = "#" * filled + "-" * (30 - filled)
        marker = " <--" if i == pred else ""
        print(f"  {name}: |{bar}| {probs[i]:.1%}{marker}")
    print("="*60 + "\n")
    return pred


# ============================================================================
#  主入口
#  解析命令行参数，编排完整训练/评估/预测流程。
# ============================================================================

# 存放 KMeans 聚类名称，供交互预测使用
cluster_names = None

def main():
    parser = argparse.ArgumentParser(description="GNN 游戏用户分类")
    parser.add_argument("--train", action="store_true", help="训练模型")
    parser.add_argument("--predict", action="store_true", help="交互式预测")
    parser.add_argument("--compare-mlp", action="store_true", help="对比 MLP 基线")
    parser.add_argument("--encoder", type=str, default="gat",
                        choices=["gat", "gcn", "sage"])
    parser.add_argument("--decoder", type=str, default="mlp",
                        choices=["mlp", "linear"])
    parser.add_argument("--pretrained", type=str, default=None, help="预训练权重路径")
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--samples", type=int, default=NUM_SAMPLES)
    parser.add_argument("--no-plot", action="store_true")
    args = parser.parse_args()

    if not args.train and not args.predict:
        args.train = args.predict = True

    print("\n" + "="*60)
    print("  GNN 游戏用户分类系统")
    print("  编码器: " + args.encoder.upper() + " | 解码器: " + args.decoder.upper())
    if args.pretrained:
        print("  模式: 迁移学习（加载预训练编码器）")
    print("="*60)

    # ===== [1/5] 加载数据 =====
    print("\n[1/5] 加载 Kaggle Steam 真实数据...")
    global cluster_names
    X, y = load_real_data(n_samples=args.samples)
    cluster_names = [
        f"{'+'.join(GENRE_NAMES[i] for i in np.argsort(X[y==c].mean(axis=0))[::-1][:2])}"
        for c in range(NUM_CLASSES)
    ]
    print(f"  加载了 {len(X)} 个用户样本, {INPUT_DIM} 维游戏类型分布")
    print("\n  各类别特征均值:")
    header = "  {:<30}".format("Cluster") + "".join(["{:>8}".format(g[:6]) for g in GENRE_NAMES])
    print(header)
    for lbl in range(NUM_CLASSES):
        m = y == lbl
        means = X[m].mean(axis=0)
        row = "  {:<30}".format(cluster_names[lbl][:29]) + "".join(["{:>8.3f}".format(means[i]) for i in range(INPUT_DIM)])
        print(row)

    # ===== [2/5] 构建图 =====
    print("\n[2/5] 构建用户相似度图 (k-NN)...")
    x_t, y_t, edge_index, scaler = build_graph(X, y, k=K_NEIGHBORS)
    print(f"  节点数: {x_t.shape[0]}, 边数: {edge_index.shape[1]}, "
          f"平均度数: {edge_index.shape[1]/x_t.shape[0]:.1f}")
    x_t, y_t, edge_index, train_mask, val_mask = split_graph(x_t, y_t, edge_index)

    # ===== [3/5] 构建模型 & 训练 =====
    encoder = build_encoder(args.encoder, in_dim=INPUT_DIM, hidden_dim=HIDDEN_DIM,
                            out_dim=HIDDEN_DIM, dropout=DROPOUT)
    if args.pretrained:
        print(f"\n  [迁移] 加载预训练编码器: {args.pretrained}")
        encoder, info = transfer_pretrained_weights(encoder, args.pretrained, device=DEVICE)
        print_transfer_report(info)

    decoder = build_decoder(args.decoder, embed_dim=HIDDEN_DIM, hidden_dim=HIDDEN_DIM,
                            num_classes=NUM_CLASSES, dropout=DROPOUT)
    model = GNNClassifier(encoder, decoder).to(DEVICE)

    if args.train:
        pt_tag = " [from pretrained]" if args.pretrained else ""
        print(f"\n[3/5] 训练模型 (encoder={args.encoder.upper()}, "
              f"decoder={args.decoder.upper()}){pt_tag}...")
        model, gnn_hist, gnn_acc = train_model(model, x_t, y_t, edge_index,
                                                 train_mask, val_mask, epochs=args.epochs)

        if args.compare_mlp:
            print("\n[3b/5] 训练 MLP 基线（对比）...")
            mlp = MLPClassifier(in_dim=INPUT_DIM, hidden_dim=HIDDEN_DIM,
                                out_dim=NUM_CLASSES, dropout=DROPOUT)
            mlp, mlp_hist, mlp_acc = train_model(mlp, x_t, y_t, edge_index,
                                                  train_mask, val_mask, epochs=args.epochs)
            diff = (gnn_acc - mlp_acc) * 100
            print(f"  >>> GNN ({args.encoder.upper()}) 准确率: {gnn_acc:.4f}  "
                  f"vs  MLP 准确率: {mlp_acc:.4f}")
            print(f"  >>> GNN 提升: {diff:+.1f} 个百分点")

        torch.save({
            "encoder_state": encoder.state_dict(),
            "decoder_state": decoder.state_dict(),
            "scaler": scaler,
            "cluster_names": cluster_names,
        }, "gnn_classifier.pt")
        print("  [模型已保存到 gnn_classifier.pt]")

        # ===== [4/5] 评估 =====
        if not args.no_plot:
            print("\n[4/5] 评估与可视化...")
            evaluate_and_plot(model, x_t, y_t, edge_index, train_mask, val_mask, scaler, gnn_hist)
        else:
            model.eval()
            with torch.no_grad():
                logits = model(x_t.to(DEVICE), edge_index.to(DEVICE))
                val_acc = (logits[val_mask].argmax(1).cpu() == y_t[val_mask]).float().mean().item()
            print(f"\n  验证准确率: {val_acc:.4f}")

    else:
        # 加载已有模型做预测
        print("\n[3/5] 加载已保存的模型...")
        try:
            ckpt = torch.load("gnn_classifier.pt", map_location=DEVICE, weights_only=False)
            encoder.load_state_dict(ckpt["encoder_state"])
            decoder.load_state_dict(ckpt["decoder_state"])
            scaler = ckpt["scaler"]
            cluster_names = ckpt.get("cluster_names", [f"Cluster{i}" for i in range(NUM_CLASSES)])
            model = model.to(DEVICE)
            print("  [模型已加载]")
        except FileNotFoundError:
            print("  [!] 未找到 gnn_classifier.pt，请先运行 --train")
            return

    # ===== [5/5] 交互预测 =====
    if args.predict:
        print("\n[5/5] 交互式预测...")
        while True:
            interactive_predict(model, scaler)
            if input("  继续预测? (y/n): ").strip().lower() != "y":
                print("  再见!\n")
                break


if __name__ == "__main__":
    main()
