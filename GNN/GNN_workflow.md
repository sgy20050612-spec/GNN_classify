#For the English version, please check the content after line 164. The Chinese version at the beginning is kept as a reference for my future model optimization.
# GNN 游戏用户分类 — 开发流程

我做这个个人项目的初衷主要是为了探索如何将所学习的ai中的图神经网络（GNN）技术应用到游戏开发的环节之中，这是我个人的第一次尝试，且还在进行之中。后续还希望在游戏画面中使用类似于时序分析来做一些画面帧率处理（例如：预测玩家操作输入（而非画面本身），提前渲染并传输可能的未来帧，在服务器端用LSTM分析历史操作日志的时间模式，来预测最可能的下一个操作，从而减少需要提前渲染和传输的帧数量。【这个设想基于个人在所学习的领域的相关知识，预计在完成当前阶段学业任务和其他项目后会开始尝试实践】）

## 项目目标

使用图神经网络(GAT)基于 Steam 真实用户数据将玩家分为 5 种类型。

## 项目结构

```
npc_ai/
├── common.py          # 基础层（BaseLayer2D, MLPClassifier）
├── encoder.py         # GNN 编码器（GAT/GCN/GraphSAGE）+ 权重迁移函数
├── decoder.py         # 分类解码器（MLP/Linear）
├── main.py            # 主程序（数据加载 + 训练 + 可视化 + 交互预测）
└── GNN开发流程.md      # 本文档
```

## 整体流程

```
Kaggle Steam 真实数据 -> 游戏名匹配类型 -> 用户聚合(10维) -> KMeans聚类标签 -> GAT分类 -> 交互预测
```

## 一、数据准备

### 1.1 数据来源
- 来源：Kaggle `tamber/steam-video-games`（11,350 用户 x 3,600 游戏 x 200,000 条记录）
- 获取链接：https://www.kaggle.com/datasets/tamber/steam-video-games
- 下载方式：kagglehub 自动下载，首次运行自动缓存

### 1.2 数据处理流程
1. 下载原始 CSV（user_id, game_title, behavior, hours）
2. 用内置的 150+ 游戏关键词映射表将每个游戏归类到 10 种类型（Action/RPG/Strategy/Simulation/Casual/Social_MMO/Sports/Adventure/Puzzle/Indie）
3. 按用户聚合：统计每个用户在 10 种类型上的总游玩时长
4. L1 归一化：每个用户的 10 维向量除以总时长，得到"类型时长分布"
5. 过滤：至少玩过 2 种类型、总时长 > 2 小时的用户
6. KMeans 聚类：将用户自动分为 5 类，每类用最高权重的 2 个类型命名

### 1.3 加载说明
- main.py 中 load_steam_data() 负责加载数据
- 优先使用 kagglehub 自动下载，失败时尝试本地 CSV
- 加载失败时自动回退到 sklearn 生成模拟数据

## 二、图构建

- 节点 = 用户，边 = k-NN 余弦相似度（k=10）
- 先 StandardScaler 标准化（消除量纲差异）
- 余弦距离衡量"游戏偏好模式相似度"（看方向不看绝对值）
- 构建无向图，添加反向边供 GNN 消息传递

## 三、模型架构

### 3.1 编码器（encoder.py）
```
Input(10维) -> InputProjection -> GATConv x 2 -> Embedding(128维)
```
- 第 1 层 GAT：4 注意力头，输出 128x4=512 维
- 第 2 层 GAT：1 注意力头，输出 128 维
- 每层后有 BatchNorm + Dropout

### 3.2 解码器（decoder.py）
```
Embedding(128) -> MLP(128->128) -> MLP(128->5) -> Class Logits
```

## 四、训练结果
 如图gnn_results.png所展示的

- 使用 3,000 真实 Steam 用户，KMeans 聚类为 5 类
- 聚类结果示例：
  - Cluster0: Action+Adventure（812 人）
  - Cluster1: Indie+RPG（1440 人）
  - Cluster2: Social/MMO+Sports（509 人）
  - Cluster3: Strategy+RPG（179 人）
  - Cluster4: Sports+Simulation（60 人）
- 训练配置：CrossEntropyLoss + Adam + CosineAnnealingLR，早停 patience=60
- **验证准确率：96.8%**

## 五、交互预测

- 终端输入在 10 种游戏类型中的游玩时长（小时）
- 自动 L1 归一化（与训练数据一致）
- 模型推理输出玩家所属的类型 + 置信度
- 运行方式：`python main.py --predict`

## 六、关键技术决策

| 决策 | 选择 | 原因 |
|------|------|------|
| GNN 层类型 | GAT | 注意力机制比 GCN 等权聚合更灵活 |
| 图构建方式 | k-NN 余弦 | 适合连续特征，衡量回答模式相似度 |
| 标签生成 | KMeans 聚类 | 无监督自动分组，避免人工标注 |
| 数据源 | 真实 Steam 数据 | 比合成数据更接近实际玩家行为 |
| 框架 | PyTorch + PyG | 生态成熟，GATConv 开箱即用 |

## 七、当前进度

目前处于**模型训练完成阶段**，已使用 Kaggle Steam 真实数据进行训练，验证准确率 96.8%。交互预测功能可用。后续可在数据层面和算法层面继续优化。

## 八、优化方案

### 8.1 数据层面

最大的问题：游戏类型映射覆盖率仅 52%，近一半游戏未能归类。小众类型（Puzzle 23人、Casual 23人、Adventure 79人）样本量严重不足。

**优化方向 A：提高游戏类型映射覆盖率**
- 接入 IGDB API 或 SteamSpy 数据库，用游戏 ID 精确查询类型（但隐私和权限问题可能限制可行性）

- **使用句向量提高映射覆盖率**：对未匹配的 48% 游戏，用 NLP 方法做分类。AI 给我提供的思路是用 TF-IDF + 余弦相似度。我的想法是：
  1. 用句向量（embedding + 平均池化）替代 TF-IDF，效果更好但计算量更大，可以先尝试
  2. 用玩家评论做 NLP 比用标题更好：评论更客观，是玩家替我们分好了类，更多的文字内容在 NLP 处理后可以得到更精确的分类
- **引入 Steam 官方标签系统**：标签比类型更细粒度（如"Roguelike"、"Open World"），只需要在分类标签中添加更多类别即可

#### 优化方向 B：扩充样本量

- 使用 Kaggle 上完整的 Steam Reviews 数据集（百万级用户）
- 引入 SteamSpy 的估算数据

#### 优化方向 C：丰富特征维度

- 扩展到 20-30 种 Steam 标签
- 加入用户行为特征：游戏频率、平均时长、付费倾向
- 加入时间维度：用 LSTM/RNN 分析玩家游戏类型偏好的变化趋势

### 8.2 算法层面（部分由ai指导）

当前问题：预训练任务为掩码特征重建，虽有效但较简单。下游任务 GAT + MLP 在合成数据上 93%，实际泛化能力未知。

#### 优化方向 A：更强的预训练任务

- **对比学习（Contrastive Learning）**：随机扰动同一用户的特征，让 GNN 学会"扰动前后是同一个用户"
- **子图掩码（GraphMAE）**：不仅掩盖节点特征，还随机掩盖图中的边，重建被删掉的边
- **多任务预训练**：同时做掩码重建 + 类型分类 + 图结构重建，学到更丰富的表示

#### 优化方向 B：更强的编码器

- **Graph Transformer**：替换 GAT，引入全局注意力，不只是邻居（相当于引入 Transformer 架构）
- **增加 GNN 层数**：从当前 2 层扩展到 3-4 层，使用连续的残差连接，提升深度和准确度
- **Heterogeneous GNN**：区分"用户-用户"边和"用户-游戏"边，构建异构图（学习异构图的构建方法）【ai提供的思路，还在学习】

#### 优化方向 C：更好的训练策略

- **学习率 Warmup + Cosine Decay**：当前仅使用 Cosine Decay，加入 Warmup 能让训练初期更稳定
- **Mixup 数据增强**：在特征空间线性插值两个用户，生成新的训练样本
- **Label Smoothing**：软标签替代硬标签，减轻过拟合
- **5-fold 交叉验证**：当前仅单次 train/val 划分，无法评估稳定性。可以对 channel 进行随机 dropout（每折随机 drop 10% 的 channel），进行 5 折实验取平均值，提升泛化性

### 8.3 评估层面
- 引入真实问卷数据验证，可以增加更多问题（10 个或更多）
- A/B 测试：GAT vs GCN vs GraphSAGE vs 纯 MLP
- 消融实验：去掉图结构 / 去掉注意力，逐一验证各模块贡献
- 可视化分析：用 t-SNE 对比训练前后嵌入分布

---

## (English Translation)

# GNN Game User Classification — Development Process

I started this personal project primarily to explore how to apply the Graph Neural Network (GNN) techniques I learned in AI to game development. This is my first attempt and is still in progress. In the future, I also hope to use time-series analysis for frame rate processing in game graphics (for example: predicting player input operations rather than the screen itself, pre-rendering and transmitting possible future frames, using LSTM on the server side to analyze time patterns in historical operation logs to predict the most likely next operation, thereby reducing the number of frames that need to be pre-rendered and transmitted). [This idea is based on my knowledge in the relevant fields I have studied, and I expect to start experimenting with it after completing my current academic tasks and other projects.]

## Project Objective

Use Graph Neural Networks (GAT) to classify players into 5 types based on real Steam user data.

## Project Structure

```
npc_ai/
├── common.py          # Base layers (BaseLayer2D, MLPClassifier)
├── encoder.py         # GNN Encoders (GAT/GCN/GraphSAGE) + weight transfer functions
├── decoder.py         # Classification decoders (MLP/Linear)
├── main.py            # Main program (data loading + training + visualization + interactive prediction)
└── GNN_workflow.md      # This document
```

## Overall Flow

```
Kaggle Steam real data -> game title type matching -> user aggregation (10-dim) -> KMeans clustering labels -> GAT classification -> interactive prediction
```

## 1. Data Preparation

### 1.1 Data Source
- Source: Kaggle `tamber/steam-video-games` (11,350 users x 3,600 games x 200,000 records)
- Download link: https://www.kaggle.com/datasets/tamber/steam-video-games
- Download method: kagglehub automatic download, auto-cached on first run

### 1.2 Data Processing Pipeline
1. Download raw CSV (user_id, game_title, behavior, hours)
2. Use built-in 150+ game keyword mapping table to classify each game into 10 genres (Action/RPG/Strategy/Simulation/Casual/Social_MMO/Sports/Adventure/Puzzle/Indie)
3. Aggregate by user: calculate each user's total playtime across the 10 genres
4. L1 normalization: divide each user's 10-dim vector by total playtime to obtain "genre time distribution"
5. Filter: keep users who have played at least 2 genres and have total playtime > 2 hours
6. KMeans clustering: automatically group users into 5 categories, each named by its 2 highest-weight genres

### 1.3 Loading Notes
- `load_steam_data()` in main.py is responsible for loading the data
- Prioritizes kagglehub automatic download, falls back to local CSV on failure
- Falls back to sklearn-generated simulation data if loading fails entirely

## 2. Graph Construction

- Nodes = users, Edges = k-NN cosine similarity (k=10)
- StandardScaler normalization first (eliminates dimensional differences)
- Cosine distance measures "game preference pattern similarity" (looks at direction, not absolute values)
- Build undirected graph, add reverse edges for GNN message passing

## 3. Model Architecture

### 3.1 Encoder (encoder.py)
```
Input(10-dim) -> InputProjection -> GATConv x 2 -> Embedding(128-dim)
```
- Layer 1 GAT: 4 attention heads, output 128x4=512 dim
- Layer 2 GAT: 1 attention head, output 128 dim
- BatchNorm + Dropout after each layer

### 3.2 Decoder (decoder.py)
```
Embedding(128) -> MLP(128->128) -> MLP(128->5) -> Class Logits
```

## 4. Training Results
As shown in gnn_results.png

- Used 3,000 real Steam users, KMeans clustered into 5 categories
- Sample clustering results:
  - Cluster0: Action+Adventure (812 users)
  - Cluster1: Indie+RPG (1440 users)
  - Cluster2: Social/MMO+Sports (509 users)
  - Cluster3: Strategy+RPG (179 users)
  - Cluster4: Sports+Simulation (60 users)
- Training configuration: CrossEntropyLoss + Adam + CosineAnnealingLR, early stopping patience=60
- **Validation accuracy: 96.8%**

## 5. Interactive Prediction

- Input playtime (hours) in each of the 10 game genres via terminal
- Automatic L1 normalization (consistent with training data)
- Model inference outputs the user's predicted type + confidence scores
- Usage: `python main.py --predict`

## 6. Key Technical Decisions

| Decision | Choice | Reason |
|----------|--------|--------|
| GNN layer type | GAT | Attention mechanism is more flexible than GCN's equal-weight aggregation |
| Graph construction | k-NN cosine | Suitable for continuous features, measures pattern similarity |
| Label generation | KMeans clustering | Unsupervised automatic grouping, avoids manual labeling |
| Data source | Real Steam data | Closer to actual player behavior than synthetic data |
| Framework | PyTorch + PyG | Mature ecosystem, GATConv works out of the box |

## 7. Current Progress

Currently in the **model training completed phase**. Kaggle Steam real data has been used for training, with validation accuracy of 96.8%. Interactive prediction functionality is available. Further optimization can be done at both the data level and algorithm level.

## 8. Optimization Plan

### 8.1 Data Level

The biggest problem: game genre mapping coverage is only 52%, nearly half of all games are not classified. Niche genres (Puzzle 23 users, Casual 23 users, Adventure 79 users) have severely insufficient sample sizes.

**Optimization Direction A: Improve game genre mapping coverage**
- Connect to IGDB API or SteamSpy database to query genres precisely by game ID (but privacy and permission issues may limit feasibility)

- **Use sentence vectors to improve mapping coverage**: For the unmatched 48% of games, use NLP methods for classification. The AI suggested using TF-IDF + cosine similarity. My thoughts:
  1. Use sentence vectors (embedding + mean pooling) instead of TF-IDF; better results but higher computation cost, worth trying first
  2. Using player reviews for NLP is better than using titles: reviews are more objective, players have essentially classified the games for us, and the greater amount of text content yields more accurate token classification after NLP processing
- **Introduce Steam's official tag system**: Tags are more granular than genres (e.g., "Roguelike", "Open World"), just need to add more categories to the classification labels

#### Optimization Direction B: Expand sample size

- Use the complete Steam Reviews dataset on Kaggle (millions of users)
- Incorporate SteamSpy estimated data

#### Optimization Direction C: Enrich feature dimensions

- Expand to 20-30 Steam tags
- Add user behavioral features: gaming frequency, average playtime, payment tendency
- Add temporal dimension: use LSTM/RNN to analyze changes in player genre preferences over time

### 8.2 Algorithm Level (partially guided by AI)

Current issue: The pretraining task is Masked Feature Reconstruction, which is effective but relatively simple. The downstream GAT + MLP achieves 93% on synthetic data, but real generalization ability is unknown.

#### Optimization Direction A: Stronger pretraining tasks

- **Contrastive Learning**: Randomly perturb the same user's features, teaching the GNN that "the user before and after perturbation is the same user"
- **GraphMAE**: Not only mask node features, but also randomly mask edges in the graph, reconstructing the deleted edges
- **Multi-task pretraining**: Simultaneously perform mask reconstruction + type classification + graph structure reconstruction, learning richer representations

#### Optimization Direction B: Stronger encoders

- **Graph Transformer**: Replace GAT, introduce global attention, not just neighbors (equivalent to introducing Transformer architecture)
- **Increase GNN layers**: Expand from current 2 layers to 3-4 layers, using continuous residual connections to increase depth and accuracy
- **Heterogeneous GNN**: Distinguish "user-user" edges from "user-game" edges, build heterogeneous graphs (learning how to construct heterogeneous graphs) [idea provided by AI, still learning]

#### Optimization Direction C: Better training strategies

- **Learning rate Warmup + Cosine Decay**: Currently only using Cosine Decay; adding Warmup makes training more stable in the early phase
- **Mixup data augmentation**: Linearly interpolate two users in feature space to generate new training samples
- **Label Smoothing**: Replace hard labels with soft labels to reduce overfitting
- **5-fold cross-validation**: Currently only a single train/val split, unable to assess stability. Can randomly dropout channels (drop 10% of channels per fold), run 5-fold experiments and take the average to improve generalization

### 8.3 Evaluation Level
- Introduce real survey data for validation, with the possibility of adding more questions (10 or more)
- A/B testing: GAT vs GCN vs GraphSAGE vs pure MLP
- Ablation studies: remove graph structure / remove attention, verify each module's contribution one by one
- Visualization analysis: use t-SNE to compare embedding distributions before and after training
