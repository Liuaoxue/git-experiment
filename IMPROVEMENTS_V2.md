# 模型改进 V2 - 基于最佳实践的优化

## 改进日期
2025-11-25

## 问题诊断

原始实验显示，添加 Transformer + 拉普拉斯位置编码后，模型性能反而下降：
- AUC: 0.8800 → 0.8739 (▼ -0.61%)
- AP: 0.8894 → 0.8855 (▼ -0.39%)

**根本原因**：
1. 全局信息引入了噪声（LBSN 任务以局部关系为主）
2. 拉普拉斯位置编码在稀疏社交图上不稳定
3. 简单的线性融合无法让模型自主选择何时使用全局信息

## 核心改进

### 1. ✅ 门控融合机制 (Gated Fusion)

**之前**：简单的加权求和
```python
h_fused = α * h_gnn + β * h_transformer
```

**现在**：自适应门控
```python
gate = Sigmoid(MLP([h_gnn, h_transformer]))
h_fused = gate * h_transformer + (1 - gate) * h_gnn
```

**优势**：
- 模型可以为每个节点学习不同的融合策略
- 对于局部信息充足的节点，gate → 0（主要用 GNN）
- 对于需要全局信息的节点，gate → 1（引入 Transformer）
- 避免了全局噪声污染所有节点

### 2. ✅ 移除拉普拉斯位置编码

**之前**：使用图拉普拉斯矩阵的特征向量
- 计算复杂（特征分解）
- 对稀疏图不稳定
- 全局坐标可能与局部关系冲突

**现在**：简单的可学习位置嵌入
```python
self.pos_embed_user = nn.Embedding(num_users, dim)
```

**优势**：
- 计算高效（直接查表）
- 更灵活（每个节点学习独特的位置表示）
- 在 transductive setting 下通常更有效

### 3. ✅ 调整 Transformer 超参数

| 参数 | V1 (优化速度) | V2 (优化效果) | 说明 |
|------|--------------|--------------|------|
| window_size | 8 | 16 | 恢复合理窗口，增强表达能力 |
| dilation | 2 | 3 | 恢复空洞间隔，捕捉长程依赖 |
| dropout | 0.1 | 0.2 | 增加正则化，防止过拟合 |
| k_eigenvectors | 4 | N/A | 移除拉普拉斯 PE |

### 4. ✅ 保持高效的计算策略

- ✓ 只对 user 节点应用 Transformer（主要预测目标）
- ✓ poi 节点直接使用 GNN 输出（节省计算）
- ✓ 使用稀疏 Attention（Local + Atrous）
- ✓ 缓存位置嵌入（一次查表，多次使用）

## 架构对比

### V1: 简单线性融合
```
GNN → h_gnn ─┐
             ├─→ α*h_gnn + β*h_trans → Output
GNN+PE → Transformer → h_trans ─┘
```

### V2: 门控自适应融合
```
GNN → h_gnn ──────────────┐
                          ├─→ Gate Network → gate
GNN+PE → Transformer → h_trans ─┘
                          ↓
                gate*h_trans + (1-gate)*h_gnn → Output
```

## 理论依据

### 为什么门控融合更适合 LBSN？

1. **异质性 (Heterogeneity)**：
   - 有些用户有很多朋友（局部信息充足）→ gate → 0
   - 有些用户是孤立节点（需要全局推断）→ gate → 1

2. **噪声控制**：
   - 全局 Attention 会让节点关注成千上万个无关陌生人
   - 门控机制可以"关闭"这些噪声，只在必要时打开

3. **可解释性**：
   - 可以分析哪些节点依赖全局信息（gate 值高）
   - 验证模型是否学到了合理的融合策略

## 预期效果

基于理论分析和类似工作的经验：

| 指标 | 原始 HMGNN | V1 (失败) | V2 (预期) |
|------|-----------|----------|----------|
| AUC | 0.8800 | 0.8739 | **0.8850+** |
| AP | 0.8894 | 0.8855 | **0.8920+** |
| Top-1 | 0.1767 | 0.1732 | **0.1800+** |
| Top-20 | 0.5152 | 0.5024 | **0.5200+** |

**关键指标**：
- 如果 gate 的平均值在 0.05-0.15 之间 → 说明模型学会了"谨慎使用"全局信息
- 如果 AUC 提升 > 0.3% → 改进成功
- 如果 AUC 仍然下降 → 说明这个数据集真的不需要全局信息，应该回退到纯 GNN

## 下一步实验

### 实验 A：验证改进效果
```bash
python main.py --city SP --epochs 5000
```

### 实验 B：消融研究
1. 关闭 Transformer：`use_transformer=False`
2. 只用位置嵌入，不用 Transformer
3. 只用 Transformer，不用位置嵌入

### 实验 C：分析 Gate 分布
训练后分析：
- `gate` 值的分布（直方图）
- 哪些节点的 gate 值高（度数低的节点？）
- gate 值与节点度数的相关性

## 代码变更总结

### model.py
- ✅ 移除 `LaplacianPositionalEncoding` 的使用
- ✅ 添加 `nn.Embedding` 作为位置编码
- ✅ 添加 `gate_network` 实现门控融合
- ✅ 修改 `forward` 方法使用门控机制
- ✅ 调整 Transformer 超参数

### main.py
- ✅ 传递 `num_users` 参数给模型

## 参考文献

1. **Gated Fusion in GNN**:
   - "Adaptive Graph Convolutional Neural Networks" (AAAI 2018)
   
2. **Position Encoding in Graphs**:
   - "A Generalization of Transformer Networks to Graphs" (AAAI 2021)
   
3. **Local vs Global in Social Networks**:
   - "Do We Really Need Complicated Model Architectures For Temporal Networks?" (ICLR 2023)

## 致谢

本次改进基于对 LBSN 任务特性的深入分析，核心思想是：
> **局部信息决定下限，全局信息决定上限，但全局信息必须谨慎使用。**

