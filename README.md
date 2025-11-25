# HMGCL
This is code for our WWWJ paper《HMGCL: Heterogeneous Multigraph Contrastive Learning for LBSN Friend Recommendation》


Before to run *HMGCL*, it is necessary to install the following packages:
<br/>
``pip install dgl``
<br/>
``pip install torch``
<br/>
``pip install scikit-learn``

## Requirements

- numpy ==1.13.1
- torch ==1.7.1
- scikit-learn==1.0.2
- dgl ==0.7.2

【September 11, 2024】The modified code now supports the latest versions of torch and DGL.

### Data Set


You can download whole [raw Foursquare Dataset](https://sites.google.com/site/yangdingqi/home/foursquare-dataset) here.

Our data can be found at [here](https://drive.google.com/file/d/1i6W2oz0PEidhG2md6pn1u-HRT3wmWyh7/view?usp=sharing).

【March 3, 2025】 Please refer to the code implementation in [H3GNN](https://github.com/liyongkang123/H3GNN) to learn more about processing heterogeneous multigraph data. 
We provide an updated version of the code that can complete the data preprocessing for a single city within 10 minutes.

### Basic Usage
 
- --run  main.py to train the HMGCL. and it probably need at least 11G GPU memory 
- --run  test.py to estimate the performance of HMGCL based on the user representations that we learned during our experiments. You can also use this code to individually test the effects of your own learned representation.

### Miscellaneous

*Note:* This is only a reference implementation of *HMGCL*. Our code implementation is partially based on the DGL library, for which we are grateful.

# Citation
If you find this work helpful, please consider citing our paper:
```bibtex
@article{li2023hmgcl,
  title={HMGCL: Heterogeneous multigraph contrastive learning for LBSN friend recommendation},
  author={Li, Yongkang and Fan, Zipei and Yin, Du and Jiang, Renhe and Deng, Jinliang and Song, Xuan},
  journal={World Wide Web},
  volume={26},
  number={4},
  pages={1625--1648},
  year={2023},
  publisher={Springer},
  url = {https://doi.org/10.1007/s11280-022-01092-5},
  doi = {10.1007/s11280-022-01092-5},
}
```
## 模型改进说明

### 2024年11月 - Transformer模块融合改进

在Transformer编码层的基础上，添加了以下两个关键改进：

#### 1. 位置编码（Positional Encoding）
- **目的**：帮助模型捕捉节点的顺序信息
- **实现**：使用标准三角函数位置编码
  - sin编码：位置偶数维度
  - cos编码：位置奇数维度
- **频率公式**：`PE(pos, 2i) = sin(pos / 10000^(2i/d_model))`
- **应用时机**：在输入Transformer之前添加位置编码

#### 2. 残差连接与融合层（Residual Connection + Fusion Layer）
- **目的**：有效融合GNN原始输出与Transformer的输出，防止信息损失
- **实现细节**：
  - 保存Conv2后的原始GNN输出
  - 使用可学习的权重参数（gnn_weight和transformer_weight）控制融合比例
  - 公式：`fused = α * gnn_output + β * transformer_output + gnn_output`
  - 使用LayerNorm稳定训练过程
  - 初始权重均设为0.5，训练过程中自动调整

#### 3. 代码结构
```python
# 新增类
- PositionalEncoding: 位置编码层
- TransformerFusionLayer: 融合层（残差+可学习权重）

# HMGNN类修改
- __init__: 添加位置编码和融合层初始化
- forward: 添加位置编码应用和残差融合过程
```

### 实验建议
1. 监控训练过程中fusion_layers中的权重参数变化
2. 比较有/无位置编码的性能差异
3. 调整融合层的融合策略（可以改为拼接、门控单元等）