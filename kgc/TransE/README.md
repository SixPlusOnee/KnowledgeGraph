# TransE

### Paper
[Translating Embeddings for Modeling Multi-relational Data](https://papers.nips.cc/paper/2013/file/1cecc7a77928ca8133fa24680a88d2f9-Paper.pdf)

### 核心思想
- 给定KG中已有的三元组(s, r, t)，在嵌入空间中，使得 s + r 与 t 的距离尽可能小
- 优化目标: argmin(distance(s + r - t) - distance(s' + r' - t'))，(s', r', t')为负采样三元组

### 实现
transE.py使用tensorflow 2.0(keras)实现了transE
main.py使用transE对FB15k-237数据集进行了实践

### Run
python3 main.py
