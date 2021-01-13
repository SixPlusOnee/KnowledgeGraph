# Knowledge Graph Completion

### 目前主要的方法都是先构建实体和关系的embedding，再用embedding进行预测

### Definitions
- h: head, 头实体
- r: relation, 关系
- t: tail, 尾实体
- (h, r, t): triple, 三元组
- ?: 预测目标

### 知识图谱补全，主要有以下3个任务:
- 给定头实体、关系类型，预测尾实体 => (h, r, ?)
- 给定关系类型、尾实体，预测头实习 => (?, r, t)
- 给定头尾实体，预测关系类型 => (h, ?, t)

### Benchmark
- FB15k-237

