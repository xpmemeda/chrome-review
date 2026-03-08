# 并发定义

## Data Parallelism（DP，数据并行）

将输入数据（batch / tokens）切分到多个副本上，每个副本拥有完整模型参数，独立前向/反向计算，通过参数梯度或权重更新进行同步。

## Tensor Parallelism（TP，张量并行）

将单个算子的参数张量按维度切分到多个设备上，同一层的计算由多设备协同完成，并在算子边界通过集合通信（如 all-reduce / all-gather）合并结果。

## Expert Parallelism（EP，专家并行）

在 MoE 模型中，将不同的 expert 分布到不同设备，token 根据路由结果被分发到对应 expert 计算，并通过 token 级通信完成结果回收。

## Pipeline Parallelism（PP，流水线并行）

将模型的层或阶段按顺序切分到多个设备上，输入数据按微批次在各阶段之间流水传递，实现模型级并行。

# 通信开销

|OP     |Shape                  |~Numel |Numel                  |
|-      |-                      |       |                       |
|AG     |[m / p, n] -> [m, n]   |mn     |(p - 1) / p * mn       |
|AR     |[m, n]     -> [m, n]   |2mn    |2 * (p - 1) / p * mn   |
|A2A    |[m, n]     -> [m, n]   |mn     |-                      |


# MoE: TP vs EP

## Feed-Forward Network

$$
\mathrm{FFN}(x) = \sigma(x W_1) W_2
$$

## SwiGLU

$$
\mathrm{SwiGLU}(x)
=
\left( \mathrm{SiLU}(x W_1) \odot (x W_3) \right) W_2
$$

---

## 假设场景一

- 所有 GPU 均拥有完整的输入激活 [m, n]。
- top-k 路由，token 分布假设均匀。

|Step                   |TP = 16                                                            |EP = 16                                                        |
|-                      |-                                                                  |-                                                              |
|Input                  |[m, n]                                                             |[m, n]                                                         |
|Top-k Expansion        |[m, n] -> [m * topk, n]                                            |[m, n]                                 -> [m * topk, n]        |
|EP Dispatch (No-Comm)  |-                                                                  |[m * topk, n]                          -> [m * topk / 16, n]   |
|SwiGLU (FFN)           |[m * topk, n] x [n, 4n / 16] x [4n / 16, n]    -> [m * topk, n]    |[m * topk / 16, n] x [n, 4n] x [4n, n] -> [m * topk / 16, n]   |
|TP Top-k Aggregation   |[m * topk, n] -> [m, n]                                            |-                                                              |
|TP AR                  |[m, n] -> [m, n]                                                   |-                                                              |
|EP Combine (AG)        |                                                                   |[m * topk / 16, n]                     -> [m * topk , n]       |
|EP Top-k Aggregation   |-                                                                  |[m * topk , n]                         -> [m , n]              |
|Sum-Communication      |2mn                                                                |topk * mn                                                      |

在上述设定下：

**只要 topk > 2，TP 的通信量更小**。

## 假设场景二

- 每块 GPU 只拥有部分输入激活 [m / 16, n]，比如同时开启了 DP16。
- top-k 路由，token 分布假设均匀。

|Step                   |TP = 16                                                            |EP = 16                                                        |
|-                      |-                                                                  |-                                                              |
|Input                  |[m / 16, n]                                                        |[m / 16, n]                                                    |
|TP AG                  |[m / 16, n]                                    -> [m, n]           |-                                                              |
|Top-k Expansion        |[m, n]                                         -> [m * topk, n]    |[m / 16, n]                            -> [m / 16 * topk, n]   |
|EP Dispatch (A2A)      |-                                                                  |[m / 16 * topk, n]                     -> [m / 16 * topk, n]   |
|SwiGLU (FFN)           |[m * topk, n] x [n, 4n / 16] x [4n / 16, n]    -> [m * topk, n]    |[m / 16 * topk, n] x [n, 4n] x [4n, n] -> [m / 16 * topk, n]   |
|TP Top-k Aggregation   |[m * topk, n]                                  -> [m, n]           |-                                                              |
|TP AR                  |[m, n]                                         -> [m, n]           |-                                                              |
|EP Combine (A2A)       |-                                                                  |[m / 16 * topk, n] -> [m / 16 * topk, n]                       |
|EP Top-k Aggregation   |-                                                                  |[m / 16 * topk, n] -> [m / 16, n]                              |
|Sum-Communication      |3mn                                                                |2 / p(16) * topk * mn                                          |

在上述设定下：

**只要并发数大于 topk，EP 的通信量更小。**
