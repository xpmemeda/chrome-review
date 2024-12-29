# 不同长度context下，矩阵乘和Attention的耗时占比

## Qwen2-7B + A10

- hidden_size: 3584
- intermediate_size: 18944
- vocab_size: 152064
- NQ/NK: 28/4

- P: profill
- D: decode
- PM: MM in Profill
- PA: Attn in Profill
- DM: MM in Decode
- DA: Attn in Decode

- bsz = 4

|       |P wnr  |D wnr  |PM     |PA (vllm)  |DM     |DA (vllm)  |DA (wnr)       |
|-      |-      |-      |-      |-          |-      |-          |-              |
|256    |199.51 |31.89  |       |0.85%      |       |1.91%      |0.62 (1.94%)   |
|1024   |805.99 |32.23  |       |1.92%      |       |3.54%      |1.23 (3.82%)   |
|2048   |1651.07|32.93  |       |3.64%      |       |5.37%      |1.88 (5.71%)   |
|4096   |                       |6.68%      |       |7.42%      |OOM            |


## Baichuan-7B + A10

- hidden_size: 4096
- intermediate_size: 11008
- vocab_size: 64000
- NQ/NK: 32/32

- bsz = 4

|       |P wnr  |D wnr  |PM     |PA (vllm)  |DM     |DA (vllm)  |DA (wnr)       |
|-      |-      |-      |-      |-          |-      |-          |-              |
|256    |192.74 |30.85  |       |           |       |           |1.34 (4.34%)   |
|1024   |793.52 |34.12  |       |           |       |           |4.48 (13.13%)  |
|2048   |825.59 |38.38  |       |4.59%      |       |22.60%     |8.83 (23.01%)  |
|4096   |                                           |           |OOM            |
