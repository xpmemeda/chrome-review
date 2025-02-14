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


# DeepSeek

- select_experts

DeepSeek有n_routed_experts=256个experts，n_group=8个一组，选择函数根据门控组件的输出 router_logits (num_tokens, num_experts)
（1）根据专家组中的最大值，先topk出概率最高的那topk_group=4个组，没有被选中的组中的专家，概率再高也要被置零。
（2）从选中的4个专家组中，选择概率最大的num_experts_per_tok=8个专家。
返回
（1）topk_weights: 选中的专家的概率(num_tokens, num_experts_per_tok)
（2）topk_ids: 选中的专家的索引(ditto)

- run_moe_ep_preproess

计算一些辅助信息，根据topk_ids(num_tokens, num_experts_per_tok)
（1）调用sort，把topk_ids压扁排序，得到reorder_topk_ids
（2）计算src2dst，作用是通过topk_ids获取reorder_topk_ids，而不需要sort。（估计后面会通过这个方式重新布局hidden_states，然后专家通过seg_indptr来取值）
（3）seg_indptr(num_experts + 1)，每个专家要取的输入切片是seg_indptr[export_id, export_id + 1]

- pre_reorder_triton_kernel

（猜测）把hidden_states重排成 reorder_topk_ids 这个形状 (num_tokens, num_experts_per_tok, hidden_size)，输出 gateup_input.

- grouped_gemm_runner

实际上所有的卡都会拥有完整的hidden_state输入，也都会调用前面的 run_moe_ep_preproess 和 pre_reorder_triton_kernel ，生成所有experts的输入。只不过在 grouped_gemm_runner kernel内部，会通过自身rank来决定算哪些。

这个kernel的结果是 gateup_output （num_tokens * num_experts_per_tok, w13_size）
再 down_output (num_tokens * num_experts_per_tok, w2_size)

- post_reorder_triton_kernel

每个experts都会算出一份 output （num_tokens, hidden_size）,这个函数把每个卡的experts的输出可以直接加权相加，然后输出。

后面不同卡的experts会通过一次 all_reduce 合并结果。
