# Memory pool capacity

|Model                              |Arch   |kv |Layers |Heads  |Features   |Bytes  |kB/token   |tokens/GB  |
|-                                  |-      |-  |-      |-      |-          |-      |-          |-          |
|Qwen2.5-7B                         |MHA    |2  |28     |4      |128        |2      |56         |18724      |
|Qwen3-0.6B                         |MHA    |2  |28     |8      |128        |2      |112        |9362       |
|Qwen3-1.7B                         |MHA    |2  |28     |8      |128        |2      |112        |9362       |
|QwQ-32B                            |MHA    |2  |64     |8      |128        |2      |256        |4096       |
|Qwen3-Coder-480B-A35B-Instruct-FP8 |MHA    |2  |62     |8      |64         |1      |62         |16912      |
|DeepSeek-R1                        |MLA    |1  |61     |1      |512        |2      |61         |17189      |
|GLM-5-FP8                          |MLA    |1  |78     |1      |576        |2      |87.75      |11949.58   |
