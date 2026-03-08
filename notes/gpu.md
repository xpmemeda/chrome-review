# 常见卡的对比

|Processor  |Release    |Foundry    |Size   |Arch       |CC     |GPCs   |TPCs   |SMs    |CUDAs  |CUDAs/SM   |TC Gen |TCs    |TCs/SM |L1(KB)/SM  |L2(MB) |
|-          |-          |-          |-      |-          |-      |-      |-      |-      |-      |-          |-      |-      |-      |-          |-      |
|GH100      |23/03      |TSMC       |5nm    |Hopper     |9.0    |8      |72     |144    |18432  |128        |4th    |576    |4      |256        |60     |
|AD102      |22/09      |TSMC       |5nm    |Lovelace   |8.9    |12     |72     |144    |18432  |128        |4th    |576    |4      |128        |96     |
|GA102      |20/09      |Samsung    |8nm    |Ampere     |8.6    |7      |42     |84     |10752  |128        |3rd    |336    |4      |128        |6      |
|GA100      |20/05      |TSMC       |7nm    |Ampere     |8.0    |8      |64     |128    |8192   |64         |3rd    |512    |4      |192        |80     |
|TU104      |18/08      |TSMC       |12nm   |Turing     |7.5    |6      |24     |48     |3072   |64         |2nd    |384    |8      |64         |4      |
|GV100      |17/06      |TSMC       |12nm   |Volta      |7.0    |6      |42     |84     |5376   |64         |1st    |672    |8      |128        |6      |

|GPU            |Processor  |SMs        |CUDAs  |TCs    |L1(KB)/SM  |L2(MB) |Gmem (GB)  |Gmem BW    |F32    |F16    |TDP    |
|-              |-          |-          |-      |-      |-          |-      |-          |-          |-      |-      |-      |
|H100 SXM5 96GB |GH100      |132/144    |16896  |528    |256        |60     |96 HBM3    |3.36 TB/s  |67     |989    |700    |
|H20            |GH100      | 78/144    |9984   |312    |256        |60     |96 HBM3    |4.0  TB/s  |44     |148   ?|400    |
|L40            |AD102      |142/144    |18176  |568    |128        |96     |48 GDDR6   |864  GB/s  |90.5   |181    |300    |
|L20            |AD102      | 92/144    |11776  |368    |128        |96     |48 GDDR6   |864  GB/s  |59.8   |119.5  |275    |
|RTX 4090       |AD102      |128/144    |16384  |512    |128        |72     |24 GDDR6X  |1008 GB/s  |83.8   |167.6  |450    |
|A10            |GA102      | 72/84     |9216   |288    |128        |6      |24 GDDR6   |600  GB/s  |62.5   |125    |150    |
|A100 SXM4 40GB |GA100      |108/128    |6912   |432    |192        |40     |40 HBM2e   |1.56 TB/s  |19.5   |312    |300    |
|T4             |TU104      | 40/48     |2560   |160    |64         |4      |16 GDDR6   |320  GB/s  |8.1    |65     |70     |
|V100 SXM2 32GB |GV100      | 80/84     |5120   |640    |128        |6      |32 HBM2    |900  GB/s  |15.7   |125    |300    |

- GH100: https://www.techpowerup.com/gpu-specs/nvidia-gh100.g1011
- AD102: https://www.techpowerup.com/gpu-specs/nvidia-ad102.g1005
- GA102: https://www.techpowerup.com/gpu-specs/nvidia-ga102.g930
- GA100: https://www.techpowerup.com/gpu-specs/nvidia-ga100.g931
- TU104: https://www.techpowerup.com/gpu-specs/nvidia-tu104.g854
- GV100: https://www.techpowerup.com/gpu-specs/nvidia-gv100.g809

## specifications

- CC: Compute capability
- FP32: FP32 TFLOPS(CUDA Core)
- FP16: FP16 TFLOPS(Tensor Core)
- INT8: INT8 TOPS(Tensor Core)
- CUDA: CUDA cores
- TC: Tensor cores
- L1/Smem: KB/SM
- Gmem: GB
- Gmem BW: GB/s
- TDP: Thermal Design Power (W)


|GPU            |Arch       |Launch |Core   |CC     |F32    |TF32   |F16    |I8     |SM     |CUDA   |TC     |L1 (K) |L2 (M) |Gmem (GB)  |Gmem BW (GB/s) |TDP    |
|-              |-          |-      |-      |-      |-      |-      |-      |-      |-      |-      |-      |-      |-      |-          |-              |-      |
|V100 XSM2      |Volta      |17/06  |GV100  |7.0    |15.7   |       |125    |       |80     |5120   |640    |128    |6      |32/16 HBM2 |900            |300    |
|T4             |Turing     |18/09  |TU104  |7.5    |8.1    |       |65     |130    |40     |2560   |160    |64     |4      |16 GDDR6   |320            |70     |
|A10            |Ampere     |21/04  |GA102  |8.6    |31.2   |62.5   |125    |250    |72     |9216   |288    |128    |6      |24 GDDR6   |600            |150    |
|A100 40G PCIe  |Ampere     |20/06  |GA100  |8.0    |19.5   |156    |312    |624    |108    |6912   |432    |192    |40     |40 HBM2e   |1560           |300    |
|A100 80G PCIe  |Ampere     |21/06  |GA100  |8.0    |19.5   |156    |312    |624    |108    |6912   |432    |192    |80     |80 HBM2e   |1935           |300    |
|L20            |Lovelace   |23/11  |AD102  |8.9    |59.8   |       |119.5  |239    |92     |11776  |368    |128    |96     |48 GDDR6   |864            |275    |
|L40            |Lovelace   |22/10  |AD102  |8.9    |90.5   |90.5   |181    |362    |142    |18176  |568    |128    |96     |48 GDDR6   |864            |300    |
|RTX 4090       |Lovelace   |22/10  |AD102  |8.9    |83.8   |       |167.6  |330    |128    |16384  |512    |128    |72     |24 GDDR6X  |1008           |450    |
|H100 PCIe      |Hopper     |23/03  |GH100  |9.0    |60     |-      |835    |-      |114    |14592  |456    |256    |50     |94 HBM2e   |3900           |400    |
|H100 SXM       |Hopper     |23/03  |GH100  |9.0    |67     |-      |989    |-      |114    |14592  |456    |256    |50     |80 HBM2e   |3350           |700    |
|H20            |Hopper     |23/12  |       |9.0    |44     |74     |148    |296    |78     |       |       |228    |60     |96 HBM3    |4096           |400    |

Refs:

- https://en.wikipedia.org/wiki/List_of_Nvidia_graphics_processing_units
- https://www.techpowerup.com/


# HGEMM 理论性能分析

HGEMM 是计算瓶颈还是访存瓶颈，理论输入最大吞吐是多少？

## H20

硬件规格

- 算力：148 TFLOPS（FP16）
- 带宽：4 TB/s
- Operational Intensity (OI)：148TFLOPS / 4TB/s = 37 FLOP/B

### 8x4096x4096

OI ~= O(M) = 8

- 计算：2(fma) * M * K * N = 2 * 8 * 4096 * 4096 = 256 MFLOP
- 访存：2(fp16,2bytes) * (M * K + K * N + 2 * M * N) ~= 32 MB
- Operational Intensity (OI)：256MFLOP / 32MB ~= 8 FLOP/B

任务 OI 显著小于硬件 OI，因此是**访存瓶颈**。

理论输入最大吞吐：

只看访存部分的最大吞吐，总的访存大小为 32MB，耗时 32MB / 4TB/s。
输入吞吐则是 2(fp16,2bytes) * M * K / 耗时 = 2 * 8 * 4096 * 4TB/s / 32MB = 8 GB/s

Pytorch 实测吞吐 3.41GB/s，达成率 3.41 / 8 = 42.6%。

### 4096x4096x4096

OI ~= O(K/4) = 1024

- 计算：2(fma) * M * K * N = 2 * 4096 * 4096 * 4096 = 128 GFLOP
- 访存：2(fp16,2bytes) * (M * K + K * N + 2 * M * N) = 128 MB
- Operational Intensity (OI)：128GFLOP / 128MB = 1000 FLOP/B

任务 OI 显著高于硬件 OI，因此是**计算瓶颈**。

理论输入最大吞吐：

只看计算部分的最大吞吐，总的计算量是 128GFLOP，耗时 128GFLOP / 148TFLOPS。
输入吞吐则是 2(fp16,2bytes) * M * K / 耗时 = 2 * 4096 * 4096 * 148TFLOPS / 128GFLOP = 37 GB/s

Pytorch 实测吞吐 32.20GB/s，达成率 32.20 / 37 = 87.0%。

## A100-40GB-PCIe

硬件规格

- 算力：312 TFLOPS (FP16)
- 带宽：1.56 TB/s

算力是 H20 的 2.11 倍，带宽是 H20 的 0.39 倍。

### 8x4096x4096

OI ~= O(M) = 8

访存瓶颈，理论输入最大吞吐应是 H20 的 0.39 倍，即 8 * 0.39 = 3.12 GB/s，
Pytorch 实测 1.32 GB/s，达成率 42.3%，和 H20 的达成率差不多。

### 4096x4096x4096

OI ~= O(K/4) = 1024

计算瓶颈，理论输入最大吞吐应是 H20 的 2.11 倍，即 37 * 2.11 = 78.97 GB/s，
Pytorch 实测 51.52 GB/s，达成率 65.2%，比 H20 低。

# 不同卡 HGEMM 实测数据

## 脚本

```python
import triton
import torch


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["M"],
        x_vals=[8, 128, 4096, 16384],
        line_arg="NK",
        line_vals=[1024, 2048, 4096, 7168],
        line_names=["1024", "2048", "4096", "7168"],
        styles=[("blue", "-"), ("green", "-"), ("red", "-"), ("yellow", "-")],
        ylabel="GB/s",
        plot_name="GEMM",
        args={},
    )
)
def benchmark(M, NK, device=torch.device("cuda:0")):
    x = torch.randn(M, NK, device=device, dtype=torch.float16)
    y = torch.randn(NK, NK, device=device, dtype=torch.float16)

    ms = triton.testing.do_bench(lambda: torch.matmul(x, y))

    gbps = lambda ms: x.nbytes * 1e-9 / (ms * 1e-3)

    return gbps(ms)


if __name__ == "__main__":
    benchmark.run(print_data=True, save_path=".")
```

## Operational Intensity (OI)

|GPU    |FLOPS      |BW     |OI (FLOP/B)    |
|-      |-          |-      |-              |
|T4     |65T        |320G   |203            |
|A10    |125T       |600G   |208            |
|A100   |312T       |1560G  |200            |
|L20    |119.5T     |864G   |138            |
|L40    |181T       |864G   |215            |
|H20    |148        |4096G  |37             |

- fma   2
- half  2
- OI = 2 * M * N * K / (2 * M * K + 2 * K * N + 2 * M * N * 2)
  - M << K == N: OI ~= O(M)
  - M == K == N: OI ~= O(K / 4)
  - M >> K == N: OI ~= O(K / 3)

## M=8; OI ~= O(M) = 8

|GPU    |8x1024x1024    |8x2048x2048    |8x4096x4096    |8x7168x7168    |
|-      |-              |-              |-              |-              |
|T4     |0.82 GB/s      |0.53 GB/s      |0.39 GB/s      |0.27 GB/s      |
|A10    |1.13 GB/s      |1.21 GB/s      |0.81 GB/s      |0.51 GB/s      |
|A100   |0.96 GB/s      |1.31 GB/s      |1.32 GB/s      |1.06 GB/s      |
|L20    |1.00 GB/s      |0.83 GB/s      |0.92 GB/s      |0.54 GB/s      |
|L40    |1.00 GB/s      |0.96 GB/s      |0.83 GB/s      |0.55 GB/s      |
|H20    |1.93 GB/s      |3.07 GB/s      |3.41 GB/s      |2.45 GB/s      |

## M=128; OI ~= O(M) = 128

|GPU    |128x1024x1024  |128x2048x2048  |128x4096x4096  |128x7168x7168  |
|-      |-              |-              |-              |-              |
|T4     |10.62 GB/s     |9.35  GB/s     |6.90  GB/s     |3.98  GB/s     |
|A10    |19.33 GB/s     |17.92 GB/s     |12.90 GB/s     |8.08  GB/s     |
|A100   |15.21 GB/s     |21.83 GB/s     |19.06 GB/s     |14.64 GB/s     |
|L20    |11.27 GB/s     |22.05 GB/s     |12.83 GB/s     |8.03  GB/s     |
|L40    |12.46 GB/s     |19.63 GB/s     |14.03 GB/s     |8.04  GB/s     |
|H20    |28.21 GB/s     |31.75 GB/s     |22.01 GB/s     |17.31 GB/s     |

## M=4096; OI ~= O(K / 3) = 341 ~ 2390

|GPU    |4096x1024x1024 |4096x2048x2048 |4096x4096x4096 |4096x7168x7168 |
|-      |-              |-              |-              |-              |
|T4     |38.11  GB/s    |17.30 GB/s     |6.28  GB/s     |2.52  GB/s     |
|A10    |76.18  GB/s    |37.83 GB/s     |15.99 GB/s     |9.45 GB/s      |
|A100   |118.48 GB/s    |94.35 GB/s     |51.52 GB/s     |31.80 GB/s     |
|L20    |87.72  GB/s    |50.78 GB/s     |27.88 GB/s     |16.15 GB/s     |
|L40    |108.71 GB/s    |68.58 GB/s     |31.99 GB/s     |31.05 GB/s     |
|H20    |106.59 GB/s    |62.57 GB/s     |32.20 GB/s     |18.91 GB/s     |

## M=16384; OI ~= O(K / 4) = 256 ~ 1792

|GPU    |16384x1024x1024|16384x2048x2048|16384x4096x4096|16384x7168x7168|
|-      |-              |-              |-              |-              |
|T4     |26.79  GB/s    |13.27  GB/s    |5.67  GB/s     |2.71  GB/s     |
|A10    |67.04 GB/s     |27.80  GB/s    |18.56  GB/s    |10.82 GB/s     |
|A100   |177.56 GB/s    |98.39  GB/s    |54.01  GB/s    |31.15 GB/s     |
|L20    |100.34 GB/s    |54.93  GB/s    |28.00  GB/s    |16.02 GB/s     |
|L40    |117.98 GB/s    |53.38  GB/s    |25.93  GB/s    |14.07 GB/s     |
|H20    |128.69 GB/s    |66.49 GB/s     |33.64  GB/s    |19.71 GB/s     |

## Models

Latency: ms, smaller is better.

- bert: bert-base-uncased, bs = (8, 256)
- qwen2-7b: Qwen2-7B, bso = (8, 1024, 32)

|GPU            |bert   |qwen2-7b   |
|-              |-      |-          |
|T4             |6.80   |OOM        |
|A10            |7.02   |3508.67    |
|A100 40G PCIe  |7.64   |1470.28    |
|L20            |4.01   |2232.77    |
|L40            |3.80   |1980.56    |

## nvidia gpu参数里面的Shading Units和CUDA cores有什么区别和联系？

在 NVIDIA GPU 的参数中，"Shading Units" 和 "CUDA Cores" 是两个常见的术语，它们有一定的联系，但也有区别。以下是对这两个术语的详细解释：

CUDA Cores
CUDA Cores 是 NVIDIA GPU 中的基本计算单元。每个 CUDA Core 可以执行简单的算术运算，如加法、乘法等。CUDA Cores 是 NVIDIA 的并行计算架构 CUDA（Compute Unified Device Architecture）的核心组成部分，专门用于处理并行计算任务。CUDA Cores 的数量通常是衡量 GPU 计算能力的一个重要指标。

Shading Units
Shading Units 通常指的是 GPU 中用于执行着色（shading）任务的计算单元。在现代 GPU 架构中，着色任务包括顶点着色、像素着色、几何着色和计算着色等。Shading Units 负责处理这些任务，以生成最终的图像。

联系
在 NVIDIA 的 GPU 架构中，Shading Units 和 CUDA Cores 实际上是同一个东西。也就是说，CUDA Cores 就是用于执行着色任务的计算单元。因此，在 NVIDIA 的 GPU 参数中，Shading Units 和 CUDA Cores 通常是同义词，指的是同一组计算单元。

区别
虽然在 NVIDIA 的 GPU 中，Shading Units 和 CUDA Cores 是同一个东西，但在其他 GPU 架构（如 AMD 的 GPU）中，Shading Units 可能有不同的定义和实现方式。因此，Shading Units 这个术语在不同的 GPU 架构中可能会有不同的含义。

总结
在 NVIDIA 的 GPU 参数中，Shading Units 和 CUDA Cores 是同一个东西，指的是 GPU 中的基本计算单元。这些计算单元既可以用于并行计算任务，也可以用于执行着色任务。虽然这两个术语在 NVIDIA 的 GPU 中是同义词，但在其他 GPU 架构中，Shading Units 可能有不同的定义和实现方式。

## 为什么H100和H800用的是同一款芯片，也即GH100，为什么会有性能差距？

NVIDIA H100 和 H800 都基于相同的 GH100 芯片，但它们之间可能存在性能差距，原因主要包括以下几个方面：

1. 芯片配置和启用的单元数量
虽然 H100 和 H800 都使用 GH100 芯片，但它们可能启用了不同数量的计算单元（如 CUDA 核心、Tensor 核心等）。在制造过程中，芯片可能会有一些缺陷，NVIDIA 可能会禁用部分有缺陷的单元，以提高芯片的良品率。这种做法可以产生不同配置的产品，即使它们使用相同的基础芯片。

2. 时钟频率
H100 和 H800 可能运行在不同的时钟频率下。较高的时钟频率通常意味着更高的性能，但也会增加功耗和发热量。NVIDIA 可能会根据产品定位和目标市场，调整不同型号的时钟频率，以平衡性能和功耗。

3. 显存配置
显存的类型、容量和带宽也会影响 GPU 的整体性能。H100 和 H800 可能配备了不同类型或数量的显存，从而导致性能差异。例如，H100 可能配备了更高带宽的显存，而 H800 可能使用了较低带宽的显存。

4. 功耗和散热设计
不同型号的 GPU 可能有不同的功耗限制和散热设计。较高的功耗限制和更高效的散热设计可以允许 GPU 在更高的性能水平下运行。H100 可能设计用于更高的功耗和更强的散热系统，从而提供更高的性能。

5. 市场定位和优化
NVIDIA 可能会根据不同的市场需求，对 H100 和 H800 进行不同的优化。例如，H100 可能针对高性能计算和数据中心应用进行了优化，而 H800 可能针对其他特定应用场景进行了优化。这些优化可能包括驱动程序、固件和软件层面的调整，从而影响实际性能。

6. 产品分级和市场策略
NVIDIA 可能会根据市场策略，对同一款芯片进行不同的分级和定价。通过这种方式，NVIDIA 可以覆盖更广泛的市场需求，并提供不同价位和性能水平的产品。

总结
尽管 H100 和 H800 都基于相同的 GH100 芯片，但它们之间的性能差距可能源于芯片配置、时钟频率、显存配置、功耗和散热设计、市场定位和优化等多个因素。通过这些调整，NVIDIA 可以提供不同性能和价格的产品，以满足不同的市场需求。