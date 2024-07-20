# 常见卡的对比

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


## benchmark

script: cuda/kernels/benchmark/benchmark-torch.py
unit: ms, smaller is better.

- mm-1: torch.matmul, bmnk = (1, 8, 4096, 4096)
- mm-2: torch.matmul, bmnk = (8, 4096, 4096, 4096)
- mm-3: torch.matmul, bmnk = (256, 4096, 4096, 4096)
- bert: bert-base-uncased, bs = (8, 256)
- qwen2-7b: Qwen2-7B, bso = (8, 1024, 32)

|GPU            |mm-1   |mm-2   |mm-3   |bert   |qwen2-7b   |
|-              |-      |-      |-      |-      |-          |
|T4             |0.155  |40.87  |OOM    |6.80   |OOM        |
|A10            |0.081  |13.98  |456.81 |7.02   |3508.67    |
|A100 40G PCIe  |0.046  |4.14   |137.25 |7.64   |1470.28    |
|L20            |0.073  |9.51   |298.27 |4.01   |2232.77    |
|L40            |0.078  |6.97   |282.05 |3.80   |1980.56    |

## What's the CUDA primary context

The primary context is unique per device and shared with the CUDA runtime API.

Ref: https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__PRIMARY__CTX.html

## cudaInitDevice和cudaSetDevice的区别是什么？

两者都会初始化设备和Primary Context，但是前者不会把设备和Context绑定起来，后者会。

- cudaInitDevice: This function will initialize the CUDA Runtime structures and primary context on device when called, but the context will not be made current to device.

- cudaSetDevice: This function will immediately initialize the runtime state on the primary context, and the context will be current on device immediately.

## libcupti.so和libcufft.so分别是什么？

libcupti.so和libcufft.so是CUDA工具包中的两个不同库，它们分别用于不同的目的：

libcupti.so 是 CUDA Profiling Tools Interface (CUPTI) 库。CUPTI 提供了一组 API，用于收集和分析 CUDA 程序的性能数据。它主要用于性能分析和调试，帮助开发者优化 CUDA 应用程序。CUPTI 可以收集各种性能指标，如内核执行时间、内存传输时间、硬件计数器等。

libcufft.so 是 CUDA Fast Fourier Transform (cuFFT) 库。cuFFT 提供了一组 API，用于在 NVIDIA GPU 上执行快速傅里叶变换 (FFT)。FFT 是一种广泛应用于信号处理、图像处理、音频处理和科学计算的数学变换。

## libcublasLt.so是什么？

libcublasLt.soNVIDIA cuBLAS库的一部分，它是一个线性代数库。Lt代表"Low Level Tensor Operations"，这是一个更底层的接口，提供了更多的灵活性和更高的性能。cuBLAS Lt API允许用户更细粒度地控制矩阵运算，包括矩阵布局、计算类型、硬件资源的使用等。这使得用户可以更好地优化他们的应用程序，以获得最高的性能。
