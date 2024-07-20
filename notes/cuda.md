# 常见卡的规格对比

|Entry                          |T4     |A10(PCIe)  |A100(PCIe) |L40        |RTX 4090   |
|-                              |-      |-          |-          |-          |-          |
|Arch                           |Turing |Ampere     |Ampere     |Lovelace   |Lovelace   |
|Compute capability             |7.5    |8.6        |8.0        |8.9        |8.9        |
|FP32 TFLOPS(CUDA Core)         |8.1    |31.2       |19.5       |90.5       |83.8       |
|FP16 TFLOPS(Tensor Core)       |65     |125        |312        |181        |167.6      |
|INT8 TOPS(Tensor Core)         |130    |250        |624        |362        |1330       |
|SM                             |40     |72         |108        |142        |128        |
|CUDA Cores                     |2560   |9216       |6912       |18176      |16384      |
|Tensor Cores                   |320    |288        |432        |568        |512        |
|L1/shared memory(KB/SM)        |64     |64         |164        |128        |128        |
|GPU memory(GiB)                |16     |24         |40/80      |48         |24         |
|GPU memory bandwidth(GiB/s)    |300    |600        |1935       |864        |1008       |
|TDP(Thermal Design Power)(W)   |70     |150        |300        |300        |450        |

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
