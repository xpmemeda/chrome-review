# 常见卡的规格对比

- CC: Compute capability
- FP32: FP32 TFLOPS(CUDA Core)
- FP16: FP16 TFLOPS(Tensor Core)
- INT8: INT8 TOPS(Tensor Core)
- CUDA: CUDA cores
- TENSOR: Tensor cores
- L1/Smem: KB/SM
- Gmem: GB
- Gmem BW:  GB/s
- TDP: Thermal Design Power (W)

Ref: https://en.wikipedia.org/wiki/List_of_Nvidia_graphics_processing_units

- 8.9: SM(128, 4, 128KB)

|GPU        |Arch       |Launch |Core       |CC     |F32    |F16    |I8     |SM     |CUDA   |TENSOR |L1/SMem|Gmem       |Gmem BW    |TDP    |
|-          |-          |-      |-          |-      |-      |-      |-      |-      |-      |-      |-      |-          |-          |-      |
|T4         |Turing     |18.09  |TU104      |7.5    |8.1    |65     |130    |40     |2560   |320    |64     |16         |300        |70     |
|A10        |Ampere     |21.04  |GA102      |8.6    |31.2   |125    |250    |72     |9216   |288    |64     |24 GDDR6   |600        |150    |
|A100       |Ampere     |20.05  |GA100      |8.0    |19.5   |312    |624    |108    |6912   |432    |164    |40/80 HBM2e|1935       |300    |
|L20        |Lovelace   |       |           |8.9    |59.8   |119.5  |239    |92     |11776  |368    |100    |48         |864        |?      |
|L40        |Lovelace   |22.10  |AD102      |8.9    |90.5   |181    |362    |142    |18176  |568    |128    |48 GDDR6   |864        |300    |
|RTX 4090   |Lovelace   |22.10  |AD102-300  |8.9    |83.8   |167.6  |1330   |128    |16384  |512    |128    |24 GDDR6X  |1008       |450    |

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
