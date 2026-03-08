# 性能调优的思路

## 每个 SM 上可同时调度的 Warp 数量是否足够？Warp 数量不够会导致 Lantency 不能被掩盖，吞吐不能达到理论值。

Warp 的数量受到这三个方面限制：

1. 寄存器：一个 SM 最多使用 65535 个寄存器。
2. 共享内存限制：一个 SM 的共享内存大小是固定的，比如 H20 大小为 256 KB。
3. CTA 大小限制：H20 一个 SM 最多可以有 2048 个线程。

根据上面的三点都可以通过 ncu 打印出来，然后计算每个 SM 可以容纳几个 CTAs（Block）（向下取整），换算得到每个 SM 可以同时调度多少 Warps。

比如说一个 kernel 在的理论 Warps 数量是 12，H20 的规格是 64。那么这里 Occupancy.Theoretical_Occupancy 就是 12 / 64 = 18.75%。

另外，如果 Grids < SMs，Occupancy.Achieved_Occupancy = Occupancy.Theoretical_Occupancy * Grids/SMs。也就是说会更小。

那么多少 Warps/SM 算多呢？此处给出 GPT5 的回答：

- ALU / FMA-heavy   :  8 - 16
- L2 hit            : 16 - 32
- HBM Miss          : 32 - 64

NOTE:

- 同一个 CTA 内的 Warp 不可以被调度到不同的 SM 执行。
- 一个 CTA 不能被拆成多份 Launch，一个 SM 必须一次性调度一个完整的 CTA。

## CTA 的总数是不是可以被 SM's CTAs 的数量整除？一个满的 Wave 和一个几乎空的 Wave 耗时差不多。

举个例子，H20 有 78 个 SMs。假设每个 SM 只能调度一个 CTA，那么 78 个 CTA 需要一个 Wave，79 个 CTA 则要两个 Wave，耗时直接翻倍。

NOTE:

- Launch kernel 时，如果可以控制 CTA 的数量，尽量避免 Tail effect（也即非满波执行）。
- 运行时可以通过 cudaOccupancyMaxActiveBlocksPerMultiprocessor 拿到每个 SM 可以调度多少 CTAs。


# 术语解释

## Waves

"Waves" 表示在一个 SM 上需要多少批次（waves）才能完成所有分配给该 SM 的线程块的执行。

在 CUDA 中，线程块是执行的基本单位。每个 SM 可以同时执行多个线程块，但由于硬件资源（如寄存器、共享内存等）的限制，一个 SM 上能同时执行的线程块数量是有限的。

一个 wave 中包含的线程块数量并不是一个固定的数值，而是根据具体的内核（kernel）和设备的资源限制来变化的。

如果waves小于1，那么ncu会提示增加grid的大小。

## Wavefront

Wavefront 这个术语在 AMD GPU 架构中更为常见，但在 NVIDIA GPU 中，类似的概念通常被称为 warp。

## Scheduler

scheduler（调度器）是指 GPU 内部负责管理和调度线程执行的硬件单元。

调度器的主要任务是根据资源可用性和依赖关系，决定哪些线程可以在当前时钟周期内执行，从而最大化 GPU 的利用率和性能。

每个 Streaming Multiprocessor (SM) 包含多个调度器（scheduler），每个调度器可以管理一定数量的线程束（warps）。

Volta、Turing、Ampere、Ada架构都是 4 schedulers/SM，16 warps/scheduler。

每个调度器（scheduler）中的线程束（warp）可以来自于不同的线程块（thread block）。

## Sector

在 GPU 内存架构中，sector 通常指的是内存访问的基本单位。具体来说，sector 是内存控制器一次可以处理的最小数据块。

对于 NVIDIA GPU，sector 的大小通常是 32 字节。


## bank conflict

bank conflict（存储体冲突）是指多个线程在同一个时钟周期内访问同一个共享内存存储体（bank），导致访问冲突，从而引起性能下降。

共享内存被划分为多个存储体，每个存储体可以在一个时钟周期内处理一个访问请求。

如果多个线程在同一个时钟周期内访问同一个存储体，就会发生存储体冲突，导致这些访问请求被串行化处理，从而降低性能。

在 NVIDIA GPU 中，共享内存通常被划分为 32 个存储体。每个存储体的大小通常是 4 字节（32 位），但这可能会根据具体的 GPU 架构有所不同。

```
Bank    |      1      |      2      |      3      |     ...     |      16     |
Address |  0  1  2  3 |  4  5  6  7 |  8  9 10 11 |     ...     | 60 61 62 63 |
Address | 64 65 66 67 | 68 69 70 71 | 72 73 74 75 |     ...     |     ...     |
...
```

# Compute Workload Analysis

## 主页词条

- Executed Ipc Elapsed

    Executed Ipc Elapsed 表示在整个内核执行期间，每个时钟周期内实际执行的指令数。它是一个全局的指标，反映了内核在整个执行过程中指令执行的平均情况。

    解释：Executed Ipc Elapsed = 总执行指令数 / 总时钟周期数

    用途：用于评估内核在整个执行期间的指令执行效率。较高的 Executed Ipc Elapsed 值通常表示内核在大部分时间内都在有效地执行指令。

- Executed Ipc Active

    Executed Ipc Active 表示在内核执行期间，活跃时钟周期内每个时钟周期实际执行的指令数。活跃时钟周期是指至少有一个 warp 在执行指令的时钟周期。

    解释：Executed Ipc Active = 总执行指令数 / 活跃时钟周期数

    用途：用于评估内核在活跃时钟周期内的指令执行效率。较高的 Executed Ipc Active 值表示在活跃时钟周期内，内核能够有效地执行指令。

- Issued Ipc Active

    Issued Ipc Active 表示在内核执行期间，活跃时钟周期内每个时钟周期发出的指令数。发出的指令数包括所有被调度但未必执行的指令。

    解释：Issued Ipc Active = 总发出指令数 / 活跃时钟周期数

    用途：用于评估内核在活跃时钟周期内的指令调度效率。较高的 Issued Ipc Active 值表示在活跃时钟周期内，内核能够有效地调度指令。

- SM Busy

    SM 的忙碌程度，也即某个时间段内，SM 正在执行指令的比例。这个指标反映了 SM 的利用率。

    定义：SM Busy 是指在给定的时间段内，至少有一个 warp 在 SM 上执行指令的时间比例。

    意义：高 SM Busy 值表示 SM 大部分时间都在执行指令，说明计算资源利用率较高。低 SM Busy 值可能意味着存在资源闲置，可能是由于内存带宽限制、指令调度问题或其他瓶颈。

- Issue Slots Busy

    指令调度器的忙碌程度。具体来说，它表示在某个时间段内，指令调度器正在发出指令的比例。

    定义：Issue Slots Busy 是指在给定的时间段内，指令调度器正在发出指令的时间比例。

    意义：高 Issue Slots Busy 值表示指令调度器大部分时间都在发出指令，说明指令流动顺畅。低 Issue Slots Busy 值可能意味着指令调度器存在空闲时间，可能是由于指令依赖、资源冲突或其他调度问题。

# Memory Workload Analysis

## 主页词条

- Memory Throughput [Gbyte/second]

    内核执行期间的平均传输速度，也即实际带宽。

- Mem Busy

    Mem Busy 表示内存子系统在内核执行期间的忙碌程度。具体来说，它反映了内存控制器在处理内存请求时的忙碌时间比例。

    解释：Mem Busy = 内存控制器忙碌时间 / 内核执行总时间

    用途：用于评估内存子系统的利用率。较高的 Mem Busy 值表示内存控制器在大部分时间内都在处理内存请求，可能存在内存带宽瓶颈。

- Mem Pipes Busy

    Mem Pipes Busy 表示内存管道在内核执行期间的忙碌程度。具体来说，它反映了内存管道在处理内存请求时的忙碌时间比例。

    解释：Mem Pipes Busy = 内存管道忙碌时间 / 内核执行总时间

    用途：用于评估内存管道的利用率。较高的 Mem Pipes Busy 值表示内存管道在大部分时间内都在处理内存请求，可能存在内存带宽瓶颈。

- Max Bandwidth

    Max Bandwidth 表示内核在执行期间所达到的最大内存带宽对于理论带宽的比值，反映了内核在这一时刻的内存传输速率。

    解释：Max Bandwidth = 内核执行期间的最大内存传输速率

    用途：用于评估内核在执行期间的内存带宽利用情况。较高的 Max Bandwidth 值表示内核在某一时刻能够充分利用内存带宽。

## 词条关联性

- Mem Busy vs. Mem Pipes Busy

    Mem Busy 主要反映内存控制器的忙碌程度，而 Mem Pipes Busy 反映内存管道的忙碌程度。

    这两个指标都用于评估内存子系统的利用率，但它们关注的具体组件不同。内存控制器负责管理内存请求的调度和处理，而内存管道负责实际的数据传输。

- Memory Throughput vs. Mem Pipes Busy

    实际带宽和理论带宽的比值，通常略小于Mem Pipes Busy。Memory Throughput / Theoretical Memory Throughput <= Mem Pipes Busy。

# Warp State Statistics

## 阻塞原因

1. Stall Long Scoreboard

    含义：线程束（warp）由于等待长延迟操作（如全局内存访问、线程的局部内存访问）而停滞的时间。

    原因：通常是由于等待全局内存加载或存储操作完成。

2. Stall Barrier

    含义：线程束由于等待同步操作（如 __syncthreads()）而停滞的时间。

    原因：线程束在同步点等待其他线程束到达。

3. Selected

    含义：线程束被调度器选中并正在执行的时间。貌似为1，并且以此来衡量其他项的开销。

    原因：表示线程束正在执行指令。

4. Stall Wait

    含义：waiting on a fixed latency execution dependency

    原因：未知（猜测是指令流水线中的数值依赖导致等待）。通常来说这个值不大，优先考虑其他项。

5. Stall Short Scoreboard

    含义：线程束由于等待短延迟操作而停滞的时间。
    
    原因：通常是访问共享内存和一些特殊的数学计算如MUFU。

6. Stall Branch Resolving

    含义：线程束由于等待分支指令的解析而停滞的时间。

    原因：通常是由于分支预测错误或分支指令的延迟。

7. Stall Not Selected

    含义：线程束由于未被调度器选中而停滞的时间。

    原因：调度器选择其他线程束执行，当前线程束未被选中。
