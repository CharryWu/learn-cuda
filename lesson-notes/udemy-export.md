# Mastering GPU Parallel Programming with CUDA: (HW & SW)

_Exported from intuit.udemy.com · 166 notes_

---

## GPU vs CPU (very important)

*(Lecture 1)_

**[2:38]** ALU is a core in both GPU and CPUs

**CPU:**

- Powerful ALU, much more powerful and complex than GPU cores
  - CPU cores/ALU are designed for general purpose computing, executing single thread quickly (sequential execution)
- Large Cache memory per ALU
- Good for sequential applications

**GPU:**

- Thousands of ALUs/cores
  - Optimized to process multiple threads of same tasks (parallel execution)
- small caches per core
- Good for parallelized applications, each instructions is less dependent on its predecessor

**DRAM:**

- Exists in both CPU & GPU
- Called "Device global memory" on GPU

---

**[10:48]** If there're 1000 instructions independent of each other, they can be put onto GPU that has 1000 cores which results in 1 cycle of runtime; however, if they're dependent of each other, 1000 instructions will take AT LEAST 1000 cycles to execute since later instructions must wait the result of prev ones

---

**[15:26]** Streaming Multiprocessor (SM): contains multiple groups of cores, each group dedicated to a specific task (float point operations, integer operations, etc)

Contains:

- L1 cache shared by all ALUs inside SM
- Scheduler unit, dispatcher unit
- Register files
- Tensor cores (designed specifically for matrix multiplications)
- Float point cores (FP32, FP64)
- Integer operation cores
- Special function units (logarithm calculations)
- Load & store units (reading & writing data to memory)

---

## Architectures and Generations relationship [Hopper, Ampere, GeForce and Tesla]

*(Lecture 3)_

**[1:27]** GPU architecture is how GPU is built and cores are arranged, data flows, determines GPU efficiency, performance, and **capabilities**

For example, Turing introduced real-time ray tracing and tensor core units to support AI computing. Other older architecture codenames: Hopper, Ampere, Volta

---

**[2:58]** A separate thing is product "category", or "generation". NVDA offers its product in 2 configurations: Standard consumer grade, AND HPC

RTX are standard category (Tegra, GeForce, Quadro generation codename), while H100, A100, V100 are HPC (Tesla is generation codename)

---

**[5:05]** Ampere architecture includes GPU from multiple generations — GeForce RTX 3090 & Tesla A100

We can say that 3090 and A100 are based on the same architecture but for different purpose: A100 is for supercomputers & cloud servers, 3090 is for gaming. Note that Tesla was also an architecture codename.

---

**[6:52]** Tegra: Mobile devices, low-power consumption, smart phones, tablets. Example: Nintendo Switch, Microsoft Zune

GeForce: Gaming, Video editing, regular consumers, PC graphics — RTX 3060, 3070, 3080, 3090

Quadro (Replaced by RTX label): Professional cards, designers, engineers — Z4 G4 workstation, RTX A4000, RTX A6000. Note: RTX Axxxx = Ampere architecture

---

## How to Know the Architecture and Generation

*(Lecture 4)_

**[1:13]** Techpowerup to search GPU and its architectures and detailed specs; top500.org for top supercomputers

---

**[5:55]** V100, A100 do NOT have cooling fans because they are typically installed in datacenters which have robust cooling systems (air or liquid cooling)

---

## The Difference Between the GPU and the GPU Chip

*(Lecture 5)_

**[2:05]** GPU itself is a whole product containing GPU chip, memory, and peripheral interfaces (Cooling Fan / radiator 散热片), PCIe bus

---

**[3:06]** GPU chip on A100 GPU is GA100 (Ampere architecture); GPU chip on F100 GPU (Fermi architecture) is GF100

GPU chip is the brain of GPU; its architectural diagram & graphs can also be inspected on techpowerup

---

## The Architectures and the Corresponding Chips

*(Lecture 6)_

**[1:34]** "Tesla Ampere": Tesla is the generation (used for HPC & supercomputers), while Ampere is the architecture name

---

**[1:52]** GA102 chip is usually used in RTX, but also has a specific version for HPC: GA102-890-A1

---

**[4:48]** There's a minute difference between vendors of NVIDIA chips and minor perf difference — pay attention to those. For example, GIGABYTE RTX 3090 boost clock is +4% than NVIDIA RTX 3090

---

## Nvidia GPU Architectures From Fermi to Hopper

*(Lecture 7)_

**[4:46]** Higher clock speed = more power & energy consumption, not necessarily a good thing. HPC chips typically have lower clock speed than consumer chips because of power consumption considerations

---

**[5:59]** Single precision (float point): texture & pixel rate metrics. Double precision: scientific computing

---

## Parameters Required to Compare Between Different Architectures

*(Lecture 8)_

**[1:15]** throughput = speed, measured in Tflops = Core count × clock speed of GPU cores

> Note: GPU has two clock speeds — one for memory, one for GPU cores

---

**[5:00]** Higher bandwidth = shorter response to resolve multi-core read requests = smoother gaming experience

---

**[5:48]** Bandwidth = memory speed × bus width

- bus width = # bits that can be read at the same time
- memory speed = clock speed of memory; essential aspect of GPU performance

---

**[11:36]** DDR4, DDR6, HBM _(memory types)_

---

**[17:47]** 更多 GPU 核心并不等于更快的处理性能 — Higher GPU core count may lead to longer nanoseconds per cycle, therefore canceling the improvement of higher GPU core counts

---

**[20:48]** Higher core + higher speed = increased energy consumption; companies may opt for lower speed GPUs simply to reduce utility and cooling bills

---

**[21:58]** Tensor cores were introduced in Volta architecture in 2017; enhances matrix computation significantly ~8× vs. Pascal on the same task

---

## Half, Single and Double Precision Operations

*(Lecture 9)_

**[2:52]** FP16 (half precision): suitable for machine learning and graphic processing since they occupy less space (ML doesn't need high value representations)

FP32 (single precision): middle ground, go-to choice for many tasks

FP64 (double precision): powerhouse for scientific computing and complex simulations; very high precision needed where every decimal point matters!

---

**[5:19]** Double precision addition takes 4 cycles to complete while single precision addition takes 2 cycles

---

**[6:18]** Limitation of half precision: it only supports up to 2 or 3 digits after decimal point

---

## Compute Capability and Utilizations of the GPUs

*(Lecture 10)_

**[0:16]** Compute Capability (CC) number: Nvidia's proprietary classification/measurement of GPU computing power

Format: X.X — each version corresponds to a specific generation of GPUs and capabilities they support; first X is major, second minor

---

**[2:38]** Some operations need a minimum CC. CC also determines which CUDA kit can run on it: higher CC supports newer CUDA version

---

**[4:59]** PTX programming compatibility

---

**[7:11]** On Hopper architecture, you cannot use CUDA toolkit < 11.8; otherwise you will get an incompatible version error

---

## Before Reading Any Whitepapers!! Look at This

*(Lecture 11)_

**[0:18]** GPU Whitepapers — in-depth technical sheet

---

**[3:58]** Whitepaper structure:

1. Intro (new features, novel enhancements in HW & SW)
2. Streaming Multiprocessor
   - Each SM contains some execution units, called "cores" (single FP32/double FP64 float point)
3. Performance comparisons
4. Technical Specification

---

## Volta + Ampere + Pascal + SIMD (Don't skip)

*(Lecture 12)_

**[10:22]** Ampere IADD = 2 cycles, Volta = 4 cycles, Pascal = 6 cycles. Nvidia not only enhances the # cores on GPU chips but also enhances the instruction cycles

---

**[12:08]** Whitepaper also includes latencies (measured in GPU cycles) of tensor cores, affected by input matrix dtype & shape

---

**[13:30]** Volta's 2017 introduction of tensor core improves performance: 25 TFLOPS (P100) → 125 TFLOPS (V100)

- Volta architecture allows you to run BOTH integer & float point operations at the same time
- Introduced Multi-Process Service (MPS): enables improved performance, isolation, and QoS for multiple compute applications sharing the same GPU (48 processes)
- Unified memory — enhanced GPU and CPU memories into one single memory

---

**[24:14]** SM block is divided into different partitions — allowing smaller assignments of software tasks. Each partition is a combination of different types of computational units

---

**[26:27]** A group of threads is called a CUDA block. Group of 32 threads is called a Warp (线程束). CUDA blocks are grouped into a grid. A kernel is executed as a grid of blocks of threads. 多个线程块组合成一个网格。同一网格中的所有线程块包含相同数量的线程。

- A single thread runs on a single core
- A thread block runs on an SM
- A CUDA kernel runs on a GPU device

---

**[27:08]** SIMT = Single Instruction Multiple Thread — an execution model where a single central "Control Unit" broadcasts an instruction to multiple "Processing Units" for them to all _optionally_ perform simultaneous synchronous and fully-independent parallel execution of that one instruction. Each PU has its own **independent data and address registers** and its own independent Memory, but **no** PU in the array has a Program Counter.

Warp scheduler in Nvidia chips are essentially built on the SIMT concept. Each warp is 32 threads independently executing the same instruction. For example, in vector addition, 32 threads in the same Warp execute the same add instruction but on different data (element-wise on each vector).

---

**[30:50]** In A100 architecture, all 4 partitions of a SM have independent L0 instruction cache, warp scheduler, dispatch unit, register file and PU (cores), but share L1 data memory and texture memories

---

**[35:28]** Mem size (GB) × bus width (bits) × speed (clock rate, MHz) = memory bandwidth (GB/sec)

---

**[40:03]** Accumulator: the register that stores the result — For FP16 × FP16, result might be too large to store in FP16 registers, so FP32 registers are needed as accumulators

---

**[43:32]** Ampere architecture has fewer tensor cores than Volta architecture; however, fewer cores does NOT mean worse — overall performance must consider single-core performance as well as other specs

---

**[34:55]** Pascal CAN execute integer operations, but not on a standalone unit; when executing INT calculations, it cannot perform FP32/FP64 operations simultaneously

---

**[46:02]** NVLinks is NVidia's proprietary interconnect technology which connects GPU-GPU, GPU-CPU; provides significantly more performance than PCIe bus

---

**[46:36]** NVIDIA product encompasses all scenarios — single GPU for gamers/video editors, single GPU for scientific computing & ML workstations, then GPU servers (4~8 GPUs interconnected by NVLink)

---

**[48:03]** Pascal has 1 link between each GPU; Volta has 2 links

---

## What Features Are Installed with the CUDA Toolkit?

*(Lecture 13)_

**[0:10]** CUDA — Compute Unified Device Architecture

---

**[1:21]** nvcc is CUDA's compiler. CUDA code → PTX code.

PTX (Parallel Thread Execution) is a virtual assembly language for NVIDIA GPUs — a stable intermediate representation that can be compiled to specific hardware.

PTX code is translated into SASS (Shader Assembly) for the target GPU architecture. SASS is specific to the underlying NVIDIA GPU architecture and represents the actual microcode that the GPU's processing units execute.

---

**[1:45]** Under the CUDA umbrella, its capability is enriched by numerous libraries:

- cuBLAS: GPU accelerated Basic Linear Algebra Subprograms
- cuFFT: Fast Fourier Transform library for GPUs
- cuRAND: Random number generation library
- cuDNN: Deep Neural Network library for deep learning applications

These libraries provide pre-optimized, ready-to-use functions that save time and enhance performance

---

**[3:39]** CUDA Runtime and Driver APIs: `cudaMalloc()`, `cudaMemcpy()`

---

**[4:02]** Nvidia debugging tools:

1. CUDA-GDB: Debugging CUDA applications on *nix systems
2. CUDA-memcheck: tool to detect and diagnose memory errors in CUDA applications
3. Nsight Systems: For system-wide performance analysis
4. Nsight Compute: For detailed CUDA kernel profiling and analysis

---

**[4:44]** CUDA samples are more than demo/toy codes; they demonstrate various CUDA programming techniques as a guide to writing efficient CUDA code — from basic concepts to advanced operations. Studying CUDA samples can give invaluable insights, complementing theoretical learnings with practical insights.

---

**[5:52]** CUDA toolkit = compiler (NVCC) + profiling + debugging + guide/reference

---

## Mapping SW from CUDA to HW + Introducing CUDA

*(Lecture 18)_

**[0:38]** GPGPU = General-purpose computing on Graphics Processing Units

---

**[1:40]** NVCC is MORE than just a compiler — it optimizes CUDA applications

---

**[2:00]** Host: CPU + associated memory (RAM)

Device: Nvidia GPU + GDRAM inside GPU device

Understanding Host code vs. Device code is crucial in crafting code structure, data transfer, and memory handling

---

**[6:25]** GPU Level: CUDA application as a whole is executed on entire GPU

SM Level: Each (thread) block in CUDA application is executed on SM level

- Imagine block = software tasks/jobs, SM = software workers
- GigaThread Unit is SM-level thread block manager & dispatcher

Partition level: Warp scheduler divides thread block into smaller units — warp of threads, assigns one of 8 warps from same block to the partition, in each GPU cycle

Core level: each computing core (FP32/FP64) is assigned a single thread from 32-thread warp

---

**[12:16]** 2 key params during CUDA app initialization:

- Total # (thread) blocks

- # threads within each block

`# warp / block = (# threads / block) / 32`

---

## 001 Hello World Program (threads - Blocks)

*(Lecture 19)_

**[2:31]** Kernel in CUDA is equivalent to a function in C lang

Kernels start with `__global__` and are executed on GPU, not CPU

---

**[6:08]** Pre-defined variables accessible in kernel:

- `blockDim` — total # threads in block
- `blockIdx` — ID of current block (0 ~ 9)

---

**[9:35]** 调用 CUDA kernel 基础语法：

`kernel_name<<<grid_size, block_size>>>(param1, param2, ...);`

`grid_size` 是整个 kernel 需要的线程块 (block) 数量；`block_size` 是每个线程块里面的线程数量

<https://stackoverflow.com/a/26774770>

---

**[15:31]** **\*\*IMPORTANT: what will happen if init param exceeds max limit of threads\*\***

When you initiate a kernel of 2048 threads per block on A100 GPU, the code will compile but it doesn't work; no output will be printed. A100 supports up to 1024 threads per block.

---

**[17:27]** If your CUDA application needs more threads than the max limit per block, you can increase total # blocks `<<<num_blocks, num_threads_per_block>>>`.

- You can write more than 32 blocks on A100 — 32 is blocks per SM, not per GPU!

---

## 002 Hello World Program (Warp_IDs)

*(Lecture 21)_

**[3:42]** Warp ID = threadIdx.x / 32

1 warp = 32 threads

In terms of scale: grid > block > warp > thread

---

## 003 Vector Addition + the Steps for Any CUDA Project

*(Lecture 22)_

**[5:53]** This code does NOT need any looping on GPU compared to CPU:

```cuda
__global__ void vectorAdd(int *A, int *B, int *C, int n) {
  int i = threadIdx.x;
  C[i] = A[i] + B[i];
}
```

The instructions of `C[i] = A[i] + B[i]` execute in parallel on each GPU SM

---

**[8:26]** Before running a kernel, input data must be copied from host memory to device memory:

`cudaMemcpy(void* dest, void* src, size, cudaMemcpyHostToDevice);`

After computation, output data must be copied from device to host:

`cudaMemcpy(void* dest, void* src, size, cudaMemcpyDeviceToHost);`

---

**[10:03]** Final step is to release memory spaces on both CPU & GPU:

```c
cudaFree(device_A);
cudaFree(device_B);
cudaFree(A);
cudaFree(B);
```

---

**[13:54]** Initialize memory on device:

```c
cudaMalloc((void**) &device_A, size);
cudaMalloc((void**) &device_B, size);
```

Compare to initialize on host (regular C syntax):

```c
A = (int *) malloc(size);
B = (int *) malloc(size);
```

---

## 004 Vector Addition + Blocks and Thread Indexing + GPU Performance

*(Lecture 23)_

**[1:20]** 作为 CUDA 开发者，你可以自由决定程序所用的线程 blocks 数量和单个 block 里面的线程数量

---

**[5:11]** 把一整个长数组拆分成符合 GPU 架构的小数组：

`arrIndex = threadID + blockID * threads_per_block`

`int i = blockIdx.x * blockDim.x + threadIdx.x`

---

**[8:37]** GPU 运用率 = 被实际调用的流式处理器 SM 个数 / GPU 内总流式处理器 SM 个数

---

**[8:42]** 每个线程块在一个流式处理器上运行。每个线程块最多可以有 1024 个线程（nvidia CC 从 2.x/3.x 开始到当前最新架构的限制），最少可以有 32 个线程（单个 warp）

---

**[18:02]** 1. **Parallel Execution (Inside a Warp)**

The fundamental unit of execution on an SM is the **warp** — a group of 32 threads.

**SIMT Execution:** At any given clock cycle, the SM's scheduler issues one instruction to an entire warp. All 32 threads execute that _same_ instruction at the _exact same time_, but on their own private data.

---

**[18:21]** 2. **Concurrent Execution (Between Warps)**

A single block is made of one or more warps. An SM can hold and manage many warps at once, often from multiple different blocks.

**Latency Hiding:** If the active warp has to pause (e.g., waiting for a slow memory read), the SM **immediately switches to another "ready" warp**. This is the concurrent execution.

---

**[18:03]** Two key execution modes:

1. **Parallel Execution**: Multiple SMs run their assigned blocks at the same time. A block runs entirely on a single SM from start to finish.
2. **Concurrency on one SM**: A single SM can hold and execute multiple blocks concurrently. The SM switches between active warps from all resident blocks to keep the processing units busy.

---

**[12:43]** 有两种方式去衡量运行时间性能：

1. CUDA 自身的计时 API: `cudaEventCreate`, `cudaEventRecord`, `cudaEventSynchronize`, `cudaEventElapsedTime`
2. Nvidia 打包好的 profiler，比手动调用 API 更加精准: nsight systems, nsight compute

---

## 005 Levels of Parallelization — Vector Addition with Extra-Large Vectors

*(Lecture 24)_

**[18:31]** If vector is too large for computation, you need to divide inputs into chunks

---

## Query the Device Properties Using the Runtime APIs

*(Lecture 25)_

**[0:54]** CUDA Runtime API: GPU limits and constraints such as max number of threads per block. API provides functions that allow you to query that value directly from GPU. This is an alternative to reading specs from whitepaper.

---

**[1:50]** `cudaGetDeviceProperties` — allows you to query the properties and capabilities of GPU device

Returns:

- `maxThreadsDim` — maximum threads per block
- `name[256]` — name of GPU device
- `totalGlobalMem` — total byte size of global memory spaces

---

**[8:34]** `2 * prop.memoryClockRate * (prop.memoryBusWidth/8) / 1.0e6` ← get peak value of memory bandwidth

---

**[9:41]** <https://stackoverflow.com/questions/2392250/understanding-cuda-grid-dimensions-block-dimensions-and-threads-organization-s>

Threads are organized in blocks. A block is executed by a multiprocessing unit. Threads of a block can be identified using 1D (x), 2D (x,y), or 3D (x,y,z) indexes, where x\*y\*z <= 768 (other restrictions apply). If you need more threads, you need more blocks.

---

**[15:33]** 线程块是软件编程概念，SM 流处理器是硬件概念

---

**[16:53]** A grid is a collection of blocks that collectively form the entire parallel computation launched by a CUDA kernel.

- Blocks within a grid are independent and cannot directly communicate via shared memory or `__syncthreads()`.
- Grids can be arranged in 1D, 2D, or 3D configurations.
- Each block within a grid has a unique `blockIdx` (e.g., `blockIdx.x`, `blockIdx.y`, `blockIdx.z`)

---

**[18:04]** `cudaError_t`: 1) Return Success 2) Return Error code indicating why the kernel failed to execute

`__host__ __device__ cudaError_t cudaGetDeviceCount(int* count)`

---

**[4:57]** CU code snippet to get device properties:

```c
#include <cuda_runtime.h>
// ...
int device;
cudaGetDevice(&device);
cudaDeviceProp prop;
cudaGetDeviceProperties(&prop, device);
```

> `cudaSetDevice` and `cudaGetDevice` always work on **logical** visible devices from `0` to `num_visible_devices`. For example, if you `setenv CUDA_VISIBLE_DEVICES 3,6`, then `cudaSetDevice(1)` will actually work on **physical** device `6`, not `1`.

---

```c
#include <stdio.h>

int main() {
    int nDevices;
    cudaGetDeviceCount(&nDevices);
    for (int i=0; i<nDevices; i++) {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, i);
        printf("Device ID: %d\n", i);
        printf("Device name: %s\n", prop.name);
        printf("Memory Clock Rate (KHz): %d\n", prop.memoryClockRate);
        printf("Memory Bus Width (bits): %d\n", prop.memoryBusWidth);
        printf("Compute capability: %d.%d\n", prop.major, prop.minor);
        printf("Peak memory bandwidth (GB/s): %f\n\n", 2.0 * prop.memoryClockRate * (prop.memoryBusWidth/8) / 1.0e6);
        printf("Number of SMs: %d\n", prop.multiProcessorCount);
        printf("Number of maxThreadsPerBlock: %d\n", prop.maxThreadsPerBlock);
    }
}
```

---

## Nvidia-smi and Its Configurations (Linux User)

*(Lecture 26)_

**[0:16]** `nvidia-smi` is Nvidia's System Management Interface (系统管理界面) — a command-line utility to monitor and manage GPUs

---

**[7:09]** nvidia-smi has three components: performance monitoring (utilization, memory usage, temp, power), settings management (fan speed, power limits), and GPU system info queries (name, driver version, etc.)

```bash
nvidia-smi -l         # continuously monitor
nvidia-smi -L         # list all detected GPUs
nvidia-smi -r         # reset the GPU
nvidia-smi --query-gpu=gpu_name,driver_version,temperature.gpu --format=csv
nvidia-smi -i 0 -pl 150   # limit GPU 0 power cap to 150W
nvidia-smi -l 5 > gpu_log.txt   # refresh every 5s, output to file
```

---

**[nvidia-smi will NOT capture other applications]** nvidia-smi's "processes" section only captures CUDA applications running on GPU; it will NOT capture video recording/playing/games running on GPU

---

**[18:33]** NVIDIA GPU persistence mode **keeps the driver and GPU state initialized even when no applications are using it**, which reduces startup latency for subsequent applications.

Enable via `nvidia-smi -pm 1` (temporary) or configure `nvidia-persistenced` service for boot-time setting.

> 很多时候，如果需要改其他设置，需要先打开 persistence mode `nvidia-smi -pm 1` 才能修改

---

**[20:45]** `-d TYPE, --display=TYPE` — Display only selected info: MEMORY, UTILIZATION, ECC, TEMPERATURE, POWER, CLOCK, COMPUTE, PIDS, PERFORMANCE, SUPPORTED_CLOCKS, etc. Flags can be combined with comma e.g. `"MEMORY,ECC"`.

---

**[21:44]** Clock speed is auto-lowered by GPU to save power consumption, but can be fixed manually

---

**[23:42]** Memory frequency and/or GPU clock frequency must be one of the hardware-supported integer values (MHz):

```bash
nvidia-smi -q -d SUPPORTED_CLOCKS
nvidia-smi --application-clocks=1215,765
```

---

## The GPU's Occupancy and Latency Hiding

*(Lecture 27)_

**[1:25]** Occupancy: 利用率 — measure of GPU utilization

- **Theoretical occupancy** = (warp used by kernel) / (max warps per SM) = (threads used by kernel) / (max threads per SM)
  - Warp limits on GPU from whitepaper OR from runtime API
  - 理论利用率计算公式很简单，直接除就是了。不考虑任何内存/计算/线程间数据依赖的瓶颈

---

**[2:55]** CU code snippet to get device properties + `cudaDeviceGetAttribute`:

```c
#include <cuda_runtime.h>
// ...
int device = 0;
cudaGetDevice(&device);
int maxThreadsPerMP = 0;
cudaDeviceGetAttribute(&maxThreadsPerMP, cudaDevAttrMaxThreadsPerMultiProcessor, device);
```

<https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__DEVICE.html>

---

**[7:11]** List of attributes for `cudaDeviceGetAttribute`:

<https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html#group__CUDART__TYPES_1g49e2f8c2c0bd6fe264f2fc970912e5cd>

- `cudaDevAttrMaxThreadsPerBlock`: 每个线程块最高的线程数量
- `cudaDevAttrMaxThreadsPerSM`: 每个流式处理器最多的线程块数量
- `cudaDevAttrMaxRegistersPerBlock`: 每个线程块可用的 32 位寄存器数量
- `cudaDevAttrMultiProcessorCount`: GPU 里面总共的流式处理器数量
- `cudaDevAttrMaxThreadsPerMultiProcessor`: 每个处理器可同时处理的线程数量
- `cudaDevAttrMemoryClockRate`: 内存频率

以上的属性都有对应的 nvidia-smi 命令去获取

---

**[13:50]** You can profile CUDA programs using nsight compute command `ncu ./cuda_app`

---

**[25:21]** 再次说明一下:

- CUDA 应用在 GPU 上运行
- 应用里面的一个线程块在一个流式处理器上运行
- 应用里面的线程束 warp 在流式处理器中的一个 partition 里运行（CUDA 最小执行单位）
- 最后是单个线程在单个运算单元（FP64, FP32, INT, TENSOR CORE）与其他线程一并运行

---

**[27:28]** NVBIT: the trace collector of nvidia libraries. SASS instructions — GPU assembly code

---

**[27:30]** **流水线停顿** stall cycles: 当所有就绪的线程束都在等从内存中读取数据时就会发生流水线停顿（GPU 干等）

---

**[32:32]** GPU implements zero-cost switching between threads. If a thread stalls, a different thread takes its place. Load latency and instruction latency are the most common latencies to be hidden, typically in that order.

> This immediately tells us the most important mechanism to hide latency: run lots of threads so there is always another thread ready to run. That is the programmer's responsibility.

<https://forums.developer.nvidia.com/t/how-to-understand-the-hide-latency/258938>

---

**[38:42]** CPU 处理器会将指令重新排序，把无任何依赖的指令插到前面空位执行；但 GPU 并不是这样 — 由于有很多线程束待执行，GPU 调度器通常能在多个就绪的线程束找到下个指令，而无需调整执行顺序

---

**[40:05]** 当一个线程束指令被发送到 GPU 里面发生执行的时候，并不是单一的指令，而是多个指令一起执行在不同的数据上。例如:

```
FMUL R3 R4 R5
FMUL R6 R7 R8
FMUL R9 R10 R11
```

---

**[44:26]** Computation of one instruction = 1 cycle; instruction fetching + computing = 2 cycles

---

**[48:07]** LDG load global instruction: 30–300 cycles

---

**[Achieved Occupancy]** Achieved occupancy = active cycles / total cycles

Achieved 比 theoretical 难计算，因为需要在运行时 profile 程序，依赖指令集状态的动态变化

---

**[先计算每个流式处理器的每个分区的 achieved occupancy]** 先计算每个流式处理器的每个分区的 achieved occupancy；然后再取平均，算出该流式处理器的利用率；最后再取所有流式处理器平均利用率得到 GPU 的综合利用率

高利用率不总是等于高性能:

1. GPU 瓶颈也受到 CPU 和主内存限制
2. GPU 载荷分为计算类载荷与内存读取类载荷，后者不能充分利用 GPU 高并行特点
3. 如果 GPU 太热，它也会降频来避免烧坏元件

---

## Allocated Active Blocks per SM (Important)

*(Lecture 28)_

**[0:38]** Allocated Active Blocks per SM (AABS): 每个流式处理器能运行时的 block 指标，和利用率指标一样重要

---

**[2:11]** 再次重申 CUDA 编程模型: Grid > Block > Warp > Thread

A group of threads is called a **CUDA block**. **CUDA blocks** are grouped into a _**grid**_. A kernel is executed as a _**grid**_ of blocks of threads.

---

**[2:15]** The number of warps in a block = total threads in block / 32. For example, a block with 96 threads would contain 3 warps (96÷32=3).

---

**[2:15]** GPU 参数有两个限制，影响着运行时 CUDA 程序的并发数，两者取少（木桶短板效应）:

- **warp limit per SM**: 限制了一个流式处理器上能同时跑多少个线程束
- **block limit per SM**: 限制了一个流式处理器上能同时跑多少个线程块

在某些情况下，block limit per SM × # warps/block 可能大于 warp limit per SM，这时就是木桶短板效应起作用，取较小值作为实际能运行的处理器并发数量。nvcc 编译器会自动干预降低限制。`max register per SM` 也受到相同限制。

---

**[5:29]** We can assign millions of blocks/GPU, more than 1000 blocks per SM. No more than 32 blocks can run concurrently on each SM.

---

**[8:26]** 计算 AABS（寄存器）: 每个 CUDA 程序都有不同的 AABS 参数。程序本身需要的 thread per register 会影响它:

```
app required SM registers = # threads per block × # needed registers per thread
AABS = max registers per SM / app required SM registers
```

> 通常小于 max thread blocks per SM

---

**[11:37]** 总体来说，AABS 会受到 warp、register、shared memory 三个硬件参数的限制。GPU 调度器会取以下最小值作为最终的 AABS:

```
Block warp limit: 4
Block register limit: 8
Block shared memory limit: 16
→ Final AABS = 4
```

---

**[13:10]** `ncu` profiler command reports: Speed of light throughput, Launch statistics, Occupancy ← computed block limits based on specs of warp, register, shared memory

---

## The GPU's Occupancy and Latency Hiding (cont.)

*(Lecture 27 continued)_

**[13:50]** You can profile CUDA programs using nsight compute command `ncu ./cuda_app`

---

## All Profiling Tools from NVidia (Nsight Systems - Compute - nvprof ...)

*(Lecture 30)_

**[3:09]** 7 nvidia debugging/profiling tools:

- CUDA-MEMCHECK: memory error diagnose + CUDA app correctness checking
- CUDA-GDB
- NVIDIA Visual Profiler (nvvp)

---

**[3:09]** Nsight Systems: CUDA 应用总体的分析，是 compute 和 graphics 的 parent tools. Analyze on multi-kernel, multi-load scale

- Server profiling tool, not client profiling tool

Nsight Compute: 细粒度 (kernel level) 分析 — 是单个 kernel 分析主要的工具

Nsight Graphics: 专门用来分析图形学 3D 渲染的 CUDA 程序

---

## Error Checking APIs

*(Lecture 31)_

**[0:42]** 为什么 CUDA 去专门 error checking 很重要？因为很多时候 CUDA 程序中的错误（超出硬件限制，数据错误）并不会被编译器直接捕捉并汇报。编译器依然能成功编译。

CUDA 错误一般分为两种:

- **同步错误**: cudaMalloc 是同步类错误。如果无法分配内存，会阻止 host (CPU) 执行下一个指令。cudaMalloc 如果失败会返回错误码。
- **异步错误**: GPU kernel 启动不会阻塞 host/CPU

---

**[6:33]** INT Array vectors: 4 bytes each (32 bit int) in CUDA C

---

**[12:05]** C 编译器会默认所有整数数值为 int 类型，但如果进行连乘操作，数值特别大会溢出，编译器会报错。所以要告诉编译器数值是长类型的: `1024LL`

`1024LL * 1024 * 1024 * 20 // makes every value LL in multiplication`

`long long` is a data type used to store integers that require a larger range than `int` or `long`. Requires at least 64 bits of storage.

---

**[14:38]** 获取同步执行结果 + 错误处理例子：

```c
cudaError_t err;
err = cudaMalloc((void **) &d_A, size);
if (err != cudaSuccess) {
  fprintf(stderr, "Failed to allocate device memory - %s\n", cudaGetErrorString(err));
}
```

与之相对，如果是异步执行代码（如 kernel），通过这种方式获取的 err 可能不准确，因为异步在 GPU 设备上的代码会花时间，晚于 host 执行完毕。

---

**[21:25]** Asynchronous CUDA functions do not block the host/CPU from executing; that's why `cudaGetLastError` is needed:

```c
foo<<<1,1>>>(0); // async

cudaDeviceSynchronize();

cudaError_t error = cudaGetLastError();
if(error != cudaSuccess) {
  printf("CUDA error: %s\n", cudaGetErrorString(error));
  exit(-1);
}
```

<https://stackoverflow.com/a/34372414>

---

**[23:25]** `cudaGetLastError` returns the last error produced by any runtime call in the same instance of the CUDA Runtime library in the host thread and resets it to `cudaSuccess`.

- May also return error codes from previous, asynchronous launches.
- May also return `cudaErrorInitializationError`, `cudaErrorInsufficientDriver` or `cudaErrorNoDevice`.

---

**[25:34]** To avoid repetition of `if (err != cudaSuccess) { .... }`, use this C macro:

```c
#define cudaCheckError(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true) {
   if (code != cudaSuccess) {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}
```

Usage:

```c
cudaCheckError(cudaMalloc((void **)&d_A, size));
```

---

## Nsight Compute Performance Using Command Line Analysis

*(Lecture 32)_

**[5:51]** Launch Statistics displays the configured kernel information — block size, grid size, # SMs, Threads used, registers per thread, etc.

---

**[12:37]** <https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html>

---

**[14:49]** You have **100K metrics** you can collect from nsight compute

---

**[18:18]** `gpu`, `draw`, `lts`, `sm`, `l1tex` are the 5 most important metrics to observe

---

**[21:22]** `.sum` metrics is unexpectedly the `.max` metric multiplied by # SM counts, **not** `.avg` metric — 长板理论

---

## Vector Addition with a Size Not Power of 2 (!!important)

*(Lecture 35)_

**[10:49]** 一个 CUDA 程序最少调用 32 线程，即 1 warp

---

**[10:49]** 当数组元素不为 2 的次方时，可以改写 CUDA 程序内核算法本身，让 block 的数量最优（不然效用率低下）。如果 block 数量是 GPU 流处理器的倍数时，就能最大化利用时间（仅一波就执行完毕，而不是两波）

---

## Performance Analysis

*(Lecture 34)_

**[0:52]** `cudaFree` before or after `free`

**Handle device memory first:** It is generally good practice to free GPU memory with `cudaFree()` first. In many CUDA implementations, `cudaFree()` acts as a synchronization point, ensuring all previous CUDA operations are complete before the memory is released.

---

**[2:09]** CUDA C kernel 编程一般流程:

- 定义 host + device 指针变量
- `malloc` + `cudaMalloc` 给变量分配内存
- 初始化主机上的数据，并用 `cudaMemcpy` 拷贝到设备内存上
- 将指针变量当成参数传入内核并执行。启动参数 `<<<blocksPerGrid, threadsPerBlock>>>`
- 用 `cudaMemcpy` 把数据从设备内存拷贝到主机内存上
- 释放内存 `cudaFree`, `free`

---

**[21:26]** 一般来说，blocksPerGrid 参数由以下公式决定：

`int blocksPerGrid = (SIZE + threadsPerBlock - 1) / threadsPerBlock;`

---

**[24:49]** Total blocks / (AABS × # SMs on GPU) = number of waves to run per SM

For example, if AABS = 16 blocks, each SM will run 16 blocks at one time; after that, it will ask for the next batch.

> Next wave cannot run before prev wave finishes (even if prev wave blocks are all stalled)

---

**[26:03]** Block size × total blocks = Total GPU elements

---

## Matrices Addition Using 2D of Blocks and Threads

*(Lecture 36)_

**[2:19]** If 1D:

```
Block size = blockDimX = number of threads per block = 1024
Grid size = GridDimX = 1024 blocks
```

If 2D: `blockDimX * blockDimY` must be <= max(blockDimX)

---

**[4:46]** The RTX 3060 is based on the **Ampere architecture** with **Compute Capability 8.6** (GA106 chip, **28 SMs**).

- Max Threads per Block: 1024
- Max Warps per Block: 32
- Max Thread Dimensions: (1024, 1024, 64)
- Max Grid Dimensions: (2^31 - 1, 65535, 65535)

The maximum number of active blocks in a given time is linked to the Compute Capability of your GPU.

---

**[17:16]** 矩阵乘法本质：在一维化的内存空间计算二维度的矩阵相乘

---

**[18:51]** 计算矩阵加法：矩阵元素下标映射到一维的内存地址下标

```cuda
__global__ void matMul(int * A, int * B, int * C, unsigned int N, unsigned int M) {
  unsigned int ix = threadIdx.x + blockIdx.x * blockDim.x;
  unsigned int iy = threadIdx.y + blockIdx.y * blockDim.y;
  unsigned int nx = gridDim.x * blockDim.x; // number of threads per grid row
  if (ix < M && iy < N) {
    unsigned int idx = iy * nx + ix;
    C[idx] = A[idx] + B[idx];
  }
}
```

---

**[23:56]** 在 CUDA 处理矩阵类数据时，有三个坐标体系需要计算：

1. 矩阵元素本身的 ij 下标
2. 线程块 block 在 grid 中的下标，与线程 thread 在 block 中的下标：
   - `gridDim.x = ceil(M / blockDim.x) = (M + blockDim.x - 1) // blockDim.x`
   - `gridDim.y = ceil(N / blockDim.y) = (N + blockDim.y - 1) // blockDim.y`
3. 一维化的**内存地址**数组下标：总共需要 `(gridDim.x × gridDim.y) × (blockDim.x × blockDim.y)` 个线程

---

## Why L1 Hit-Rate is Zero?

*(Lecture 37)_

**[5:36]** GPU cache & memory hierarchy:

- 192KB L1 Data Cache + Shared memory occupy same GPU cache
- L2 Cache
- HBM

---

**[9:43]** Nsight compute tool assumes all **write** operations to L2 cache as a hit, even if it's not a hit.

For example, Matrix A + B = C has 2 read operations and one write operation, registering 1/3 assumed hit rate in Nsight compute. Therefore, you should focus on **READ hit rate of L2 cache** as accurate. L1 hit rate doesn't have this issue.

---

**[9:41]** Matrix addition always has hit rate of 0, since there's no reuse of matrix data loaded onto memory.

Matrix multiplication has hit rate > 0; there IS reuse of matrix data loaded onto memory.

---

## The Shared Memory

*(Lecture 38)_

**[0:42]** GPU 内存层次结构 (全局 → 共享 → 寄存器):

- **全局内存**: 所有 GPU 线程与 CPU 都能访问到
- **GPU L2 cache**: 所有 GPU 线程都能访问到
- **GPU L1 cache**: 在某个流式处理器内部，只能被该流式处理器的所有线程访问到 (private to the SM). 也被称作 hardware cache，只能被 GPU 硬件管理
- **Share memory**: 和 L1 cache 或 ALU 运算单元是一个层级（都在芯片上）。速度非常快，只能当前 block 内的所有 thread 访问，能被软件管理的内存。允许线程间通信和数据共享。
- **GPU 寄存器**: 一个寄存器属于一个线程，它的生命周期和线程一样长

---

**[8:41]** 共享内存存在的意义：如果多个线程需要用到同一份数据，只需要将其加载一次入共享内存里面就能减少全局内存的访问频率，降低内存瓶颈。全局内存非常非常慢。

而且也允许线程间通信，同理也能大大加快线程间通信的速度。

共享内存为什么会比 L1 cache 快？因为访问 L1 cache 需要进行地址搜索操作，需要算 address tag 和各种各样的前缀。访问共享内存没有这些操作。

实际上共享内存与 L1 cache 处在同一个内存单元里面。他们的总内存大小一样。

---

**[9:51]** 如何在共享内存上定义变量？

```cuda
__shared__ float tileA[32][32];
__shared__ float tileB[32][32];
```

---

**[11:17]** 在安培架构上，共享内存大小可以被设置成 (0, 8, 16, 32, 64, 100, 132, 164) KB，和 L1 cache 共享一个流处理器上的物理内存单元（合 192KB）

---

**[15:46]** 每次读取并不是单个内存，而是整个缓存行。每个内存行有 128 byte 的数据，32 个 bank，每个 bank 4 字节。所以每个 GPU cycle 总共可以读一个 cache line 的数据

---

**[19:54]** Shared memory Access Pattern: 想象内存布局是一个矩阵，有许多缓存行，每个缓存行有许多列，每一列的 cell 都是一个 bank，每个 bank 有 4 Byte 大小。

- 同一个 GPU cycle 中，同一 warp 的 threads 可以访问一个甚至多个缓存行不同列位置的 memory bank。如果访问列位置相同，但不在同一缓存行的 bank 就会有冲突
- 如果多个线程在同一个 GPU cycle 中访问同一个 bank，那么没有任何冲突。这叫 broadcasting

综上可以得出，每一列都有 number of accessed cells，取所有列该值的最大值：`max accessed cells per bank column` 为 A，那么至少需要 A 个 GPU cycles 才能完成全部数据访问。

`# bank conflicts = A - 1`（注意不需要乘 number of columns）

---

**[24:49]** 如果总共数据大小超过一个缓存行的大小，那么一定会有 bank conflict，需要不止一个 GPU cycle 读取所有数据

---

**[33:10]** nsight compute bank conflict will give out incorrect reading

---

## Warp Divergence

*(Lecture 39)_

**[6:51]** 当 kernel 中有 if-else 分支的时候，同一个 warp 里面的 thread 会执行不同的代码分支而产生 warp divergence。如果存在 warp divergence 的话，那么其中一个分支的线程会等待另一个分支所有线程执行完毕才开始执行。他们之间按顺序执行。所以会产生 GPU 空闲，浪费计算资源。

---

## Vector Reduction Using Global Memory Only (Baseline)

*(Lecture 41)_

**[15:07]** vector reduction: 输入是一维向量数组，输出是单个值的算法

Tree-based approach to vector reduction: 先将输入向量对应到树状结构的叶子节点，树是倒置的，然后每次计算都是上一层的两个（或者多个）叶子节点合并，计算结果就是到下层的节点。每层节点数递减。

---

**[20:55]** 在线程块的层面上没有"同步"功能，warp 和 kernel 层级上才有 synchronize 功能。因为实现线程块的硬件成本太高 — Expensive to build in hardware for GPU with high processor count

---

**[25:19]** Use 2 kernels and synchronization to compute vector reduction:

- **First kernel**: M blocks, each block maps to consecutive N elements → produces M-sized vector of intermediate/partial sum results
- **Second Kernel**: takes output of M size and outputs a singular final result

Both kernels resemble an inverted tree.

---

**[26:47]** Off-by-one error: `if (index+1 > n) ...`

---

**[48:46]** 任何输入为矩阵/数组的内核都需要同时传入他们的尺寸作为参数

---

**[27:18]** 分批累加一个元素和它的下一个步长 stride 后的元素

Filter by stride of 1, 2, 4 (sum every [2, 4, 8, ...] element) to compute the partial result:

```cuda
tid = threadIdx.x;
index = blockIdx.x * blockDim.x + threadIdx.x;
for (int stride = 1; stride < blockDim.x; stride *= 2) {
  __syncThreads(); // ensure all threads have completed the previous operations
  if (tid % (2 * stride) == 0 && index+stride < n) {
    input[index] += input[index+stride];
  }
}
if (tid==0) { // store the reduced result into first position of this block's index
  input[blockIdx.x] = input[blockIdx.x*blockDim.x];
}
// input[0] -> input[0], input[8] -> input[1], input[16] -> input[2], ...
```

Step1, stride = 1: {0,2,4,6,8,10}

Step2, stride = 2: [0,4,8,12]

Step3, stride = 4: [0,8]

Step4, stride = 8: [0]

---

**[GPU 内核不能直接返回任何值]** GPU 内核不能直接返回任何值，计算结果必须以 cudaMemcpy 形式返回

---

**[1:01:58]** We should work on # elements = # blocks, each block is going to provide only one value
