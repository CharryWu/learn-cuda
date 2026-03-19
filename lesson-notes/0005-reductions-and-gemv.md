# 0005: Parallel Reductions & GEMV

## 1. `__syncthreads()` vs. `__shfl_down_sync()`

* **`__syncthreads()`:** A block-level barrier. Expensive. Forces the hardware's Barrier Control Unit to wait for the slowest thread in a block (up to 1024 threads). Requires Shared Memory to exchange data.
* **`__shfl_down_sync()`:** A warp-level intrinsic. Ultra-fast. Uses the SM's Warp Crossbar to route data directly from one thread's register to another's in near-zero cycles.

---

## 2. Tree-Based Warp Reduction

To find the sum of an array, we use a logarithmic $O(\log N)$ tree reduction rather than a linear loop.

```cpp
__device__ float warpReduceSum(float val) {
    // Fold 32 threads down to 1 using register shuffles
    for (int offset = 16; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val; // Thread 0 now holds the sum of all 32 threads
}
```

---

## 3. The Grid-Stride Loop

To scale a kernel to handle matrices of any size $N$ (e.g., millions of elements) using a fixed number of threads, we stride the grid across the data.

This must happen *before* communication/reduction to maximize memory throughput and minimize shuffle operations.

```cpp
float sum = 0.0f;
int idx = blockIdx.x * blockDim.x + threadIdx.x;
int stride = blockDim.x * gridDim.x;

// Local accumulation in private registers
for (int i = idx; i < n; i += stride) {
    sum += in[i];
}
// ONLY reduce after local accumulation is complete
sum = warpReduceSum(sum);
```

---

## 4. GEMV (Matrix-Vector Multiplication)

Combining the Grid-Stride Loop and Warp Reduction gives us an optimal GEMV kernel, heavily used in LLM inference.

**Strategy (One Warp per Row):**

1. Assign exactly 32 threads (1 warp) to a single row of Matrix $A$.
2. Threads use a Grid-Stride Loop to coalesced-read the row and vector $x$, keeping a running sum.
3. The warp performs a final shuffle reduction to produce the dot product for that row.

```cpp
__global__ void gemv_warp_kernel(const float* A, const float* x, float* y, int M, int N) {
    int row = blockIdx.x * (blockDim.x / 32) + (threadIdx.x / 32);
    if (row >= M) return;

    int lane = threadIdx.x % 32;
    float sum = 0.0f;

    for (int col = lane; col < N; col += 32) {
        sum += A[row * N + col] * x[col]; // Coalesced memory read
    }

    sum = warpReduceSum(sum);

    if (lane == 0) y[row] = sum;
}
```
