# 0006: Softmax & Numerical Stability

## 1. The "Leaking Bucket" Problem (Overflow)

The standard Softmax formula is $P_i = \frac{e^{x_i}}{\sum e^{x_j}}$.
In `float32`, if $x_i = 1000$, $e^{1000}$ results in $\infty$ (overflow), breaking the entire neural network.

**The Safe Softmax Fix:**
We subtract the maximum value of the row ($M$) from every element before calculating the exponential:
$$P_i = \frac{e^{x_i - M}}{\sum e^{x_j - M}}$$
This mathematically identical formulation ensures the largest value becomes $e^0 = 1$, and all other values are $\le 1$, completely eliminating the risk of overflow.

---

## 2. The 3-Pass Block-Level Implementation

A standard safe softmax requires three distinct passes over the data. We assign **One Thread Block per Row** for optimal SM occupancy.

1. **Pass 1 (Max):** Find the row maximum to prevent overflow.
2. **Pass 2 (Sum):** Compute the sum of the exponentials ($e^{x - M}$).
3. **Pass 3 (Normalize):** Divide each exponential by the sum and write to global memory.

```cpp
__global__ void safe_softmax_kernel(float* input, float* output, int width) {
    int row = blockIdx.x;
    int tid = threadIdx.x;

    // 1. FIND MAX (Pass 1)
    float local_max = -FLT_MAX;
    for (int i = tid; i < width; i += blockDim.x) {
        local_max = fmaxf(local_max, input[row * width + i]);
    }
    float row_max = blockReduceMax(local_max); // Uses warp shuffles internally
    __shared__ float shared_row_max;
    if (tid == 0) shared_row_max = row_max;
    __syncthreads();

    // 2. COMPUTE SUM OF EXPS (Pass 2)
    float local_sum = 0.0f;
    for (int i = tid; i < width; i += blockDim.x) {
        local_sum += expf(input[row * width + i] - shared_row_max);
    }
    
    float row_sum = blockReduceSum(local_sum);
    __shared__ float shared_row_sum;
    if (tid == 0) shared_row_sum = row_sum;
    __syncthreads();

    // 3. NORMALIZE AND WRITE (Pass 3)
    for (int i = tid; i < width; i += blockDim.x) {
        int idx = row * width + i;
        output[idx] = expf(input[idx] - shared_row_max) / shared_row_sum;
    }
}
```

*Note: In production (like vLLM), this is often optimized into a 2-pass "Online Softmax" by maintaining a running corrected sum, but the 3-pass is the standard interview baseline.*
