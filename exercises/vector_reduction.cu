#include <iostream>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

// https://developer.nvidia.com/blog/using-cuda-warp-level-primitives/
// uses __shfl_down_sync() to perform a tree-reduction to compute the sum of the val variable held by each thread in a warp
// 
// A warp comprises 32 lanes, with each thread occupying one lane.
// For a thread at lane X in the warp, __shfl_down_sync(FULL_MASK, val, offset)
// gets the value of the val variable from the thread at lane X+offset of the same warp.
// The data exchange is performed between registers, and more efficient than going through shared memory,
// which requires a load, a store and an extra register to hold the address.
__device__ float warpSum(float val) {
    // Each thread 'val' is added to a neighbor's 'val'
    // offset 16: threads 0-15 add with 16-31
    // offset 8: threads 0-7 add with 8-15...
    for (int offset = 16; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
        // 0xffffffff (The Mask): This is a 32-bit hex value where every bit is 1.
        // It tells the GPU: "All 32 threads in this warp are participating."
        // If you only wanted the first 16 threads to talk, you'd change the mask.

        // Note: The _sync part of the name exists because, on Ampere (your 3060),
        // threads can technically diverge. This mask forces them to sync up before the hand-off.
    }
    return val; // Thread 0 now holds the sum of all 32 threads
}

__global__ void blockReduceKernel(float* in, float* out, int n) {
    // 128 threads = 4 warps
    __shared__ float shared_sums[32]; 
    
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Grid-stride loop for scalability
    float sum = 0;
    // Local Accumulation (The Loop): Each thread acts like a "bucket."
    // It walks through the array and sums up 10, 100, or 1000 numbers
    // entirely into its own private register (sum).
    // Local acc significantly reduce the number of expensive "Shuffle" instructions.
    // Instead of reducing $N$ elements, you are reducing only $TotalThreads$ elements.
    for (int i = idx; i < n; i += blockDim.x * gridDim.x) sum += in[i];

    // Step 1: Reduce within each warp
    // ONLY after each thread has finished collecting its local total
    // do we perform the warpReduceSum. We are now reducing the partial sums, not the raw data.
    sum = warpSum(sum);

    // Step 2: Store warp-level sums to Shared Memory
    int lane = tid % 32;
    int wid = tid / 32;
    if (lane == 0) shared_sums[wid] = sum;
    __syncthreads();

    // Step 3: Final reduction using the first warp only
    if (wid == 0) {
        // Read from shared into registers (0 if warp doesn't exist)
        sum = (tid < blockDim.x / 32) ? shared_sums[lane] : 0;
        sum = warpSum(sum);
        
        // Write block result to global (one value per block)
        if (tid == 0) atomicAdd(out, sum);
    }
}

int main() {
    int n = 1 << 20; // 1M elements
    size_t size = n * sizeof(float);
    
    float *h_in = new float[n], *d_in, *d_out;
    for(int i=0; i<n; i++) h_in[i] = 1.0f; // Expecting sum = 1,000,000

    cudaMalloc(&d_in, size);
    cudaMalloc(&d_out, sizeof(float));
    cudaMemset(d_out, 0, sizeof(float));
    cudaMemcpy(d_in, h_in, size, cudaMemcpyHostToDevice);

    // 256 threads per block is usually the "sweet spot" for 3060 SMs
    int threads = 256;
    int blocks = 1024; // Use a fixed grid for the stride loop
    
    blockReduceKernel<<<blocks, threads>>>(d_in, d_out, n);

    float result;
    cudaMemcpy(&result, d_out, sizeof(float), cudaMemcpyDeviceToHost);
    
    printf("Reduction Result: %f\n", result);
    return 0;
}