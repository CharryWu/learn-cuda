#include <cuda_runtime.h>

/**
 * Grid-Stride implementation of `vectorAdd`: Why is this better? (The Interview "Why")
 * When an interviewer asks why you used a loop instead of a simple if (i < n), give them these three reasons:
 * - Flexibility (Decoupling): You can process an array of any size regardless of your grid size. If you have 1 billion elements but only a small GPU, you can launch a small grid and it will still finish (it just takes more "steps").
 * - Memory Coalescing: Because stride is a multiple of the warp size (32), all threads in a warp still access memory in a contiguous block during every iteration. This keeps memory bandwidth high.
 * - Debugging/Portability: You can easily test your kernel with a single block (<<<1, 256>>>) to ensure the math is right before scaling it up to a massive grid.
 */
__global__ void vectorAddGridStride(const float* A, const float* B, float* C, int n) {
    // 1. Get the starting index for this specific thread
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    
    // 2. Get the total number of threads in the entire grid (The Stride)
    int stride = blockDim.x * gridDim.x;

    // 3. The Loop: Thread processes its element, then jumps by 'stride'
    for (int i = index; i < n; i += stride) {
        C[i] = A[i] + B[i];
    }
}

// Driver code from leetgpu.com (compare this to `main` in vector_add_x_only.cu)
// A, B, C are device pointers (i.e. pointers to memory on the GPU)
extern "C" void solve(const float* A, const float* B, float* C, int N) {
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    vectorAddGridStride<<<blocksPerGrid, threadsPerBlock>>>(A, B, C, N);
    cudaDeviceSynchronize();
}
