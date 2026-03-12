#include <cuda_runtime.h>

__global__ void matrixTranspose(const float *in, float *out, unsigned int M, unsigned int N) {
    const unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;

    // input has N number of columns on each input row, and y is input row index, x is input col index
    const unsigned int in_idx = y * N + x;
    // output has M number of columns on each output row, and x is output row index, y is output col index
    const unsigned int out_idx = x * M + y;

    if (x < N && y < M) {
        out[out_idx] = in[in_idx];
    }
}

// input, output are device pointers (i.e. pointers to memory on the GPU)
extern "C" void solve(const float* input, float* output, int rows, int cols) {
    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((cols + threadsPerBlock.x - 1) / threadsPerBlock.x,
                       (rows + threadsPerBlock.y - 1) / threadsPerBlock.y);

    matrixTranspose<<<blocksPerGrid, threadsPerBlock>>>(input, output, rows, cols);
    cudaDeviceSynchronize();
}