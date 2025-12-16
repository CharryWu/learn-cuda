#include <cmath>
#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#define N 10000

/**
 * @brief vectorAdd is a kernel function that adds two vectors
 * @param a the first vector
 * @param b the second vector
 * @param c the result vector
 * @param n the length of the vectors
 */
__global__ void vectorAdd(int *a, int *b, int *c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        c[i] = a[i] + b[i];
    }
}

int main(void) {
    int *a = (int*)malloc(N * sizeof(int));
    int *b = (int*)malloc(N * sizeof(int));
    int *c = (int*)malloc(N * sizeof(int));

    for (int i = 0; i < N; i++) {
        a[i] = i;
        b[i] = i;
    }
    
    int *d_a, *d_b, *d_c;
    cudaMalloc((void**)&d_a, N * sizeof(int));
    cudaMalloc((void**)&d_b, N * sizeof(int));
    cudaMalloc((void**)&d_c, N * sizeof(int));

    cudaMemcpy(d_a, a, N * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, N * sizeof(int), cudaMemcpyHostToDevice);
    
    // IMPORTANT: we need to set correct number of threads and blocks to avoid warps divergence
    // If this is not set correctly, the kernel will still compile but not run correctly (output sum will be all 0)
    int thr_per_blk = 256;
    int blk_in_grid = ceil( float(N) / thr_per_blk );
    vectorAdd<<<blk_in_grid, thr_per_blk>>>(d_a, d_b, d_c, N);

    cudaError_t error = cudaGetLastError();
    if(error != cudaSuccess)
    {
        printf("CUDA error: %s\n", cudaGetErrorString(error));
        exit(-1);
    }

    cudaMemcpy(c, d_c, N * sizeof(int), cudaMemcpyDeviceToHost);
    
    for (int i = 0; i < N; i++) {
        printf("%d + %d = %d\n", a[i], b[i], c[i]);
    }
    
    free(a);
    free(b);
    free(c);
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    return 0;
}