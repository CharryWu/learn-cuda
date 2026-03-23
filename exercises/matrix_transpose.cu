
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdlib.h>
#include <stdio.h>

__global__ void matrixTranspose(const float* in, float* out, const unsigned int M, const unsigned int N)
{
    const unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < N && y < M)
    {
        const unsigned int inIndex = y * N + x;
        const unsigned int outIndex = x * M + y;
        out[y] = in[x];
    }
}

void printMatrix(const char* label, const float* mat, const unsigned int M, const unsigned int N) {
    printf("%s (%dx%d):\n", label, M, N);
    for (int r = 0; r < M; r++) {
        printf("  [");
        for (int c = 0; c < N; c++) {
            printf("%6.1f", mat[r * N + c]);
            if (c < N - 1) printf(",");
        }
        printf(" ]\n");
    }
    printf("\n");
}

void runTranspose(const char* name, const float* host_in, const unsigned int M, const unsigned int N) {
    const size_t in_bytes = M * N * sizeof(float);
    const size_t out_bytes = N * M * sizeof(float);

    float *device_in, *device_out;
    cudaMalloc((void**)&device_in, in_bytes);
    cudaMalloc((void**)&device_out, out_bytes);
    cudaMemcpy((void*)device_in, (const void*)host_in, in_bytes, cudaMemcpyHostToDevice);

    dim3 threads(16, 16); // Common config: 256 threads per block
    dim3 blocks(
        (N + threads.x - 1) / threads.x, // Round up to cover all columns
        (M + threads.y - 1) / threads.y  // Round up to cover all rows
    );

    matrixTranspose<<<blocks, threads>>>(device_in, device_out, M, N);
    cudaDeviceSynchronize();

    float* host_out = (float*)malloc(out_bytes);
    cudaMemcpy(host_out, device_out, out_bytes, cudaMemcpyDeviceToHost);

    printf("=== %s ===\n", name);
    printMatrix("Input ", host_in, M, N);
    printMatrix("Output", host_out, N, M);   // transposed: N rows, M cols

    free(host_out);
    cudaFree(device_in);
    cudaFree(device_out);
}

int main()
{
    // --- Matrix 1: 4x6, sequential 1..24 ---
    // Easy to verify: element [r][c] should appear at [c][r] after transpose.
    const float mat1[4 * 6] = {
         1,  2,  3,  4,  5,  6,
         7,  8,  9, 10, 11, 12,
        13, 14, 15, 16, 17, 18,
        19, 20, 21, 22, 23, 24,
    };
    runTranspose("4x6 sequential", mat1, 4, 6);

    // --- Matrix 2: 6x4, column index as value (highlights how rows/cols swap) ---
    const float mat2[6 * 4] = {
        0, 1, 2, 3,
        0, 1, 2, 3,
        0, 1, 2, 3,
        0, 1, 2, 3,
        0, 1, 2, 3,
        0, 1, 2, 3,
    };
    runTranspose("6x4 column-index pattern", mat2, 6, 4);

    // --- Matrix 3: 5x5 diagonal identity-like, non-square feel from print format ---
    const float mat3[5 * 5] = {
        1, 0, 0, 0, 0,
        0, 2, 0, 0, 0,
        0, 0, 3, 0, 0,
        0, 0, 0, 4, 0,
        0, 0, 0, 0, 5,
    };
    runTranspose("5x5 diagonal (should be unchanged)", mat3, 5, 5);

    return 0;
}
