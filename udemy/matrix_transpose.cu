#include <cuda_runtime.h>
#include <stdio.h>

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

void printMatrix(const char *label, const float *mat, int rows, int cols) {
    printf("%s (%dx%d):\n", label, rows, cols);
    for (int r = 0; r < rows; r++) {
        printf("  [");
        for (int c = 0; c < cols; c++) {
            printf("%6.1f", mat[r * cols + c]);
            if (c < cols - 1) printf(",");
        }
        printf(" ]\n");
    }
    printf("\n");
}

void runTranspose(const char *name, const float *h_in, int M, int N) {
    int in_bytes  = M * N * sizeof(float);
    int out_bytes = N * M * sizeof(float);   // same size, different shape

    float *d_in, *d_out;
    cudaMalloc((void**)&d_in,  in_bytes);
    cudaMalloc((void**)&d_out, out_bytes);
    cudaMemcpy(d_in, h_in, in_bytes, cudaMemcpyHostToDevice);

    dim3 threads(16, 16);
    dim3 blocks((N + threads.x - 1) / threads.x,
                (M + threads.y - 1) / threads.y);
    matrixTranspose<<<blocks, threads>>>(d_in, d_out, M, N);
    cudaDeviceSynchronize();

    float *h_out = (float*)malloc(out_bytes);
    cudaMemcpy(h_out, d_out, out_bytes, cudaMemcpyDeviceToHost);

    printf("=== %s ===\n", name);
    printMatrix("Input ", h_in,  M, N);
    printMatrix("Output", h_out, N, M);   // transposed: N rows, M cols

    free(h_out);
    cudaFree(d_in);
    cudaFree(d_out);
}

int main() {
    // --- Matrix 1: 4x6, sequential 1..24 ---
    // Easy to verify: element [r][c] should appear at [c][r] after transpose.
    const float mat1[4*6] = {
         1,  2,  3,  4,  5,  6,
         7,  8,  9, 10, 11, 12,
        13, 14, 15, 16, 17, 18,
        19, 20, 21, 22, 23, 24,
    };
    runTranspose("4x6 sequential", mat1, 4, 6);

    // --- Matrix 2: 6x4, column index as value (highlights how rows/cols swap) ---
    const float mat2[6*4] = {
        0, 1, 2, 3,
        0, 1, 2, 3,
        0, 1, 2, 3,
        0, 1, 2, 3,
        0, 1, 2, 3,
        0, 1, 2, 3,
    };
    runTranspose("6x4 column-index pattern", mat2, 6, 4);

    // --- Matrix 3: 5x5 diagonal identity-like, non-square feel from print format ---
    const float mat3[5*5] = {
        1, 0, 0, 0, 0,
        0, 2, 0, 0, 0,
        0, 0, 3, 0, 0,
        0, 0, 0, 4, 0,
        0, 0, 0, 0, 5,
    };
    runTranspose("5x5 diagonal (should be unchanged)", mat3, 5, 5);

    return 0;
}