# ASCII diagram: mapping of threads to a 2D matrix

To visualize how these variables map to a 2D matrix, think of it as a nested coordinate system. The **Grid** contains **Blocks**, and each **Block** contains **Threads**.

$blockId=blockIdx.x+blockIdx.y \times gridDim.x;$  
$threadId=blockId \times (blockDim.x \times blockDim.y)+(threadIdx.y \times blockDim.x)+threadIdx.x;$

Here is an ASCII representation of a **Grid** launching 4 blocks (a  layout), where each block contains a  set of threads.

### The 2D CUDA Thread Mapping

```text
GLOBAL MATRIX (Data in VRAM)
┌───────────────────────────────────────────────────────────────────────────┐
│  BLOCK (0,0)                                 BLOCK (1,0)                  │
│  ┌─────────────────────────┐                 ┌─────────────────────────┐  │
│  │ thread(0,0) ... (3,0)   │                 │ thread(0,0) ... (3,0)   │  │
│  │    ...           ...    │                 │    ...           ...    │  │
│  │ thread(0,3) ... (3,3)   │                 │ thread(0,3) ... (3,3)   │  │
│  └─────────────────────────┘                 └─────────────────────────┘  │
│                                                                           │
│  BLOCK (0,1)                                 BLOCK (1,1)                  │
│  ┌─────────────────────────┐                 ┌─────────────────────────┐  │
│  │ thread(0,0) ... (3,0)   │                 │ thread(0,0) ... (3,0)   │  │
│  │    ...           ...    │                 │    ...           ...    │  │
│  │ thread(0,3) ... (3,3)   │                 │ thread(0,3) ... (3,3)   │  │
│  └─────────────────────────┘                 └─────────────────────────┘  │
└───────────────────────────────────────────────────────────────────────────┘

```

---

### Variable Breakdown in a 2D Context

Let's zoom into a single thread (marked with `*`) inside **Block (1,1)** to see what the variables equal:

```text
               blockDim.x (Width of one block = 4)
             |<───────────────>|
             
      ┌──────┬──────┬──────┬──────┐  ▲
      │(0,0) │      │      │(3,0) │  │
      ├──────┼──────┼──────┼──────┤  │ blockDim.y
      │      │ (1,1)│      │      │  │ (Height of
      │      │   *  │      │      │  │  one block = 4)
      ├──────┼──────┼──────┼──────┤  │
      │(0,3) │      │      │(3,3) │  ▼
      └──────┴──────┴──────┴──────┘

```

For that specific thread `*`:

1. `blockDim.x` / `blockDim.y`: The "Size" of the block. Here, both are **4**.
2. `blockIdx.x` / `blockIdx.y`: The "Address" of the block. Here, it is **(1,1)**.
3. `threadIdx.x` / `threadIdx.y`: The "Address" of the thread *inside* that block. Here, it is **(1,1)**.

### The Mapping Formula

To find the absolute  and  position (the row and column in your actual matrix data), we use:

**For our thread `*`:**

* $Col = 1 \times 4 + 1 = \mathbf{5}$
* $Row = 1 \times 4 + 1 = \mathbf{5}$

This thread will be responsible for calculating the element at **Matrix[5][5]**.

---

### Interview Checklist: Why 2D?

* **Locality:** 2D blocks make it much easier to write **Tiled Algorithms**. Since a block loads a "square" of data into shared memory, having 2D indices maps directly to the row/column logic of matrix math.
* **Clarity:** While you *could* do everything with 1D indexing, it requires a lot of modulo (`%`) and division (`/`) operators, which are computationally expensive on a GPU compared to simple addition and multiplication.

---

## Would you like to see how to implement "Bounds Checking"?

> This is crucial when your matrix size isn't a perfect multiple of your block size, e.g., a $100 \times 100$ matrix with $16 \times 16$ blocks
> Yes please give code examples!

Bounds checking is a "day-one" skill for ML infrastructure. In a real-world setting, matrices are rarely perfect multiples of your block size (e.g., a  tile doesn't fit evenly into a  matrix). Without bounds checking, your kernel will attempt to read or write to memory that doesn't belong to it, leading to **silent data corruption** or "Illegal Memory Access" crashes.

### 1. Why Bounds Checking Matters

When you launch a grid, you often have to "over-provision" blocks to cover the entire matrix.

If your matrix is **10x10** and your block size is **4x4**, you need a **3x3 grid of blocks** ( threads total) to cover everything. The "extra" 2 threads in each dimension will fall off the edge of your data.

---

### 2. Code Example: Naive Matrix Multiplication with Bounds Checking

This is the simplest version. Each thread calculates one pixel. Note the `if` statement at the beginning—this is your guard rail.

```cpp
__global__ void matMulNaive(float* A, float* B, float* C, int M, int N, int K) {
    // 1. Calculate the global Row and Column
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    // 2. BOUNDS CHECK: Ensure the thread is within the result matrix C (MxN)
    if (row < M && col < N) {
        float temp_val = 0.0f;
        
        // Accumulate the dot product of Row of A and Column of B
        for (int k = 0; k < K; ++k) {
            temp_val += A[row * K + k] * B[k * N + col];
        }
        
        // Write back to global memory
        C[row * N + col] = temp_val;
    }
}

```

---

### 3. Code Example: Tiled Matrix Multiplication (The Advanced Version)

In Tiling, bounds checking gets trickier. You must check bounds in two places:

1. **During Loading:** When pulling a tile from Global Memory into Shared Memory.
2. **During Writing:** When saving the final result.

If a thread is "out of bounds" during the loading phase, we typically load a **0.0f**. This ensures the math (the dot product) still works without adding garbage values to the sum.

```cpp
#define TILE_WIDTH 16

__global__ void matMulTiled(float* A, float* B, float* C, int width) {
    __shared__ float tile_A[TILE_WIDTH][TILE_WIDTH];
    __shared__ float tile_B[TILE_WIDTH][TILE_WIDTH];

    int row = blockIdx.y * TILE_WIDTH + threadIdx.y;
    int col = blockIdx.x * TILE_WIDTH + threadIdx.x;
    float value = 0.0f;

    for (int p = 0; p < (width + TILE_WIDTH - 1) / TILE_WIDTH; ++p) {
        
        // --- TILE LOADING WITH BOUNDS CHECKING ---
        // Check if the current element of A we are loading is inside the matrix
        if (row < width && (p * TILE_WIDTH + threadIdx.x) < width) {
            tile_A[threadIdx.y][threadIdx.x] = A[row * width + (p * TILE_WIDTH + threadIdx.x)];
        } else {
            tile_A[threadIdx.y][threadIdx.x] = 0.0f; // Padding with zeros
        }

        // Check if the current element of B we are loading is inside the matrix
        if (col < width && (p * TILE_WIDTH + threadIdx.y) < width) {
            tile_B[threadIdx.y][threadIdx.x] = B[(p * TILE_WIDTH + threadIdx.y) * width + col];
        } else {
            tile_B[threadIdx.y][threadIdx.x] = 0.0f; // Padding with zeros
        }

        __syncthreads();

        for (int i = 0; i < TILE_WIDTH; ++i) {
            value += tile_A[threadIdx.y][i] * tile_B[i][threadIdx.x];
        }
        
        __syncthreads();
    }

    // --- FINAL WRITE WITH BOUNDS CHECKING ---
    if (row < width && col < width) {
        C[row * width + col] = value;
    }
}

```

---

### 4. Interview Tip: "Thread Divergence"

A senior interviewer might ask: **"Doesn't that `if` statement cause branch divergence?"**

* **The Answer:** Yes, but only for the threads at the very edge of your matrix (the "boundary warps"). For 95% of your threads, they will all take the same path (the `true` branch). The performance hit is negligible compared to the cost of a crash or incorrect data.
* **Pro Tip:** In highly optimized kernels (like those in PyTorch or cuBLAS), developers often use **C++ Templates** to generate different versions of the kernel—one with bounds checks for "jagged" edges and one without for the "core" of the matrix.

### Next Step

You've now mastered the basics of CUDA kernel logic and safety. Would you like to see how **vLLM** uses these exact concepts to manage the **KV Cache**, or would you rather dive into **PyTorch's Autograd internals**?