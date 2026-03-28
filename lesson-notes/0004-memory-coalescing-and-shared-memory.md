# 0004: Memory Coalescing & Shared Memory Banks

## 1. Global Memory Coalescing

The GPU's memory controller fetches data from Global VRAM in **128-byte segments**.

* **Coalesced Access (Optimal):** When 32 threads in a warp request 32 consecutive 4-byte `float` values, they fall perfectly into a 128-byte window. The GPU makes **one trip** to VRAM.
* **Uncoalesced Access (Disaster):** When threads request scattered memory locations (e.g., reading a matrix column by column), the GPU must make up to 32 separate trips to VRAM to fulfill the warp's request.

**Standard Coalescing Pattern:**

```cpp
int idx = blockIdx.x * blockDim.x + threadIdx.x; // Guarantees adjacent threads read adjacent memory
```

---

## 2. Shared Memory Bank Conflicts

Shared memory is physically divided into **32 independent memory modules** called **Banks**.

* Each bank can service exactly one 32-bit request per clock cycle.
* **Conflict:** If multiple threads in a warp access *different* addresses that fall into the *same* bank simultaneously, the hardware serializes the requests.

**Address to Bank Mapping:**
$$\text{Bank} = (\text{Address} / 4 \text{ bytes}) \pmod{32}$$

---

## 3. Resolving Conflicts (Matrix Transpose Example)

When reading a matrix by column from Shared Memory, every thread hits the same bank, causing a massive 32-way conflict.

### Method A: Padding (Industry Standard)

Add a dummy column to shift the bank alignment of every row.

```cpp
__shared__ float tile[32][33]; // The extra column staggers the memory layout
```

### Method B: Swizzling / XOR (The "Pro" Choice)

Permute the column index based on the row index using a bitwise XOR ($\oplus$). This uses 0% extra memory and perfectly distributes column reads across all banks.
$$\text{Swizzled\_Col} = \text{col} \oplus \text{row}$$

```cpp
// Swizzled Write to Shared Memory
int swizzle_x = threadIdx.x ^ threadIdx.y;
tile[threadIdx.y][swizzle_x] = global_in[...];

// Swizzled Read from Shared Memory
int swizzle_y = threadIdx.y ^ threadIdx.x;
global_out[...] = tile[threadIdx.x][swizzle_y];
```

https://leimao.github.io/blog/CUDA-Shared-Memory-Swizzling/
