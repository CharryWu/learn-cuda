# 0002: CUDA Hardware Constraints & Execution Model

Understanding the difference between the **Logical API** (what your code requests) and the **Physical Hardware** (what the chip actually does) is the foundation of high-performance ML kernels.

## 1. Logical Constraints vs. RTX 3060 Limits

| Level | Constraint | Limit |
| :--- | :--- | :--- |
| **Block** | Max Threads (Total) | 1024 |
| **Block** | Max Z-dimension | 64 |
| **Grid** | Max X-dimension | 2,147,483,647 |
| **Grid** | Max Y/Z-dimension | 65,535 |
| **Hardware** | Streaming Multiprocessors (SMs) | 28 (on RTX 3060) |
| **Hardware** | Max Threads per SM | 1536 |

---

## 2. The SM Sub-Architecture (Ampere)

An SM (Streaming Multiprocessor) is not a single execution unit. On the Ampere architecture (RTX 30 series), each SM is physically divided into **4 Sub-Partitions**.

Each partition is an independent execution engine containing its own private hardware:

* **Warp Scheduler & Dispatch Unit**
* **Register File:** 16K registers per partition (64K total per SM). Warps are physically locked to a partition to stay near their registers.
* **ALUs (Cores):** 16 FP32 Cores, 16 FP32/INT32 Cores, 2 Special Function Units (SFUs).

**Shared Hardware (SM Level):**

* L1 Instruction Cache
* Shared Memory / L1 Data Cache (This allows threads across different partitions within the same block to communicate).
* Tensor Cores

---

## 3. Execution & Latency Hiding

The GPU hardware scheduler treats the Grid as a massive "work queue."

1. **Block Issuance:** The scheduler assigns Blocks to SMs until the SM's Register or Shared Memory limits are hit.
2. **Warp Partitioning:** The SM divides the Block into 32-thread Warps and assigns them to its 4 sub-partitions. All assigned warps are **resident** (kept in physical registers).
3. **Zero-Overhead Context Switching:** If Warp 0 stalls (e.g., waiting 400 cycles for VRAM data), the partition's Warp Scheduler instantly switches to another resident warp (e.g., Warp 4) on the very next clock cycle.

**Senior Insight:** This is why it is better to launch thousands of blocks rather than exactly 28. A deep queue of work ensures the 28 SMs can constantly swap warps to hide memory latency, ensuring the math ALUs are never idle.
