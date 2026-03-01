# 0003: Control Flow & Warp Divergence

The "Golden Rule" of GPU partitions is the SIMT (Single Instruction, Multiple Threads) model: **All 32 threads in a warp must execute the exact same instruction at the exact same clock cycle.**

## 1. The Mechanics of Divergence

Warp divergence occurs when an `if-else` statement causes threads *within the same warp* to take different execution paths.

The hardware handles this through **Serial Execution via Masking**:

1. The scheduler masks out (disables) the `else` threads and runs the `if` path.
2. The scheduler then masks out the `if` threads and runs the `else` path.
**Result:** Throughput for that code block drops by 50%.

---

## 2. Safe vs. Unsafe Branching

Not all `if` statements cause divergence. It only happens if threads *within the same 32-thread group* disagree.

* **No Divergence (Safe):** `if (blockIdx.x > 5)`
  All 32 threads in any given warp belong to the same block. Therefore, every thread in the warp evaluates the condition identically. They execute in lockstep at full speed.
* **Divergence (Unsafe):** `if (threadIdx.x % 2 == 0)`
  Inside a warp, even threads evaluate `True` and odd threads evaluate `False`. The warp splits, and execution is serialized.

---

## 3. Branchless Code Optimization

To achieve uniformity and keep all ALUs in the partition firing at once, we replace conditional branches with mathematical equivalents (like intrinsics or ternary operators).

**Divergent (Bad):**

```cpp
if (val < 0.0f) {
    val = 0.0f; 
} else {
    val = val * 2.0f; 
}
```

**Branchless (Senior/Optimal):**

```cpp
// fmaxf is a single hardware instruction on Ampere.
// No branch to resolve; the Warp Scheduler issues one instruction for all 32 threads.
val = fmaxf(val, 0.0f) * (1.0f + (float)(val >= 0.0f));
```
