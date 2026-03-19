# learn-cuda

A self-study repo for ML infrastructure engineering. Goal: interview-ready CUDA/PyTorch/vLLM skills for ML infra roles at AI startups.

## Repo Structure

```
exercises/      # CUDA kernel exercises (.cu) — hands-on code samples
lesson-notes/   # Progressive study notes (0001–…) from teacher conversations
*.md            # Topic deep-dives (warp divergence, bank conflicts, shared memory, etc.)
*.cu            # Top-level kernel experiments
```

## Learning Curriculum (Phase Order)

1. **Phase 1 — CUDA Foundations**: Hardware execution model, thread/block/grid, memory hierarchy, warp divergence, coalescing, shared memory, reductions
2. **Phase 2 — PyTorch Internals**: Strides, autograd, `torch.utils.cpp_extension` (JIT + `load_inline`)
3. **Phase 3 — Inference Systems**: vLLM, PagedAttention, KV cache, continuous batching
4. **Phase 4 — Distributed Training**: DDP, FSDP, tensor parallelism

## Running CUDA Kernels

`.cu` files are executed remotely via [ferris-compute-cuda](https://github.com/CharryWu/ferris-compute-cuda) — a Rust gRPC client that sends local `.cu` files to a GPU server.

```bash
# Single file
cargo run -p client -- run --server "http://<SERVER_IP>:50051" exercises/<KERNEL>.cu

# Multi-file (first arg = entry point)
cargo run -p client -- run --server "http://<SERVER_IP>:50051" exercises/main.cu exercises/utils.cuh
```

Requires `FERRIS_AUTH_TOKEN` in `.env` or environment. Client lives at `~/ferris-compute-cuda`.

## Conventions

- **Kernel files**: `exercises/` holds course exercises; top-level `.cu` files are standalone experiments
- **Lesson notes**: `lesson-notes/NNNN-topic.md` — numbered sequentially; each covers one concept cluster with code snippets
- **Topic notes**: standalone `.md` files at root for single-topic deep dives (e.g., `warp-divergence.md`)

## When Helping

- New lessons go in `lesson-notes/` with the next sequential number and a descriptive slug
- Benchmark comparisons (naive vs. optimized) are valuable — this repo is also a portfolio
