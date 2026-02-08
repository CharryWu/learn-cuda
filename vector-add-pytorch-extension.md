# The PyTorch C++ Extension

To get that senior offer, you need to show you can integrate these kernels into a Python-first AI stack. We are going to wrap your **Grid-Stride Vector Add** into a PyTorch extension. This is the "glue code" that companies like OpenAI and Anthropic write every day.

You need two files: a **C++ wrapper** and a **Python setup script**.

### **1. The Wrapper (`vector_add_kernel.cu`)**

This file contains your CUDA kernel and a C++ function that checks the tensors (ensuring they are on the GPU and have the right types).

```cpp
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// Your Grid-Stride Kernel
__global__ void vector_add_stride_kernel(const float* a, const float* b, float* c, int n) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = index; i < n; i += stride) {
        c[i] = a[i] + b[i];
    }
}

// The C++ "Glue" function
void vector_add_cuda(torch::Tensor a, torch::Tensor b, torch::Tensor c) {
    const int n = a.size(0);
    
    // Hard-coded for simplicity, but in production, 
    // you'd calculate these based on 'n' and GPU properties
    const int threads = 256;
    const int blocks = (n + threads - 1) / threads;

    // Launch the kernel
    vector_add_stride_kernel<<<blocks, threads>>>(
        a.data_ptr<float>(), 
        b.data_ptr<float>(), 
        c.data_ptr<float>(), 
        n
    );
}

// Bind the C++ function to a Python name
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("add", &vector_add_cuda, "Grid-Stride Vector Add (CUDA)");
}

```

---

### **2. The Build Script (`setup.py`)**

This uses PyTorch's `CUDAExtension` to compile your code into a `.so` (Linux) or `.pyd` (Windows) file that Python can import.

```python
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='my_cuda_ops',
    ext_modules=[
        CUDAExtension('my_cuda_ops', [
            'vector_add_kernel.cu',
        ])
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)

```

---

### **3. How to use it in your Notebook**

After running `pip install .` in your terminal:

```python
import torch
import my_cuda_ops # This is your custom C++ library!

# 1. Create Tensors on the GPU
size = 1000000
a = torch.ones(size, device='cuda')
b = torch.ones(size, device='cuda')
c = torch.empty(size, device='cuda')

# 2. Call your custom CUDA kernel
my_cuda_ops.add(a, b, c)

# 3. Verify
print(c[:5]) # Should be all 2.0s

```

---

**Next Step for You:**
Would you like to try implementing the **Softmax** kernel? It’s significantly harder because it requires **two** passes:

1. Find the `max` of the row (to prevent overflow).
2. Compute `exp(x - max)` and sum them.

This will force you to use **Shared Memory** and **Warp-level Primitives** effectively. Shall we?
