# 0001: Environment Setup & PyTorch C++ Extensions

## 1. The CUDA 13.0 & PyTorch Alignment

When building custom ML infrastructure on the "bleeding edge," aligning your system compiler with PyTorch's internal runtime is critical for stability.

* **System Toolkit (`nvcc`):** Used for building C++/CUDA extensions from source.
* **PyTorch Runtime:** The internal CUDA libraries PyTorch uses to run standard operations.

To install the most compatible PyTorch version for a system running CUDA 13.0:

```powershell
pip install torch torchvision --index-url [https://download.pytorch.org/whl/cu130](https://download.pytorch.org/whl/cu130)
```

## 2. The JIT "Smoke Test"

To verify that your MSVC (`cl.exe`), CUDA (`nvcc`), and Python environments are correctly linked, use PyTorch's Just-In-Time (JIT) compilation. This avoids complex `setup.py` configurations for rapid prototyping.

```python
import torch
from torch.utils.cpp_extension import load_inline

cuda_source = '''
__global__ void hello_cuda_kernel(float* out) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid == 0) {
        printf("Hello from GPU thread 0! Your CUDA setup is working.\\n");
    }
}

void launch_hello(torch::Tensor out) {
    hello_cuda_kernel<<<1, 1>>>(out.data_ptr<float>());
}
'''

cpp_source = "void launch_hello(torch::Tensor out);"

print("Attempting to compile CUDA kernel...")
try:
    hello_module = load_inline(
        name='hello_cuda',
        cpp_sources=[cpp_source],
        cuda_sources=[cuda_source],
        functions=['launch_hello'],
        verbose=True
    )
    
    out = torch.zeros(1, device='cuda')
    hello_module.launch_hello(out)
    torch.cuda.synchronize()
    print("Verification Successful!")
    
except Exception as e:
    print("\n--- COMPILATION FAILED ---")
    print(e)

```

---

## 3. Resolving Windows Build Errors

When moving from JIT to a formal `setup.py` on Windows (using the "x64 Native Tools Command Prompt"), you will often encounter `DISTUTILS_USE_SDK` warnings or Setuptools deprecation errors.

**The Fix:**
Modern ML engineering relies on "Editable Installs" (`pip install -e .`) rather than the deprecated `python setup.py install`. Before building, you must explicitly tell PyTorch that the Visual Studio environment is already active.

**Standard Clean Build Workflow:**

```powershell
# 1. Clean zombie build files
Remove-Item -Recurse -Force build, dist, *.egg-info -ErrorAction SilentlyContinue

# 2. Set environment variables to prevent circular SDK activations
$env:DISTUTILS_USE_SDK=1
$env:FORCE_CUDA=1

# 3. Perform an editable install
pip install -e .

```
