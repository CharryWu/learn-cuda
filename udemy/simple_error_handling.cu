#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
 
/**
 * @brief foo is a kernel function that sets the value of the pointer to 7
 * @param ptr the pointer to the value to be set
 */
__global__ void foo(int *ptr) {
  *ptr = 7;
}

int main(void) {
  // allocate memory on the host and the device
  int *d_ptr, *h_ptr;
  h_ptr = (int*)malloc(sizeof(int));
  cudaMalloc((void**)&d_ptr, sizeof(int));

  // launch the kernel
  foo<<<1,1>>>(d_ptr); // async
  cudaDeviceSynchronize();

  // check for error
  cudaError_t error = cudaGetLastError();
  if(error != cudaSuccess)
  {
    // print the CUDA error message and exit
    printf("CUDA error: %s\n", cudaGetErrorString(error));
    exit(-1);
  }

  // copy the result from the device to the host
  cudaMemcpy(h_ptr, d_ptr, sizeof(int), cudaMemcpyDeviceToHost);

  printf("d_ptr: %d\n", *d_ptr);
  // make the host block until the device is finished with foo
  printf("h_ptr: %d\n", *h_ptr);
 

  cudaFree(d_ptr);
  free(h_ptr);
 
  return 0;
}
