#include <cstdio>

__global__ void hello() {
  printf("Hello from block %d, thread %d\n", blockIdx.x, threadIdx.x);
}

int main(){
  hello <<<2, 4>>>(); //2 blocks of 4 threads = 8 threads.
  cudaDeviceSynchronize(); 
  return 0;
}
