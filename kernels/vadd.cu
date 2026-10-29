```cuda-cpp
#include <cstdio>
#include <cstdlib>

#define CUDA_CHECK(call) do {                                 \
    cudaError_t err = (call);                                 \
    if (err != cudaSuccess) {                                 \
        fprintf(stderr, "CUDA error: %s (%s:%d)\n",           \
                cudaGetErrorString(err), __FILE__, __LINE__); \
        exit(1);                                              \
    }                                                         \
} while (0)

__global__ void add(const float* a, const float* b, float* c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) c[i] = a[i] + b[i];      // the grid may have extra threads; they do nothing
}

int main() {
    int n = 1 << 20;                    // 1,048,576 #s
    size_t bytes = n * sizeof(float);

    // 1. assign Host (CPU) memory
    float *a = (float*)malloc(bytes), *b = (float*)malloc(bytes), *c = (float*)malloc(bytes);
    for (int i = 0; i < n; i++) { a[i] = i; b[i] = 2 * i; }

    // 2. Device (GPU) memory
    float *d_a, *d_b, *d_c;
    CUDA_CHECK(cudaMalloc(&d_a, bytes));
    CUDA_CHECK(cudaMalloc(&d_b, bytes));
    CUDA_CHECK(cudaMalloc(&d_c, bytes));

    // 3. Copy inputs to the GPU
    CUDA_CHECK(cudaMemcpy(d_a, a, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, b, bytes, cudaMemcpyHostToDevice));

    // 4. Launch enough blocks to cover n (round up), timed with GPU events
    int threads = 256;
    int blocks = (n + threads - 1) / threads;

    cudaEvent_t start, stop; // here we finna time the kernel. erm.
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    CUDA_CHECK(cudaEventRecord(start));
    add<<<blocks, threads>>>(d_a, d_b, d_c, n);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float ms = 0;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    printf("Kernel took %.3f ms\n", ms);

    // 5. Copy the result back (this waits for the kernel to finish)
    CUDA_CHECK(cudaMemcpy(c, d_c, bytes, cudaMemcpyDeviceToHost));

    for (int i = 0; i < n; i++) {
        if (c[i] != 3.0f * i) { printf("Wrong at %d\n", i); return 1; }
    }
    printf("All %d results correct\n", n);

    // Cleanup
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaFree(d_a)); CUDA_CHECK(cudaFree(d_b)); CUDA_CHECK(cudaFree(d_c));
    free(a); free(b); free(c);
    return 0;
}
```