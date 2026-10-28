
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
    // erm we finna add gpu timers. 
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    add<<<blocks, threads>>>(d_a, d_b, d_c, n);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float ms = 0;
    cudaEventElaspedTime(&ms, start, stop);
    printf("Kernel took .%.3f ms \n", ms);







    int n = 1 << 20;                    // 1,048,576 numbers hopefully
    size_t bytes = n * sizeof(float);

    // 1. Host (CPU) memory
    float *a = (float*)malloc(bytes), *b = (float*)malloc(bytes), *c = (float*)malloc(bytes);
    for (int i = 0; i < n; i++) { a[i] = i; b[i] = 2 * i; }

    // 2. Device (GPU) memory
    float *d_a, *d_b, *d_c;
    CUDA_CHECK(cudaMalloc(&d_a, bytes));
    CUDA_CHECK(cudaMalloc(&d_b, bytes));
    CUDA_CHECK(cudaMalloc(&d_c, bytes));

    // 3. Copy inputs to gpu device
    CUDA_CHECK(cudaMemcpy(d_a, a, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, b, bytes, cudaMemcpyHostToDevice));

    // 4. Launch enough blocks to cover n (round up)
    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    add<<<blocks, threads>>>(d_a, d_b, d_c, n);
    CUDA_CHECK(cudaGetLastError());

    // 5. Copy the result back (we finna wait fo the kernel to finish cuz time complexity)
    CUDA_CHECK(cudaMemcpy(c, d_c, bytes, cudaMemcpyDeviceToHost));

    for (int i = 0; i < n; i++) {
        if (c[i] != 3.0f * i) { printf("Wrong at %d\n", i); return 1; }
    }
    printf("All %d results correct\n", n);

    cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);
    free(a); free(b); free(c);
    return 0;
}