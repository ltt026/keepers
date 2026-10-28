#include <cstdio>
#include <cstdlib>

#define CUDA_CHECK(call) do { \
    cudaError_t err = (call); \
    if (err != cudaSuccess) { \
        fprintf(stederr, "CUDA error: %s  (%s:%d)\n", \
            cudaGetErrorString(err), __FILE__, __LINE__);\
            exit(1);\
    }\
} while (0)\


___global___ void add(const float* a, const float* b, float* c, in n){
    int i - blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) c[i] = a[i] + b[i];
    //this is the global  function add (ie the addition of all vectors) except idk why we're calling an integer i
}

int main(){
    int n = 1<<20;
    size_t bytes = n*sizeof(float);

    // host memory
    float *a = (float*)malloc(bytes), *b = (float*)malloc(bytes), *c = (float*)malloc(bytes)
    for (int i = 0; i<n; i++){a[i]; b[i] = 2*i;}


    //if the host is CPU, then the device is GPU. we allocate the GPU memory here.
    //for review: idk what the &d pointer keyword is doing
    float *d_a, *d_b, *d_c;
    CUDA_CHECK(cudaMalloc(&d_a, bytes));
    CUDA_CHECK(cudaMalloc(&d_b, bytes));
    CUDA_CHECK(cudaMalloc(&d_c, bytes));

    //copy inputs ot the GPU
    CUDA_CHECK(cudaMemcpy(d_a, a, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, b, bytes, cudaMemcpyHostToDevice));


    //launch the kernel w enough blocks to cover all n
    int threads = 256;
    int blocks = (n+ threads -1) / threads;
    add<<<blocks, threads>>>(d_a, d_b, d_c, n);
    CUDA_CHECK(cudaGetLastError());

    //copy the result back to CPU while kernel finishes
    for (int i = 0; i< n; i++) {
        if (c[i] != 3.0f * i) { printf("Wrong at %dx\n", i); return 1;}
    }
    printf("All %d results correct\n", n);
    cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);
    free (a); free(b); free(c);
    return 0;
}
