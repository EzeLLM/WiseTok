#include <stdio.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(err)); \
            exit(1); \
        } \
    } while(0)

#define BLOCK_SIZE 16

__global__ void matmul_kernel(const float* A, const float* B, float* C, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < N && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < N; ++k) {
            sum += A[row * N + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}

__global__ void matmul_shared_kernel(const float* A, const float* B, float* C, int N) {
    __shared__ float sA[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float sB[BLOCK_SIZE][BLOCK_SIZE];

    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int row = by * BLOCK_SIZE + ty;
    int col = bx * BLOCK_SIZE + tx;

    float sum = 0.0f;

    for (int k = 0; k < (N + BLOCK_SIZE - 1) / BLOCK_SIZE; ++k) {
        if (row < N && (k * BLOCK_SIZE + tx) < N) {
            sA[ty][tx] = A[row * N + k * BLOCK_SIZE + tx];
        } else {
            sA[ty][tx] = 0.0f;
        }

        if ((k * BLOCK_SIZE + ty) < N && col < N) {
            sB[ty][tx] = B[(k * BLOCK_SIZE + ty) * N + col];
        } else {
            sB[ty][tx] = 0.0f;
        }

        __syncthreads();

        for (int i = 0; i < BLOCK_SIZE; ++i) {
            sum += sA[ty][i] * sB[i][tx];
        }

        __syncthreads();
    }

    if (row < N && col < N) {
        C[row * N + col] = sum;
    }
}

__global__ void reduce_kernel(const float* input, float* output, int N) {
    extern __shared__ float sdata[];

    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

    sdata[tid] = (i < N) ? input[i] : 0.0f;
    __syncthreads();

    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        output[blockIdx.x] = sdata[0];
    }
}

__global__ void vector_add_kernel(const float* A, const float* B, float* C, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        C[idx] = A[idx] + B[idx];
    }
}

class CudaMatrix {
private:
    float* d_data;
    int rows, cols;

public:
    CudaMatrix(int r, int c) : rows(r), cols(c) {
        CUDA_CHECK(cudaMalloc(&d_data, rows * cols * sizeof(float)));
    }

    ~CudaMatrix() {
        if (d_data) {
            CUDA_CHECK(cudaFree(d_data));
        }
    }

    void copyFromHost(const float* h_data) {
        CUDA_CHECK(cudaMemcpy(d_data, h_data, rows * cols * sizeof(float),
                             cudaMemcpyHostToDevice));
    }

    void copyToHost(float* h_data) {
        CUDA_CHECK(cudaMemcpy(h_data, d_data, rows * cols * sizeof(float),
                             cudaMemcpyDeviceToHost));
    }

    float* getData() const { return d_data; }
    int getRows() const { return rows; }
    int getCols() const { return cols; }

    void multiplyShared(const CudaMatrix& B, CudaMatrix& C) {
        dim3 blockDim(BLOCK_SIZE, BLOCK_SIZE);
        dim3 gridDim((B.getCols() + BLOCK_SIZE - 1) / BLOCK_SIZE,
                    (rows + BLOCK_SIZE - 1) / BLOCK_SIZE);
        matmul_shared_kernel<<<gridDim, blockDim>>>(d_data, B.getData(), C.getData(), rows);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
    }
};

void runMatrixMultiplication() {
    const int N = 1024;
    float* h_A = new float[N * N];
    float* h_B = new float[N * N];
    float* h_C = new float[N * N];

    for (int i = 0; i < N * N; ++i) {
        h_A[i] = 1.0f;
        h_B[i] = 2.0f;
    }

    CudaMatrix dA(N, N), dB(N, N), dC(N, N);
    dA.copyFromHost(h_A);
    dB.copyFromHost(h_B);

    dim3 blockDim(BLOCK_SIZE, BLOCK_SIZE);
    dim3 gridDim((N + BLOCK_SIZE - 1) / BLOCK_SIZE,
                (N + BLOCK_SIZE - 1) / BLOCK_SIZE);

    matmul_shared_kernel<<<gridDim, blockDim>>>(dA.getData(), dB.getData(),
                                               dC.getData(), N);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    dC.copyToHost(h_C);

    printf("Result C[0][0] = %f (expected %f)\n", h_C[0], (float)(N * 2.0f));

    delete[] h_A;
    delete[] h_B;
    delete[] h_C;
}

int main() {
    int device_count;
    CUDA_CHECK(cudaGetDeviceCount(&device_count));
    printf("Found %d CUDA device(s)\n", device_count);

    if (device_count > 0) {
        CUDA_CHECK(cudaSetDevice(0));
        cudaDeviceProp prop;
        CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
        printf("Using device: %s\n", prop.name);

        runMatrixMultiplication();
    }

    return 0;
}
