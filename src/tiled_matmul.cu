#include "testing.h"

#include <math.h>
#include <stdio.h>
#include <iostream>

// // CUDA kernel for vector addition
// __global__
// void vec_add_kernel(float* a, float* b, float* c, int n) {
//     int i = blockDim.x * blockIdx.x + threadIdx.x;
//     if (i < n) {
//         c[i] = a[i] + b[i];
//     }
// }

// // CUDA vector addition
// void vec_add(float* a_h, float* b_h, float* c_h, int n) {
//     int size = n * sizeof(float);
//     float *a_d, *b_d, *c_d;

//     cudaMalloc((void**) &a_d, size);
//     cudaMalloc((void**) &b_d, size);
//     cudaMalloc((void**) &c_d, size);

//     cudaMemcpy(a_d, a_h, size, cudaMemcpyHostToDevice);
//     cudaMemcpy(b_d, b_h, size, cudaMemcpyHostToDevice);

//     // Computation
//     dim3 dim_grid(ceil(n / 256.0), 1, 1);
//     dim3 dim_block(256, 1, 1);
//     vec_add_kernel<<<dim_grid, dim_block>>>(a_d, b_d, c_d, n);

//     cudaMemcpy(c_h, c_d, size, cudaMemcpyDeviceToHost);

//     cudaFree(a_d);
//     cudaFree(b_d);
//     cudaFree(c_d);
// }

// __global__
// void convert_to_gray_kernel(unsigned char* p_in, unsigned char* p_out, int width, int height) {
//     int col = blockIdx.x * blockDim.x + threadIdx.x;
//     int row = blockIdx.y * blockDim.y + threadIdx.y;

//     if (col < width && row < height) {
//         int gray_offset = row * width + col;

//         int rgb_offset = gray_offset * 3; // 3 bytes needed to store RGB data (could increase if there was more channels)
//         unsigned char r = p_in[rgb_offset];
//         unsigned char g = p_in[rgb_offset + 1];
//         unsigned char b = p_in[rgb_offset + 2];

//         p_out[gray_offset] = 0.21f*r + 0.72f*g + 0.07f*b;
//     }
// }

// #define BLUR_SIZE = 1

// __global__
// void blur_kernel(unsigned char* p_in, unsigned char* p_out, int width, int height) {
//     int col = blockIdx.x * blockDim.x + threadIdx.x;
//     int row = blockIdx.y * blockDim.y + threadIdx.y;

//     if (col < width && row < height) {
//         int pix_vals = 0;
//         int pixels = 0;

//         for (int blur_row = BLUR_SIZE - 1; blur_row <= BLUR_SIZE; ++blur_row) {
//             cur_row = row + blur_row;
//             for (int blur_col = BLUR_SIZE - 1; blur_col <= BLUR_SIZE; ++blur_col) {
//                 cur_col = col + blur_col;

//                 if (cur_row >= 0 && cur_row < height && cur_col >= 0 && cur_col < width) {
//                     pix_vals += p_in[cur_row * width + cur_col];
//                     ++pixels;
//                 }
//             }
//         }

//         p_out[row * width + col] = (unsigned char) (pix_vals / pixels);
//     }
// }

// // Only square matrices
// __global__
// void matmul_kernel(float* a, float* b, float* p, int width) {
//     int col = blockIdx.x * blockDim.x + threadIdx.x;
//     int row = blockIdx.y * blockDim.y + threadIdx.y;

//     if (row < width && col < width) {
//         float p_val = 0;
//         for (int i = 0; i < width; ++i) {
//             p_val += a[row * width + i] * b[i * width + col];
//         }

//         p[row * width + col] = p_val;
//     }
// }

// // Matrix multiplication with one row per thread
// __global__
// void matmul_row_kernel(float* a, float* b, float* p, int width) {
//     int row = blockIdx.y * blockDim.y + threadIdx.y;

//     if (row < width) {
//         for (int i = 0; i < width; ++i) {
//             float p_value = 0;
//             for (int k = 0; k < width; ++k) {
//                 p_value += a[row * width + k] * b[k * width + i];
//             }
//             p[row * width + i] = p_value;
//         }
//     }
// }

// // Matrix multiplication with one column per thread
// __global__
// void matmul_col_kernel(float* a, float* b, float* p, int width) {
//     int col = blockIdx.x * blockDim.x + threadIdx.x;

//     if (col < width) {
//         for (int row = 0; row < width; ++row) {
//             float p_value = 0;
//             for (int k = 0; k < width; ++k) {
//                 p_value += a[row * width + k] * b[k * width + col];
//             }
//             p[row * width + col] = p_value;
//         }
//     }
// }

// void cuda_matmul(float* a_h, float* b_h, float* p_h, int width) {
//     int size = (width * width) * sizeof(float);
//     float *a_d, *b_d, *p_d;

//     // Allocate space on device and copy over data
//     cudaMalloc((void**) &a_d, size);
//     cudaMalloc((void**) &b_d, size);
//     cudaMalloc((void**) &p_d, size);

//     cudaMemcpy(a_d, a_h, size, cudaMemcpyHostToDevice);
//     cudaMemcpy(b_d, b_h, size, cudaMemcpyHostToDevice);

//     // Compute matrix multiplication and copy result to host
//     dim3 grid_dim(1, 1, 1);
//     dim3 block_dim(width, 1, 1);
//     matmul_col_kernel<<<grid_dim, block_dim>>>(a_d, b_d, p_d, width);

//     cudaMemcpy(p_h, p_d, size, cudaMemcpyDeviceToHost);

//     // Free memory on device
//     cudaFree(a_d);
//     cudaFree(b_d);
//     cudaFree(p_d);
// }

// __global__
// void matrix_vector_mult_kernel(float* mat, float* vec, float* res, int width) {
//     int row = blockIdx.y * blockDim.y + threadIdx.y;

//     float out_value = 0;
//     for (int i = 0; i < width; ++i) {
//         out_value += vec[i] * mat[row * width + i];
//     }

//     res[row] = out_value;
// }

// void cuda_matvec_mul(float* mat_h, float* vec_h, float* res_h, int width) {
//     int size = width * sizeof(float);
//     float *mat_d, *vec_d, *res_d;

//     // Assign space in device memory and copy over data
//     cudaMalloc((void**) &mat_d, size * size);
//     cudaMalloc((void**) &vec_d, size);
//     cudaMalloc((void**) &res_d, size);

//     cudaMemcpy(mat_d, mat_h, size * size, cudaMemcpyHostToDevice);
//     cudaMemcpy(vec_d, vec_h, size, cudaMemcpyHostToDevice);

//     // Compute the result and copy back to the host
//     dim3 grid_dim(1, 1, 1);
//     dim3 block_dim(1, width, 1);
//     matrix_vector_mult_kernel<<<grid_dim, block_dim>>>(mat_d, vec_d, res_d, width);

//     cudaMemcpy(res_h, res_d, size, cudaMemcpyDeviceToHost);

//     // Free device memmory
//     cudaFree(mat_d);
//     cudaFree(vec_d);
//     cudaFree(res_d);
// }

#define TILE_WIDTH 2
__global__ void tiled_matmul_kernel(float* a, float* b, float* res, int j, int k, int l) {
    // Initialize the arrays for tiling purposes in shared block memory
    __shared__ float a_tile [TILE_WIDTH][TILE_WIDTH];
    __shared__ float b_tile [TILE_WIDTH][TILE_WIDTH];

    // Identify the row/column of the result matrix to calculate in this thread
    int col = blockIdx.x * TILE_WIDTH + threadIdx.x;
    int row = blockIdx.y * TILE_WIDTH + threadIdx.y;

    float p_value = 0.0f;

    for (int phase = 0; phase < ceil(static_cast<float>(k) / TILE_WIDTH); ++phase) {
        // Load values into shared memory
        if (row < j && (phase * TILE_WIDTH + threadIdx.x) < j) {
            a_tile[threadIdx.y][threadIdx.x] = a[row * k + phase * TILE_WIDTH + threadIdx.x];
        } else {
            a_tile[threadIdx.y][threadIdx.x] = 0.0f;
        }

        if ((phase * TILE_WIDTH + threadIdx.y) < l && col < l) {
            b_tile[threadIdx.y][threadIdx.x] = b[(phase * TILE_WIDTH + threadIdx.y) * l + col];
        } else {
            b_tile[threadIdx.y][threadIdx.x] = 0.0f;
        }

        __syncthreads();

        // Compute the partial p_value
        for (int i = 0; i < TILE_WIDTH; ++i) {
            p_value += a_tile[threadIdx.y][i] * b_tile[i][threadIdx.x];
        }

        __syncthreads();
    }

    if (row < j && col < l) {
        res[row * l + col] = p_value;
    }
}

void cuda_matmul_tile(float* a_h, float* b_h, float* res_h, int j, int k, int l, float* time_to_execute) {
    float *a_d, *b_d, *res_d;

    cudaMalloc((void**) &a_d, j * k * sizeof(float));
    cudaMalloc((void**) &b_d, k * l * sizeof(float));
    cudaMalloc((void**) &res_d, j * l * sizeof(float));

    cudaMemcpy(a_d, a_h, j * k * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(b_d, b_h, k * l * sizeof(float), cudaMemcpyHostToDevice);

    
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    dim3 grid_dim(4, 4, 1);
    dim3 block_dim(2, 2, 1);

    cudaEventRecord(start);
    tiled_matmul_kernel<<<grid_dim, block_dim>>>(a_d, b_d, res_d, j, k, l);
    cudaEventRecord(stop);
  
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    *time_to_execute = milliseconds;

    // std::cout << "Kernel execution took " << milliseconds << " milliseconds" << '\n';

    cudaMemcpy(res_h, res_d, j * l * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(a_d);
    cudaFree(b_d);
    cudaFree(res_d);
}

int main() {
    std::vector<int> matrix_dims = {
        32, 64, 128, 256, 512, 1024, // Dims to test size with powers of two
        200, 400, 600, 800, 1000, 1200 // Dims to test generalization
    };

    // Number of times to run each configuration
    int num_runs = 10;

    std::cout << "--- Benchmarking Tiled Normal Matmul --- \n";

    for (int dim : matrix_dims) {
        int i = dim;
        int j = dim;
        int k = dim;

        float total_time = 0.0;

        for (int run = 0; run < num_runs; run++) {
            std::vector<float> mat1 = generate_random_matrix(i, j);
            std::vector<float> mat2 = generate_random_matrix(j, k);
            float res[dim * dim];

            float time_to_execute;
            cuda_matmul_tile(mat1.data(), mat2.data(), res, i, j, k, &time_to_execute);

            total_time += time_to_execute;
        }

        double average_time = total_time / num_runs;

        std::cout << "Dim: " << dim << "\n Average runtime: " << average_time << " milliseconds\n\n";
    }
    
    // float a[4] = {5.47, 3.08, 1.0, 1.5};
    // float b[6] = {5.47, 3.08, 1.5, -7.59, 1, 2.5};
    // float p[6];
    
    // cuda_matmul_tile(a, b, p, 2, 2, 3);

    // for (int i = 0; i < 6; ++i) {
    //     printf("%.2f\n", p[i]);
    // }

    return 0;
}
