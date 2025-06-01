#include "quantization.h"
#include "testing.h"

#include <vector>
#include <cstdint>
#include <cstring>
#include <math.h>

#include <iostream>


void print_gpu_memory_info() {
    size_t free_mem, total_mem;
    cudaMemGetInfo(&free_mem, &total_mem);

    std::cout << "Used GPU Memory: " << total_mem - free_mem << "\n\n";
}

#define TILE_WIDTH 2

__global__
void quantized_mult_kernel(int8_t* mat1, int8_t* mat2, float* res, int i, int j, int k, size_t tile_width, float scale_mat1, float scale_mat2) {
    // Initialize the arrays for tiling purposes in shared block memory
    extern __shared__ int8_t mat1_mat2_tiles[];

    int8_t* mat1_tile = (int8_t *) mat1_mat2_tiles;
    int8_t* mat2_tile = (int8_t *) mat1_mat2_tiles + tile_width * tile_width;

    int col = blockIdx.x * tile_width + threadIdx.x;
    int row = blockIdx.y * tile_width + threadIdx.y;

    int32_t p_value = 0;

    // Iterate through each phase of the tiling process
    for (int phase = 0; phase < ceilf(static_cast<float>(j) / tile_width); ++phase) {
        // Load values into shared memory
        int thread_index = threadIdx.y * tile_width + threadIdx.x;
        if (row < i && (phase * tile_width + threadIdx.x) < j) {
            mat1_tile[thread_index] = mat1[row * j + phase * tile_width + threadIdx.x];
        } else {
            mat1_tile[thread_index] = 0;
        }

        if (col < k && (phase * tile_width + threadIdx.y) < j) {
            mat2_tile[thread_index] = mat2[(phase * tile_width + threadIdx.y) * k + col];
        } else {
            mat2_tile[thread_index] = 0;
        }

        __syncthreads();

        // Accumulate p_value for tiled values
        for (int p = 0; p < tile_width; ++p) {
            p_value += static_cast<int32_t>(mat1_tile[threadIdx.y * tile_width + p])
                     * static_cast<int32_t>(mat2_tile[p * tile_width + threadIdx.x]);
        }

        __syncthreads();
    }

    if (row < i && col < k) {
        res[row * k + col] = p_value / (scale_mat1 * scale_mat2);
    }
}

std::vector<float> quantize_multiply_cuda(const std::vector<float>& mat1, const std::vector<float>& mat2, int i, int j, int k, float* time_to_execute) {
    float* res = new float[i * k];

    std::pair<std::vector<int8_t>, float> quantized_result_mat1 = symmetric_quantize_int8(mat1);
    std::pair<std::vector<int8_t>, float> quantized_result_mat2 = symmetric_quantize_int8(mat2);

    int8_t* quantized_data_mat1 = quantized_result_mat1.first.data();
    float quantized_scale_mat1 = quantized_result_mat1.second;
    
    int8_t* quantized_data_mat2 = quantized_result_mat2.first.data();
    float quantized_scale_mat2 = quantized_result_mat2.second;

    int8_t *mat1_d, *mat2_d;
    float *res_d;

    cudaMalloc((void**) &mat1_d, i * j * sizeof(int8_t));
    cudaMalloc((void**) &mat2_d, j * k * sizeof(int8_t));
    cudaMalloc((void**) &res_d, i * k * sizeof(float));

    cudaMemcpy(mat1_d, quantized_data_mat1, i * j * sizeof(int8_t), cudaMemcpyHostToDevice);
    cudaMemcpy(mat2_d, quantized_data_mat2, j * k * sizeof(int8_t), cudaMemcpyHostToDevice);

    int dev_count;
    cudaGetDeviceCount(&dev_count);
    cudaDeviceProp dev_prop;
    size_t max_block_mem = 0;
    
    size_t tile_width = TILE_WIDTH;
    size_t needed_space = 2 * tile_width * tile_width * sizeof(int8_t);

    for (int i = 0; i < dev_count; ++i) {
        cudaGetDeviceProperties(&dev_prop, i);
        size_t device_block_mem = dev_prop.sharedMemPerBlock;

        if (device_block_mem > max_block_mem) {
            max_block_mem = device_block_mem;
        }
    }

    while (max_block_mem < needed_space) {
        tile_width /= 2;
        needed_space = 2 * tile_width * tile_width * sizeof(int8_t);
    }

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Execute the kernel
    dim3 dim_grid((k + tile_width - 1) / tile_width, (i + tile_width - 1) / tile_width, 1);
    dim3 dim_block(tile_width, tile_width,  1);

    cudaEventRecord(start);
    quantized_mult_kernel<<<dim_grid, dim_block, needed_space>>>(mat1_d, mat2_d, res_d, i, j, k, tile_width, quantized_scale_mat1, quantized_scale_mat2);
    cudaEventRecord(stop);

    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    *time_to_execute = milliseconds;

    // std::cout << "Kernel execution took " << milliseconds << " milliseconds" << '\n';

    cudaDeviceSynchronize(); // Wait for the kernel to complete
    cudaError_t err = cudaGetLastError(); // Check for any errors that occurred during kernel execution
    if (err != cudaSuccess) {
        std::cerr << "CUDA error after kernel launch: " << cudaGetErrorString(err) << std::endl;
    }

    cudaMemcpy(res, res_d, i * k * sizeof(float), cudaMemcpyDeviceToHost);

    std::vector<float> res_vec(res, res + i * k);

    cudaFree(mat1_d);
    cudaFree(mat2_d);
    cudaFree(res_d);

    delete[] res;

    return res_vec;
}

int main() {
    std::vector<int> matrix_dims = {
        32, 64, 128, 256, 512, 1024, // Dims to test size with powers of two
        200, 400, 600, 800, 1000, 1200 // Dims to test generalization
    };

    // Number of times to run each configuration
    int num_runs = 10;

    std::cout << "--- Benchmarking Tiled Quantized Matmul --- \n";

    for (int dim : matrix_dims) {
        int i = dim;
        int j = dim;
        int k = dim;

        float total_time = 0.0;

        for (int run = 0; run < num_runs; run++) {
            std::vector<float> mat1 = generate_random_matrix(i, j);
            std::vector<float> mat2 = generate_random_matrix(j, k);

            float time_to_execute;
            quantize_multiply_cuda(mat1, mat2, i, j, k, &time_to_execute);

            total_time += time_to_execute;
        }

        double average_time = total_time / num_runs;

        std::cout << "Dim: " << dim << "\n Average runtime: " << average_time << " milliseconds\n\n";
    }
    
    // std::vector<float> test_mat1 = {5.47, 3.08, 1.0, 1.5, 2.5};
    // std::vector<float> test_mat2 = {5.47, 3.08, 1.5, -7.59, 1, 2.5};

    // std::vector<float> quantized_mat_result = quantize_multiply_cuda(test_mat1, test_mat2, 2, 2, 3);

    // for (float val : quantized_mat_result) {
    //     std::cout << val << '\n';
    // }

    return 0;
}
