#include <torch/extension.h>
#include <vector>

// 设置c矩阵的元素
__device__ void setElement(float* matrix, int i, int j, int dim_x, int dim_y, int value){
    matrix[i*dim_y+j] = value;
}

// 获取矩阵的索引为[i][j]的元素
__device__ int getElement(float* matrix, int i, int j, int dim_x, int dim_y){
    return matrix[i*dim_y + j];
}

__global__ void matmul_kernel(float* a, float* b, float* c, int dim_x, int dim_y, int dim_a_x, int dim_a_y, int dim_b_x, int dim_b_y){
    printf("%d, %d, %d, %d, %d, %d", dim_x, dim_y, dim_a_x, dim_a_y, dim_b_x, dim_b_y);
    int thread_x = blockIdx.x * blockDim.x + threadIdx.x;
    int thread_y = blockIdx.y * blockDim.y + threadIdx.y;
    int stride_x = blockDim.x * gridDim.x;
    int stride_y = blockDim.y * gridDim.y;

    for(int i=thread_y; i< dim_y; i = i + stride_y){
        for(int j=thread_x; j < dim_x;j = j + stride_x){
            printf("%d, %d", i, j);
            int value = 0;
            for(int k=0; k< dim_a_y;k++){
                value = value + getElement(a, i, k, dim_a_x, dim_a_y)*getElement(a, k, j, dim_b_x, dim_b_y);
            }
            setElement(c, i, j, dim_x, dim_y, value);
        }
    }
}

void launch_mat_mul(torch::Tensor a, torch::Tensor b, torch::Tensor &c){
    dim3 block_per_grid = (1);
    dim3 threads_per_block = (10, 10);
    float* a_data_ptr = (float*)a.data_ptr<float>();
    float* b_data_ptr = (float*)b.data_ptr<float>();
    float* c_data_ptr = (float*)c.data_ptr<float>();
    int dim_x = c.size(0);
    int dim_y = c.size(1);
    int dim_a_x = a.size(0);
    int dim_a_y = a.size(1);
    int dim_b_x = b.size(0);
    int dim_b_y = b.size(1);
    matmul_kernel<<<block_per_grid, threads_per_block>>>(a_data_ptr, b_data_ptr, c_data_ptr, dim_x, dim_y, dim_a_x, dim_a_y, dim_b_x, dim_b_y);
    return;
}


