#include <torch/extension.h>
#include <vector>


__global__ void vector_add_kernel(const float* a, const float* b, float* c){
    auto dim1 = blockIdx.x * blockDim.x + threadIdx.x;
    auto dim2 = blockIdx.y * blockDim.y + threadIdx.y;
    printf("hello! %d, %d\n", dim1, dim2);
    // printf(c);
    printf("\n");
    // c[dim1][dim2] = a[dim1][dim2] + b[dim1][dim2];
}

// 通过host可以通过传递引用对tensor直接进行修改，但是通过cuda，不能通过直接传递tensor引用的方式，此处有bug
void vector_add(torch::Tensor a, torch::Tensor b, torch::Tensor &c){
    dim3 block_per_grid(2, 2);
    dim3 threads_per_block(2, 2);
    vector_add_kernel<<<block_per_grid, threads_per_block>>>(
        (float*)a.data_ptr(),
        (float*)b.data_ptr(),
        (float*)c.data_ptr());
    std::cout << c.size(0) << std::endl;
    for(int i = 0; i< a.size(0);i++){
        std::cout << (float*)a.data_ptr()+i <<std::endl;
        std::cout << (float*)b.data_ptr()+i <<std::endl;
    }
    // std::cout << c << std::endl;
    return;
}

