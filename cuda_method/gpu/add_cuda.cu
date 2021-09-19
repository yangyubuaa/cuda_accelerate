#include <torch/extension.h>
#include <vector>

// 通过传递指针进行计算

__global__ void vector_add_kernel(float* a, float* b, float* c, int dim_x, int dim_y){
    auto dim1 = blockIdx.x * blockDim.x + threadIdx.x;
    auto dim2 = blockIdx.y * blockDim.y + threadIdx.y;
    if((dim2*dim_y + dim1)<dim_x*dim_y){
        c[dim2*dim_y + dim1] = a[dim2*dim_y + dim1] + b[dim2*dim_y + dim1];
    }
}

// 通过host可以通过传递引用对tensor直接进行修改，但是通过cuda，不能通过直接传递tensor引用的方式，此处有bug
void vector_add(torch::Tensor a, torch::Tensor b, torch::Tensor &c){
    dim3 block_per_grid(1);
    dim3 threads_per_block(10, 10);
    vector_add_kernel<<<block_per_grid, threads_per_block>>>(
        (float*)a.data_ptr<float>(),
        (float*)b.data_ptr<float>(),
        (float*)c.data_ptr<float>(),
        c.size(0),
        c.size(1));
    // for(int i = 0; i< a.size(0);i++){
    //     // 通过指针访问
    //     std::cout << *((float*)a.data_ptr<float>()+i) <<std::endl;
    //     std::cout << *((float*)b.data_ptr<float>()+i) <<std::endl;
    // }

    return;
}

