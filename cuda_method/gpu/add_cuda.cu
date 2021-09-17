#include <torch/extension.h>
#include <vector>
// __global__ void vector_add_kernel(() a, torch::Tensor b, torch::Tensor c){
//     auto dim1 = blockIdx.x * blockDim.x + threadIdx.x;
//     auto dim2 = blockIdx.y * blockDim.y + threadIdx.y;
//     c[dim1][dim2] = a[dim1][dim2] + b[dim1][dim2];
// }

void vector_add(torch::Tensor a, torch::Tensor b, torch::Tensor c){
    // dim3 block_per_grid(2, 2);
    // dim3 threads_per_block(2, 2);
    auto a_data = a.data();
    auto b_data = b.data();
    auto c_data = c.data();
    std::cout << a_data << b_data << c_data << std::endl;
    // vector_add_kernel<<<block_per_grid, threads_per_block>>>(a_data, b_data, c_data);
    return;
}

