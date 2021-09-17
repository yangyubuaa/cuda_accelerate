__global__ void vector_add_kernel(const torch::Tensor &a, const torch::Tensor &b, torch::Tensor &c){
    dim1 = blockIndex.x * blockDim.x + threadIdx.x;
    dim2 = blockIndex.y * blockDim.y + threadIdx.y;
    c[dim1][dim2] = a[dim1][dim2] + b[dim1][dim2];
}

torch::Tensor vector_add(const torch::Tensor &a, const torch::Tensor &b, torch::Tensor &c){
    return vector_add_kernel(a, b, c);
}

std::vector<torch::Tensor> vector_add_backward(const torch::Tensor &result){
    return {result, result};
}

