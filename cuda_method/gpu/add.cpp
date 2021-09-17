#include "add.h"

add_forward_bind(const torch::Tensor &a, const torch::Tensor &b, tensor::Tensor &c){
    vector_add(a, b, c);

}

std::vector<torch::Tensor> add_backward_bind(torch::Tensor result){
//    std::cout << result << std::endl;
    return vector_add_backward(result);
//    return {result, result};
}

// pybind11 绑定
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &add_forward_bind, "TEST forward");
  m.def("backward", &add_backward_bind, "TEST backward");
}