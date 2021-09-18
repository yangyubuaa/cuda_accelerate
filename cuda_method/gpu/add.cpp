#include "add.h"

// 输入两个tensor，返回两个tensor的相加
void add_forward_bind(torch::Tensor a, torch::Tensor b, torch::Tensor &c){
  vector_add(a, b, c);
  std::cout << c << std::endl;
}

// pybind11 绑定
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &add_forward_bind, "TEST forward");
}