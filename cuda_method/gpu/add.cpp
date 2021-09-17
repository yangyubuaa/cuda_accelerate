#include "add.h"

void add_forward_bind(torch::Tensor a, torch::Tensor b, torch::Tensor c){
    vector_add(a, b, c);
}

// pybind11 绑定
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &add_forward_bind, "TEST forward");
}