#include "matmul.h"
#include <torch/extension.h>
#include <vector>
void launch_mat_mul(torch::Tensor a, torch::Tensor b, torch::Tensor &c);

void mat_mul_function(torch::Tensor a, torch::Tensor b, torch::Tensor &c){
    launch_mat_mul(a, b, c);
    return;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m){
  m.def("matmul_cuda", &mat_mul_function, "TEST matmul");
}
