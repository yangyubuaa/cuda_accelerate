#include "add.h"

// 前向传播，两个 Tensor 相加。这里只关注 C++ 扩展的流程，具体实现不深入探讨。
torch::Tensor Test_forward_cpu(const torch::Tensor& x,
                            const torch::Tensor& y) {
    AT_ASSERTM(x.sizes() == y.sizes(), "x must be the same size as y");
    torch::Tensor z = torch::zeros(x.sizes());
    z = 2 * x + y;
    return z;
}

// 反向传播
// 在这个例子中，z对x的导数是2，z对y的导数是1。
// 至于这个backward函数的接口（参数，返回值）为何要这样设计，后面会讲。
std::vector<torch::Tensor> Test_backward_cpu(const torch::Tensor& gradOutput) {
    torch::Tensor gradOutputX = 2 * gradOutput * torch::ones(gradOutput.sizes());
    torch::Tensor gradOutputY = gradOutput * torch::ones(gradOutput.sizes());
    return {gradOutputX, gradOutputY};
}

// pybind11 绑定
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &Test_forward_cpu, "TEST forward");
  m.def("backward", &Test_backward_cpu, "TEST backward");
}