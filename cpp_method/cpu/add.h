#include <torch/extension.h>
#include <vector>

// 前向传播
torch::Tensor Test_forward_cpu(const torch::Tensor& inputA,
                            const torch::Tensor& inputB);
// 反向传播
std::vector<torch::Tensor> Test_backward_cpu(const torch::Tensor& gradOutput);