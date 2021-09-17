#include <torch/extension.h>
#include <vector>

torch::Tensor vector_add(torch::Tensor a, torch::Tensor b);
vector<torch::Tensor> vector_add_backward(torch::Tensor result);
