#include <torch/extension.h>
#include <vector>

void launch_mat_mul(torch::Tensor a, torch::Tensor b, torch::Tensor &c);


