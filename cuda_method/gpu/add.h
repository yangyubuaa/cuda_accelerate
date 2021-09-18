#include <torch/extension.h>
#include <vector>

void vector_add(torch::Tensor a, torch::Tensor b, torch::Tensor &c);

