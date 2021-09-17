#include "add.h"

torch::Tensor add_forward_bind(torch::Tensor a, torch::Tensor b){
    std::cout << a << b << std::endl;
    std::cout << a[1][1] << std::endl;
    std::cout << a.data_ptr() << std::endl;
    auto dim1 = a.size(0);
    auto dim2 = a.size(1);
    auto total_nums = dim1 * dim2;
    std::cout << dim1 << std::endl;
    std::cout << dim2 << std::endl;
    auto* ptr = (float*)a.data_ptr();
    std::cout << ptr[5] << std::endl;
    int i = total_nums;
    for(int i = 0; i < total_nums; i++){
        std::cout <<"```"<< *(ptr+i) << std::endl;
    }
    return a;
//    return vector_add(a, b);
}

std::vector<torch::Tensor> add_backward_bind(torch::Tensor result){
    std::cout << result << std::endl;
//    return vector_add_backward(result)
    return {result, result};
}

// pybind11 绑定
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &add_forward_bind, "TEST forward");
  m.def("backward", &add_backward_bind, "TEST backward");
}