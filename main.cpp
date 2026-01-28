#include "tensor.hpp"
#include "ops.hpp"
#include "loss.hpp"
#include "optimizer.hpp"
#include "sgd.hpp"
#include "adam.hpp"
#include "layer.hpp"
#include "model.hpp"

#include <iostream>
#include <vector>
#include <memory>
#include <cstddef>

int main() {
    std::shared_ptr<Tensor> tensor_a1 = std::make_shared<Tensor>(std::vector<std::size_t>{8, 8}, 1.0f, true);
    std::shared_ptr<Tensor> tensor_b1 = std::make_shared<Tensor>(std::vector<std::size_t>{3, 3}, 2.0f, true);
    std::shared_ptr<Tensor> tensor_c1 = Ops::convolution_2d(tensor_a1, tensor_b1, 1, Ops::PaddingFill::REPLICATE);
    tensor_c1->forward();
    tensor_c1->set_gradients(std::vector<float>(tensor_c1->size(), 1.0f));
    std::cout << "performing backward pass..." << std::endl;
    tensor_c1->backward();
    std::cout << "Tensor A1: " << tensor_a1->to_string() << std::endl;
    std::cout << "Tensor B1: " << tensor_b1->to_string() << std::endl;
    std::cout << "Tensor C1: " << tensor_c1->to_string() << std::endl;
    return 0;
}