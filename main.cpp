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
    std::shared_ptr<Tensor> tensor_a1 = std::make_shared<Tensor>(std::vector<std::size_t>{2, 3}, std::vector<float>{1.0, 2.0, 3.0, 4.0, 5.0, 6.0}, true);
    std::shared_ptr<Tensor> tensor_b1 = std::make_shared<Tensor>(std::vector<std::size_t>{3, 2}, 2.0f, true);
    std::shared_ptr<Tensor> tensor_c1 = Ops::tensor_product(tensor_a1, tensor_b1);
    tensor_c1->forward();
    tensor_c1->set_gradients(std::vector<float>(tensor_c1->size(), 1.0f));
    tensor_c1->backward();
    std::cout << "Tensor A1: " << tensor_a1->to_string() << std::endl;
    std::cout << "Tensor B1: " << tensor_b1->to_string() << std::endl;
    std::cout << "Tensor C1: " << tensor_c1->to_string() << std::endl;
    return 0;
}