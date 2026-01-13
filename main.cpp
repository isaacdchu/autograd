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
    std::shared_ptr<Tensor> tensor_a = std::make_shared<Tensor>(std::vector<std::size_t>{2, 3}, std::vector<float>{1.0, 2.0, 3.0, 4.0, 5.0, 6.0}, true);
    std::shared_ptr<Tensor> tensor_b = std::make_shared<Tensor>(std::vector<std::size_t>{3, 2}, 2.0f, true);
    std::shared_ptr<Tensor> tensor_c = Ops::tensor_product(tensor_a, tensor_b);
    // std::shared_ptr<Tensor> target = std::make_shared<Tensor>(std::vector<std::size_t>{2, 2}, 3.0f, false);
    // std::shared_ptr<Tensor> loss = Loss::mse(tensor_c, target);
    // loss->set_gradients(std::vector<float>{1.0f});
    // loss->backward();
    tensor_c->set_gradients(std::vector<float>(tensor_c->size(), 1.0f));
    tensor_c->backward();
    std::cout << "Tensor A: " << tensor_a->to_string() << std::endl;
    std::cout << "Tensor B: " << tensor_b->to_string() << std::endl;
    std::cout << "Tensor C: " << tensor_c->to_string() << std::endl;
    // std::cout << "Loss: " << loss->to_string() << std::endl;
    return 0;
}