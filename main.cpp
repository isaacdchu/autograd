#include "tensor.hpp"
#include "ops.hpp"
#include "loss.hpp"
#include "optimizer.hpp"
#include "sgd.hpp"
#include "adam.hpp"

#include <iostream>
#include <vector>
#include <memory>
#include <cstddef>

int main() {
    std::shared_ptr<Tensor> tensor_a = std::make_shared<Tensor>(std::vector<std::size_t>{2, 3}, std::vector<float>{1.0, 2.0, 3.0, 4.0, 5.0, 6.0}, true);
    std::shared_ptr<Tensor> tensor_b = std::make_shared<Tensor>(std::vector<std::size_t>{2, 3}, 2.0f, true);
    std::shared_ptr<Tensor> tensor_c = Ops::element_wise_multiply(tensor_a, tensor_b);
    std::shared_ptr<Tensor> tensor_d = Ops::transpose(tensor_c);
    std::shared_ptr<Tensor> tensor_e = Ops::scale(tensor_d, 3.0f);
    std::shared_ptr<Tensor> target = std::make_shared<Tensor>(std::vector<std::size_t>{3, 2}, 36.0f);
    std::shared_ptr<Tensor> loss = Loss::mse(tensor_e, target);
    std::unique_ptr<Optimizer> optimizer = std::make_unique<Adam>(std::vector<std::shared_ptr<Tensor>>{tensor_a, tensor_b}, 0.01f);
    loss->set_gradients(std::vector<float>{1.0f});
    loss->backward();
    std::cout << "Tensor A: " << tensor_a->to_string() << std::endl;
    std::cout << "Tensor B: " << tensor_b->to_string() << std::endl;
    std::cout << "Tensor C: " << tensor_c->to_string() << std::endl;
    std::cout << "Tensor D: " << tensor_d->to_string() << std::endl;
    std::cout << "Tensor E: " << tensor_e->to_string() << std::endl;
    std::cout << "Loss: " << loss->to_string() << std::endl;
    optimizer->step();
    std::cout << "After stepping optimizer:" << std::endl;
    std::cout << "Tensor A: " << tensor_a->to_string() << std::endl;
    std::cout << "Tensor B: " << tensor_b->to_string() << std::endl;
    std::cout << "Tensor C: " << tensor_c->to_string() << std::endl;
    std::cout << "Tensor D: " << tensor_d->to_string() << std::endl;
    std::cout << "Tensor E: " << tensor_e->to_string() << std::endl;
    std::cout << "Loss: " << loss->to_string() << std::endl;
    loss->zero_grad();
    loss->forward();
    std::cout << "After zero grad and forward:" << std::endl;
    std::cout << "Tensor A: " << tensor_a->to_string() << std::endl;
    std::cout << "Tensor B: " << tensor_b->to_string() << std::endl;
    std::cout << "Tensor C: " << tensor_c->to_string() << std::endl;
    std::cout << "Tensor D: " << tensor_d->to_string() << std::endl;
    std::cout << "Tensor E: " << tensor_e->to_string() << std::endl;
    std::cout << "Loss: " << loss->to_string() << std::endl;
    return 0;
}