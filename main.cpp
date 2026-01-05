#include "tensor.hpp"
#include "ops.hpp"

#include <iostream>
#include <vector>
#include <memory>
#include <cstddef>

int main() {
    std::shared_ptr<Tensor> tensor_a = std::make_shared<Tensor>(std::vector<std::size_t>{2, 2}, 1.0f, true);
    std::shared_ptr<Tensor> tensor_b = std::make_shared<Tensor>(std::vector<std::size_t>{2, 2}, 2.0f, false);
    std::shared_ptr<Tensor> tensor_c = Ops::element_wise_multiply(tensor_a, tensor_b);
    std::shared_ptr<Tensor> tensor_d = Ops::scale(tensor_c, 3.0f);
    tensor_d->set_gradients(std::vector<float>{1.0f, 1.0f, 1.0f, 0.0f});
    tensor_d->backward();
    std::cout << "Tensor A: " << tensor_a->to_string() << std::endl;
    std::cout << "Tensor B: " << tensor_b->to_string() << std::endl;
    std::cout << "Tensor C: " << tensor_c->to_string() << std::endl;
    std::cout << "Tensor D: " << tensor_d->to_string() << std::endl;
    tensor_d->zero_grad();
    tensor_a->set_values(tensor_b);
    tensor_b->set_requires_grad(true);
    tensor_d->forward();
    tensor_d->set_gradients(std::vector<float>{1.0f, 0.0f, 1.0f, 0.0f});
    tensor_d->backward();
    std::cout << "After modifying Tensor A and forwarding:" << std::endl;
    std::cout << "Tensor A: " << tensor_a->to_string() << std::endl;
    std::cout << "Tensor B: " << tensor_b->to_string() << std::endl;
    std::cout << "Tensor C: " << tensor_c->to_string() << std::endl;
    std::cout << "Tensor D: " << tensor_d->to_string() << std::endl;
    return 0;
}