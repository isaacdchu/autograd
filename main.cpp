#include "tensor.hpp"
#include "ops.hpp"

#include <iostream>
#include <vector>
#include <memory>
#include <cstddef>

int main() {
    std::shared_ptr<Tensor> tensor_a = std::make_shared<Tensor>(std::vector<std::size_t>{2, 3}, std::vector<float>{1.0, 2.0, 3.0, 4.0, 5.0, 6.0}, true);
    std::shared_ptr<Tensor> tensor_b = std::make_shared<Tensor>(std::vector<std::size_t>{2, 3}, 2.0f, false);
    std::shared_ptr<Tensor> tensor_c = Ops::element_wise_multiply(tensor_a, tensor_b);
    std::shared_ptr<Tensor> tensor_d = Ops::transpose(tensor_c);
    std::shared_ptr<Tensor> tensor_e = Ops::scale(tensor_d, 3.0f);
    tensor_e->set_gradients(std::vector<float>{1.0f, 1.0f, 2.0f, 1.0f, 1.0f, 2.0f});
    tensor_e->backward();
    std::cout << "Tensor A: " << tensor_a->to_string() << std::endl;
    std::cout << "Tensor B: " << tensor_b->to_string() << std::endl;
    std::cout << "Tensor C: " << tensor_c->to_string() << std::endl;
    std::cout << "Tensor D: " << tensor_d->to_string() << std::endl;
    std::cout << "Tensor E: " << tensor_e->to_string() << std::endl;
    tensor_e->zero_grad();
    tensor_b->set_values(tensor_a);
    tensor_b->set_requires_grad(true);
    tensor_e->forward();
    tensor_e->set_gradients(std::vector<float>{1.0f, 0.0f, 1.0f, 0.0f, 1.0f, 0.0f});
    tensor_e->backward();
    std::cout << "After modifying Tensor A and forwarding:" << std::endl;
    std::cout << "Tensor A: " << tensor_a->to_string() << std::endl;
    std::cout << "Tensor B: " << tensor_b->to_string() << std::endl;
    std::cout << "Tensor C: " << tensor_c->to_string() << std::endl;
    std::cout << "Tensor D: " << tensor_d->to_string() << std::endl;
    std::cout << "Tensor E: " << tensor_e->to_string() << std::endl;
    return 0;
}