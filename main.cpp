#include "tensor.hpp"
#include "ops.hpp"
#include "loss.hpp"
#include "optimizer.hpp"
#include "sgd.hpp"
#include "adam.hpp"
#include "layer.hpp"
#include "dense_layer.hpp"
#include "loss_layer.hpp"
#include "model.hpp"

#include <iostream>
#include <vector>
#include <memory>
#include <cstddef>

int main() {
    auto model = Model({
        std::make_shared<DenseLayer>(2, 3),
        std::make_shared<DenseLayer>(3, 1)
    }, std::make_shared<LossLayer>(1, Loss::mse));
    auto optimizer = SGD(model.parameters(), 0.01f);
    std::vector<std::shared_ptr<Tensor>> inputs = {
        std::make_shared<Tensor>(std::vector<std::size_t>{1, 2}, std::vector<float>{1.0f, 2.0f}, true),
        std::make_shared<Tensor>(std::vector<std::size_t>{1, 2}, std::vector<float>{3.0f, 4.0f}, true)
    };
    // optimizer.zero_grad();
    auto outputs = model.forward(inputs);
    for (const auto& output : outputs) {
        std::cout << "Output: ";
        for (float value : output->values()) {
            std::cout << value << " ";
        }
        std::cout << std::endl;
    }
    std::vector<std::shared_ptr<Tensor>> labels = {
        std::make_shared<Tensor>(std::vector<std::size_t>{1, 1}, std::vector<float>{3.0f}, true),
        std::make_shared<Tensor>(std::vector<std::size_t>{1, 1}, std::vector<float>{2.0f}, true)
    };
    model.backward(labels);
    optimizer.step();
    for (const auto& param : model.parameters()) {
        std::cout << param->to_string() << std::endl;
    }
    outputs = model.forward(inputs);
    for (const auto& output : outputs) {
        std::cout << "Output: ";
        for (float value : output->values()) {
            std::cout << value << " ";
        }
        std::cout << std::endl;
    }
    return 0;
}