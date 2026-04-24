#ifndef DENSE_LAYER_HPP
#define DENSE_LAYER_HPP

#include "layer.hpp"
#include "tensor.hpp"
#include "ops.hpp"

#include <iostream>
#include <memory>
#include <vector>
#include <cstddef>

class DenseLayer : public Layer {
private:
    std::shared_ptr<Tensor> weights_;
    std::shared_ptr<Tensor> biases_;
    std::size_t input_size_;
    std::size_t output_size_;
    std::shared_ptr<Tensor> input_;
    std::shared_ptr<Tensor> product_;
    std::shared_ptr<Tensor> output_;

public:
    DenseLayer(std::size_t input_size, std::size_t output_size) 
        : input_size_(input_size), output_size_(output_size) {
        weights_ = std::make_shared<Tensor>(
            std::vector<std::size_t>{input_size_, output_size_},
            std::vector<float>(input_size_ * output_size_, 0.01f),
            true
        );
        biases_ = std::make_shared<Tensor>(
            std::vector<std::size_t>{1, output_size_},
            std::vector<float>(output_size_, 0.1f),
            true
        );
        input_ = std::make_shared<Tensor>(std::vector<std::size_t>{1, input_size_}, 0.0f, true);
        product_ = Ops::tensor_product(input_, weights_);
        output_ = Ops::add(product_, biases_);
    }

    std::vector<std::shared_ptr<Tensor>> forward(const std::vector<std::shared_ptr<Tensor>>& inputs) override {
        std::vector<std::shared_ptr<Tensor>> outputs;
        for (const auto& input : inputs) {
            input_->set_values(input->values());
            output_->forward();
            outputs.emplace_back(std::make_shared<Tensor>(output_->shape(), output_->values(), true));
        }
        return outputs;
    }

    std::vector<std::shared_ptr<Tensor>> backward(const std::vector<std::shared_ptr<Tensor>>& output_gradients) override {
        std::vector<std::shared_ptr<Tensor>> input_gradients;
        for (auto& output_gradient : output_gradients) {
            output_->set_gradients(output_gradient->values());
            output_->backward();
            input_gradients.emplace_back(std::make_shared<Tensor>(input_->shape(), input_->gradients(), true));
        }
        return input_gradients;
    }

    std::vector<std::shared_ptr<Tensor>> parameters() override {
        return {weights_, biases_};
    }
};

#endif // DENSE_LAYER_HPP