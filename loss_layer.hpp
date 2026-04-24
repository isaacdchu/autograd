#ifndef LOSS_LAYER_HPP
#define LOSS_LAYER_HPP

#include "layer.hpp"
#include "tensor.hpp"
#include "ops.hpp"

#include <iostream>
#include <functional>
#include <memory>
#include <vector>
#include <cstddef>

class LossLayer {
private:
    std::size_t input_size_;
    std::shared_ptr<Tensor> input_;
    std::shared_ptr<Tensor> label_;
    std::shared_ptr<Tensor> output_;

public:
    LossLayer(std::size_t input_size, std::function<std::shared_ptr<Tensor>(std::shared_ptr<Tensor>, std::shared_ptr<Tensor>)> loss_function) 
        : input_size_(input_size) {
        input_ = std::make_shared<Tensor>(std::vector<std::size_t>{1, input_size_}, 0.0f, true);
        label_ = std::make_shared<Tensor>(std::vector<std::size_t>{1, input_size_}, 0.0f, true);
        output_ = loss_function(input_, label_);
    }

    std::vector<std::shared_ptr<Tensor>> forward(
        const std::vector<std::shared_ptr<Tensor>>& inputs,
        const std::vector<std::shared_ptr<Tensor>>& labels
    ) {
        std::vector<std::shared_ptr<Tensor>> outputs;
        for (std::size_t i = 0; i < inputs.size(); i++) {
            input_->set_values(inputs[i]->values());
            label_->set_values(labels[i]->values());
            output_->forward();
            outputs.emplace_back(std::make_shared<Tensor>(output_->shape(), output_->values(), false));
        }
        return outputs;
    }

    std::vector<std::shared_ptr<Tensor>> backward(const std::vector<std::shared_ptr<Tensor>>& output_gradients) {
        std::vector<std::shared_ptr<Tensor>> input_gradients;
        const std::vector<float> ones(output_->size(), 1.0f);
        for (std::size_t i = 0; i < output_gradients.size(); i++) {
            output_->set_gradients(ones);
            output_->backward();
            input_gradients.emplace_back(std::make_shared<Tensor>(input_->shape(), input_->gradients(), false));
        }
        return input_gradients;
    }
};

#endif // LOSS_LAYER_HPP