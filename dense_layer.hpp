#ifndef DENSE_LAYER_HPP
#define DENSE_LAYER_HPP

#include "layer.hpp"
#include "tensor.hpp"

#include <memory>
#include <vector>
#include <cstddef>

class DenseLayer : public Layer {
private:
    std::shared_ptr<Tensor> weights_;
    std::shared_ptr<Tensor> biases_;
    std::size_t input_size_;
    std::size_t output_size_;

public:
    DenseLayer(std::size_t input_size, std::size_t output_size) 
        : input_size_(input_size), output_size_(output_size) {
        weights_ = std::make_shared<Tensor>(
            std::vector<std::size_t>{input_size_, output_size_},
            std::vector<float>(input_size_ * output_size_, 0.01f),
            true
        );
        biases_ = std::make_shared<Tensor>(
            std::vector<std::size_t>{output_size_},
            std::vector<float>(output_size_, 0.0f),
            true
        );
    }

    std::vector<std::shared_ptr<Tensor>> forward(std::vector<std::shared_ptr<Tensor>>& inputs) override {
        return {};
    }
};

#endif // DENSE_LAYER_HPP