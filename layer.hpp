#ifndef LAYER_HPP
#define LAYER_HPP

#include "tensor.hpp"

#include <memory>
#include <stdexcept>
#include <vector>
#include <cstddef>

class Layer {
public:
    virtual std::vector<std::shared_ptr<Tensor>> forward(const std::vector<std::shared_ptr<Tensor>>& inputs) = 0;
    virtual std::vector<std::shared_ptr<Tensor>> backward(const std::vector<std::shared_ptr<Tensor>>& output_gradients) = 0;
    virtual std::vector<std::shared_ptr<Tensor>> parameters() = 0;
};

#endif // LAYER_HPP