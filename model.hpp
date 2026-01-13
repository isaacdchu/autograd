#ifndef MODEL_HPP
#define MODEL_HPP

#include "layer.hpp"
#include "tensor.hpp"
#include "ops.hpp"
#include "loss.hpp"
#include "optimizer.hpp"

#include <memory>
#include <stdexcept>
#include <vector>
#include <cstddef>

class Model {
protected:
    std::vector<std::shared_ptr<Layer>> layers_;
    std::vector<std::shared_ptr<Tensor>> parameters_;

public:
    Model(const std::vector<std::shared_ptr<Layer>>& layers)
        : layers_(layers) {
        if (layers.empty()) {
            throw std::invalid_argument("Model requires at least one layer.");
        }
        for (std::shared_ptr<Layer>& layer : layers_) {
            parameters_.append_range(layer->parameters());
        }
    }
    std::vector<std::shared_ptr<Tensor>> forward(std::vector<std::shared_ptr<Tensor>>& inputs) {
        std::vector<std::shared_ptr<Tensor>> current_outputs = inputs;
        for (std::shared_ptr<Layer>& layer : layers_) {
            current_outputs = layer->forward(current_outputs);
        }
        return current_outputs;
    }
    std::vector<std::shared_ptr<Tensor>>& parameters() {
        return parameters_;
    }
};

#endif // MODEL_HPP