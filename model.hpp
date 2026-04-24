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
    std::shared_ptr<LossLayer> loss_layer_;
    std::vector<std::shared_ptr<Tensor>> parameters_;
    std::vector<std::shared_ptr<Tensor>> last_output_;

public:
    Model(const std::vector<std::shared_ptr<Layer>>& layers, std::shared_ptr<LossLayer> loss_layer)
        : layers_(layers), loss_layer_(loss_layer) {
        if (layers.empty()) {
            throw std::invalid_argument("Model requires at least one layer.");
        }
        if (!loss_layer) {
            throw std::invalid_argument("Model requires a loss layer.");
        }
        for (std::shared_ptr<Layer>& layer : layers_) {
            parameters_.append_range(layer->parameters());
        }
    }

    std::vector<std::shared_ptr<Tensor>> forward(const std::vector<std::shared_ptr<Tensor>>& inputs) {
        last_output_ = inputs;
        for (std::shared_ptr<Layer>& layer : layers_) {
            last_output_ = layer->forward(last_output_);
        }
        return last_output_;
    }

    void backward(const std::vector<std::shared_ptr<Tensor>>& labels) {
        loss_layer_->forward(last_output_, labels);
        std::vector<std::shared_ptr<Tensor>> current_gradients = loss_layer_->backward({});
        for (auto it = layers_.rbegin(); it != layers_.rend(); ++it) {
            current_gradients = (*it)->backward(current_gradients);
        }
    }

    std::vector<std::shared_ptr<Tensor>>& parameters() {
        return parameters_;
    }
};

#endif // MODEL_HPP