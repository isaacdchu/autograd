#ifndef SGD_HPP
#define SGD_HPP

#include "optimizer.hpp"
#include "tensor.hpp"

#include <memory>
#include <stdexcept>
#include <vector>
#include <cstddef>

class SGD : public Optimizer {
private:
    const float learning_rate_;

public:
    SGD(const std::vector<std::shared_ptr<Tensor>>& parameters, float learning_rate = 0.001f)
        : Optimizer(parameters), learning_rate_(learning_rate) {
        if (learning_rate_ <= 0.0f) {
            throw std::invalid_argument("Learning rate must be positive.");
        }
    }

    void step() override {
        for (std::shared_ptr<Tensor>& param : parameters_) {
            if (!param->requires_grad()) {
                continue;
            }
            std::vector<float> values = param->values();
            std::vector<float> gradients = param->gradients();
            if (values.size() != gradients.size()) {
                gradients.resize(values.size(), 0.0f);
            }
            for (std::size_t i = 0; i < values.size(); i++) {
                values[i] -= learning_rate_ * gradients[i];
            }
            param->set_values(values);
        }
    }
};

#endif // SGD_HPP