#ifndef OPTIMIZER_HPP
#define OPTIMIZER_HPP

#include "tensor.hpp"

#include <functional>
#include <memory>
#include <stdexcept>
#include <vector>

class Optimizer {
protected:
    std::vector<std::shared_ptr<Tensor>> parameters_;
public:
    Optimizer(const std::vector<std::shared_ptr<Tensor>>& parameters)
        : parameters_(parameters) {
        if (parameters.empty()) {
            throw std::invalid_argument("Optimizer requires at least one parameter.");
        }
    }
    virtual void step() = 0;
    void zero_grad() {
        for (std::shared_ptr<Tensor>& param : parameters_) {
            if (!param->requires_grad()) {
                continue;
            }
            param->zero_grad();
        }
    }
};

#endif // OPTIMIZER_HPP