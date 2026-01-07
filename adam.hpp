#ifndef ADAM_HPP
#define ADAM_HPP

#include "optimizer.hpp"
#include "tensor.hpp"

#include <iostream>
#include <memory>
#include <vector>
#include <stdexcept>
#include <cstddef>
#include <cmath>

class Adam : public Optimizer {
private:
    const float learning_rate_;
    const float beta_1_;
    const float beta_2_;
    const float epsilon_;
    std::vector<std::vector<float>> m_;
    std::vector<std::vector<float>> v_;
    std::size_t t_;

public:
    Adam(
        const std::vector<std::shared_ptr<Tensor>>& parameters,
        float learning_rate = 0.001f,
        float beta_1 = 0.9f,
        float beta_2 = 0.999f,
        float epsilon = 1e-8f
    ) : Optimizer(parameters),
        learning_rate_(learning_rate),
        beta_1_(beta_1),
        beta_2_(beta_2),
        epsilon_(epsilon),
        t_(0) {
        if (learning_rate_ <= 0.0f) {
            throw std::invalid_argument("Learning rate must be positive.");
        }
        if (beta_1_ < 0.0f || beta_1_ >= 1.0f) {
            throw std::invalid_argument("Beta_1 must be in the range [0, 1).");
        }
        if (beta_2_ < 0.0f || beta_2_ >= 1.0f) {
            throw std::invalid_argument("Beta_2 must be in the range [0, 1).");
        }
        if (epsilon_ <= 0.0f) {
            throw std::invalid_argument("Epsilon must be positive.");
        }
        for (const std::shared_ptr<Tensor>& param : parameters_) {
            m_.emplace_back(std::vector<float>(param->size(), 0.0f));
            v_.emplace_back(std::vector<float>(param->size(), 0.0f));
        }
    }

    void step() override {
        t_++;
        for (std::size_t idx = 0; idx < parameters_.size(); ++idx) {
            std::shared_ptr<Tensor> param = parameters_[idx];
            if (!param->requires_grad()) {
                continue;
            }
            std::vector<float> values = param->values();
            std::vector<float> gradients = param->gradients();
            if (gradients.size() != values.size()) {
                gradients.resize(values.size(), 0.0f);
            }
            std::vector<float>& m = m_[idx];
            std::vector<float>& v = v_[idx];
            for (std::size_t i = 0; i < values.size(); i++) {
                m[i] = beta_1_ * m[i] + (1.0f - beta_1_) * gradients[i];
                v[i] = beta_2_ * v[i] + (1.0f - beta_2_) * (gradients[i] * gradients[i]);
                const float m_hat = m[i] / (1.0f - std::powf(beta_1_, t_));
                const float v_hat = v[i] / (1.0f - std::powf(beta_2_, t_));
                values[i] -= learning_rate_ * m_hat / (std::sqrt(v_hat) + epsilon_);
            }
            param->set_values(values);
        }
    }
};

#endif // ADAM_HPP