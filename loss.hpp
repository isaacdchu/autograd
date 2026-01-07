#ifndef LOSS_HPP
#define LOSS_HPP

#include "tensor.hpp"

#include <functional>
#include <memory>
#include <stdexcept>
#include <vector>
#include <cstddef>

class Loss {
public:
    static std::shared_ptr<Tensor> mse(std::shared_ptr<Tensor> predictions, std::shared_ptr<Tensor> targets) {
        if (!Tensor::vectors_are_equal(predictions->shape(), targets->shape())) {
            throw std::invalid_argument("(Loss::mse) Predictions and targets must have the same shape.");
        }
        float mse_value = 0.0f;
        for (std::size_t i = 0; i < predictions->size(); i++) {
            const float diff = predictions->values()[i] - targets->values()[i];
            mse_value += diff * diff;
        }
        mse_value /= static_cast<float>(predictions->size());
        std::shared_ptr<Tensor> mse = std::make_shared<Tensor>(std::vector<std::size_t>{1}, std::vector<float>{mse_value}, predictions->requires_grad() || targets->requires_grad());
        std::vector<float> grad_predictions = std::vector<float>();
        if (predictions->requires_grad()) {
            grad_predictions.resize(predictions->size());
            for (std::size_t i = 0; i < grad_predictions.size(); i++) {
                grad_predictions[i] = (2.0f * predictions->values()[i] - targets->values()[i]) / static_cast<float>(predictions->size());
            }
        }
        std::function<std::vector<float>()> grad_predictions_initializer = [predictions_weak = std::weak_ptr<Tensor>(predictions), targets_weak = std::weak_ptr<Tensor>(targets)]() {
            std::shared_ptr<Tensor> predictions_shared = predictions_weak.lock();
            std::shared_ptr<Tensor> targets_shared = targets_weak.lock();
            if (!predictions_shared || !targets_shared) {
                throw std::runtime_error("(Loss::mse) Predecessor tensor has been deallocated.");
            }
            std::vector<float> grad_predictions(predictions_shared->size());
            for (std::size_t i = 0; i < grad_predictions.size(); i++) {
                grad_predictions[i] = (2.0f * predictions_shared->values()[i] - targets_shared->values()[i]) / static_cast<float>(predictions_shared->size());
            }
            return grad_predictions;
        };
        std::vector<float> grad_targets = std::vector<float>();
        if (targets->requires_grad()) {
            grad_targets.resize(targets->size());
            for (std::size_t i = 0; i < targets->size(); i++) {
                grad_targets[i] = (2.0f * predictions->values()[i] - targets->values()[i]) / static_cast<float>(predictions->size());
            }
        }
        std::function<std::vector<float>()> grad_targets_initializer = [predictions_weak = std::weak_ptr<Tensor>(predictions), targets_weak = std::weak_ptr<Tensor>(targets)]() {
            std::shared_ptr<Tensor> predictions_shared = predictions_weak.lock();
            std::shared_ptr<Tensor> targets_shared = targets_weak.lock();
            if (!predictions_shared || !targets_shared) {
                throw std::runtime_error("(Loss::mse) Predecessor tensor has been deallocated.");
            }
            std::vector<float> grad_targets(targets_shared->size());
            for (std::size_t i = 0; i < grad_targets.size(); i++) {
                grad_targets[i] = (2.0f * predictions_shared->values()[i] - targets_shared->values()[i]) / static_cast<float>(predictions_shared->size());
            }
            return grad_targets;
        };
        std::function<void(std::vector<float>&, const std::vector<float>&, const std::vector<float>&)> update_function = [](
            std::vector<float> &pred_tensor_gradients,
            const std::vector<float> &current_gradients,
            const std::vector<float> &pred_struct_gradients
        ) {
            for (std::size_t i = 0; i < pred_tensor_gradients.size(); i++) {
                pred_tensor_gradients.at(i) += current_gradients.at(0) * pred_struct_gradients.at(i);
            }
        };
        mse->add_predecessor(predictions, grad_predictions, predictions->requires_grad(), grad_predictions_initializer, update_function);
        mse->add_predecessor(targets, grad_targets, targets->requires_grad(), grad_targets_initializer, update_function);
        mse->forward_function_ = [](std::vector<float>& values, const std::vector<Predecessor>& preds) {
            const std::shared_ptr<Tensor> predictions = preds.at(0).tensor.lock();
            const std::shared_ptr<Tensor> targets = preds.at(1).tensor.lock();
            if (!predictions || !targets) {
                throw std::runtime_error("(Loss::mse) Predecessor tensor has been deallocated.");
            }
            float mse_value = 0.0f;
            for (std::size_t i = 0; i < predictions->size(); i++) {
                const float diff = predictions->values()[i] - targets->values()[i];
                mse_value += diff * diff;
            }
            values[0] = mse_value / static_cast<float>(predictions->size());
        };
        return mse;
    }
};

#endif // LOSS_HPP