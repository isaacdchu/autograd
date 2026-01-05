#ifndef OPS_HPP
#define OPS_HPP

#include "tensor.hpp"

#include <algorithm>
#include <functional>
#include <iostream>
#include <memory>
#include <vector>
#include <stdexcept>
#include <cmath>
#include <cstddef>

class Ops {
public:
    static std::shared_ptr<Tensor> add(std::shared_ptr<Tensor> a, std::shared_ptr<Tensor> b) {
        if (!Tensor::vectors_are_equal(a->shape(), b->shape())) {
            throw std::invalid_argument("(Ops::add) Tensors must have the same shape for addition.");
        }
        std::vector<float> result_data(a->size());
        for (std::size_t i = 0; i < result_data.size(); i++) {
            result_data[i] = a->values_[i] + b->values_[i];
        }
        std::shared_ptr<Tensor> result = std::make_shared<Tensor>(a->shape(), result_data, a->requires_grad() || b->requires_grad());
        std::vector<float> grad_a;
        if (a->requires_grad()) {
            grad_a = std::vector<float>(result_data.size(), 1.0f);
        }
        std::function<std::vector<float>()> grad_a_initializer = [size = result_data.size()]() {
            return std::vector<float>(size, 1.0f);
        };
        result->add_predecessor(a, grad_a, a->requires_grad(), grad_a_initializer);
        std::vector<float> grad_b;
        if (b->requires_grad()) {
            grad_b = std::vector<float>(result_data.size(), 1.0f);
        }
        std::function<std::vector<float>()> grad_b_initializer = [size = result_data.size()]() {
            return std::vector<float>(size, 1.0f);
        };
        result->add_predecessor(b, grad_b, b->requires_grad(), grad_b_initializer);
        result->forward_function_ = [](std::vector<float>& values, const std::vector<Predecessor>& preds) {
            const std::shared_ptr<Tensor> a = preds.at(0).tensor.lock();
            const std::shared_ptr<Tensor> b = preds.at(1).tensor.lock();
            if (!a || !b) {
                throw std::runtime_error("(Ops::add) Predecessor tensor has been deallocated.");
            }
            for (std::size_t i = 0; i < values.size(); i++) {
                values[i] = a->values_[i] + b->values_[i];
            }
        };
        return result;
    }

    static std::shared_ptr<Tensor> element_wise_multiply(std::shared_ptr<Tensor> a, std::shared_ptr<Tensor> b) {
        if (!Tensor::vectors_are_equal(a->shape(), b->shape())) {
            throw std::invalid_argument("(Ops::element_wise_multiply) Tensors must have the same shape for element-wise multiplication.");
        }
        std::vector<float> result_data(a->size());
        for (std::size_t i = 0; i < result_data.size(); i++) {
            result_data[i] = a->values_[i] * b->values_[i];
        }
        std::shared_ptr<Tensor> result = std::make_shared<Tensor>(a->shape(), result_data, a->requires_grad() || b->requires_grad());
        std::vector<float> grad_a;
        if (a->requires_grad()) {
            grad_a = b->values();
        }
        std::function<std::vector<float>()> grad_a_initializer = [b_weak = std::weak_ptr<Tensor>(b)]() {
            std::shared_ptr<Tensor> b_shared = b_weak.lock();
            if (!b_shared) {
                throw std::runtime_error("(Ops::element_wise_multiply) Predecessor tensor has been deallocated.");
            }
            return b_shared->values();
        };
        result->add_predecessor(a, grad_a, a->requires_grad(), grad_a_initializer);
        std::vector<float> grad_b;
        if (b->requires_grad()) {
            grad_b = a->values();
        }
        std::function<std::vector<float>()> grad_b_initializer = [a_weak = std::weak_ptr<Tensor>(a)]() {
            std::shared_ptr<Tensor> a_shared = a_weak.lock();
            if (!a_shared) {
                throw std::runtime_error("(Ops::element_wise_multiply) Predecessor tensor has been deallocated.");
            }
            return a_shared->values();
        };
        result->add_predecessor(b, grad_b, b->requires_grad(), grad_b_initializer);
        result->forward_function_ = [](std::vector<float>& values, const std::vector<Predecessor>& preds) {
            const std::shared_ptr<Tensor> a = preds.at(0).tensor.lock();
            const std::shared_ptr<Tensor> b = preds.at(1).tensor.lock();
            if (!a || !b) {
                throw std::runtime_error("(Ops::element_wise_multiply) Predecessor tensor has been deallocated.");
            }
            for (std::size_t i = 0; i < values.size(); i++) {
                values[i] = a->values()[i] * b->values()[i];
            }
        };
        return result;
    }

    static std::shared_ptr<Tensor> scale(std::shared_ptr<Tensor> tensor, float scalar) {
        if (std::isnan(scalar) || std::isinf(scalar)) {
            throw std::invalid_argument("(Ops::scale) Scalar value must be a valid finite number.");
        }
        std::vector<float> result_data(tensor->size());
        for (std::size_t i = 0; i < result_data.size(); i++) {
            result_data[i] = tensor->values_[i] * scalar;
        }
        std::shared_ptr<Tensor> result = std::make_shared<Tensor>(tensor->shape(), result_data, tensor->requires_grad());
        std::vector<float> gradients;
        if (tensor->requires_grad()) {
            gradients = std::vector<float>(result_data.size(), scalar);
        }
        std::function<std::vector<float>()> grad_initializer = [size = result_data.size(), scalar]() {
            return std::vector<float>(size, scalar);
        };
        result->add_predecessor(tensor, gradients, tensor->requires_grad(), grad_initializer);
        result->forward_function_ = [scalar](std::vector<float>& values, const std::vector<Predecessor>& preds) {
            const std::shared_ptr<Tensor> tensor = preds.at(0).tensor.lock();
            if (!tensor) {
                throw std::runtime_error("(Ops::scale) Predecessor tensor has been deallocated.");
            }
            for (std::size_t i = 0; i < values.size(); i++) {
                values[i] = tensor->values()[i] * scalar;
            }
        };
        return result;
    }

    static std::shared_ptr<Tensor> transpose(std::shared_ptr<Tensor> tensor, std::size_t dim_1 = 0, std::size_t dim_2 = 1) {
        if (dim_1 >= tensor->ndim() || dim_2 >= tensor->ndim()) {
            throw std::invalid_argument("(Ops::transpose) Dimension indices are out of bounds.");
        }
        std::vector<std::size_t> new_shape = tensor->shape();
        std::swap(new_shape[dim_1], new_shape[dim_2]);
        std::shared_ptr<Tensor> result = std::make_shared<Tensor>(new_shape, 0.0f, tensor->requires_grad());
        
        for (std::size_t i = 0; i < tensor->size(); i++) {
            std::vector<std::size_t> original_indices(tensor->ndim());
            std::size_t remainder = i;
            for (std::size_t d = 0; d < tensor->ndim(); d++) {
                original_indices[d] = remainder / tensor->strides()[d];
                remainder = remainder % tensor->strides()[d];
            }
            std::swap(original_indices[dim_1], original_indices[dim_2]);
            std::size_t new_index = 0;
            for (std::size_t d = 0; d < tensor->ndim(); d++) {
                new_index += original_indices[d] * result->strides()[d];
            }
            result->values_[new_index] = tensor->values()[i];
        }
        std::vector<float> gradients(result->size(), 1.0f);
        result->add_predecessor(
            tensor,
            gradients,
            tensor->requires_grad(),
            [size = tensor->size()]() {
                return std::vector<float>(size, 1.0f);
            },
            [tensor_weak = std::weak_ptr<Tensor>(tensor), new_strides = result->strides(), dim_1, dim_2](
                std::vector<float>& pred_tensor_gradients,
                const std::vector<float>& current_gradients,
                const std::vector<float>& pred_struct_gradients
            ) -> void {
                std::shared_ptr<Tensor> tensor = tensor_weak.lock();
                if (!tensor) {
                    throw std::runtime_error("(Ops::transpose) Predecessor tensor has been deallocated.");
                }
                for (std::size_t i = 0; i < current_gradients.size(); i++) {
                    std::vector<std::size_t> transposed_indices(tensor->ndim());
                    std::size_t remainder = i;
                    for (std::size_t d = 0; d < tensor->ndim(); d++) {
                        transposed_indices[d] = remainder / new_strides[d];
                        remainder = remainder % new_strides[d];
                    }
                    std::swap(transposed_indices[dim_1], transposed_indices[dim_2]);
                    std::size_t original_index = 0;
                    for (std::size_t d = 0; d < tensor->ndim(); d++) {
                        original_index += transposed_indices[d] * tensor->strides()[d];
                    }
                    pred_tensor_gradients.at(original_index) += current_gradients.at(i);
                    // "* pred_struct_gradients.at(i)" is equivalent to "* 1", so omitted
                }
            }
        );
        result->forward_function_ = [new_strides = result->strides(), dim_1, dim_2](std::vector<float>& values, const std::vector<Predecessor>& preds) {
            const std::shared_ptr<Tensor> tensor = preds.at(0).tensor.lock();
            if (!tensor) {
                throw std::runtime_error("(Ops::transpose) Predecessor tensor has been deallocated.");
            }
            for (std::size_t i = 0; i < tensor->size(); i++) {
                std::vector<std::size_t> original_indices(tensor->ndim());
                std::size_t remainder = i;
                for (std::size_t d = 0; d < tensor->ndim(); d++) {
                    original_indices[d] = remainder / tensor->strides()[d];
                    remainder = remainder % tensor->strides()[d];
                }
                std::swap(original_indices[dim_1], original_indices[dim_2]);
                std::size_t new_index = 0;
                for (std::size_t d = 0; d < tensor->ndim(); d++) {
                    new_index += original_indices[d] * new_strides[d];
                }
                values[new_index] = tensor->values_[i];
            }
        };
        return result;
    }
};

#endif // OPS_HPP