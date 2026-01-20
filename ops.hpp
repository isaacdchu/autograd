#ifndef OPS_HPP
#define OPS_HPP

#include "tensor.hpp"

#include <algorithm>
#include <functional>
#include <iostream>
#include <memory>
#include <vector>
#include <stack>
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
        std::function<std::vector<float>()> grad_initializer = [size = result_data.size()]() {
            return std::vector<float>(size, 1.0f);
        };
        std::vector<float> grad_a;
        if (a->requires_grad()) {
            grad_a = grad_initializer();
        }
        result->add_predecessor(a, grad_a, a->requires_grad(), grad_initializer);
        std::vector<float> grad_b;
        if (b->requires_grad()) {
            grad_b = grad_initializer();
        }
        result->add_predecessor(b, grad_b, b->requires_grad(), grad_initializer);
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
        std::function<std::vector<float>()> grad_initializer = [size = result_data.size(), scalar]() {
            return std::vector<float>(size, scalar);
        };
        std::vector<float> gradients;
        if (tensor->requires_grad()) {
            gradients = grad_initializer();
        }
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

    static std::shared_ptr<Tensor> tensor_product(std::shared_ptr<Tensor> a, std::shared_ptr<Tensor> b, int contractions = 1) {
        if (contractions < 1 || contractions >= static_cast<int>(std::min(a->ndim(), b->ndim()))) {
            throw std::invalid_argument("(Ops::tensor_product) Number of contractions is out of valid range.");
        }
        std::vector<size_t> contraction_dims = std::vector<size_t>(a->shape().end() - contractions, a->shape().end());
        for (size_t i = 0; i < contraction_dims.size(); i++) {
            if (contraction_dims[i] != b->shape()[i]) {
                throw std::invalid_argument("(Ops::tensor_product) Contraction dimensions must match.");
            }
        }
        // a1 x a2 x a3 x ... x an * b1 x b2 x ... x bm
        // = a1 x a2 x ... x a(n-contractions) x b(1+contractions) x ... x bm
        std::vector<size_t> a_non_contraction_dims = std::vector<size_t>(a->shape().begin(), a->shape().end() - contractions);
        std::vector<size_t> b_non_contraction_dims = std::vector<size_t>(b->shape().begin() + contractions, b->shape().end());
        std::vector<std::size_t> result_shape(a_non_contraction_dims);
        result_shape.append_range(b_non_contraction_dims);
        std::size_t result_size = 1;
        for (std::size_t dim : result_shape) {
            result_size *= dim;
        }
        std::vector<float> result_data(result_size);
        std::function<float(const std::vector<std::size_t>&)> helper = [&](const std::vector<std::size_t>& indices) -> float {
            float sum = 0.0f;
            std::vector<std::size_t> a_indices = std::vector<std::size_t>(indices.begin(), indices.begin() + a->ndim() - contraction_dims.size());
            std::vector<std::size_t> b_indices = std::vector<std::size_t>(indices.begin() + a->ndim() - contraction_dims.size(), indices.end());
            std::vector<std::size_t> k_indices(contraction_dims.size(), 0);
            bool done = false;
            while (!done) {
                std::vector<std::size_t> full_a_indices = a_indices;
                full_a_indices.append_range(k_indices);
                std::vector<std::size_t> full_b_indices = k_indices;
                full_b_indices.append_range(b_indices);
                sum += a->operator()(full_a_indices) * b->operator()(full_b_indices);
                for (int d = contraction_dims.size() - 1; d >= 0; d--) {
                    k_indices[d]++;
                    if (k_indices[d] < contraction_dims[d]) {
                        break;
                    } else if (d == 0) {
                        done = true;
                    } else {
                        k_indices[d] = 0;
                    }
                }
            }
            return sum;
        };
        for (std::size_t i = 0; i < result_size; i++) {
            std::vector<std::size_t> indices = Tensor::unravel_index(i, result_shape);
            result_data[i] = helper(indices);
        }
        std::shared_ptr<Tensor> result = std::make_shared<Tensor>(result_shape, result_data, a->requires_grad() || b->requires_grad());
        // gradients aren't stored in predessor struct
        std::function<std::vector<float>()> grad_initializer = [a_size = a->size()]() -> std::vector<float> {
            return std::vector<float>();
        };
        auto compute_a = [contractions, result_shape](
            const std::vector<std::size_t>& indices,
            const std::vector<float>& current_gradients,
            const std::shared_ptr<Tensor>& other
        ) -> float {
            float sum = 0.0f;
            // compute the sum over non-contraction indices of b of (current_gradients * b tensor values)
            std::vector<std::size_t> k_indices(other->ndim() - contractions, 0);
            bool done = false;
            while (!done) {
                std::vector<std::size_t> full_grad_indices(indices.begin(), indices.end() - k_indices.size());
                full_grad_indices.append_range(k_indices);
                std::vector<std::size_t> full_other_indices(indices.end() - k_indices.size(), indices.end());
                full_other_indices.append_range(k_indices);
                std::size_t grad_flat_index = Tensor::ravel_index(full_grad_indices, result_shape);
                sum += current_gradients[grad_flat_index] * other->operator()(full_other_indices);
                for (int d = k_indices.size() - 1; d >= 0; d--) {
                    k_indices[d]++;
                    if (k_indices[d] < other->shape()[d + contractions]) {
                        break;
                    } else if (d == 0) {
                        done = true;
                    } else {
                        k_indices[d] = 0;
                    }
                }
            }
            return sum;
        };
        auto compute_b = [contractions, result_shape](
            const std::vector<std::size_t>& indices,
            const std::vector<float>& current_gradients,
            const std::shared_ptr<Tensor>& other
        ) -> float {
            float sum = 0.0f;
            // compute the sum over non-contraction indices of a of (a tensor values * current_gradients)
            std::vector<std::size_t> k_indices(other->ndim() - contractions, 0);
            std::vector<std::size_t> contraction_indices(indices.begin(), indices.begin() + contractions);
            std::vector<std::size_t> non_contraction_indices(indices.begin() + contractions, indices.end());
            bool done = false;
            while (!done) {
                std::vector<std::size_t> full_other_indices(k_indices);
                full_other_indices.append_range(contraction_indices);
                std::vector<std::size_t> full_grad_indices(k_indices);
                full_grad_indices.append_range(non_contraction_indices);
                std::size_t grad_flat_index = Tensor::ravel_index(full_grad_indices, result_shape);
                sum += current_gradients[grad_flat_index] * other->operator()(full_other_indices);
                for (int d = k_indices.size() - 1; d >= 0; d--) {
                    k_indices[d]++;
                    if (k_indices[d] < other->shape()[d]) {
                        break;
                    } else if (d == 0) {
                        done = true;
                    } else {
                        k_indices[d] = 0;
                    }
                }
            }
            return sum;
        };
        auto update_function_a = \
        [b_weak = std::weak_ptr<Tensor>(b), shape = a->shape(), contractions, result_shape = result->shape(), compute_a](
            std::vector<float>& pred_tensor_gradients,
            const std::vector<float>& current_gradients,
            const std::vector<float>& pred_struct_gradients
        ) -> void {
            // pred_tensor_gradients += sum over non-contractions of (current_gradients * other tensor values)
            std::shared_ptr<Tensor> b = b_weak.lock();
            if (!b) {
                throw std::runtime_error("(Ops::tensor_product) Predecessor tensor has been deallocated.");
            }
            for (std::size_t i = 0; i < pred_tensor_gradients.size(); i++) {
                std::vector<std::size_t> indices = Tensor::unravel_index(i, shape);
                pred_tensor_gradients[i] += compute_a(indices, current_gradients, b);
            }
        };
        auto update_function_b = \
        [a_weak = std::weak_ptr<Tensor>(a), shape = b->shape(), contractions, result_shape = result->shape(), compute_b](
            std::vector<float>& pred_tensor_gradients,
            const std::vector<float>& current_gradients,
            const std::vector<float>& pred_struct_gradients
        ) -> void {
            // pred_tensor_gradients += sum over non-contractions of (current_gradients * other tensor values)
            std::shared_ptr<Tensor> a = a_weak.lock();
            if (!a) {
                throw std::runtime_error("(Ops::tensor_product) Predecessor tensor has been deallocated.");
            }
            for (std::size_t i = 0; i < pred_tensor_gradients.size(); i++) {
                std::vector<std::size_t> indices = Tensor::unravel_index(i, shape);
                pred_tensor_gradients[i] += compute_b(indices, current_gradients, a);
            }
        };
        std::vector<float> grad_a;
        if (a->requires_grad()) {
            grad_a = grad_initializer();
        }
        result->add_predecessor(a, grad_a, a->requires_grad(), grad_initializer, update_function_a);
        std::vector<float> grad_b;
        if (b->requires_grad()) {
            grad_b = grad_initializer();
        }
        result->add_predecessor(b, grad_b, b->requires_grad(), grad_initializer, update_function_b);
        result->forward_function_ = [a_weak = std::weak_ptr<Tensor>(a), b_weak = std::weak_ptr<Tensor>(b), contractions, result_shape, helper](
            std::vector<float>& values,
            const std::vector<Predecessor>& preds
        ) -> void {
            std::shared_ptr<Tensor> a = a_weak.lock();
            std::shared_ptr<Tensor> b = b_weak.lock();
            if (!a || !b) {
                throw std::runtime_error("(Ops::tensor_product) Predecessor tensor has been deallocated.");
            }
            std::vector<size_t> contraction_dims = std::vector<size_t>(a->shape().end() - contractions, a->shape().end());
            for (size_t i = 0; i < contraction_dims.size(); i++) {
                if (contraction_dims[i] != b->shape()[i]) {
                    throw std::invalid_argument("(Ops::tensor_product) Contraction dimensions must match.");
                }
            }
            for (std::size_t i = 0; i < values.size(); i++) {
                std::vector<std::size_t> indices = Tensor::unravel_index(i, result_shape);
                values[i] = helper(indices);
            }
        };
        return result;
    }
};

#endif // OPS_HPP