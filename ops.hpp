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
        result->forward_function_ = [a_weak = std::weak_ptr<Tensor>(a), b_weak = std::weak_ptr<Tensor>(b), contractions, result_shape](
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
            std::function<float(const std::vector<std::size_t>&)> helper = [a, b, contraction_dims](const std::vector<std::size_t>& indices) -> float {
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
            for (std::size_t i = 0; i < values.size(); i++) {
                std::vector<std::size_t> indices = Tensor::unravel_index(i, result_shape);
                values[i] = helper(indices);
            }
        };
        return result;
    }

    enum class PaddingFill {
        ZERO,
        REPLICATE
    };
    static std::shared_ptr<Tensor> convolution_2d(
        std::shared_ptr<Tensor> input,
        std::shared_ptr<Tensor> kernel,
        std::size_t stride = 1,
        PaddingFill padding_fill = PaddingFill::ZERO
    ) {
        if (input->ndim() != 2) {
            throw std::invalid_argument("(Ops::convolution_2d) Input tensor must be 2D.");
        }
        if (kernel->ndim() != 2) {
            throw std::invalid_argument("(Ops::convolution_2d) Kernel tensor must be 2D.");
        }
        if (stride == 0) {
            throw std::invalid_argument("(Ops::convolution_2d) Stride must be a positive integer.");
        }
        const std::size_t input_height = input->shape()[input->ndim() - 2];
        const std::size_t input_width = input->shape()[input->ndim() - 1];
        std::vector<std::size_t> output_shape(input->shape());
        const std::size_t pad_height = (kernel->shape()[0] - 1) / 2;
        const std::size_t pad_width = (kernel->shape()[1] - 1) / 2;
        // make result tensor
        std::shared_ptr<Tensor> result = std::make_shared<Tensor>(output_shape, 0.0f, input->requires_grad() || kernel->requires_grad());
        std::function<float(std::size_t, std::size_t)> get_padded_value;
        switch (padding_fill) {
            case PaddingFill::ZERO:
                get_padded_value = [input, pad_height, pad_width, input_height, input_width](std::size_t i, std::size_t j) -> float {
                    if (i < pad_height || j < pad_width || i >= input_height + pad_height || j >= input_width + pad_width) {
                        return 0.0f;
                    }
                    return input->operator()({i - pad_height, j - pad_width});
                };
                break;
            case PaddingFill::REPLICATE:
                get_padded_value = [input, pad_height, pad_width, input_height, input_width](std::size_t i, std::size_t j) -> float {
                    std::size_t clamped_i = std::min(std::max(i, pad_height), input_height + pad_height - 1) - pad_height;
                    std::size_t clamped_j = std::min(std::max(j, pad_width), input_width + pad_width - 1) - pad_width;
                    return input->operator()({clamped_i, clamped_j});
                };
                break;
            default:
                throw std::invalid_argument("(Ops::convolution_2d) Invalid padding fill mode.");
        }
        auto convolve = [get_padded_value, kernel, stride](std::size_t out_i, std::size_t out_j) -> float {
            float sum = 0.0f;
            for (std::size_t k_i = 0; k_i < kernel->shape()[0]; k_i++) {
                for (std::size_t k_j = 0; k_j < kernel->shape()[1]; k_j++) {
                    const std::size_t in_i = out_i * stride + k_i;
                    const std::size_t in_j = out_j * stride + k_j;
                    sum += get_padded_value(in_i, in_j) * kernel->operator()({k_i, k_j});
                }
            }
            return sum;
        };
        for (std::size_t i = 0; i < result->shape()[result->shape().size() - 2]; i++) {
            for (std::size_t j = 0; j < result->shape()[result->shape().size() - 1]; j++) {
                result->values_[Tensor::ravel_index({i, j}, result->shape())] = convolve(i, j);
            }
        }
        // gradients aren't stored in predessor struct
        std::function<std::vector<float>()> grad_initializer = []() {
            return std::vector<float>();
        };
        std::vector<float> grad_input;
        if (input->requires_grad()) {
            grad_input = grad_initializer();
        }
        auto update_function_input = [
            kernel_weak = std::weak_ptr<Tensor>(kernel),
            input_height,
            input_width,
            output_shape
        ](
            std::vector<float>& pred_tensor_gradients,
            const std::vector<float>& current_gradients,
            const std::vector<float>& pred_struct_gradients
        ) -> void {
            // pred_tensor_gradients += convolution backpropagation w.r.t. input
            // (full convolution of current_gradients with 180 rotated kernel)
            std::shared_ptr<Tensor> kernel = kernel_weak.lock();
            if (!kernel) {
                throw std::runtime_error("(Ops::convolution_2d) Predecessor tensor has been deallocated.");
            }
            auto rotate_180 = [](const std::shared_ptr<Tensor>& kernel) -> std::shared_ptr<Tensor> {
                std::shared_ptr<Tensor> rotated = std::make_shared<Tensor>(kernel->shape(), 0.0f, false);
                for (std::size_t i = 0; i < kernel->shape()[0]; i++) {
                    for (std::size_t j = 0; j < kernel->shape()[1]; j++) {
                        const std::size_t x = kernel->shape()[0] - 1 - i;
                        const std::size_t y = kernel->shape()[1] - 1 - j;
                        rotated->values_.at(Tensor::ravel_index({x, y}, kernel->shape())) = kernel->operator()({i, j});
                    }
                }
                return rotated;
            };
            std::shared_ptr<Tensor> rotated_kernel = rotate_180(kernel);
            auto convolve = [output_shape, current_gradients, rotated_kernel](std::size_t in_i, std::size_t in_j) -> float {
                float sum = 0.0f;
                const std::size_t output_height = output_shape[output_shape.size() - 2];
                const std::size_t output_width = output_shape[output_shape.size() - 1];
                for (std::size_t out_i = 0; out_i < output_height; out_i++) {
                    for (std::size_t out_j = 0; out_j < output_width; out_j++) {
                        const std::size_t k_i = in_i + rotated_kernel->shape()[0] - 1 - out_i * 1;
                        const std::size_t k_j = in_j + rotated_kernel->shape()[1] - 1 - out_j * 1;
                        if (k_i < rotated_kernel->shape()[0] && k_j < rotated_kernel->shape()[1]) {
                            sum += current_gradients.at(Tensor::ravel_index({out_i, out_j}, {output_height, output_width})) * rotated_kernel->operator()({k_i, k_j});
                        }
                    }
                }
                return sum;
            };
            for (std::size_t in_i = 0; in_i < input_height; in_i++) {
                for (std::size_t in_j = 0; in_j < input_width; in_j++) {
                    pred_tensor_gradients.at(Tensor::ravel_index({in_i, in_j}, {input_height, input_width})) += convolve(in_i, in_j);
                }
            }
        };
        result->add_predecessor(input, grad_input, input->requires_grad(), grad_initializer, update_function_input);
        // gradients aren't stored in predecessor struct
        std::vector<float> grad_kernel;
        if (kernel->requires_grad()) {
            grad_kernel = grad_initializer();
        }
        auto update_function_kernel = [
            input_weak = std::weak_ptr<Tensor>(input),
            kernel_height = kernel->shape()[0],
            kernel_width = kernel->shape()[1],
            padding_fill,
            pad_height,
            pad_width,
            input_height,
            input_width,
            output_shape,
            stride
        ](
            std::vector<float> &pred_tensor_gradients,
            const std::vector<float> &current_gradients,
            const std::vector<float> &pred_struct_gradients
        ) -> void {
            // pred_tensor_gradients += convolution backpropagation w.r.t. kernel
            // (valid convolution of input with current_gradients)
            std::shared_ptr<Tensor> input = input_weak.lock();
            if (!input) {
                throw std::runtime_error("(Ops::convolution_2d) Predecessor tensor has been deallocated.");
            }
            std::function<float(std::size_t, std::size_t)> get_padded_value;
            switch (padding_fill) {
                case PaddingFill::ZERO:
                    get_padded_value = [input, pad_height, pad_width, input_height, input_width](std::size_t i, std::size_t j) -> float {
                        if (i < pad_height || j < pad_width || i >= input_height + pad_height || j >= input_width + pad_width) {
                            return 0.0f;
                        }
                        return input->operator()({i - pad_height, j - pad_width});
                    };
                    break;
                case PaddingFill::REPLICATE:
                    get_padded_value = [input, pad_height, pad_width, input_height, input_width](std::size_t i, std::size_t j) -> float {
                        std::size_t clamped_i = std::min(std::max(i, pad_height), input_height + pad_height - 1) - pad_height;
                        std::size_t clamped_j = std::min(std::max(j, pad_width), input_width + pad_width - 1) - pad_width;
                        return input->operator()({clamped_i, clamped_j});
                    };
                    break;
                default:
                    throw std::invalid_argument("(Ops::convolution_2d) Invalid padding fill mode.");
            }
            auto convolve = [output_shape, stride, current_gradients, get_padded_value](std::size_t k_i, std::size_t k_j) -> float {
                float sum = 0.0f;
                const std::size_t output_height = output_shape[output_shape.size() - 2];
                const std::size_t output_width = output_shape[output_shape.size() - 1];
                for (std::size_t out_i = 0; out_i < output_height; out_i++) {
                    for (std::size_t out_j = 0; out_j < output_width; out_j++) {
                        const std::size_t in_i = out_i * stride + k_i;
                        const std::size_t in_j = out_j * stride + k_j;
                        sum += get_padded_value(in_i, in_j) * current_gradients.at(Tensor::ravel_index({out_i, out_j}, {output_height, output_width}));
                    }
                }
                return sum;
            };
            for (std::size_t k_i = 0; k_i < kernel_height; k_i++) {
                for (std::size_t k_j = 0; k_j < kernel_width; k_j++) {
                    pred_tensor_gradients.at(Tensor::ravel_index({k_i, k_j}, {kernel_height, kernel_width})) += convolve(k_i, k_j);
                }
            }
        };
        result->add_predecessor(kernel, grad_kernel, kernel->requires_grad(), grad_initializer, update_function_kernel);
        result->forward_function_ = [output_shape, padding_fill, pad_height, pad_width, stride](
            std::vector<float>& values,
            const std::vector<Predecessor>& preds
        ) -> void {
            const std::shared_ptr<Tensor> input = preds.at(0).tensor.lock();
            const std::shared_ptr<Tensor> kernel = preds.at(1).tensor.lock();
            if (!input || !kernel) {
                throw std::runtime_error("(Ops::convolution_2d) Predecessor tensor has been deallocated.");
            }
            std::function<float(std::size_t, std::size_t)> get_padded_value;
            switch (padding_fill) {
                case PaddingFill::ZERO:
                    get_padded_value = [input, pad_height, pad_width](std::size_t i, std::size_t j) -> float {
                        if (i < pad_height || j < pad_width || i >= input->shape()[0] + pad_height || j >= input->shape()[1] + pad_width) {
                            return 0.0f;
                        }
                        return input->operator()({i - pad_height, j - pad_width});
                    };
                    break;
                case PaddingFill::REPLICATE:
                    get_padded_value = [input, pad_height, pad_width](std::size_t i, std::size_t j) -> float {
                        std::size_t clamped_i = std::min(std::max(i, pad_height), input->shape()[0] + pad_height - 1) - pad_height;
                        std::size_t clamped_j = std::min(std::max(j, pad_width), input->shape()[1] + pad_width - 1) - pad_width;
                        return input->operator()({clamped_i, clamped_j});
                    };
                    break;
                default:
                    throw std::invalid_argument("(Ops::convolution_2d) Invalid padding fill mode.");
            }
            auto convolve = [get_padded_value, kernel, stride](std::size_t out_i, std::size_t out_j) -> float {
                float sum = 0.0f;
                for (std::size_t k_i = 0; k_i < kernel->shape()[0]; k_i++) {
                    for (std::size_t k_j = 0; k_j < kernel->shape()[1]; k_j++) {
                        const std::size_t in_i = out_i * stride + k_i;
                        const std::size_t in_j = out_j * stride + k_j;
                        sum += get_padded_value(in_i, in_j) * kernel->operator()({k_i, k_j});
                    }
                }
                return sum;
            };
            for (std::size_t i = 0; i < output_shape[output_shape.size() - 2]; i++) {
                for (std::size_t j = 0; j < output_shape[output_shape.size() - 1]; j++) {
                    values.at(Tensor::ravel_index({i, j}, output_shape)) = convolve(i, j);
                }
            }
        };
        return result;
    }
};

#endif // OPS_HPP