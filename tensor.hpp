#ifndef TENSOR_HPP
#define TENSOR_HPP

#include <functional>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <vector>
#include <cstddef>

class Tensor;

struct Predecessor {
    std::weak_ptr<Tensor> tensor;
    std::vector<float> gradients;
    bool requires_grad;
    std::function<std::vector<float>()> gradient_initializer;
    std::function<void(std::vector<float>&, const std::vector<float>&, const std::vector<float>&)> update_function;
    Predecessor(
        std::weak_ptr<Tensor> tensor,
        std::vector<float> gradients,
        bool requires_grad,
        std::function<std::vector<float>()> gradient_initializer,
        std::function<void(std::vector<float>&, const std::vector<float>&, const std::vector<float>&)> update_function
    ) : tensor(tensor),
        gradients(gradients),
        requires_grad(requires_grad),
        gradient_initializer(gradient_initializer),
        update_function(update_function)
    {}
};

class Tensor : public std::enable_shared_from_this<Tensor> {
friend class Ops;
friend class Optimizer;
friend class Loss;
private:
    std::vector<std::size_t> shape_;
    std::size_t ndim_;
    std::size_t size_;
    std::vector<float> values_;
    std::vector<float> gradients_;
    std::vector<std::size_t> strides_;
    bool requires_grad_;
    std::vector<Predecessor> predecessors_;
    std::vector<std::weak_ptr<Tensor>> successors_;
    std::vector<std::shared_ptr<Tensor>> backward_list_;
    std::function<void(std::vector<float>&, const std::vector<Predecessor>&)> forward_function_;

    void add_predecessor(
        std::shared_ptr<Tensor> tensor,
        const std::vector<float>& gradients,
        bool requires_grad,
        std::function<std::vector<float>()> gradient_initializer,
        std::function<void(std::vector<float>&, const std::vector<float>&, const std::vector<float>&)> update_function = std::bind(&Tensor::default_update_function, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3)
    ) {
        predecessors_.emplace_back(tensor, gradients, requires_grad, gradient_initializer, update_function);
        tensor->successors_.emplace_back(shared_from_this());
    }

    void compute_size() {
        size_ = 1;
        for (std::size_t dim : shape_) {
            if (dim <= 0) {
                throw std::invalid_argument("(Tensor::compute_size) Tensor dimensions must be positive.");
            }
            size_ *= dim;
        }
    }

    void compute_strides(const std::vector<std::size_t>& shape) {
        std::size_t stride = 1;
        for (std::size_t i = shape.size(); i-- > 0;) {
            strides_[i] = stride;
            stride *= shape[i];
        }
    }

    void create_backward_list() {
        std::unordered_set<std::shared_ptr<Tensor>> visited;
        std::unordered_set<std::shared_ptr<Tensor>> temp_mark;
        backward_list_.clear();
        std::function<void(std::shared_ptr<Tensor>)> dfs = [&](std::shared_ptr<Tensor> tensor) {
            if (visited.contains(tensor)) {
                return;
            }
            if (temp_mark.contains(tensor)) {
                throw std::runtime_error("(Tensor::create_backward_list) Cycle detected in the computation graph.");
            }

            temp_mark.insert(tensor);
            for (const Predecessor& pred : tensor->predecessors_) {
                if (std::shared_ptr<Tensor> pred_tensor = pred.tensor.lock()) {
                    dfs(pred_tensor);
                }
            }
            temp_mark.erase(tensor);
            visited.insert(tensor);
            backward_list_.push_back(tensor);
        };
        dfs(shared_from_this());
    }

    void default_forward_function(std::vector<float>& values, const std::vector<Predecessor>& preds) const {
        // Default forward function does nothing
    }

    static void default_update_function(
        std::vector<float>& pred_tensor_gradients,
        const std::vector<float>& current_gradients,
        const std::vector<float>& pred_struct_gradients
    ) {
        for (std::size_t i = 0; i < current_gradients.size(); i++) {
            pred_tensor_gradients.at(i) += current_gradients.at(i) * pred_struct_gradients.at(i);
        }
    }

    static bool vectors_are_equal(const std::vector<std::size_t>& a, const std::vector<std::size_t>& b) {
        if (a.size() != b.size()) {
            return false;
        }
        for (std::size_t i = 0; i < a.size(); i++) {
            if (a[i] != b[i]) {
                return false;
            }
        }
        return true;
    }

public:
    Tensor(const std::vector<std::size_t>& shape, float init_value = 0.0f, bool requires_grad = true)
        : shape_(shape), requires_grad_(requires_grad), successors_() {
        compute_size();
        strides_.resize(shape.size());
        compute_strides(shape);
        values_.resize(size_, init_value);
        gradients_ = std::vector<float>(size_);
        ndim_ = shape.size();
        forward_function_ = std::bind(&Tensor::default_forward_function, this, std::placeholders::_1, std::placeholders::_2);
    }

    Tensor(const std::vector<std::size_t>& shape, const std::vector<float>& init_data, bool requires_grad = true)
        : shape_(shape), values_(init_data), requires_grad_(requires_grad), successors_() {
        compute_size();
        if (init_data.size() != size_) {
            throw std::invalid_argument("(Tensor Constructor) Initial data size does not match tensor size.");
        }
        strides_.resize(shape.size());
        compute_strides(shape);
        gradients_ = std::vector<float>(size_);
        ndim_ = shape.size();
        forward_function_ = std::bind(&Tensor::default_forward_function, this, std::placeholders::_1, std::placeholders::_2);
    }

    float operator()(const std::vector<std::size_t>& indices) const {
        if (indices.size() != ndim_) {
            throw std::invalid_argument("(Tensor::operator()) Number of indices must match tensor dimensions.");
        }
        std::size_t flat_index = 0;
        for (std::size_t i = 0; i < ndim_; ++i) {
            flat_index += indices[i] * strides_[i];
        }
        if (flat_index >= values_.size()) {
            throw std::out_of_range("(Tensor::operator()) Index out of bounds.");
        }
        return values_[flat_index];
    }

    void set_values(const std::vector<float>& vals) {
        if (vals.size() != size_) {
            throw std::invalid_argument("(Tensor::set_values) Input size does not match tensor size.");
        }
        values_ = vals;
    }

    void set_values(std::vector<float>&& vals) {
        if (vals.size() != size_) {
            throw std::invalid_argument("(Tensor::set_values) Input size does not match tensor size.");
        }
        values_ = std::move(vals);
    }

    void set_values(const Tensor& val_tensor) {
        if (val_tensor.values_.size() != size_) {
            throw std::invalid_argument("(Tensor::set_values) Input size does not match tensor size.");
        }
        values_ = val_tensor.values();
    }

    void set_values(const std::shared_ptr<Tensor>& val_tensor) {
        if (val_tensor->values_.size() != size_) {
            throw std::invalid_argument("(Tensor::set_values) Input size does not match tensor size.");
        }
        values_ = val_tensor->values();
    }

    void set_gradients(const std::vector<float>& grads) {
        if (!requires_grad_) {
            throw std::runtime_error("(Tensor::set_gradients) Cannot set gradients on a tensor that does not require gradients.");
        }
        if (grads.size() != size_) {
            throw std::invalid_argument("(Tensor::set_gradients) Gradient size does not match tensor size.");
        }
        gradients_ = grads;
    }

    void set_gradients(std::vector<float>&& grads) {
        if (!requires_grad_) {
            throw std::runtime_error("(Tensor::set_gradients) Cannot set gradients on a tensor that does not require gradients.");
        }
        if (grads.size() != size_) {
            throw std::invalid_argument("(Tensor::set_gradients) Gradient size does not match tensor size.");
        }
        gradients_ = std::move(grads);
    }

    void set_gradients(const Tensor& grad_tensor) {
        if (!requires_grad_) {
            throw std::runtime_error("(Tensor::set_gradients) Cannot set gradients on a tensor that does not require gradients.");
        }
        if (vectors_are_equal(grad_tensor.shape_, shape_) == false) {
            throw std::invalid_argument("(Tensor::set_gradients) Gradient shape does not match tensor shape.");
        }
        gradients_ = grad_tensor.values();
    }

    void set_gradients(const std::shared_ptr<Tensor>& grad_tensor) {
        if (!requires_grad_) {
            throw std::runtime_error("(Tensor::set_gradients) Cannot set gradients on a tensor that does not require gradients.");
        }
        if (vectors_are_equal(grad_tensor->shape_, shape_) == false) {
            throw std::invalid_argument("(Tensor::set_gradients) Gradient shape does not match tensor shape.");
        }
        gradients_ = grad_tensor->values();
    }

    void set_requires_grad(bool requires_grad) {
        if (requires_grad_ == requires_grad) {
            // No changes needed
            return;
        }
        if (requires_grad_ && !requires_grad) {
            // requires_grad_ changed from true to false
            gradients_.clear();
            gradients_.shrink_to_fit();
        } else {
            // requires_grad_ changed from false to true
            gradients_.resize(size_, 0.0f);
        }
        requires_grad_ = requires_grad;
        for (const std::weak_ptr<Tensor>& succ_weak : successors_) {
            std::shared_ptr<Tensor> succ = succ_weak.lock();
            if (!succ) {
                throw std::runtime_error("(Tensor::set_requires_grad) Successor tensor has been deallocated.");
            }
            for (Predecessor& pred : succ->predecessors_) {
                std::shared_ptr<Tensor> pred_tensor = pred.tensor.lock();
                if (!pred_tensor) {
                    throw std::runtime_error("(Tensor::set_requires_grad) Predecessor tensor has been deallocated.");
                }
                if (pred_tensor.get() != this) {
                    continue;
                }
                if (pred.requires_grad == requires_grad) {
                    continue;
                }
                pred.requires_grad = requires_grad;
                pred.gradients = pred.gradient_initializer();
            }
            if (requires_grad) {
                succ->set_requires_grad(requires_grad);
            }
        }
    }

    void forward() {
        if (backward_list_.empty()) {
            create_backward_list();
        }
        for (std::shared_ptr<Tensor> tensor : backward_list_) {
            tensor->forward_function_(tensor->values_, tensor->predecessors());
        }
    }

    void backward() {
        if (!requires_grad_) {
            throw std::runtime_error("(Tensor::backward) Cannot perform backward on a tensor that does not require gradients.");
        }
        if (backward_list_.empty()) {
            create_backward_list();
        }
        for (auto it = backward_list_.rbegin(); it != backward_list_.rend(); ++it) {
            std::shared_ptr<Tensor> tensor = *it;
            for (const Predecessor& pred : tensor->predecessors()) {
                if (!pred.requires_grad) {
                    continue;
                }
                std::shared_ptr<Tensor> pred_tensor = pred.tensor.lock();
                if (!pred_tensor) {
                    throw std::runtime_error("(Tensor::backward) Predecessor tensor has been deallocated.");
                }
                // if (pred.gradients.empty()) {
                //     std::cerr << "YOU HAVE A BUG IN gradients_initializer FUNCTION!" << std::endl;
                //     throw std::runtime_error("(Tensor::backward) Predecessor gradients are not initialized.");
                // }
                if (pred_tensor->gradients_.size() != pred_tensor->size()) {
                    pred_tensor->gradients_.resize(pred_tensor->size());
                }
                if (tensor->gradients_.size() != tensor->size_) {
                    tensor->gradients_.resize(tensor->size_, 0.0f);
                }
                // std::cout << "calling update_function for predecessor tensor: " << pred_tensor->to_string() << std::endl;
                pred.update_function(pred_tensor->gradients_, tensor->gradients(), pred.gradients);
                // std::cout << "success" << std::endl;
            }
        }
    }

    void zero_grad() {
        if (!requires_grad_) {
            throw std::runtime_error("(Tensor::zero_grad) Cannot perform zero_grad on a tensor that does not require gradients.");
        }
        gradients_ = std::vector<float>(size_);
        if (backward_list_.empty()) {
            create_backward_list();
        }
        for (std::shared_ptr<Tensor> tensor : backward_list_) {
            for (const Predecessor& pred : tensor->predecessors()) {
                if (!pred.requires_grad) {
                    continue;
                }
                std::shared_ptr<Tensor> pred_tensor = pred.tensor.lock();
                if (!pred_tensor) {
                    throw std::runtime_error("(Tensor::zero_grad) Predecessor tensor has been deallocated.");
                }
                pred_tensor->gradients_ = std::vector<float>();
            }
        }
    }

    const std::vector<std::size_t>& shape() const {
        return shape_;
    }

    std::size_t ndim() const {
        return ndim_;
    }

    std::size_t size() const {
        return size_;
    }

    const std::vector<float>& values() const {
        return values_;
    }

    const std::vector<float>& gradients() const {
        return gradients_;
    }

    const std::vector<std::size_t>& strides() const {
        return strides_;
    }

    bool requires_grad() const {
        return requires_grad_;
    }

    const std::vector<Predecessor>& predecessors() const {
        return predecessors_;
    }

    std::string to_string() const {
        std::string repr = "Tensor(\n\tshape=[";
        for (std::size_t i = 0; i < shape_.size(); ++i) {
            repr += std::to_string(shape_[i]);
            if (i < shape_.size() - 1) {
                repr += ", ";
            }
        }
        repr += "],\n\tvalues=[";
        for (std::size_t i = 0; i < values_.size(); ++i) {
            repr += std::to_string(values_[i]);
            if (i < values_.size() - 1) {
                repr += ", ";
            }
        }
        repr += "],\n\tgradients=[";
        for (std::size_t i = 0; i < gradients_.size(); ++i) {
            repr += std::to_string(gradients_[i]);
            if (i < gradients_.size() - 1) {
                repr += ", ";
            }
        }
        repr += "],\n\trequires_grad=" + std::string(requires_grad_ ? "true" : "false") + "\n)";
        return repr;
    }

    static std::size_t ravel_index(const std::vector<std::size_t>& indices, const std::vector<std::size_t>& shape) {
        if (indices.size() != shape.size()) {
            throw std::invalid_argument("(Tensor::ravel_index) Number of indices must match shape dimensions.");
        }
        std::size_t flat_index = 0;
        std::size_t stride = 1;
        for (std::size_t i = shape.size(); i-- > 0;) {
            flat_index += indices[i] * stride;
            stride *= shape[i];
        }
        if (flat_index >= stride) {
            throw std::out_of_range("(Tensor::ravel_index) Index out of bounds.");
        }
        return flat_index;
    }

    static std::vector<std::size_t> unravel_index(std::size_t index, const std::vector<std::size_t>& shape) {
        std::vector<std::size_t> indices(shape.size());
        for (std::size_t i = shape.size(); i-- > 0;) {
            indices[i] = index % shape[i];
            index /= shape[i];
        }
        return indices;
    }
};

#endif // TENSOR_HPP