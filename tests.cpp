#include "tensor.hpp"
#include "ops.hpp"

#include <iostream>
#include <memory>
#include <stdexcept>
#include <vector>
#include <cstddef>
#include <cmath>

bool test_tensor_creation() {
    std::shared_ptr<Tensor> tensor_a = std::make_shared<Tensor>(std::vector<std::size_t>{3, 3}, 0.0f, true);
    if (tensor_a->shape() != std::vector<std::size_t>{3, 3}) {
        return false;
    }
    if (tensor_a->size() != 9) {
        return false;
    }
    for (std::size_t i = 0; i < tensor_a->size(); i++) {
        if (tensor_a->values()[i] != 0.0f) {
            return false;
        }
    }
    std::shared_ptr<Tensor> tensor_b = std::make_shared<Tensor>(std::vector<std::size_t>{2, 3, 1}, std::vector<float>{1.0, 2.0, 3.0, 4.0, 5.0, 6.0}, false);
    if (tensor_b->shape() != std::vector<std::size_t>{2, 3, 1}) {
        return false;
    }
    if (tensor_b->size() != 6) {
        return false;
    }
    for (std::size_t i = 0; i < tensor_b->size(); i++) {
        if (tensor_b->values()[i] != static_cast<float>(i + 1)) {
            return false;
        }
    }
    return true;
}

bool test_invalid_tensor_creation() {
    try {
        std::shared_ptr<Tensor> tensor = std::make_shared<Tensor>(std::vector<std::size_t>{3, 0}, 0.0f, true);
    } catch (const std::invalid_argument& e) {
        return true; // Exception was thrown as expected
    }
    return false; // No exception thrown, test failed
}

bool test_addition() {
    std::shared_ptr<Tensor> tensor_a = std::make_shared<Tensor>(std::vector<std::size_t>{2, 2, 4}, 1.0f, true);
    std::shared_ptr<Tensor> tensor_b = std::make_shared<Tensor>(std::vector<std::size_t>{2, 2, 4}, 2.0f, false);
    std::shared_ptr<Tensor> tensor_c = Ops::add(tensor_a, tensor_b);
    for (std::size_t i = 0; i < tensor_c->size(); i++) {
        if (tensor_c->values()[i] != 3.0f) {
            return false;
        }
    }
    return true;
}

bool test_invalid_addition() {
    try {
        std::shared_ptr<Tensor> tensor_a = std::make_shared<Tensor>(std::vector<std::size_t>{2, 3}, 1.0f, true);
        std::shared_ptr<Tensor> tensor_b = std::make_shared<Tensor>(std::vector<std::size_t>{3, 2}, 2.0f, false);
        std::shared_ptr<Tensor> tensor_c = Ops::add(tensor_a, tensor_b);
    } catch (const std::invalid_argument& e) {
        return true; // Exception was thrown as expected
    }
    return false; // No exception thrown, test failed
}

bool test_element_wise_multiply() {
    std::shared_ptr<Tensor> tensor_a = std::make_shared<Tensor>(std::vector<std::size_t>{2, 2}, 3.0f, true);
    std::shared_ptr<Tensor> tensor_b = std::make_shared<Tensor>(std::vector<std::size_t>{2, 2}, 2.0f, false);
    std::shared_ptr<Tensor> tensor_c = Ops::element_wise_multiply(tensor_a, tensor_b);
    for (std::size_t i = 0; i < tensor_c->size(); i++) {
        if (tensor_c->values()[i] != 6.0f) {
            return false;
        }
    }
    return true;
}

bool test_invalid_element_wise_multiply() {
    try {
        std::shared_ptr<Tensor> tensor_a = std::make_shared<Tensor>(std::vector<std::size_t>{2, 2}, 1.0f, true);
        std::shared_ptr<Tensor> tensor_b = std::make_shared<Tensor>(std::vector<std::size_t>{2, 3}, 2.0f, false);
        std::shared_ptr<Tensor> tensor_c = Ops::element_wise_multiply(tensor_a, tensor_b);
    } catch (const std::invalid_argument& e) {
        return true; // Exception was thrown as expected
    }
    return false; // No exception thrown, test failed
}

bool test_scale() {
    std::shared_ptr<Tensor> tensor_a = std::make_shared<Tensor>(std::vector<std::size_t>{2, 3}, 4.0f, true);
    float scale_factor = 2.5f;
    std::shared_ptr<Tensor> tensor_b = Ops::scale(tensor_a, scale_factor);
    for (std::size_t i = 0; i < tensor_b->size(); i++) {
        if (tensor_b->values()[i] != 10.0f) {
            return false;
        }
    }
    return true;
}

bool test_invalid_scale() {
    try {
        std::shared_ptr<Tensor> tensor_a = std::make_shared<Tensor>(std::vector<std::size_t>{2, 3}, 1.0f, true);
        float scale_factor = std::nanf("");
        std::shared_ptr<Tensor> tensor_b = Ops::scale(tensor_a, scale_factor);
    } catch (const std::invalid_argument& e) {
        return true; // Exception was thrown as expected
    }
    return false; // No exception thrown, test failed
}

int main() {
    std::vector<std::pair<std::string, bool (*)()>> tests = {
        {"Tensor Creation Test", test_tensor_creation},
        {"Addition Test", test_addition},
        {"Invalid Addition Test", test_invalid_addition},
        {"Element-wise Multiplication Test", test_element_wise_multiply},
        {"Invalid Element-wise Multiplication Test", test_invalid_element_wise_multiply},
        {"Scaling Test", test_scale},
        {"Invalid Scaling Test", test_invalid_scale}
    };
    for (const auto& [test_name, test_func] : tests) {
        bool result = test_func();
        std::cout << test_name << ": " << (result ? "PASSED" : "FAILED") << std::endl;
    }
    return 0;
}