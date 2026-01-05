# autograd
Reverse mode automatic differentiation implemented in C++
Supports the following features:
- Operations on n-dimensional tensors (add, element-wise multiply, scale, transpose)
- Forward pass (update values)
- Backward pass (update/accumulate gradients)
- Zero grad (set gradients and preceding gradients to 0)
- Togglable gradients (turn on/off gradient updates)

# Usage
- Compile `main.cpp`
```bash
make compile
```
- Run
```bash
make run
```

# How it works
This is written from my understanding, so some things may be inaccurate.

Suppose we have a "complicated" function like
```math
f(x) = 3x^2 + 5
```
Its derivative is easy to calculate by hand
```math
f'(x) = 6x
```
However, making a computer take a derivative of function implemented in code in this way would be very difficult. This is the motivation of automatic differentiation: people don't want to hardcode derivatives into their programs, especially for complicated functions that use multiple variables or matrices, like in machine learning.

The solution is to use the chain rule to separate one big function into many tiny functions that have simple derivates. This allows programmers to create the complicated functions by composing simple ones. In this example, we can write
```math
y_1 = f_1(x) = x^2 \newline
y_2 = f_2(y_1) = 3y_1 \newline
y_3 = f_3(y_2) = y_2 + 5 \newline
```
Therefore,
```math
f(x) = y_3 = f_3(f_2(f_1(x)))
```
By the chain rule:
```math
f'(x) = f_3'(f_2(f_1(x))) \cdot f_2'(f_1(x)) \cdot f_1'(x) \newline
= 1 \cdot 3 \cdot 2x \newline
= 6x
```
This makes a little more sense with Leibniz notation:
```math
f'(x) = \frac{dy_3}{dx} = \frac{dy_3}{dy_2} \cdot \frac{dy_2}{dy_1} \cdot \frac{dy_1}{dx}
```
Essentially, the derivative of $f$ with respect to $x$ can be calculated by multiplying the intermediate derivatives together, each of which being simple enough to do. In practice, we have functions of multiple variables, so partial derivatives are used instead, but the key idea is still the same.

There are a few unique implementations of automatic differentiation. Namely, forward mode ones and reverse mode ones.

"Forward mode" means calculating the derivatives during the forward pass. This means when we evaluate $f(x)$, our program also computes $f'(x)$ along the way. The first main way of doing this is by starting with $\frac{dx}{dx} = 1$ and using that to "seed" the chain rule. The other way, which I personally find quite interesting, is to use the relationship that the dual numbers have with derivatives. (Dual numbers are similar to complex numbers, where a dual number $x \in \mathbb{R}(\varepsilon)$ is represented by $x = a + b\varepsilon$ for $a,b \in \mathbb{R}$, and $\varepsilon$ has the property such that $\varepsilon^2=0$. This leads to the useful property that for any real function $f$, we have $f(x+\varepsilon) - f(x) = f'(x) \cdot \varepsilon$, which may look familiar if you remember the limit definition of the derivative.)

"Reverse mode" means calculating the derivatives separately, on a "backwards" pass. Note that the forward pass is still necessary before starting the backwards pass. It starts with the output gradient, and then passes that information on to prior variables (also using the chain rule). In this case, the "seed" is $\frac{dy_3}{dy_3} = 1$. The implementation is typically done with a DAG to represent the compuational graph, where the roots* are the inputs, and the leaves* are the outputs (*depending on how you think about it). Each node keeps track of its own value, derivative (wrt the final output), predecessors, and predecessors derivatives (wrt itself). For example, the node for $y_2$ would store $y_2, \frac{dy_2}{dx}, y_1, \frac{dy_1}{dy_2}$. During the backwards pass, the order of traversal is output, output's predecessors, the predecessors of those predecessors, etc.

Reverse mode is more used in machine learning, since it is more efficient when there are more input variables than output variables. In machine learning, it's very common to have only one output variable (the loss), but many input variables (the features). Forward mode is less efficient for this kind of task, making it far less suitable for machine learning purposes.