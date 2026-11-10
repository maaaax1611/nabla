---
title: Autograd
desc: How nabla records a computation as a graph and walks it backwards with the chain rule to get every gradient in one pass.
tags: [Chain rule, Computational graph, Backward, Broadcasting, Gradient check]
order: 1
eyebrow: Core · Reverse-mode differentiation
footer: The figures use the scalar example <code>d = a * b</code>, <code>e = d + a</code>, <code>L = e ** 2</code>. Code excerpts are from <a href="https://github.com/maaaax1611/nabla/blob/main/src/nabla/tensor.py"><code>tensor.py</code></a> and <a href="https://github.com/maaaax1611/nabla/blob/main/src/nabla/function.py"><code>function.py</code></a>, shortened to the essentials.
---

Training a network means following the gradient of one number, the loss, with respect to millions of parameters. Autograd computes all of these derivatives automatically: it records what the program does on the way forward and then applies the chain rule on the way back. This page covers the two classes that carry the mechanism, the backward pass itself, and how operations plug into it.

## One output, many inputs {#why group="Idea" toc="Why reverse mode"}

A loss $L$ depends on parameters $\theta_1, \dots, \theta_n$ through a long chain of operations. There are two ways to push the chain rule through such a chain:

- **Forward mode** fixes one input and carries $\partial(\cdot)/\partial\theta_i$ along with every intermediate value. One pass gives the derivative of everything with respect to *one* parameter, so $n$ parameters need $n$ passes.
- **Reverse mode** fixes the output and carries $\partial L/\partial(\cdot)$ backwards. One pass gives the derivative of *one* output with respect to everything.

A network has one loss and many parameters, so reverse mode is the right direction: all gradients cost about as much as a few evaluations of the function itself. The price is memory, because the backward pass needs the intermediate values of the forward pass.

> [!abstract] Key points
> - Reverse mode: one backward pass yields $\partial L/\partial\theta_i$ for all $i$.
> - It needs the forward pass recorded, values included.

## Tensors and functions {#graph toc="Tensor and Function"}

nabla records the computation while it runs. Two classes are enough:

- A **`Tensor`** wraps an array (`data`) and has a slot for its gradient (`grad`). If it was computed by an operation, it also remembers which one (`_ctx`) and from which tensors (`_prev`).
- A **`Function`** is one differentiable operation. `forward` computes the result, `backward` turns the gradient of the output into the gradients of the inputs.

Every operation runs through one method, `Function.apply`:

```python
@classmethod
def apply(cls, *inputs, **kwargs):
    ctx = cls(**kwargs)
    out = Tensor(ctx.forward(*inputs))
    if _grad_enabled and any(t.requires_grad for t in inputs):
        out._ctx = ctx          # the operation that produced out
        out._prev = inputs      # the tensors it was computed from
        out.requires_grad = True
    return out
```

`a * b` calls `Multiply.apply(a, b)`. The result is an ordinary tensor that additionally points back at its inputs. A chain of operations therefore leaves behind a **computational graph**: tensors as nodes, connected through the operations that produced them. Tensors without `_ctx` are the **leaves**, typically inputs and parameters.

::: widget ag-01-forward

The graph is built on the fly, in the order in which Python executes the lines (*define-by-run*). Loops and `if` statements need no special treatment: each run records exactly the operations it performed.

## The chain rule on a graph {#chain-rule group="Backward pass" toc="Chain rule"}

For every tensor $y$ in the graph we want $\partial L/\partial y$. If $y$ is used by several operations with results $z_1, z_2, \dots$, the multivariable chain rule adds one term per use:

$$
\frac{\partial L}{\partial y} = \sum_{k} \frac{\partial L}{\partial z_k}\,\frac{\partial z_k}{\partial y}
$$

Each term has two factors. $\partial L/\partial z_k$ is the gradient of the result, which arrives from further down the graph. $\partial z_k/\partial y$ is the **local derivative** of a single operation and depends only on that operation and its inputs. An operation never needs to know what the rest of the network looks like.

In the example graph, `a` feeds into both `d` and `e`:

$$
\frac{\partial L}{\partial a}
= \frac{\partial L}{\partial e}\cdot\underbrace{\frac{\partial e}{\partial a}}_{1}
+ \frac{\partial L}{\partial d}\cdot\underbrace{\frac{\partial d}{\partial a}}_{b}
$$

> [!abstract] Key points
> - Gradient of a tensor = sum over its uses of (gradient of the result) × (local derivative).
> - Local derivatives are a property of one operation alone.

## Walking the graph backwards {#backward toc="backward()"}

The sum above is only complete once *every* use of $y$ has delivered its term. The tensors must therefore be processed in an order in which each result comes before its inputs. That is a **reverse topological order** of the graph, and a depth-first search produces it:

```python
def backward(self):
    self.grad = ones_like(self.data)          # dL/dL = 1

    topo, visited = [], set()
    def visit(t):
        if id(t) not in visited:
            visited.add(id(t))
            for parent in t._prev:
                visit(parent)
            topo.append(t)                    # after all of its inputs
    visit(self)

    for t in reversed(topo):                  # results before inputs
        if t._ctx:
            grads = t._ctx.backward(t.grad)
            for parent, g in zip(t._prev, grads):
                if parent.requires_grad:
                    parent.grad = g if parent.grad is None else parent.grad + g
```

The last line **adds** instead of assigning. That is the sum of the chain rule: a tensor that is used twice receives two contributions.

::: widget ag-02-backward

Two consequences of accumulating:

- Gradients of the leaves survive the call. A training loop has to reset them before the next backward pass (`zero_grad()`), otherwise the gradients of successive steps pile up.
- After `backward()` nabla drops `_ctx` and `_prev` of every tensor it visited. The intermediate values are no longer needed and their memory is released at once. A second `backward()` through the same graph is therefore not possible; the next forward pass builds a new one.

The implementation in the repository performs the same traversal with an explicit stack instead of recursion, which also works for graphs deeper than Python's recursion limit.

## Writing an operation {#functions group="Operations" toc="Operations"}

An operation implements two methods. `forward` receives the input tensors and returns an array. `backward` receives $\partial L/\partial\text{out}$ and returns one array per input, $\partial L/\partial\text{input}$, in the same order:

```python
class Multiply(Function):
    def forward(self, x, y):
        self.save_for_backward(x, y)
        return x.data * y.data

    def backward(self, grad_output):
        x, y = self.saved_tensors
        return grad_output * y.data, grad_output * x.data
```

`backward` never returns the local derivative on its own. It returns the product with the incoming gradient, which keeps every result in the shape of the corresponding input and avoids building Jacobian matrices. What an operation saves for the backward pass is its own choice: `Multiply` needs both inputs, `Exp` only its own output.

| Operation | Forward | `backward(g)` returns |
| --- | --- | --- |
| `Add` | $x + y$ | $g,\; g$ |
| `Multiply` | $x \cdot y$ | $g\,y,\; g\,x$ |
| `Divide` | $x / y$ | $g / y,\; -g\,x / y^2$ |
| `Pow` | $x^p$, constant $p$ | $g\,p\,x^{p-1}$ |
| `Exp` | $e^x$ | $g\,e^x$ |
| `Log` | $\ln x$ | $g / x$ |
| `MatMul` | $A B$ | $g\,B^\top,\; A^\top g$ |
| `Sum` | $\sum_i x_i$ | $g$ copied to every $x_i$ |

Python numbers are accepted as operands: `x * 2` and `1 - x` wrap the constant in a tensor with the dtype and device of `x`.

## Broadcasting {#broadcasting}

NumPy lets `x + b` combine shapes such as (4, 3) and (3,): `b` is copied along the missing axis. For the gradient this means that one entry of `b` has influenced several entries of the output. By the chain rule its gradient is the **sum** over all of them.

::: widget ag-03-broadcast

Every elementwise operation therefore passes its gradients through `unbroadcast`, which sums over the axes that were added or stretched until the shape of the input is restored:

```python
def unbroadcast(grad, target_shape):
    while grad.ndim > len(target_shape):          # axes that were added in front
        grad = grad.sum(axis=0)
    for axis, (g, t) in enumerate(zip(grad.shape, target_shape)):
        if t == 1 and g != 1:                     # axes of size 1 that were stretched
            grad = grad.sum(axis=axis, keepdims=True)
    return grad
```

A reduction is the mirror image: `x.sum(axis=0)` sums forward, so backward its gradient is *copied* along that axis. `x.mean(axis=0)` does the same with a factor $1/n$.

> [!abstract] Key points
> - Copied forward → summed backward (broadcasting).
> - Summed forward → copied backward (`sum`, `mean`).
> - The gradient of a tensor always has the shape of the tensor.

## Composing and fusing {#composition toc="Compose or fuse"}

Not every function needs its own `Function`. Anything that can be written with existing operations gets its backward pass for free. Softmax in three operations:

```python
e = x.exp()
p = e / e.sum(axis=-1, keepdims=True)
```

nabla nevertheless ships a hand-written `Softmax`. The composed version computes $e^{x}$ directly and overflows for large inputs; the fused operation can subtract the row maximum first, and its backward pass is a single expression instead of three chained ones. The rule of thumb: compose by default, fuse when it buys numerical stability or a much simpler gradient.

## Switching recording off {#no-grad toc="no_grad"}

Recording only pays off if `backward()` is called afterwards. During evaluation it is not, and each recorded operation would keep its inputs alive for nothing. Inside `no_grad` the results are plain tensors without `_ctx` and `_prev`:

```python
with nabla.no_grad():
    logits = model(x)
```

The same check in `Function.apply` skips recording whenever no input requires a gradient, so arithmetic on plain data never builds a graph.

## Checking a gradient {#gradcheck group="Testing" toc="Gradient check"}

A wrong `backward` rarely crashes. The network just trains worse. Each operation is therefore tested against a derivative obtained without any calculus, the **central difference**:

$$
\frac{\partial f}{\partial x_i} \approx \frac{f(x + h\,e_i) - f(x - h\,e_i)}{2h}
$$

Its truncation error falls with $h^2$, compared with $h$ for the one-sided quotient $\bigl(f(x+h) - f(x)\bigr)/h$. But $h$ cannot be made arbitrarily small: the numerator is the difference of two nearly equal floating-point numbers, and its rounding error is divided by $h$.

::: widget ag-04-gradcheck

The tests perturb one input element at a time with $h = 10^{-5}$ in double precision and compare with the result of `backward` (`np.allclose`). The check needs two forward passes per element, which is why it is used on small tensors in tests and not for training.

> [!abstract] Key points
> - Central difference: error $\propto h^2$, limited by rounding for small $h$.
> - Run gradient checks in `float64`; single precision leaves too few digits.
