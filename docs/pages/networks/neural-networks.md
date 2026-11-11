---
title: Neural networks
desc: From a single linear layer to a trained multilayer perceptron — modules, activations, initialization and the softmax cross-entropy loss.
tags: [Module, Linear, Activation, Initialization, Softmax, Cross-entropy, MLP]
order: 1
eyebrow: Networks · Multilayer perceptron
footer: Code excerpts are from <a href="https://github.com/maaaax1611/nabla/tree/main/src/nabla/nn"><code>nabla/nn</code></a>, <a href="https://github.com/maaaax1611/nabla/blob/main/src/nabla/init.py"><code>init.py</code></a> and <a href="https://github.com/maaaax1611/nabla/blob/main/src/nabla/ops/loss.py"><code>ops/loss.py</code></a>, shortened to the essentials. The last figure trains the network from <a href="https://github.com/maaaax1611/nabla/blob/main/examples/xor.py"><code>examples/xor.py</code></a> in the browser.
---

A neural network is a function with many adjustable numbers, built by stacking simple layers. This page assembles the smallest useful one, the multilayer perceptron: linear layers, a nonlinearity between them and a loss on top. Everything is built from the operations of the [Autograd](../core/autograd.html) page, so none of it needs a backward pass written by hand, with one deliberate exception.

## Modules and parameters {#module group="Layers" toc="Module"}

A layer bundles a computation with the tensors it learns, its **parameters**. An optimizer needs all parameters of a model as one flat list. `Module` collects them by looking at its own attributes:

```python
class Module:
    def parameters(self):
        params = []
        for value in self.__dict__.values():
            if isinstance(value, Tensor) and value.requires_grad:
                params.append(value)                # a parameter of this module
            elif isinstance(value, Module):
                params.extend(value.parameters())   # a sub-module: recurse
        return params
```

Assigning a tensor with `requires_grad=True` or another module to `self` is all the registration there is. The same walk over the attributes implements `zero_grad()`, `train()` / `eval()` and `to(device)`. Layers kept in a plain Python list are invisible to it; `ModuleList` exists for that case.

A model is a module whose `forward` calls its sub-modules:

```python
class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(2, 8)
        self.fc2 = nn.Linear(8, 1)

    def forward(self, x):
        return self.fc2(self.fc1(x).relu())
```

## The linear layer {#linear toc="Linear"}

A linear layer maps a batch of row vectors $X \in \mathbb R^{N \times d_\text{in}}$ to

$$
Y = XW + b, \qquad W \in \mathbb R^{d_\text{in} \times d_\text{out}},\quad b \in \mathbb R^{d_\text{out}}
$$

Each of the $d_\text{out}$ outputs is a weighted sum of the inputs plus an offset. In nabla the layer is one line, composed of a matrix product and an addition:

```python
def forward(self, x):
    return x.matmul(self.weight) + self.bias
```

Its gradients therefore come from `MatMul` and `Add`. With $G = \partial L/\partial Y$:

$$
\frac{\partial L}{\partial W} = X^\top G, \qquad
\frac{\partial L}{\partial X} = G\,W^\top, \qquad
\frac{\partial L}{\partial b} = \sum_{n=1}^{N} G_{n,:}
$$

The shapes are a useful check: each gradient has the shape of its tensor, and there is only one way to multiply $G$ with the other factor that produces it. The bias was broadcast over the batch, so its gradient is summed over the batch.

> [!abstract] Key points
> - `Module.parameters()` finds parameters by walking the attributes, recursively.
> - Linear layer: $Y = XW + b$ with $W$ of shape $(d_\text{in}, d_\text{out})$.
> - $\partial L/\partial W = X^\top G$, $\;\partial L/\partial X = G\,W^\top$.

## Activations {#activations group="Nonlinearity"}

Two linear layers in a row are still one linear layer:

$$
(XW_1 + b_1)\,W_2 + b_2 = X\,(W_1 W_2) + (b_1 W_2 + b_2)
$$

Depth only adds expressive power if a nonlinear function, the **activation**, is applied elementwise between the layers.

| Activation | $f(x)$ | $f'(x)$ |
| --- | --- | --- |
| ReLU | $\max(0, x)$ | $1$ for $x > 0$, else $0$ |
| Sigmoid | $\sigma(x) = \dfrac{1}{1 + e^{-x}}$ | $\sigma(x)\,(1 - \sigma(x))$ |
| Tanh | $\tanh x$ | $1 - \tanh^2 x$ |
| GELU | $x\,\Phi(x)$ | $\Phi(x) + x\,\varphi(x)$ |

::: widget nn-01-activations

In the backward pass an activation multiplies the gradient elementwise by $f'(x)$. This factor decides how well gradients survive many layers:

- **Sigmoid** has a derivative of at most $0.25$, and close to $0$ for large $|x|$. Across several layers the product becomes very small (*vanishing gradients*).
- **Tanh** reaches $1$ at the origin but saturates in the same way.
- **ReLU** passes the gradient unchanged wherever its input is positive, and blocks it elsewhere.
- **GELU** is a smooth variant of ReLU, with $\Phi$ the distribution function of the standard normal distribution. nabla uses the common approximation $\Phi(x) \approx \tfrac12\bigl(1 + \tanh\bigl(\sqrt{2/\pi}\,(x + 0.044715\,x^3)\bigr)\bigr)$, because NumPy has no error function.

For sigmoid and tanh the derivative can be written with the function value itself, so these operations store their output and not their input.

## Initialization {#init toc="Initialization"}

Before training the weights are random. Their scale matters: every layer multiplies the size of the signal by some factor, and across many layers this factor is raised to the power of the depth.

Take a layer with $n$ inputs and independent weights of variance $\sigma^2$. An output is a sum of $n$ terms, so for inputs with mean square $q$ the pre-activation has variance $n\,\sigma^2\,q$. The signal keeps its size if this factor is balanced against what the activation does:

- **Xavier (Glorot)** assumes an activation that is roughly the identity near zero, such as tanh: $\sigma^2 = 1/n$. To keep the gradients stable on the way back as well, it averages over inputs and outputs: $\sigma^2 = 2/(n_\text{in} + n_\text{out})$.
- **He (Kaiming)** accounts for ReLU, which sets half of the pre-activations to zero and thereby halves the mean square: $\sigma^2 = 2/n_\text{in}$.

nabla initializes `Linear` and `Conv2D` with He by default and offers Xavier as an alternative; biases start at zero. For a convolution $n_\text{in}$ is the number of values that enter one output: `in_channels * kh * kw`.

> [!abstract] Key points
> - Without an activation, depth collapses into one linear layer.
> - The derivative of the activation scales the gradient in every layer.
> - He: $\sigma^2 = 2/n_\text{in}$ (ReLU). Xavier: $\sigma^2 = 2/(n_\text{in} + n_\text{out})$ (tanh, sigmoid).

## Mean squared error {#mse group="Loss" toc="MSE"}

A loss reduces the outputs of a batch to one number that measures the error. For regression the usual choice is the mean squared error:

$$
L = \frac{1}{n}\sum_{i=1}^{n} (\hat y_i - y_i)^2, \qquad \frac{\partial L}{\partial \hat y_i} = \frac{2}{n}\,(\hat y_i - y_i)
$$

In nabla it is composed of a subtraction, a multiplication and a mean:

```python
diff = predictions - targets
return (diff * diff).mean()
```

## Softmax and cross-entropy {#cross-entropy toc="Softmax, cross-entropy"}

For classification the network outputs one score per class, the **logits** $z \in \mathbb R^K$. **Softmax** turns them into a probability distribution:

$$
p_k = \frac{e^{z_k}}{\sum_{j} e^{z_j}}
$$

The **cross-entropy** loss is the negative log-probability of the correct class $t$. With the one-hot vector $y$ of the target:

$$
L = -\ln p_t = -\sum_k y_k \ln p_k
$$

It is $0$ when the network is certain and correct and grows without bound when it is certain and wrong. Taken together, the two steps have a remarkably simple gradient. Inserting the softmax gives $L = -z_t + \ln\sum_j e^{z_j}$, and therefore

$$
\frac{\partial L}{\partial z_k} = p_k - y_k
$$

::: widget nn-03-softmax

This is the one place where nabla implements a composition as a single operation, `SoftmaxCrossEntropy`, for two reasons:

- **Stability.** Computed separately, $e^{z_k}$ overflows for large logits and $\ln p_t$ becomes $-\infty$ once $p_t$ rounds to zero. The fused operation subtracts the largest logit first and computes $\ln p$ directly as $z - \max z - \ln\sum_j e^{z_j - \max z}$ (*log-sum-exp*).
- **A simple backward pass.** Instead of chaining the Jacobian of the softmax with the derivative of the logarithm, backward returns $(p - y)/N$ for a batch of $N$ samples.

```python
def backward(self, grad_output):
    grad = self.probs.copy()
    grad[np.arange(N), targets] -= 1          # p - y
    return grad * grad_output / N, ...
```

A model for classification therefore ends with a linear layer and no activation; `CrossEntropyLoss` expects raw logits. The standalone `Softmax` operation is still needed where probabilities are used inside the network, as in attention.

> [!abstract] Key points
> - Softmax: logits → probabilities. Cross-entropy: $L = -\ln p_t$.
> - Fused: $\partial L/\partial z = p - y$, computed stably with log-sum-exp.
> - `CrossEntropyLoss` takes logits, not probabilities.

## Training a network {#xor group="Example" toc="XOR"}

XOR is the smallest problem a single linear layer cannot solve: no straight line separates $(0,1)$ and $(1,0)$ from $(0,0)$ and $(1,1)$. One hidden layer is enough.

```python
net = XORNet()                       # Linear(2, 8) → ReLU → Linear(8, 1) → Sigmoid
criterion = nn.MSELoss()
optimizer = optim.SGD(net.parameters(), lr=0.1)

for epoch in range(1000):
    loss = criterion(net(X), y)      # forward pass builds the graph
    net.zero_grad()                  # clear the gradients of the last step
    loss.backward()                  # gradient of the loss for every parameter
    optimizer.step()                 # move each parameter against its gradient
```

These four lines are the whole training loop. `zero_grad()` is needed because `backward()` adds to existing gradients.

::: widget nn-04-xor

Set the hidden activation to *none* to see the collapse from the Activations section in practice: the network is then a single linear layer followed by a sigmoid, the best it can do is output $0.5$ everywhere, and the loss stays at $0.25$.
