# Softmax

## The idea

[`ops/loss.py`](../src/nabla/ops/loss.py) already computes softmax
internally, fused with cross-entropy — but that fusion only works when a
cross-entropy loss is the very next thing downstream. Once you want softmax
somewhere *else* (most immediately: turning attention scores into weights
for a Transformer), it needs to be its own differentiable
[`Function`](../src/nabla/ops/softmax.py), unaware of whatever comes after
it.

That difference matters for the backward pass. `SoftmaxCrossEntropy`'s
gradient collapses all the way down to `probs - one_hot`
(see [Softmax + Cross-Entropy](05-softmax-cross-entropy.md#the-gradient-why-fusing-the-two-ops-pays-off))
precisely *because* it knows the very next op is a log + gather against a
one-hot target. A standalone `Softmax` has no such luck — it has to return
the correct gradient for *any* downstream computation, which means
implementing softmax's actual Jacobian-vector product, not a shortcut.

## The math

For $x \in \mathbb{R}^K$ along the chosen axis:

$$
\text{softmax}(x)_k = \frac{e^{x_k}}{\sum_{j=1}^K e^{x_j}}
$$

computed the same numerically-stable way as everywhere else in nabla — the
log-sum-exp trick, subtracting the row's max before exponentiating so
`exp()` never overflows:

```python
shifted = x.data - x.data.max(axis=self.axis, keepdims=True)
exp_shifted = np.exp(shifted)
sum_exp = exp_shifted.sum(axis=self.axis, keepdims=True)
self.probs = exp_shifted / sum_exp
```

## Backward: the Jacobian-vector product

Softmax's defining trait is that **every output depends on every input**,
because they all divide by the same sum. Its local Jacobian is:

$$
\frac{\partial p_j}{\partial x_k} = p_j (\delta_{jk} - p_k)
\qquad\text{(} \delta_{jk}=1 \text{ if } j=k \text{, else } 0 \text{)}
$$

split into a diagonal term ($j=k$: $p_k(1-p_k)$) and an off-diagonal term
($j \ne k$: $-p_j p_k$). For a $K$-class row that's a full $K \times K$
matrix — never actually built in the code below, because the chain rule
collapses it to something far cheaper:

$$
\frac{\partial L}{\partial x_k} = \sum_j \frac{\partial L}{\partial p_j}\cdot\frac{\partial p_j}{\partial x_k}
= \sum_j \text{grad\_output}_j \cdot p_j(\delta_{jk} - p_k)
$$

Split the sum at $\delta_{jk}$ — one term survives only at $j=k$, the other
has $p_k$ constant in $j$ and factors out:

$$
= \underbrace{\text{grad\_output}_k \cdot p_k}_{j=k \text{ term}} \;-\; \underbrace{p_k \sum_j \text{grad\_output}_j \cdot p_j}_{p_k \text{ factored out}}
$$

$$
\frac{\partial L}{\partial x_k} = p_k \left( \text{grad\_output}_k - \sum_j \text{grad\_output}_j \cdot p_j \right)
$$

`sum_j grad_output_j * p_j` is one scalar per row — the probability-weighted
average of the incoming gradient — subtracted from every element before
scaling by that element's own probability. Intuitively: nudging $x_k$
doesn't just move $p_k$, it also nudges every other $p_j$ slightly (they all
share the same normalizer), and that shared-normalizer term is exactly what
this correction accounts for.

```python
def backward(self, grad_output: NDArray) -> tuple[NDArray]:
    weighted_sum = np.sum(grad_output * self.probs, axis=self.axis, keepdims=True)
    return (self.probs * (grad_output - weighted_sum),)
```

## Implementation

[`ops/softmax.py`](../src/nabla/ops/softmax.py) takes an `axis` argument
(default: the last axis) rather than hardcoding which axis gets normalized
— `Tensor.softmax(axis=-1)` needs that flexibility for attention, where
softmax is applied over the key/sequence axis, not necessarily the last one
depending on how scores are laid out.
[`tests/test_softmax_ops.py`](../tests/test_softmax_ops.py) checks forward
against a naive reference, numerical stability on very large logits,
backward against a numerical gradient (including a non-default `axis`),
and one property specific to this gradient formula: if `grad_output` is
uniform, `sum_j grad_output_j * p_j` reduces to `grad_output` itself
(since `probs` sums to 1), so the correction term cancels `grad_output`
exactly and `x.grad` comes out as zero everywhere.

## `nn.Softmax` has no parameters or train/eval split

[`nn/softmax.py`](../src/nabla/nn/softmax.py) is a thin `Module` wrapper
purely so softmax composes like any other layer inside a bigger model
(e.g. as one line in an attention block's `forward`) — it holds no
parameters and, like [LayerNorm](07-layernorm.md#no-traineval-split),
behaves identically regardless of `self.training`.
