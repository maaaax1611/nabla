# Softmax + Cross-Entropy

## The idea

For multi-class classification, a network outputs one raw score
("logit") per class, and we need two things: a way to turn those scores
into a probability distribution (**softmax**), and a way to measure how far
that distribution is from the true label (**cross-entropy**). nabla
implements both together as a single `Function`,
[`SoftmaxCrossEntropy`](../src/nabla/ops/loss.py) — not because they can't
be expressed as separate ops, but because fusing them is both more
numerically stable *and* produces a dramatically simpler gradient. Both
reasons are worth spelling out.

## Softmax

For logits $z \in \mathbb{R}^K$ ($K$ classes):

$$
\text{softmax}(z)_k = \frac{e^{z_k}}{\sum_{j=1}^K e^{z_j}}
$$

This is a genuine problem to compute directly: if any logit is large (say
$z_k = 1000$), $e^{z_k}$ overflows a float64 long before it matters. The
fix is the **log-sum-exp trick** — softmax is invariant to subtracting any
constant $c$ from every logit (it cancels in the ratio), so subtracting the
row's max keeps every exponent $\le 0$:

$$
\text{softmax}(z)_k = \frac{e^{z_k - c}}{\sum_j e^{z_j - c}}, \qquad c = \max_j z_j
$$

```python
shifted = logits.data - logits.data.max(axis=1, keepdims=True)
exp_shifted = np.exp(shifted)
probs = exp_shifted / exp_shifted.sum(axis=1, keepdims=True)
```

## Cross-entropy

Given the true class index $y$ (an integer, not a `Tensor` in a
differentiable sense — see the note at the end), cross-entropy is just the
negative log-probability the model assigned to the correct class, averaged
over the batch:

$$
L = -\frac{1}{B}\sum_{n=1}^{B} \log \text{softmax}(z^{(n)})_{y^{(n)}}
$$

Computing $\log(\text{softmax}(z)_k)$ directly from `probs` would first
compute a (possibly tiny) probability and then take its log — losing
precision for exactly the confident, correct predictions you care about
getting right at the end of training. Since `shifted` and
`log(sum_exp)` are already on hand from the softmax computation above,
`log_probs` is computed directly instead:

$$
\log \text{softmax}(z)_k = (z_k - c) - \log \sum_j e^{z_j - c}
$$

```python
log_probs = shifted - np.log(sum_exp)
picked = log_probs[np.arange(batch_size), target_idx]
loss = -picked.mean()
```

## The gradient: why fusing the two ops pays off

Naively, backpropagating through softmax followed by a gather-and-log step
would mean deriving (and implementing) the Jacobian of softmax itself —
softmax's Jacobian is a full $K \times K$ matrix per sample (every output
depends on every input, because of the shared normalizing sum), which is
both more code and more computation than necessary.

But composed with cross-entropy specifically, the two Jacobians collapse
into something remarkably simple. Writing $p = \text{softmax}(z)$ and $y$
as the one-hot vector of the true class:

$$
\frac{\partial L}{\partial z_k} = p_k - y_k
$$

*(Sketch: $\partial L/\partial p_y = -1/p_y$ from the log, and
$\partial p_k/\partial z_j = p_k(\delta_{kj} - p_j)$ is softmax's own
Jacobian; multiplying those out for a single target class and summing over
$j$ makes every term except $p_k$ and $y_k$ cancel.)* The result: the
gradient w.r.t. the logits is just the predicted distribution minus the
target distribution — subtract 1 from the probability of the true class,
leave everything else untouched, done:

```python
grad_logits = self.probs.copy()
grad_logits[np.arange(self.batch_size), self.target_idx] -= 1
grad_logits *= grad_output / self.batch_size
```

This is one of the cleanest gradients in the whole library — a nice
illustration of how choosing *where* to draw the boundary between ops can
matter as much for the backward pass as for the forward one. A
[`tests/test_loss.py`](../tests/test_loss.py) test checks exactly this
property directly: `grad_logits.sum(axis=1)` should be `0` for every row,
since both `probs` and the one-hot target sum to 1.

## `targets` isn't differentiable

`targets` (the integer class labels) is still passed through `Function.apply`
as a `Tensor` — mainly so it participates in the graph machinery the same
way every other input does — but it's never meaningful to differentiate
with respect to a class index. `backward` returns a zero array for it, and
since `CrossEntropyLoss` (the [`nn/loss.py`](../src/nabla/nn/loss.py)
wrapper) never sets `requires_grad=True` on it, that gradient is computed
but immediately discarded by `Tensor.backward()`'s
`if parent.requires_grad:` check (see
[Autodiff basics](01-autodiff.md#the-chain-rule-as-a-topological-sort)).
