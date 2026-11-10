---
title: "Attention and the Transformer"
description: "Embeddings, positional encoding, multi-head attention, LayerNorm, the Transformer block, and a character-level language model."
---

# Attention and the Transformer

Convolutions hard-code which positions may talk to each other. Attention
lets the model decide: every position looks at every other position and
computes how much to take from each. This post builds the pieces one at a
time — a normalization that works on sequences, attention itself, token
embeddings, positional information — assembles them into a Transformer
block, and ends with a character-level language model trained on
Shakespeare.

## LayerNorm


Like [BatchNorm2D](04-cnn.md#batchnorm2d), LayerNorm re-centers and re-scales
activations to zero mean and unit variance, then lets the network learn its
own scale and shift (`gamma`, `beta`) back on top. The difference is *which*
elements get averaged together to compute that mean/variance:

- BatchNorm2D normalizes **per channel**, over every other sample in the
  batch (+ spatial axes) — so it needs a reasonably large batch to get a
  stable estimate, and behaves differently at train vs. eval time.
- LayerNorm normalizes **per sample**, over that sample's own feature axis
  — every sample is normalized independently of every other one.

That independence is exactly why LayerNorm is the normalization of choice
for Transformers: sequence batches often mix samples of different
effective lengths (padding), and you don't want one sample's statistics
leaking into another's, or the normalization to depend on batch size at
all. It also means LayerNorm needs no running statistics and behaves
identically whether the model is training or not — see
["No train/eval split"](#no-traineval-split) below.

### LayerNorm: the math

For input $x$ of shape `(*, \text{features})` — any number of leading axes,
with `features` last — mean and variance are computed **per sample**, over
just that last axis, using $N = \text{features}$ values:

$$
\mu = \frac{1}{N}\sum_k x_k \qquad \sigma^2 = \frac{1}{N}\sum_k (x_k - \mu)^2
$$

$$
\hat{x} = \frac{x - \mu}{\sqrt{\sigma^2 + \varepsilon}} \qquad \text{out} = \gamma \cdot \hat{x} + \beta
$$

This is the *identical* formula to BatchNorm2D — only which axis gets
reduced over changes. `gamma`/`beta` still have shape `(features,)`, but
because `features` is already the *last* axis of `x`, they broadcast
against it directly with no `.reshape(...)` needed (BatchNorm2D's channels
sit on the second axis of `(batch, C, H, W)`, which is why *that* op needs
to reshape `gamma`/`beta` to `(1, C, 1, 1)` first).

### Backward: same graph, different axis

Because the math is identical, so is the computational graph — decomposing
`out = gamma * x_hat + beta` into atomic operations produces the exact same
nine-node graph documented in
[BatchNorm2D's backward derivation](04-cnn.md#backward-derived-from-the-computational-graph):
`x → mu`, `x - mu → xmu`, `xmu² → sq`, `mean(sq) → var`, and so on through
`sqrtvar`, `ivar`, `xhat`, `gammax`, `out` — with `x` still forking into two
paths (direct, through `xmu`, and indirect, through `mu`) that get summed
at the very end. The only thing that changes walking that graph backward is
*which axis* every `sum`/`mean` reduces over — `axis=(0, 2, 3)` becomes
`axis=-1` — and what `N` means: batch·H·W becomes just `features`.

```python
leading_axes = tuple(range(grad_output.ndim - 1))
grad_beta = np.sum(grad_output, axis=leading_axes)
grad_gamma = np.sum(grad_output * self.x_hat, axis=leading_axes)

grad_x_hat = grad_output * gamma.data

grad_var = np.sum(
    grad_x_hat * self.x_centered * -0.5 * (self.var + self.eps) ** -1.5,
    axis=-1,
    keepdims=True,
)
grad_mean = np.sum(grad_x_hat * -1 / np.sqrt(self.var + self.eps), axis=-1, keepdims=True)
grad_mean += grad_var * np.sum(-2 * self.x_centered, axis=-1, keepdims=True) / self.N

grad_x = grad_x_hat / np.sqrt(self.var + self.eps)
grad_x += grad_var * 2 * self.x_centered / self.N
grad_x += grad_mean / self.N
```

Note `grad_beta`/`grad_gamma` sum over `leading_axes` (every axis *except*
the last one) rather than a fixed `axis=(0, 2, 3)` — LayerNorm has to work
for any number of leading axes (a plain `(batch, features)` input, or a
Transformer's `(batch, seq_len, features)`), so which axes count as
"leading" depends on the input's rank, computed once from
`grad_output.ndim` rather than hardcoded.

### LayerNorm: implementation

[`ops/layernorm.py`](https://github.com/maaaax1611/nabla/blob/main/src/nabla/ops/layernorm.py) caches exactly what the
backward table above needs — `x_centered`, `x_hat`, `var`, and `N` — the
same "what does forward need to *cache* for backward" question that came up
for [BatchNorm2D](04-cnn.md#batchnorm2d), just answered once and
gotten right this time. Correctness is checked the same way as every other
op in the library: a naive reference implementation for forward, and
numerical gradient checks for backward — including a `(batch, seq_len,
features)` case to specifically cover the Transformer-shaped input LayerNorm
exists for, in
[`tests/test_layernorm_ops.py`](https://github.com/maaaax1611/nabla/blob/main/tests/test_layernorm_ops.py).

### No train/eval split

Unlike [`nn/batchnorm.py`](https://github.com/maaaax1611/nabla/blob/main/src/nabla/nn/batchnorm.py),
[`nn/layernorm.py`](https://github.com/maaaax1611/nabla/blob/main/src/nabla/nn/layernorm.py) has no `running_mean` /
`running_var`, and its `forward` doesn't branch on `self.training` at all
— it always calls the same `Function`:

```python
def forward(self, x: Tensor) -> Tensor:
    return LayerNormFunction.apply(x, self.gamma, self.beta, eps=self.eps)
```

That's a direct consequence of normalizing per sample instead of per batch:
there's no batch-level statistic to smooth into a running average in the
first place, so eval mode has nothing different to do than training mode.

## Scaled Dot-Product Attention


Attention lets a model decide, per query position, how much to weigh each
of several key/value pairs — instead of every position in a sequence
processing itself in isolation (like a `Linear` layer would), attention
lets a query position pull in information from wherever in the sequence is
relevant to it. That's the operation multi-head attention (and Transformers
generally) build on:

$$
\text{Attention}(Q, K, V) = \text{softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}}\right) V
$$

- $Q$ (queries), $K$ (keys), $V$ (values) are all just matrices — for a
  single sequence, $Q$ has one row per query position, $K$/$V$ have one row
  per key/value position (often the same sequence as $Q$, for
  *self*-attention).
- $QK^\top$ computes a dot-product similarity between every query and every
  key — row $i$, column $j$ is "how much does query $i$ relate to key $j$".
- Dividing by $\sqrt{d_k}$ (the query/key feature dimension) keeps those dot
  products from growing large as $d_k$ grows, which would otherwise push
  softmax into a saturated regime with near-zero gradients almost
  everywhere except the single largest score.
- `softmax` (row-wise) turns those similarity scores into a proper
  probability distribution per query — the *attention weights*.
- Multiplying by $V$ turns those weights into a weighted average of the
  value vectors — each query position's output is a blend of the values it
  attends to.

### Why this isn't its own `Function`

Every other op documented so far ([Conv2D](04-cnn.md#conv2d),
[Pooling](04-cnn.md#pooling), [BatchNorm2D](04-cnn.md#batchnorm2d)) is its own
`Function` with a hand-derived `backward`, because each one needed a
genuine reformulation trick (im2col, argmax-routing, the batch-statistics
chain rule) that doesn't fall out of composing existing ops. Attention is
different: $QK^\top$, the scale, the mask, the softmax, and the final
multiply by $V$ are *all* already differentiable — nothing here is new
math. So [`functional.scaled_dot_product_attention`](https://github.com/maaaax1611/nabla/blob/main/src/nabla/functional.py)
is just those five steps written out as ordinary `Tensor` expressions, and
autodiff assembles the correct backward pass automatically by walking the
graph they build:

```python
scores = matmul(Q, _swap_last_two_axes(K))
scores = scores * (1 / math.sqrt(d_k))
if mask is not None:
    scores = scores + Tensor(xp.asarray(mask, dtype=scores.data.dtype))

weights = softmax(scores, axis=-1)
return matmul(weights, V)
```

This is exactly why the prerequisite step was making
[`MatMul`](https://github.com/maaaax1611/nabla/blob/main/src/nabla/ops/transform.py) support batched matmul (`np.matmul`
+ `np.swapaxes(-1, -2)` instead of `np.dot` + `.T`, with `broadcast.unbroadcast`
reducing any batch axes that got broadcast — the same helper `Add`/`Multiply`
already used for elementwise broadcasting) rather than adding a whole new
`Function` for attention — once the primitive is right, attention (and
later, multi-head attention) is just composition, not new calculus.

### Why it lives in `functional.py`, not as a `Tensor` method

Every argument here — $Q$, $K$, $V$ — is equally important; there's no
natural choice of which one should be `self` for a method call like
`Q.attention(K, V)`. That's exactly the case
[`functional.py`](https://github.com/maaaax1611/nabla/blob/main/src/nabla/functional.py) exists for: a free function
`scaled_dot_product_attention(Q, K, V, mask=None)`, the same shape as
PyTorch's `torch.nn.functional` module and free functions like
`torch.matmul`. See the module's own docstring for the full reasoning.

### Shapes and batching

$$
Q: (*, L_q, d_k) \qquad K: (*, L_k, d_k) \qquad V: (*, L_k, d_v) \qquad (L = \text{seq len})
$$

The leading `*` axes are arbitrary — a plain batch axis today, and (once
multi-head attention splits `embed_dim` into per-head slices) a `num_heads`
axis too, with no code changes needed here. That's what
`_swap_last_two_axes` is for: it transposes only the trailing two axes of
`K` to get $K^\top$, no matter how many leading axes come before them —

```python
def _swap_last_two_axes(t: Tensor) -> Tensor:
    ndim = t.data.ndim
    axes = tuple(range(ndim - 2)) + (ndim - 1, ndim - 2)
    return transpose(t, axes)
```

— which only works at all because `MatMul` now batches over those leading
axes the same way `np.matmul` does.

### Attention: the mask

`mask` is a plain `ndarray`, not a `Tensor` — there's nothing meaningful to
differentiate with respect to a fixed masking pattern. It's added *before*
the softmax, conventionally with a large negative value (e.g. `-1e9`) at
positions that must not be attended to: after softmax, `exp(-1e9)` underflows
to `0`, so those positions get (as close as floating point allows) exactly
zero attention weight without needing a separate masked-softmax
implementation.

#### Testing

Because this is pure composition, correctness mostly means checking that
the composition is *assembled* correctly, not re-deriving any gradient by
hand. [`tests/test_functional.py`](https://github.com/maaaax1611/nabla/blob/main/tests/test_functional.py) checks
forward against a naive reference (including the masked case), and backward
against a numerical gradient for all three inputs — the same numerical
gradient check used everywhere else in the library, just now confirming
that a whole chain of already-tested ops still composes into a correct
gradient end to end.

## Multi-Head Attention


Running [scaled dot-product attention](#scaled-dot-product-attention) once gives a model
one single way of relating positions to each other — one set of Q/K/V
projections, one attention pattern. Multi-head attention instead projects
Q/K/V into `num_heads` smaller subspaces, runs attention independently in
each, and concatenates the results back together, letting different heads
specialize in different kinds of relationships (e.g. one head tracking
nearby positions, another tracking a specific long-range dependency) instead
of forcing everything through one shared attention pattern.

Crucially, single-head attention with learned Q/K/V/output projections
*isn't a special case* that needs separate code — it's just `num_heads=1`.
[`nn/attention.py`](https://github.com/maaaax1611/nabla/blob/main/src/nabla/nn/attention.py) is a single `Attention`
class covering both: nothing in the implementation branches on whether
`num_heads` is 1 or more.

### Splitting into heads

Given `embed_dim` and `num_heads` (which must divide it evenly), each head
operates on a `head_dim = embed_dim // num_heads` slice. Projected Q/K/V
start out as `(batch, seq_len, embed_dim)` and need to become
`(batch, num_heads, seq_len, head_dim)` — a reshape to expose the head axis,
followed by a transpose to move it before the sequence axis (so
[scaled dot-product attention's batching](#shapes-and-batching)
treats `num_heads` as just another batch dimension, no code changes needed
there at all):

```python
def _split_heads(self, x: Tensor) -> Tensor:
    batch, seq_len, _ = x.data.shape
    x = x.reshape((batch, seq_len, self.num_heads, self.head_dim))
    return x.transpose((0, 2, 1, 3))
```

`_merge_heads` undoes exactly this after attention runs — transpose the head
axis back next to `head_dim`, then reshape the two back into one
`embed_dim` axis:

```python
def _merge_heads(self, x: Tensor) -> Tensor:
    batch, num_heads, seq_len, head_dim = x.data.shape
    x = x.transpose((0, 2, 1, 3))
    return x.reshape((batch, seq_len, num_heads * head_dim))
```

Both are ordinary compositions of `Reshape`/`Transpose` — already
differentiable, so (like [scaled dot-product attention](#scaled-dot-product-attention)
itself) no new backward pass needed anywhere in this layer.

### The four projections

```
Q, K, V = q_proj(query), k_proj(key), v_proj(value)   # each: Linear(embed_dim, embed_dim)
attended = scaled_dot_product_attention(split_heads(Q), split_heads(K), split_heads(V), mask)
out = out_proj(merge_heads(attended))                 # Linear(embed_dim, embed_dim)
```

Four separate `nn.Linear(embed_dim, embed_dim)` layers: three to project the
raw input into query/key/value space before splitting into heads, one more
(`out_proj`) to mix the concatenated heads back together after attention —
without it, the heads' outputs would just sit side by side with no way for
information to move between them.

### Self-attention vs. cross-attention

`forward(query, key=None, value=None, mask=None)` defaults `key`/`value` to
`query` when omitted, so `layer(x)` runs self-attention (every position
attends to every other position in the same sequence — the Transformer
encoder case). Passing `key`/`value` explicitly (e.g. an encoder's output,
with a possibly different sequence length than `query`) runs
cross-attention instead — the same code path either way, since Q/K/V are
always projected and split independently and
[scaled dot-product attention](#scaled-dot-product-attention) never assumed
`seq_len_q == seq_len_k` in the first place.

#### Testing

[`tests/test_nn_attention.py`](https://github.com/maaaax1611/nabla/blob/main/tests/test_nn_attention.py) checks shapes
for self- and cross-attention and for `num_heads=1`, that `_split_heads` and
`_merge_heads` are exact inverses of each other, gradients against a
numerical check (through all four projections, not just the attention core),
and one behavioral property of masking: perturbing a key/value position that
every query is masked away from must leave the output completely unchanged.

## Embedding

### The idea — not to be confused with "embeddings" the output

Two different things share the name "embedding," worth separating up
front:

- **The output of an encoder** — a dense vector representing a whole input
  (word, sentence, image), produced by many layers of transformation. This
  is what "embedding" usually means in a foundation-model context, and
  it's the *result* of running a model, not a layer in one.
- **`nn.Embedding`** (this section) — the very first layer of a language model,
  turning a discrete token ID into the dense vector that then *feeds into*
  the encoder. It's a learnable lookup table, nothing more: shape
  `(vocab_size, embed_dim)`, one trainable row per vocabulary entry.

### The math (or lack of it)

$$
\text{out} = \text{table}[\text{indices}]
$$

Forward is pure indexing — for every integer ID in `indices`, look up the
corresponding row of `table`. No matmul, no activation, nothing to derive.
That simplicity is exactly what makes the *backward* pass worth pausing on.

### Why the backward pass isn't just "write the gradient back"

If `indices` never repeated within a call, backward would be trivial:
scatter `grad_output`'s rows back to the corresponding rows of `table`.
But token IDs *do* repeat — the same word can appear twice in one sequence,
or the same token across different sequences in a batch — and each
occurrence independently pulled `table`'s row into the output, so each
occurrence must independently contribute to that row's gradient. Writing
naively:

```python
grad_table[indices] = grad_output  # WRONG: last occurrence silently wins
```

would let a later repeat of the same ID silently overwrite an earlier
contribution instead of adding to it. This is exactly the same scatter/
accumulate problem [`MaxPool2D`'s backward](04-cnn.md#pooling) solves with
`np.add.at` — routing possibly-overlapping contributions back to a smaller
array without one clobbering another — so `Embedding` uses the identical
tool:

```python
def backward(self, grad_output: NDArray) -> tuple[NDArray, NDArray]:
    grad_table = np.zeros(self.table_shape)
    np.add.at(grad_table, self.indices, grad_output)

    grad_indices = np.zeros_like(self.indices, dtype=float)
    return grad_table, grad_indices
```

`np.add.at(grad_table, self.indices, grad_output)` accumulates
`grad_output`'s rows into `grad_table` at the positions given by
`self.indices`, correctly summing when an index repeats — unlike plain
fancy-index assignment (`grad_table[self.indices] = ...`), which the NumPy
docs explicitly call out as *not* accumulating for repeated indices.

### `indices` isn't differentiable

Same pattern as `targets` in
[`SoftmaxCrossEntropy`](02-mlp.md#targets-isnt-differentiable):
`indices` is still passed through `Function.apply` as a `Tensor`, purely so
it participates in the graph machinery like any other input, but integer
token IDs have no meaningful gradient. `backward` returns an all-zero array
for it, and `nn/embedding.py` never sets `requires_grad=True` on the
indices it's called with, so `Tensor.backward()` discards that gradient via
the same `if parent.requires_grad:` check.

### Initialization is different from Linear/Conv2D

`nn/embedding.py` doesn't default to He or Xavier init — both derive their
scale from `fan_in`/`fan_out`, which assumes every output depends on *all*
of a layer's inputs (true for a matmul, false here: each output row of an
embedding lookup depends on exactly *one* row of `table`, never a
combination of several). Instead it defaults to small-normal
initialization (`N(0, 0.01)`), matching PyTorch's `nn.Embedding` default:

```python
def _small_normal(shape: tuple[int, ...]) -> NDArray:
    return np.random.randn(*shape) * 0.01
```

#### Testing

[`tests/test_embedding_ops.py`](https://github.com/maaaax1611/nabla/blob/main/tests/test_embedding_ops.py) checks
forward against plain NumPy fancy indexing, backward against a numerical
gradient, and — the property that actually exercises the scatter-add fix —
that looking up the same index multiple times in one call produces a
gradient that's the *sum* of every occurrence's contribution, not just the
last one.

## Positional Encoding


Attention is permutation-invariant: `softmax(QK^T/sqrt(d_k))V` treats the
sequence axis purely as a set of positions that can attend to one another
— nothing in that formula encodes *which* position came before which.
Shuffle a sentence's tokens and shuffle the output back, and attention
alone would produce the same result either way; "the dog bites the man"
and "the man bites the dog" would look identical to an attention layer
with no other signal. Positional encoding is what breaks that symmetry —
it's added directly to the token embeddings *before* they reach any
attention layer, so position information is baked into the values Q/K/V
get computed from.

### Why it needs no new `Function`

Unlike [`Embedding`](#embedding), this table is not learned — it's a
fixed mathematical pattern, computed once from `embed_dim` and `max_len`
and then simply **added** to the input:

```python
return x + Tensor(self.table[:seq_len])
```

Adding a constant only shifts the input; $\partial(x+c)/\partial x = 1$
either way, so the existing `Add` `Function` (see
[Autodiff basics](01-autograd.md#autodiff-basics)) already produces the correct gradient
with no new backward derivation needed — the constant Tensor just never
has `requires_grad=True`, so its own (correctly computed, but irrelevant)
gradient gets discarded the same way `targets`/`indices` are elsewhere in
the library.

### Positional Encoding: the formula

For position $pos$ (0, 1, 2, ...) and embedding dimension $i$, alternating
sine and cosine:

$$
PE(pos, 2i) = \sin\!\left(\frac{pos}{10000^{2i/d}}\right) \qquad
PE(pos, 2i{+}1) = \cos\!\left(\frac{pos}{10000^{2i/d}}\right)
$$

```python
position = np.arange(max_len)[:, None]          # (max_len, 1)
i = np.arange(embed_dim)[None, :]                # (1, embed_dim)
angle_rates = 1.0 / np.power(10000.0, (2 * (i // 2)) / embed_dim)
angles = position * angle_rates                  # (max_len, embed_dim)

table = np.zeros((max_len, embed_dim))
table[:, 0::2] = np.sin(angles[:, 0::2])
table[:, 1::2] = np.cos(angles[:, 1::2])
```

### Why this specific formula, not something simpler

**Why not just add the raw position number?** Because it's unbounded —
`table[:, i] += pos` would grow arbitrarily large for long sequences,
swamping the embedding values it's added to. Every entry of the sinusoidal
table stays in $[-1, 1]$ regardless of sequence length, so it never
dominates or destabilizes the scale of what it's added to.

**Why does the frequency depend on $i$?** `10000^{2i/d}` makes early
dimensions ($i$ small) oscillate quickly across positions and later
dimensions oscillate slowly — like a clock with hands moving at very
different speeds. Together, all dimensions form a unique combination per
position, the same way a clock's second/minute/hour hands together
identify a unique moment even though each hand alone repeats constantly.

**Why alternate sine and cosine instead of just sine?** This is the part
that makes *relative* positions easy for the network to use, not just
absolute ones. Because of the angle-addition identities

$$
\sin(a+b) = \sin a \cos b + \cos a \sin b \qquad \cos(a+b) = \cos a \cos b - \sin a \sin b
$$

the encoding at position $pos + k$ can be written as a fixed **linear**
combination of the encoding at position $pos$ (for a constant offset $k$,
the coefficients $\sin(k\omega)$/$\cos(k\omega)$ don't depend on $pos$).
That means a simple linear layer downstream is, in principle, capable of
learning to attend to "the token $k$ positions back" — a relative
relationship — directly from the additive positional signal, not just to
memorize absolute position numbers.

#### Testing

[`tests/test_nn_positional_encoding.py`](https://github.com/maaaax1611/nabla/blob/main/tests/test_nn_positional_encoding.py)
checks the table directly (bounded values, `sin(0)=0`/`cos(0)=1` at
position 0, every position gets a distinct row) and the layer's behavior
(same table slice broadcast across every batch element, sequence-length and
embed-dim validation, and — since this op involves no new math — that
`backward` really does just pass the gradient straight through unchanged).
`PositionalEncoding.parameters()` is also checked to return nothing: the
table is a plain `ndarray` attribute, never wrapped in a `Tensor`, so
[`Module.parameters()`](01-autograd.md#autodiff-basics) never picks it up as trainable.

## Transformer Block


Every building block from the last several docs —
[`Attention`](#multi-head-attention), [`LayerNorm`](#layernorm),
[`Dropout`](03-training.md#dropout) — comes together here into one reusable unit:
[`nn/transformer.py`](https://github.com/maaaax1611/nabla/blob/main/src/nabla/nn/transformer.py)'s `TransformerBlock`,
the same layer PyTorch ships as `nn.TransformerEncoderLayer`. Stack several
of these and you have a Transformer encoder.

```python
attended = self.attn(self.norm1(x), mask=mask)
x = x + self.dropout(attended)

fed_forward = self.ff(self.norm2(x))
x = x + self.dropout(fed_forward)
```

Two sublayers, each wrapped the same way: normalize, run the sublayer,
dropout, add back to the input.

### The feedforward sublayer

[`nn/feedforward.py`](https://github.com/maaaax1611/nabla/blob/main/src/nabla/nn/feedforward.py) is deliberately
small — `Linear(embed_dim, hidden_dim) -> ReLU -> Linear(hidden_dim,
embed_dim)`, applied identically and independently to every position in
the sequence (hence "position-wise": no mixing across the sequence axis at
all, that already happened in attention). Attention lets positions
exchange information; the feedforward sublayer is where the model actually
does per-position computation on what it gathered.

### Residual connections: why `x + sublayer(x)`, not just `sublayer(x)`

Without the `x +`, a stack of $N$ `TransformerBlock`s would force gradients
to flow back through every single sublayer in sequence to reach an early
layer — the same vanishing-gradient problem that makes any sufficiently
deep network hard to train. With the residual connection, the `+` gives
`backward()` a direct, unimpeded path straight back through every block
(`Add`'s backward is just the identity on both branches — see
[Autodiff basics](01-autograd.md#autodiff-basics)), regardless of how many blocks are
stacked in between. The sublayer only has to learn a *correction* to add
on top of `x`, not reproduce `x` itself in order to pass it through.

### Pre-norm vs. post-norm

The original Transformer paper normalizes *after* each sublayer:
`x = LayerNorm(x + Sublayer(x))`. This implementation instead uses
**pre-norm** — normalize *before* the sublayer runs:

```python
x = x + Dropout(Sublayer(LayerNorm(x)))
```

the convention most models have used since GPT-2. The difference matters
for exactly the same reason residual connections do: with pre-norm, the
running sum `x` down the residual path is never itself passed through a
normalization — it stays a clean, unnormalized accumulation of every
sublayer's output, which trains more stably as more blocks get stacked.
With post-norm, that accumulated sum gets renormalized after every single
block, which empirically makes deep stacks harder to train without extra
tricks (learning-rate warmup, careful initialization).

### The mask stays generic

`TransformerBlock.forward(x, mask=None)` just forwards `mask` straight
through to [`Attention`](#multi-head-attention) — it doesn't build or
know anything about what kind of mask gets passed in. That's intentional:
a causal (look-ahead) mask for language modeling, a padding mask for
variable-length batches, or no mask at all for a plain encoder are all
*task*-specific decisions, not something a generic reusable block should
hardcode. Building the actual causal mask is left to whatever example uses
this block for language modeling.

#### Testing

[`tests/test_nn_transformer.py`](https://github.com/maaaax1611/nabla/blob/main/tests/test_nn_transformer.py) checks
shapes, gradients through every parameter (all four sub-modules:
`attn`, `norm1`, `ff`, `norm2`), that `eval()`/`train()` correctly
propagates down to the internal `Dropout`, and one behavioral property of
masking that's worth calling out specifically: with a causal mask applied,
drastically perturbing the *last* position in a sequence must leave every
*earlier* position's output completely unchanged — a direct, executable
check that the mask is doing what "causal" is supposed to mean, not just
that shapes come out right.

## ModuleList

### The problem it solves

Stacking several of the same layer — several
[`TransformerBlock`s](#transformer-block), for instance — reads
naturally as a Python list:

```python
self.blocks = [TransformerBlock(embed_dim, num_heads, hidden_dim) for _ in range(num_layers)]
```

But [`Module.parameters()`/`train()`/`eval()`](01-autograd.md#autodiff-basics) only look at
attributes that are themselves a `Tensor` or a `Module` — a plain `list`
attribute is neither, so every parameter inside those blocks would be
silently invisible to `parameters()` (no gradient updates would ever reach
them) and `train()`/`eval()` would never propagate into them either. This
is exactly the gap PyTorch's `nn.ModuleList` exists to close — a plain list
doesn't work there either, for the identical reason.

### How `ModuleList` closes it

[`nn/container.py`](https://github.com/maaaax1611/nabla/blob/main/src/nabla/nn/container.py) doesn't add any new
logic to `Module.parameters()`/`train()`/`eval()` at all — it just stores
each submodule under its own generated attribute name, so the *existing*
introspection (which already recurses into any `Module`-typed attribute)
picks every one of them up automatically:

```python
def __init__(self, modules: Iterable[Module]) -> None:
    super().__init__()
    self._items: list[Module] = list(modules)
    for i, module in enumerate(self._items):
        setattr(self, f"_module_{i}", module)
```

`self._items` itself stays a plain list (harmless — it's not a `Tensor` or
`Module`, so `parameters()` skips right over it), used only for Python-side
indexing and iteration (`__getitem__`, `__iter__`, `__len__`), while
`self._module_0`, `self._module_1`, ... are what actually make every
submodule's parameters and train/eval state reachable.

#### Testing

[`tests/test_nn_container.py`](https://github.com/maaaax1611/nabla/blob/main/tests/test_nn_container.py) checks
indexing and iteration order, that `parameters()` returns every submodule's
parameters (not just the first, or none), that `train()`/`eval()`
propagates to every item, and that gradients flow correctly through a small
stack of layers end to end.

## Example: A Character-Level Transformer on Tiny Shakespeare

### Shakespeare example: what this ties together

[`examples/shakespeare_transformer.py`](https://github.com/maaaax1611/nabla/blob/main/examples/shakespeare_transformer.py)
is where every building block from this post and the [training post](03-training.md) finally comes
together into an actual trained model: `Embedding` +
`PositionalEncoding` feed a stack of `TransformerBlock`s (held in a
`ModuleList`), followed by a final `LayerNorm` and a `Linear` "head"
projecting back to vocabulary size — the same architecture as
nanoGPT, just running on plain NumPy instead of a GPU.

```python
class CharTransformerLM(Module):
    def forward(self, token_ids, mask=None):
        x = self.token_embedding(token_ids)
        x = self.pos_encoding(x)
        for block in self.blocks:
            x = block(x, mask=mask)
        x = self.norm_out(x)
        return self.lm_head(x)
```

### Why character-level, not BPE

The task is next-character prediction on the Tiny Shakespeare corpus.
Tokenization here is deliberately the simplest possible scheme — one
integer per distinct character, no merges, no subword logic (see
[`examples/shakespeare_data.py`](https://github.com/maaaax1611/nabla/blob/main/examples/shakespeare_data.py)'s
`CharTokenizer`). Implementing byte-pair encoding was considered and
rejected: BPE is a text-preprocessing algorithm, not an autodiff
concern, and would add real scope without exercising any new part of
nabla. Character-level tokenization keeps the example focused on what
it's actually meant to demonstrate — the Transformer internals — while
still needing a real `Embedding` lookup table.

### Why a causal mask, and why it's built here and not in `TransformerBlock`

Language modeling means position `t` can only attend to positions
`<= t` — it must never see the token it's trying to predict, or any
token after it. The [Transformer block section](#transformer-block)
already explains why `TransformerBlock` keeps `mask` fully generic
instead of hardcoding any particular masking strategy: causal masking
is a property of *this task* (autoregressive language modeling), not
of the block itself. So `causal_mask()` — an additive `-1e9` above the
diagonal, added to the attention scores before softmax — lives in this
example:

```python
def causal_mask(seq_len: int) -> np.ndarray:
    return np.triu(np.full((seq_len, seq_len), -1e9), k=1)
```

### Shakespeare example: training loop

Nothing new here — `CrossEntropyLoss`, `Adam`, `model.zero_grad()` /
`loss.backward()` / `optimizer.step()`, the same pattern as every
other example. The only wrinkle is flattening the `(batch, block_size,
vocab_size)` logits and `(batch, block_size)` targets down to 2D and 1D
before handing them to `CrossEntropyLoss`, since it expects one
prediction row per example, not per sequence:

```python
def compute_loss(model, X, y, mask, criterion):
    logits = model(Tensor(X), mask=mask)
    batch, block_size, vocab_size = logits.data.shape
    logits_flat = logits.reshape((batch * block_size, vocab_size))
    targets_flat = y.reshape(batch * block_size)
    return criterion(logits_flat, targets_flat)
```

At the example's configuration (`embed_dim=64, num_heads=4,
hidden_dim=256, num_layers=3, block_size=64, batch_size=16`), 1000
training steps take roughly two minutes on CPU.

### Generation

`generate()` runs the model autoregressively: encode a prompt, repeatedly
feed the last `block_size` tokens through the model with a causal mask,
sample the next character from the softmax distribution over the final
position's logits, append it, and repeat. This is a proof-of-concept at
a tiny scale and budget — don't expect coherent English. What you should
see is the *shape* of English emerging from what started as uniform
noise: plausible word lengths, apostrophes and punctuation in sensible
places, line breaks, even character-name-like capitalized words followed
by a colon — structure the model had to learn from nothing but
next-character prediction.

### Shakespeare example: running it

```bash
uv run python examples/shakespeare_transformer.py
```

The corpus downloads once and is cached under
`examples/.shakespeare_cache/` (git-ignored) on first run.

---

Previous: [Convolutional networks](04-cnn.md) · Next: [Vision Transformer](06-vit.md)
