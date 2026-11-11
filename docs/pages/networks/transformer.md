---
title: Transformers
desc: Token embeddings, positional encoding, scaled dot-product and multi-head attention, LayerNorm, the Transformer block, a character-level language model and a Vision Transformer.
tags: [Attention, Embedding, Positional encoding, LayerNorm, Transformer block, ViT]
order: 3
eyebrow: Networks · Sequences
footer: Code excerpts are from <a href="https://github.com/maaaax1611/nabla/blob/main/src/nabla/functional.py"><code>functional.py</code></a> and <a href="https://github.com/maaaax1611/nabla/tree/main/src/nabla/nn"><code>nabla/nn</code></a>, shortened to the essentials. The examples are <a href="https://github.com/maaaax1611/nabla/blob/main/examples/shakespeare_transformer.py"><code>examples/shakespeare_transformer.py</code></a> and <a href="https://github.com/maaaax1611/nabla/blob/main/examples/mnist_vit.py"><code>examples/mnist_vit.py</code></a>. Q and K in the figures are hand-picked or random, not taken from a trained model.
---

A convolution lets each position look at a fixed neighbourhood. A Transformer lets every position look at every other position and decide for itself which ones matter. The mechanism that does this, attention, needs almost nothing new from the library: it is a matrix product, a softmax and another matrix product. This page builds a Transformer from the input side up, then uses the same blocks twice: for text, one character at a time, and for images, one patch at a time.

## Tokens to vectors {#embedding group="Input" toc="Embedding"}

A model computes with vectors, not with characters. Text is first split into **tokens** from a fixed vocabulary, each replaced by its index. The Shakespeare example uses single characters as tokens: the vocabulary is the 65 distinct characters of the text, so every string can be encoded and no tokenizer has to be trained.

An **embedding** maps each index to a learned vector. It is a table $E \in \mathbb R^{V \times d}$ with one row per token, and the lookup simply selects rows:

```python
class Embedding(Function):
    def forward(self, table, indices):
        self.indices = indices.data
        return table.data[indices.data]                  # (*, d)

    def backward(self, grad_output):
        grad_table = zeros(self.table_shape, dtype=grad_output.dtype)
        add.at(grad_table, self.indices, grad_output)    # repeated tokens add up
        return grad_table, zeros_like(self.indices)
```

The backward pass writes each output gradient back into the row it came from. A token that occurs several times in a batch was read several times, so its row collects the sum of all these gradients, which is why `add.at` is used instead of a plain assignment. The indices themselves are integers and get no gradient.

The table starts with small random values, $\mathcal N(0, 0.01^2)$. He or Xavier initialization do not apply: they balance a sum over many inputs, and a lookup has no such sum.

## Where a token is {#posenc toc="Positional encoding"}

Attention, as defined below, treats its input as a set. Shuffle the tokens and the outputs are shuffled the same way, but otherwise unchanged; "dog bites man" and "man bites dog" would look identical. The order has to be put into the vectors themselves. nabla adds a fixed **sinusoidal positional encoding** to the embeddings, as in the original Transformer:

$$
\mathrm{PE}_{p,\,2i} = \sin\!\left(\frac{p}{10000^{2i/d}}\right), \qquad
\mathrm{PE}_{p,\,2i+1} = \cos\!\left(\frac{p}{10000^{2i/d}}\right)
$$

Each pair of dimensions is a sine and a cosine of the position $p$ with its own frequency $\omega_i = 10000^{-2i/d}$, ranging from one radian per position down to a very slow wave.

::: widget tf-03-posenc

Why this pattern? For each frequency, the pair $(\sin p\,\omega_i,\ \cos p\,\omega_i)$ is a point on a circle, and moving $k$ positions rotates it by the angle $k\,\omega_i$. So $\mathrm{PE}(p + k)$ is a fixed linear function of $\mathrm{PE}(p)$, the same for every $p$, and a layer can learn to look "three positions back" with one set of weights. The dot product of two encodings depends only on their distance:

$$
\mathrm{PE}(p) \cdot \mathrm{PE}(q) = \sum_i \cos\bigl((p - q)\,\omega_i\bigr)
$$

The table is computed once and added with an ordinary `+`. It is a constant, so it needs no backward pass and is not a parameter.

> [!abstract] Key points
> - Embedding: row lookup in a learned table; backward adds gradients into the rows that were read.
> - Attention ignores order, so a position signal is added to the embeddings.
> - Sinusoidal encoding: a shift by $k$ positions is a rotation, the same for every position.

## Attention as a weighted mean {#attention group="Attention" toc="Attention"}

Attention computes, for every position, a new vector as a **weighted mean** of vectors from all positions. Three roles are involved, each a linear projection of the input:

- the **query** $q$ of the position that is being updated: what it is looking for,
- a **key** $k_j$ for every position: what that position offers,
- a **value** $v_j$ for every position: what it passes on if chosen.

The weight of position $j$ is the softmax of the scaled dot products of the query with all keys, and the output is the weighted mean of the values:

$$
a_j = \frac{\exp(q \cdot k_j / \sqrt{d})}{\sum_l \exp(q \cdot k_l / \sqrt{d})}, \qquad
y = \sum_j a_j\, v_j
$$

::: widget tf-01-attention

The dot product is large when query and key point in the same direction, so attention is a soft lookup: instead of picking the best-matching key, it blends all values in proportion to how well they match. Because everything is differentiable, the projections that produce queries, keys and values are learned like any other weights.

## Scaled dot-product attention {#sdpa toc="Matrix form"}

For all positions at once, the queries, keys and values of a sequence of length $n$ are stacked into matrices $Q, K \in \mathbb R^{n \times d_k}$ and $V \in \mathbb R^{n \times d_v}$:

$$
\operatorname{Attention}(Q, K, V) = \operatorname{softmax}\!\left(\frac{Q K^\top}{\sqrt{d_k}} + M\right) V
$$

The softmax runs over each row, so every row of the $n \times n$ weight matrix sums to 1. $M$ is an optional **mask** added to the scores: an entry of $-\infty$ gives weight 0 after the softmax. A language model predicts the next token, so position $t$ must not see the positions after it. The **causal mask** has $-\infty$ above the diagonal and 0 elsewhere.

::: widget tf-02-mask

**Why divide by $\sqrt{d_k}$.** If the components of $q$ and $k$ are independent with variance 1, their dot product is a sum of $d_k$ terms and has variance $d_k$. With $d_k = 64$ the scores spread over a range of about $\pm 16$, the softmax puts almost all weight on one key, and its gradient becomes tiny. Dividing by $\sqrt{d_k}$ brings the variance back to 1.

In nabla, attention is not a `Function` of its own. It is written with operations that already have a backward pass:

```python
def scaled_dot_product_attention(Q, K, V, mask=None):
    d_k = Q.data.shape[-1]
    scores = matmul(Q, swap_last_two_axes(K)) * (1 / math.sqrt(d_k))
    if mask is not None:
        scores = scores + Tensor(mask)
    weights = softmax(scores, axis=-1)
    return matmul(weights, V)
```

Autograd assembles the backward pass from `MatMul`, `Multiply`, `Add` and `Softmax`. The one hand-written piece is the softmax backward, which multiplies the incoming gradient $g$ by the softmax Jacobian without forming it: for each row, $\partial L/\partial s = p \odot \bigl(g - \textstyle\sum_j g_j p_j\bigr)$. `MatMul` works on the last two axes and treats all leading axes as batch axes, so the same function serves a batch of sequences and, below, several heads at once. The $n \times n$ score matrix is also the main cost: memory and compute grow with the square of the sequence length.

> [!abstract] Key points
> - $\operatorname{softmax}(QK^\top/\sqrt{d_k} + M)\,V$: one weighted mean per query.
> - Causal mask: $-\infty$ above the diagonal, no look into the future.
> - Composed from existing operations; no new backward pass.

## Multi-head attention {#multi-head toc="Multi-head"}

One softmax produces one weighting per position. Often a position needs several at once, for example the previous character and the start of the current word. **Multi-head attention** runs $h$ attentions in parallel, each on its own $d/h$-dimensional slice of the projected queries, keys and values, and combines their results with a final projection:

```python
def forward(self, query, key=None, value=None, mask=None):
    key = query if key is None else key
    value = query if value is None else value
    Q = self._split_heads(self.q_proj(query))     # (B, n, d) → (B, h, n, d/h)
    K = self._split_heads(self.k_proj(key))
    V = self._split_heads(self.v_proj(value))
    out = scaled_dot_product_attention(Q, K, V, mask=mask)
    return self.out_proj(self._merge_heads(out))  # (B, h, n, d/h) → (B, n, d)
```

Splitting is a `reshape` to $(B, n, h, d/h)$ followed by a `transpose` to $(B, h, n, d/h)$; the heads become one more batch axis, and the attention function above handles them without a loop. Merging reverses both steps. Since each head works on $d/h$ dimensions, $h$ heads cost about as much as one head of full width. Single-head attention is just `num_heads=1`. If `key` and `value` come from another sequence, the same layer computes cross-attention.

## LayerNorm {#layernorm group="The block"}

Deep stacks of attention layers need normalization as well, but batch normalization is a poor fit for sequences: its statistics mix tokens from different sequences, including padding, and the output of one token depends on the rest of the batch. **Layer normalization** normalizes each token over its own $d$ features instead:

$$
\mu = \frac1d \sum_{i=1}^{d} x_i, \qquad
\sigma^2 = \frac1d \sum_{i=1}^{d} (x_i - \mu)^2, \qquad
y_i = \gamma_i\, \frac{x_i - \mu}{\sqrt{\sigma^2 + \epsilon}} + \beta_i
$$

::: widget tf-04-norm

Every token is normalized independently, so there are no running statistics and no difference between training and evaluation. The backward pass is the batch-normalization formula from the [Convolutional networks](cnn.html#batchnorm) page, with the sums taken over the feature axis and $N = d$. $\gamma$ and $\beta$ are per feature and receive gradients summed over all tokens.

## The Transformer block {#block toc="Transformer block"}

A block combines attention, which mixes information between positions, with a **feedforward** network applied to every position separately: `Linear(d, 4d)`, ReLU, `Linear(4d, d)`. Each of the two sublayers is wrapped in a **residual connection** and preceded by LayerNorm:

$$
x \leftarrow x + \operatorname{Attention}(\operatorname{LN}(x)), \qquad
x \leftarrow x + \operatorname{FFN}(\operatorname{LN}(x))
$$

::: widget tf-05-block

- **Residual connections.** The block computes a change to $x$, not a replacement. In the backward pass the `+` passes the gradient unchanged to $x$, so every block has a direct path to the loss, however many blocks are stacked.
- **Pre-norm.** The original Transformer normalized after the addition (post-norm). nabla, like GPT-2 and most later models, normalizes the input of each sublayer instead. The residual stream itself is then a plain sum of all sublayer outputs, which trains more stably in deep stacks.

Blocks are stacked in a `ModuleList`. A plain Python list would hide them from `Module.parameters()`, which only looks at attributes that are tensors or modules; `ModuleList` stores each block under an attribute of its own.

> [!abstract] Key points
> - LayerNorm: per token over its features; independent of the batch.
> - Block: $x + \operatorname{Attn}(\operatorname{LN}(x))$, then $x + \operatorname{FFN}(\operatorname{LN}(x))$.
> - Residuals give the gradient a direct path through every block.

## A character-level language model {#shakespeare group="Examples" toc="Shakespeare"}

[`examples/shakespeare_transformer.py`](https://github.com/maaaax1611/nabla/blob/main/examples/shakespeare_transformer.py) trains a model to continue text from Shakespeare's plays, one character at a time:

```python
def forward(self, token_ids, mask=None):
    x = self.token_embedding(token_ids)       # (B, n) → (B, n, 256)
    x = self.pos_encoding(x)
    for block in self.blocks:                 # 6 blocks, 8 heads each
        x = block(x, mask=mask)
    x = self.norm_out(x)
    return self.lm_head(x)                    # (B, n, 65): logits for the next character
```

- **Data.** Training samples are windows of 128 characters. The target sequence is the same window shifted by one, so every position is trained to predict the character that follows it.
- **Causal mask.** It lives in the example, not in `TransformerBlock`. Which positions may see each other is a property of the task; the ViT below uses the same block without a mask.
- **Loss.** The logits are flattened to $(B \cdot n, 65)$ and passed to `CrossEntropyLoss` with the flattened targets.
- **Training.** Adam with learning rate $3 \cdot 10^{-4}$, a warmup and cosine learning-rate schedule, gradient clipping, checkpoints and a metrics log.
- **Generation.** Start from a prompt, compute the logits for the last position, sample the next character from their softmax, append it and repeat, always keeping the last 128 characters as context. Generation runs under `no_grad()`.

## Vision Transformer {#vit toc="ViT"}

A Transformer block works on any sequence of vectors. The **Vision Transformer** (Dosovitskiy et al., 2020) turns an image into one: it cuts the image into patches and treats every patch as a token.

::: widget tf-06-vit

```python
def forward(self, images):
    x = self.patch_embed(images)                       # (B, 1, 28, 28) → (B, 49, 64)
    cls = self.cls_token + Tensor(zeros((B, 1, 64)))   # one CLS token per image
    x = F.concat([cls, x], axis=1)                     # (B, 50, 64)
    x = x + self.pos_embedding                         # learned, one vector per position
    for block in self.blocks:
        x = block(x)                                   # no mask
    return self.head(self.norm_out(x)[:, 0])           # classify from the CLS token
```

Four details differ from the language model, and each reuses an existing operation:

- **Patches via `Conv2D`.** Flattening each $p \times p$ patch and applying a linear layer is the same as a convolution with kernel size and stride $p$: the windows do not overlap, and each output channel is one output of the linear layer. `PatchEmbedding` is therefore one `Conv2D` followed by a reshape and transpose.
- **CLS token.** A learned vector is placed in front of the patches with `concat`. It belongs to no patch, gathers information from all of them through attention, and its final state is the input of the classifier. Its gradient is the sum over the batch, since the same parameter is broadcast to every image.
- **Learned positions.** The positional embedding is a parameter of shape $(1, n + 1, d)$ added to the sequence, instead of the fixed sinusoids.
- **Readout.** `x[:, 0]` selects the CLS token. Its backward pass writes the gradient into position 0 and zeros elsewhere.

On MNIST with 3 000 training images and patch size 4, a recorded run reached 81.6 % test accuracy after ten epochs. A CNN does better on this budget: convolutions have locality and translation built in, while a ViT has to learn them from data, which is why ViTs are usually trained on much larger datasets.

```bash
uv run python examples/shakespeare_transformer.py
uv run python examples/mnist_vit.py
```
