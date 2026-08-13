# Transformer Block

## The idea

Every building block from the last several docs —
[`Attention`](10-multi-head-attention.md), [`LayerNorm`](07-layernorm.md),
[`Dropout`](06-dropout.md) — comes together here into one reusable unit:
[`nn/transformer.py`](../src/nabla/nn/transformer.py)'s `TransformerBlock`,
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

## The feedforward sublayer

[`nn/feedforward.py`](../src/nabla/nn/feedforward.py) is deliberately
small — `Linear(embed_dim, hidden_dim) -> ReLU -> Linear(hidden_dim,
embed_dim)`, applied identically and independently to every position in
the sequence (hence "position-wise": no mixing across the sequence axis at
all, that already happened in attention). Attention lets positions
exchange information; the feedforward sublayer is where the model actually
does per-position computation on what it gathered.

## Residual connections: why `x + sublayer(x)`, not just `sublayer(x)`

Without the `x +`, a stack of $N$ `TransformerBlock`s would force gradients
to flow back through every single sublayer in sequence to reach an early
layer — the same vanishing-gradient problem that makes any sufficiently
deep network hard to train. With the residual connection, the `+` gives
`backward()` a direct, unimpeded path straight back through every block
(`Add`'s backward is just the identity on both branches — see
[Autodiff basics](01-autodiff.md)), regardless of how many blocks are
stacked in between. The sublayer only has to learn a *correction* to add
on top of `x`, not reproduce `x` itself in order to pass it through.

## Pre-norm vs. post-norm

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

## The mask stays generic

`TransformerBlock.forward(x, mask=None)` just forwards `mask` straight
through to [`Attention`](10-multi-head-attention.md) — it doesn't build or
know anything about what kind of mask gets passed in. That's intentional:
a causal (look-ahead) mask for language modeling, a padding mask for
variable-length batches, or no mask at all for a plain encoder are all
*task*-specific decisions, not something a generic reusable block should
hardcode. Building the actual causal mask is left to whatever example uses
this block for language modeling.

## Testing

[`tests/test_nn_transformer.py`](../tests/test_nn_transformer.py) checks
shapes, gradients through every parameter (all four sub-modules:
`attn`, `norm1`, `ff`, `norm2`), that `eval()`/`train()` correctly
propagates down to the internal `Dropout`, and one behavioral property of
masking that's worth calling out specifically: with a causal mask applied,
drastically perturbing the *last* position in a sequence must leave every
*earlier* position's output completely unchanged — a direct, executable
check that the mask is doing what "causal" is supposed to mean, not just
that shapes come out right.
