# Multi-Head Attention

## The idea

Running [scaled dot-product attention](09-attention.md) once gives a model
one single way of relating positions to each other — one set of Q/K/V
projections, one attention pattern. Multi-head attention instead projects
Q/K/V into `num_heads` smaller subspaces, runs attention independently in
each, and concatenates the results back together, letting different heads
specialize in different kinds of relationships (e.g. one head tracking
nearby positions, another tracking a specific long-range dependency) instead
of forcing everything through one shared attention pattern.

Crucially, single-head attention with learned Q/K/V/output projections
*isn't a special case* that needs separate code — it's just `num_heads=1`.
[`nn/attention.py`](../src/nabla/nn/attention.py) is a single `Attention`
class covering both: nothing in the implementation branches on whether
`num_heads` is 1 or more.

## Splitting into heads

Given `embed_dim` and `num_heads` (which must divide it evenly), each head
operates on a `head_dim = embed_dim // num_heads` slice. Projected Q/K/V
start out as `(batch, seq_len, embed_dim)` and need to become
`(batch, num_heads, seq_len, head_dim)` — a reshape to expose the head axis,
followed by a transpose to move it before the sequence axis (so
[scaled dot-product attention's batching](09-attention.md#shapes-and-batching)
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
differentiable, so (like [scaled dot-product attention](09-attention.md)
itself) no new backward pass needed anywhere in this layer.

## The four projections

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

## Self-attention vs. cross-attention

`forward(query, key=None, value=None, mask=None)` defaults `key`/`value` to
`query` when omitted, so `layer(x)` runs self-attention (every position
attends to every other position in the same sequence — the Transformer
encoder case). Passing `key`/`value` explicitly (e.g. an encoder's output,
with a possibly different sequence length than `query`) runs
cross-attention instead — the same code path either way, since Q/K/V are
always projected and split independently and
[scaled dot-product attention](09-attention.md) never assumed
`seq_len_q == seq_len_k` in the first place.

## Testing

[`tests/test_nn_attention.py`](../tests/test_nn_attention.py) checks shapes
for self- and cross-attention and for `num_heads=1`, that `_split_heads` and
`_merge_heads` are exact inverses of each other, gradients against a
numerical check (through all four projections, not just the attention core),
and one behavioral property of masking: perturbing a key/value position that
every query is masked away from must leave the output completely unchanged.
