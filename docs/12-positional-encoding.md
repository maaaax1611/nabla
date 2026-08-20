# Positional Encoding

## The idea

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

## Why it needs no new `Function`

Unlike [`Embedding`](11-embedding.md), this table is not learned — it's a
fixed mathematical pattern, computed once from `embed_dim` and `max_len`
and then simply **added** to the input:

```python
return x + Tensor(self.table[:seq_len])
```

Adding a constant only shifts the input; $\partial(x+c)/\partial x = 1$
either way, so the existing `Add` `Function` (see
[Autodiff basics](01-autodiff.md)) already produces the correct gradient
with no new backward derivation needed — the constant Tensor just never
has `requires_grad=True`, so its own gradient gets discarded the same 
way `targets`/`indices` are elsewhere in the library.

## The formula

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

## Why this specific formula, not something simpler

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

## Testing

[`tests/test_nn_positional_encoding.py`](../tests/test_nn_positional_encoding.py)
checks the table directly (bounded values, `sin(0)=0`/`cos(0)=1` at
position 0, every position gets a distinct row) and the layer's behavior
(same table slice broadcast across every batch element, sequence-length and
embed-dim validation, and — since this op involves no new math — that
`backward` really does just pass the gradient straight through unchanged).
`PositionalEncoding.parameters()` is also checked to return nothing: the
table is a plain `ndarray` attribute, never wrapped in a `Tensor`, so
[`Module.parameters()`](01-autodiff.md) never picks it up as trainable.
