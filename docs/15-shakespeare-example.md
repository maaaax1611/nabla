# Example: A Character-Level Transformer on Tiny Shakespeare

## What this ties together

[`examples/shakespeare_transformer.py`](../examples/shakespeare_transformer.py)
is where every building block from docs 06 through 14 finally comes
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

## Why character-level, not BPE

The task is next-character prediction on the Tiny Shakespeare corpus.
Tokenization here is deliberately the simplest possible scheme — one
integer per distinct character, no merges, no subword logic (see
[`examples/shakespeare_data.py`](../examples/shakespeare_data.py)'s
`CharTokenizer`). Implementing byte-pair encoding was considered and
rejected: BPE is a text-preprocessing algorithm, not an autodiff
concern, and would add real scope without exercising any new part of
nabla. Character-level tokenization keeps the example focused on what
it's actually meant to demonstrate — the Transformer internals — while
still needing a real `Embedding` lookup table.

## Why a causal mask, and why it's built here and not in `TransformerBlock`

Language modeling means position `t` can only attend to positions
`<= t` — it must never see the token it's trying to predict, or any
token after it. [`docs/13-transformer-block.md`](13-transformer-block.md)
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

## Training loop

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

## Generation

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

## Running it

```bash
uv run python examples/shakespeare_transformer.py
```

The corpus downloads once and is cached under
`examples/.shakespeare_cache/` (git-ignored) on first run.
