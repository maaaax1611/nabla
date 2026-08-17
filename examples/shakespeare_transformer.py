"""Tiny character-level Transformer trained on Tiny Shakespeare.

Combines every Transformer building block documented in docs/09 through
docs/14: Embedding + PositionalEncoding feed a stack of TransformerBlocks
(held in a ModuleList), followed by a final LayerNorm and a Linear "head"
projecting back to vocabulary size. Trained with a causal mask so position
t can only attend to positions <= t - the actual language-modeling task
(predict the next character) is what makes that mask necessary, which is
why it's built here rather than inside TransformerBlock itself (see
docs/13-transformer-block.md).

This is a small proof-of-concept, not a serious language model: nabla runs
on plain NumPy (no GPU), so the model and training budget here are kept
deliberately small to finish in a reasonable time on a CPU. Don't expect
Shakespeare-quality output - expect recognizable word-shapes and structure
emerging from what started as random noise, which is the point.
"""

import numpy as np
from shakespeare_data import get_batch, load_shakespeare

from nabla.nn.container import ModuleList
from nabla.nn.embedding import Embedding
from nabla.nn.layernorm import LayerNorm
from nabla.nn.linear import Linear
from nabla.nn.loss import CrossEntropyLoss
from nabla.nn.module import Module
from nabla.nn.positional_encoding import PositionalEncoding
from nabla.nn.transformer import TransformerBlock
from nabla.optim.adam import Adam
from nabla.tensor import Tensor


class CharTransformerLM(Module):
    def __init__(
        self,
        vocab_size: int,
        embed_dim: int,
        num_heads: int,
        hidden_dim: int,
        num_layers: int,
        max_len: int,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.token_embedding = Embedding(vocab_size, embed_dim)
        self.pos_encoding = PositionalEncoding(embed_dim, max_len)
        self.blocks = ModuleList(
            [TransformerBlock(embed_dim, num_heads, hidden_dim, dropout) for _ in range(num_layers)]
        )
        self.norm_out = LayerNorm(embed_dim)
        self.lm_head = Linear(embed_dim, vocab_size)

    def forward(self, token_ids: Tensor, mask: np.ndarray | None = None) -> Tensor:
        x = self.token_embedding(token_ids)
        x = self.pos_encoding(x)
        for block in self.blocks:
            x = block(x, mask=mask)
        x = self.norm_out(x)
        return self.lm_head(x)


def causal_mask(seq_len: int) -> np.ndarray:
    """Position i may attend to positions <= i, never to the future."""
    return np.triu(np.full((seq_len, seq_len), -1e9), k=1)


def compute_loss(model: CharTransformerLM, X: np.ndarray, y: np.ndarray, mask: np.ndarray, criterion) -> Tensor:
    logits = model(Tensor(X), mask=mask)  # (batch, block_size, vocab_size)
    batch, block_size, vocab_size = logits.data.shape
    logits_flat = logits.reshape((batch * block_size, vocab_size))
    targets_flat = y.reshape(batch * block_size)
    return criterion(logits_flat, targets_flat)


def generate(model: CharTransformerLM, tokenizer, prompt: str, num_new_tokens: int, block_size: int) -> str:
    model.eval()
    ids = list(tokenizer.encode(prompt))
    for _ in range(num_new_tokens):
        context = np.array(ids[-block_size:])
        mask = causal_mask(len(context))
        logits = model(Tensor(context[None, :]), mask=mask)
        last_logits = logits.data[0, -1]  # (vocab_size,) - next-token prediction
        probs = np.exp(last_logits - last_logits.max())
        probs /= probs.sum()
        next_id = np.random.choice(len(probs), p=probs)
        ids.append(int(next_id))
    model.train()
    return tokenizer.decode(np.array(ids))


def main() -> None:
    print("Loading Tiny Shakespeare (downloads + caches on first run)...")
    train_ids, val_ids, tokenizer = load_shakespeare()
    print(f"Vocab size: {tokenizer.vocab_size}, train chars: {len(train_ids)}, val chars: {len(val_ids)}")

    block_size = 64
    batch_size = 16
    steps = 1000
    eval_interval = 100

    model = CharTransformerLM(
        vocab_size=tokenizer.vocab_size,
        embed_dim=64,
        num_heads=4,
        hidden_dim=256,
        num_layers=3,
        max_len=block_size,
        dropout=0.1,
    )
    criterion = CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=3e-4)
    mask = causal_mask(block_size)
    rng = np.random.default_rng(0)

    print(f"Model parameters: {sum(p.data.size for p in model.parameters()):,}")

    model.train()
    for step in range(1, steps + 1):
        X, y = get_batch(train_ids, block_size, batch_size, rng)
        loss = compute_loss(model, X, y, mask, criterion)

        model.zero_grad()
        loss.backward()
        optimizer.step()

        if step % eval_interval == 0 or step == 1:
            model.eval()
            X_val, y_val = get_batch(val_ids, block_size, batch_size, rng)
            val_loss = compute_loss(model, X_val, y_val, mask, criterion)
            model.train()
            print(f"Step {step:4d} | Train loss: {float(loss.data):.4f} | Val loss: {float(val_loss.data):.4f}")

    print("\nSample generation (random weights would look like noise; compare to the loss trend above):")
    print(generate(model, tokenizer, prompt="\n", num_new_tokens=300, block_size=block_size))


if __name__ == "__main__":
    main()
