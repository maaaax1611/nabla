import os

import numpy as np
from shakespeare_data import load_shakespeare

from nabla.backend import gpu_available, to_device
from nabla.checkpoint import load_checkpoint
from nabla.tensor import Tensor
from nabla.nn.module import Module
from nabla.nn.embedding import Embedding
from nabla.nn.positional_encoding import PositionalEncoding
from nabla.nn.container import ModuleList
from nabla.nn.transformer import TransformerBlock
from nabla.nn.layernorm import LayerNorm
from nabla.nn.linear import Linear


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
    return np.triu(np.full((seq_len, seq_len), -1e9), k=1)


def generate(
        model: CharTransformerLM, tokenizer, prompt: str, num_new_tokens: int, block_size: int, device: str
) -> str:
    model.eval()
    ids = list(tokenizer.encode(prompt))
    for _ in range(num_new_tokens):
        context = np.array(ids[-block_size:])
        mask = causal_mask(len(context))
        logits = model(Tensor(context[None, :]).to(device), mask=mask)
        last_logits = to_device(logits.data[0, -1], "cpu")  # (vocab_size,) - plain ndarray, no autodiff needed here
        probs = np.exp(last_logits - last_logits.max())  # numerically stable softmax
        probs /= probs.sum()
        next_id = np.random.choice(len(probs), p=probs)
        ids.append(int(next_id))
    return tokenizer.decode(np.array(ids))


CHECKPOINT_PATH = os.path.join(os.path.dirname(__file__), ".shakespeare_large_checkpoint.pkl")

# must match examples/shakespeare_transformer.py's config exactly - the
# checkpoint only stores raw weight values (see Module.state_dict()), not
# the architecture that produced them, so there's nothing to check this
# against at load time; a mismatch here fails with a parameter-count error
# from Module.load_state_dict() at best, or silently loads garbage at worst.
BLOCK_SIZE = 128
EMBED_DIM = 256
NUM_HEADS = 8
HIDDEN_DIM = 1024
NUM_LAYERS = 6


def main() -> None:
    print("Loading Tiny Shakespeare (for the tokenizer's vocabulary - downloads + caches on first run)...")
    _, _, tokenizer = load_shakespeare()

    device = "cuda" if gpu_available() else "cpu"
    print(f"Running on: {device}")

    model = CharTransformerLM(
        vocab_size=tokenizer.vocab_size,
        embed_dim=EMBED_DIM,
        num_heads=NUM_HEADS,
        hidden_dim=HIDDEN_DIM,
        num_layers=NUM_LAYERS,
        max_len=BLOCK_SIZE,
        dropout=0.1,
    )
    model.to(device)  # before loading weights, same ordering rule as training - see docs/19-gpu-support.md

    prompt = "\n"
    num_new_tokens = 500

    # generate once with the fresh, randomly-initialized weights - before
    # touching the checkpoint at all - so there's an actual untrained
    # baseline to compare the trained output against, not just a guess at
    # what "random" would sound like
    print("\n=== Untrained (random init) ===\n")
    print(generate(model, tokenizer, prompt=prompt, num_new_tokens=num_new_tokens, block_size=BLOCK_SIZE, device=device))

    if os.path.exists(CHECKPOINT_PATH):
        metadata = load_checkpoint(CHECKPOINT_PATH, model)
        print(f"\n=== Trained (checkpoint step {metadata['step']}, train loss {metadata['train_loss']:.4f}) ===\n")
        print(generate(model, tokenizer, prompt=prompt, num_new_tokens=num_new_tokens, block_size=BLOCK_SIZE, device=device))
    else:
        print(f"\nNo checkpoint found at {CHECKPOINT_PATH} - run examples/shakespeare_transformer.py first "
              "to train one, then re-run this script to see the trained output too.")


if __name__ == "__main__":
    main()
