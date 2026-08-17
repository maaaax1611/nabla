# nabla docs

nabla is a small automatic differentiation library built on NumPy, modeled on
PyTorch's design. These docs walk through the ideas behind each building
block — the math, the derivation of the backward pass, and how that maps
onto the actual code — rather than just restating what the code does.

Read them in order if you're new to the codebase; each one builds on the
last.

1. [Autodiff basics](01-autodiff.md) — `Tensor`, `Function`, the computational
   graph, and how `backward()` turns the chain rule into a topological sort.
2. [Conv2D](02-conv2d.md) — 2D convolution via the im2col trick.
3. [Pooling](03-pooling.md) — max and average pooling, and why their
   gradients look so different from each other.
4. [BatchNorm2D](04-batchnorm.md) — batch normalization, with the full
   chain-rule derivation through the batch mean and variance.
5. [Softmax + Cross-Entropy](05-softmax-cross-entropy.md) — why softmax and
   cross-entropy are implemented as a single fused op, and how that produces
   one of the simplest gradients in the whole library.
6. [Dropout](06-dropout.md) — regularization via random masking, why it's
   the only op in nabla that isn't a pure function of its input, and how
   that changes the way its gradient gets tested.
7. [LayerNorm](07-layernorm.md) — per-sample normalization instead of
   BatchNorm2D's per-batch statistics, and why that means no running stats
   and no train/eval split.
8. [Softmax](08-softmax.md) — the standalone op, and why its backward pass
   needs softmax's actual Jacobian-vector product instead of the shortcut
   `SoftmaxCrossEntropy` gets to take.
9. [Scaled Dot-Product Attention](09-attention.md) — why, unlike every op
   before it, this one needed no new backward derivation at all — just
   batched matmul and composition of ops that already existed.
10. [Multi-Head Attention](10-multi-head-attention.md) — splitting Q/K/V
    into heads via reshape + transpose, and why single-head attention is
    just the `num_heads=1` case, not separate code.
11. [Embedding](11-embedding.md) — a learnable lookup table, and why its
    backward needs the same scatter-accumulate trick as `MaxPool2D` once
    an ID repeats within a call.
12. [Positional Encoding](12-positional-encoding.md) — the fixed sin/cos
    signal that gives attention a sense of order, and why the specific
    formula makes relative positions learnable as a linear operation.
13. [Transformer Block](13-transformer-block.md) — assembling Attention,
    LayerNorm, and a feedforward sublayer into one reusable unit, and why
    residual connections and pre-norm placement matter for training deep
    stacks of them.
14. [ModuleList](14-module-list.md) — why stacking layers in a plain
    Python list silently hides their parameters, and how `ModuleList`
    fixes that with no new introspection logic at all.
15. [Example: Character-Level Transformer on Tiny Shakespeare](15-shakespeare-example.md) —
    every building block from docs 06-14 assembled into an actual
    trained model, with a causal mask for next-character prediction.
16. [Concat](16-concat.md) — the first op that takes a variable number of
    tensors, needed to prepend a learnable CLS token to a sequence, and
    an off-by-one bug in its backward pass that numerical gradient
    checking caught immediately.

Each doc also calls out real bugs that came up while implementing these ops —
they're often more informative than the happy path.
