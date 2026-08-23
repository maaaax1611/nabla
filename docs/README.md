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
17. [Vision Transformer (ViT)](17-vision-transformer.md) — turning images
    into patch sequences via `Conv2D`, a CLS token via `Concat`, and a
    learned positional embedding. (Originally picked the CLS token out
    via a one-hot `MatMul` for lack of an indexing op — see
    [doc 24](24-slicing.md) for how that got replaced.)
18. [Example: MNIST with a Vision Transformer](18-vit-example.md) — training
    `VisionTransformer` end to end, and why it trails a CNN's accuracy
    at the same budget (a real, well-known ViT property, not a bug).
19. [GPU Support via CuPy](19-gpu-support.md) — the `get_array_module`
    dispatch trick that lets every op run on NumPy or CuPy unchanged,
    `Tensor`/`Module.to(device)`, and four device-mismatch bugs that
    only surfaced by actually running on a GPU.
20. [Learning-Rate Schedules](20-lr-schedules.md) — why a scheduler is
    just an external object mutating `optimizer.lr`, warmup + cosine
    decay, and why that keeps it fully decoupled from the optimizer.
21. [Gradient Clipping](21-gradient-clipping.md) — clipping one combined
    norm across every parameter instead of each independently, so a
    clipped update keeps its direction and only loses magnitude.
22. [Checkpointing](22-checkpointing.md) — `state_dict()`/`load_state_dict()`
    on `Module` and `Optimizer`, resuming a run (including its LR
    schedule) from the last saved step instead of from scratch.
23. [Logging](23-logging.md) — tracking metrics logged at different
    step cadences (train loss every step, val loss every eval interval)
    without padding gaps, exported as tidy/long-format CSV.
24. [Slicing](24-slicing.md) — basic indexing (`x[:, 0]`, `x[1:3]`), why its
    backward pass is a plain scatter instead of `Embedding`-style
    scatter-accumulate, and retiring the one-hot-matmul CLS-token hack
    from the Vision Transformer now that real indexing exists.
25. [Graph Memory and GPU Stalls](25-graph-memory-and-gpu-stalls.md) — two
    reference cycles (`Tensor`/`Function`, and a recursive closure inside
    `backward()` itself) that left every step's graph alive until
    Python's cyclic GC happened to run, and why that turned into
    multi-second GPU stalls only once the model got big enough to matter.
26. [Upsample](26-upsample.md) — nearest-neighbor upsampling for a U-Net
    decoder, why its backward is a reshape-and-sum that mirrors forward's
    `repeat`, and a first (wrong) `roll`-based attempt that shows why.
27. [Dice Loss](27-dice-loss.md) — overlap-based segmentation loss instead
    of per-pixel cross-entropy, why `eps` placement matters for the
    empty-mask edge case, and the quotient-rule backward that needs a
    reshape to broadcast correctly against pixel-shaped gradients.
28. [U-Net](28-unet.md) — encoder-decoder segmentation architecture with
    skip connections, composing existing ops with no new backward math.
    First doc written in the newer, more structured
    Overview/Math/Implementation/Testing format.
29. [no_grad](29-no-grad.md) — a context manager to skip graph
    construction entirely for forward passes that never call
    `.backward()` (validation/inference), avoiding gigabytes of retained
    im2col buffers that a real training loop uncovered on an 8GB GPU.
30. [BCEWithLogitsLoss](30-bce-with-logits.md) — fused sigmoid +
    cross-entropy for numerical stability, the same clean
    `sigmoid(x) - y` gradient shape as `SoftmaxCrossEntropy`, and why
    it's typically combined with Dice loss to escape the "predict all
    background" plateau pure Dice gets stuck in on rare-class targets.

Each doc also calls out real bugs that came up while implementing these ops —
they're often more informative than the happy path.
