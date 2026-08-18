import numpy as np

from nabla.tensor import Tensor


class TestGraphClearedAfterBackward:
    """Tensor._ctx and Function.saved_tensors form a reference cycle
    (child -> Function -> parent tensors, and every non-leaf parent has
    the same shape of link one level further back). Plain refcounting
    can never collect a cycle on its own - only Python's cyclic garbage
    collector can, and it doesn't run on every allocation. Left alone,
    each step's graph (and everything it holds, including GPU arrays on
    CuPy tensors) only gets freed whenever that collector happens to run.
    Tensor.backward() now breaks the cycle itself once it's done with the
    graph, freeing it via plain refcounting instead of waiting on gc.
    """

    def test_intermediate_and_root_graph_links_are_cleared(self):
        x = Tensor(np.array([1.0, 2.0, 3.0]), requires_grad=True)
        y = Tensor(np.array([4.0, 5.0, 6.0]), requires_grad=True)
        z = x + y  # intermediate node
        out = z * x  # root

        assert z._ctx is not None and z._prev != ()
        assert out._ctx is not None and out._prev != ()

        out.backward()

        assert out._ctx is None and out._prev == ()
        assert z._ctx is None and z._prev == ()

    def test_leaf_tensors_are_untouched(self):
        # leaves never had a graph link to begin with - backward() must
        # not disturb them, and they must stay usable in a fresh graph
        x = Tensor(np.array([2.0]), requires_grad=True)
        y = Tensor(np.array([3.0]), requires_grad=True)
        (x * y).backward()

        assert x._ctx is None and x._prev == ()
        assert y._ctx is None and y._prev == ()

        # same leaves, new computation - must still work after backward
        x.grad = None
        y.grad = None
        (x + y).backward()
        assert np.array_equal(x.grad, np.array([1.0]))
        assert np.array_equal(y.grad, np.array([1.0]))

    def test_gradients_are_unaffected_by_clearing(self):
        # the fix must only drop graph *links*, never the accumulated
        # gradients those links were used to compute
        x = Tensor(np.array([1.0, 2.0]), requires_grad=True)
        y = Tensor(np.array([3.0, 4.0]), requires_grad=True)
        out = (x * y).sum()
        out.backward()

        assert np.array_equal(x.grad, y.data)
        assert np.array_equal(y.grad, x.data)

    def test_no_cyclic_garbage_left_behind(self):
        import gc

        x = Tensor(np.random.randn(50, 50), requires_grad=True)
        w = Tensor(np.random.randn(50, 50), requires_grad=True)

        gc.collect()
        gc.disable()
        try:
            out = (x.matmul(w)).sum()
            out.backward()
            del out
            # nothing left for the cyclic collector to do - a plain
            # refcounting pass (gc.collect() while disabled still runs
            # the cycle-finder) should find no garbage from this graph
            collected = gc.collect()
            assert collected == 0
        finally:
            gc.enable()
