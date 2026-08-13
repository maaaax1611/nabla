from __future__ import annotations

from typing import Iterable, Iterator

from nabla.nn.module import Module


class ModuleList(Module):
    """A list of submodules that Module.parameters()/train()/eval() can see.

    A plain Python list assigned as an attribute (`self.blocks = [...]`)
    would be invisible to Module's introspection: parameters()/train()/
    eval() only recurse into attributes that are themselves a Module or
    Tensor, and a list is neither. Storing each submodule under its own
    generated attribute name (`self._module_0`, `self._module_1`, ...)
    makes every one of them show up automatically through that same
    existing mechanism - no separate handling needed anywhere else. This
    is the same reason PyTorch has `nn.ModuleList` instead of letting a
    plain list just work.

    Args:
        modules: An iterable of Modules to store, in order.
    """

    def __init__(self, modules: Iterable[Module]) -> None:
        super().__init__()
        self._items: list[Module] = list(modules)
        for i, module in enumerate(self._items):
            setattr(self, f"_module_{i}", module)

    def __getitem__(self, index: int) -> Module:
        return self._items[index]

    def __iter__(self) -> Iterator[Module]:
        return iter(self._items)

    def __len__(self) -> int:
        return len(self._items)
