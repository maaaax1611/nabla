from __future__ import annotations

from typing import TYPE_CHECKING

from numpy.typing import NDArray

from nabla.backend import get_array_module
from nabla.function import Function

if TYPE_CHECKING:
    from nabla.tensor import Tensor


class Upsample(Function):
    """Nearest-neighbor upsampling by an integer scale factor.

    Shapes:
        x:   (batch, channels, H, W)
        out: (batch, channels, H * scale, W * scale)

    Forward: jedes Pixel wird zu einem scale x scale Block dupliziert.
    Backward: TODO - überlege dir, was die Umkehrung von "duplizieren"
    für den Gradienten bedeutet. Wenn ein Eingabe-Pixel im Forward-Pass
    zu mehreren Ausgabe-Pixeln beigetragen hat, was passiert dann mit
    deren jeweiligen Gradienten beim Zurückfließen?
    """

    def __init__(self, scale: int) -> None:
        super().__init__()
        self.scale = scale

    def forward(self, x: Tensor) -> NDArray:
        xp = get_array_module(x.data)
        self.input_shape = x.data.shape

        # TODO: erzeuge Output mit Shape (N, C, H*scale, W*scale),
        # wobei jeder Wert scale x scale mal wiederholt wird.
        # Hinweis: xp.repeat entlang der richtigen Achsen, zweimal
        # angewendet (einmal für H, einmal für W).
        out_h = xp.repeat(x.data, self.scale, axis=2)
        out = xp.repeat(out_h, self.scale, axis=3)
        return out

    def backward(self, grad_output: NDArray) -> tuple[NDArray]:
        # TODO: grad_output hat Shape (N, C, H*scale, W*scale).
        # Jedes Eingabe-Pixel hat scale*scale Kopien im Output erzeugt -
        # summiere die Gradienten jedes scale x scale Fensters zu einem
        # einzigen Wert pro Eingabe-Pixel zusammen (downsampling der
        # Gradienten - siehe deine Antwort dazu).
        xp = get_array_module(grad_output)
        b, c, h, w = self.input_shape
        # reshape grad_output to (b, c, h, scale, w, scale) so that we can sum over the scale dimensions
        grad_reshaped = grad_output.reshape(b, c, h, self.scale, w, self.scale)
        # sum over the scale dimensions (axis 3 and 5)
        grad_input = grad_reshaped.sum(axis=(3, 5))
        return (grad_input,)