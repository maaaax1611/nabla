from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from numpy.typing import NDArray

from nabla.function import Function

if TYPE_CHECKING:
    from nabla.tensor import Tensor


class Conv2D(Function):
    """2D convolution: out = conv(x, weight) + bias.

    Shapes:
        x:      (batch, in_channels, H, W)
        weight: (out_channels, in_channels, kh, kw)
        bias:   (out_channels,)
        out:    (batch, out_channels, out_h, out_w)

    where out_h = (H + 2*padding - kh) // stride + 1  (analog für out_w).
    """

    def __init__(self, stride: int = 1, padding: int = 0) -> None:
        super().__init__()
        self.stride = stride
        self.padding = padding

    def forward(self, x: Tensor, weight: Tensor, bias: Tensor) -> NDArray:
        # TODO 1: Falls self.padding > 0, x.data mit np.pad auf allen Seiten
        #         der H/W-Achsen mit Nullen auffüllen. Ergebnis in self.x_padded
        #         speichern (brauchst du für backward!).
        #         Achtung: nur H/W padden, nicht batch/channel-Achsen.
        if self.padding > 0:
            self.x_padded = np.pad(
                x.data,
                ((0, 0), (0, 0), (self.padding, self.padding), (self.padding, self.padding)),
                mode="constant",
            )
        else:
            self.x_padded = x.data

        # TODO 2: out_h / out_w berechnen (siehe Docstring-Formel oben).
        kh = weight.data.shape[2] # kernel height
        kw = weight.data.shape[3] # kernel width
        out_h = (x.data.shape[2] + 2 * self.padding - kh) // self.stride + 1
        out_w = (x.data.shape[3] + 2 * self.padding - kw) // self.stride + 1

        # TODO 3: sliding_window_view(self.x_padded, (kh, kw), axis=(2, 3))
        #         anwenden. Ergebnis-Shape: (batch, C, out_h_all, out_w_all, kh, kw)
        #         (out_h_all/out_w_all = alle möglichen Fenster bei stride=1).
        windows = sliding_window_view(self.x_padded, (kh, kw), axis=(2, 3)) # (batch, C, out_h_all, out_w_all, kh, kw)


        # TODO 4: Stride berücksichtigen: von den "all"-Fenstern nur jedes
        #         self.stride-te in H- und W-Richtung nehmen
        #         (z.B. windows[:, :, ::stride, ::stride, :, :]).
        windows_strided = windows[:, :, ::self.stride, ::self.stride, :, :]

        # TODO 5: Achsen so umordnen (transpose) und reshapen, dass du eine
        #         "cols"-Matrix der Form (C*kh*kw, batch*out_h*out_w) bekommst.
        #         Tipp: erst transpose auf (C, kh, kw, batch, out_h, out_w),
        #         dann reshape.
        #         WICHTIG: cols für backward speichern (z.B. self.cols),
        #         genauso wie self.out_h, self.out_w, self.batch_size.
        self.cols = windows_strided.transpose(1, 4, 5, 0, 2, 3).reshape(weight.data.shape[1]*kh*kw, -1) # (C*kh*kw, batch*out_h*out_w)

        # TODO 6: weight.data zu (out_channels, C*kh*kw) reshapen.

        # TODO 7: weight_flat @ cols  ->  Shape (out_channels, batch*out_h*out_w)
        #         + bias (Broadcasting beachten: bias hat Shape (out_channels,),
        #         muss auf (out_channels, 1) reshaped werden zum Addieren).

        # TODO 8: Ergebnis zurück in (batch, out_channels, out_h, out_w) bringen.
        #         Achtung bei der Reihenfolge: das Ergebnis aus TODO 7 hat die
        #         Achsen-Reihenfolge (out_channels, batch, out_h, out_w) - erst
        #         transpose auf (batch, out_channels, out_h, out_w), dann erst
        #         ist reshape sicher (reshape allein reicht NICHT, da das die
        #         Speicher-Reihenfolge nicht ändert).

        raise NotImplementedError

    def backward(self, grad_output: NDArray) -> tuple[NDArray, NDArray, NDArray]:
        # grad_output shape: (batch, out_channels, out_h, out_w)

        # TODO 9: grad_bias = grad_output über batch, out_h, out_w summieren
        #         -> Shape (out_channels,)

        # TODO 10: grad_output so umformen, dass es zu deiner "cols"-Matrix aus
        #          forward passt: (out_channels, batch*out_h*out_w)
        #          (Achsen-Reihenfolge beachten - Umkehrung von TODO 8!)

        # TODO 11: grad_weight_flat = grad_output_flat @ self.cols.T
        #          -> Shape (out_channels, C*kh*kw), dann zurück zu
        #          weight.data.shape reshapen.

        # TODO 12: grad_cols = weight_flat.T @ grad_output_flat
        #          -> Shape (C*kh*kw, batch*out_h*out_w)
        #          Das ist der Gradient bezüglich der extrahierten Patches.

        # TODO 13: grad_x_padded mit Nullen initialisieren, gleiche Shape wie
        #          self.x_padded.
        #          Schleife über alle Ausgabepositionen (i in range(out_h),
        #          j in range(out_w)):
        #            - hole die passende Spalte/Spalten aus grad_cols für
        #              Position (i, j) über alle batches
        #            - reshape zu (batch, C, kh, kw)
        #            - addiere (+=!) in das entsprechende Fenster von
        #              grad_x_padded bei
        #              [:, :, i*stride:i*stride+kh, j*stride:j*stride+kw]
        #          WICHTIG: += nicht =, da sich Fenster bei stride < kernel
        #          überlappen können und Gradienten sich dort aufaddieren müssen!

        # TODO 14: Falls self.padding > 0, das Padding von grad_x_padded wieder
        #          abschneiden (Slicing), um auf die Original-Shape von x zu
        #          kommen.

        raise NotImplementedError
