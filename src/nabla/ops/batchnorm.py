from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

from nabla.function import Function

if TYPE_CHECKING:
    from nabla.tensor import Tensor


class BatchNorm2D(Function):
    """2D batch normalization using per-batch statistics (training mode).

    out = gamma * (x - mean) / sqrt(var + eps) + beta

    Shapes:
        x:     (batch, channels, H, W)
        gamma: (channels,)
        beta:  (channels,)
        out:   (batch, channels, H, W)

    mean/var werden pro Channel über (batch, H, W) berechnet, d.h. jeder
    Channel wird unabhängig von den anderen normalisiert. Anders als bei
    Conv2D/Pooling ändert sich die Shape hier nicht.

    Hinweis: Das hier ist "nur" die differenzierbare Normalisierung mit den
    Batch-Statistiken (training mode). Das Tracking von running_mean/
    running_var für den eval-Modus passiert später separat im nn-Layer,
    nicht hier in der Function.
    """

    def __init__(self, eps: float = 1e-5) -> None:
        super().__init__()
        self.eps = eps

    def forward(self, x: Tensor, gamma: Tensor, beta: Tensor) -> NDArray:
        # TODO 1: batch_mean berechnen: Mittelwert von x.data über die Achsen
        #         (0, 2, 3) (batch, H, W), mit keepdims=True, damit du später
        #         gegen x broadcasten kannst. Shape: (1, C, 1, 1).
        self.batch_mean = np.mean(x.data, axis=(0, 2, 3), keepdims=True) # (1, C, 1, 1)

        # TODO 2: batch_var berechnen: Varianz von x.data über dieselben
        #         Achsen (0, 2, 3), keepdims=True. WICHTIG: biased variance
        #         verwenden (durch N teilen, nicht N-1) - also entweder
        #         np.var(..., ddof=0) oder manuell
        #         mean((x - batch_mean)**2). Shape: (1, C, 1, 1).
        self.batch_var = np.var(x.data, axis=(0, 2, 3), keepdims=True, ddof=0) # (1, C, 1, 1)

        # TODO 3: x_hat berechnen: (x.data - batch_mean) / sqrt(batch_var + eps).
        #         Das ist der normalisierte Input, Shape (batch, C, H, W).
        #         WICHTIG: x_hat für backward speichern (self.x_hat), genauso
        #         wie batch_var (self.batch_var) und alles andere, was du in
        #         backward brauchst (z.B. self.N = batch*H*W, die Anzahl
        #         Elemente pro Channel, über die gemittelt wurde).
        self.x_centered = x.data - self.batch_mean # (batch, C, H, W)
        self.x_hat = self.x_centered / np.sqrt(self.batch_var + self.eps) # (batch, C, H, W)
        batch, _, H, W = x.data.shape
        self.N = batch * H * W


        # TODO 4: gamma/beta auf (1, C, 1, 1) reshapen, damit sie gegen
        #         x_hat broadcasten. gamma und beta selbst (nicht nur ihre
        #         Werte) für backward speichern, z.B. über
        #         self.save_for_backward(gamma, beta).
        gamma_reshaped = gamma.data.reshape(1, -1, 1, 1)
        beta_reshaped = beta.data.reshape(1, -1, 1, 1)
        self.save_for_backward(gamma, beta)

        # TODO 5: out = gamma_reshaped * x_hat + beta_reshaped zurückgeben.
        out = gamma_reshaped * self.x_hat + beta_reshaped
        return out

    def backward(self, grad_output: NDArray) -> tuple[NDArray, NDArray, NDArray]:
        # grad_output shape: (batch, channels, H, W)

        # TODO 6: grad_beta = grad_output über (0, 2, 3) summieren.
        #         -> Shape (channels,). (beta wurde nur addiert, also ist das
        #         die Summe aller eingehenden Gradienten pro Channel.)
        grad_beta = np.sum(grad_output, axis=(0, 2, 3)) # (channels,)

        # TODO 7: grad_gamma = (grad_output * self.x_hat) über (0, 2, 3)
        #         summieren. -> Shape (channels,).
        #         (out = gamma * x_hat + beta, also d(out)/d(gamma) = x_hat.)
        grad_gamma = np.sum(grad_output * self.x_hat, axis=(0, 2, 3)) # (channels,)

        # TODO 8: grad_x_hat = grad_output * gamma_reshaped
        #         (gamma wieder auf (1, C, 1, 1) reshapen wie in forward).
        #         Shape (batch, C, H, W). Das ist der Gradient bezüglich des
        #         normalisierten Inputs - jetzt musst du durch die
        #         Normalisierung "durchdifferenzieren" (TODO 9-11).
        gamma, _ = self.saved_tensors
        gamma_reshaped = gamma.data.reshape(1, -1, 1, 1)
        grad_x_hat = grad_output * gamma_reshaped # (batch, C, H, W)

        # TODO 9: grad_var berechnen (Gradient bezüglich batch_var):
        #         grad_var = sum(grad_x_hat * (x - batch_mean) * -0.5 *
        #                        (batch_var + eps)^(-1.5),
        #                        axis=(0,2,3), keepdims=True)
        #         Du brauchst hier (x - batch_mean) wieder - entweder x.data
        #         nochmal aus saved_tensors holen und batch_mean neu
        #         berechnen, oder (x - batch_mean) direkt in forward
        #         zusätzlich speichern (z.B. self.x_centered).
        grad_var = np.sum(grad_x_hat * self.x_centered * -0.5 * (self.batch_var + self.eps)**(-1.5), axis=(0, 2, 3), keepdims=True)

        # TODO 10: grad_mean berechnen (Gradient bezüglich batch_mean):
        #          grad_mean = sum(grad_x_hat * -1/sqrt(batch_var + eps),
        #                          axis=(0,2,3), keepdims=True)
        #                    + grad_var * mean(-2 * (x - batch_mean),
        #                                      axis=(0,2,3), keepdims=True)
        #          (zweiter Term: batch_var hängt auch von batch_mean ab,
        #          daher der zusätzliche Pfad über die Kettenregel.)
        grad_mean = np.sum(grad_x_hat * -1 / np.sqrt(self.batch_var + self.eps), axis=(0, 2, 3), keepdims=True)
        grad_mean += grad_var * np.mean(-2 * self.x_centered, axis=(0, 2, 3), keepdims=True)

        # TODO 11: grad_x zusammensetzen (drei Pfade addieren):
        #          grad_x = grad_x_hat / sqrt(batch_var + eps)
        #                 + grad_var * 2 * (x - batch_mean) / self.N
        #                 + grad_mean / self.N
        #          Shape (batch, C, H, W).

        grad_x = grad_x_hat / np.sqrt(self.batch_var + self.eps)
        grad_x += grad_var * 2 * self.x_centered / self.N
        grad_x += grad_mean / self.N

        return grad_x, grad_gamma, grad_beta
