from cellflow.preprocessing._premetric import SpectralNystroem
import jax.numpy as np


class RFMInterpolation:

    def __init__(self, nystroem: SpectralNystroem):
        self.nystroem = nystroem

    def interpolate(self, x0, x1, t, steps):
        dt = t / steps
        xt = x0.copy()

        for _ in range(steps):
            v = self.u_t(xt, x1)
            xt = xt + dt * v

        return xt

    def u_t(self, xt, x1):

        # ----- spectral squared distance -----
        d2 = self.nystroem.spectral_distance(xt, x1)
        d = np.sqrt(d2 + 1e-12)

        # ----- gradient of squared distance -----
        grad_d2 = self.nystroem.spectral_distance_grad(xt, x1)

        # convert to gradient of distance
        grad_d = grad_d2 / (2.0 * d + 1e-12)

        # ----- Riemannian norm term -----
        g = self.nystroem.g_norm_fast(xt, x1) + 1e-12

        # ----- RFM vector field -----
        u_t = - (d * grad_d) / g

        return u_t




