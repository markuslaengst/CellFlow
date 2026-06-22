from cellflow.preprocessing._premetric import SpectralNystroem
import jax.numpy as jnp


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
        d, grad_d, g = self.nystroem.distance_grad_and_norm_batch(xt, x1)

        u_t = -(d[:, None] * grad_d) / g[:, None]

        return u_t




