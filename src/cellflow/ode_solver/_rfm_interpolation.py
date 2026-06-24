from cellflow.preprocessing._premetric import SpectralNystroem
import jax.numpy as jnp


class RFMInterpolation:

    def __init__(self, nystroem: SpectralNystroem):
        self.nystroem = nystroem

    def interpolate(self, x0, x1, t, steps):
        dt = t / steps
        xt = x0.copy()
        current_t = 0
        for _ in range(steps):
            v = self.u_t(xt, x1, current_t)
            xt = xt + dt * v
            current_t = current_t + dt

        return xt

    def u_t(self, xt, x1, t):
        d, grad_d, g = self.nystroem.distance_grad_and_norm_batch(xt, x1)

        u_t = (d[:, None] * grad_d) / g[:, None] / (1 - t)

        return u_t




