from collections.abc import Callable

import jax.numpy as jnp

from cellflow.preprocessing._premetric import SpectralNystroem


def project_to_unit_sphere(x, eps: float = 1e-12):
    """Project points to the boundary of the unit sphere."""
    norm = jnp.linalg.norm(x, axis=-1, keepdims=True)
    return x / jnp.maximum(norm, eps)


class RFMInterpolation:

    def __init__(self, nystroem: SpectralNystroem, projection_fn: Callable | None = None):
        self.nystroem = nystroem
        self.projection_fn = projection_fn

    def interpolate(self, x0, x1, t, steps):
        dt = t / steps
        xt = x0.copy()
        current_t = 0
        for _ in range(steps):
            v = self.u_t(xt, x1, current_t)
            xt = xt + dt * v
            if self.projection_fn is not None:
                xt = self.projection_fn(xt)
            current_t = current_t + dt

        return xt

    def u_t(self, xt, x1, t):
        d, grad_d, g = self.nystroem.distance_grad_and_norm_batch(xt, x1)

        u_t = (d[:, None] * grad_d) / g[:, None] / (1 - t)

        return u_t




