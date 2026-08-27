from collections.abc import Callable

import jax
import jax.numpy as jnp

from cellflow.preprocessing._premetric import SpectralNystroem


def project_to_unit_sphere(x, eps: float = 1e-12):
    """Project points to the boundary of the unit sphere."""
    norm = jnp.linalg.norm(x, axis=-1, keepdims=True)
    return x / jnp.maximum(norm, eps)


def project_to_nearest_point(x, manifold_points):
    """Project each point to its nearest manifold support point."""
    x = jnp.asarray(x)
    manifold_points = jnp.asarray(manifold_points)

    if manifold_points.ndim != 2:
        raise ValueError("manifold_points must have shape (n_points, n_features).")
    if x.shape[-1] != manifold_points.shape[-1]:
        raise ValueError(
            "x and manifold_points must have the same feature dimension, "
            f"got {x.shape[-1]} and {manifold_points.shape[-1]}."
        )

    flat_x = x.reshape((-1, manifold_points.shape[-1]))

    def nearest_point(point):
        squared_distances = jnp.sum((manifold_points - point) ** 2, axis=1)
        return manifold_points[jnp.argmin(squared_distances)]

    projected = jax.vmap(nearest_point)(flat_x)
    return projected.reshape(x.shape)


class RFMInterpolation:
    """Riemannian flow-matching interpolation using a spectral distance."""

    def __init__(
        self,
        nystroem: SpectralNystroem,
        projection_fn: Callable | None = None,
        project_to_manifold: bool = False,
        min_gradient_norm_sq: float = 0.0,
        time_epsilon: float = 1e-6,
        max_velocity_norm: float | None = None,
    ):
        if min_gradient_norm_sq < 0:
            raise ValueError("min_gradient_norm_sq must be non-negative.")
        if time_epsilon <= 0:
            raise ValueError("time_epsilon must be positive.")
        if max_velocity_norm is not None and max_velocity_norm <= 0:
            raise ValueError("max_velocity_norm must be positive or None.")
        if project_to_manifold and not hasattr(nystroem, "X"):
            raise ValueError("project_to_manifold=True requires nystroem to expose training points as `X`.")

        self.nystroem = nystroem
        self.projection_fn = projection_fn
        self.project_to_manifold = project_to_manifold
        self.min_gradient_norm_sq = min_gradient_norm_sq
        self.time_epsilon = time_epsilon
        self.max_velocity_norm = max_velocity_norm

    def project(self, x):
        if self.projection_fn is not None:
            x = self.projection_fn(x)
        if self.project_to_manifold:
            x = project_to_nearest_point(x, self.nystroem.X)
        return x

    def interpolate(self, x0, x1, t, steps):
        if steps <= 0:
            raise ValueError("steps must be positive.")

        xt = jnp.asarray(x0)
        x1 = jnp.asarray(x1)
        if xt.shape != x1.shape:
            raise ValueError(f"x0 and x1 must have the same shape, got {xt.shape} and {x1.shape}.")

        dt = jnp.asarray(t) / steps
        if dt.ndim == 1:
            dt = dt[:, None]
        current_t = jnp.zeros_like(dt)
        for _ in range(steps):
            v = self.u_t(xt, x1, current_t)
            xt = xt + dt * v
            xt = self.project(xt)
            current_t = current_t + dt

        return xt

    def u_t(self, xt, x1, t):
        d, grad_d, g = self.nystroem.distance_grad_and_norm_batch(xt, x1)

        gradient_norm_sq = jnp.maximum(g, self.min_gradient_norm_sq)
        gradient_norm_sq = jnp.where(
            gradient_norm_sq > 0.0,
            gradient_norm_sq,
            jnp.finfo(gradient_norm_sq.dtype).tiny,
        )
        time_remaining = jnp.maximum(1.0 - jnp.asarray(t), self.time_epsilon)
        if time_remaining.ndim > 0:
            time_remaining = time_remaining.reshape((-1, 1))

        u_t = -(d[:, None] * grad_d) / gradient_norm_sq[:, None] / time_remaining

        if self.max_velocity_norm is not None:
            velocity_norm = jnp.linalg.norm(u_t, axis=1, keepdims=True)
            clip_scale = jnp.minimum(1.0, self.max_velocity_norm / jnp.maximum(velocity_norm, 1e-12))
            u_t = u_t * clip_scale

        return u_t
