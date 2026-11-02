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

    def trajectory(self, x0, x1, steps, offset=None, t_end=1.0, include_end=False):
        """Integrate the RFM path from ``t=0`` towards ``t_end`` and record every Euler state.

        With ``h = t_end / steps``, the time grid of sample ``i`` is ``t_k = (offset_i + k) h`` for
        ``k = 0, ..., steps - 1``. A first Euler step of length ``offset_i h`` moves from ``t=0`` to
        ``t_0``; every later step has length ``h``. With ``offset ~ U[0, 1)`` and ``t_end=1``, every
        recorded time ``t_k`` is uniform on ``[k / steps, (k + 1) / steps)``, so the recorded times
        cover ``[0, 1)`` uniformly. With ``offset=None`` the grid is the fixed ``t_k = k h``.

        Parameters
        ----------
        x0
            Source points of shape ``(n, d)``.
        x1
            Target points of shape ``(n, d)``.
        steps
            Number of recorded states (and full-length Euler steps).
        offset
            Per-sample grid offsets in ``[0, 1)`` of shape ``(n,)`` or ``(n, 1)``. :obj:`None` uses
            the fixed grid ``t_k = k h``.
        t_end
            End time of the integration, in ``(0, 1]``.
        include_end
            Also record the state reached after the last Euler step, at ``t = (offset + steps) h``
            (``t_end`` for a zero offset), with its target velocity. This is the state that
            :meth:`interpolate` returns.

        Returns
        -------
        ``(t, x, u)`` with shapes ``(m, n, 1)``, ``(m, n, d)`` and ``(m, n, d)``, where ``m = steps``
        (``steps + 1`` with ``include_end``): the times, the interpolated states, and the RFM target
        velocity at each state, which is also the velocity used for the following Euler step.
        """
        if steps <= 0:
            raise ValueError("steps must be positive.")
        if not 0.0 < t_end <= 1.0:
            raise ValueError(f"t_end must be in (0, 1], got {t_end}.")

        x0 = jnp.asarray(x0)
        x1 = jnp.asarray(x1)
        if x0.shape != x1.shape:
            raise ValueError(f"x0 and x1 must have the same shape, got {x0.shape} and {x1.shape}.")

        n = x0.shape[0]
        offset = jnp.zeros((n, 1), dtype=x0.dtype) if offset is None else jnp.asarray(offset, dtype=x0.dtype)
        offset = offset.reshape((n, 1))
        dt = t_end / steps

        # First (partial) step from t=0 to t_0 = offset * dt.
        t_start = offset * dt
        x = self.project(x0 + t_start * self.u_t(x0, x1, jnp.zeros_like(offset)))

        def body(x, k):
            t = (offset + k) * dt
            u = self.u_t(x, x1, t)
            x_next = self.project(x + dt * u)
            return x_next, (t, x, u)

        x_last, (ts, xs, us) = jax.lax.scan(body, x, jnp.arange(steps, dtype=x0.dtype))
        if include_end:
            t_last = (offset + steps) * dt
            u_last = self.u_t(x_last, x1, t_last)
            ts = jnp.concatenate((ts, t_last[None]), axis=0)
            xs = jnp.concatenate((xs, x_last[None]), axis=0)
            us = jnp.concatenate((us, u_last[None]), axis=0)
        return ts, xs, us

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


class GuardedRFMInterpolation(RFMInterpolation):
    """RFM interpolation that evaluates the field on the manifold and filters its velocity.

    :meth:`u_t` first projects the query with ``projection_fn``, evaluates the RFM target
    velocity there, and passes ``(projected_point, velocity)`` to ``velocity_guard_fn``.
    Typical guards remove the normal component (tangent-space projection) or the
    outward component at a boundary (tangent-cone projection).

    Parameters
    ----------
    nystroem
        Spectral premetric, as in :class:`RFMInterpolation`.
    projection_fn
        Point projection onto the domain. Also used after every Euler step.
    velocity_guard_fn
        Function ``(x, v) -> v`` applied to the velocity at the projected point ``x``.
    kwargs
        Further keyword arguments for :class:`RFMInterpolation`.
    """

    def __init__(self, nystroem: SpectralNystroem, projection_fn: Callable, velocity_guard_fn: Callable, **kwargs):
        if projection_fn is None or velocity_guard_fn is None:
            raise ValueError("projection_fn and velocity_guard_fn are required.")
        super().__init__(nystroem, projection_fn=projection_fn, **kwargs)
        self.velocity_guard_fn = velocity_guard_fn

    def u_t(self, xt, x1, t):
        projected = self.projection_fn(xt)
        return self.velocity_guard_fn(projected, super().u_t(projected, x1, t))
