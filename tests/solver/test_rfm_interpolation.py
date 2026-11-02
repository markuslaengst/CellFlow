import jax.numpy as jnp
import numpy as np
import pytest

from cellflow.ode_solver import GuardedRFMInterpolation, RFMInterpolation, project_to_unit_sphere
from cellflow.preprocessing._premetric import SpectralNystroem, spectral_distance


class ConstantPremetric:
    def distance_grad_and_norm_batch(self, xt, x1):
        batch_size = xt.shape[0]
        return (
            jnp.ones((batch_size,), dtype=xt.dtype),
            jnp.ones_like(xt),
            jnp.ones((batch_size,), dtype=xt.dtype),
        )


class EuclideanPremetric:
    def distance_grad_and_norm_batch(self, xt, x1):
        diff = xt - x1
        d = jnp.linalg.norm(diff, axis=1)
        grad = diff / jnp.maximum(d[:, None], 1e-12)
        grad = jnp.where(d[:, None] == 0.0, jnp.zeros_like(grad), grad)
        g = jnp.sum(grad * grad, axis=1)
        return d, grad, g


class ScaledEuclideanPremetric:
    def __init__(self, scale):
        self.scale = scale

    def distance_grad_and_norm_batch(self, xt, x1):
        diff = xt - x1
        norm = jnp.linalg.norm(diff, axis=1)
        d = self.scale * norm
        grad = self.scale * diff / jnp.maximum(norm[:, None], 1e-12)
        grad = jnp.where(norm[:, None] == 0.0, jnp.zeros_like(grad), grad)
        g = jnp.sum(grad * grad, axis=1)
        return d, grad, g


def test_project_to_unit_sphere():
    x = jnp.array([[2.0, 0.0], [0.0, -3.0], [3.0, 4.0]])

    projected = np.asarray(project_to_unit_sphere(x))

    assert np.allclose(np.linalg.norm(projected, axis=1), 1.0)


def test_interpolate_applies_projection():
    interpolator = RFMInterpolation(ConstantPremetric(), projection_fn=project_to_unit_sphere)
    x0 = jnp.array([[1.0, 0.0]])
    x1 = jnp.array([[0.0, 1.0]])

    xt = np.asarray(interpolator.interpolate(x0, x1, t=0.5, steps=5))

    assert np.allclose(np.linalg.norm(xt, axis=1), 1.0)


def test_interpolate_defaults_do_not_clip_long_paths():
    interpolator = RFMInterpolation(EuclideanPremetric())
    x0 = jnp.array([[0.0]])
    x1 = jnp.array([[20.0]])

    xt = np.asarray(interpolator.interpolate(x0, x1, t=0.5, steps=64))

    assert np.allclose(xt, [[10.0]], atol=1e-4)


def test_interpolate_defaults_do_not_floor_small_gradient_norms():
    interpolator = RFMInterpolation(ScaledEuclideanPremetric(scale=1e-4))
    x0 = jnp.array([[0.0]])
    x1 = jnp.array([[1.0]])

    xt = np.asarray(interpolator.interpolate(x0, x1, t=1.0, steps=64))

    assert np.allclose(xt, [[1.0]], atol=1e-4)


def test_interpolate_accepts_per_sample_times():
    interpolator = RFMInterpolation(EuclideanPremetric())
    x0 = jnp.array([[0.0, 0.0], [0.0, 0.0]])
    x1 = jnp.array([[1.0, 0.0], [0.0, 2.0]])

    xt = np.asarray(interpolator.interpolate(x0, x1, t=jnp.array([0.5, 0.25]), steps=64))

    assert np.allclose(xt, [[0.5, 0.0], [0.0, 0.5]], atol=1e-4)


def test_spectral_distance_returns_unsquared_distance_by_default():
    psi = [[0.0, 0.0], [3.0, 4.0]]

    assert spectral_distance(0, 1, psi) == 5.0
    assert spectral_distance(0, 1, psi, squared=True) == 25.0


def test_spectral_nystroem_filters_unstable_components():
    x = np.array([[0.0], [1.0], [2.0]])
    phi = np.arange(12.0).reshape(3, 4) + 1.0
    eigvals = np.array([0.0, 0.5, 1.0, 1.5])
    psi = phi.copy()

    nystroem = SpectralNystroem(
        x,
        phi,
        eigvals,
        sigma=1.0,
        psi=psi,
        x_to_id=x,
        k=2,
        train_degrees=np.ones(3),
    )

    assert np.allclose(np.asarray(nystroem.lam), [0.5, 1.5])
    assert nystroem.Phi.shape == (3, 2)
    assert nystroem.Psi.shape == (3, 2)
    assert nystroem.n_dropped_spectral_components == 2
    assert np.all(np.isfinite(np.asarray(nystroem.embedding_scale)))


def test_trajectory_states_match_interpolate():
    interpolator = RFMInterpolation(EuclideanPremetric(), projection_fn=project_to_unit_sphere)
    x0 = jnp.array([[1.0, 0.0], [0.0, 1.0]])
    x1 = jnp.array([[0.0, 1.0], [-1.0, 0.0]])
    steps = 8

    ts, xs, us = interpolator.trajectory(x0, x1, steps)

    assert ts.shape == (steps, 2, 1) and xs.shape == (steps, 2, 2) and us.shape == (steps, 2, 2)
    np.testing.assert_allclose(np.asarray(ts[:, 0, 0]), np.arange(steps) / steps, atol=1e-7)
    np.testing.assert_allclose(np.asarray(xs[0]), np.asarray(x0), atol=1e-6)
    for k in range(1, steps):
        expected = interpolator.interpolate(x0, x1, t=k / steps, steps=k)
        np.testing.assert_allclose(np.asarray(xs[k]), np.asarray(expected), atol=1e-5)
        np.testing.assert_allclose(
            np.asarray(us[k]), np.asarray(interpolator.u_t(xs[k], x1, ts[k])), atol=1e-5
        )


def test_trajectory_offset_shifts_time_grid():
    interpolator = RFMInterpolation(EuclideanPremetric())
    x0 = jnp.array([[0.0], [0.0]])
    x1 = jnp.array([[1.0], [1.0]])
    offset = jnp.array([[0.25], [0.75]])

    ts, xs, _ = interpolator.trajectory(x0, x1, steps=4, offset=offset)

    np.testing.assert_allclose(np.asarray(ts[:, :, 0]), (np.arange(4)[:, None] + [0.25, 0.75]) / 4, atol=1e-7)
    # Euclidean RFM is the straight path x_t = t * x1, which Euler integrates exactly.
    np.testing.assert_allclose(np.asarray(xs[:, :, 0]), np.asarray(ts[:, :, 0]), atol=1e-5)


def test_trajectory_fixed_end_time_includes_interpolate_endpoint():
    interpolator = RFMInterpolation(EuclideanPremetric(), projection_fn=project_to_unit_sphere)
    x0 = jnp.array([[1.0, 0.0], [0.0, 1.0]])
    x1 = jnp.array([[0.0, 1.0], [-1.0, 0.0]])
    steps, t_end = 10, 0.99

    ts, xs, us = interpolator.trajectory(x0, x1, steps, t_end=t_end, include_end=True)

    assert ts.shape == (steps + 1, 2, 1) and xs.shape == (steps + 1, 2, 2)
    np.testing.assert_allclose(np.asarray(ts[:, 0, 0]), np.arange(steps + 1) * t_end / steps, atol=1e-6)
    np.testing.assert_allclose(
        np.asarray(xs[-1]), np.asarray(interpolator.interpolate(x0, x1, t=t_end, steps=steps)), atol=1e-5
    )
    np.testing.assert_allclose(np.asarray(us[-1]), np.asarray(interpolator.u_t(xs[-1], x1, t_end)), atol=1e-5)


def test_guarded_interpolation_projects_query_and_filters_velocity():
    def remove_radial_component(x, v):
        radial = x / jnp.linalg.norm(x, axis=1, keepdims=True)
        return v - jnp.sum(v * radial, axis=1, keepdims=True) * radial

    interpolator = GuardedRFMInterpolation(
        EuclideanPremetric(), projection_fn=project_to_unit_sphere, velocity_guard_fn=remove_radial_component
    )
    x = jnp.array([[2.0, 0.0], [0.0, 0.5]])
    x1 = jnp.array([[0.0, 1.0], [1.0, 0.0]])

    guarded = np.asarray(interpolator.u_t(x, x1, 0.0))
    projected = np.asarray(project_to_unit_sphere(x))
    raw_at_projection = np.asarray(RFMInterpolation(EuclideanPremetric()).u_t(jnp.asarray(projected), x1, 0.0))

    np.testing.assert_allclose(np.sum(guarded * projected, axis=1), 0.0, atol=1e-6)
    np.testing.assert_allclose(
        guarded, raw_at_projection - np.sum(raw_at_projection * projected, axis=1, keepdims=True) * projected, atol=1e-6
    )
    xt = np.asarray(interpolator.interpolate(x, x1, t=0.5, steps=4))
    np.testing.assert_allclose(np.linalg.norm(xt, axis=1), 1.0, atol=1e-6)


def test_guarded_interpolation_requires_projection_and_guard():
    with pytest.raises(ValueError, match="required"):
        GuardedRFMInterpolation(EuclideanPremetric(), projection_fn=None, velocity_guard_fn=lambda x, v: v)
