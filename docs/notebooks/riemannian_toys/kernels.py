"""Kernel-profile comparison on a toy domain: fit, raw fields, Euler trajectories, refinement."""

import time

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd

from cellflow.ode_solver import RFMInterpolation
from cellflow.preprocessing import fit_kernel_nystroem

KERNEL_PROFILES = {
    "fixed Gaussian": "fixed_gaussian",
    "variable Gaussian": "gaussian",
    "compact polynomial": "polynomial",
    "smooth bump": "bump",
    "binary CkNN": "indicator",
}
"""Display name → :class:`~cellflow.preprocessing._kernels.SymmetricBandwidthKernel` profile."""

RAW_MODE = "unprojected"


def fit_kernel_profiles(construction_points, profiles=None, *, density_k, delta, n_modes, laplacian_type,
                        spectral_weight_type, reference_profile="binary CkNN"):
    """Fit one Nyström extension per kernel profile on the same construction points.

    Returns
    -------
    ``(models, kernels, graphs, fit_df)``: dictionaries keyed by profile name, and a table of
    fit diagnostics with the edge count relative to ``reference_profile``.
    """
    profiles = KERNEL_PROFILES if profiles is None else profiles
    models, kernels, graphs, rows = {}, {}, {}, []
    for kernel_name, profile in profiles.items():
        fit = fit_kernel_nystroem(
            construction_points, profile, density_k=density_k, delta=delta, n_modes=n_modes,
            laplacian_type=laplacian_type, spectral_weight_type=spectral_weight_type,
        )
        models[kernel_name], kernels[kernel_name], graphs[kernel_name] = fit.model, fit.kernel, fit.graph
        rows.append({"kernel": kernel_name, **fit.diagnostics})
        print(f"{kernel_name}: fitted in {fit.diagnostics['fit_seconds']:.2f}s", flush=True)
    fit_df = pd.DataFrame(rows).set_index("kernel")
    if reference_profile in fit_df.index:
        fit_df["retained_edge_fraction"] = fit_df["edges"] / fit_df.loc[reference_profile, "edges"]
    return models, kernels, graphs, fit_df


def gradient_check(model, query, target, difference_step=1e-6):
    """Relative error of the analytic distance gradient against autodiff and central differences.

    Runs in float64. Returns ``(autodiff_error, finite_difference_error)``.
    """
    with jax.enable_x64():
        query = jnp.asarray(query, dtype=jnp.float64)
        target = jnp.asarray(target, dtype=jnp.float64)

        def distance(point):
            residual = model.extend_point(point) - model.extend_point(target)
            return jnp.sqrt(jnp.dot(residual, residual) + model.eps)

        _, analytic, _ = model.distance_grad_and_norm(query, target)
        autodiff = jax.grad(distance)(query)
        finite_difference = np.array([
            (float(distance(query + direction)) - float(distance(query - direction))) / (2.0 * difference_step)
            for direction in np.eye(len(query)) * difference_step
        ])
        scale = max(float(jnp.linalg.norm(analytic)), 1e-10)
        return (
            float(jnp.linalg.norm(analytic - autodiff)) / scale,
            float(np.linalg.norm(np.asarray(analytic) - finite_difference)) / scale,
        )


def support_counts(kernel, points):
    """Number of construction points with nonzero affinity at each query point."""
    return np.asarray(jax.jit(jax.vmap(lambda point: jnp.count_nonzero(kernel(point))))(jnp.asarray(points)))


def raw_field_diagnostics(models, kernels, geometry, field_points, target, time, gradient_query):
    """Evaluate each raw RFM field on ``field_points`` and check gradients and probe behavior.

    Returns
    -------
    ``(fields, field_df)``: velocities per kernel and a diagnostics table.
    """
    field_points = jnp.asarray(field_points)
    targets = jnp.broadcast_to(jnp.asarray(target), field_points.shape)
    probe_targets = jnp.broadcast_to(jnp.asarray(target), geometry.probe_points.shape)
    fields, rows = {}, []
    for kernel_name, model in models.items():
        interpolator = RFMInterpolation(model, project_to_manifold=False)
        velocities = np.asarray(jax.jit(interpolator.u_t)(field_points, targets, time))
        if not np.all(np.isfinite(velocities)):
            raise ValueError(f"Non-finite vector field for {kernel_name}.")
        speeds = np.linalg.norm(velocities, axis=1)
        autodiff_error, finite_difference_error = gradient_check(model, gradient_query, target)
        if autodiff_error > 1e-5:
            raise AssertionError(f"{kernel_name}: analytic/autodiff gradient mismatch.")
        probe_velocities = np.asarray(jax.jit(interpolator.u_t)(jnp.asarray(geometry.probe_points), probe_targets, time))
        fields[kernel_name] = velocities
        rows.append({
            "kernel": kernel_name,
            "median_speed": np.median(speeds),
            "p95_speed": np.quantile(speeds, 0.95),
            "zero_speed_fraction": np.mean(speeds < 1e-8),
            "unsupported_probe_fraction": np.mean(support_counts(kernels[kernel_name], field_points) == 0),
            **geometry.probe_diagnostic(probe_velocities),
            "autodiff_relative_error": autodiff_error,
            "finite_difference_relative_error": finite_difference_error,
        })
    return fields, pd.DataFrame(rows).set_index("kernel")


def integrate_euler_paths(interpolator, starts, target, final_time, steps):
    """Euler-integrate the RFM field from ``t=0`` to ``final_time`` towards one target.

    Each step is ``x ← project(x + h u_t(x))``. Returns ``(path, proposed)`` with shapes
    ``(steps + 1, n, d)`` (including the starts) and ``(steps, n, d)`` (states before projection).
    """
    starts = jnp.asarray(starts)
    targets = jnp.broadcast_to(jnp.asarray(target), starts.shape)
    time_step = final_time / steps
    step_times = np.linspace(0.0, final_time, steps + 1, dtype=np.float32)[:-1]

    def integrate(initial):
        def step(state, current_time):
            proposed = state + time_step * interpolator.u_t(state, targets, current_time)
            next_state = interpolator.project(proposed)
            return next_state, (next_state, proposed)

        _, (path, proposed_path) = jax.lax.scan(step, initial, jnp.asarray(step_times))
        return jnp.concatenate((initial[None], path), axis=0), proposed_path

    path, proposed = jax.jit(integrate)(starts)
    return np.asarray(path), np.asarray(proposed)


def make_interpolator(model, geometry, guarded):
    return geometry.guarded_interpolator(model) if guarded else RFMInterpolation(model, project_to_manifold=False)


def spectral_distance_trace(model, path, target, stride=8):
    """Spectral distance to the target at every ``stride``-th saved state, shape ``(n_saved, n)``."""
    sampled = jnp.asarray(path[::stride])
    target = jnp.broadcast_to(jnp.asarray(target), sampled.shape[1:])
    return np.asarray(jax.jit(lambda states: jax.lax.map(
        lambda batch: model.distance_grad_and_norm_batch(batch, target)[0], states
    ))(sampled))


def trajectory_comparison(models, kernels, geometry, starts, target, final_time, steps):
    """Raw and guarded Euler trajectories for every kernel, with endpoint and path diagnostics.

    Returns
    -------
    ``(trajectories, distance_traces, trajectory_df)``; the dictionaries are keyed by
    ``(kernel_name, mode)`` with ``mode`` in ``("unprojected", geometry.guarded_mode)``.
    """
    trajectories, distance_traces, rows = {}, {}, []
    for kernel_name, model in models.items():
        for mode, guarded in [(RAW_MODE, False), (geometry.guarded_mode, True)]:
            started = time.perf_counter()
            path, proposed = integrate_euler_paths(make_interpolator(model, geometry, guarded), starts, target,
                                                   final_time, steps)
            if not np.all(np.isfinite(path)):
                raise ValueError(f"Non-finite trajectories for {kernel_name}, {mode}.")
            distances = spectral_distance_trace(model, path, target)
            final_errors = np.linalg.norm(path[-1] - target, axis=1)
            initial_errors = np.linalg.norm(path[0] - target, axis=1)
            trajectories[kernel_name, mode] = path
            distance_traces[kernel_name, mode] = distances
            rows.append({
                "kernel": kernel_name,
                "mode": mode,
                "mean_final_euclidean": np.mean(final_errors),
                "p90_final_euclidean": np.quantile(final_errors, 0.9),
                "mean_relative_euclidean": np.mean(final_errors / initial_errors),
                "fraction_within_0.1": np.mean(final_errors < 0.1),
                "mean_final_relative_spectral": np.mean(distances[-1] / np.maximum(distances[0], 1e-12)),
                "mean_path_length": np.mean(np.linalg.norm(np.diff(path, axis=0), axis=-1).sum(axis=0)),
                f"max_{geometry.name}_violation": np.max(geometry.violation(path)),
                "max_proposed_violation": np.max(geometry.violation(proposed)),
                "unsupported_final_fraction": np.mean(support_counts(kernels[kernel_name], path[-1]) == 0),
                "integration_seconds": time.perf_counter() - started,
                **geometry.path_geometry(path),
            })
            print(
                f"{kernel_name}, {mode}: mean endpoint error {np.mean(final_errors):.4f}, "
                f"{np.mean(final_errors < 0.1):.0%} within 0.1; {rows[-1]['integration_seconds']:.2f}s",
                flush=True,
            )
    return trajectories, distance_traces, pd.DataFrame(rows)


def passes_segment_checks(row, geometry, tolerance=1e-3):
    return all(row[column] <= tolerance for column in geometry.segment_check_columns)


def report_best_guarded_kernel(trajectory_df, geometry, tolerance=1e-3):
    """Print the guarded profile with the lowest endpoint error among those passing all segment checks."""
    guarded = trajectory_df[trajectory_df["mode"].eq(geometry.guarded_mode)].sort_values("mean_final_euclidean")
    valid = guarded[
        guarded.apply(lambda row: passes_segment_checks(row, geometry, tolerance), axis=1)
        & guarded["unsupported_final_fraction"].eq(0.0)
    ]
    if len(valid):
        best = valid.iloc[0]
        print(f"Lowest endpoint error passing domain/segment checks: {best['kernel']} "
              f"({best['mean_final_euclidean']:.4f}).")
    print("Binary CkNN is a graph baseline: its almost-everywhere zero gradients "
          "leave the moving-point interpolation stationary.")
    print(f"{geometry.name.capitalize()} projection enforces the domain. Compare the raw {geometry.probe_field_label} "
          "diagnostic and unprojected paths to assess the pre-metric itself.")


def refinement_check(models, geometry, coarse_trajectories, starts, target, final_time, steps):
    """Repeat the guarded integration with ``steps`` Euler steps and compare with the coarse run.

    Returns
    -------
    ``(refined_trajectories, refinement_df)``.
    """
    refined_trajectories, rows = {}, []
    for kernel_name, model in models.items():
        started = time.perf_counter()
        refined, _ = integrate_euler_paths(geometry.guarded_interpolator(model), starts, target, final_time, steps)
        if not np.all(np.isfinite(refined)):
            raise ValueError(f"{kernel_name}: non-finite refined trajectories.")
        refined_errors = np.linalg.norm(refined[-1] - target, axis=1)
        coarse = coarse_trajectories[kernel_name, geometry.guarded_mode]
        geometry_metrics = geometry.path_geometry(refined)
        refined_trajectories[kernel_name] = refined
        rows.append({
            "kernel": kernel_name,
            "coarse_mean_endpoint_error": np.mean(np.linalg.norm(coarse[-1] - target, axis=1)),
            "refined_mean_endpoint_error": np.mean(refined_errors),
            "mean_endpoint_shift": np.linalg.norm(refined[-1] - coarse[-1], axis=1).mean(),
            "refined_fraction_within_0.1": np.mean(refined_errors < 0.1),
            "passes_domain_segment_checks": passes_segment_checks(geometry_metrics, geometry),
            **geometry_metrics,
            "refinement_seconds": time.perf_counter() - started,
        })
        print(
            f"{kernel_name}, {steps} guarded steps: mean endpoint error {np.mean(refined_errors):.4f}; "
            f"maximum hole penetration {geometry_metrics['max_segment_hole_penetration']:.4f}",
            flush=True,
        )
    return refined_trajectories, pd.DataFrame(rows).set_index("kernel")
