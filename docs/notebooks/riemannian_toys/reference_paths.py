"""Conditional reference paths of FM/RFM training compared to exact geodesics.

FM regresses the straight-line field ``u_t = x1 - x0``. RFM regresses the field that the
:class:`~cellflow.ode_solver.RFMInterpolation` Euler integration follows from ``x0`` towards ``x1``.
This module integrates both for coupled pairs, compares every path with the exact geodesic of the
domain, and summarizes the curve distances below.

Curve distances (all in the units of the coordinates):

``hausdorff``
    Largest distance from a point of either curve to the other curve: the worst local deviation.
``chamfer``
    Mean distance from a point of one curve to the other curve, averaged over both directions and
    taken uniformly along arc length: the typical deviation.
``wasserstein_time``
    W2 between the laws of ``x_t`` and ``gamma(t)`` for ``t ~ U[0, 1]``, where ``gamma`` is the
    geodesic at constant speed. It compares where each path spends its time, so it is also sensitive
    to the speed profile, not just the shape.
``wasserstein_arc``
    W2 between the uniform arc-length measures of the two curves. It compares shapes only, but unlike
    Hausdorff and Chamfer it requires mass to be matched, so a path that covers only part of the
    geodesic is penalized.
"""

import jax
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist

CURVE_METRICS = ["hausdorff", "chamfer", "wasserstein_time", "wasserstein_arc"]


# Reference paths -----------------------------------------------------------------------------------
def sample_coupled_pairs(coupling, source, target, n_pairs, rng):
    """``(x0, x1)`` pairs drawn from a (normalized) coupling matrix."""
    flat = coupling.ravel() / coupling.sum()
    source_ids, target_ids = np.unravel_index(rng.choice(flat.size, n_pairs, p=flat), coupling.shape)
    return source[source_ids], target[target_ids]


def fm_reference_paths(x0, x1, times):
    """FM conditional path ``x_t = (1 - t) x0 + t x1`` at ``times``, shape ``(n, m, d)``."""
    times = np.asarray(times)[None, :, None]
    return (1.0 - times) * np.asarray(x0)[:, None] + times * np.asarray(x1)[:, None]


def rfm_reference_paths(interpolator, x0, x1, steps):
    """RFM conditional paths: ``steps`` Euler steps of the training interpolator from 0 to 1.

    Returns ``(times, paths)`` with shapes ``(steps + 1,)`` and ``(n, steps + 1, d)``. The states are
    the same ones the training interpolation visits on the fixed grid ``t_k = k / steps``.
    """
    trajectory = jax.jit(interpolator.trajectory, static_argnums=2, static_argnames="include_end")
    ts, xs, _ = trajectory(np.asarray(x0, np.float32), np.asarray(x1, np.float32), steps, include_end=True)
    return np.asarray(ts)[:, 0, 0].astype(np.float64), np.asarray(xs).transpose(1, 0, 2).astype(np.float64)


# Curve distances -----------------------------------------------------------------------------------
def resample_by_arc_length(path, n_points):
    """``n_points`` points spaced uniformly along the arc length of a polyline ``(m, d)``."""
    cumulative = np.concatenate(([0.0], np.cumsum(np.linalg.norm(np.diff(path, axis=0), axis=1))))
    if cumulative[-1] <= 0.0:
        return np.repeat(path[:1], n_points, axis=0)
    s = np.linspace(0.0, cumulative[-1], n_points)
    return np.column_stack([np.interp(s, cumulative, path[:, k]) for k in range(path.shape[1])])


def point_to_polyline(points, polyline):
    """Distance from each point ``(p, d)`` to the polyline ``(m, d)``, segments included."""
    starts, segments = polyline[:-1], np.diff(polyline, axis=0)
    if len(segments) == 0:
        return np.linalg.norm(points - polyline[0], axis=1)
    offsets = points[:, None, :] - starts[None]
    fractions = np.clip(
        np.sum(offsets * segments[None], axis=-1) / np.maximum(np.sum(segments**2, axis=-1), 1e-20), 0.0, 1.0
    )
    closest = starts[None] + fractions[..., None] * segments[None]
    return np.linalg.norm(points[:, None, :] - closest, axis=-1).min(axis=1)


def wasserstein2(a, b):
    """W2 between two uniform empirical measures with the same number of atoms (exact assignment)."""
    cost = cdist(a, b, "sqeuclidean")
    rows, cols = linear_sum_assignment(cost)
    return float(np.sqrt(cost[rows, cols].mean()))


def curve_distances(path, geodesic_at_times, geodesic_dense, n_curve_points=256):
    """Hausdorff, Chamfer and both Wasserstein distances between one path and its geodesic.

    ``path`` and ``geodesic_at_times`` share the same (uniform) time grid; ``geodesic_dense`` is a
    finely sampled copy of the geodesic used as its polyline.
    """
    path_curve = resample_by_arc_length(path, n_curve_points)
    geodesic_curve = resample_by_arc_length(geodesic_dense, n_curve_points)
    path_to_geodesic = point_to_polyline(path_curve, geodesic_dense)
    geodesic_to_path = point_to_polyline(geodesic_curve, path)
    return {
        "hausdorff": float(max(path_to_geodesic.max(), geodesic_to_path.max())),
        "chamfer": float(0.5 * (path_to_geodesic.mean() + geodesic_to_path.mean())),
        "wasserstein_time": wasserstein2(path, geodesic_at_times),
        "wasserstein_arc": wasserstein2(path_curve, geodesic_curve),
    }


def compare_to_geodesics(geometry, x0, x1, times, paths, n_dense=1025, n_curve_points=256):
    """Per-pair curve distances of ``paths`` ``(n, m, d)`` at ``times`` to the geodesics of ``geometry``.

    Also reports the endpoint error ``|x_1 - path(1)|``, the ratio of path length to geodesic length,
    the fraction of the path's arc length covered by ``t = 0.5`` (0.5 for a constant-speed path) and,
    if the geometry defines ``winding_angle``, whether the path passes the hole on the other side than
    the geodesic.
    """
    geodesics_at_times, lengths = geometry.geodesic(x0, x1, times)
    geodesics_dense, _ = geometry.geodesic(x0, x1, np.linspace(0.0, 1.0, n_dense))
    half = int(np.argmin(np.abs(np.asarray(times) - 0.5)))
    rows = []
    for path, at_times, dense, x_end, length in zip(paths, geodesics_at_times, geodesics_dense, x1, lengths,
                                                    strict=True):
        rows.append({
            **curve_distances(path, at_times, dense, n_curve_points),
            "endpoint_error": float(np.linalg.norm(path[-1] - x_end)),
            "length_ratio": float(np.linalg.norm(np.diff(path, axis=0), axis=1).sum() / max(length, 1e-12)),
            "arc_fraction_at_half_time": float(arc_progress(path)[half]),
            "geodesic_length": float(length),
        })
    frame = pd.DataFrame(rows)
    # Dense resampling so that a chord between two saved states that cuts through the hole counts.
    dense_paths = np.stack([resample_by_arc_length(path, n_dense) for path in paths])
    leaves = geometry.violation(dense_paths).max(axis=1) > 1e-3
    frame["leaves_domain"] = leaves
    if hasattr(geometry, "winding_angle"):
        wrong = np.sign(geometry.winding_angle(paths)) != np.sign(geometry.winding_angle(geodesics_dense))
        frame["wrong_side"] = np.where(leaves, np.nan, wrong.astype(float))  # undefined through the hole
    return frame


def arc_progress(path):
    """Fraction of the total arc length covered at each saved state of a path ``(m, d)``."""
    cumulative = np.concatenate(([0.0], np.cumsum(np.linalg.norm(np.diff(path, axis=0), axis=1))))
    return cumulative / max(cumulative[-1], 1e-12)


SUMMARY_METRICS = (*CURVE_METRICS, "endpoint_error", "length_ratio", "arc_fraction_at_half_time")


def summarize(per_pair, metrics=SUMMARY_METRICS, stats=("mean", "median", "max")):
    """Statistics of each metric over pairs, for a dict ``{index: per-pair frame}``.

    Columns are ``(metric, statistic)``, plus the fraction of paths leaving the domain and, when
    available, the fraction of in-domain paths passing the hole on the wrong side. Tuple keys become a
    row MultiIndex.
    """
    rows = {}
    for key, frame in per_pair.items():
        rows[key] = {(metric, stat): frame[metric].agg(stat) for metric in metrics for stat in stats}
        rows[key]["leaves_domain", "fraction"] = frame["leaves_domain"].mean()
        if "wrong_side" in frame:
            rows[key]["wrong_side", "fraction"] = frame["wrong_side"].mean()  # among paths staying inside
    table = pd.DataFrame(rows).T
    table.columns = pd.MultiIndex.from_tuples(table.columns)
    if all(isinstance(key, tuple) for key in per_pair):
        table.index = pd.MultiIndex.from_tuples(table.index)
    return table


# Figures -------------------------------------------------------------------------------------------
def plot_reference_paths(geometry, x0, x1, paths, colors, geodesic_times, pair_ids, title=None, ncols=3):
    """Selected pairs: each reference path in ``paths`` (``{name: (n, m, d)}``) against the geodesic."""
    geodesics, _ = geometry.geodesic(x0[pair_ids], x1[pair_ids], geodesic_times)
    nrows = int(np.ceil(len(pair_ids) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.4 * ncols, 4.4 * nrows), squeeze=False)
    for ax, pair_id, geodesic in zip(axes.flat, pair_ids, geodesics, strict=False):
        ax.plot(*geodesic.T, color="black", linestyle="--", linewidth=2.5, label="geodesic", zorder=5)
        for name, path in paths.items():
            ax.plot(*path[pair_id].T, color=colors[name], linewidth=1.6, alpha=0.9, label=name)
        ax.scatter(*x0[pair_id], color="tab:blue", s=50, zorder=6)
        ax.scatter(*x1[pair_id], color="tab:orange", marker="X", s=70, zorder=6)
        geometry.finish_axes(ax)
        ax.set_title(f"pair {pair_id}")
    for ax in axes.flat[len(pair_ids):]:
        ax.set_visible(False)
    axes.flat[0].legend(loc="lower left", fontsize=8)
    if title:
        fig.suptitle(title)
    fig.tight_layout()
    plt.show()


def plot_arc_progress(paths, colors, title=None):
    """Mean fraction of arc length covered against time; ``paths`` maps names to ``(times, paths)``.

    A constant-speed geodesic follows the diagonal.
    """
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    ax.plot([0, 1], [0, 1], color="black", linestyle="--", linewidth=2, label="geodesic (constant speed)")
    for name, (times, trajectories) in paths.items():
        progress = np.stack([arc_progress(path) for path in trajectories])
        ax.plot(times, progress.mean(axis=0), color=colors[name], linewidth=2, label=name)
        ax.fill_between(times, *np.quantile(progress, [0.1, 0.9], axis=0), color=colors[name], alpha=0.15)
    ax.set(xlabel="t", ylabel="fraction of path length covered", xlim=(0, 1), ylim=(0, 1.02),
           title=title or "Speed profile of the reference paths (mean, 10-90% band)")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    plt.show()


def plot_distance_distributions(per_pair, colors, metrics=CURVE_METRICS, groups=None):
    """Box plots of per-pair distances; one panel per metric, one box per reference path and group.

    ``per_pair`` maps ``(group, name)`` to per-pair frames; ``colors`` maps ``name`` to a color.
    """
    groups = groups or list(dict.fromkeys(group for group, _ in per_pair))
    names = list(dict.fromkeys(name for _, name in per_pair))
    fig, axes = plt.subplots(1, len(metrics), figsize=(4.6 * len(metrics), 4.2))
    width = 0.8 / len(names)
    for ax, metric in zip(np.atleast_1d(axes), metrics, strict=True):
        for j, name in enumerate(names):
            positions = np.arange(len(groups)) + (j - (len(names) - 1) / 2) * width
            data = [per_pair[group, name][metric].to_numpy() for group in groups]
            box = ax.boxplot(data, positions=positions, widths=width * 0.9, patch_artist=True, showfliers=False)
            for patch in box["boxes"]:
                patch.set(facecolor=colors[name], alpha=0.6)
            for median in box["medians"]:
                median.set(color="black")
        ax.set_xticks(np.arange(len(groups)), groups)
        ax.set(title=metric.replace("_", " "), ylabel="distance to geodesic")
        ax.grid(alpha=0.25, axis="y")
    handles = [plt.Rectangle((0, 0), 1, 1, color=colors[name], alpha=0.6) for name in names]
    fig.legend(handles, names, loc="lower center", ncol=len(names), bbox_to_anchor=(0.5, -0.06))
    fig.tight_layout()
    plt.show()
