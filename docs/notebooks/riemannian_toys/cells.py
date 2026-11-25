"""Helpers for Riemannian flow matching on real single-cell data in a few genes.

Used by ``riemannian_eb_genes_example.ipynb``: loading a gene view of an AnnData time course,
a stratified, group-aware train/validation/test split, the graph construction set, a
domain description without an analytic manifold, metrics and 3-D plots.
"""

from dataclasses import dataclass

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.optimize import linear_sum_assignment
from scipy.spatial import cKDTree
from scipy.spatial.distance import cdist

SPLITS = ("train", "val", "test")


# Data --------------------------------------------------------------------------------------------
def find_genes(var_names, symbols):
    """``var_names`` entries whose gene symbol (text before a space or ``(``) matches each symbol."""
    base = pd.Series(np.asarray(var_names), index=var_names).str.split(r"[\s(]", n=1, regex=True).str[0].str.upper()
    found = []
    for symbol in symbols:
        matches = base.index[base == symbol.upper()]
        if len(matches) != 1:
            raise KeyError(f"{symbol}: {len(matches)} matching genes.")
        found.append(matches[0])
    return found


def gene_view(adata, genes, layer=None):
    """Dense ``float32`` expression matrix of ``genes``, shape ``(n_cells, len(genes))``."""
    matrix = adata[:, genes].X if layer is None else adata[:, genes].layers[layer]
    matrix = matrix.toarray() if sp.issparse(matrix) else np.asarray(matrix)
    return matrix.astype(np.float32)


def stratified_group_split(strata, groups, fractions, seed):
    """Assign every cell to ``train``/``val``/``test``, per stratum, keeping each group in one split.

    ``fractions`` maps split name to its share. Within each stratum (time point) the groups are
    shuffled and filled into the splits in order until each split has its share of cells, so the
    shares hold per time point up to the size of one group.
    """
    if not np.isclose(sum(fractions.values()), 1.0):
        raise ValueError("Split fractions must sum to one.")
    strata, groups = np.asarray(strata), np.asarray(groups)
    rng = np.random.default_rng(seed)
    labels = np.empty(len(strata), dtype=object)
    names = list(fractions)
    for stratum in np.unique(strata):
        in_stratum = np.flatnonzero(strata == stratum)
        group_ids, group_sizes = np.unique(groups[in_stratum], return_counts=True)
        order = rng.permutation(len(group_ids))
        bounds = np.cumsum([fractions[name] for name in names]) * len(in_stratum)
        filled = np.cumsum(group_sizes[order]) - group_sizes[order] / 2  # group midpoints
        split_of_group = dict(zip(group_ids[order], np.array(names)[np.searchsorted(bounds, filled)], strict=True))
        labels[in_stratum] = [split_of_group[g] for g in groups[in_stratum]]
    return labels


def split_table(split, time):
    """Cells per split and time point, with row and column totals."""
    return pd.crosstab(pd.Series(split, name="split"), pd.Series(time, name="time"), margins=True).reindex(
        [*SPLITS, "All"]
    )


def unique_points(points, decimals=5):
    """Distinct rows of ``points`` (after rounding), their multiplicities and the inverse index."""
    rounded = np.round(np.asarray(points, dtype=np.float64), decimals)
    unique, inverse, counts = np.unique(rounded, axis=0, return_inverse=True, return_counts=True)
    return unique.astype(np.float32), counts, inverse.ravel()


# Domain ------------------------------------------------------------------------------------------
@dataclass
class CellCloud:
    """The data domain without an analytic manifold: the graph points and their kernel support.

    A point is *supported* when the fitted kernel gives it at least one non-zero affinity, i.e. it
    lies within the CkNN support of some construction point. Unsupported points have no Nyström
    embedding gradient, so RFM cannot move them. Log-expression is non-negative, so
    :meth:`project` clips to the non-negative orthant.
    """

    construction_points: np.ndarray
    kernel: object
    name: str = "cell support"

    def __post_init__(self):
        self.construction_points = np.asarray(self.construction_points, dtype=np.float64)
        self.tree = cKDTree(self.construction_points)
        self.local_scales = np.asarray(self.kernel.local_scales, dtype=np.float64)
        self.delta = float(self.kernel.delta)

    @staticmethod
    def project(points):
        return jnp.maximum(points, 0.0)

    def nearest_distance(self, points):
        points = np.asarray(points, dtype=np.float64).reshape(-1, self.construction_points.shape[1])
        return self.tree.query(points)[0]

    def support_ratio(self, points, k=32):
        """``min_j ||x - x_j|| / (δ ρ_j)`` over the ``k`` nearest construction points: < 1 means supported.

        This is the training-point side of the CkNN support; the query bandwidth only matters near
        the support boundary.
        """
        shape = np.shape(points)[:-1]
        points = np.asarray(points, dtype=np.float64).reshape(-1, self.construction_points.shape[1])
        distances, ids = self.tree.query(points, k=k)
        return (distances / (self.delta * self.local_scales[ids])).min(axis=1).reshape(shape)

    def contains(self, points, tol=0.0):
        return self.support_ratio(points) < 1.0 + tol

    def negative_fraction(self, points):
        return float(np.mean(np.any(np.asarray(points) < -1e-6, axis=-1)))

    def diverged_fraction(self, points, factor=2.0):
        """Fraction of points with a coordinate beyond ``factor`` times the largest construction coordinate.

        Such points have left the data entirely; a few of them dominate mean-based metrics.
        """
        limit = factor * float(np.abs(self.construction_points).max())
        return float(np.mean(np.any(np.abs(np.asarray(points)) > limit, axis=-1)))


# Models ------------------------------------------------------------------------------------------
def predict_cells(model, starts):
    """Push any start cells through ``solver.predict`` with the single target condition.

    Uses the same inference state, condition and Tsit5 settings as ``CellFlow.predict``, but
    accepts cells that are not the control cells of the training AnnData (e.g. held-out cells).
    """
    from riemannian_toys.flows import inference_inputs

    _, condition, _ = inference_inputs(model)
    predicted = np.asarray(model.solver.predict(np.asarray(starts, dtype=np.float32), condition))
    assert predicted.shape == np.shape(starts) and np.all(np.isfinite(predicted))
    return predicted


def exact_rfm_paths(models, cloud, starts, target, final_time, steps, max_velocity_norm=None):
    """Euler paths of the exact RFM field ``u_t(x | x_1)`` towards one target, per kernel profile.

    Every step is ``x ← max(x + h u_t(x), 0)``. Returns ``(paths, table)``; paths have shape
    ``(steps + 1, n_starts, d)``.
    """
    from cellflow.ode_solver import RFMInterpolation
    from riemannian_toys.kernels import integrate_euler_paths, spectral_distance_trace

    paths, rows = {}, []
    for name, model in models.items():
        interpolator = RFMInterpolation(model, projection_fn=cloud.project, max_velocity_norm=max_velocity_norm)
        path, proposed = integrate_euler_paths(interpolator, starts, target, final_time, steps)
        if not np.all(np.isfinite(path)):
            raise ValueError(f"Non-finite exact RFM paths for {name}.")
        errors = np.linalg.norm(path[-1] - target, axis=1)
        distances = spectral_distance_trace(model, path, target)
        paths[name] = path
        rows.append({
            "kernel": name,
            "mean_final_error": errors.mean(),
            "max_final_error": errors.max(),
            "mean_relative_error": np.mean(errors / np.maximum(np.linalg.norm(path[0] - target, axis=1), 1e-12)),
            "mean_final_relative_biharmonic": np.mean(distances[-1] / np.maximum(distances[0], 1e-12)),
            "mean_path_length": np.linalg.norm(np.diff(path, axis=0), axis=-1).sum(axis=0).mean(),
            "mean_straight_distance": np.linalg.norm(path[0] - target, axis=1).mean(),
            "max_support_ratio": cloud.support_ratio(np.swapaxes(path, 0, 1)).max(),
            "clipped_step_fraction": float(np.mean(np.any(proposed < 0, axis=-1))),
        })
    return paths, pd.DataFrame(rows).set_index("kernel")


# Metrics -----------------------------------------------------------------------------------------
def energy_distance_squared(samples, reference):
    samples, reference = np.asarray(samples, dtype=np.float64), np.asarray(reference, dtype=np.float64)
    energy = 2 * cdist(samples, reference).mean() - cdist(samples, samples).mean() - cdist(reference, reference).mean()
    return float(max(energy, 0.0))


def wasserstein_2(samples, reference, max_points=1500, seed=0):
    """Empirical W2 by exact assignment between equal-size uniform subsamples."""
    rng = np.random.default_rng(seed)
    n = min(len(samples), len(reference), max_points)
    a = np.asarray(samples)[rng.choice(len(samples), n, replace=False)]
    b = np.asarray(reference)[rng.choice(len(reference), n, replace=False)]
    cost = cdist(a, b, "sqeuclidean")
    rows, cols = linear_sum_assignment(cost)
    return float(np.sqrt(cost[rows, cols].mean()))


def distribution_metrics(samples, reference, cloud):
    """Distribution distances to ``reference`` (no pairing assumed) and support diagnostics."""
    samples = np.asarray(samples)
    return {
        "energy_distance_squared": energy_distance_squared(samples, reference),
        "wasserstein_2": wasserstein_2(samples, reference),
        "mean_error_to_target": float(np.linalg.norm(samples.mean(axis=0) - np.asarray(reference).mean(axis=0))),
        "unsupported_fraction": float(np.mean(~cloud.contains(samples))),
        "negative_expression_fraction": cloud.negative_fraction(samples),
        "median_distance_to_graph": float(np.median(cloud.nearest_distance(samples))),
        "diverged_fraction": cloud.diverged_fraction(samples),
    }


def metric_table(clouds, reference, cloud):
    return pd.DataFrame({name: distribution_metrics(samples, reference, cloud) for name, samples in clouds.items()}).T


def path_metrics(cloud, trajectories):
    """Per-model path diagnostics: length, support and non-negativity along the saved states."""
    trajectories = np.asarray(trajectories)
    supported = cloud.contains(trajectories)
    return {
        "mean_path_length": float(np.linalg.norm(np.diff(trajectories, axis=1), axis=-1).sum(axis=1).mean()),
        "paths_leaving_support": float(np.mean(~supported.all(axis=1))),
        "fraction_saved_points_supported": float(supported.mean()),
        "max_support_ratio": float(cloud.support_ratio(trajectories).max()),
        "paths_with_negative_expression": float(np.mean(np.any(trajectories < -1e-6, axis=(1, 2)))),
        "median_distance_to_graph": float(np.median(cloud.nearest_distance(trajectories))),
    }


# Plotting ----------------------------------------------------------------------------------------
TIME_CMAP = "viridis"
PAIRS = ((0, 1), (0, 2), (1, 2))


def _short(gene):
    return gene.split(" ")[0]


def plot_gene_view(points, time, genes, title, background=None, max_points=6000, seed=0, s=3):
    """A 3-D scatter and the three pairwise projections, coloured by time."""
    rng = np.random.default_rng(seed)
    ids = rng.permutation(len(points))[:max_points]
    points, time = np.asarray(points)[ids], np.asarray(time)[ids]
    fig = plt.figure(figsize=(17, 4.2))
    kw = {"c": time, "cmap": TIME_CMAP, "s": s, "alpha": 0.5, "vmin": 0.0, "vmax": 1.0, "rasterized": True}
    ax = fig.add_subplot(1, 4, 1, projection="3d")
    ax.scatter(*points.T, **kw)
    ax.set(xlabel=_short(genes[0]), ylabel=_short(genes[1]), zlabel=_short(genes[2]))
    for panel, (a, b) in enumerate(PAIRS, start=2):
        ax = fig.add_subplot(1, 4, panel)
        if background is not None:
            ax.scatter(background[:, a], background[:, b], s=2, c="lightgrey", alpha=0.3, rasterized=True)
        image = ax.scatter(points[:, a], points[:, b], **kw)
        ax.set(xlabel=_short(genes[a]), ylabel=_short(genes[b]))
    fig.colorbar(image, ax=ax, label="experimental time")
    fig.suptitle(title)
    fig.tight_layout()
    plt.show()


def plot_time_trends(points, time, genes, title="Expression against time"):
    levels = np.unique(time)
    jitter = np.random.default_rng(0).uniform(-0.05, 0.05, len(time))
    fig, axes = plt.subplots(1, len(genes), figsize=(4.2 * len(genes), 3.4))
    for j, ax in enumerate(np.atleast_1d(axes)):
        ax.scatter(time + jitter, points[:, j], c=time, cmap=TIME_CMAP, s=2, alpha=0.25, rasterized=True)
        ax.plot(levels, [points[time == t, j].mean() for t in levels], "k-o", ms=4)
        ax.set(title=_short(genes[j]), xlabel="experimental time", ylabel="log expression")
    fig.suptitle(title)
    fig.tight_layout()
    plt.show()


def plot_spectral_coordinates(psi, time, n_coordinates=3, title="Biharmonic coordinates of the graph points"):
    """Leading weighted spectral coordinates against each other, coloured by time."""
    fig, axes = plt.subplots(1, n_coordinates, figsize=(4.6 * n_coordinates, 4))
    pairs = [(i, j) for i in range(n_coordinates) for j in range(i + 1, n_coordinates)][:n_coordinates]
    for ax, (i, j) in zip(np.atleast_1d(axes), pairs, strict=False):
        image = ax.scatter(psi[:, i], psi[:, j], c=time, cmap=TIME_CMAP, s=3, alpha=0.6, vmin=0, vmax=1,
                           rasterized=True)
        ax.set(xlabel=rf"$\Psi_{{{i + 1}}}$", ylabel=rf"$\Psi_{{{j + 1}}}$")
    fig.colorbar(image, ax=ax, label="mean experimental time of the point")
    fig.suptitle(title)
    fig.tight_layout()
    plt.show()


def plot_clouds(background, clouds, genes, title, max_points=1500, seed=0):
    """Pairwise projections of several point clouds over a grey background cloud.

    ``clouds`` maps a label to ``(points, scatter_kwargs)``.
    """
    rng = np.random.default_rng(seed)
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.6))
    for ax, (a, b) in zip(axes, PAIRS, strict=True):
        ax.scatter(background[:, a], background[:, b], s=2, c="lightgrey", alpha=0.35, rasterized=True)
        for label, (points, kwargs) in clouds.items():
            points = np.asarray(points)
            ids = rng.permutation(len(points))[:max_points]
            ax.scatter(points[ids, a], points[ids, b], label=label, **{"s": 4, "alpha": 0.5, **kwargs})
        ax.set(xlabel=_short(genes[a]), ylabel=_short(genes[b]))
    axes[0].legend(fontsize=8, markerscale=3)
    fig.suptitle(title)
    fig.tight_layout()
    plt.show()


def plot_paths(background, background_time, trajectories, times, genes, title, colors=None, axes=None):
    """Pairwise projections of learned paths (start: circle, end: cross) over the graph points."""
    trajectories = {"paths": trajectories} if not isinstance(trajectories, dict) else trajectories
    colors = colors or dict.fromkeys(trajectories, "tab:red")
    fig, axes = plt.subplots(len(trajectories), 3, figsize=(16, 4.4 * len(trajectories)), squeeze=False)
    background = np.asarray(background)
    low, high = background.min(axis=0) - 1.5, background.max(axis=0) + 1.5  # diverging paths leave the frame
    for row, (name, paths) in zip(axes, trajectories.items(), strict=True):
        n_off = int(np.sum(np.any((np.asarray(paths) < low) | (np.asarray(paths) > high), axis=(1, 2))))
        for ax, (a, b) in zip(row, PAIRS, strict=True):
            ax.set(xlim=(low[a], high[a]), ylim=(low[b], high[b]))
            ax.scatter(background[:, a], background[:, b], c=background_time, cmap=TIME_CMAP, s=2, alpha=0.25,
                       vmin=0, vmax=1, rasterized=True)
            for path in np.asarray(paths):
                ax.plot(path[:, a], path[:, b], color=colors[name], lw=1.1, alpha=0.8)
                ax.scatter(*path[0, [a, b]], marker="o", s=22, color=colors[name], edgecolors="black", zorder=3)
                ax.scatter(*path[-1, [a, b]], marker="x", s=30, color="black", zorder=3)
            ax.set(xlabel=_short(genes[a]), ylabel=_short(genes[b]))
        row[0].set_title(name + (f" ({n_off} path(s) leave the frame)" if n_off else ""), loc="left")
    fig.suptitle(title)
    fig.tight_layout()
    plt.show()


def plot_metric_bars(table, metrics, colors, title=None):
    """One bar panel per metric; rows of ``table`` are models (or other clouds)."""
    fig, axes = plt.subplots(1, len(metrics), figsize=(4.6 * len(metrics), 3.6))
    for ax, (column, label) in zip(np.atleast_1d(axes), metrics, strict=True):
        values = table[column]
        ax.bar(range(len(values)), values, color=[colors.get(name, "tab:gray") for name in values.index])
        ax.set_xticks(range(len(values)), values.index, rotation=35, ha="right", fontsize=8)
        ax.set_title(label, fontsize=10)
        ax.grid(axis="y", alpha=0.2)
    if title:
        fig.suptitle(title)
    fig.tight_layout()
    plt.show()
