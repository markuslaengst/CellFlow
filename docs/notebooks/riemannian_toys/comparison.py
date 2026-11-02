"""Comparing several trained models: couplings, training, endpoints, paths and a factor summary."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from riemannian_toys.annulus import straight_segment_min_radius
from riemannian_toys.flows import check_coupling, endpoint_metrics, integrate_learned_paths, loss_summary, train_timed


def coupling_comparison_table(couplings, cost_matrices, source, target, hole_radius):
    """Marginals, transported costs, and how much mass pairs the same half or crosses the hole.

    ``same_half_mass`` is the coupled mass whose endpoints lie on the same side of the x-axis;
    ``chord_through_hole_mass`` the mass whose straight chord passes within ``hole_radius`` of
    the origin. An independent coupling is added as a reference.
    """
    n_source, n_target = len(source), len(target)
    pair_min_radius = straight_segment_min_radius(source[:, None, :], target[None, :, :])
    same_half = np.sign(source[:, None, 1]) == np.sign(target[None, :, 1])
    independent = np.full((n_source, n_target), 1.0 / (n_source * n_target))
    rows = []
    for name, coupling in [*couplings.items(), ("independent", independent)]:
        check_coupling(coupling, n_source, n_target)
        mass = coupling / coupling.sum()
        rows.append({
            "coupling": name,
            "coupling_mass": coupling.sum(),
            "max_source_marginal_error": np.max(np.abs(coupling.sum(axis=1) - 1 / n_source)),
            "max_target_marginal_error": np.max(np.abs(coupling.sum(axis=0) - 1 / n_target)),
            **{f"mean_{cost_name.lower()}_cost": np.sum(mass * cost) for cost_name, cost in cost_matrices.items()},
            "same_half_mass": np.sum(mass * same_half),
            "chord_through_hole_mass": np.sum(mass * (pair_min_radius < hole_radius)),
        })
    return pd.DataFrame(rows).set_index("coupling")


def plot_coupling_comparison(geometry, couplings, source, target, seed, n_pairs=40):
    """Couplings sorted by height (top) and the most likely partner of sampled sources (bottom)."""
    source_order, target_order = np.argsort(source[:, 1]), np.argsort(target[:, 1])
    pair_sources = np.random.default_rng(seed).choice(len(source), n_pairs, replace=False)
    fig, axes = plt.subplots(2, len(couplings), figsize=(6 * len(couplings), 11), constrained_layout=True)
    for column, (name, coupling) in enumerate(couplings.items()):
        image = axes[0, column].imshow(coupling[np.ix_(source_order, target_order)], aspect="auto", origin="lower")
        axes[0, column].set(xlabel="Target index (sorted by y)", ylabel="Source index (sorted by y)",
                            title=f"{name} entropic OT coupling")
        fig.colorbar(image, ax=axes[0, column])
        ax = axes[1, column]
        geometry.scatter(ax, source, s=8, alpha=0.25, color="tab:blue", label="Source")
        geometry.scatter(ax, target, s=8, alpha=0.25, color="tab:orange", label="Target")
        for source_index, target_index in zip(pair_sources, coupling[pair_sources].argmax(axis=1)):
            geometry.plot_path(ax, np.stack((source[source_index], target[target_index])),
                               color="gray", alpha=0.5, linewidth=0.8)
        geometry.finish_axes(ax)
        ax.set_title(f"{name}: most likely partner of {n_pairs} sources\n(lines show pairs, not paths)")
        ax.legend(loc="upper right", fontsize=8)
    plt.show()


def train_untrained(models, losses, training_seconds, num_iterations, batch_size):
    """Train every model without an entry in ``losses``; fill ``losses`` and ``training_seconds`` in place."""
    for name, model in models.items():
        if name in losses:
            continue
        print(f"Training {name}", flush=True)
        losses[name], training_seconds[name] = train_timed(model, num_iterations, batch_size)
        assert int(model.solver.vf_state.step) == num_iterations
        print(f"{name}: {training_seconds[name]:.1f}s; {loss_summary(losses[name])}", flush=True)
    return pd.DataFrame({
        "training_seconds": training_seconds,
        "mean_loss_first_25": {name: values[:25].mean() for name, values in losses.items()},
        "mean_loss_last_25": {name: values[-25:].mean() for name, values in losses.items()},
    }).loc[list(models)]


def endpoint_table(predictions, source, target, geometry):
    """:func:`~riemannian_toys.flows.endpoint_metrics` of the unchanged source and of every prediction."""
    return pd.DataFrame(
        [endpoint_metrics(source, target, geometry)]
        + [endpoint_metrics(samples, target, geometry) for samples in predictions.values()],
        index=["unchanged source", *predictions],
    )


def integrate_all_paths(models, starts, times):
    """Learned paths of every model from the same starts; returns ``(trajectories, solver_steps)``."""
    trajectories, solver_steps = {}, {}
    for name, model in models.items():
        trajectories[name], solver_steps[name] = integrate_learned_paths(model, starts, times)
    return trajectories, solver_steps


def path_table(geometry, trajectories, solver_steps):
    """:meth:`geometry.path_metrics` per model, plus the mean number of adaptive solver steps."""
    return pd.DataFrame([
        {"model": name, **geometry.path_metrics(paths), "mean_adaptive_solver_steps": float(np.mean(solver_steps[name]))}
        for name, paths in trajectories.items()
    ]).set_index("model")


def factor_summary(endpoint_df, path_df, training_seconds, model_names, endpoint_columns, path_columns):
    """One row per model, indexed by ``(interpolation, coupling)`` parsed from ``"RFM + biharmonic OT"``."""
    summary = endpoint_df.loc[list(model_names), endpoint_columns].join(path_df[path_columns])
    summary = summary.assign(training_seconds=pd.Series(training_seconds))
    summary.index = pd.MultiIndex.from_tuples([tuple(name.split(" + ")) for name in summary.index],
                                              names=["interpolation", "coupling"])
    return summary


def plot_factor_bars(summary, metrics, interpolations=("RFM", "FM"), couplings=("biharmonic OT", "Euclidean OT")):
    """2 × 2 bar charts: interpolation on the x-axis, coupling as bar color."""
    fig, axes = plt.subplots(1, len(metrics), figsize=(6 * len(metrics), 4.5), constrained_layout=True)
    for ax, (column, title) in zip(axes, metrics):
        summary[column].unstack("coupling").loc[list(interpolations), list(couplings)].plot.bar(
            ax=ax, rot=0, color=["tab:green", "tab:gray"])
        ax.set(title=title, xlabel="Interpolation", ylabel=column)
        ax.legend(title="Coupling", fontsize=8)
        ax.grid(axis="y", alpha=0.2)
    fig.suptitle("2 × 2 comparison: interpolation (x-axis) × coupling (bar color)")
    plt.show()
