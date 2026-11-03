"""Figures shared by the annulus, swiss-roll and torus notebooks.

Every function draws the domain through the geometry object (2D axes for the annulus and swiss roll,
3D axes for the torus) and calls ``plt.show()``.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import Normalize
from matplotlib.lines import Line2D

from riemannian_toys.flows import evaluate_learned_field

PATH_PALETTE = plt.get_cmap("tab10")


def domain_axes(geometry, nrows=1, ncols=1, figsize=(7, 6), **kwargs):
    """``plt.subplots`` with axes suited to the domain."""
    return plt.subplots(nrows, ncols, figsize=figsize, subplot_kw=geometry.subplot_kw, **kwargs)


def add_domain_axis(fig, geometry, *args):
    return fig.add_subplot(*args, **geometry.subplot_kw)


def rolling_mean(values, window=25):
    return pd.Series(values).rolling(window, min_periods=1).mean()


# Data ---------------------------------------------------------------------------------------------
def plot_point_clouds(geometry, clouds, title, figsize=(7, 6), legend_loc="upper right", label_boundaries=False,
                      legend_fontsize=None):
    """Scatter several clouds on the domain. ``clouds`` is a list of ``(points, scatter_kwargs)``."""
    fig, ax = domain_axes(geometry, figsize=figsize)
    for points, style in clouds:
        geometry.scatter(ax, points, **style)
    geometry.finish_axes(ax, label_boundaries=label_boundaries, grid_alpha=0.25)
    ax.set_title(title)
    if ax.get_legend_handles_labels()[0]:
        ax.legend(loc=legend_loc, fontsize=legend_fontsize)
    plt.show()


# Kernel comparison -------------------------------------------------------------------------------
def plot_kernel_fields(geometry, field_points, fields, target, time, arrow_length=0.18):
    """Direction-normalized raw RFM fields per kernel, colored by log speed on a shared scale."""
    log_speeds = {name: np.log1p(np.linalg.norm(v, axis=1)) for name, v in fields.items()}
    norm = Normalize(0.0, max(float(np.quantile(np.concatenate(list(log_speeds.values())), 0.98)), 1e-6))
    fig, axes = domain_axes(geometry, 2, 3, figsize=(16, 10) if geometry.dim == 2 else (18, 11),
                            constrained_layout=True)
    for ax, (name, velocities) in zip(axes.ravel(), fields.items()):
        speeds = np.linalg.norm(velocities, axis=1)
        directions = arrow_length * velocities / np.maximum(speeds[:, None], 1e-8)
        geometry.draw_arrows(ax, field_points, directions, log_speeds[name], norm, "plasma", width=0.005)
        geometry.finish_axes(ax, grid_alpha=0.15)
        geometry.scatter(ax, np.asarray(target)[None], s=100, marker="*", color="black", zorder=5)
        ax.set_title(f"{name}\nzero velocities: {np.mean(speeds < 1e-8):.0%}")
    for ax in axes.ravel()[len(fields):]:
        fig.delaxes(ax)
    fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap="plasma"), ax=list(axes.ravel())[:len(fields)],
                 label="log(1 + raw speed), shared scale", shrink=1.0 if geometry.dim == 2 else 0.65)
    fig.suptitle(f"Raw RFM fields at t={time:g}; arrows show direction")
    plt.show()


def plot_kernel_trajectories(geometry, trajectories, starts, target, trajectory_df, steps):
    """Raw (top) and guarded (bottom) Euler paths for every kernel."""
    kernel_names = list(dict.fromkeys(name for name, _ in trajectories))
    modes = list(dict.fromkeys(mode for _, mode in trajectories))
    fig, axes = domain_axes(geometry, len(modes), len(kernel_names), figsize=(25, 10), constrained_layout=True)
    for column, kernel_name in enumerate(kernel_names):
        for row, mode in enumerate(modes):
            ax = axes[row, column]
            path = trajectories[kernel_name, mode]
            for start_index in range(len(starts)):
                geometry.plot_path(ax, path[:, start_index], linewidth=1.1, alpha=0.7)
            geometry.scatter(ax, starts, s=16, color="tab:blue")
            geometry.scatter(ax, path[-1], s=18, color="tab:orange", marker="x")
            geometry.finish_axes(ax, grid_alpha=0.15)
            geometry.scatter(ax, np.asarray(target)[None], s=100, marker="*", color="black", zorder=5)
            result = trajectory_df[trajectory_df["kernel"].eq(kernel_name) & trajectory_df["mode"].eq(mode)].iloc[0]
            ax.set_title(f"{kernel_name}: {mode}\nmean endpoint error {result['mean_final_euclidean']:.3f}")
    fig.suptitle(f"Shared starts and target; blue starts, orange endpoints; {steps} Euler steps")
    plt.show()


def plot_kernel_convergence(trajectories, distance_traces, step_times, target, stride=8):
    """Mean Euclidean distance to the target and mean relative spectral distance over time."""
    kernel_names = list(dict.fromkeys(name for name, _ in trajectories))
    modes = list(dict.fromkeys(mode for _, mode in trajectories))
    fig, axes = plt.subplots(2, len(modes), figsize=(14, 10), constrained_layout=True)
    for column, mode in enumerate(modes):
        for kernel_name in kernel_names:
            path = trajectories[kernel_name, mode]
            axes[0, column].plot(step_times, np.linalg.norm(path - target, axis=-1).mean(axis=1), label=kernel_name)
            distances = distance_traces[kernel_name, mode]
            axes[1, column].plot(step_times[::stride], (distances / np.maximum(distances[0], 1e-12)).mean(axis=1),
                                 label=kernel_name)
        axes[1, column].plot(step_times, 1.0 - step_times, "k--", label="ideal 1 - t")
        axes[0, column].set(title=mode, xlabel="time", ylabel="mean Euclidean distance to target")
        axes[1, column].set(title=mode, xlabel="time", ylabel="mean relative spectral distance")
        for ax in axes[:, column]:
            ax.legend()
            ax.grid(alpha=0.2)
    fig.suptitle("Endpoint convergence and spectral decay for every kernel")
    plt.show()


def plot_refinement_errors(refinement_df, coarse_steps, refined_steps):
    fig, ax = plt.subplots(figsize=(9, 4.5), constrained_layout=True)
    refinement_df[["coarse_mean_endpoint_error", "refined_mean_endpoint_error"]].rename(columns={
        "coarse_mean_endpoint_error": f"{coarse_steps} steps",
        "refined_mean_endpoint_error": f"{refined_steps} steps",
    }).plot.bar(ax=ax, rot=15, logy=True)
    ax.set_ylabel("mean Euclidean endpoint error (log scale)")
    ax.set_title("Guarded trajectory resolution check; inspect segment validity in the table")
    ax.grid(axis="y", alpha=0.2)
    plt.show()


# Training ----------------------------------------------------------------------------------------
def plot_matrices(panels, figsize=(11, 4)):
    """Heatmaps of ``(matrix, title)`` pairs, e.g. a cost matrix and its OT coupling."""
    fig, axes = plt.subplots(1, len(panels), figsize=figsize)
    for ax, (values, title) in zip(np.atleast_1d(axes), panels):
        image = ax.imshow(values, aspect="auto", origin="lower")
        ax.set(xlabel="Target index", ylabel="Source index", title=title)
        fig.colorbar(image, ax=ax)
    fig.tight_layout()
    plt.show()


def plot_training_result(geometry, losses, source, target, predicted, target_label="Target"):
    """Training loss, then source vs. target and prediction vs. target."""
    fig = plt.figure(figsize=(16 if geometry.dim == 2 else 18, 4.5 if geometry.dim == 2 else 5))
    loss_ax = fig.add_subplot(131)
    loss_ax.plot(losses, alpha=0.35, label="Training loss")
    loss_ax.plot(rolling_mean(losses), label="25-step mean")
    loss_ax.set(xlabel="Iteration", ylabel="Flow matching loss", title="RFM training")
    loss_ax.legend()
    for position, (cloud, title, color, label) in enumerate([
        (source, "Source and target", "tab:blue", "Source"),
        (predicted, "RFM prediction and target", "tab:green", "Prediction"),
    ]):
        ax = add_domain_axis(fig, geometry, 132 + position)
        geometry.scatter(ax, target, s=14, alpha=0.4, color="tab:orange", label=target_label)
        geometry.scatter(ax, cloud, s=14, alpha=0.6, color=color, label=label)
        geometry.finish_axes(ax, extra_points=cloud)
        ax.set_title(title)
        ax.legend(fontsize=8)
    fig.tight_layout()
    plt.show()


# Learned fields and paths ------------------------------------------------------------------------
def plot_learned_field(geometry, model, point, time, source, target, grid_size=27, arrow_dt=0.04):
    """Learned instantaneous velocity on a domain grid; the selected point's arrow is highlighted.

    Arrow length is ``arrow_dt`` times velocity everywhere. Returns ``(fig, ax, v(time, point))``.
    """
    point = np.asarray(point, dtype=np.float32)
    if point.shape != (geometry.dim,) or not np.all(np.isfinite(point)):
        raise ValueError(f"point must contain {geometry.dim} finite coordinates.")
    if not np.isfinite(time) or not 0.0 <= time <= 1.0:
        raise ValueError("time must be between 0 and 1.")
    if grid_size < 3 or arrow_dt <= 0:
        raise ValueError("grid_size must be at least 3 and arrow_dt must be positive.")
    grid = geometry.learned_field_grid(grid_size)
    velocities = evaluate_learned_field(model, np.vstack((grid, point[None])), time)
    grid_velocities, point_velocity = velocities[:-1], velocities[-1]
    speeds = np.linalg.norm(grid_velocities, axis=1)
    norm = Normalize(0.0, max(float(speeds.max()), 1e-6))

    fig, ax = domain_axes(geometry, figsize=(8, 7) if geometry.dim == 2 else (10, 8))
    geometry.scatter(ax, source, s=6, alpha=0.15, color="tab:blue", label="Source cloud (t=0)")
    geometry.scatter(ax, target, s=6, alpha=0.15, color="tab:orange", label="Target cloud (t=1)")
    geometry.draw_arrows(ax, grid, arrow_dt * grid_velocities, speeds, norm, "viridis", alpha=0.85)
    fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap="viridis"), ax=ax, label="Learned speed ||v(t, x)||",
                 shrink=1.0 if geometry.dim == 2 else 0.65)
    geometry.scatter(ax, point[None], s=70, color="tab:red", edgecolor="white", zorder=5, label="Selected point")
    geometry.draw_highlight_arrow(ax, point, arrow_dt * point_velocity)
    geometry.finish_axes(ax, extra_points=np.vstack((point, point + arrow_dt * point_velocity)), outer_color="gray")
    ax.set_title(f"Learned RFM field at t={time:.2f}\n"
                 f"x={np.round(point, 2)}, v={np.round(point_velocity, 3)}")
    text = ax.text if geometry.dim == 2 else ax.text2D
    text(0.02, 0.02, f"Arrow length = {arrow_dt:g} × velocity", transform=ax.transAxes, fontsize=9)
    ax.legend(loc="upper right" if geometry.dim == 2 else "upper left", fontsize=9)
    fig.tight_layout()
    plt.show()
    return fig, ax, point_velocity


def _draw_paths(geometry, ax, trajectories, label_paths=False, linewidth=2, marker_size=(65, 85)):
    for index, path in enumerate(trajectories):
        color = PATH_PALETTE(index % 10)
        geometry.plot_path(ax, path, color=color, linewidth=linewidth, label=f"Path {index + 1}" if label_paths else None)
        geometry.scatter(ax, path[0][None], s=marker_size[0], color=color, edgecolor="white", zorder=5)
        geometry.scatter(ax, path[-1][None], marker="X", s=marker_size[1], color=color, edgecolor="white", zorder=6)
        geometry.draw_direction_arrows(ax, path, color)


def plot_learned_paths(geometry, trajectories, times, source, target, model_name="RFM"):
    """Learned paths from fresh source samples, and the radial coordinate along each path."""
    fig = plt.figure(figsize=(13, 6) if geometry.dim == 2 else (15, 6))
    path_ax = add_domain_axis(fig, geometry, 121)
    radius_ax = fig.add_subplot(122)
    geometry.scatter(path_ax, source, s=9, alpha=0.2, color="tab:blue", label="Source cloud")
    geometry.scatter(path_ax, target, s=9, alpha=0.2, color="tab:orange", label="Target cloud")
    geometry.draw_radial_reference(radius_ax)
    _draw_paths(geometry, path_ax, trajectories, label_paths=True)
    radial = geometry.radial_coordinate(trajectories)
    for index in range(len(trajectories)):
        radius_ax.plot(times, radial[index], color=PATH_PALETTE(index % 10), linewidth=2, label=f"Path {index + 1}")
    geometry.finish_axes(path_ax, extra_points=trajectories.reshape(-1, geometry.dim), outer_color="gray")
    path_ax.set_title(f"Learned {model_name} trajectories from fresh source samples")
    handles, _ = path_ax.get_legend_handles_labels()
    handles += [Line2D([], [], linestyle="none", marker="o", color="gray", label="Start (t=0)"),
                Line2D([], [], linestyle="none", marker="X", color="gray", label="End (t=1)")]
    path_ax.legend(handles=handles, loc="upper right" if geometry.dim == 2 else "upper left", fontsize=8)
    radius_ax.set(xlabel="Time t", ylabel=geometry.radial_label, xlim=(0, 1),
                  title=f"{geometry.radial_label} along each path")
    radius_ax.legend(fontsize=8, loc="best")
    radius_ax.grid(alpha=0.2)
    fig.tight_layout()
    plt.show()


# Model comparison --------------------------------------------------------------------------------
def plot_loss_panels(losses, colors, title):
    """One training-loss panel per model, with a 25-step rolling mean."""
    fig, axes = plt.subplots(1, len(losses), figsize=(5 * len(losses), 4), constrained_layout=True)
    for ax, (name, values) in zip(np.atleast_1d(axes), losses.items()):
        ax.plot(values, alpha=0.3, color=colors[name], label="Training loss")
        ax.plot(rolling_mean(values), color="black", linewidth=1.2, label="25-step mean")
        ax.set(xlabel="Iteration", ylabel="Flow matching loss", title=name)
        ax.legend(fontsize=8)
        ax.grid(alpha=0.2)
    fig.suptitle(title)
    plt.show()


def plot_prediction_panels(geometry, source, target, predictions, colors, metrics=None, title=None):
    """Source vs. target, then each model's prediction vs. target, on a common extent."""
    panels = [("Source and target", source, "tab:blue", "Source")]
    panels += [(name, samples, colors[name], "Prediction") for name, samples in predictions.items()]
    extra = np.vstack([samples for _, samples, _, _ in panels])
    fig, axes = domain_axes(geometry, 1, len(panels), figsize=(5 * len(panels), 5.5), constrained_layout=True)
    outside = f"outside_{geometry.name}_fraction"
    for ax, (name, samples, color, label) in zip(axes, panels):
        geometry.scatter(ax, target, s=12, alpha=0.35, color="tab:orange", label="Target")
        geometry.scatter(ax, samples, s=12, alpha=0.6, color=color, label=label)
        geometry.finish_axes(ax, extra_points=extra)
        if metrics is not None and name in metrics.index:
            name = (f"{name}\nenergy² {metrics.loc[name, 'energy_distance_squared']:.3f}, "
                    f"outside {metrics.loc[name, outside]:.0%}")
        ax.set_title(name)
        ax.legend(loc="upper right" if geometry.dim == 2 else "upper left", fontsize=8)
    fig.suptitle(title or "Same data, architecture, optimizer, batch size, seed, and iteration count")
    plt.show()


def plot_field_comparison(geometry, models, query_point, time, source, target, grid_size=27, arrow_dt=0.04):
    """Learned fields of several models at a common time, grid and speed scale.

    Returns the velocities per model; the last row of each array is the query point.
    """
    grid = geometry.learned_field_grid(grid_size)
    queries = np.vstack((grid, np.asarray(query_point, dtype=np.float32)[None]))
    fields = {name: evaluate_learned_field(model, queries, time) for name, model in models.items()}
    norm = Normalize(0.0, max(max(float(np.linalg.norm(v[:-1], axis=1).max()) for v in fields.values()), 1e-6))
    n_columns = 2
    n_rows = int(np.ceil(len(fields) / n_columns))
    figsize = (15, 6.5 * n_rows) if geometry.dim == 2 else (16, 7 * n_rows)
    fig, axes = domain_axes(geometry, n_rows, n_columns, figsize=figsize, constrained_layout=True, squeeze=False)
    for ax, (name, velocities) in zip(axes.ravel(), fields.items()):
        grid_velocities, point_velocity = velocities[:-1], velocities[-1]
        geometry.draw_arrows(ax, grid, arrow_dt * grid_velocities, np.linalg.norm(grid_velocities, axis=1), norm,
                             "viridis", alpha=0.85)
        geometry.scatter(ax, source, s=6, alpha=0.15, color="tab:blue", label="Source cloud (t=0)")
        geometry.scatter(ax, target, s=6, alpha=0.15, color="tab:orange", label="Target cloud (t=1)")
        geometry.scatter(ax, np.asarray(query_point)[None], s=70, color="tab:red", edgecolor="white", zorder=5,
                         label="Query point")
        geometry.draw_highlight_arrow(ax, query_point, arrow_dt * point_velocity)
        geometry.finish_axes(ax, extra_points=np.vstack((query_point, query_point + arrow_dt * point_velocity)),
                             outer_color="gray")
        ax.set_title(f"{name} at t={time:.2f}\nv(query) = {np.round(point_velocity, 3)}")
        ax.legend(loc="upper right" if geometry.dim == 2 else "upper left", fontsize=8)
    for ax in axes.ravel()[len(fields):]:
        fig.delaxes(ax)
    fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap="viridis"), ax=list(axes.ravel())[:len(fields)],
                 label="Learned speed ||v(t, x)||, shared scale", shrink=0.6)
    fig.suptitle(f"Arrow length = {arrow_dt:g} × velocity; common point and time")
    plt.show()
    return fields


def plot_path_comparison(geometry, trajectories, times, source, target, colors, path_metrics=None):
    """Each model's learned paths from the same starts, plus mean and range of the radial coordinate."""
    n_models = len(trajectories)
    n_rows = 2 if n_models > 2 else 1
    n_columns = int(np.ceil(n_models / n_rows))
    fig = plt.figure(figsize=(7.5 * (n_columns + 1), 6 * n_rows), constrained_layout=True)
    grid = fig.add_gridspec(n_rows, n_columns + 1)
    path_axes = [add_domain_axis(fig, geometry, grid[row, column])
                 for row in range(n_rows) for column in range(n_columns)][:n_models]
    radius_ax = fig.add_subplot(grid[:, n_columns])
    geometry.draw_radial_reference(radius_ax)
    extra = np.concatenate([paths.reshape(-1, geometry.dim) for paths in trajectories.values()])
    leaving = f"paths_leaving_{geometry.name}"
    for ax, (name, paths) in zip(path_axes, trajectories.items()):
        geometry.scatter(ax, source, s=8, alpha=0.15, color="tab:blue", label="Source cloud")
        geometry.scatter(ax, target, s=8, alpha=0.15, color="tab:orange", label="Target cloud")
        _draw_paths(geometry, ax, paths, marker_size=(60, 80))
        geometry.finish_axes(ax, extra_points=extra, outer_color="gray")
        title = name
        if path_metrics is not None and leaving in path_metrics.columns:
            title += f"\n{int(path_metrics.loc[name, leaving])}/{len(paths)} paths leave the {geometry.name}"
        ax.set_title(title)
        ax.legend(loc="upper right" if geometry.dim == 2 else "upper left", fontsize=8)
        radial = geometry.radial_coordinate(paths)
        radius_ax.plot(times, radial.mean(axis=0), color=colors[name], linewidth=2, label=f"{name} mean")
        radius_ax.fill_between(times, radial.min(axis=0), radial.max(axis=0), color=colors[name], alpha=0.12)
    radius_ax.set(xlabel="Time t", ylabel=geometry.radial_label, xlim=(0, 1),
                  title="Mean; shaded range across the same paths")
    radius_ax.legend(fontsize=8, loc="best")
    radius_ax.grid(alpha=0.2)
    fig.suptitle("Learned paths from the same fresh source samples; circles start, crosses end")
    plt.show()
