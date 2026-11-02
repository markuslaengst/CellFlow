"""RFM loss modes: final interpolation point vs. every interpolation point.

Single-run helpers (train, field error on held-out RFM states, endpoints and paths, figures) and
the multi-seed experiment with per-iteration timing.
"""

import json
import time

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from IPython.display import display

from cellflow.data._dataloader import TrainSampler
from cellflow.ode_solver import RFMInterpolation
from riemannian_toys.flows import (
    endpoint_metrics,
    evaluate_learned_field,
    inference_inputs,
    integrate_learned_paths,
    predict_target,
    prepare_toy_model,
)
from riemannian_toys.plotting import PATH_PALETTE, rolling_mean

LOSS_MODE_COLORS = {"final": "tab:green", "trajectory": "tab:red"}

MULTISEED_CONFIGS = [
    # (setting, variant, interpolation_steps, loss_mode, trajectory_end_time)
    ("32 steps", "final", 32, "final", None),
    ("32 steps", "trajectory, random offset", 32, "trajectory", None),
    ("100 steps", "final", 100, "final", None),
    ("100 steps", "trajectory, fixed grid to 0.99", 100, "trajectory", 0.99),
    ("100 steps", "trajectory, random offset", 100, "trajectory", None),
]
MULTISEED_COLORS = {
    "final": "tab:green",
    "trajectory, fixed grid to 0.99": "tab:red",
    "trajectory, random offset": "tab:purple",
}
MULTISEED_METRICS = [
    "energy_distance_squared", "outside_annulus_fraction", "fresh_paths_leaving_annulus",
    "max_hole_penetration", "field_mse", "field_mse_t<0.9",
]


# Single run --------------------------------------------------------------------------------------
def interpolator_like(model, nystroem):
    """An :class:`RFMInterpolation` with ``model``'s training interpolation settings."""
    solver = model.solver
    return RFMInterpolation(
        nystroem,
        projection_fn=solver.projection_fn,
        project_to_manifold=solver.project_to_manifold,
        min_gradient_norm_sq=solver.min_gradient_norm_sq,
        time_epsilon=solver.time_epsilon,
        max_velocity_norm=solver.max_velocity_norm,
    )


def sample_rfm_eval_states(interpolator, coupling, source, target, n_pairs, steps, rng, offsets=None):
    """Exact RFM states for pairs drawn from ``coupling``, on random-offset time grids.

    Returns flattened ``(t, x, u)`` with shapes ``(n_pairs * steps, 1)``, ``(…, d)``, ``(…, d)``.
    """
    flat = coupling.ravel() / coupling.sum()
    source_ids, target_ids = np.unravel_index(rng.choice(flat.size, n_pairs, p=flat), coupling.shape)
    if offsets is None:
        offsets = rng.uniform(size=(n_pairs, 1)).astype(np.float32)
    ts, xs, us = jax.jit(interpolator.trajectory, static_argnums=2)(
        jnp.asarray(source[source_ids]), jnp.asarray(target[target_ids]), steps, jnp.asarray(offsets)
    )
    return tuple(np.asarray(a).reshape(-1, a.shape[-1]) for a in (ts, xs, us))


def field_error(model, eval_states, time_bins=None):
    """Squared, relative and cosine error of the learned field against exact RFM targets.

    Returns ``(metrics, binned_squared_error)``; ``field_mse`` is per coordinate.
    """
    ts, xs, us = eval_states
    vs = evaluate_learned_field(model, xs, ts)
    sq_error = np.sum((vs - us) ** 2, axis=1)
    relative = np.linalg.norm(vs - us, axis=1) / np.maximum(np.linalg.norm(us, axis=1), 1e-8)
    cosine = np.sum(vs * us, axis=1) / np.maximum(np.linalg.norm(vs, axis=1) * np.linalg.norm(us, axis=1), 1e-8)
    binned = None
    if time_bins is not None:
        bin_ids = np.clip(np.digitize(ts[:, 0], time_bins) - 1, 0, len(time_bins) - 2)
        binned = np.array([sq_error[bin_ids == b].mean() for b in range(len(time_bins) - 1)])
    metrics = {
        "field_mse": sq_error.mean() / xs.shape[1],
        "field_mse_t<0.9": sq_error[ts[:, 0] < 0.9].mean() / xs.shape[1],
        "median_relative_error": np.median(relative),
        "mean_cosine_similarity": cosine.mean(),
    }
    return metrics, binned


def endpoint_and_path_metrics(model, adata, target_key, condition_name, target, geometry, fresh_starts, save_times):
    """Prediction of the fitted source, learned paths of the fresh starts, and their metrics.

    Returns ``(predicted, paths, metrics)``.
    """
    predicted = predict_target(model, adata, target_key, condition_name)
    paths, _ = integrate_learned_paths(model, fresh_starts, save_times)
    path_metrics = geometry.path_metrics(paths)
    metrics = {
        **endpoint_metrics(predicted, target, geometry),
        **{key: path_metrics[key] for key in (
            f"paths_leaving_{geometry.name}", f"fraction_saved_points_in_{geometry.name}",
            "max_segment_hole_penetration")},
    }
    return predicted, paths, metrics


def plot_single_run(losses, binned_errors, time_bins, couplings, predicted, trajectories, summary, geometry, source,
                    target):
    """Training losses and binned field error; then predictions and fresh-sample paths per model."""
    fig, axes = plt.subplots(1, len(couplings) + 1, figsize=(20, 4.8), constrained_layout=True)
    for ax, coupling_name in zip(axes, couplings):
        for mode, color in LOSS_MODE_COLORS.items():
            values = losses[mode, coupling_name]
            ax.plot(values, alpha=0.2, color=color)
            ax.plot(rolling_mean(values), color=color, linewidth=2, label=f"{mode} (25-step mean)")
        ax.set(xlabel="Iteration", ylabel="RFM loss (same expectation for both modes)",
               title=f"Training loss, {coupling_name}", yscale="log")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.2)
    ax = axes[-1]
    centers = 0.5 * (time_bins[:-1] + time_bins[1:])
    for (mode, coupling_name), errors in binned_errors.items():
        ax.plot(centers, errors, marker="o", color=LOSS_MODE_COLORS[mode],
                linestyle="-" if coupling_name == list(couplings)[0] else "--", label=f"{mode}, {coupling_name}")
    ax.set(xlabel="Time t", ylabel="Mean ||v - u||^2 on held-out RFM states", title="Field error vs. time",
           yscale="log")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.2)
    plt.show()

    extra = np.concatenate([np.vstack(list(predicted.values()))]
                           + [paths.reshape(-1, geometry.dim) for paths in trajectories.values()])
    fig, axes = plt.subplots(2, len(summary), figsize=(6 * len(summary), 12), constrained_layout=True,
                             subplot_kw=geometry.subplot_kw)
    outside = f"outside_{geometry.name}_fraction"
    leaving = f"paths_leaving_{geometry.name}"
    for column, key in enumerate(summary.index):
        mode, coupling_name = key
        metrics = summary.loc[key]
        ax = axes[0, column]
        geometry.scatter(ax, target, s=10, alpha=0.3, color="tab:orange", label="Target")
        geometry.scatter(ax, predicted[key], s=10, alpha=0.6, color=LOSS_MODE_COLORS[mode], label="Prediction")
        ax.set_title(f"{mode} loss, {coupling_name}\nenergy² {metrics['energy_distance_squared']:.3f}, "
                     f"outside {metrics[outside]:.0%}")
        ax.legend(loc="upper right", fontsize=8)
        ax = axes[1, column]
        geometry.scatter(ax, source, s=6, alpha=0.12, color="tab:blue")
        geometry.scatter(ax, target, s=6, alpha=0.12, color="tab:orange")
        for index, path in enumerate(trajectories[key]):
            color = PATH_PALETTE(index % 10)
            geometry.plot_path(ax, path, color=color, linewidth=2)
            geometry.scatter(ax, path[0][None], s=50, color=color, edgecolor="white", zorder=5)
            geometry.scatter(ax, path[-1][None], marker="X", s=70, color=color, edgecolor="white", zorder=6)
        ax.set_title(f"Fresh-sample paths: {int(metrics[leaving])}/{len(trajectories[key])} leave the {geometry.name}")
        for ax in axes[:, column]:
            geometry.finish_axes(ax, extra_points=extra)
    fig.suptitle("RFM final-point loss vs. trajectory loss; same data, architecture, seed and iterations")
    plt.show()


# Multi-seed experiment ---------------------------------------------------------------------------
def train_seeded(model, seed, n_iterations, batch_size):
    """``CellFlowTrainer.train``'s loop without validation, with seeded batches and time samples.

    Returns ``(losses, iteration_seconds)``; each step is timed until its loss is available.
    """
    sampler = TrainSampler(model.train_data, batch_size=batch_size)
    rng_np, rng_jax = np.random.default_rng(seed), jax.random.PRNGKey(seed)
    losses, iteration_seconds = [], []
    for _ in range(n_iterations):
        rng_jax, key = jax.random.split(rng_jax)
        batch = sampler.sample(rng_np)
        started = time.perf_counter()
        losses.append(float(model.solver.step_fn(key, batch)))  # float() waits for the result
        iteration_seconds.append(time.perf_counter() - started)
    model.solver.is_trained = True
    return np.array(losses), np.array(iteration_seconds)


def multiseed_metrics(model, eval_states, source, target, geometry, fresh_starts, save_times):
    """Endpoint, fresh-path and field-error metrics of one multi-seed run."""
    ts, _, _ = eval_states
    sq = np.sum((evaluate_learned_field(model, eval_states[1], ts) - eval_states[2]) ** 2, axis=1)
    _, condition, _ = inference_inputs(model)
    predicted = np.asarray(model.solver.predict(source, condition))
    paths, _ = integrate_learned_paths(model, fresh_starts, save_times, check_prediction=False)
    radii = geometry.radial_coordinate(paths)
    inside = geometry.contains(paths)
    distribution = endpoint_metrics(predicted, target, geometry)
    return {
        "energy_distance_squared": distribution["energy_distance_squared"],
        f"outside_{geometry.name}_fraction": distribution[f"outside_{geometry.name}_fraction"],
        f"fresh_paths_leaving_{geometry.name}": float(np.mean(~inside.all(axis=1))),
        f"fresh_saved_points_in_{geometry.name}": float(inside.mean()),
        "max_hole_penetration": float(np.maximum(geometry.inner_radius - radii, 0).max()),
        "field_mse": float(sq.mean() / 2),
        "field_mse_t<0.9": float(sq[ts[:, 0] < 0.9].mean() / 2),
        "field_mse_by_t_decile": [
            float(sq[(ts[:, 0] >= b / 10) & (ts[:, 0] < (b + 1) / 10)].mean() / 2) for b in range(10)
        ],
    }


def run_multiseed(*, configs, seeds, adata, target_key, architecture_kwargs, optimizer, match_fn, nystroem, geometry,
                  source, target, coupling, fresh_starts, n_iterations=300, batch_size=32, max_velocity_norm=30.0,
                  n_eval_pairs=512, eval_seed=999):
    """Train one RFM model per configuration and seed; return one result row per run.

    Each row holds the metrics of :func:`multiseed_metrics`, timing (first iteration including
    compilation, median iteration, total), the last-25 mean loss, and the raw loss and timing traces.
    """
    eval_rng = np.random.default_rng(eval_seed)
    flat = coupling.ravel() / coupling.sum()
    pair_ids = eval_rng.choice(flat.size, n_eval_pairs, p=flat)
    offsets = eval_rng.uniform(size=(n_eval_pairs, 1)).astype(np.float32)
    eval_interpolator = RFMInterpolation(nystroem, projection_fn=geometry.project, max_velocity_norm=max_velocity_norm)
    save_times = jnp.linspace(0.0, 1.0, 201)
    eval_states_by_steps, rows = {}, []
    for setting, variant, steps, loss_mode, trajectory_end_time in configs:
        if steps not in eval_states_by_steps:
            source_ids, target_ids = np.unravel_index(pair_ids, coupling.shape)
            ts, xs, us = jax.jit(eval_interpolator.trajectory, static_argnums=2)(
                jnp.asarray(source[source_ids]), jnp.asarray(target[target_ids]), steps, jnp.asarray(offsets))
            eval_states_by_steps[steps] = tuple(np.asarray(a).reshape(-1, a.shape[-1]) for a in (ts, xs, us))
        for seed in seeds:
            started = time.perf_counter()
            model = prepare_toy_model(
                adata.copy(), "otrfm", target_key, **architecture_kwargs, match_fn=match_fn, nystroem=nystroem,
                optimizer=optimizer, seed=seed,
                solver_kwargs={
                    "interpolation_steps": steps, "projection_fn": geometry.project, "project_to_manifold": False,
                    "max_velocity_norm": max_velocity_norm, "loss_mode": loss_mode,
                    "trajectory_end_time": trajectory_end_time,
                },
            )
            losses, iteration_seconds = train_seeded(model, seed, n_iterations, batch_size)
            rows.append({
                "setting": setting, "variant": variant, "seed": seed, "loss_mode": loss_mode,
                **multiseed_metrics(model, eval_states_by_steps[steps], source, target, geometry, fresh_starts,
                                    save_times),
                "first_iter_seconds_incl_compile": float(iteration_seconds[0]),
                "median_iter_seconds": float(np.median(iteration_seconds[1:])),
                "train_seconds_total": float(iteration_seconds.sum()),
                "build_train_eval_seconds": time.perf_counter() - started,
                "loss_last_25": float(losses[-25:].mean()),
                "losses": losses.tolist(), "iter_seconds": iteration_seconds.tolist(),
            })
            print(f"{setting}, {variant}, seed {seed}: done", flush=True)
    return rows


def load_or_run_multiseed(results_path, run, configs, **run_kwargs):
    """Run :func:`run_multiseed` and save the rows if ``run``; otherwise load the saved rows."""
    if run:
        rows = run_multiseed(configs=configs, **run_kwargs)
        with open(results_path, "w", encoding="utf-8") as handle:
            json.dump({"configs": configs, "rows": rows}, handle, indent=1)
    else:
        with open(results_path, encoding="utf-8") as handle:
            rows = json.load(handle)["rows"]
    return rows


def mean_sd_table(frame, columns, order, digits=3):
    grouped = frame.groupby(["setting", "variant"])[columns]
    mean, sd = grouped.mean(), grouped.std()
    table = mean.round(digits).astype(str) + " ± " + sd.round(digits).astype(str)
    return table.loc[[key for key in order if key in table.index]]


def show_multiseed_tables(rows, configs, seeds, metrics=MULTISEED_METRICS):
    """Display mean ± sd per configuration, per-seed wins against the final-point loss, and timing."""
    frame = pd.DataFrame(rows)
    order = [(setting, variant) for setting, variant, *_ in configs]
    print("Results, mean ± sd over seeds (lower is better for every column)")
    display(mean_sd_table(frame, metrics, order))

    final = frame[frame["variant"].eq("final")].set_index(["setting", "seed"])[metrics]
    trajectory = frame[frame["loss_mode"].eq("trajectory")].set_index(["setting", "variant", "seed"])[metrics]
    wins = trajectory.lt(final.loc[trajectory.index.droplevel("variant")].to_numpy())
    wins = wins.groupby(level=["setting", "variant"]).sum().astype(int)
    print(f"Seeds (of {len(seeds)}) on which the trajectory variant beats the final-point loss with the same step count")
    display(wins.loc[[key for key in order if key in wins.index]])

    print("Computation time at 100 steps, mean ± sd over seeds (seconds)")
    display(mean_sd_table(frame[frame["setting"].eq("100 steps")],
                          ["first_iter_seconds_incl_compile", "median_iter_seconds", "train_seconds_total"], order,
                          digits=2))
    print("The 32-step runs recorded only build + train + evaluation time per model:")
    display(mean_sd_table(frame[frame["setting"].eq("32 steps")], ["build_train_eval_seconds"], order, digits=1))
    return frame


def plot_multiseed(frame, configs, seeds, colors=MULTISEED_COLORS):
    """Per-seed energy distance and endpoints outside; 100-step field error by time and training progress."""
    settings = list(dict.fromkeys(config[0] for config in configs))
    variants_100 = [variant for setting, variant, *_ in configs if setting == "100 steps"]
    fig, axes = plt.subplots(1, 4, figsize=(24, 4.8), constrained_layout=True)
    for ax, column, title in [
        (axes[0], "energy_distance_squared", "Squared energy distance to target"),
        (axes[1], "outside_annulus_fraction", "Predicted endpoints outside annulus"),
    ]:
        labelled = set()
        for position, setting in enumerate(settings):
            variants = [variant for s, variant, *_ in configs if s == setting]
            for offset, variant in zip(np.linspace(-0.12, 0.12, len(variants)) * (len(variants) > 1), variants):
                values = frame.loc[frame["setting"].eq(setting) & frame["variant"].eq(variant), column]
                ax.scatter(np.full(len(values), position + offset), values, color=colors[variant],
                           label=None if variant in labelled else variant, zorder=3)
                ax.hlines(values.mean(), position + offset - 0.05, position + offset + 0.05,
                          color=colors[variant], linewidth=2.5)
                labelled.add(variant)
        ax.set(xticks=range(len(settings)), xticklabels=settings, title=title + " (dots: seeds)")
        ax.legend(fontsize=8)
        ax.grid(axis="y", alpha=0.2)
    ax = axes[2]
    for variant in variants_100:
        deciles = np.array(frame.loc[frame["setting"].eq("100 steps") & frame["variant"].eq(variant),
                                     "field_mse_by_t_decile"].tolist())
        ax.plot(np.arange(10) / 10 + 0.05, deciles.mean(axis=0), marker="o", color=colors[variant], label=variant)
    ax.set(xlabel="Time t", ylabel="Field MSE vs. exact RFM target", yscale="log",
           title="100 steps: field error by time (mean over seeds)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.2)
    ax = axes[3]
    for variant in variants_100:
        for row in frame[frame["setting"].eq("100 steps") & frame["variant"].eq(variant)].itertuples():
            ax.plot(np.cumsum(row.iter_seconds), np.arange(1, len(row.iter_seconds) + 1), color=colors[variant],
                    alpha=0.7, label=variant if row.seed == seeds[0] else None)
    ax.set(xlabel="Wall-clock seconds", ylabel="Iterations completed",
           title="100 steps: training progress over time (one line per seed)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.2)
    plt.show()
