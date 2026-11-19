"""RFM target velocity scale: ``d(x_t, x1) / (1 - t)`` (CellFlow default) vs. ``d(x0, x1)`` (reference).

Both point along ``-grad d(x_t, x1) / ||grad d||^2`` and agree on the exact path. They differ once the
path is discretized (Euler steps, projection, velocity clipping) and in how large the regression target
gets near ``t = 1``. This module compares them

1. on the conditional reference paths alone (no network): distance to the exact geodesic, endpoint error
   and target speed along the path, as a function of the number of Euler steps; and
2. after training CellFlow with each target, over several seeds and both loss modes.
"""

import json
import time

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from flax.traverse_util import flatten_dict, unflatten_dict

from cellflow.ode_solver import RFMInterpolation
from riemannian_toys import reference_paths
from riemannian_toys.flows import (
    endpoint_metrics,
    evaluate_learned_field,
    inference_inputs,
    integrate_learned_paths,
    prepare_toy_model,
)
from riemannian_toys.loss_modes import train_seeded
from riemannian_toys.plotting import rolling_mean

VARIANTS = {
    # name: (velocity_scale, clip velocities at the configured max_velocity_norm)
    "current": ("current", True),
    "initial": ("initial", True),
    "initial, no clip": ("initial", False),
}
"""``current`` is CellFlow's default; ``initial, no clip`` is the reference implementation's target."""
VARIANT_LABELS = {
    "current": "d(x_t,x1)/(1-t), clipped  [CellFlow]",
    "initial": "d(x0,x1), clipped",
    "initial, no clip": "d(x0,x1), unclipped  [reference]",
}
VARIANT_COLORS = {"current": "tab:green", "initial": "tab:blue", "initial, no clip": "tab:purple"}
LOSS_MODE_STYLES = {"final": "--", "trajectory": "-"}


def variant_settings(variant, max_velocity_norm):
    """``(velocity_scale, max_velocity_norm)`` of a variant in :data:`VARIANTS`."""
    scale, clip = VARIANTS[variant]
    return scale, max_velocity_norm if clip else None


def make_interpolator(nystroem, variant, projection_fn, max_velocity_norm):
    scale, clip_norm = variant_settings(variant, max_velocity_norm)
    return RFMInterpolation(nystroem, projection_fn=projection_fn, max_velocity_norm=clip_norm, velocity_scale=scale)


# 1. Conditional reference paths --------------------------------------------------------------------
def reference_path_study(geometry, nystroem, x0, x1, steps_list, projection_fn, max_velocity_norm,
                         variants=tuple(VARIANTS)):
    """Integrate every variant with every step count and compare each path with the exact geodesic.

    Returns ``(per_pair, paths, speeds)``: per-pair frames of
    :func:`reference_paths.compare_to_geodesics` keyed by ``(steps, variant)``, the paths
    ``(times, (n, steps + 1, d))`` keyed the same way, and the target speeds ``(steps, n)`` at the states
    before ``t = 1``, as used for the Euler step (``clipped``) and before clipping (``unclipped``).
    """
    per_pair, paths, speeds = {}, {}, {}
    for steps in steps_list:
        for variant in variants:
            scale, _ = VARIANTS[variant]
            clipped = make_interpolator(nystroem, variant, projection_fn, max_velocity_norm)
            unclipped = RFMInterpolation(nystroem, projection_fn=projection_fn, velocity_scale=scale)
            times, trajectories = reference_paths.rfm_reference_paths(clipped, x0, x1, steps)
            per_pair[steps, variant] = reference_paths.compare_to_geodesics(geometry, x0, x1, times, trajectories)
            paths[steps, variant] = (times, trajectories)
            # Target speed at the recorded states before t = 1 (never supervised), with and without clipping.
            states = jnp.asarray(trajectories[:, :-1].transpose(1, 0, 2), dtype=jnp.float32)  # (m, n, d)
            target = jnp.broadcast_to(jnp.asarray(x1, jnp.float32), states.shape)
            t = jnp.broadcast_to(jnp.asarray(times[:-1], jnp.float32)[:, None, None], states.shape[:2] + (1,))
            d0 = clipped.initial_distance(jnp.asarray(x0, jnp.float32), jnp.asarray(x1, jnp.float32))

            # One time step at a time: evaluating all (m, n) states at once needs memory ~ m * n * construction points.
            @jax.jit
            def step_speeds(x, y, tt, d0=d0):
                return tuple(jnp.linalg.norm(interp.u_t(x, y, tt, d0=d0), axis=-1) for interp in (clipped, unclipped))

            per_step = [step_speeds(states[k], target[k], t[k]) for k in range(states.shape[0])]
            speeds[steps, variant] = {
                "times": times[:-1],
                "clipped": np.stack([np.asarray(c) for c, _ in per_step]),
                "unclipped": np.stack([np.asarray(u) for _, u in per_step]),
            }
    return per_pair, paths, speeds


def reference_summary(per_pair, speeds, max_velocity_norm,
                      metrics=("hausdorff", "wasserstein_time", "endpoint_error", "length_ratio",
                               "arc_fraction_at_half_time")):
    """One row per ``(steps, variant)``: mean/max of the path metrics and the share of states whose
    unclipped target speed exceeds ``max_velocity_norm`` (clipped in the clipped variants)."""
    rows = {}
    for key, frame in per_pair.items():
        row = {}
        for metric in metrics:
            row[metric, "mean"] = frame[metric].mean()
            row[metric, "max"] = frame[metric].max()
        row["leaves_domain", "fraction"] = frame["leaves_domain"].mean()
        unclipped = speeds[key]["unclipped"]
        row["target_speed", "median"] = np.median(unclipped)
        row["target_speed", "max"] = unclipped.max()
        if max_velocity_norm is not None:
            row["clipped_states", "fraction"] = np.mean(unclipped > max_velocity_norm)
        rows[key] = row
    table = pd.DataFrame(rows).T
    table.columns = pd.MultiIndex.from_tuples(table.columns)
    table.index = pd.MultiIndex.from_tuples(table.index, names=["steps", "variant"])
    return table


def plot_reference_convergence(per_pair, steps_list, metrics=("endpoint_error", "hausdorff", "wasserstein_time",
                                                               "leaves_domain")):
    """Mean (and 10-90% band) of each path metric against the number of Euler steps."""
    variants = list(dict.fromkeys(variant for _, variant in per_pair))
    fig, axes = plt.subplots(1, len(metrics), figsize=(4.6 * len(metrics), 4.0), constrained_layout=True)
    for ax, metric in zip(axes, metrics, strict=True):
        for variant in variants:
            values = np.stack([per_pair[steps, variant][metric].to_numpy(dtype=float) for steps in steps_list])
            color = VARIANT_COLORS[variant]
            ax.plot(steps_list, values.mean(axis=1), marker="o", color=color, label=VARIANT_LABELS[variant])
            if metric != "leaves_domain":
                ax.fill_between(steps_list, *np.quantile(values, [0.1, 0.9], axis=1), color=color, alpha=0.12)
        title = "fraction of paths leaving the roll" if metric == "leaves_domain" else metric.replace("_", " ")
        ax.set(xscale="log", xlabel="Euler steps", title=title)
        if metric != "leaves_domain":
            ax.set_yscale("log")
        ax.grid(alpha=0.25)
    axes[0].legend(fontsize=8)
    fig.suptitle("Conditional RFM paths vs. exact geodesics (mean over pairs, 10-90% band)")
    plt.show()


def plot_target_speeds(speeds, steps, max_velocity_norm=None):
    """Unclipped target speed ``||u_t||`` at the states of each variant's own path (median, 10-90% band)."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2), constrained_layout=True)
    for variant in dict.fromkeys(variant for _, variant in speeds):
        entry = speeds[steps, variant]
        times, values = entry["times"], entry["unclipped"]
        normalized = values / np.maximum(values[:1], 1e-12)
        color = VARIANT_COLORS[variant]
        for ax, data in zip(axes, (values, normalized), strict=True):
            ax.plot(times, np.median(data, axis=1), color=color, label=VARIANT_LABELS[variant])
            ax.fill_between(times, *np.quantile(data, [0.1, 0.9], axis=1), color=color, alpha=0.12)
    if max_velocity_norm is not None:
        axes[0].axhline(max_velocity_norm, color="black", linestyle=":", label="max_velocity_norm (clipping)")
    axes[0].set(xlabel="t", ylabel="||u_t|| (unclipped)", yscale="log", title=f"Target speed, {steps} Euler steps")
    axes[1].set(xlabel="t", ylabel="||u_t|| / ||u_0||", yscale="log", title="Target speed relative to t = 0")
    for ax in axes:
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8)
    plt.show()


def plot_example_paths(geometry, x0, x1, paths, steps_list, pair_ids, geodesic_times=np.linspace(0.0, 1.0, 400)):
    """For each selected pair (columns) and variant (rows), the geodesic and the Euler path per step count."""
    variants = list(dict.fromkeys(variant for _, variant in paths))
    geodesics, _ = geometry.geodesic(x0[pair_ids], x1[pair_ids], geodesic_times)
    fig, axes = plt.subplots(len(variants), len(pair_ids), figsize=(4.2 * len(pair_ids), 4.2 * len(variants)),
                             squeeze=False)
    shades = plt.get_cmap("viridis")(np.linspace(0.0, 0.85, len(steps_list)))
    for row, scale in enumerate(variants):
        for column, (pair_id, geodesic) in enumerate(zip(pair_ids, geodesics, strict=True)):
            ax = axes[row, column]
            ax.plot(*geodesic.T, color="black", linestyle="--", linewidth=2.2, label="geodesic", zorder=5)
            for shade, steps in zip(shades, steps_list, strict=True):
                ax.plot(*paths[steps, scale][1][pair_id].T, color=shade, linewidth=1.4, label=f"{steps} steps")
            ax.scatter(*x0[pair_id], color="tab:blue", s=50, zorder=6)
            ax.scatter(*x1[pair_id], color="tab:orange", marker="X", s=80, zorder=7)
            geometry.finish_axes(ax)
            ax.set_title(f"{scale}: pair {pair_id}", fontsize=10)
    axes[0, 0].legend(fontsize=7, loc="lower left")
    fig.suptitle("Euler paths of the RFM targets; rows: " + "; ".join(VARIANT_LABELS[v] for v in variants))
    fig.tight_layout()
    plt.show()


# 2. Training --------------------------------------------------------------------------------------
def held_out_states(nystroem, variant, projection_fn, max_velocity_norm, source, target, coupling, steps, n_pairs,
                    seed):
    """RFM states and the variant's own targets for held-out coupled pairs, on random-offset grids."""
    rng = np.random.default_rng(seed)
    flat = coupling.ravel() / coupling.sum()
    source_ids, target_ids = np.unravel_index(rng.choice(flat.size, n_pairs, p=flat), coupling.shape)
    offsets = rng.uniform(size=(n_pairs, 1)).astype(np.float32)
    interpolator = make_interpolator(nystroem, variant, projection_fn, max_velocity_norm)
    ts, xs, us = jax.jit(interpolator.trajectory, static_argnums=2)(
        jnp.asarray(source[source_ids]), jnp.asarray(target[target_ids]), steps, jnp.asarray(offsets))
    return tuple(np.asarray(a).reshape(-1, a.shape[-1]) for a in (ts, xs, us))


def run_metrics(model, eval_states, source, target, geometry, fresh_starts, save_times):
    """Endpoint metrics of the fitted source, fresh-path metrics, and field error against the own target."""
    ts, xs, us = eval_states
    error = evaluate_learned_field(model, xs, ts) - us
    sq = np.sum(error**2, axis=1)
    relative = np.linalg.norm(error, axis=1) / np.maximum(np.linalg.norm(us, axis=1), 1e-8)
    _, condition, _ = inference_inputs(model)
    predicted = np.asarray(model.solver.predict(source, condition))
    paths, solver_steps = integrate_learned_paths(model, fresh_starts, save_times, check_prediction=False)
    path_metrics = geometry.path_metrics(paths)
    return {
        **{key: float(value) for key, value in endpoint_metrics(predicted, target, geometry).items()},
        "fresh_paths_leaving_roll_fraction": path_metrics["paths_leaving_roll"] / len(paths),
        "fresh_paths_skipping_a_turn": path_metrics["paths_skipping_a_turn"],
        "fresh_saved_points_in_roll": path_metrics["fraction_saved_points_in_roll"],
        "max_segment_gap_penetration": path_metrics["max_segment_gap_penetration"],
        "fresh_mean_path_length": path_metrics["mean_path_length"],
        "mean_adaptive_solver_steps": float(np.mean(solver_steps)),
        "field_mse": float(sq.mean() / xs.shape[1]),
        "field_median_relative_error": float(np.median(relative)),
        "field_mse_by_t_decile": [
            float(sq[(ts[:, 0] >= b / 10) & (ts[:, 0] < (b + 1) / 10)].mean() / xs.shape[1]) for b in range(10)
        ],
        "predicted": predicted.tolist(),
    }


def build_model(variant, loss_mode, seed, *, adata, target_key, architecture_kwargs, optimizer, match_fn, nystroem,
                geometry, interpolation_steps, max_velocity_norm):
    """Untrained RFM model with the variant's target speed and clipping; everything else is shared."""
    scale, clip_norm = variant_settings(variant, max_velocity_norm)
    model = prepare_toy_model(
        adata.copy(), "otrfm", target_key, **architecture_kwargs, match_fn=match_fn, nystroem=nystroem,
        optimizer=optimizer, seed=seed,
        solver_kwargs={
            "interpolation_steps": interpolation_steps, "projection_fn": geometry.project,
            "project_to_manifold": False, "max_velocity_norm": clip_norm,
            "loss_mode": loss_mode, "velocity_scale": scale,
        },
    )
    assert model.solver.velocity_scale == scale and model.solver.max_velocity_norm == clip_norm
    return model


def run_experiment(*, variants, loss_modes, seeds, adata, target_key, architecture_kwargs, optimizer, match_fn,
                   nystroem, geometry, source, target, coupling, fresh_starts, interpolation_steps,
                   max_velocity_norm, n_iterations, batch_size, n_eval_pairs=512, eval_seed=999):
    """Train one model per ``(variant, loss_mode, seed)``; return one row per run.

    The field error of each model is measured against its own variant's targets on held-out states.
    """
    save_times = jnp.linspace(0.0, 1.0, 201)
    eval_states = {
        variant: held_out_states(nystroem, variant, geometry.project, max_velocity_norm, source, target, coupling,
                                 interpolation_steps, n_eval_pairs, eval_seed)
        for variant in variants
    }
    rows = []
    for variant in variants:
        scale, clip_norm = variant_settings(variant, max_velocity_norm)
        for loss_mode in loss_modes:
            for seed in seeds:
                started = time.perf_counter()
                model = build_model(
                    variant, loss_mode, seed, adata=adata, target_key=target_key,
                    architecture_kwargs=architecture_kwargs, optimizer=optimizer, match_fn=match_fn,
                    nystroem=nystroem, geometry=geometry, interpolation_steps=interpolation_steps,
                    max_velocity_norm=max_velocity_norm,
                )
                losses, iteration_seconds = train_seeded(model, seed, n_iterations, batch_size)
                rows.append({
                    "variant": variant, "velocity_scale": scale, "max_velocity_norm": clip_norm,
                    "loss_mode": loss_mode, "seed": seed,
                    **run_metrics(model, eval_states[variant], source, target, geometry, fresh_starts, save_times),
                    "median_iter_seconds": float(np.median(iteration_seconds[1:])),
                    "train_seconds_total": float(iteration_seconds.sum()),
                    "loss_first_25": float(losses[:25].mean()),
                    "loss_last_25": float(losses[-25:].mean()),
                    "losses": losses.tolist(),
                })
                print(f"{variant:>16}, {loss_mode:>10}, seed {seed}: energy² {rows[-1]['energy_distance_squared']:.4f}, "
                      f"{time.perf_counter() - started:.0f}s", flush=True)
    return rows


def save_rows(results_path, rows, settings):
    with open(results_path, "w", encoding="utf-8") as handle:
        json.dump({"settings": settings, "rows": rows}, handle)


def load_or_run(results_path, run, settings, **run_kwargs):
    """Run :func:`run_experiment` and save the rows if ``run``; otherwise load the saved rows.

    ``settings`` (JSON-serializable) is stored with the rows, and checked against the saved ones on load.
    """
    if run:
        rows = run_experiment(**run_kwargs)
        save_rows(results_path, rows, settings)
    else:
        with open(results_path, encoding="utf-8") as handle:
            saved = json.load(handle)
        if saved["settings"] != settings:
            raise ValueError(f"Saved results use different settings: {saved['settings']} != {settings}.")
        rows = saved["rows"]
    return rows


def results_table(frame, columns, digits=4):
    """Mean ± sd over seeds per ``(loss_mode, variant)``."""
    grouped = frame.groupby(["loss_mode", "variant"], sort=False)[columns]
    mean, sd = grouped.mean(), grouped.std()
    return mean.round(digits).astype(str) + " ± " + sd.round(digits).astype(str)


def wins_against(frame, columns, baseline="current"):
    """Per loss mode and variant: on how many seeds the variant has a lower value than ``baseline``."""
    indexed = frame.set_index(["loss_mode", "variant", "seed"])[columns]
    base = indexed.xs(baseline, level="variant")
    others = indexed[indexed.index.get_level_values("variant") != baseline]
    reference = base.loc[others.index.droplevel("variant")].to_numpy()
    return others.lt(reference).groupby(level=["loss_mode", "variant"], sort=False).sum().astype(int)


def plot_training(frame, loss_modes):
    """Loss curves (per seed) and field error by time decile (mean over seeds)."""
    fig, axes = plt.subplots(1, len(loss_modes) + 1, figsize=(6 * (len(loss_modes) + 1), 4.4),
                             constrained_layout=True)
    for ax, loss_mode in zip(axes, loss_modes, strict=False):
        for row in frame[frame["loss_mode"].eq(loss_mode)].itertuples():
            ax.plot(rolling_mean(np.asarray(row.losses)), color=VARIANT_COLORS[row.variant], alpha=0.8,
                    label=VARIANT_LABELS[row.variant] if row.seed == frame["seed"].min() else None)
        ax.set(xlabel="Iteration", ylabel="Training loss (25-step mean)", yscale="log",
               title=f"{loss_mode} loss (each target regressed onto itself)")
        ax.grid(alpha=0.2)
        ax.legend(fontsize=8)
    ax = axes[-1]
    for (loss_mode, variant), group in frame.groupby(["loss_mode", "variant"], sort=False):
        deciles = np.array(group["field_mse_by_t_decile"].tolist()).mean(axis=0)
        ax.plot(np.arange(10) / 10 + 0.05, deciles, marker="o", color=VARIANT_COLORS[variant],
                linestyle=LOSS_MODE_STYLES[loss_mode], label=f"{variant}, {loss_mode}")
    ax.set(xlabel="t", ylabel="MSE vs. own target on held-out states", yscale="log",
           title="Field error by time (mean over seeds)")
    ax.grid(alpha=0.2)
    ax.legend(fontsize=8)
    plt.show()


def plot_metric_dots(frame, metrics):
    """Per-seed values of each metric, grouped by loss mode, one color per variant."""
    loss_modes = list(dict.fromkeys(frame["loss_mode"]))
    variants = list(dict.fromkeys(frame["variant"]))
    offsets = np.linspace(-0.2, 0.2, len(variants))
    fig, axes = plt.subplots(1, len(metrics), figsize=(4.8 * len(metrics), 4.2), constrained_layout=True)
    for ax, (column, title) in zip(np.atleast_1d(axes), metrics, strict=True):
        for position, loss_mode in enumerate(loss_modes):
            for offset, variant in zip(offsets, variants, strict=True):
                values = frame.loc[frame["loss_mode"].eq(loss_mode) & frame["variant"].eq(variant), column]
                color = VARIANT_COLORS[variant]
                ax.scatter(np.full(len(values), position + offset), values, color=color, zorder=3,
                           label=VARIANT_LABELS[variant] if position == 0 else None)
                ax.hlines(values.mean(), position + offset - 0.06, position + offset + 0.06, color=color,
                          linewidth=2.5)
        ax.set(xticks=range(len(loss_modes)), xticklabels=loss_modes, title=title)
        ax.grid(axis="y", alpha=0.2)
    np.atleast_1d(axes)[0].legend(fontsize=8)
    plt.show()


def plot_predictions(geometry, frame, source, target, seed):
    """Predicted endpoints of one seed for every ``(loss_mode, variant)``: loss modes in rows."""
    loss_modes = list(dict.fromkeys(frame["loss_mode"]))
    variants = list(dict.fromkeys(frame["variant"]))
    fig, axes = plt.subplots(len(loss_modes), len(variants), figsize=(5 * len(variants), 5 * len(loss_modes)),
                             constrained_layout=True, squeeze=False)
    for (row_id, loss_mode), (column, scale) in ((r, c) for r in enumerate(loss_modes) for c in enumerate(variants)):
        ax = axes[row_id, column]
        row = frame[frame["loss_mode"].eq(loss_mode) & frame["variant"].eq(scale) & frame["seed"].eq(seed)]
        predicted = np.asarray(row["predicted"].iloc[0])
        geometry.scatter(ax, source, s=6, alpha=0.15, color="tab:gray", label="Source")
        geometry.scatter(ax, target, s=8, alpha=0.3, color="tab:orange", label="Target")
        geometry.scatter(ax, predicted, s=8, alpha=0.7, color=VARIANT_COLORS[scale], label="Prediction")
        ax.set_title(f"{scale}, {loss_mode} (seed {seed})\nenergy² "
                     f"{row['energy_distance_squared'].iloc[0]:.4f}, outside "
                     f"{row['outside_roll_fraction'].iloc[0]:.0%}", fontsize=10)
        geometry.finish_axes(ax)
    axes[0, 0].legend(fontsize=8, loc="upper right")
    plt.show()


# 3. Example models: learned fields and fresh-sample paths -----------------------------------------
EXAMPLE_COLORS = {
    ("final", "current"): "tab:green", ("final", "initial"): "tab:blue", ("final", "initial, no clip"): "tab:purple",
    ("trajectory", "current"): "darkolivegreen", ("trajectory", "initial"): "tab:cyan",
    ("trajectory", "initial, no clip"): "tab:pink",
}


def example_name(variant, loss_mode):
    return f"{variant} · {loss_mode} loss"


def _set_inference_params(model, flat_params):
    params = unflatten_dict({tuple(name.split("/")): jnp.asarray(value) for name, value in flat_params.items()})
    solver = model.solver
    solver.vf_state = solver.vf_state.replace(params=params)
    solver.vf_state_inference = solver.vf_state_inference.replace(params=params)
    solver.is_trained = True


def load_or_train_examples(path, run, settings, *, variants, loss_modes, seed, n_iterations, batch_size,
                           **build_kwargs):
    """One trained model per ``(variant, loss_mode)`` for ``seed``, keyed ``(variant, loss_mode)``.

    The models are built by :func:`build_model` and trained by the same seeded loop as :func:`run_experiment`.
    With ``run``, their inference weights are saved to ``path`` (``.npz``) together with ``settings``;
    otherwise the weights are loaded from there and ``settings`` must match the saved ones.
    """
    models = {(variant, loss_mode): build_model(variant, loss_mode, seed, **build_kwargs)
              for variant in variants for loss_mode in loss_modes}
    if run:
        arrays = {"__settings__": np.array(json.dumps(settings))}
        for (variant, loss_mode), model in models.items():
            started = time.perf_counter()
            train_seeded(model, seed, n_iterations, batch_size)
            for name, value in flatten_dict(model.solver.vf_state_inference.params, sep="/").items():
                arrays[f"{variant}|{loss_mode}|{name}"] = np.asarray(value)
            print(f"Trained {example_name(variant, loss_mode)} in {time.perf_counter() - started:.0f}s", flush=True)
        np.savez(path, **arrays)
    else:
        with np.load(path) as saved:
            if json.loads(str(saved["__settings__"])) != settings:
                raise ValueError(f"Saved example models use different settings than {settings}.")
            for (variant, loss_mode), model in models.items():
                prefix = f"{variant}|{loss_mode}|"
                _set_inference_params(model, {key[len(prefix):]: saved[key] for key in saved.files
                                              if key.startswith(prefix)})
    return models


def check_examples_against_runs(models, frame, seed, source):
    """Largest endpoint difference between each example model and the saved run with the same settings."""
    rows = []
    for (variant, loss_mode), model in models.items():
        saved = frame[(frame["variant"] == variant) & (frame["loss_mode"] == loss_mode) & (frame["seed"] == seed)]
        _, condition, _ = inference_inputs(model)
        predicted = np.asarray(model.solver.predict(source, condition))
        rows.append({"variant": variant, "loss_mode": loss_mode,
                     "max_endpoint_difference": float(np.abs(predicted - np.asarray(saved["predicted"].iloc[0])).max()),
                     "saved_energy_distance_squared": float(saved["energy_distance_squared"].iloc[0])})
    return pd.DataFrame(rows).set_index(["loss_mode", "variant"])


def field_table(models, grid, times, baseline="current"):
    """Speed of each learned field on ``grid`` and its direction agreement with ``baseline`` (same loss mode)."""
    fields = {(key, t): evaluate_learned_field(model, grid, t) for key, model in models.items() for t in times}
    rows = []
    for ((variant, loss_mode), t), velocities in fields.items():
        reference = fields[(baseline, loss_mode), t]
        speed, reference_speed = np.linalg.norm(velocities, axis=1), np.linalg.norm(reference, axis=1)
        cosine = np.sum(velocities * reference, axis=1) / np.maximum(speed * reference_speed, 1e-12)
        rows.append({"loss_mode": loss_mode, "variant": variant, "t": t, "median_speed": np.median(speed),
                     "max_speed": speed.max(), f"mean_cosine_to_{baseline}": cosine.mean(),
                     f"fraction_cosine_to_{baseline}_below_0.5": np.mean(cosine < 0.5)})
    return pd.DataFrame(rows).set_index(["loss_mode", "t", "variant"]).sort_index(level=["loss_mode", "t"],
                                                                                 sort_remaining=False)


def plot_variant_fields(geometry, models, loss_mode, times, source, target, grid_size=27, arrow_length=0.35):
    """Learned fields of every variant (columns) at several times (rows) for one loss mode.

    Within a row, arrows and colors share one scale: the fastest arrow in the row is ``arrow_length`` long.
    Rows have their own scale, because the speeds differ by orders of magnitude between early and late times.
    """
    from matplotlib.colors import Normalize

    from riemannian_toys.plotting import domain_axes

    grid = geometry.learned_field_grid(grid_size)
    variants = [variant for variant, mode in models if mode == loss_mode]
    fig, axes = domain_axes(geometry, len(times), len(variants), figsize=(5.4 * len(variants), 5.0 * len(times)),
                            constrained_layout=True, squeeze=False)
    for row, t in zip(axes, times, strict=True):
        fields = {variant: evaluate_learned_field(models[variant, loss_mode], grid, t) for variant in variants}
        speeds = {variant: np.linalg.norm(velocities, axis=1) for variant, velocities in fields.items()}
        max_speed = max(max(float(s.max()) for s in speeds.values()), 1e-6)
        norm = Normalize(0.0, max_speed)
        for ax, variant in zip(row, variants, strict=True):
            geometry.scatter(ax, source, s=5, alpha=0.12, color="tab:blue")
            geometry.scatter(ax, target, s=5, alpha=0.12, color="tab:orange")
            geometry.draw_arrows(ax, grid, arrow_length / max_speed * fields[variant], speeds[variant], norm, "viridis",
                                 alpha=0.85)
            geometry.finish_axes(ax, outer_color="gray")
            ax.set_title(f"{VARIANT_LABELS[variant]}\nt = {t:.2f}, median speed {np.median(speeds[variant]):.1f}",
                         fontsize=10)
        fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap="viridis"), ax=list(row), shrink=0.85,
                     label=f"learned speed at t = {t:.2f}")
    fig.suptitle(f"Learned fields, {loss_mode} loss: one arrow and color scale per row "
                 f"(the fastest arrow in a row is {arrow_length:g} long); source blue, target orange")
    plt.show()


__all__ = [
    "EXAMPLE_COLORS", "PROBLEM_SETTINGS", "VARIANTS", "VARIANT_COLORS", "VARIANT_LABELS", "build_model",
    "build_swiss_roll_problem", "check_examples_against_runs", "example_name", "field_table", "load_or_run",
    "load_or_train_examples", "make_interpolator", "plot_example_paths", "plot_metric_dots", "plot_predictions",
    "plot_reference_convergence", "plot_target_speeds", "plot_training", "plot_variant_fields",
    "reference_path_study", "reference_summary", "results_table", "run_experiment", "save_rows", "variant_settings",
    "wins_against",
]


# Shared problem ----------------------------------------------------------------------------------
PROBLEM_SETTINGS = {
    "roll": {"inner_radius": 1.0, "pitch": 1.2, "half_width": 0.3, "turns": 1.5},
    "n_construction_points": 5000, "construction_seed": 42,
    "gaussian_seed": 123, "variance": 0.09, "n_points_per_gaussian": 250, "end_offset": 1.2,
    "kernel": "fixed_gaussian", "density_k": 24, "delta": 2.0, "n_modes": 32,
    "laplacian_type": "unnormalized", "spectral_weight_type": "biharmonic",
    "ot_epsilon": 0.05, "n_fresh_starts": 200, "fresh_seed": 4242,
}
"""Same swiss roll, clouds, geometry and coupling as ``riemannian_swiss_roll_toy_example.ipynb``."""


def build_swiss_roll_problem(settings=PROBLEM_SETTINGS):
    """Roll, source (outer end) and target (inner end) clouds, fixed-Gaussian biharmonic Nyström model,
    biharmonic entropic OT, its full-cloud coupling, and fresh source samples for path diagnostics."""
    from types import SimpleNamespace

    from riemannian_toys import flows
    from riemannian_toys.swiss_roll import SwissRoll
    from cellflow.preprocessing import fit_kernel_nystroem

    roll = SwissRoll(**settings["roll"])
    construction_points = roll.sample_uniform(
        settings["n_construction_points"], np.random.default_rng(settings["construction_seed"])).astype(np.float32)
    rng = np.random.default_rng(settings["gaussian_seed"])
    end_offset, variance, n = settings["end_offset"], settings["variance"], settings["n_points_per_gaussian"]
    mean_outer = roll.strip_point(roll.theta_at_arc_length(roll.centerline_length - end_offset))
    mean_inner = roll.strip_point(roll.theta_at_arc_length(end_offset))
    outer_window = (roll.theta_max - np.pi, roll.theta_max)
    source = roll.sample_gaussian(mean_outer, variance, n, rng, theta_window=outer_window)[0].astype(np.float32)
    target = roll.sample_gaussian(mean_inner, variance, n, rng, theta_window=(0.0, np.pi))[0].astype(np.float32)
    fit = fit_kernel_nystroem(
        construction_points, settings["kernel"], density_k=settings["density_k"], delta=settings["delta"],
        n_modes=settings["n_modes"], laplacian_type=settings["laplacian_type"],
        spectral_weight_type=settings["spectral_weight_type"],
    )
    embed = jax.jit(jax.vmap(fit.model.extend_point))
    match_fn = flows.make_entropic_match_fn(settings["ot_epsilon"], embed=embed)
    coupling = np.asarray(jax.jit(match_fn)(source, target))
    flows.check_coupling(coupling, len(source), len(target))
    fresh_starts = roll.sample_gaussian(
        mean_outer, variance, settings["n_fresh_starts"], np.random.default_rng(settings["fresh_seed"]),
        theta_window=outer_window)[0].astype(np.float32)
    return SimpleNamespace(
        roll=roll, construction_points=construction_points, source=source, target=target, nystroem=fit.model,
        fit_diagnostics=fit.diagnostics, match_fn=match_fn, coupling=coupling, fresh_starts=fresh_starts,
        mean_outer=mean_outer, mean_inner=mean_inner,
    )


TRAINING_SETTINGS = {
    "loss_modes": ["final", "trajectory"], "seeds": [2026, 1, 2], "learning_rate": 1e-3,
    "interpolation_steps": 32, "max_velocity_norm": 30.0, "n_iterations": 300, "batch_size": 32,
}
"""Same training settings as the swiss-roll notebook's RFM model (32 steps, 300 iterations, batch 32)."""


def run_settings():
    """Everything that determines the saved training results."""
    return {"problem": PROBLEM_SETTINGS, "training": TRAINING_SETTINGS, "variants": {k: list(v) for k, v in VARIANTS.items()}}
