"""CellFlow helpers for the toy notebooks: data, OT matching, training, prediction, learned fields and paths."""

import time

import anndata as ad
import diffrax
import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
from ott.geometry import costs
from scipy.spatial.distance import cdist

from cellflow.data._dataloader import TrainSampler
from cellflow.model import CellFlow
from cellflow.utils import match_linear

SOURCE_KEY = "control"

DEFAULT_ARCHITECTURE = {
    "condition_mode": "deterministic",
    "pooling": "mean",
    "layers_before_pool": [],
    "layers_after_pool": [],
    "condition_embedding_dim": 16,
    "cond_output_dropout": 0.0,
    "time_freqs": 16,
    "time_encoder_dims": (32, 32),
    "hidden_dims": (64, 64),
    "decoder_dims": (64, 64),
    "conditioning_kwargs": {},
}
"""Compact velocity-field network used by every toy model."""

ARCHITECTURE_NAMES = (
    "condition_mode", "regularization", "condition_embedding_dim",
    "pooling", "pooling_kwargs", "layers_before_pool", "layers_after_pool",
    "cond_output_dropout", "condition_encoder_kwargs",
    "time_freqs", "time_max_period", "time_encoder_dims", "time_encoder_dropout",
    "hidden_dims", "hidden_dropout", "conditioning", "conditioning_kwargs",
    "decoder_dims", "decoder_dropout",
    "layer_norm_before_concatenation", "linear_projection_before_concatenation",
)
"""Velocity-field attributes that :func:`copy_architecture_kwargs` copies from a fitted model."""


# Data --------------------------------------------------------------------------------------------
def make_toy_adata(source, target, target_key):
    """AnnData with the source cloud as control and the target cloud flagged by ``target_key``."""
    n_source, n_target = len(source), len(target)
    return ad.AnnData(
        X=np.vstack((source, target)),
        obs=pd.DataFrame(
            {
                SOURCE_KEY: np.r_[np.ones(n_source, dtype=bool), np.zeros(n_target, dtype=bool)],
                target_key: np.r_[np.zeros(n_source, dtype=bool), np.ones(n_target, dtype=bool)],
            },
            index=[f"toy_{i}" for i in range(n_source + n_target)],
        ),
    )


def prepare_toy_model(adata, solver, target_key, **prepare_model_kwargs):
    """``CellFlow`` on ``adata`` with data prepared for the single transport condition, then ``prepare_model``."""
    model = CellFlow(adata, solver=solver)
    model.prepare_data(sample_rep="X", control_key=SOURCE_KEY, perturbation_covariates={"transport": [target_key]})
    model.prepare_model(**prepare_model_kwargs)
    return model


def check_source_target_split(model, batch_size, seed, is_source, is_target):
    """Assert that the default dataloader samples source cells as source and target cells as target."""
    batch = TrainSampler(model.train_data, batch_size=batch_size).sample(np.random.default_rng(seed))
    assert np.all(is_source(np.asarray(batch["src_cell_data"])))
    assert np.all(is_target(np.asarray(batch["tgt_cell_data"])))


# OT matching -------------------------------------------------------------------------------------
def make_entropic_match_fn(epsilon, embed=None):
    """Balanced entropic OT with unsquared Euclidean cost, optionally in an embedding.

    With ``embed`` set to a Nyström embedding, the Euclidean cost in the weighted spectral
    coordinates is the biharmonic distance. ``epsilon`` is relative to the mean batch cost.
    """
    embed = (lambda points: points) if embed is None else embed

    def match(source_batch, target_batch):
        return match_linear(
            embed(jnp.asarray(source_batch)),
            embed(jnp.asarray(target_batch)),
            cost_fn=costs.Euclidean(),
            epsilon=epsilon,
            scale_cost="mean",
            tau_a=1.0,
            tau_b=1.0,
            threshold=1e-5,
            max_iterations=5000,
        )

    return match


def check_coupling(coupling, n_source, n_target, atol=5e-5):
    """Assert a finite, non-negative coupling with uniform marginals."""
    assert coupling.shape == (n_source, n_target)
    assert np.all(np.isfinite(coupling)) and np.all(coupling >= 0.0)
    np.testing.assert_allclose(coupling.sum(axis=1), 1 / n_source, atol=atol)
    np.testing.assert_allclose(coupling.sum(axis=0), 1 / n_target, atol=atol)


def check_rectangular_batch(match_jit, source, target):
    """A reordered, rectangular batch also works without global point indices."""
    coupling = np.asarray(match_jit(source[7:0:-1], target[2:11]))
    assert coupling.shape == (7, 9) and np.all(np.isfinite(coupling))


def coupling_diagnostics(coupling, cost_matrix, cost_name, label="coupling diagnostics"):
    """Mass, marginal errors and transported vs. independent mean cost, as a one-column table."""
    n_source, n_target = coupling.shape
    return pd.DataFrame({
        "coupling_mass": coupling.sum(),
        "max_source_marginal_error": np.max(np.abs(coupling.sum(axis=1) - 1 / n_source)),
        "max_target_marginal_error": np.max(np.abs(coupling.sum(axis=0) - 1 / n_target)),
        f"OT_mean_{cost_name}_cost": np.sum(coupling * cost_matrix),
        f"independent_mean_{cost_name}_cost": cost_matrix.mean(),
    }, index=[label]).T


# Training and prediction -------------------------------------------------------------------------
def train_timed(model, num_iterations, batch_size, valid_freq=100):
    """Train with ``CellFlow.train`` and return ``(losses, seconds)``."""
    started = time.perf_counter()
    model.train(num_iterations=num_iterations, batch_size=batch_size, valid_freq=valid_freq)
    seconds = time.perf_counter() - started
    losses = np.asarray(model.trainer.training_logs["loss"])
    assert len(losses) == num_iterations and np.all(np.isfinite(losses))
    assert model.solver.is_trained
    return losses, seconds


def loss_summary(losses):
    return f"mean loss first 25 {losses[:25].mean():.5f}, last 25 {losses[-25:].mean():.5f}"


def predict_target(model, adata, target_key, condition_name):
    """Push the control cells of ``adata`` through the public ``predict`` API."""
    source_adata = adata[adata.obs[SOURCE_KEY]].copy()
    predictions = model.predict(
        source_adata,
        covariate_data=pd.DataFrame({SOURCE_KEY: [True], target_key: [True], "condition": [condition_name]}),
        condition_id_key="condition",
    )
    predicted = np.asarray(predictions[condition_name])
    assert predicted.shape == (len(source_adata), adata.n_vars) and np.all(np.isfinite(predicted))
    return predicted


def distribution_metrics(samples, target, geometry):
    """Squared energy distance, mean-position error and off-domain fraction; no pairing assumed."""
    samples = np.asarray(samples)
    energy = 2 * cdist(samples, target).mean() - cdist(samples, samples).mean() - cdist(target, target).mean()
    return {
        "energy_distance_squared": float(max(energy, 0.0)),
        "mean_error_to_target": float(np.linalg.norm(samples.mean(axis=0) - target.mean(axis=0))),
        f"outside_{geometry.name}_fraction": float(np.mean(~geometry.contains(samples))),
    }


def endpoint_metrics(samples, target, geometry):
    """:func:`distribution_metrics` plus the geometry's endpoint diagnostics."""
    return {**distribution_metrics(samples, target, geometry), **geometry.endpoint_metrics(samples)}


# Comparison models -------------------------------------------------------------------------------
def copy_architecture_kwargs(model):
    """``prepare_model`` keyword arguments that reproduce ``model``'s velocity-field network."""
    kwargs = {name: getattr(model.vf, name) for name in ARCHITECTURE_NAMES}
    kwargs["vf_act_fn"] = model.vf.act_fn
    return kwargs


def shared_solver_kwargs(model):
    return {"time_sampler": model.solver.time_sampler, "ema": model.solver.ema}


def rfm_solver_kwargs(model, **overrides):
    """Solver keyword arguments that reproduce ``model``'s Riemannian interpolation settings."""
    solver = model.solver
    return {
        **shared_solver_kwargs(model),
        "interpolation_steps": solver.interpolation_steps,
        "projection_fn": solver.projection_fn,
        "project_to_manifold": solver.project_to_manifold,
        "min_gradient_norm_sq": solver.min_gradient_norm_sq,
        "time_epsilon": solver.time_epsilon,
        "max_velocity_norm": solver.max_velocity_norm,
        **overrides,
    }


def build_matched_model(reference, adata, solver, match_fn, target_key, seed, nystroem=None, solver_kwargs=None):
    """A fresh model on a copy of ``adata`` with ``reference``'s network, optimizer and training settings.

    ``solver="otrfm"`` reuses the reference's Riemannian interpolation settings (overridable via
    ``solver_kwargs``) and ``nystroem``. ``solver="otfm"`` uses the zero-noise straight path.
    """
    from cellflow.solvers import OTFlowMatching
    from cellflow.solvers._otrfm import OTRFlowMatching

    common = {
        **copy_architecture_kwargs(reference), "match_fn": match_fn,
        "optimizer": reference.solver.vf_state.tx, "seed": seed,
    }
    if solver == "otrfm":
        model = prepare_toy_model(
            adata.copy(), solver, target_key, **common, nystroem=nystroem,
            solver_kwargs=rfm_solver_kwargs(reference, **(solver_kwargs or {})),
        )
        assert type(model.solver) is OTRFlowMatching and model.solver.nystroem is nystroem
    else:
        model = prepare_toy_model(
            adata.copy(), solver, target_key, **common, probability_path={"constant_noise": 0.0},
            solver_kwargs={**shared_solver_kwargs(reference), **(solver_kwargs or {})},
        )
        assert type(model.solver) is OTFlowMatching and model.solver.probability_path.sigma == 0.0
    assert model.vf == reference.vf
    assert int(model.solver.vf_state.step) == 0
    np.testing.assert_array_equal(model.train_data.cell_data, reference.train_data.cell_data)
    for key in reference.train_data.condition_data:
        np.testing.assert_array_equal(model.train_data.condition_data[key], reference.train_data.condition_data[key])
    return model


# Learned fields and paths ------------------------------------------------------------------------
def inference_inputs(model):
    """Inference state, the single target condition, and deterministic encoder noise, as in ``predict``."""
    state = model.solver.vf_state_inference
    condition = {key: jnp.asarray(values[:1]) for key, values in model.train_data.condition_data.items()}
    encoder_noise = jnp.zeros((1, model.solver.vf.condition_embedding_dim))
    return state, condition, encoder_noise


def evaluate_learned_field(model, positions, times):
    """``v_θ(t, x | target)`` with inference weights and dropout disabled.

    ``times`` is a scalar or one time per position, shape ``(n, 1)``.
    """
    if not model.solver.is_trained:
        raise ValueError("Train the model first.")
    state, condition, encoder_noise = inference_inputs(model)
    positions = jnp.asarray(positions)
    times = jnp.broadcast_to(jnp.asarray(times, dtype=positions.dtype), (positions.shape[0], 1))

    @jax.jit
    def evaluate(query_times, query_positions):
        return state.apply_fn(
            {"params": state.params}, query_times, query_positions, condition, encoder_noise=encoder_noise,
            train=False,
        )[0]

    velocities = np.asarray(evaluate(times, positions))
    if not np.all(np.isfinite(velocities)):
        raise ValueError("The learned field contains non-finite velocities.")
    return velocities


def integrate_learned_paths(model, starts, save_times, rtol=1e-5, atol=1e-5, check_prediction=True):
    """Integrate the learned field from each start with the predictor's Tsit5 settings, saving all times.

    Returns ``(paths, solver_steps)`` with paths of shape ``(n, len(save_times), d)``. With
    ``check_prediction``, the endpoints must match ``solver.predict``.
    """
    if not model.solver.is_trained:
        raise ValueError("Train the model first.")
    state, condition, encoder_noise = inference_inputs(model)
    starts = np.asarray(starts)

    def velocity(t, position, params):
        return state.apply_fn({"params": params}, t, position, condition, encoder_noise, train=False)[0]

    def integrate_one(params, start):
        solution = diffrax.diffeqsolve(
            diffrax.ODETerm(velocity), solver=diffrax.Tsit5(), t0=0.0, t1=1.0, dt0=None, y0=start, args=params,
            stepsize_controller=diffrax.PIDController(rtol=rtol, atol=atol),
            saveat=diffrax.SaveAt(ts=jnp.asarray(save_times)),
        )
        return solution.ys, solution.stats["num_steps"]

    paths, steps = jax.jit(jax.vmap(integrate_one, in_axes=(None, 0)))(state.params, jnp.asarray(starts))
    paths = np.asarray(paths)
    assert paths.shape == (len(starts), len(save_times), starts.shape[1]) and np.all(np.isfinite(paths))
    np.testing.assert_allclose(paths[:, 0], starts)
    if check_prediction:
        np.testing.assert_allclose(paths[:, -1], model.solver.predict(starts, condition), rtol=1e-4, atol=1e-4)
    return paths, np.asarray(steps)


def path_domain_summary(geometry, starts, trajectories, solver_steps, tol=1e-4):
    """Per-path start, end, range of the radial coordinate, and domain membership."""
    radial = geometry.radial_coordinate(trajectories)
    inside = geometry.contains(trajectories, tol=tol)
    axes = "xyz"[: trajectories.shape[-1]]
    radial_name = "radius" if geometry.name == "annulus" else "tube_radius"
    return pd.DataFrame(
        {
            **{f"start_{a}": starts[:, i] for i, a in enumerate(axes)},
            **{f"end_{a}": trajectories[:, -1, i] for i, a in enumerate(axes)},
            f"min_sampled_{radial_name}": radial.min(axis=1),
            f"max_sampled_{radial_name}": radial.max(axis=1),
            f"endpoint_in_{geometry.name}": inside[:, -1],
            f"all_saved_points_in_{geometry.name}": inside.all(axis=1),
            "adaptive_solver_steps": np.asarray(solver_steps),
        },
        index=[f"Path {i + 1}" for i in range(len(starts))],
    )
