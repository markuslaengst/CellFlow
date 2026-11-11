"""Intermediate marginals of learned flows compared to a Gaussian that travels along the swiss roll.

The reference family ``p_t`` is an isotropic Gaussian conditioned on the strip whose mean moves at
constant speed in centerline arc length from the source mean (``t = 0``) to the target mean
(``t = 1``). Each learned field pushes the fitted source cloud from ``0`` to ``t``; the resulting
cloud is compared with a sample of ``p_t`` using the notebook's distribution metrics (squared energy
distance, mean-position error, domain diagnostics), the empirical W2 distance, and point-set versions
of the Hausdorff and Chamfer distances.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist

from riemannian_toys.flows import endpoint_metrics, integrate_learned_paths
from riemannian_toys.plotting import domain_axes
from riemannian_toys.reference_paths import wasserstein2


class TravellingRollGaussian:
    """``p_t``: isotropic Gaussian on the roll whose mean moves linearly in centerline arc length.

    ``mean(t) = c(θ(s_t))`` with ``s_t = (1 - t) s_0 + t s_1``, where ``s_0`` and ``s_1`` are the arc
    positions of the source and target means. As for the end clouds, samples are conditioned on the
    strip and on a roll-angle window of width ``window_width`` centered at the mean (clipped to the
    roll), which keeps the cloud off the neighboring turns.
    """

    def __init__(self, roll, source_arc, target_arc, variance, window_width=np.pi):
        self.roll = roll
        self.source_arc = float(source_arc)
        self.target_arc = float(target_arc)
        self.variance = float(variance)
        self.window_width = float(window_width)

    def arc_position(self, t):
        return (1.0 - t) * self.source_arc + t * self.target_arc

    def theta(self, t):
        return float(self.roll.theta_at_arc_length(self.arc_position(t)))

    def mean(self, t):
        return self.roll.strip_point(self.theta(t))

    def theta_window(self, t):
        theta = self.theta(t)
        half = 0.5 * self.window_width
        return max(theta - half, 0.0), min(theta + half, self.roll.theta_max)

    def sample(self, t, n_samples, rng):
        return self.roll.sample_gaussian(self.mean(t), self.variance, n_samples, rng,
                                         theta_window=self.theta_window(t))[0]


def push_forward(models, starts, time):
    """Learned states at ``time`` from ``starts``, per model (endpoints are checked against ``predict``)."""
    save_times = np.array([0.0, time, 1.0], dtype=np.float32)
    return {name: integrate_learned_paths(model, starts, save_times)[0][:, 1] for name, model in models.items()}


def point_set_distances(samples, reference):
    """Hausdorff and Chamfer distances between two point clouds (nearest-neighbor based)."""
    distances = cdist(samples, reference)
    to_reference, to_samples = distances.min(axis=1), distances.min(axis=0)
    return {
        "hausdorff": float(max(to_reference.max(), to_samples.max())),
        "chamfer": float(0.5 * (to_reference.mean() + to_samples.mean())),
    }


def marginal_metrics(samples, reference, roll):
    """Endpoint metrics of the notebook plus W2, point-set Hausdorff/Chamfer and the mean arc position."""
    samples = np.asarray(samples, dtype=float)
    reference = np.asarray(reference, dtype=float)
    if len(samples) != len(reference):
        raise ValueError("W2 uses an exact assignment; use clouds of equal size.")
    return {
        **endpoint_metrics(samples, reference, roll),
        "wasserstein_2": wasserstein2(samples, reference),
        **point_set_distances(samples, reference),
        "mean_arc_position": float(roll.arc_length(roll.roll_angle(samples)).mean()),
    }


def marginal_table(clouds, reference, roll):
    """:func:`marginal_metrics` for every named cloud against the same reference sample."""
    return pd.DataFrame({name: marginal_metrics(cloud, reference, roll) for name, cloud in clouds.items()}).T


def plot_marginals(roll, family, time, reference, predictions, colors, metrics, source, target):
    """Reference family on the left, then each model's learned state at ``time`` against ``p_time``."""
    panels = len(predictions) + 1
    extra = np.vstack([source, target, reference, *predictions.values()])
    fig, axes = domain_axes(roll, 1, panels, figsize=(4.6 * panels, 5.2), constrained_layout=True)

    ax = axes[0]
    roll.scatter(ax, source, s=10, alpha=0.4, color="tab:blue", label="$p_0$ (source)")
    roll.scatter(ax, reference, s=10, alpha=0.6, color="black", label=f"$p_{{{time:g}}}$")
    roll.scatter(ax, target, s=10, alpha=0.4, color="tab:orange", label="$p_1$ (target)")
    means = np.vstack([family.mean(s) for s in np.linspace(0.0, 1.0, 41)])
    ax.plot(means[:, 0], means[:, 1], color="tab:red", linewidth=1.5, label="mean of $p_t$")
    roll.finish_axes(ax, extra_points=extra)
    ax.set_title(f"Reference family, $p_{{{time:g}}}$ in black")
    ax.legend(loc="upper right", fontsize=8)

    for ax, (name, samples) in zip(axes[1:], predictions.items(), strict=True):
        roll.scatter(ax, reference, s=10, alpha=0.35, color="black", label=f"$p_{{{time:g}}}$")
        roll.scatter(ax, samples, s=10, alpha=0.65, color=colors[name], label=f"learned $x_{{{time:g}}}$")
        roll.finish_axes(ax, extra_points=extra)
        ax.set_title(f"{name}\nenergy² {metrics.loc[name, 'energy_distance_squared']:.3f}, "
                     f"W2 {metrics.loc[name, 'wasserstein_2']:.2f}, "
                     f"outside {metrics.loc[name, f'outside_{roll.name}_fraction']:.0%}")
        ax.legend(loc="upper right", fontsize=8)
    fig.suptitle(f"Learned marginals at t = {time:g} vs. the travelling Gaussian $p_{{{time:g}}}$")
    plt.show()
