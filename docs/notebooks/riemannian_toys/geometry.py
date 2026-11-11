"""Common interface of the toy domains used by the shared experiment and plotting helpers."""

import numpy as np

from cellflow.ode_solver import GuardedRFMInterpolation


class ToyManifold:
    """A toy domain in its ambient space.

    Subclasses provide the domain-specific pieces: point projection, the velocity guard
    used by guarded RFM trajectories, domain diagnostics, and drawing. Everything else
    (kernel comparison, training, evaluation, figures) is written against this interface.

    Attributes
    ----------
    name
        Short domain name used in column names, e.g. ``"outside_annulus_fraction"``.
    dim
        Ambient dimension.
    guarded_mode
        Label of the guarded trajectory mode in the kernel comparison.
    probe_field_label
        Label of the raw-field probe diagnostic, used in printed summaries.
    segment_check_columns
        Path-geometry columns that must stay below the segment tolerance for a guarded path to pass.
    subplot_kw
        Matplotlib ``subplot_kw`` for axes that show the domain.
    """

    name = "domain"
    dim = 2
    guarded_mode = "guarded"
    probe_field_label = "probe-field"
    segment_check_columns = ("max_domain_violation",)
    subplot_kw: dict = {}
    radial_label = "radius"
    radial_column = "radius"
    """Column suffix for :meth:`radial_coordinate` in per-path summaries."""
    penetration_name = "max_hole_penetration"
    """Name of :meth:`max_penetration` in the multi-seed loss-mode results."""

    # Geometry -------------------------------------------------------------------------------
    def project(self, x):
        """Project points onto the domain (JAX)."""
        raise NotImplementedError

    def guard_velocity(self, x, v):
        """Filter a velocity at an on-domain point ``x`` (JAX)."""
        raise NotImplementedError

    def guarded_interpolator(self, nystroem, **kwargs) -> GuardedRFMInterpolation:
        """RFM interpolation that projects queries and Euler states and guards the velocity."""
        return GuardedRFMInterpolation(
            nystroem,
            projection_fn=self.project,
            velocity_guard_fn=self.guard_velocity,
            project_to_manifold=False,
            **kwargs,
        )

    def contains(self, samples, tol=1e-4):
        """Boolean mask of samples within ``tol`` of the domain."""
        raise NotImplementedError

    def violation(self, samples):
        """Non-negative distance-like violation of the domain constraint, per sample."""
        raise NotImplementedError

    def radial_coordinate(self, samples):
        """Scalar coordinate plotted against time to show departures from the domain."""
        raise NotImplementedError

    def max_penetration(self, trajectories) -> float:
        """Worst shortcut depth of saved path states, used by the multi-seed loss-mode comparison."""
        raise NotImplementedError

    def loss_mode_domain_metrics(self) -> list:
        """Domain columns reported by the multi-seed loss-mode comparison (lower is better)."""
        return [f"outside_{self.name}_fraction", f"fresh_paths_leaving_{self.name}", self.penetration_name]

    def pair_masks(self, source, target) -> dict:
        """Boolean ``(n_source, n_target)`` pair diagnostics for the coupling comparison table."""
        raise NotImplementedError

    def path_geometry(self, path) -> dict:
        """Segment-level diagnostics for Euler paths of shape ``(n_times, n_paths, dim)``."""
        raise NotImplementedError

    def path_metrics(self, trajectories) -> dict:
        """Domain diagnostics of saved learned paths of shape ``(n_paths, n_times, dim)``."""
        raise NotImplementedError

    def endpoint_metrics(self, samples) -> dict:
        """Domain diagnostics of predicted endpoints, added to the distribution metrics."""
        raise NotImplementedError

    def probe_diagnostic(self, velocities) -> dict:
        """Diagnostic of the raw field at ``self.probe_points``, as ``{column: value}``."""
        raise NotImplementedError

    def learned_field_grid(self, grid_size=27):
        """Query grid on the domain for learned-field plots."""
        raise NotImplementedError

    # Drawing --------------------------------------------------------------------------------
    @staticmethod
    def scatter(ax, points, **kwargs):
        return ax.scatter(*np.asarray(points).T, **kwargs)

    @staticmethod
    def plot_path(ax, path, **kwargs):
        return ax.plot(*np.asarray(path).T, **kwargs)

    def finish_axes(self, ax, extra_points=None, **kwargs):
        """Draw the domain outline and set limits, labels and grid."""
        raise NotImplementedError

    def draw_arrows(self, ax, points, vectors, values, norm, cmap, alpha=1.0, width=0.003):
        """Draw arrows of the given ambient length, colored by ``values``."""
        raise NotImplementedError

    def draw_highlight_arrow(self, ax, point, vector, color="tab:red"):
        """Draw one emphasized arrow of the given ambient length."""
        raise NotImplementedError

    def draw_direction_arrows(self, ax, path, color, linewidth=1.8, n_arrows=4):
        """Small arrows along a saved path that show the travel direction."""
        raise NotImplementedError

    def draw_radial_reference(self, ax):
        """Reference band or line for :meth:`radial_coordinate` plots."""
        raise NotImplementedError

    @staticmethod
    def direction_arrow_indices(n_times, n_arrows=4):
        return np.linspace(25, n_times - 15, n_arrows, dtype=int)
