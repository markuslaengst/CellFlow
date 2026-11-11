"""The torus surface in R³: parametrization, sampling, projection, tangent guard, drawing.

``T(u, v) = ((R + r cos v) cos u, (R + r cos v) sin u, r sin v)`` with major radius ``R``
and minor (tube) radius ``r``.
"""

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

from riemannian_toys.geometry import ToyManifold


class Torus(ToyManifold):
    """Torus surface. The guard projects velocities onto the tangent plane."""

    name = "torus"
    dim = 3
    guarded_mode = "torus + tangent"
    probe_field_label = "normal-field"
    segment_check_columns = (
        "max_segment_hole_penetration",
        "max_domain_violation",
        "max_sampled_segment_surface_deviation",
    )
    subplot_kw = {"projection": "3d"}
    radial_label = "Distance to the major circle"
    radial_column = "tube_radius"
    penetration_name = "max_surface_deviation"

    def __init__(self, major_radius=2.0, minor_radius=0.75, n_probe_major=24, n_probe_minor=8):
        if not major_radius > minor_radius > 0.0:
            raise ValueError("Require major_radius > minor_radius > 0.")
        self.major_radius = float(major_radius)
        self.minor_radius = float(minor_radius)
        self.probe_points, self.probe_normals = self.sample_probes(n_probe_major, n_probe_minor)
        plot_u, plot_v = np.meshgrid(np.linspace(0.0, 2.0 * np.pi, 65), np.linspace(0.0, 2.0 * np.pi, 33))
        self._plot_surface = self.angles_to_points(np.stack((plot_u, plot_v), axis=-1))

    # Parametrization and sampling -----------------------------------------------------------
    def angles_to_points(self, angles):
        angles = np.asarray(angles)
        u, v = angles[..., 0], angles[..., 1]
        cylindrical_radius = self.major_radius + self.minor_radius * np.cos(v)
        return np.stack(
            (cylindrical_radius * np.cos(u), cylindrical_radius * np.sin(u), self.minor_radius * np.sin(v)), axis=-1
        )

    def tube_radius(self, samples):
        """Distance to the major circle; equals ``minor_radius`` on the surface."""
        samples = np.asarray(samples)
        cylindrical_radius = np.linalg.norm(samples[..., :2], axis=-1)
        return np.sqrt((cylindrical_radius - self.major_radius) ** 2 + samples[..., 2] ** 2)

    def sample_uniform(self, n_samples, rng, batch_size=10_000):
        """Area-uniform samples. ``dA = r (R + r cos v) du dv``, so uniform angles need rejection.

        Returns ``(points, angles)``.
        """
        accepted, n_accepted = [], 0
        while n_accepted < n_samples:
            candidates = rng.uniform(0.0, 2.0 * np.pi, size=(batch_size, 2))
            acceptance = (self.major_radius + self.minor_radius * np.cos(candidates[:, 1])) / (
                self.major_radius + self.minor_radius
            )
            kept = candidates[rng.uniform(size=batch_size) < acceptance]
            accepted.append(kept)
            n_accepted += len(kept)
        angles = np.vstack(accepted)[:n_samples]
        return self.angles_to_points(angles), angles

    def sample_angular_gaussian(self, mean_angles, angular_std, n_samples, rng, half_width, batch_size=10_000):
        """Angular Gaussian truncated to ``|angle - mean| <= half_width`` per coordinate.

        Returns ``(points, angles)``.
        """
        mean_angles = np.asarray(mean_angles)
        accepted, n_accepted = [], 0
        while n_accepted < n_samples:
            candidates = rng.normal(loc=mean_angles, scale=angular_std, size=(batch_size, 2))
            kept = candidates[np.all(np.abs(candidates - mean_angles) <= half_width, axis=1)]
            accepted.append(kept)
            n_accepted += len(kept)
        angles = np.vstack(accepted)[:n_samples]
        return self.angles_to_points(angles), angles

    def sample_probes(self, n_major=24, n_minor=8):
        """Angle-grid probe points and their outward unit normals."""
        u, v = np.meshgrid(
            np.linspace(0.0, 2.0 * np.pi, n_major, endpoint=False),
            np.linspace(0.0, 2.0 * np.pi, n_minor, endpoint=False),
        )
        u, v = u.ravel(), v.ravel()
        points = self.angles_to_points(np.column_stack((u, v))).astype(np.float32)
        normals = np.column_stack((np.cos(v) * np.cos(u), np.cos(v) * np.sin(u), np.sin(v))).astype(np.float32)
        return points, normals

    def angle_grid(self, n_major, n_minor):
        """Points on an angle grid, without the closing endpoint."""
        u, v = np.meshgrid(
            np.linspace(0.0, 2.0 * np.pi, n_major, endpoint=False),
            np.linspace(0.0, 2.0 * np.pi, n_minor, endpoint=False),
        )
        return self.angles_to_points(np.column_stack((u.ravel(), v.ravel()))).astype(np.float32)

    # Geometry -------------------------------------------------------------------------------
    def project(self, x, eps=1e-12):
        """Closest surface point, with deterministic fallbacks on the axis and the major circle."""
        x = jnp.asarray(x)
        cylindrical_radius = jnp.linalg.norm(x[..., :2], axis=-1, keepdims=True)
        fallback_radial = jnp.zeros_like(x[..., :2]).at[..., 0].set(1.0)
        radial = jnp.where(
            cylindrical_radius > eps, x[..., :2] / jnp.maximum(cylindrical_radius, eps), fallback_radial
        )
        core = jnp.concatenate((self.major_radius * radial, jnp.zeros_like(x[..., 2:3])), axis=-1)
        offset = x - core
        offset_norm = jnp.linalg.norm(offset, axis=-1, keepdims=True)
        fallback_normal = jnp.concatenate((radial, jnp.zeros_like(x[..., 2:3])), axis=-1)
        normal = jnp.where(offset_norm > eps, offset / jnp.maximum(offset_norm, eps), fallback_normal)
        return core + self.minor_radius * normal

    def surface_normal(self, x):
        y = self.project(x)
        cylindrical_radius = jnp.linalg.norm(y[..., :2], axis=-1, keepdims=True)
        radial = y[..., :2] / cylindrical_radius
        core = jnp.concatenate((self.major_radius * radial, jnp.zeros_like(y[..., 2:3])), axis=-1)
        return (y - core) / self.minor_radius

    def guard_velocity(self, x, v):
        """Tangent-plane projection ``v - (v·n) n``."""
        v = jnp.asarray(v)
        normal = self.surface_normal(x)
        return v - jnp.sum(v * normal, axis=-1, keepdims=True) * normal

    def contains(self, samples, tol=1e-4):
        return np.abs(self.tube_radius(samples) - self.minor_radius) <= tol

    def violation(self, samples):
        return np.abs(self.tube_radius(samples) - self.minor_radius)

    def radial_coordinate(self, samples):
        return self.tube_radius(samples)

    def max_penetration(self, trajectories):
        """Largest surface deviation over saved path states (a shortcut through the tube or the hole)."""
        return float(self.violation(trajectories).max())

    def loss_mode_domain_metrics(self):
        """Unconstrained predictions are never exactly on the surface, so the binary inside/outside
        counts are uninformative; report continuous surface deviations instead."""
        return ["mean_surface_deviation", "fresh_paths_mean_max_violation", self.penetration_name]

    def path_geometry(self, path):
        segment_starts = path[:-1]
        segments = np.diff(path, axis=0)
        # Exact minimum cylindrical radius on each straight Euler segment.
        planar_segments = segments[..., :2]
        closest_fractions = np.clip(
            -np.sum(segment_starts[..., :2] * planar_segments, axis=-1)
            / np.maximum(np.sum(planar_segments**2, axis=-1), 1e-20),
            0.0,
            1.0,
        )
        closest_radii = np.linalg.norm(segment_starts[..., :2] + closest_fractions[..., None] * planar_segments, axis=-1)
        # Also inspect the tube surface between projected endpoints.
        fractions = np.linspace(0.0, 1.0, 17)
        segment_probes = segment_starts[..., None, :] + fractions[None, None, :, None] * segments[..., None, :]
        hole_radius = self.major_radius - self.minor_radius
        return {
            "max_step_length": np.sqrt(np.sum(segments**2, axis=-1).max()),
            "max_segment_hole_penetration": np.maximum(hole_radius - closest_radii, 0.0).max(),
            "hole_crossing_segment_fraction": np.mean(closest_radii < hole_radius - 1e-3),
            "max_domain_violation": self.violation(path).max(),
            "max_sampled_segment_surface_deviation": self.violation(segment_probes).max(),
        }

    def probe_diagnostic(self, velocities):
        return {
            "normal_velocity_fraction": np.mean(np.abs(np.sum(velocities * self.probe_normals, axis=1)) > 1e-6),
        }

    def learned_field_grid(self, grid_size=27):
        return self.angle_grid(grid_size, max(3, grid_size // 3))

    def endpoint_metrics(self, samples):
        """Torus-specific endpoint metrics added to the distribution metrics."""
        deviation = self.violation(samples)
        return {
            "mean_surface_deviation": float(deviation.mean()),
            "p95_surface_deviation": float(np.quantile(deviation, 0.95)),
        }

    def path_metrics(self, trajectories):
        """Surface deviation of saved path states, for paths ``(n, T, 3)``."""
        deviation = self.violation(trajectories)
        inside = self.contains(trajectories)
        return {
            "mean_max_surface_deviation": deviation.max(axis=1).mean(),
            "max_surface_deviation": deviation.max(),
            "mean_endpoint_surface_deviation": deviation[:, -1].mean(),
            "paths_leaving_torus": int(np.count_nonzero(~inside.all(axis=1))),
            "fraction_saved_points_in_torus": np.mean(inside),
        }

    # Drawing --------------------------------------------------------------------------------
    def finish_axes(self, ax, extra_points=None, **_):
        """Translucent surface, equal-scale box, fixed view."""
        ax.plot_surface(*self._plot_surface.transpose(2, 0, 1), color="lightgray", alpha=0.12, linewidth=0)
        outer = self.major_radius + self.minor_radius + 0.2
        limits = np.array([outer, outer, self.minor_radius + 0.2])
        if extra_points is not None and len(extra_points):
            limits = np.maximum(limits, np.max(np.abs(np.asarray(extra_points).reshape(-1, 3)), axis=0) + 0.2)
        ax.set(
            xlim=(-limits[0], limits[0]), ylim=(-limits[1], limits[1]), zlim=(-limits[2], limits[2]),
            xlabel="x", ylabel="y", zlabel="z",
        )
        ax.set_box_aspect(2 * limits)
        ax.view_init(elev=28, azim=35)
        ax.grid(alpha=0.2)

    def draw_arrows(self, ax, points, vectors, values, norm, cmap, alpha=1.0, width=None):
        return ax.quiver(
            *points.T, *vectors.T, colors=plt.get_cmap(cmap)(norm(values)),
            length=1.0, normalize=False, arrow_length_ratio=0.3, alpha=alpha,
        )

    def draw_highlight_arrow(self, ax, point, vector, color="tab:red"):
        return ax.quiver(*point, *vector, length=1.0, normalize=False, color=color, linewidth=2,
                         arrow_length_ratio=0.3)

    def draw_direction_arrows(self, ax, path, color, linewidth=1.8, n_arrows=4):
        for index in self.direction_arrow_indices(len(path), n_arrows):
            ax.quiver(*path[index], *(path[index + 4] - path[index]), length=1.0, normalize=False,
                      color=color, linewidth=linewidth, arrow_length_ratio=0.4)

    def draw_radial_reference(self, ax):
        ax.axhline(self.minor_radius, color="gray", linestyle="--", label="Torus surface")


def certify_unique_geodesics(torus, source_angle_center, target_angle_center, angle_half_width,
                             comparison_midpoint_offset=0.3, n_intervals=64):
    """Check that every pair of the two truncated patches has a unique shortest surface geodesic.

    Implements the comparison-path argument of the torus notebook: an upper bound on the
    shortest-path length from auxiliary curves, compared with the injectivity-radius lower
    bound of the universal cover and the lengths of winding competitors.

    Returns
    -------
    dict with the bounds; raises :class:`AssertionError` if any check fails.
    """
    R, r = torus.major_radius, torus.minor_radius
    source_angle_center = np.asarray(source_angle_center, dtype=float)
    target_angle_center = np.asarray(target_angle_center, dtype=float)
    maximum_angle_difference = np.abs(target_angle_center - source_angle_center) + 2.0 * angle_half_width
    assert np.all(maximum_angle_difference < np.pi)
    max_curvature = 1.0 / (r * (R + r))
    injectivity_lower = np.pi / np.sqrt(max_curvature)

    source_v_min, source_v_max = source_angle_center[1] + np.array([-1, 1]) * angle_half_width
    target_v_min, target_v_max = target_angle_center[1] + np.array([-1, 1]) * angle_half_width
    assert target_v_min <= 0.0 < target_v_max < source_v_min < source_v_max < np.pi
    assert 2.0 * comparison_midpoint_offset < source_v_min - target_v_max

    # Each interval bounds all source tube angles it contains, and all target tube angles.
    source_v_edges = np.linspace(source_v_min, source_v_max, n_intervals + 1)
    source_v_lower, source_v_upper = source_v_edges[:-1], source_v_edges[1:]
    midpoints = (source_v_lower + target_v_min) / 2.0 + comparison_midpoint_offset
    assert np.all((midpoints > 0.0) & (midpoints < source_v_lower))

    def mean_squared_parallel_radius(v0, v1):
        difference = v0 - v1
        return (
            R**2
            + 2.0 * R * r * (np.sin(v0) - np.sin(v1)) / difference
            + r**2 * (0.5 + (np.sin(2.0 * v0) - np.sin(2.0 * v1)) / (4.0 * difference))
        )

    # Clamp the lower comparison curve at v=0 for a pointwise upper bound on (R + r cos v)^2,
    # then integrate that bound exactly on both linear pieces.
    positive_fraction = midpoints / (midpoints - target_v_min)
    second_half_bound = (
        positive_fraction * mean_squared_parallel_radius(midpoints, 0.0) + (1.0 - positive_fraction) * (R + r) ** 2
    )
    mean_radius_sq_upper = 0.5 * (mean_squared_parallel_radius(source_v_lower, midpoints) + second_half_bound)
    tube_difference_upper = source_v_upper - target_v_min
    energy_upper = maximum_angle_difference[0] ** 2 * mean_radius_sq_upper + r**2 * (
        tube_difference_upper**2 + 4.0 * comparison_midpoint_offset**2
    )
    path_length_upper = np.sqrt(energy_upper.max())
    azimuth_winding_lower = (R - r) * (2.0 * np.pi - maximum_angle_difference[0])
    # Joint angular lower bounds exclude a competing tube winding.
    tube_winding_margins = r**2 * (
        (2.0 * np.pi - tube_difference_upper) ** 2 - tube_difference_upper**2 - 4.0 * comparison_midpoint_offset**2
    ) - (mean_radius_sq_upper - (R - r) ** 2) * maximum_angle_difference[0] ** 2
    assert path_length_upper < injectivity_lower
    assert path_length_upper < azimuth_winding_lower
    assert tube_winding_margins.min() > 0.0
    return {
        "surface_path_length_upper_bound": path_length_upper,
        "cover_injectivity_radius_lower_bound": injectivity_lower,
        "azimuth_winding_length_lower_bound": azimuth_winding_lower,
        "tube_winding_squared_margin_lower_bound": tube_winding_margins.min(),
    }


def describe_patch_placement(torus, source_angle_center, target_angle_center, bounds):
    """Print the geodesic-uniqueness bounds and where the two patch centers lie."""
    source_center = torus.angles_to_points(source_angle_center)
    target_center = torus.angles_to_points(target_angle_center)
    print(f"All support pairs: distance <= {bounds['surface_path_length_upper_bound']:.4f}"
          f" < cover injectivity bound {bounds['cover_injectivity_radius_lower_bound']:.4f}"
          f" and azimuth-winding bound {bounds['azimuth_winding_length_lower_bound']:.4f}")
    print(f"Tube-winding squared-length margin >= {bounds['tube_winding_squared_margin_lower_bound']:.4f}")
    print("Every source–target pair has a unique shortest surface geodesic.")
    major_gap = abs(target_angle_center[0] - source_angle_center[0])
    minor_gap = abs(target_angle_center[1] - source_angle_center[1])
    chord = np.linspace(source_center, target_center, 101)
    print(f"Source azimuth: {np.degrees(source_angle_center[0]):.2f} degrees;"
          f" target azimuth: {np.degrees(target_angle_center[0]):.2f} degrees;"
          f" separation: {np.degrees(major_gap):.2f} degrees")
    print(f"Source tube angle: {np.degrees(source_angle_center[1]):.2f} degrees (top);"
          f" target tube angle: {np.degrees(target_angle_center[1]):.2f} degrees (front)")
    print(f"Meridian component of the center angle change: {torus.minor_radius * minor_gap:.4f}")
    print(f"Straight center chord departs from the surface by {np.max(torus.violation(chord)):.4f}.")
