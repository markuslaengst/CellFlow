"""The planar annulus ``inner_radius <= |x| <= outer_radius``: sampling, projection, guards, drawing."""

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

from riemannian_toys.geometry import ToyManifold


def straight_segment_min_radius(starts, ends):
    """Smallest distance to the origin along each straight segment (broadcasts)."""
    segments = ends - starts
    fractions = np.clip(
        -np.sum(starts * segments, axis=-1) / np.maximum(np.sum(segments**2, axis=-1), 1e-20),
        0.0,
        1.0,
    )
    return np.linalg.norm(starts + fractions[..., None] * segments, axis=-1)


class Annulus(ToyManifold):
    """Annulus in the plane. Its interior tangent space is the whole plane.

    At the boundary, the tangent-cone guard removes outward velocity and keeps inward and
    tangential components.
    """

    name = "annulus"
    dim = 2
    guarded_mode = "annulus + cone"
    probe_field_label = "boundary-field"
    segment_check_columns = ("max_segment_hole_penetration", "max_domain_violation")
    subplot_kw: dict = {}
    radial_label = "Radius ||x(t)||"

    def __init__(self, inner_radius=2.0, outer_radius=3.0, boundary_tol=1e-4, n_probe_angles=96):
        if not 0.0 < inner_radius < outer_radius:
            raise ValueError("Require 0 < inner_radius < outer_radius.")
        self.inner_radius = float(inner_radius)
        self.outer_radius = float(outer_radius)
        self.boundary_tol = boundary_tol
        self.probe_points, self.probe_normals = self.sample_boundary(n_probe_angles)

    # Sampling -------------------------------------------------------------------------------
    def sample_uniform(self, n_samples, rng):
        """Area-uniform samples: uniform angle, radius with density proportional to r."""
        theta = rng.uniform(0.0, 2.0 * np.pi, n_samples)
        radius = np.sqrt(rng.uniform(self.inner_radius**2, self.outer_radius**2, n_samples))
        return np.column_stack((radius * np.cos(theta), radius * np.sin(theta)))

    def sample_gaussian(self, mean, variance, n_samples, rng, batch_size=10_000):
        """Isotropic Gaussian conditioned on the annulus (rejection sampling).

        Returns ``(samples, n_drawn, n_accepted)``.
        """
        std = np.sqrt(variance)
        accepted, n_drawn, n_accepted = [], 0, 0
        while n_accepted < n_samples:
            candidates = rng.normal(loc=mean, scale=std, size=(batch_size, 2))
            n_drawn += len(candidates)
            kept = candidates[self.contains(candidates, tol=0.0)]
            if len(kept) > 0:
                accepted.append(kept)
                n_accepted += len(kept)
        return np.vstack(accepted)[:n_samples], n_drawn, n_accepted

    def sample_boundary(self, n_angles=96):
        """Points on both boundary circles and their interior unit normals."""
        angles = np.linspace(0.0, 2.0 * np.pi, n_angles, endpoint=False, dtype=np.float32)
        radial = np.column_stack((np.cos(angles), np.sin(angles))).astype(np.float32)
        points = np.vstack([self.inner_radius * radial, self.outer_radius * radial]).astype(np.float32)
        normals = np.vstack([radial, -radial]).astype(np.float32)
        return points, normals

    # Geometry -------------------------------------------------------------------------------
    def project(self, x, eps=1e-12):
        x = jnp.asarray(x)
        radius = jnp.linalg.norm(x, axis=-1, keepdims=True)
        clipped_radius = jnp.clip(radius, self.inner_radius, self.outer_radius)
        fallback_direction = jnp.zeros_like(x).at[..., 0].set(1.0)
        direction = jnp.where(radius > eps, x / jnp.maximum(radius, eps), fallback_direction)
        return direction * clipped_radius

    def interior_normal(self, x, eps=1e-12):
        """Interior unit normal at boundary points (zero elsewhere) and the boundary mask."""
        y = self.project(x, eps=eps)
        radius = jnp.linalg.norm(y, axis=-1, keepdims=True)
        radial = y / jnp.maximum(radius, eps)
        inner_mask = radius <= self.inner_radius + self.boundary_tol
        outer_mask = radius >= self.outer_radius - self.boundary_tol
        normal = jnp.where(inner_mask, radial, jnp.where(outer_mask, -radial, jnp.zeros_like(radial)))
        return normal, inner_mask | outer_mask

    def guard_velocity(self, x, v):
        """Tangent-cone projection: remove the outward normal component at the boundary."""
        v = jnp.asarray(v)
        normal, boundary_mask = self.interior_normal(x)
        normal_component = jnp.sum(v * normal, axis=-1, keepdims=True)
        outward = boundary_mask & (normal_component < 0.0)
        return jnp.where(outward, v - normal_component * normal, v)

    def contains(self, samples, tol=1e-4):
        radii = np.linalg.norm(samples, axis=-1)
        return (radii >= self.inner_radius - tol) & (radii <= self.outer_radius + tol)

    def violation(self, samples):
        radii = np.linalg.norm(samples, axis=-1)
        return np.maximum(self.inner_radius - radii, 0.0) + np.maximum(radii - self.outer_radius, 0.0)

    def radial_coordinate(self, samples):
        return np.linalg.norm(samples, axis=-1)

    def max_penetration(self, trajectories):
        """Deepest saved path state inside the hole."""
        return float(np.maximum(self.inner_radius - self.radial_coordinate(trajectories), 0).max())

    def pair_masks(self, source, target):
        """Same upper/lower half, and straight chord passing through the hole."""
        pair_min_radius = straight_segment_min_radius(source[:, None, :], target[None, :, :])
        return {
            "same_half": np.sign(source[:, None, 1]) == np.sign(target[None, :, 1]),
            "chord_through_hole": pair_min_radius < self.inner_radius,
        }

    def geodesic(self, x0, x1, times):
        """Exact shortest paths in the annulus, traversed at constant speed.

        The outer disk is convex, so only the hole is an obstacle. A pair whose straight chord
        misses the hole is joined by that chord. Otherwise the shortest path consists of a tangent
        segment to the inner circle, an arc along it, and a second tangent segment, wrapped around
        the shorter side of the hole.

        Parameters
        ----------
        x0, x1
            Start and end points of shape ``(n, 2)`` inside the annulus.
        times
            Times in ``[0, 1]`` of shape ``(m,)``; ``t`` maps to arc length ``t * length``.

        Returns
        -------
        ``(paths, lengths)`` with shapes ``(n, m, 2)`` and ``(n,)``.
        """
        x0, x1 = np.asarray(x0, dtype=np.float64), np.asarray(x1, dtype=np.float64)
        times = np.asarray(times, dtype=np.float64)
        r = self.inner_radius
        rho0 = np.maximum(np.linalg.norm(x0, axis=-1), r)
        rho1 = np.maximum(np.linalg.norm(x1, axis=-1), r)
        phi0, phi1 = np.arctan2(x0[:, 1], x0[:, 0]), np.arctan2(x1[:, 1], x1[:, 0])
        alpha0, alpha1 = np.arccos(np.clip(r / rho0, -1, 1)), np.arccos(np.clip(r / rho1, -1, 1))
        tangent0, tangent1 = np.sqrt(rho0**2 - r**2), np.sqrt(rho1**2 - r**2)

        # Counter-clockwise (+1) and clockwise (-1) wraps: leave the circle at phi0 + s*alpha0,
        # follow it for `arc` radians in direction s, rejoin at phi1 - s*alpha1.
        arcs = {s: np.mod(s * ((phi1 - s * alpha1) - (phi0 + s * alpha0)), 2 * np.pi) for s in (1, -1)}
        sign = np.where(arcs[1] <= arcs[-1], 1.0, -1.0)
        arc = np.minimum(arcs[1], arcs[-1])
        wrap_length = tangent0 + r * arc + tangent1
        straight = straight_segment_min_radius(x0, x1) >= r
        lengths = np.where(straight, np.linalg.norm(x1 - x0, axis=-1), wrap_length)

        s = times[None, :] * lengths[:, None]  # arc length along each path, (n, m)
        chord = x0[:, None] + times[None, :, None] * (x1 - x0)[:, None]
        leave = phi0 + sign * alpha0
        leave_point = r * np.stack((np.cos(leave), np.sin(leave)), axis=-1)
        join = leave + sign * arc
        join_point = r * np.stack((np.cos(join), np.sin(join)), axis=-1)
        first = x0[:, None] + (s / np.maximum(tangent0, 1e-12)[:, None])[..., None] * (leave_point - x0)[:, None]
        angle = leave[:, None] + sign[:, None] * (s - tangent0[:, None]) / r
        on_arc = r * np.stack((np.cos(angle), np.sin(angle)), axis=-1)
        last_fraction = (s - tangent0[:, None] - r * arc[:, None]) / np.maximum(tangent1, 1e-12)[:, None]
        last = join_point[:, None] + np.clip(last_fraction, 0, 1)[..., None] * (x1 - join_point)[:, None]
        wrapped = np.where(
            (s <= tangent0[:, None])[..., None], first,
            np.where((s <= (tangent0 + r * arc)[:, None])[..., None], on_arc, last),
        )
        return np.where(straight[:, None, None], chord, wrapped), lengths

    def winding_angle(self, paths):
        """Total change of the polar angle along paths ``(..., m, 2)``: its sign tells the side of the hole."""
        angles = np.unwrap(np.arctan2(paths[..., 1], paths[..., 0]), axis=-1)
        return angles[..., -1] - angles[..., 0]

    def path_geometry(self, path):
        segments = np.diff(path, axis=0)
        closest_radii = straight_segment_min_radius(path[:-1], path[1:])
        return {
            "max_step_length": np.sqrt(np.sum(segments**2, axis=-1).max()),
            "max_segment_hole_penetration": np.maximum(self.inner_radius - closest_radii, 0.0).max(),
            "hole_crossing_segment_fraction": np.mean(closest_radii < self.inner_radius - 1e-3),
            "max_domain_violation": self.violation(path).max(),
        }

    def probe_diagnostic(self, velocities):
        return {
            "outward_boundary_fraction": np.mean(np.sum(velocities * self.probe_normals, axis=1) < -1e-6),
        }

    def polar_grid(self, n_angles=30, n_radii=6, margin=0.07):
        """Polar grid strictly inside the annulus, for raw kernel fields."""
        angles, radii = np.meshgrid(
            np.linspace(0.0, 2.0 * np.pi, n_angles, endpoint=False),
            np.linspace(self.inner_radius + margin, self.outer_radius - margin, n_radii),
        )
        return np.column_stack(((radii * np.cos(angles)).ravel(), (radii * np.sin(angles)).ravel())).astype(np.float32)

    def learned_field_grid(self, grid_size=27):
        """Square grid restricted to the annulus."""
        axis_values = np.linspace(-self.outer_radius, self.outer_radius, grid_size, dtype=np.float32)
        grid_x, grid_y = np.meshgrid(axis_values, axis_values)
        grid = np.column_stack((grid_x.ravel(), grid_y.ravel()))
        return grid[self.contains(grid, tol=0.0)]

    def path_metrics(self, trajectories):
        """Domain and hole diagnostics of saved learned paths ``(n, T, 2)``.

        Segment penetration uses a closest-point test on each straight segment between saved
        times, so it also detects chords that cut through the hole between two saved times.
        """
        radii = np.linalg.norm(trajectories, axis=-1)
        inside = self.contains(trajectories)
        steps = np.linalg.norm(np.diff(trajectories, axis=1), axis=-1)
        min_radii = straight_segment_min_radius(trajectories[:, :-1], trajectories[:, 1:])
        return {
            "paths_leaving_annulus": int(np.count_nonzero(~inside.all(axis=1))),
            "endpoints_in_annulus": float(np.mean(inside[:, -1])),
            "fraction_saved_points_in_annulus": float(np.mean(inside)),
            "min_sampled_radius": float(radii.min()),
            "max_sampled_radius": float(radii.max()),
            "mean_path_length": float(steps.sum(axis=1).mean()),
            "max_step_length": float(steps.max()),
            "max_segment_hole_penetration": float(np.maximum(self.inner_radius - min_radii, 0.0).max()),
            "hole_crossing_segment_fraction": float(np.mean(min_radii < self.inner_radius - 1e-3)),
            "max_annulus_violation": float(self.violation(trajectories).max()),
        }

    def endpoint_metrics(self, samples):
        """Annulus-specific endpoint metrics added to the distribution metrics."""
        radii = np.linalg.norm(samples, axis=-1)
        violation = self.violation(samples)
        return {
            "inside_hole_fraction": float(np.mean(radii < self.inner_radius - 1e-4)),
            "beyond_outer_fraction": float(np.mean(radii > self.outer_radius + 1e-4)),
            "mean_annulus_violation": float(violation.mean()),
            "p95_annulus_violation": float(np.quantile(violation, 0.95)),
        }

    # Drawing --------------------------------------------------------------------------------
    def draw_boundaries(self, ax, outer_color="black", label=False):
        """Dashed inner circle and solid outer circle."""
        style = {"linewidth": 2} if label else {}
        ax.add_patch(plt.Circle(
            (0, 0), self.inner_radius, fill=False, linestyle="--", color="tab:red",
            label=f"inner radius = {self.inner_radius:g}" if label else None, **style,
        ))
        ax.add_patch(plt.Circle(
            (0, 0), self.outer_radius, fill=False, edgecolor=outer_color,
            label=f"outer radius = {self.outer_radius:g}" if label else None, **style,
        ))

    def finish_axes(self, ax, extra_points=None, extent=None, outer_color="black", label_boundaries=False,
                    grid_alpha=0.2):
        self.draw_boundaries(ax, outer_color=outer_color, label=label_boundaries)
        if extent is None:
            extent = self.outer_radius + 0.3
            if extra_points is not None and len(extra_points):
                extent = max(extent, float(np.max(np.abs(extra_points))) + 0.2)
        ax.set(aspect="equal", xlim=(-extent, extent), ylim=(-extent, extent), xlabel="x", ylabel="y")
        ax.grid(alpha=grid_alpha)

    def draw_arrows(self, ax, points, vectors, values, norm, cmap, alpha=1.0, width=0.003):
        return ax.quiver(
            points[:, 0], points[:, 1], vectors[:, 0], vectors[:, 1], values,
            cmap=cmap, clim=(norm.vmin, norm.vmax), angles="xy", scale_units="xy", scale=1,
            width=width, alpha=alpha,
        )

    def draw_highlight_arrow(self, ax, point, vector, color="tab:red"):
        return ax.quiver(
            point[0], point[1], vector[0], vector[1], angles="xy", scale_units="xy", scale=1,
            color=color, width=0.008, zorder=6,
        )

    def draw_direction_arrows(self, ax, path, color, linewidth=1.8, n_arrows=4):
        for index in self.direction_arrow_indices(len(path), n_arrows):
            ax.annotate("", xy=path[index + 4], xytext=path[index],
                        arrowprops={"arrowstyle": "->", "color": color, "lw": linewidth})

    def draw_radial_reference(self, ax):
        ax.axhspan(self.inner_radius, self.outer_radius, color="tab:green", alpha=0.1, label="Annulus")
        ax.axhline(self.inner_radius, color="tab:red", linestyle="--")
        ax.axhline(self.outer_radius, color="gray", linestyle="--")
