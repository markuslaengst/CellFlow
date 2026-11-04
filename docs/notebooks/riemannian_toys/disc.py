"""The planar disc ``|x| <= radius``: a convex domain without any geometric obstacle.

Every geodesic is the straight chord, so flow matching with straight-line paths is already exact.
The disc only bounds the region on which the graph and its Nyström extension are built.
"""

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

from riemannian_toys.geometry import ToyManifold


class Disc(ToyManifold):
    """Disc in the plane. Its interior tangent space is the whole plane.

    At the boundary circle, the tangent-cone guard removes outward velocity and keeps inward and
    tangential components.
    """

    name = "disc"
    dim = 2
    guarded_mode = "disc + cone"
    probe_field_label = "boundary-field"
    segment_check_columns = ("max_domain_violation",)
    subplot_kw: dict = {}
    radial_label = "Radius ||x(t)||"
    penetration_name = "max_disc_violation"

    def __init__(self, radius=4.0, boundary_tol=1e-4, n_probe_angles=96):
        if radius <= 0.0:
            raise ValueError("Require radius > 0.")
        self.radius = float(radius)
        self.boundary_tol = boundary_tol
        self.probe_points, self.probe_normals = self.sample_boundary(n_probe_angles)

    # Sampling -------------------------------------------------------------------------------
    def sample_uniform(self, n_samples, rng):
        """Area-uniform samples: uniform angle, radius with density proportional to r."""
        theta = rng.uniform(0.0, 2.0 * np.pi, n_samples)
        radius = self.radius * np.sqrt(rng.uniform(0.0, 1.0, n_samples))
        return np.column_stack((radius * np.cos(theta), radius * np.sin(theta)))

    def sample_gaussian(self, mean, variance, n_samples, rng, batch_size=10_000):
        """Isotropic Gaussian conditioned on the disc (rejection sampling).

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
        """Points on the boundary circle and their interior unit normals."""
        angles = np.linspace(0.0, 2.0 * np.pi, n_angles, endpoint=False, dtype=np.float32)
        radial = np.column_stack((np.cos(angles), np.sin(angles))).astype(np.float32)
        return (self.radius * radial).astype(np.float32), -radial

    # Geometry -------------------------------------------------------------------------------
    def project(self, x, eps=1e-12):
        """Radial projection onto the disc; points inside are unchanged."""
        x = jnp.asarray(x)
        radius = jnp.linalg.norm(x, axis=-1, keepdims=True)
        return x * jnp.minimum(1.0, self.radius / jnp.maximum(radius, eps))

    def guard_velocity(self, x, v, eps=1e-12):
        """Tangent-cone projection: remove the outward radial component on the boundary circle."""
        v = jnp.asarray(v)
        y = self.project(x)
        radius = jnp.linalg.norm(y, axis=-1, keepdims=True)
        outward_normal = y / jnp.maximum(radius, eps)
        outward_component = jnp.sum(v * outward_normal, axis=-1, keepdims=True)
        on_boundary = radius >= self.radius - self.boundary_tol
        return jnp.where(on_boundary & (outward_component > 0.0), v - outward_component * outward_normal, v)

    def contains(self, samples, tol=1e-4):
        return np.linalg.norm(samples, axis=-1) <= self.radius + tol

    def violation(self, samples):
        return np.maximum(np.linalg.norm(samples, axis=-1) - self.radius, 0.0)

    def radial_coordinate(self, samples):
        return np.linalg.norm(samples, axis=-1)

    def max_penetration(self, trajectories):
        """Largest distance of a saved path state outside the disc."""
        return float(self.violation(trajectories).max())

    def pair_masks(self, source, target):
        """Same upper/lower half of the plane."""
        return {"same_half": np.sign(source[:, None, 1]) == np.sign(target[None, :, 1])}

    def geodesic(self, x0, x1, times):
        """Shortest paths in the convex disc: straight chords at constant speed.

        Returns ``(paths, lengths)`` with shapes ``(n, m, 2)`` and ``(n,)``.
        """
        x0, x1 = np.asarray(x0, dtype=np.float64), np.asarray(x1, dtype=np.float64)
        times = np.asarray(times, dtype=np.float64)
        paths = x0[:, None] + times[None, :, None] * (x1 - x0)[:, None]
        return paths, np.linalg.norm(x1 - x0, axis=-1)

    def path_geometry(self, path):
        segments = np.diff(path, axis=0)
        return {
            "max_step_length": np.sqrt(np.sum(segments**2, axis=-1).max()),
            "max_domain_violation": self.violation(path).max(),
        }

    def probe_diagnostic(self, velocities):
        return {
            "outward_boundary_fraction": np.mean(np.sum(velocities * self.probe_normals, axis=1) < -1e-6),
        }

    def learned_field_grid(self, grid_size=27):
        """Square grid restricted to the disc."""
        axis_values = np.linspace(-self.radius, self.radius, grid_size, dtype=np.float32)
        grid_x, grid_y = np.meshgrid(axis_values, axis_values)
        grid = np.column_stack((grid_x.ravel(), grid_y.ravel()))
        return grid[self.contains(grid, tol=0.0)]

    def path_metrics(self, trajectories):
        """Domain diagnostics and straightness of saved learned paths ``(n, T, 2)``.

        ``mean_straightness`` is the chord length divided by the path length (1 for a straight path).
        """
        inside = self.contains(trajectories)
        steps = np.linalg.norm(np.diff(trajectories, axis=1), axis=-1)
        lengths = steps.sum(axis=1)
        chords = np.linalg.norm(trajectories[:, -1] - trajectories[:, 0], axis=-1)
        return {
            "paths_leaving_disc": int(np.count_nonzero(~inside.all(axis=1))),
            "max_sampled_radius": float(np.linalg.norm(trajectories, axis=-1).max()),
            "mean_path_length": float(lengths.mean()),
            "mean_straightness": float(np.mean(chords / np.maximum(lengths, 1e-12))),
            "max_step_length": float(steps.max()),
            "max_disc_violation": float(self.violation(trajectories).max()),
        }

    def endpoint_metrics(self, samples):
        violation = self.violation(samples)
        return {
            "mean_disc_violation": float(violation.mean()),
            "max_disc_violation": float(violation.max()),
        }

    # Drawing --------------------------------------------------------------------------------
    def finish_axes(self, ax, extra_points=None, extent=None, outer_color="black", label_boundaries=False,
                    grid_alpha=0.2):
        ax.add_patch(plt.Circle(
            (0, 0), self.radius, fill=False, edgecolor=outer_color,
            label=f"disc radius = {self.radius:g}" if label_boundaries else None,
            **({"linewidth": 2} if label_boundaries else {}),
        ))
        if extent is None:
            extent = self.radius + 0.3
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
        ax.axhspan(0.0, self.radius, color="tab:green", alpha=0.1, label="Disc")
        ax.axhline(self.radius, color="gray", linestyle="--")
