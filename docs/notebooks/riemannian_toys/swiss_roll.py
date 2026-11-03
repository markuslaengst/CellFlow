"""A planar swiss roll: a strip of constant width wound along an Archimedean spiral.

The centerline is ``c(θ) = r(θ) (cos θ, sin θ)`` with ``r(θ) = inner_radius + pitch θ / (2π)``
for ``θ ∈ [0, theta_max]``. The strip is ``{ρ e(θ) : |ρ - r(θ)| <= half_width}``, i.e. a constant
radial half-width around the centerline. ``pitch > 2 half_width`` keeps neighboring turns apart, so
every point of the plane has at most one turn containing it. ``θ = 0`` is the inner end and
``θ = theta_max`` the outer end.
"""

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

from riemannian_toys.annulus import Annulus
from riemannian_toys.geometry import ToyManifold


class SwissRoll(ToyManifold):
    """Swiss-roll strip in the plane. Its interior tangent space is the whole plane.

    At the boundary (both long edges and both end caps), the tangent-cone guard removes
    outward velocity and keeps inward and tangential components, as on the annulus.
    """

    name = "roll"
    dim = 2
    guarded_mode = "roll + cone"
    probe_field_label = "boundary-field"
    segment_check_columns = ("max_segment_gap_penetration", "max_domain_violation")
    subplot_kw: dict = {}
    radial_label = "Offset from the roll centerline"
    radial_column = "centerline_offset"
    penetration_name = "max_gap_penetration"

    def __init__(self, inner_radius=1.0, pitch=1.2, half_width=0.3, turns=1.5, boundary_tol=1e-4,
                 n_probe_points=96):
        if not (inner_radius > half_width > 0.0 and pitch > 2.0 * half_width and turns > 0.0):
            raise ValueError("Require inner_radius > half_width > 0, pitch > 2 half_width and turns > 0.")
        self.inner_radius = float(inner_radius)
        self.pitch = float(pitch)
        self.half_width = float(half_width)
        self.theta_max = float(2.0 * np.pi * turns)
        self.boundary_tol = boundary_tol
        self.outer_radius = self.centerline_radius(self.theta_max)
        # Turn indices k with φ + 2πk able to reach [0, theta_max] (and one more on each side).
        k_min = int(np.floor((-np.pi - 2.0 * np.pi) / (2.0 * np.pi)))
        k_max = int(np.ceil((self.theta_max + np.pi + 2.0 * np.pi) / (2.0 * np.pi)))
        self._turn_offsets = 2.0 * np.pi * np.arange(k_min, k_max + 1)
        dense_theta = np.linspace(0.0, self.theta_max, 20_001)
        speed = np.hypot(self.centerline_radius(dense_theta), self.pitch / (2.0 * np.pi))
        self._dense_theta = dense_theta
        self._dense_arc = np.concatenate(([0.0], np.cumsum(0.5 * (speed[1:] + speed[:-1]) * np.diff(dense_theta))))
        self.centerline_length = float(self._dense_arc[-1])
        self.probe_points, self.probe_normals = self.sample_boundary(n_probe_points)

    # Centerline -----------------------------------------------------------------------------
    def centerline_radius(self, theta):
        return self.inner_radius + self.pitch * theta / (2.0 * np.pi)

    def arc_length(self, theta):
        """Centerline arc length from the inner end."""
        return np.interp(theta, self._dense_theta, self._dense_arc)

    def theta_at_arc_length(self, arc):
        return np.interp(arc, self._dense_arc, self._dense_theta)

    def strip_point(self, theta, offset=0.0):
        """Point at roll angle ``theta`` and radial offset ``offset`` from the centerline."""
        theta = np.asarray(theta, dtype=float)
        radius = self.centerline_radius(theta) + offset
        return np.stack((radius * np.cos(theta), radius * np.sin(theta)), axis=-1)

    # Nearest turn ---------------------------------------------------------------------------
    def _nearest(self, x, xp):
        """Nearest strip point among the candidate turns, with its roll angle and radial offset.

        For each turn, the angle is clipped to ``[0, theta_max]`` and the radial component along
        that direction is clipped to the strip. Inside the strip this is exact (distance zero);
        outside it is the closest point up to the slope of the spiral (a few percent here).
        Returns ``(projected, theta, offset)``; ``offset`` is unclipped.
        """
        phi = xp.arctan2(x[..., 1], x[..., 0])
        raw_theta = phi[..., None] + xp.asarray(self._turn_offsets, dtype=x.dtype)
        theta = xp.clip(raw_theta, 0.0, self.theta_max)
        direction = xp.stack((xp.cos(theta), xp.sin(theta)), axis=-1)
        radial = xp.sum(x[..., None, :] * direction, axis=-1)
        center = self.centerline_radius(theta)
        clipped = xp.clip(radial, center - self.half_width, center + self.half_width)
        # Points inside a turn are their own candidate exactly (no rounding from cos/sin).
        exact = (raw_theta >= 0.0) & (raw_theta <= self.theta_max) & (clipped == radial)
        candidates = xp.where(exact[..., None], x[..., None, :], clipped[..., None] * direction)
        distance_sq = xp.sum((x[..., None, :] - candidates) ** 2, axis=-1)
        best = xp.argmin(distance_sq, axis=-1)[..., None]
        projected = xp.take_along_axis(candidates, best[..., None], axis=-2)[..., 0, :]
        theta_best = xp.take_along_axis(theta, best, axis=-1)[..., 0]
        offset = xp.take_along_axis(radial - center, best, axis=-1)[..., 0]
        return projected, theta_best, offset

    def roll_angle(self, samples):
        """Roll angle θ of the nearest turn (0 at the inner end), as a NumPy array."""
        return self._nearest(np.asarray(samples, dtype=float), np)[1]

    # Sampling -------------------------------------------------------------------------------
    def sample_uniform(self, n_samples, rng, batch_size=20_000):
        """Area-uniform samples by rejection from the bounding square."""
        extent = self.outer_radius + self.half_width
        accepted, n_accepted = [], 0
        while n_accepted < n_samples:
            candidates = rng.uniform(-extent, extent, size=(batch_size, 2))
            kept = candidates[self.contains(candidates, tol=0.0)]
            accepted.append(kept)
            n_accepted += len(kept)
        return np.vstack(accepted)[:n_samples]

    def sample_gaussian(self, mean, variance, n_samples, rng, theta_window=None, batch_size=10_000):
        """Isotropic Gaussian conditioned on the strip and, optionally, on ``θ ∈ theta_window``.

        The angle window keeps a cloud near one end of the roll from leaking onto the
        neighboring turn. Returns ``(samples, n_drawn, n_accepted)``.
        """
        std = np.sqrt(variance)
        accepted, n_drawn, n_accepted = [], 0, 0
        while n_accepted < n_samples:
            candidates = rng.normal(loc=mean, scale=std, size=(batch_size, 2))
            n_drawn += len(candidates)
            keep = self.contains(candidates, tol=0.0)
            if theta_window is not None:
                theta = self.roll_angle(candidates)
                keep &= (theta >= theta_window[0]) & (theta <= theta_window[1])
            kept = candidates[keep]
            if len(kept) > 0:
                accepted.append(kept)
                n_accepted += len(kept)
        return np.vstack(accepted)[:n_samples], n_drawn, n_accepted

    def sample_boundary(self, n_points=96):
        """Points on both long edges and both end caps, with their interior unit normals."""
        theta = self.theta_at_arc_length(np.linspace(0.0, self.centerline_length, n_points))[1:-1]
        points, normals = [], []
        for sign in (-1.0, 1.0):  # inner edge, outer edge
            edge = self.strip_point(theta, sign * self.half_width)
            normals.append(-sign * self._edge_normal_np(edge, theta))
            points.append(edge)
        offsets = np.linspace(-self.half_width, self.half_width, 9)[1:-1]
        for theta_end, sign in ((0.0, 1.0), (self.theta_max, -1.0)):
            points.append(self.strip_point(np.full(len(offsets), theta_end), offsets))
            normals.append(np.tile(sign * np.array([-np.sin(theta_end), np.cos(theta_end)]), (len(offsets), 1)))
        return np.vstack(points).astype(np.float32), np.vstack(normals).astype(np.float32)

    def _edge_normal_np(self, points, theta):
        """Unit gradient of ``ρ - r(θ)``: points away from the centerline across the outer edge."""
        radius = np.linalg.norm(points, axis=-1, keepdims=True)
        e_r = points / radius
        e_theta = np.stack((-e_r[..., 1], e_r[..., 0]), axis=-1)
        gradient = e_r - (self.pitch / (2.0 * np.pi)) / radius * e_theta
        return gradient / np.linalg.norm(gradient, axis=-1, keepdims=True)

    # Geometry -------------------------------------------------------------------------------
    def project(self, x):
        x = jnp.asarray(x)
        return self._nearest(x, jnp)[0]

    def interior_normals(self, x, eps=1e-12):
        """Interior unit normals of the active long edge and end cap at ``x`` (zero if inactive)."""
        y, theta, offset = self._nearest(jnp.asarray(x), jnp)
        radius = jnp.maximum(jnp.linalg.norm(y, axis=-1, keepdims=True), eps)
        e_r = y / radius
        e_theta = jnp.stack((-e_r[..., 1], e_r[..., 0]), axis=-1)
        gradient = e_r - (self.pitch / (2.0 * np.pi)) / radius * e_theta
        gradient = gradient / jnp.linalg.norm(gradient, axis=-1, keepdims=True)
        offset = offset[..., None]
        on_outer = offset >= self.half_width - self.boundary_tol
        on_inner = offset <= -self.half_width + self.boundary_tol
        edge_normal = jnp.where(on_outer, -gradient, jnp.where(on_inner, gradient, 0.0))
        arc_tol = self.boundary_tol / radius
        at_inner_end = theta[..., None] <= arc_tol
        at_outer_end = theta[..., None] >= self.theta_max - arc_tol
        end_normal = jnp.where(at_inner_end, e_theta, jnp.where(at_outer_end, -e_theta, 0.0))
        return edge_normal, end_normal

    def guard_velocity(self, x, v):
        """Tangent-cone guard: remove outward components across the active edge and end cap.

        The two normals are nearly orthogonal at the corners; two sweeps settle the remainder.
        """
        v = jnp.asarray(v)
        edge_normal, end_normal = self.interior_normals(x)
        for _ in range(2):
            for normal in (edge_normal, end_normal):
                component = jnp.sum(v * normal, axis=-1, keepdims=True)
                v = v - jnp.minimum(component, 0.0) * normal
        return v

    # Geodesics ------------------------------------------------------------------------------
    def _taut_string(self, start, end, n_points=1025, levels=(33, 129)):
        """Shortest path between two roll points as a dense polyline ``(n_points, 2)``.

        The roll is simply connected and is the box ``|s| <= half_width`` in strip coordinates
        ``(θ, s)``, where ``s`` is the radial offset from the centerline. The shortest path is the taut
        string, and it advances monotonically in ``θ``. We fix a uniform ``θ`` grid between the
        endpoints and minimize the polyline length over the offsets with box-constrained L-BFGS-B.
        Coarse levels supply the initial guess. Chords between neighboring grid points can leave the
        strip by at most their sagitta (about ``r Δθ² / 8``, below ``1e-5`` here).
        """
        from scipy.optimize import minimize

        _, (theta0, theta1), (s0, s1) = self._nearest(np.stack((start, end)).astype(float), np)
        w = self.half_width
        previous = None
        for n in (*levels, n_points):
            thetas = np.linspace(theta0, theta1, n)
            radial = np.stack((np.cos(thetas), np.sin(thetas)), axis=-1)
            base = self.centerline_radius(thetas)
            u = np.linspace(0.0, 1.0, n)
            init = s0 + u * (s1 - s0) if previous is None else np.interp(u, np.linspace(0.0, 1.0, len(previous)), previous)

            def length_and_grad(s_inner, radial=radial, base=base):
                s = np.concatenate(([s0], s_inner, [s1]))
                points = (base + s)[:, None] * radial
                segments = np.diff(points, axis=0)
                norms = np.maximum(np.linalg.norm(segments, axis=1), 1e-15)
                unit = segments / norms[:, None]
                grad_points = np.zeros_like(points)
                grad_points[:-1] -= unit
                grad_points[1:] += unit
                return norms.sum(), np.sum(grad_points * radial, axis=1)[1:-1]

            result = minimize(length_and_grad, init[1:-1], jac=True, method="L-BFGS-B",
                              bounds=[(-w, w)] * (n - 2), options={"maxiter": 20_000, "ftol": 1e-15, "gtol": 1e-12})
            previous = np.concatenate(([s0], result.x, [s1]))
        path = self.strip_point(thetas, previous)
        path[0], path[-1] = start, end
        return path

    def geodesic(self, x0, x1, times):
        """Exact (numerically converged) shortest paths in the roll, traversed at constant speed.

        Parameters
        ----------
        x0, x1
            Start and end points of shape ``(n, 2)`` inside the roll.
        times
            Times in ``[0, 1]`` of shape ``(m,)``; ``t`` maps to arc length ``t * length``.

        Returns
        -------
        ``(paths, lengths)`` with shapes ``(n, m, 2)`` and ``(n,)``. Dense solutions are cached per pair.
        """
        if not hasattr(self, "_geodesic_cache"):
            self._geodesic_cache = {}
        x0, x1 = np.asarray(x0, dtype=float), np.asarray(x1, dtype=float)
        times = np.asarray(times, dtype=float)
        paths, lengths = [], []
        for start, end in zip(x0, x1, strict=True):
            key = (start.tobytes(), end.tobytes())
            if key not in self._geodesic_cache:
                dense = self._taut_string(start, end)
                arc = np.concatenate(([0.0], np.cumsum(np.linalg.norm(np.diff(dense, axis=0), axis=1))))
                self._geodesic_cache[key] = (dense, arc)
            dense, arc = self._geodesic_cache[key]
            s = times * arc[-1]
            paths.append(np.column_stack([np.interp(s, arc, dense[:, k]) for k in range(2)]))
            lengths.append(arc[-1])
        return np.stack(paths), np.asarray(lengths)

    def violation(self, samples):
        samples = np.asarray(samples, dtype=float)
        projected = self._nearest(samples, np)[0]
        return np.linalg.norm(samples - projected, axis=-1)

    def contains(self, samples, tol=1e-4):
        return self.violation(samples) <= tol

    def radial_coordinate(self, samples):
        """Radial offset from the centerline of the nearest turn; ``|offset| <= half_width`` on the strip."""
        return self._nearest(np.asarray(samples, dtype=float), np)[2]

    def max_penetration(self, trajectories):
        """Largest distance outside the strip over saved path states."""
        return float(self.violation(trajectories).max())

    def _segment_violation(self, starts, ends, n_samples=9):
        """Distance outside the strip at ``n_samples`` points along each straight segment."""
        fractions = np.linspace(0.0, 1.0, n_samples)
        probes = starts[..., None, :] + fractions[:, None] * (ends - starts)[..., None, :]
        return self.violation(probes).max(axis=-1)

    def pair_masks(self, source, target, n_samples=33):
        """Pair diagnostics for coupling tables: same side of the centerline, chord leaving the strip."""
        same_side = np.sign(self.radial_coordinate(source))[:, None] == np.sign(self.radial_coordinate(target))[None, :]
        chord_leaves = np.zeros((len(source), len(target)), dtype=bool)
        for row in range(len(source)):
            starts = np.broadcast_to(source[row], target.shape)
            chord_leaves[row] = self._segment_violation(starts, target, n_samples) > 1e-3
        return {"same_side": same_side, "chord_leaves_roll": chord_leaves}

    def path_geometry(self, path):
        segments = np.diff(path, axis=0)
        segment_violation = self._segment_violation(path[:-1], path[1:])
        return {
            "max_step_length": np.sqrt(np.sum(segments**2, axis=-1).max()),
            "max_segment_gap_penetration": segment_violation.max(),
            "gap_crossing_segment_fraction": np.mean(segment_violation > 1e-3),
            "max_domain_violation": self.violation(path).max(),
        }

    def probe_diagnostic(self, velocities):
        return {
            "outward_boundary_fraction": np.mean(np.sum(velocities * self.probe_normals, axis=1) < -1e-6),
        }

    def field_grid(self, n_along=60, n_across=4, margin=0.05):
        """Grid strictly inside the strip, uniform in arc length and offset, for raw kernel fields."""
        theta = self.theta_at_arc_length(np.linspace(0.2, self.centerline_length - 0.2, n_along))
        offsets = np.linspace(-self.half_width + margin, self.half_width - margin, n_across)
        grid_theta, grid_offset = np.meshgrid(theta, offsets)
        return self.strip_point(grid_theta.ravel(), grid_offset.ravel()).astype(np.float32)

    def learned_field_grid(self, grid_size=27):
        """Square grid restricted to the strip."""
        extent = self.outer_radius + self.half_width
        axis_values = np.linspace(-extent, extent, grid_size, dtype=np.float32)
        grid_x, grid_y = np.meshgrid(axis_values, axis_values)
        grid = np.column_stack((grid_x.ravel(), grid_y.ravel()))
        return grid[self.contains(grid, tol=0.0)]

    def path_metrics(self, trajectories):
        """Domain diagnostics of saved learned paths ``(n, T, 2)``.

        Segment penetration samples each straight segment between saved times, so it also
        detects chords that cut across a gap between two saved times. A path *skips a turn*
        when its nearest-turn roll angle jumps by more than π between two saved times.
        """
        offsets = self.radial_coordinate(trajectories)
        inside = self.contains(trajectories)
        steps = np.linalg.norm(np.diff(trajectories, axis=1), axis=-1)
        segment_violation = self._segment_violation(trajectories[:, :-1], trajectories[:, 1:])
        theta_jumps = np.abs(np.diff(self.roll_angle(trajectories), axis=1))
        return {
            "paths_leaving_roll": int(np.count_nonzero(~inside.all(axis=1))),
            "endpoints_in_roll": float(np.mean(inside[:, -1])),
            "fraction_saved_points_in_roll": float(np.mean(inside)),
            "paths_skipping_a_turn": int(np.count_nonzero((theta_jumps > np.pi).any(axis=1))),
            "min_sampled_offset": float(offsets.min()),
            "max_sampled_offset": float(offsets.max()),
            "mean_path_length": float(steps.sum(axis=1).mean()),
            "max_step_length": float(steps.max()),
            "max_segment_gap_penetration": float(segment_violation.max()),
            "gap_crossing_segment_fraction": float(np.mean(segment_violation > 1e-3)),
            "max_roll_violation": float(self.violation(trajectories).max()),
        }

    def endpoint_metrics(self, samples):
        """Roll-specific endpoint metrics added to the distribution metrics.

        ``between_turns_fraction``: off the strip but inside its outer envelope (in a gap or the
        central hole). ``not_on_inner_turn_fraction``: endpoints whose nearest turn is more than
        one full turn from the inner end, i.e. that did not wind in to the innermost turn.
        """
        samples = np.asarray(samples, dtype=float)
        violation = self.violation(samples)
        outside = violation > 1e-4
        radii = np.linalg.norm(samples, axis=-1)
        envelope = self.outer_radius + self.half_width
        return {
            "between_turns_fraction": float(np.mean(outside & (radii <= envelope))),
            "beyond_outer_fraction": float(np.mean(outside & (radii > envelope))),
            "not_on_inner_turn_fraction": float(np.mean(self.roll_angle(samples) > 2.0 * np.pi)),
            "mean_roll_violation": float(violation.mean()),
            "p95_roll_violation": float(np.quantile(violation, 0.95)),
        }

    # Drawing --------------------------------------------------------------------------------
    def outline(self, n_points=600):
        """Closed boundary polyline: inner edge, outer end cap, outer edge back, inner end cap."""
        theta = np.linspace(0.0, self.theta_max, n_points)
        inner_edge = self.strip_point(theta, -self.half_width)
        outer_edge = self.strip_point(theta[::-1], self.half_width)
        return np.vstack((inner_edge, outer_edge, inner_edge[:1]))

    def finish_axes(self, ax, extra_points=None, extent=None, outer_color="black", label_boundaries=False,
                    grid_alpha=0.2):
        outline = self.outline()
        ax.plot(*outline.T, color=outer_color, linewidth=1.5 if label_boundaries else 1.0,
                label="roll boundary" if label_boundaries else None)
        ax.plot(*self.strip_point(np.linspace(0.0, self.theta_max, 400)).T, color="tab:red", linestyle=":",
                linewidth=0.8, alpha=0.6, label="centerline" if label_boundaries else None)
        if extent is None:
            extent = self.outer_radius + self.half_width + 0.3
            if extra_points is not None and len(extra_points):
                extent = max(extent, float(np.max(np.abs(extra_points))) + 0.2)
        ax.set(aspect="equal", xlim=(-extent, extent), ylim=(-extent, extent), xlabel="x", ylabel="y")
        ax.grid(alpha=grid_alpha)

    draw_arrows = Annulus.draw_arrows
    draw_highlight_arrow = Annulus.draw_highlight_arrow
    draw_direction_arrows = Annulus.draw_direction_arrows

    def draw_radial_reference(self, ax):
        ax.axhspan(-self.half_width, self.half_width, color="tab:green", alpha=0.1, label="Roll strip")
        ax.axhline(-self.half_width, color="gray", linestyle="--")
        ax.axhline(self.half_width, color="gray", linestyle="--")
