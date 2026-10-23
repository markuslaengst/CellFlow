import jax
import jax.numpy as jnp
import numpy as np
from scipy.sparse import coo_matrix
from scipy.spatial import cKDTree
from sklearn.neighbors import NearestNeighbors


class SymmetricBandwidthKernel:
    """Fit symmetric affinities using kth-neighbor bandwidths.

    ``gaussian``, ``polynomial``, ``bump``, and ``indicator`` evaluate a
    profile at ``||x-y||^2 / (delta^2 * rho(x) * rho(y))``. The polynomial
    profile is ``max(1-s, 0)^4``. ``fixed_gaussian`` provides a fixed-sigma
    baseline on the same CkNN support. Gaussian profiles can also be used
    without truncation by setting ``compact_support=False``.

    Training scales exclude the point itself. At a query, the bandwidth is
    the distance to the (density_k + 1)th training point. This agrees with
    the fitted scales at training points and is continuous there, including
    when a query coincides with a training point. Its gradient is piecewise
    smooth; ties between neighbor distances can produce derivative jumps.
    Truncated Gaussian and indicator profiles additionally have a jump at
    the support boundary. Their returned gradients hold away from jumps.
    """

    def __init__(
        self,
        X_train,
        profile="gaussian",
        density_k=16,
        delta=1.0,
        sigma=None,
        compact_support=True,
    ):
        training_points = np.asarray(X_train, dtype=np.float64)
        if training_points.ndim != 2 or not np.all(np.isfinite(training_points)):
            raise ValueError("X_train must be a finite two-dimensional array.")
        if not isinstance(density_k, (int, np.integer)) or not 1 <= density_k < len(training_points):
            raise ValueError("density_k must be an integer between 1 and n_points - 1.")
        if profile not in {"gaussian", "fixed_gaussian", "polynomial", "bump", "indicator"}:
            raise ValueError("Unknown kernel profile.")
        if not np.isfinite(delta) or delta <= 0:
            raise ValueError("delta must be finite and positive.")
        if not compact_support and profile not in {"gaussian", "fixed_gaussian"}:
            raise ValueError("Only Gaussian profiles support compact_support=False.")

        neighbors = NearestNeighbors(n_neighbors=density_k).fit(training_points)
        neighbor_distances, _ = neighbors.kneighbors()
        local_scales = neighbor_distances[:, -1]
        if np.any(local_scales <= 0):
            raise ValueError("Local scales must be positive; increase density_k for duplicate points.")
        if sigma is None:
            sigma = float(np.median(neighbor_distances[neighbor_distances > 0]))
        if not np.isfinite(sigma) or sigma <= 0:
            raise ValueError("sigma must be finite and positive.")

        self.X = jnp.asarray(training_points)
        self.local_scales = jnp.asarray(local_scales, dtype=self.X.dtype)
        self.profile = profile
        self.density_k = int(density_k)
        self.delta = float(delta)
        self.sigma = float(sigma)
        self.compact_support = bool(compact_support)

    def _profile_values_and_derivatives(self, scaled_distances_sq):
        if self.profile in {"gaussian", "fixed_gaussian"}:
            values = jnp.exp(-0.5 * scaled_distances_sq)
            derivatives = -0.5 * values
        elif self.profile == "polynomial":
            remainder = jnp.maximum(1.0 - scaled_distances_sq, 0.0)
            values = remainder**4
            derivatives = -4.0 * remainder**3
        elif self.profile == "bump":
            remainder = jnp.maximum(1.0 - scaled_distances_sq, jnp.finfo(scaled_distances_sq.dtype).eps)
            values = jnp.exp(1.0 - 1.0 / remainder)
            derivatives = -values / remainder**2
        else:
            values = jnp.ones_like(scaled_distances_sq)
            derivatives = jnp.zeros_like(scaled_distances_sq)
        return values, derivatives

    def _weights(self, distances_sq, query_scale, neighbor_scales):
        support_distances_sq = distances_sq / (self.delta**2 * query_scale * neighbor_scales)
        profile_distances_sq = (
            distances_sq / self.sigma**2 if self.profile == "fixed_gaussian" else support_distances_sq
        )
        values, _ = self._profile_values_and_derivatives(profile_distances_sq)
        if self.compact_support:
            values = jnp.where(support_distances_sq < 1.0, values, 0.0)
        return values

    def values_and_gradients(self, point):
        """Return affinities and their full query gradients, including bandwidth derivatives."""
        point = jnp.asarray(point)
        differences = point[None, :] - self.X
        distances_sq = jnp.sum(differences**2, axis=1)
        _, scale_neighbors = jax.lax.top_k(-distances_sq, self.density_k + 1)
        scale_index = scale_neighbors[-1]
        minimum_distance_sq = jnp.finfo(distances_sq.dtype).tiny
        query_scale = jnp.sqrt(jnp.maximum(distances_sq[scale_index], minimum_distance_sq))
        scale_gradient = jnp.where(
            distances_sq[scale_index] > minimum_distance_sq,
            differences[scale_index] / query_scale,
            jnp.zeros_like(point),
        )
        bandwidth_sq = self.delta**2 * query_scale * self.local_scales
        support_distances_sq = distances_sq / bandwidth_sq
        if self.profile == "fixed_gaussian":
            profile_distances_sq = distances_sq / self.sigma**2
            argument_gradients = 2.0 * differences / self.sigma**2
        else:
            profile_distances_sq = support_distances_sq
            argument_gradients = (
                2.0 * differences / bandwidth_sq[:, None]
                - support_distances_sq[:, None] * (scale_gradient / query_scale)[None, :]
            )
        values, derivatives = self._profile_values_and_derivatives(profile_distances_sq)
        gradients = derivatives[:, None] * argument_gradients
        if self.compact_support:
            supported = support_distances_sq < 1.0
            values = jnp.where(supported, values, 0.0)
            gradients = jnp.where(supported[:, None], gradients, 0.0)
        return values, gradients

    def __call__(self, point):
        """Evaluate affinities to all training points without a fixed-neighbor truncation."""
        return self.values_and_gradients(point)[0]

    def graph(self, include_self=False):
        """Build the full symmetric CkNN support, or all pairs for untruncated Gaussians.

        Retain self-affinities when fitting a normalized-Laplacian Nyström
        extension. Self-affinities cancel from the unnormalized Laplacian.
        """
        training_points = np.asarray(self.X, dtype=np.float64)
        local_scales = np.asarray(self.local_scales, dtype=np.float64)
        if self.compact_support:
            pairs = cKDTree(training_points).query_pairs(
                self.delta * local_scales.max(), output_type="ndarray"
            )
            row_indices, column_indices = pairs.T
        else:
            row_indices, column_indices = np.triu_indices(len(training_points), k=1)
        distances_sq = np.sum((training_points[row_indices] - training_points[column_indices]) ** 2, axis=1)
        weights = np.asarray(
            self._weights(
                jnp.asarray(distances_sq),
                self.local_scales[row_indices],
                self.local_scales[column_indices],
            )
        )
        adjacency = coo_matrix(
            (
                np.concatenate((weights, weights)),
                (np.concatenate((row_indices, column_indices)), np.concatenate((column_indices, row_indices))),
            ),
            shape=(len(training_points), len(training_points)),
        ).tocsr()
        if include_self:
            adjacency.setdiag(1.0)
        adjacency.eliminate_zeros()
        return adjacency
