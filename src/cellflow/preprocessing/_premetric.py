import jax
import jax.numpy as jnp
import numpy as np
from scipy.sparse import coo_matrix, diags
from scipy.sparse.linalg import lobpcg
from sklearn.neighbors import NearestNeighbors


# ---------------------------
# kNN graph construction
# ---------------------------
def build_knn_graph(X, k=30, sigma=None, symmetrize="max", remove_self=True):
    """
    Build a weighted kNN graph using a Gaussian kernel.

    Parameters
    ----------
    X : array, shape (n, d)
        Input data.

    k : int
        Number of non-self nearest neighbors.

    sigma : float or None
        Gaussian kernel bandwidth. If None, uses median non-self kNN distance.

    symmetrize : {"max", "mean", None}
        How to make the graph symmetric.
        - "max":  W = max(W, W.T)
        - "mean": W = 0.5 * (W + W.T)
        - None: keep directed kNN graph

    remove_self : bool
        Whether to remove self-neighbors / diagonal entries.

    Returns
    -------
    W : scipy.sparse.csr_matrix, shape (n, n)
        Weighted adjacency matrix.

    sigma : float
        Gaussian bandwidth used.
    """

    n = X.shape[0]
    if k <= 0:
        raise ValueError("k must be positive.")
    if remove_self and k >= n:
        raise ValueError(f"k must be smaller than the number of points when remove_self=True, got k={k} and n={n}.")
    if not remove_self and k > n:
        raise ValueError(f"k cannot exceed the number of points, got k={k} and n={n}.")

    if remove_self:
        n_neighbors = k + 1
    else:
        n_neighbors = k

    nn = NearestNeighbors(n_neighbors=n_neighbors).fit(X)
    dist, ind = nn.kneighbors(X)

    if remove_self:
        # Usually the first neighbor is the point itself.
        # This removes the zero-distance self-neighbor.
        dist = dist[:, 1:]
        ind = ind[:, 1:]

    if sigma is None:
        positive_distances = dist[dist > 0]
        if positive_distances.size == 0:
            raise ValueError("Cannot infer sigma because all neighbor distances are zero.")
        sigma = np.median(positive_distances)

    if sigma <= 0:
        raise ValueError(f"sigma must be positive, got sigma={sigma}")

    rows = np.repeat(np.arange(n), dist.shape[1])
    cols = ind.ravel()

    weights = np.exp(-(dist.ravel() ** 2) / (2.0 * sigma**2))

    W = coo_matrix((weights, (rows, cols)), shape=(n, n)).tocsr()

    if symmetrize == "max":
        W = W.maximum(W.T)
    elif symmetrize == "mean":
        W = 0.5 * (W + W.T)
    elif symmetrize is None:
        pass
    else:
        raise ValueError("symmetrize must be one of {'max', 'mean', None}")

    if remove_self:
        W.setdiag(0.0)
        W.eliminate_zeros()

    return W.tocsr(), sigma


# ---------------------------
# Normalized Laplacian
# ---------------------------
def normalized_laplacian(W):
    d = np.array(W.sum(axis=1)).flatten()
    if np.any(d <= 0):
        raise ValueError("Cannot build a normalized Laplacian from a graph with zero-degree nodes.")
    D_inv_sqrt = diags(1.0 / np.sqrt(d + 1e-12))
    I = diags(np.ones(len(d)))

    L = I - D_inv_sqrt @ W @ D_inv_sqrt
    return L


# ---------------------------
# Eigen decomposition
# ---------------------------
def compute_eigens(L, k=50, random_state=0, maxiter=200):
    n = L.shape[0]
    if k <= 0:
        raise ValueError("k must be positive.")
    if k > n:
        raise ValueError(f"k cannot exceed the matrix size, got k={k} and n={n}.")
    rng = np.random.default_rng(random_state)
    X0 = rng.standard_normal((n, k))

    eigvals, eigvecs = lobpcg(L, X0, largest=False, maxiter=maxiter)
    order = np.argsort(eigvals)
    return eigvals[order], eigvecs[:, order]


def spectral_embedding_scale(eigvals, weight_type="biharmonic", eps=1e-8):
    """Return the per-eigenvector scale used by the spectral embedding."""
    if weight_type == "biharmonic":
        return 1.0 / (eigvals + eps)
    elif weight_type == "diffusion":
        return np.exp(-0.5 * eigvals)
    elif weight_type == "none":
        return np.ones_like(eigvals)
    raise ValueError("weight_type must be one of {'biharmonic', 'diffusion', 'none'}")


# ---------------------------
# Spectral scaling (your choice)
# ---------------------------
def spectral_embedding(eigvals, eigvecs, weight_type="biharmonic", eps=1e-8):
    scale = spectral_embedding_scale(eigvals, weight_type=weight_type, eps=eps)

    return eigvecs * scale[None, :]


# ---------------------------
# Spectral distance
# ---------------------------
def spectral_distance(i, j, Psi, squared=False):
    Psi = np.asarray(Psi)
    diff = Psi[i] - Psi[j]
    d2 = np.dot(diff, diff)
    return d2 if squared else np.sqrt(d2)


# =========================================================
# NYSTRÖM EXTENSION (consistent version)
# =========================================================
class SpectralNystroem:
    def __init__(
        self,
        X_train,
        Phi_train,
        eigenvalues,
        sigma,
        psi,
        x_to_id,
        k=30,
        graph=None,
        train_degrees=None,
        spectral_scale=None,
        spectral_weight_type="biharmonic",
        embedding_eps=1e-8,
        drop_trivial_eigenvectors=True,
        trivial_eigenvalue_eps=1e-6,
        min_extension_eigenvalue_abs=1e-6,
        target_embedding="nystroem",
        eps=1e-12,
    ):
        """
        Phi_train must be UNWEIGHTED normalized-Laplacian eigenvectors (NOT Psi).
        """
        X_train = np.asarray(X_train)
        Phi_train = np.asarray(Phi_train)
        eigenvalues = np.asarray(eigenvalues)
        psi = np.asarray(psi)
        x_to_id = np.asarray(x_to_id)

        if X_train.ndim != 2:
            raise ValueError("X_train must have shape (n_points, n_features).")
        if Phi_train.ndim != 2:
            raise ValueError("Phi_train must have shape (n_points, n_eigenvectors).")
        if psi.ndim != 2:
            raise ValueError("psi must have shape (n_points, n_eigenvectors).")
        if eigenvalues.ndim != 1:
            raise ValueError("eigenvalues must be one-dimensional.")
        if Phi_train.shape != psi.shape:
            raise ValueError(f"Phi_train and psi must have the same shape, got {Phi_train.shape} and {psi.shape}.")
        if Phi_train.shape[0] != X_train.shape[0]:
            raise ValueError(
                f"Phi_train must have one row per training point, got {Phi_train.shape[0]} and {X_train.shape[0]}."
            )
        if Phi_train.shape[1] != eigenvalues.shape[0]:
            raise ValueError(
                "The number of eigenvectors must match the number of eigenvalues, "
                f"got {Phi_train.shape[1]} and {eigenvalues.shape[0]}."
            )
        if x_to_id.shape != X_train.shape:
            raise ValueError(f"x_to_id must have the same shape as X_train, got {x_to_id.shape} and {X_train.shape}.")
        if trivial_eigenvalue_eps < 0:
            raise ValueError("trivial_eigenvalue_eps must be non-negative.")
        if min_extension_eigenvalue_abs < 0:
            raise ValueError("min_extension_eigenvalue_abs must be non-negative.")
        if target_embedding not in {"nystroem", "exact"}:
            raise ValueError("target_embedding must be one of {'nystroem', 'exact'}.")

        keep_components = np.ones_like(eigenvalues, dtype=bool)
        if drop_trivial_eigenvectors:
            keep_components &= eigenvalues > trivial_eigenvalue_eps
        if min_extension_eigenvalue_abs > 0:
            keep_components &= np.abs(1.0 - eigenvalues) > min_extension_eigenvalue_abs
        if not np.any(keep_components):
            raise ValueError(
                "All spectral components were filtered. Use more eigenvectors or relax "
                "`trivial_eigenvalue_eps` / `min_extension_eigenvalue_abs`."
            )

        self.spectral_component_mask = jnp.array(keep_components)
        self.n_dropped_spectral_components = int(np.size(keep_components) - np.count_nonzero(keep_components))
        Phi_train = Phi_train[:, keep_components]
        eigenvalues = eigenvalues[keep_components]
        psi = psi[:, keep_components]

        self.X = jnp.array(X_train)
        self.Phi = jnp.array(Phi_train)
        self.lam = jnp.array(eigenvalues)
        self.sigma = sigma
        self.k = min(k, X_train.shape[0])
        self.Psi = jnp.array(psi)
        self.x_to_id = jnp.array(x_to_id)
        self.spectral_weight_type = spectral_weight_type
        self.embedding_eps = embedding_eps
        self.drop_trivial_eigenvectors = drop_trivial_eigenvectors
        self.trivial_eigenvalue_eps = trivial_eigenvalue_eps
        self.min_extension_eigenvalue_abs = min_extension_eigenvalue_abs
        self.target_embedding = target_embedding
        self.eps = eps

        if self.k <= 0:
            raise ValueError("k must be positive.")

        if graph is not None and train_degrees is not None:
            raise ValueError("Only one of graph or train_degrees can be provided.")

        if graph is not None:
            train_degrees = np.asarray(graph.sum(axis=1)).ravel()
        elif train_degrees is None:
            W, sigma = build_knn_graph(
                np.asarray(X_train),
                k=min(k, X_train.shape[0] - 1),
                sigma=sigma,
                symmetrize="max",
                remove_self=True,
            )
            self.sigma = sigma
            train_degrees = np.asarray(W.sum(axis=1)).ravel()

        if self.sigma is None:
            raise ValueError("sigma must be provided when graph or train_degrees is provided.")

        train_degrees = np.asarray(train_degrees)
        if train_degrees.shape != (np.asarray(X_train).shape[0],):
            raise ValueError(
                f"train_degrees must have shape ({np.asarray(X_train).shape[0]},), got {train_degrees.shape}."
            )
        if np.any(train_degrees <= 0):
            raise ValueError("All train_degrees must be positive.")

        self.train_degrees = jnp.array(train_degrees)

        if spectral_scale is None:
            spectral_scale = spectral_embedding_scale(
                eigenvalues,
                weight_type=spectral_weight_type,
                eps=embedding_eps,
            )
        else:
            spectral_scale = np.asarray(spectral_scale)
            if spectral_scale.shape == np.asarray(keep_components).shape:
                spectral_scale = spectral_scale[keep_components]
            elif spectral_scale.shape != eigenvalues.shape:
                raise ValueError(
                    "spectral_scale must have the same shape as the unfiltered or filtered eigenvalues, "
                    f"got {spectral_scale.shape}; expected {np.asarray(keep_components).shape} "
                    f"or {eigenvalues.shape}."
                )

        spectral_scale = np.asarray(spectral_scale)
        if spectral_scale.shape != eigenvalues.shape:
            raise ValueError(
                "spectral_scale must have the same shape as eigenvalues, "
                f"got {spectral_scale.shape} and {eigenvalues.shape}."
            )
        if np.any(~np.isfinite(spectral_scale)):
            raise ValueError("spectral_scale must contain only finite values.")

        self.spectral_scale = jnp.array(spectral_scale)

        adjacency_eigenvalues = 1.0 - self.lam
        adjacency_eigenvalues = jnp.where(
            jnp.abs(adjacency_eigenvalues) > min_extension_eigenvalue_abs,
            adjacency_eigenvalues,
            jnp.where(adjacency_eigenvalues < 0.0, -min_extension_eigenvalue_abs, min_extension_eigenvalue_abs),
        )
        self.embedding_scale = self.spectral_scale / adjacency_eigenvalues

        self.nn = NearestNeighbors(n_neighbors=self.k)
        self.nn.fit(X_train)

    # ---------------------------
    # Gaussian kernel
    # ---------------------------
    def _kernel(self, x, Xn):
        diff = Xn - x[None, :]
        dist2 = jnp.sum(diff**2, axis=1)
        return jnp.exp(-dist2 / (2 * self.sigma**2))

    # ---------------------------
    # Nyström extension
    # ---------------------------
    def _neighbor_indices(self, x):
        d2 = jnp.sum((self.X - x) ** 2, axis=1)
        _, idx = jax.lax.top_k(-d2, self.k)
        return idx

    def _training_embedding(self, x):
        matches = jnp.all(self.x_to_id == x, axis=1)
        idx = jnp.argmax(matches)
        return matches[idx], self.Psi[idx]

    def _embed_point_with_terms(self, x):
        idx = self._neighbor_indices(x)
        Xn = self.X[idx]
        Phin = self.Phi[idx]
        train_degrees = self.train_degrees[idx]

        diff = Xn - x[None, :]
        K = self._kernel(x, Xn)
        point_degree = jnp.maximum(jnp.sum(K), self.eps)

        degree_weighted_phi = Phin / jnp.sqrt(jnp.maximum(train_degrees, self.eps))[:, None]
        spectral_sum = jnp.sum(K[:, None] * degree_weighted_phi, axis=0)
        point_degree_scale = 1.0 / jnp.sqrt(point_degree)
        psi_x = spectral_sum * point_degree_scale * self.embedding_scale

        return psi_x, diff, K, point_degree, degree_weighted_phi, spectral_sum

    def embed_point(self, x):
        psi_x, *_ = self._embed_point_with_terms(x)
        is_training_point, psi_train = self._training_embedding(x)
        return jnp.where(is_training_point, psi_train, psi_x)

    def distance_grad_and_norm(self, x, y):
        eps = self.eps

        # Keep the differentiated x embedding continuous. By default, the target
        # embedding uses the same extension so the vector field does not chase an
        # exact support embedding that the local Nystroem map may not reproduce.
        psi_x, diff, K, point_degree, degree_weighted_phi, spectral_sum = self._embed_point_with_terms(x)
        if self.target_embedding == "exact":
            psi_y = self.embed_point(y)
        else:
            psi_y, *_ = self._embed_point_with_terms(y)

        # residual in scaled spectral space
        r = psi_x - psi_y

        d2_spec = jnp.dot(r, r) + eps
        d_spec = jnp.sqrt(d2_spec)

        scaled_residual = r * self.embedding_scale
        neighbor_coeff = degree_weighted_phi @ scaled_residual
        normalization_coeff = jnp.dot(spectral_sum, scaled_residual)
        kernel_diff = jnp.sum(K[:, None] * diff, axis=0)

        q = (
            jnp.sum((K * neighbor_coeff)[:, None] * diff, axis=0) / jnp.sqrt(point_degree)
            - 0.5 * normalization_coeff * kernel_diff / (point_degree ** 1.5)
        )

        # gradient of unsquared distance
        grad_d = q / (self.sigma ** 2 * d_spec)

        # squared Euclidean norm
        grad_d_norm_sq = jnp.dot(grad_d, grad_d) + eps

        same_point = jnp.all(x == y)
        d_spec = jnp.where(same_point, 0.0, d_spec)
        grad_d = jnp.where(same_point, jnp.zeros_like(grad_d), grad_d)
        grad_d_norm_sq = jnp.where(same_point, eps, grad_d_norm_sq)

        return d_spec, grad_d, grad_d_norm_sq

    def distance_grad_and_norm_batch(self, x, y):
        return jax.vmap(self.distance_grad_and_norm)(x, y)
