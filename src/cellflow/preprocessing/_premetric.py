import numpy as np
import jax.numpy as jnp
import jax
from sklearn.neighbors import NearestNeighbors
from scipy.sparse import coo_matrix, diags
from scipy.sparse.linalg import lobpcg


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
        sigma = np.median(dist)

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
    D_inv_sqrt = diags(1.0 / np.sqrt(d + 1e-12))
    I = diags(np.ones(len(d)))

    L = I - D_inv_sqrt @ W @ D_inv_sqrt
    return L


# ---------------------------
# Eigen decomposition
# ---------------------------
def compute_eigens(L, k=50):
    n = L.shape[0]
    X0 = np.random.randn(n, k)

    eigvals, eigvecs = lobpcg(L, X0, largest=False, maxiter=200)
    return eigvals, eigvecs


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
def spectral_distance(i, j, Psi):
    diff = Psi[i] - Psi[j]
    return np.dot(diff, diff)


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
        eps=1e-12,
    ):
        """
        Phi_train must be UNWEIGHTED normalized-Laplacian eigenvectors (NOT Psi).
        """
        self.X = jnp.array(X_train)
        self.Phi = jnp.array(Phi_train)
        self.lam = jnp.array(eigenvalues)
        self.sigma = sigma
        self.k = k
        self.Psi = jnp.array(psi)
        self.x_to_id = jnp.array(x_to_id)
        self.spectral_weight_type = spectral_weight_type
        self.embedding_eps = embedding_eps
        self.eps = eps

        if graph is not None and train_degrees is not None:
            raise ValueError("Only one of graph or train_degrees can be provided.")

        if graph is not None:
            train_degrees = np.asarray(graph.sum(axis=1)).ravel()
        elif train_degrees is None:
            W, sigma = build_knn_graph(
                np.asarray(X_train),
                k=k,
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
                np.asarray(eigenvalues),
                weight_type=spectral_weight_type,
                eps=embedding_eps,
            )

        spectral_scale = np.asarray(spectral_scale)
        if spectral_scale.shape != np.asarray(eigenvalues).shape:
            raise ValueError(
                "spectral_scale must have the same shape as eigenvalues, "
                f"got {spectral_scale.shape} and {np.asarray(eigenvalues).shape}."
            )
        if np.any(~np.isfinite(spectral_scale)):
            raise ValueError("spectral_scale must contain only finite values.")

        self.spectral_scale = jnp.array(spectral_scale)

        adjacency_eigenvalues = 1.0 - self.lam
        adjacency_eigenvalues = jnp.where(
            jnp.abs(adjacency_eigenvalues) > eps,
            adjacency_eigenvalues,
            jnp.where(adjacency_eigenvalues < 0.0, -eps, eps),
        )
        self.embedding_scale = self.spectral_scale / adjacency_eigenvalues

        self.nn = NearestNeighbors(n_neighbors=k)
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

        # Keep the differentiated x embedding continuous. Training/support points still
        # use their exact stored embedding through embed_point(), where no x-gradient is needed.
        psi_x, diff, K, point_degree, degree_weighted_phi, spectral_sum = self._embed_point_with_terms(x)
        psi_y = self.embed_point(y)

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
