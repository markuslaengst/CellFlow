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


# ---------------------------
# Spectral scaling (your choice)
# ---------------------------
def spectral_embedding(eigvals, eigvecs, weight_type="biharmonic", eps=1e-8):
    if weight_type == "biharmonic":
        w = 1.0 / (eigvals + eps)**2
    elif weight_type == "diffusion":
        w = np.exp(-eigvals)
    else:
        w = np.ones_like(eigvals)

    return eigvecs * np.sqrt(w)[None, :]


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
    def __init__(self, X_train, Phi_train, eigenvalues, sigma, psi, x_to_id, k=30):
        """
        Phi_train must be UNWEIGHTED eigenvectors (NOT Psi)
        """
        self.X = jnp.array(X_train)
        self.Phi = jnp.array(Phi_train)
        self.lam = jnp.array(eigenvalues)
        self.sigma = sigma
        self.k = k
        self.Psi = jnp.array(psi)
        self.x_to_id = x_to_id

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
    def embed_point(self, x):
        d2 = jnp.sum((self.X - x) ** 2, axis=1)
        idx = jnp.argsort(d2)[:self.k]

        Xn = self.X[idx]
        Phin = self.Phi[idx]

        K = self._kernel(x, Xn)

        return jnp.sum(K[:, None] * Phin, axis=0) / (self.lam + 1e-12)


    def distance_grad_and_norm(self, x, y):
        eps = 1e-12

        # nearest neighbors of x
        d2_all = jnp.sum((self.X - x) ** 2, axis=1)
        _, idx = jax.lax.top_k(-d2_all, self.k)

        Xn = self.X[idx]  # (k, d)
        Phin = self.Phi[idx]
        Psin = self.Psi[idx]  # raw eigenvectors, (k, m)

        diff = Xn - x[None, :]  # (k, d)
        dist2 = jnp.sum(diff ** 2, axis=1)

        K = jnp.exp(-dist2 / (2.0 * self.sigma ** 2))  # (k,)

        inv_lam = 1.0 / (self.lam + eps)  # (m,)

        # scaled Nyström embedding of x
        psi_x = jnp.sum(K[:, None] * Phin, axis=0) * inv_lam

        # get scaled embedding of y
        matches = jnp.all(self.x_to_id == y, axis=1)
        idx = jnp.argmax(matches)
        psi_y = self.Psi[idx]

        # residual in scaled spectral space
        r = psi_x - psi_y

        d2_spec = jnp.dot(r, r) + eps
        d_spec = jnp.sqrt(d2_spec)

        # inner neighbor coefficient:
        # A[a] = sum_l r_l * Phi_{a,l} / lambda_l
        A = Psin @ (r * inv_lam)  # (k,)

        # numerator of grad_d
        q = jnp.sum((K * A)[:, None] * diff, axis=0)  # (d,)

        # gradient of unsquared distance
        grad_d = q / (self.sigma ** 2 * d_spec)

        # squared Euclidean norm
        grad_d_norm_sq = jnp.dot(grad_d, grad_d) + eps

        return d_spec, grad_d, grad_d_norm_sq

    def distance_grad_and_norm_batch(self, x, y):
        return jax.vmap(self.distance_grad_and_norm)(x, y)
