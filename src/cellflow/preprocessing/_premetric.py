import numpy as np
import jax.numpy as jnp
from sklearn.neighbors import NearestNeighbors
from scipy.sparse import coo_matrix, diags
from scipy.sparse.linalg import lobpcg


# ---------------------------
# kNN graph construction
# ---------------------------
def build_knn_graph(X, k=30, sigma=None):
    n = X.shape[0]
    nn = NearestNeighbors(n_neighbors=k).fit(X)
    dist, ind = nn.kneighbors(X)

    rows = np.repeat(np.arange(n), k)
    cols = ind.flatten()

    if sigma is None:
        sigma = np.median(dist)

    # NOTE: consistent Gaussian kernel convention
    weights = np.exp(-(dist**2) / (2 * sigma**2)).flatten()

    W = coo_matrix((weights, (rows, cols)), shape=(n, n))
    W = 0.5 * (W + W.T)
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
    def __init__(self, X_train, Phi_train, eigenvalues, sigma, k=10):
        """
        Phi_train must be UNWEIGHTED eigenvectors (NOT Psi)
        """
        self.X = jnp.array(X_train)
        self.Phi = jnp.array(Phi_train)
        self.lam = jnp.array(eigenvalues)
        self.sigma = sigma
        self.k = k

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
        # Compute k-NN ids
        d2 = jnp.sum((self.X - x) ** 2, axis=1)

        idx = jnp.argsort(d2)[:self.k]

        Xn = self.X[idx]
        Phin = self.Phi[idx]

        K = self._kernel(x, Xn)

        m = self.Phi.shape[1]
        phi_x = jnp.zeros(m)

        for i in range(m):
            lam_i = self.lam[i] + 1e-12
            phi_x.at[i].set((1.0 / lam_i) * jnp.sum(K * Phin[:, i]))

        return phi_x


    # ---------------------------
    # Spectral distance
    # ---------------------------
    def spectral_distance(self, x, y):
        phi_x = self.embed_point(x)
        phi_y = self.embed_point(y)
        return jnp.sum((phi_x - phi_y)**2)


    # =====================================================
    # GRADIENT w.r.t. x
    # =====================================================
    def spectral_distance_grad(self, x, y):

        phi_x = self.embed_point(x)
        phi_y = self.embed_point(y)

        # Compute k-NN ids
        d2 = jnp.sum((self.X - x) ** 2, axis=1)

        idx = jnp.argsort(d2)[:self.k]

        Xn = self.X[idx]
        Phin = self.Phi[idx]

        diff = x[None, :] - Xn
        dist2 = jnp.sum(diff**2, axis=1)

        K = jnp.exp(-dist2 / (2 * self.sigma**2))

        grad = jnp.zeros_like(x)

        # precompute kernel-weighted directional sums
        for i in range(self.Phi.shape[1]):

            lam_i = self.lam[i] + 1e-12
            coeff = 2.0 * (phi_x[i] - phi_y[i]) / lam_i

            # ∑ K(x,xj) φ_i(xj) (xj - x)
            s = jnp.sum(
                (K * Phin[:, i])[:, None] * (Xn - x[None, :]),
                axis=0
            )

            grad = grad + coeff * s

        grad = grad * 1.0 / (self.sigma**2)

        return grad

    def g_norm_fast(self, x, y):
        """
        Fast approximation of ||∇d||_g^2 using kernel moments.
        """

        # --------------------------------------------------
        # 1. nearest neighbors
        # --------------------------------------------------
        # Compute k-NN ids
        d2 = jnp.sum((self.X - x) ** 2, axis=1)

        idx = jnp.argsort(d2)[:self.k]

        Xn = self.X[idx]
        Phin = self.Phi[idx]

        diff = Xn - x[None, :]
        dist2 = jnp.sum(diff ** 2, axis=1)

        K = jnp.exp(-dist2 / (2 * self.sigma ** 2))

        m = self.Phi.shape[1]
        d = x.shape[0]

        phi_x = jnp.zeros(m)
        M = jnp.zeros((m, d))  # spectral × spatial moment

        # --------------------------------------------------
        # 2. compute moments
        # --------------------------------------------------
        for i in range(m):
            lam = self.lam[i] + 1e-12

            w = K * Phin[:, i] / lam

            phi_x.at[i].set(jnp.sum(w))

            M.at[i].set(jnp.sum(w[:, None] * Xn, axis=0))

        # --------------------------------------------------
        # 3. embedding residual
        # --------------------------------------------------
        phi_y = self.embed_point(y)

        r = phi_x - phi_y  # if y already in embedding space
        d2 = jnp.dot(r, r) + 1e-12

        # --------------------------------------------------
        # 4. tangent projection in moment form
        # --------------------------------------------------
        v = jnp.zeros(d)

        for i in range(m):
            v = v + r[i] * (M[i] - phi_x[i] * x)

        return jnp.dot(v, v) / d2
