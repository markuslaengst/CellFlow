import numpy as np
from sklearn.neighbors import NearestNeighbors
from scipy.sparse import coo_matrix, diags
from scipy.sparse.linalg import lobpcg


# Chat-GPT snippets for knn, normalized laplacian, eigenvector and value iteration and the spectral embeddings
def build_knn_graph(X, k=30, sigma=None):
    n = X.shape[0]
    nn = NearestNeighbors(n_neighbors=k).fit(X)
    dist, ind = nn.kneighbors(X)

    rows = np.repeat(np.arange(n), k)
    cols = ind.flatten()

    if sigma is None:
        sigma = np.median(dist)

    weights = np.exp(-(dist**2) / (sigma**2)).flatten()

    W = coo_matrix((weights, (rows, cols)), shape=(n, n))
    W = 0.5 * (W + W.T)  # symmetrize
    return W.tocsr()


def normalized_laplacian(W):
    d = np.array(W.sum(axis=1)).flatten()
    D_inv_sqrt = diags(1.0 / np.sqrt(d + 1e-12))
    I = diags(np.ones(len(d)))

    L = I - D_inv_sqrt @ W @ D_inv_sqrt
    return L

def compute_eigens(L, k=50):
    n = L.shape[0]
    X = np.random.randn(n, k)

    eigvals, eigvecs = lobpcg(L, X, largest=False, maxiter=200)
    return eigvals, eigvecs

def spectral_embedding(eigvals, eigvecs, weight_type="biharmonic", eps=1e-8):
    if weight_type == "biharmonic":
        w = 1.0 / (eigvals + eps)**2
    elif weight_type == "diffusion":
        t = 1.0
        w = np.exp(-eigvals * t)
    else:
        w = np.ones_like(eigvals)

    Psi = eigvecs * np.sqrt(w)[None, :]
    return Psi

def spectral_distance(i, j, Psi, Psi_norm):
    return Psi_norm[i] + Psi_norm[j] - 2 * Psi[i] @ Psi[j]

class SpectralNyström:
    def __init__(self, X_train, Phi_train, eigenvalues, k=10, sigma=1.0):
        """
        X_train: (n, d)
        Phi_train: (n, m) spectral eigenvectors
        eigenvalues: (m,)
        k: number of nearest neighbors for extension
        sigma: RBF kernel width
        """
        self.X = X_train
        self.Phi = Phi_train
        self.lam = eigenvalues
        self.k = k
        self.sigma = sigma

        self.nn = NearestNeighbors(n_neighbors=k)
        self.nn.fit(X_train)

    def _kernel(self, x, X_neighbors):
        """
        Gaussian kernel k(x, x_j)
        """
        diff = X_neighbors - x[None, :]
        dist2 = np.sum(diff ** 2, axis=1)
        return np.exp(-dist2 / (2 * self.sigma ** 2))

    def embed_point(self, x):
        """
        Compute spectral embedding φ(x) for a new point.
        Returns: (m,) vector
        """

        # 1. find k nearest neighbors
        dist, idx = self.nn.kneighbors(x.reshape(1, -1))
        idx = idx[0]

        X_nn = self.X[idx]        # (k, d)
        Phi_nn = self.Phi[idx]    # (k, m)

        # 2. compute kernel weights
        w = self._kernel(x, X_nn)  # (k,)

        # 3. normalize weights (important in practice)
        w = w / (np.sum(w) + 1e-12)

        # 4. Nyström extension for each eigenfunction
        phi_x = np.zeros(self.Phi.shape[1])

        for i in range(self.Phi.shape[1]):
            lam_i = self.lam[i]

            # avoid divide-by-zero for constant eigenvector
            if lam_i < 1e-8:
                phi_x[i] = np.sum(w * Phi_nn[:, i])
            else:
                phi_x[i] = (1.0 / lam_i) * np.sum(w * Phi_nn[:, i])

        return phi_x
