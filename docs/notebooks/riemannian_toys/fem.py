"""Linear finite-element spectral distances on planar triangle meshes: the recipe of the RFM reference code.

Chen & Lipman (ICLR 2024), ``manifm/manifolds/mesh.py`` (commit ``b6ac1e9``):

1. ``L = -igl.cotmatrix(V, F)`` (cotangent stiffness) and ``M = igl.massmatrix(V, F, VORONOI)`` (lumped
   mixed-Voronoi areas, Meyer et al. 2003);
2. the ``n_modes + 1`` smallest eigenpairs of ``L phi = lambda M phi`` from ``eigsh(L, k, M, sigma=0, which="LM")``,
   constant mode dropped; the eigenvectors are M-orthonormal (the discrete ``int phi_i phi_j dA = delta_ij``);
3. eigenfunction values at a query point by barycentric interpolation in the closest triangle;
4. the biharmonic distance ``sqrt(sum_i ((phi_i(x) - phi_i(y)) / lambda_i)^2)`` and its gradient by autodiff.

libigl is not available in the container, so :func:`cotangent_laplacian` implements both matrices with
libigl's formulas. :class:`FEMPremetric` exposes the interface of
:class:`~cellflow.preprocessing._premetric.SpectralNystroem` that RFM training, the interpolators and
the toy helpers use (``extend_point``, ``distance_grad_and_norm(_batch)``, ``X``, ``Psi``, ``lam``, ``eps``).
"""

import time
from collections.abc import Callable
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np
from scipy.sparse import coo_matrix, csr_matrix, diags
from scipy.sparse.csgraph import connected_components
from scipy.sparse.linalg import eigsh
from scipy.spatial import Delaunay

__all__ = [
    "FEMFit",
    "FEMPremetric",
    "brute_force_locator",
    "cotangent_laplacian",
    "delaunay_domain_mesh",
    "fit_fem",
    "mesh_diagnostics",
    "strip_locator",
    "strip_mesh",
]


def _cross2(a, b):
    return a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0]


# Matrices ------------------------------------------------------------------------------------------
def cotangent_laplacian(vertices, faces):
    """``(L, M)``: ``-igl.cotmatrix`` (PSD stiffness) and ``igl.massmatrix(..., MASSMATRIX_TYPE_VORONOI)``.

    ``L_ij = -(cot a_ij + cot b_ij) / 2`` over the angles opposite edge ``ij``, ``L_ii = -sum_j L_ij``.
    ``M`` is diagonal: per triangle, each corner receives its Voronoi area if the triangle is not obtuse;
    otherwise the obtuse corner receives half and the others a quarter of the triangle area.
    """
    v = np.asarray(vertices, dtype=np.float64)
    f = np.asarray(faces, dtype=np.int64)
    p = v[f]  # (F, 3, 2)
    n = len(v)
    rows, cols, weights = [], [], []
    for i in range(3):
        j, k = (i + 1) % 3, (i + 2) % 3
        e1, e2 = p[:, j] - p[:, i], p[:, k] - p[:, i]
        cot = np.sum(e1 * e2, axis=1) / np.abs(_cross2(e1, e2))  # angle at corner i weights edge (j, k)
        rows += [f[:, j], f[:, k]]
        cols += [f[:, k], f[:, j]]
        weights += [0.5 * cot, 0.5 * cot]
    C = coo_matrix((np.concatenate(weights), (np.concatenate(rows), np.concatenate(cols))), shape=(n, n)).tocsr()
    L = diags(np.asarray(C.sum(axis=1)).ravel()) - C

    # Mixed Voronoi areas, following libigl's massmatrix_intrinsic.
    lengths = np.stack([np.linalg.norm(p[:, (i + 1) % 3] - p[:, (i + 2) % 3], axis=1) for i in range(3)], axis=1)
    double_area = np.abs(_cross2(p[:, 1] - p[:, 0], p[:, 2] - p[:, 0]))
    l0, l1, l2 = lengths.T
    cosines = np.stack([
        (l2**2 + l1**2 - l0**2) / (2 * l1 * l2),
        (l0**2 + l2**2 - l1**2) / (2 * l2 * l0),
        (l1**2 + l0**2 - l2**2) / (2 * l0 * l1),
    ], axis=1)
    barycentric = cosines * lengths
    partial = barycentric / barycentric.sum(axis=1, keepdims=True) * (0.5 * double_area)[:, None]
    quads = 0.5 * np.stack([partial[:, 1] + partial[:, 2], partial[:, 2] + partial[:, 0],
                            partial[:, 0] + partial[:, 1]], axis=1)
    for corner in range(3):
        obtuse = cosines[:, corner] < 0
        share = np.full((obtuse.sum(), 3), 0.125)
        share[:, corner] = 0.25
        quads[obtuse] = share * double_area[obtuse, None]
    mass = np.bincount(f.ravel(), weights=quads.ravel(), minlength=n)
    return csr_matrix(L), diags(mass).tocsc()


# Meshes --------------------------------------------------------------------------------------------
def delaunay_domain_mesh(points, contains: Callable, max_edge: float, tol=1e-2):
    """Delaunay triangulation of ``points`` restricted to a (possibly non-convex) domain.

    A triangle is kept if all three edges are shorter than ``max_edge`` and its centroid and edge
    midpoints lie within ``tol`` of the domain (``contains(samples, tol)``). Unused points are dropped.

    Returns ``(vertices, faces, vertex_ids)`` where ``vertex_ids`` indexes the kept points.
    """
    points = np.asarray(points, dtype=np.float64)
    faces = Delaunay(points).simplices
    p = points[faces]
    edges = np.stack([p[:, 1] - p[:, 0], p[:, 2] - p[:, 1], p[:, 0] - p[:, 2]], axis=1)
    short = np.linalg.norm(edges, axis=-1).max(axis=1) < max_edge
    probes = np.concatenate([p.mean(axis=1, keepdims=True), 0.5 * (p + np.roll(p, -1, axis=1))], axis=1)
    inside = contains(probes.reshape(-1, 2), tol=tol).reshape(len(faces), 4).all(axis=1)
    faces = faces[short & inside]
    vertex_ids = np.unique(faces)
    remap = np.full(len(points), -1)
    remap[vertex_ids] = np.arange(len(vertex_ids))
    return points[vertex_ids], remap[faces], vertex_ids


def strip_mesh(roll, n_along, n_across):
    """Structured mesh of the swiss roll in strip coordinates (arc length along the centerline, offset).

    Vertex ``(i, j)`` sits at arc length ``i * L / n_along`` and offset ``-w + j * 2w / n_across``; every
    cell is split into two triangles along its shorter diagonal ``(i + 1, j)``–``(i, j + 1)``. The radial
    offset is not orthogonal to the spiral, so the cells are skewed; this diagonal keeps the triangles
    acute (a Delaunay mesh, non-negative cotangent weights). Returns ``(vertices, faces, grid)`` with
    ``grid`` for :func:`strip_locator`.
    """
    arc = np.linspace(0.0, roll.centerline_length, n_along + 1)
    offsets = np.linspace(-roll.half_width, roll.half_width, n_across + 1)
    theta = roll.theta_at_arc_length(arc)
    vertices = roll.strip_point(theta[:, None], offsets[None, :]).reshape(-1, 2)
    i, j = np.meshgrid(np.arange(n_along), np.arange(n_across), indexing="ij")
    v00 = (i * (n_across + 1) + j).ravel()
    v10, v01 = v00 + n_across + 1, v00 + 1
    v11 = v10 + 1
    faces = np.stack([np.stack([v00, v10, v01], 1), np.stack([v10, v11, v01], 1)], axis=1).reshape(-1, 3)
    return vertices, faces, {"arc": arc, "offsets": offsets, "n_along": n_along, "n_across": n_across}


def mesh_diagnostics(vertices, faces):
    """Euler characteristic, boundary loops, non-manifold boundary vertices and angle statistics."""
    f = np.asarray(faces)
    all_edges = np.sort(np.concatenate([f[:, [0, 1]], f[:, [1, 2]], f[:, [2, 0]]]), axis=1)
    edges, counts = np.unique(all_edges, axis=0, return_counts=True)
    boundary = edges[counts == 1]
    n = len(vertices)
    graph = coo_matrix((np.ones(len(boundary)), (boundary[:, 0], boundary[:, 1])), shape=(n, n))
    boundary_vertices = np.unique(boundary)
    degree = np.bincount(boundary.ravel(), minlength=n)
    n_loops = connected_components(graph, directed=False)[0] - (n - len(boundary_vertices))
    p = np.asarray(vertices)[f]
    angles = []
    for i in range(3):
        e1, e2 = p[:, (i + 1) % 3] - p[:, i], p[:, (i + 2) % 3] - p[:, i]
        angles.append(np.degrees(np.arctan2(np.abs(_cross2(e1, e2)), np.sum(e1 * e2, axis=1))))
    angles = np.stack(angles, 1)
    return {
        "vertices": n,
        "faces": len(f),
        "euler_characteristic": n - len(edges) + len(f),
        "boundary_loops": n_loops,
        "non_manifold_boundary_vertices": int(np.count_nonzero(degree[boundary_vertices] != 2)),
        "edges_in_more_than_two_faces": int(np.count_nonzero(counts > 2)),
        "min_angle_deg": float(angles.min()),
        "obtuse_face_fraction": float(np.mean(angles.max(axis=1) > 90.0 + 1e-9)),
        "area": float(0.5 * np.abs(_cross2(p[:, 1] - p[:, 0], p[:, 2] - p[:, 0])).sum()),
    }


# Point location (JAX) ------------------------------------------------------------------------------
def barycentric(p, a, b, c):
    """Barycentric coordinates of ``p`` in ``(a, b, c)``; ``manifm.manifolds.mesh.barycenter_coordinates``."""
    v0, v1, v2 = b - a, c - a, p - a
    d00, d01, d11 = jnp.sum(v0 * v0, -1), jnp.sum(v0 * v1, -1), jnp.sum(v1 * v1, -1)
    d20, d21 = jnp.sum(v2 * v0, -1), jnp.sum(v2 * v1, -1)
    denom = d00 * d11 - d01 * d01
    v = (d11 * d20 - d01 * d21) / denom
    w = (d00 * d21 - d01 * d20) / denom
    return jnp.stack([1.0 - v - w, v, w], axis=-1)


def brute_force_locator(vertices, faces):
    """Closest triangle by a search over all faces, as ``manifm``'s ``closest_point`` (2-D version)."""
    corners = jnp.asarray(np.asarray(vertices)[np.asarray(faces)])  # (F, 3, 2)
    a, b, c = corners[:, 0], corners[:, 1], corners[:, 2]

    def segment_sq(p, s0, s1):
        d = s1 - s0
        t = jnp.clip(jnp.sum((p - s0) * d, -1) / jnp.sum(d * d, -1), 0.0, 1.0)
        return jnp.sum((p - (s0 + t[:, None] * d)) ** 2, -1)

    def locate(x):
        inside = jnp.all(barycentric(x, a, b, c) >= 0.0, axis=-1)
        outside_sq = jnp.minimum(jnp.minimum(segment_sq(x, a, b), segment_sq(x, b, c)), segment_sq(x, c, a))
        return jnp.argmin(jnp.where(inside, 0.0, outside_sq))

    return locate


def strip_locator(roll, vertices, faces, grid):
    """O(1) triangle lookup for :func:`strip_mesh`: strip coordinates, then the best of the 3 x 3 nearby cells."""
    corners = jnp.asarray(np.asarray(vertices)[np.asarray(faces)])
    n_along, n_across = grid["n_along"], grid["n_across"]
    d_arc = float(grid["arc"][1] - grid["arc"][0])
    d_off = float(grid["offsets"][1] - grid["offsets"][0])
    dense_theta, dense_arc = jnp.asarray(roll._dense_theta), jnp.asarray(roll._dense_arc)
    shifts = jnp.array([(di, dj) for di in (-1, 0, 1) for dj in (-1, 0, 1)])

    def locate(x):
        _, theta, offset = roll._nearest(x, jnp)
        arc = jnp.interp(theta, dense_theta, dense_arc)
        i0 = jnp.floor(arc / d_arc).astype(jnp.int32)
        j0 = jnp.floor((offset + roll.half_width) / d_off).astype(jnp.int32)
        i = jnp.clip(i0 + shifts[:, 0], 0, n_along - 1)
        j = jnp.clip(j0 + shifts[:, 1], 0, n_across - 1)
        cells = i * n_across + j
        candidates = jnp.concatenate([2 * cells, 2 * cells + 1])
        tri = corners[candidates]
        score = jnp.min(barycentric(x, tri[:, 0], tri[:, 1], tri[:, 2]), axis=-1)
        return candidates[jnp.argmax(score)]

    return locate


# Premetric -----------------------------------------------------------------------------------------
class FEMPremetric:
    """Biharmonic distance from FEM eigenpairs, with barycentric evaluation and an autodiff gradient.

    ``Psi`` holds the biharmonic coordinates ``phi_i / lambda_i`` at the mesh vertices. Inside a triangle
    the coordinates are linear, so the distance gradient is constant per triangle and jumps across edges.
    """

    def __init__(self, vertices, faces, eigenvalues, eigenvectors, locate, eps=1e-12, embedding_eps=1e-8):
        self.X = jnp.asarray(vertices)
        self.faces = jnp.asarray(faces)
        self.lam = jnp.asarray(eigenvalues)
        self.Phi = jnp.asarray(eigenvectors)
        self.spectral_scale = 1.0 / (self.lam + embedding_eps)
        self.Psi = self.Phi * self.spectral_scale
        self.eps = eps
        self.locate = locate

    def face_and_barycentric(self, x):
        face = jax.lax.stop_gradient(self.locate(jax.lax.stop_gradient(x)))
        corners = self.X[self.faces[face]]
        return face, barycentric(x, corners[0], corners[1], corners[2])

    def extend_point(self, x):
        face, weights = self.face_and_barycentric(x)
        return weights @ self.Psi[self.faces[face]]

    def distance_grad_and_norm(self, x, y):
        psi_y = self.extend_point(y)

        def distance(point):
            residual = self.extend_point(point) - psi_y
            return jnp.sqrt(jnp.dot(residual, residual) + self.eps)

        d_spec, grad_d = jax.value_and_grad(distance)(x)
        grad_d_norm_sq = jnp.dot(grad_d, grad_d) + self.eps
        same_point = jnp.all(x == y)
        d_spec = jnp.where(same_point, 0.0, d_spec)
        grad_d = jnp.where(same_point, jnp.zeros_like(grad_d), grad_d)
        grad_d_norm_sq = jnp.where(same_point, self.eps, grad_d_norm_sq)
        return d_spec, grad_d, grad_d_norm_sq

    def distance_grad_and_norm_batch(self, x, y):
        return jax.vmap(self.distance_grad_and_norm)(x, y)


@dataclass
class FEMFit:
    model: FEMPremetric
    eigenvalues: np.ndarray
    eigenvectors: np.ndarray
    vertices: np.ndarray
    faces: np.ndarray
    diagnostics: dict


def fit_fem(vertices, faces, n_modes, locate_factory: Callable, sigma=0.0):
    """Assemble ``L`` and ``M``, solve as the RFM code does, and wrap the result in :class:`FEMPremetric`.

    ``locate_factory(vertices, faces)`` returns the point locator. ``sigma=0`` is the reference code's
    shift; if the factorization of the singular ``L`` fails, the solve falls back to ``sigma=-1e-8``.
    """
    started = time.perf_counter()
    L, M = cotangent_laplacian(vertices, faces)
    shift = sigma
    try:
        eigenvalues, eigenvectors = eigsh(L, n_modes + 1, M, sigma=shift, which="LM", maxiter=100000)
    except Exception:  # noqa: BLE001 - singular factorization at sigma=0
        shift = -1e-8
        eigenvalues, eigenvectors = eigsh(L, n_modes + 1, M, sigma=shift, which="LM", maxiter=100000)
    order = np.argsort(eigenvalues)
    eigenvalues, eigenvectors = eigenvalues[order], eigenvectors[:, order]
    solve_seconds = time.perf_counter() - started
    mass = M.diagonal()
    gram = eigenvectors.T @ (mass[:, None] * eigenvectors)
    diagnostics = {
        "eigsh_sigma": shift,
        "lambda_0": eigenvalues[0],
        "lambda_1": eigenvalues[1],
        "lambda_max": eigenvalues[-1],
        "max_M_orthonormality_error": np.abs(gram - np.eye(len(gram))).max(),
        "max_eigen_residual": np.max(np.linalg.norm(L @ eigenvectors - (M @ eigenvectors) * eigenvalues, axis=0)),
        "fit_seconds": solve_seconds,
    }
    eigenvalues, eigenvectors = eigenvalues[1:], eigenvectors[:, 1:]
    model = FEMPremetric(vertices, faces, eigenvalues, eigenvectors, locate_factory(vertices, faces))
    return FEMFit(model, eigenvalues, eigenvectors, np.asarray(vertices), np.asarray(faces), diagnostics)
