"""Fit a continuous spectral (Nyström) premetric from a symmetric bandwidth kernel."""

import time
from dataclasses import dataclass
from typing import Literal

import jax
import jax.numpy as jnp
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from scipy.sparse.linalg import eigsh

from cellflow.preprocessing._kernels import SymmetricBandwidthKernel
from cellflow.preprocessing._premetric import (
    SpectralNystroem,
    normalized_laplacian,
    spectral_embedding_scale,
    unnormalized_laplacian,
)

__all__ = ["KernelNystroemFit", "fit_kernel_nystroem"]


@dataclass
class KernelNystroemFit:
    """Result of :func:`fit_kernel_nystroem`.

    Attributes
    ----------
    model
        The fitted continuous extension, usable as ``nystroem`` in RFM.
    kernel
        The fitted symmetric bandwidth kernel.
    graph
        The sparse affinity graph the Laplacian was built from.
    eigenvalues
        The non-trivial Laplacian eigenvalues, in increasing order.
    eigenvectors
        The matching eigenvectors, one column per eigenvalue.
    diagnostics
        Graph, eigen-solver and extension checks, see :func:`fit_kernel_nystroem`.
    """

    model: SpectralNystroem
    kernel: SymmetricBandwidthKernel
    graph: csr_matrix
    eigenvalues: np.ndarray
    eigenvectors: np.ndarray
    diagnostics: dict


def fit_kernel_nystroem(
    construction_points,
    profile: str,
    *,
    density_k: int,
    delta: float,
    n_modes: int,
    laplacian_type: Literal["unnormalized", "normalized"] = "unnormalized",
    spectral_weight_type: str = "biharmonic",
    n_reconstruction_checks: int = 20,
    eigen_seed: int = 0,
) -> KernelNystroemFit:
    """Fit a kernel graph, its Laplacian eigenpairs, and the continuous Nyström extension.

    The pipeline is: :class:`SymmetricBandwidthKernel` → sparse graph → Laplacian →
    the ``n_modes`` smallest non-trivial eigenpairs (shift-invert ``eigsh``) →
    spectral scales → :class:`SpectralNystroem` evaluated through the fitted kernel.

    Parameters
    ----------
    construction_points
        Points of shape ``(n_points, n_features)`` that define the graph.
    profile
        Kernel profile passed to :class:`SymmetricBandwidthKernel`.
    density_k
        Neighbor index of the local bandwidth.
    delta
        CkNN support scale.
    n_modes
        Number of non-trivial eigenpairs to keep.
    laplacian_type
        ``"unnormalized"`` (``L = D - W``) or ``"normalized"``.
    spectral_weight_type
        Spectral weighting passed to :func:`spectral_embedding_scale`.
    n_reconstruction_checks
        Number of evenly spaced construction points at which the extension must
        reproduce the fitted spectral coordinates.
    eigen_seed
        Seed of the eigen-solver start vector.

    Returns
    -------
    The fit. ``diagnostics`` contains ``edges`` (undirected, without self-loops),
    ``components``, ``min_degree``, ``median_degree``, ``lambda_2``,
    ``max_eigen_residual``, ``extension_relative_error`` and ``fit_seconds``.

    Raises
    ------
    ValueError
        If the graph is not connected.
    """
    started = time.perf_counter()
    construction_points = np.asarray(construction_points)
    kernel = SymmetricBandwidthKernel(construction_points, profile=profile, density_k=density_k, delta=delta)
    graph = kernel.graph(include_self=laplacian_type == "normalized").astype(np.float64)
    component_count = connected_components(graph, directed=False)[0]
    if component_count != 1:
        raise ValueError(f"{profile}: {component_count} components; increase density_k or delta.")
    laplacian = unnormalized_laplacian(graph) if laplacian_type == "unnormalized" else normalized_laplacian(graph)
    eigenvalues, eigenvectors = eigsh(
        laplacian,
        k=n_modes + 1,
        sigma=-1e-8,
        which="LM",
        tol=1e-9,
        v0=np.random.default_rng(eigen_seed).standard_normal(len(construction_points)),
    )
    order = np.argsort(eigenvalues)
    eigenvalues = eigenvalues[order][1:]
    eigenvectors = eigenvectors[:, order][:, 1:]
    scales = spectral_embedding_scale(eigenvalues, weight_type=spectral_weight_type)
    model = SpectralNystroem(
        construction_points,
        eigenvectors,
        eigenvalues,
        sigma=None,
        psi=eigenvectors * scales,
        x_to_id=construction_points,
        graph=graph,
        kernel=kernel,
        laplacian_type=laplacian_type,
        spectral_scale=scales,
        target_embedding="nystroem",
    )

    check_ids = np.linspace(0, len(construction_points) - 1, n_reconstruction_checks, dtype=int)
    reconstruction = np.asarray(jax.jit(jax.vmap(model.extend_point))(jnp.asarray(construction_points[check_ids])))
    expected = np.asarray(model.Psi)[check_ids]
    degrees = np.asarray(graph.sum(axis=1)).ravel()
    diagnostics = {
        "edges": (graph.nnz - np.count_nonzero(graph.diagonal())) // 2,
        "components": component_count,
        "min_degree": degrees.min(),
        "median_degree": np.median(degrees),
        "lambda_2": eigenvalues[0],
        "max_eigen_residual": np.max(np.linalg.norm(laplacian @ eigenvectors - eigenvectors * eigenvalues, axis=0)),
        "extension_relative_error": np.linalg.norm(reconstruction - expected) / max(np.linalg.norm(expected), 1e-12),
        "fit_seconds": time.perf_counter() - started,
    }
    return KernelNystroemFit(model, kernel, graph, eigenvalues, eigenvectors, diagnostics)
