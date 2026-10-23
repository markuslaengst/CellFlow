import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy.sparse.csgraph import connected_components

from cellflow.preprocessing._kernels import SymmetricBandwidthKernel
from cellflow.preprocessing._premetric import (
    SpectralNystroem,
    normalized_laplacian,
    unnormalized_laplacian,
)


PROFILES = ["fixed_gaussian", "gaussian", "polynomial", "bump", "indicator"]


@pytest.fixture
def training_points():
    with jax.enable_x64():
        yield np.random.default_rng(817).uniform(-1.0, 1.0, size=(36, 2))


def fit_extension(training_points, profile, laplacian_type, **kernel_options):
    kernel = SymmetricBandwidthKernel(
        training_points, profile=profile, density_k=8, delta=2.5, **kernel_options
    )
    weights = kernel.graph(include_self=laplacian_type == "normalized")
    laplacian = (
        normalized_laplacian(weights) if laplacian_type == "normalized" else unnormalized_laplacian(weights)
    )
    eigenvalues, eigenvectors = np.linalg.eigh(laplacian.toarray())
    eigenvalues = eigenvalues[1:6]
    eigenvectors = eigenvectors[:, 1:6]
    spectral_scale = np.linspace(0.5, 1.5, len(eigenvalues))
    extension = SpectralNystroem(
        training_points,
        eigenvectors,
        eigenvalues,
        sigma=None,
        psi=eigenvectors * spectral_scale,
        x_to_id=training_points,
        graph=weights,
        kernel=kernel,
        laplacian_type=laplacian_type,
        spectral_scale=spectral_scale,
    )
    return kernel, weights, extension


@pytest.mark.parametrize("profile", PROFILES)
def test_kernel_graph_is_symmetric_and_has_full_cknn_support(training_points, profile):
    kernel, weights, _ = fit_extension(training_points, profile, "unnormalized")
    pair_distances_sq = np.sum((training_points[:, None] - training_points[None, :]) ** 2, axis=-1)
    local_scales = np.asarray(kernel.local_scales)
    expected_support = pair_distances_sq < kernel.delta**2 * local_scales[:, None] * local_scales[None, :]
    np.fill_diagonal(expected_support, False)

    actual_support = weights.toarray() > 0
    assert not np.any(actual_support & ~expected_support)
    if profile == "bump":
        scaled_distances_sq = pair_distances_sq / (kernel.delta**2 * local_scales[:, None] * local_scales[None, :])
        missing_distances_sq = scaled_distances_sq[expected_support & ~actual_support]
        assert np.all(1.0 / (1.0 - missing_distances_sq) - 1.0 > 745.0)
    else:
        np.testing.assert_array_equal(actual_support, expected_support)
    assert (weights - weights.T).nnz == 0
    assert np.all(weights.diagonal() == 0)
    assert connected_components(weights, directed=False)[0] == 1
    assert np.max(np.diff(weights.indptr)) > kernel.density_k


@pytest.mark.parametrize("profile", PROFILES)
def test_kernel_gradients_include_query_bandwidth_derivative(training_points, profile):
    kernel = SymmetricBandwidthKernel(training_points, profile=profile, density_k=8, delta=1.7)
    query = jnp.array([0.123, -0.231])
    _, gradients = kernel.values_and_gradients(query)
    autodiff = jax.jacfwd(kernel)(query)
    finite_difference = np.column_stack(
        [
            (np.asarray(kernel(query + direction)) - np.asarray(kernel(query - direction))) / 2e-6
            for direction in np.eye(2) * 1e-6
        ]
    )

    np.testing.assert_allclose(gradients, autodiff, rtol=1e-9, atol=1e-9)
    np.testing.assert_allclose(gradients, finite_difference, rtol=1e-5, atol=1e-7)


@pytest.mark.parametrize("profile", PROFILES)
def test_small_unit_rescaling_preserves_cknn_affinities(training_points, profile):
    with jax.enable_x64(False):
        kernel = SymmetricBandwidthKernel(training_points, profile=profile, density_k=8, delta=1.7)
        unit_scale = 1e-8
        rescaled_kernel = SymmetricBandwidthKernel(
            training_points * unit_scale, profile=profile, density_k=8, delta=1.7
        )
        query = jnp.array([0.123, -0.231])
        values, gradients = kernel.values_and_gradients(query)
        rescaled_values, rescaled_gradients = rescaled_kernel.values_and_gradients(query * unit_scale)

        np.testing.assert_allclose(values, rescaled_values, rtol=1e-5, atol=1e-6)
        np.testing.assert_allclose(gradients, rescaled_gradients * unit_scale, rtol=1e-4, atol=1e-5)


@pytest.mark.parametrize("profile", PROFILES)
@pytest.mark.parametrize("laplacian_type", ["normalized", "unnormalized"])
def test_continuous_nystroem_reproduces_training_eigenvectors(training_points, profile, laplacian_type):
    _, _, extension = fit_extension(training_points, profile, laplacian_type)
    reconstructed = jax.vmap(extension.extend_point)(jnp.asarray(training_points))

    np.testing.assert_allclose(reconstructed, extension.Psi, rtol=1e-8, atol=1e-8)
    training_query = jnp.asarray(training_points[7])
    np.testing.assert_allclose(
        jax.jacfwd(extension.embed_point)(training_query),
        jax.jacfwd(extension.extend_point)(training_query),
        rtol=1e-9,
        atol=1e-9,
    )


@pytest.mark.parametrize("profile", PROFILES)
@pytest.mark.parametrize("laplacian_type", ["normalized", "unnormalized"])
def test_spectral_distance_gradient_matches_autodiff_and_finite_differences(
    training_points, profile, laplacian_type
):
    _, _, extension = fit_extension(training_points, profile, laplacian_type)
    query = jnp.array([0.123, -0.231])
    target = jnp.array([-0.263, 0.317])

    def distance(point):
        residual = extension.extend_point(point) - extension.extend_point(target)
        return jnp.sqrt(jnp.dot(residual, residual) + extension.eps)

    actual_distance, actual_gradient, actual_norm_sq = extension.distance_grad_and_norm(query, target)
    finite_difference = np.array(
        [
            (float(distance(query + direction)) - float(distance(query - direction))) / 2e-6
            for direction in np.eye(2) * 1e-6
        ]
    )

    np.testing.assert_allclose(actual_distance, distance(query), rtol=1e-10)
    np.testing.assert_allclose(actual_gradient, jax.grad(distance)(query), rtol=1e-8, atol=1e-9)
    np.testing.assert_allclose(actual_gradient, finite_difference, rtol=1e-5, atol=1e-7)
    np.testing.assert_allclose(actual_norm_sq, np.sum(np.asarray(actual_gradient) ** 2) + extension.eps)


@pytest.mark.parametrize("profile", ["polynomial", "bump", "indicator"])
def test_unsupported_queries_and_coincident_points_have_finite_gradients(training_points, profile):
    _, _, extension = fit_extension(training_points, profile, "normalized")
    query = jnp.array([1000.0, 1000.0])
    target = jnp.array([0.123, -0.231])
    distance, gradient, norm_sq = jax.jit(extension.distance_grad_and_norm)(query, target)

    assert np.isfinite(distance)
    assert np.all(np.isfinite(gradient))
    assert np.isfinite(norm_sq)
    np.testing.assert_array_equal(gradient, [0.0, 0.0])
    same_distance, same_gradient, _ = extension.distance_grad_and_norm(target, target)
    assert same_distance == 0.0
    np.testing.assert_array_equal(same_gradient, [0.0, 0.0])


@pytest.mark.parametrize("profile", ["polynomial", "bump"])
def test_compact_profiles_and_derivatives_vanish_at_cutoff(training_points, profile):
    kernel = SymmetricBandwidthKernel(training_points, profile=profile, density_k=8)
    arguments = jnp.array([1.0 - 1e-8, 1.0, 1.0 + 1e-8])
    values, derivatives = kernel._profile_values_and_derivatives(arguments)
    values = jnp.where(arguments < 1.0, values, 0.0)
    derivatives = jnp.where(arguments < 1.0, derivatives, 0.0)

    assert np.all(np.isfinite(values))
    assert np.all(np.isfinite(derivatives))
    assert np.max(np.abs(values)) < 1e-20
    assert np.max(np.abs(derivatives)) < 1e-20


def test_untruncated_gaussian_and_default_fixed_bandwidth_extension(training_points):
    kernel, weights, extension = fit_extension(
        training_points, "gaussian", "normalized", compact_support=False
    )
    assert weights.nnz == len(training_points) ** 2
    assert np.all(np.isfinite(jax.jit(extension.extend_point)(jnp.array([0.2, 0.1]))))

    fixed_extension = SpectralNystroem(
        training_points,
        np.asarray(extension.Phi),
        np.asarray(extension.lam),
        sigma=kernel.sigma,
        psi=np.asarray(extension.Psi),
        x_to_id=training_points,
        k=8,
        spectral_scale=np.asarray(extension.spectral_scale),
    )
    query = jnp.array([0.123, -0.231])
    target = jnp.array([-0.263, 0.317])

    def distance(point):
        residual = fixed_extension.extend_point(point) - fixed_extension.extend_point(target)
        return jnp.sqrt(jnp.dot(residual, residual) + fixed_extension.eps)

    np.testing.assert_allclose(
        fixed_extension.distance_grad_and_norm(query, target)[1], jax.grad(distance)(query), rtol=1e-8, atol=1e-9
    )


def test_normalized_kernel_extension_rejects_inconsistent_self_affinities(training_points):
    kernel, _, extension = fit_extension(training_points, "gaussian", "normalized")

    with pytest.raises(ValueError, match="include_self=True"):
        SpectralNystroem(
            training_points,
            np.asarray(extension.Phi),
            np.asarray(extension.lam),
            sigma=None,
            psi=np.asarray(extension.Psi),
            x_to_id=training_points,
            graph=kernel.graph(include_self=False),
            kernel=kernel,
        )


@pytest.mark.parametrize(
    "options",
    [
        {"density_k": 0},
        {"density_k": 100},
        {"delta": 0.0},
        {"delta": np.nan},
        {"profile": "unknown"},
        {"profile": "bump", "compact_support": False},
        {"sigma": -1.0},
    ],
)
def test_invalid_kernel_parameters_are_rejected(training_points, options):
    with pytest.raises(ValueError):
        SymmetricBandwidthKernel(training_points, **options)
