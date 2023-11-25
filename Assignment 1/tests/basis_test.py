# /bin/env python3.8

import numpy as np
import pytest

from basis_expansion import (gaussian_basis_function, generate_data,
                             initialize_gaussian_parameters, xfrm_w_gaussian)

from linear import Linear


@pytest.mark.parametrize(
    "x, mu, sigma",
    [
        (0.0, 0.0, 1.0),  # Centered at Peak
        (1.0, 0.0, 1.0),  # 1 standard deviation away [right]
        (-1.0, 0.0, 1.0),  # 1 standard deviation away [left]
        (0.01, 0.0, 0.01),  # small sigma
        (0.0, 0.0, 100.0),  # large sigma
    ],
)
def test_non_negativity(x, mu, sigma):
    y = gaussian_basis_function(x, mu, sigma)
    assert y >= 0, (
        f"For x={x}, mu={mu}, and sigma={sigma} "
        f"expected non-negative output but got {y}"
    )


@pytest.mark.parametrize(
    "x, mu, sigma",
    [
        (0.0, 0.0, 1.0),  # Centered at Peak
        (0.1, 0.0, 1.0),  # Right of peak
        (-0.1, 0.0, 1.0),  # Left of peak
    ],
)
def test_mean(x, mu, sigma):
    y_at_mu = gaussian_basis_function(mu, mu, sigma)
    y = gaussian_basis_function(x, mu, sigma)
    assert (
        y <= y_at_mu
    ), f"For x={x}, mu={mu}, and sigma={sigma}, expected y ({y}) to be less than or equal to y_at_mu ({y_at_mu})"


@pytest.mark.parametrize(
    "x, mu, sigma",
    [(0.5, 0.0, 1.0), (0.1, 0.5, 0.5), (0.2, -0.5, 2.0), (0.3, 1.0, 0.1)],
)
def test_symmetry(x, mu, sigma):
    y_positive = gaussian_basis_function(mu + x, mu, sigma)
    y_negative = gaussian_basis_function(mu - x, mu, sigma)
    assert np.isclose(
        y_positive, y_negative, atol=1e-5
    ), f"For mu={mu}, sigma={sigma}, and x={x}, expected symmetry but got y_pos={y_positive} and y_neg={y_negative}"


def test_expected_output_shape():
    x, _, _ = generate_data(num_samples=100)
    mu, sigma = initialize_gaussian_parameters(x, num_gaussians=7)
    x_transformed = xfrm_w_gaussian(x, mu, sigma)

    linear_model = Linear(num_inputs=7, num_outputs=1)
    y_hat = linear_model(x_transformed)

    assert y_hat.shape == x.shape, f"Expected shape {x.shape} but got {y_hat.shape}"


def test_gaussian_parameters():
    x, y, _ = generate_data(num_samples=100)
    mu, sigma = initialize_gaussian_parameters(x, num_gaussians=7)

    x_transformed = xfrm_w_gaussian(x, mu, sigma)
    linear_model = Linear(num_inputs=7, num_outputs=1)
    y_hat = linear_model(x_transformed)

    assert all(
        0 <= m <= 1 for m in mu.numpy()
    ), f"Some mean values are out of bounds: {mu.numpy()}"
    assert all(
        0.01 <= s <= 2 for s in sigma.numpy()
    ), f"Some sigma values are out of bounds: {sigma.numpy()}"
