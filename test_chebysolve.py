#!/usr/bin/env python

from numpy import einsum, float64
from numpy.linalg import eigvals, LinAlgError
from numpy.testing import assert_allclose
from numpy.random import default_rng

import pytest

from hypothesis import assume, given
from hypothesis.strategies import composite, floats, integers
from hypothesis.extra.numpy import arrays, array_shapes

import chebysolve


def iterate_vector_numpy(matrix, vector):
    '''
    Return the product ``matrix`` * ``matrix``^T * ``vector``.
    I.e. perform one Chebyshev iteration.

    This implementatation does not parallelise and gets very slow
    at large matrix sizes.
    '''

    # Expressed in indices: \sum_{j,k} A_ik A_jk y_j
    return einsum('ik,jk,j->i', matrix, matrix, vector)


@composite
def matrix_and_vector(
        draw,
        matrix_shape=array_shapes(
            min_dims=2, max_dims=2, min_side=1, max_side=10
        )
):
    '''
    Generate a matrix and vector with compatible shapes to multiply.
    '''
    actual_shape = draw(matrix_shape)
    matrix = draw(arrays(float64, actual_shape))
    vector = draw(arrays(float64, actual_shape[0]))
    return matrix, vector


@given(matrix_and_vector())
def test_numba_vs_numpy(matrix_vector):
    '''
    Test that our special matrix multiplier works properly.
    '''

    matrix, vector = matrix_vector

    result = chebysolve.iterate_vector(matrix, vector)
    assume((result < 1e10).all())
    assert_allclose(result, iterate_vector_numpy(matrix, vector), rtol=1e-6)
    assert_allclose(result, matrix @ matrix.T @ vector, rtol=1e-6)


def eigvals_works(matrix):
    '''
    Helper function to see whether eigvals() is able to calculate
    eigenvalues for ``matrix`` * ``matrix``^T.
    Returns True if so; False otherwise.
    '''

    try:
        eigvals(matrix @ matrix.T)
    except LinAlgError:
        return False
    else:
        return True


@given(arrays(
    float64,
    array_shapes(min_dims=2, max_dims=2, min_side=1, max_side=10),
    elements=floats(0, 1e6)
))
def test_eigenvalue(matrix):
    '''
    Test that the eigensolver does give an eigenvalue.
    '''

    assume((matrix @ matrix.T != 0).any())
    assume(eigvals_works(matrix))

    estimates, eigenvector, converged = chebysolve.largest_eigenpair(
        matrix, tolerance=1e-8
    )
    assert converged

    result = chebysolve.iterate_vector(matrix, eigenvector)
    assert_allclose(result, eigenvector * estimates[-1][0], rtol=1e-6)


@given(arrays(
    float64,
    array_shapes(min_dims=2, max_dims=2, min_side=1, max_side=10),
    elements=floats(0, 1e6)
))
def test_largest_eigenvalue(matrix):
    '''
    Independently verify that the eigenvalue is the largest.
    '''

    assume((matrix @ matrix.T != 0).any())
    assume(eigvals_works(matrix))

    eigenvalues = eigvals(matrix @ matrix.T)
    eigenvalue_estimates, _, converged = (
        chebysolve.largest_eigenpair(matrix, tolerance=1e-8)
    )
    assert converged

    assert_allclose(eigenvalue_estimates[-1][0], max(eigenvalues), rtol=1e-6)


@pytest.mark.parametrize(
    "long_axis_size, short_axis_size",
    [(10, 3), (100, 5), (1000, 10), (10_000, 25)]
)
def test_iteration_performance(benchmark, long_axis_size, short_axis_size):
    '''
    Test the performance of the custom matrix-vector multiplier.
    '''
    rng = default_rng()
    matrix = rng.integers(
        10, size=(long_axis_size, short_axis_size)
    ).astype('float64')
    vector = rng.random(long_axis_size, dtype='float64')

    chebysolve.iterate_vector(matrix, vector)
    benchmark(chebysolve.iterate_vector, matrix, vector)
