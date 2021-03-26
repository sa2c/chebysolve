#!/usr/bin/env python

"""
Chebysolve: Chebyshev eigensolver for the largest eiganvalue
and its eigenvector.

Author: Ed Bennett, 2020-03-26
Contact: e.j.bennett@swansea.ac.uk
"""

from numpy import complexfloating, floating, integer, issubdtype, zeros_like
from numpy.linalg import norm
from numpy.random import default_rng

from numba import jit, prange, float64, complex128


@jit(
    [dtype[:](dtype[:, :], dtype[:]) for dtype in (float64, complex128)],
    nopython=True,
    parallel=True
)
def iterate_vector(matrix, vector):
    '''
    Return the product ``matrix`` * ``matrix``^T * ``vector``.
    I.e. perform one Chebyshev iteration.
    '''

    result = zeros_like(vector)
    for i in prange(len(vector)):
        for j in range(len(vector)):
            for k in range(matrix.shape[1]):
                result[i] += matrix[i, k] * matrix[j, k] * vector[j]

    return result


def largest_eigenpair(
        matrix, initial_vector=None, tolerance=1e-6, max_iterations=100
):
    """
    Uses Chebyshev iteration to find the largest eigenvalue of the matrix
    M = ``matrix`` * ``matrix``^T. Starts from ``initial_vector``; if none
    is supplied, then a random vector is generated.

    Parameters:
        ``matrix``: a matrix, whose product with its transpose will be the
            target for eigensolution
        ``initial_vector``: an optional starting vector for the
            Chebyshev iteration
        ``tolerance``: the maximum standard deviation of estimates of
            the largest eigenvalue that will be considered to be converged.
        ``max_iterations``: the maximum number of iterations to perform
            before giving up

    Returns:
        Estimates of the largest eigenvalue after each iteration, and
        the standard deviation of the estimates, as a list of tuples,
        one tuple per iteration.
        The final estimate of the eigenvector.
        A boolean value indicating whether the algorithm converged.
    """

    if initial_vector is None:
        rng = default_rng(12345)
        dtype = matrix.dtype
        length = matrix.shape[0]

        if issubdtype(dtype, integer):
            initial_vector = rng.integers(255, size=length).astype(dtype)
        elif issubdtype(dtype, floating):
            initial_vector = rng.random(length).astype(dtype)
        elif issubdtype(dtype, complexfloating):
            initial_vector = (
                rng.random(length) + rng.random(length) * 1J
            ).astype(dtype)
        else:
            raise ValueError(
                "Unable to determine data type for initial vector. "
                "Please supply a starting vector to use."
            )

    # Ensure starting vector is normalised
    initial_vector = initial_vector / norm(initial_vector)

    eigenvalue_estimates = []

    for _ in range(max_iterations):
        new_vector = iterate_vector(matrix, initial_vector)

        # To estimate eigenvalue, take ratio of new to old eigenvector
        # ignoring zeroes
        mask = (new_vector != 0) & (initial_vector != 0)
        iter_eigenvalue_estimates = new_vector[mask] / initial_vector[mask]
        eigenvalue_estimate = iter_eigenvalue_estimates.mean()
        eigenvalue_error_estimate = iter_eigenvalue_estimates.std()

        eigenvalue_estimates.append(
            (eigenvalue_estimate, eigenvalue_error_estimate)
        )

        if eigenvalue_error_estimate / eigenvalue_estimate < tolerance:
            return eigenvalue_estimates, new_vector, True

        # Normalise to try to prevent overflows
        initial_vector = new_vector / norm(new_vector)
    else:
        return eigenvalue_estimates, new_vector, False
