# Chebysolve - Chebyshev eigensolver

[![Tests](https://github.com/sa2c/chebysolve/actions/workflows/python-package-conda.yml/badge.svg)](https://github.com/sa2c/chebysolve/actions/workflows/python-package-conda.yml)

Chebysolve finds the largest eigenvalue for matrices of the form _A_ = _MM_<sup>T</sup>. This is particularly useful where the matrix _A_ is too large to fit into memory, but the matrix _M_ is not&mdash;i.e. where _M_ is very long but skinny.

Chebysolve currently only finds the largest eigenvalue and its associated eigenvector.

The matrix-vector multiplier is accelerated with Numba. Currently this is CPU-only, but this could in principle work on GPUs with small changes.

Tested using Hypothesis.

An example of usage can be found in `chebysolve_example.ipynb`.

<img src="https://tfw.gov.wales/sites/default/files/inline-images/ERDF2.jpg" alt="European Regional Development Fund logo" width="150">

This project was developed with support from the [Supercomputing Wales][scw] programme, which is part-funded by the [European Regional Development Fund][erdf] (ERDF) through the [Welsh Government][welsh-gov].

[erdf]: https://ec.europa.eu/regional_policy/en/funding/erdf/
[scw]: https://supercomputing.wales
[welsh-gov]: https://gov.wales
