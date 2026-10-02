import sys

import numpy as np


def test_python_312():
    assert sys.version_info[:2] == (3, 12)


def test_eigh_hermitian_real_eigenvalues():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(8, 8)) + 1j * rng.normal(size=(8, 8))
    C = (X + X.conj().T) / 2
    w, v = np.linalg.eigh(C)
    assert np.allclose(w.imag, 0)
    assert np.allclose(C @ v, v * w)
