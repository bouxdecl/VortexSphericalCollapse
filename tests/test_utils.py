import os
import numpy as np
import pytest

from athena_collapse_analysis.utils import (
    pressure_from_conservatives,
    collapse_param_decomposition,
)


# ======================================================================
# Tests: pressure_from_conservatives
# ======================================================================

def test_pressure_from_conservatives():
    rho = np.array([1.0, 2.0, 3.0])
    Etot = np.array([10.0, 12.0, 14.0])
    v1 = np.array([1.0, 0.0, 1.0])
    v2 = np.zeros(3)
    v3 = np.zeros(3)

    p = pressure_from_conservatives(rho, Etot, v1, v2, v3, gamma=5/3)

    # Manual computation
    KE = 0.5 * rho * v1**2
    e = Etot - KE
    expected = (5/3 - 1) * e

    assert np.allclose(p, expected)


# ======================================================================
# Tests: collapse_param_decomposition
# ======================================================================

def test_collapse_param_decomposition_valid():
    R = np.array([1.0, 2.0, 4.0])
    Lz = np.array([2.0, 4.0, 8.0])

    S, alpha = collapse_param_decomposition(R, Lz, R_0=R[0], Lz_0=Lz[0])

    assert S.shape == R.shape
    assert alpha.shape == R.shape
    assert np.allclose(
        S, ( (R/R[0])**2 * Lz / Lz[0]) ** (1/3)
    )
    assert np.allclose(
        alpha, ( (R / R[0]) / (Lz / Lz[0]) ) **(1/3)
    )


def test_collapse_param_decomposition_shape_error():
    R = np.array([1, 2])
    Lz = np.array([1, 2, 3])
    with pytest.raises(ValueError):
        collapse_param_decomposition(R, Lz)

        
def test_collapse_param_decomposition_nonpositive_error():
    R = np.array([1, -2, 3])
    Lz = np.array([1, 2, 3])
    with pytest.raises(ValueError):
        collapse_param_decomposition(R, Lz)