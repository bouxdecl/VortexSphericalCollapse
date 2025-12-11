import os
import numpy as np
import pytest

from athena_collapse_analysis.io.ath_io import (
    get_hdf_files,
    open_hdf_files,
    open_hdf_files_with_collapse,
    get_collapse_profile,
    add_collapse_param,
)



# ======================================================================
# Tests: get_hdf_files
# ======================================================================

def test_get_hdf_files(sample_dir):
    files = get_hdf_files(sample_dir, verbose=False)
    assert len(files) > 0
    assert all(f.endswith(".athdf") for f in files)


# ======================================================================
# Tests: open_hdf_files
# ======================================================================

def test_open_hdf_files_basic(hdf_files):
    data = open_hdf_files(hdf_files, read_every=1, adia=True)

    # Required keys
    for key in ["x1", "x2", "x3", "time", "rho", "v1", "v2", "v3", "Etot"]:
        assert key in data

    # Dimensions consistent
    Nt = len(hdf_files)
    assert data["time"].shape == (Nt,)
    assert data["rho"].shape[0] == Nt
    assert data["v1"].shape == data["rho"].shape


def test_open_hdf_files_no_adiabatic(hdf_files):
    data = open_hdf_files(hdf_files, adia=False)
    assert "Etot" not in data or data["Etot"] is None


# ======================================================================
# Tests: get_collapse_profile
# ======================================================================

def test_get_collapse_profile_success(sample_dir):
    t, R, Lz = get_collapse_profile(sample_dir)

    assert t.ndim == 1
    assert R.shape == t.shape
    assert Lz.shape == t.shape
    assert np.all(R > 0)
    assert np.all(Lz > 0)


# ======================================================================
# Tests: add_collapse_param
# ======================================================================

def test_add_collapse_param_interpolation(hdf_files, sample_dir):
    # Load time series
    dic = open_hdf_files(hdf_files)

    # Load HST values
    t_hst, R_hst, Lz_hst = get_collapse_profile(sample_dir)

    dic2 = add_collapse_param(dic, t_hst, R_hst, Lz_hst)

    assert "Rglobal" in dic2 and "Lzglobal" in dic2
    t = dic["time"]
    assert np.allclose(dic2["Rglobal"], np.interp(t, t_hst, R_hst))
    assert np.allclose(dic2["Lzglobal"], np.interp(t, t_hst, Lz_hst))


def test_add_collapse_param_out_of_bounds():
    dic = {"time": np.array([-0.1, 0.5, 2.0])}  # -0.1 is out-of-bounds
    t_hst = np.array([0, 1, 2])
    R = np.array([1, 2, 3])
    Lz = np.array([10, 20, 30])

    with pytest.raises(ValueError, match="outside the .hst time range"):
        add_collapse_param(dic, t_hst, R, Lz)


# ======================================================================
# Tests: load_cons_with_collapse
# ======================================================================

def test_open_hdf_files_with_collapse(sample_dir, hdf_files):
    dic = open_hdf_files_with_collapse(sample_dir, hdf_files)
    assert "Rglobal" in dic and "Lzglobal" in dic
