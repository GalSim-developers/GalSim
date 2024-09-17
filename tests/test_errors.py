# Copyright (c) 2012-2023 by the GalSim developers team on GitHub
# https://github.com/GalSim-developers
#
# This file is part of GalSim: The modular galaxy image simulation toolkit.
# https://github.com/GalSim-developers/GalSim
#
# GalSim is free software: redistribution and use in source and binary forms,
# with or without modification, are permitted provided that the following
# conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions, and the disclaimer given in the accompanying LICENSE
#    file.
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions, and the disclaimer given in the documentation
#    and/or other materials provided with the distribution.
#

import galsim
from galsim_test_helpers import *

@timer
def test_galsim_error():
    """Test basic usage of GalSimError
    """
    err = galsim.GalSimError("Test")
    print('str = ',str(err))
    print('repr = ',repr(err))
    assert str(err) == "Test"
    assert isinstance(err, RuntimeError)
    check_pickle(err)


@timer
def test_galsim_value_error():
    """Test basic usage of GalSimValueError
    """
    value = 2.3
    err = galsim.GalSimValueError("Test", value)
    print('str = ',str(err))
    print('repr = ',repr(err))
    assert str(err) == "Test Value 2.3"
    assert err.value == value
    assert err.allowed_values == None
    assert isinstance(err, galsim.GalSimError)
    assert isinstance(err, ValueError)
    check_pickle(err)

    err = galsim.GalSimValueError("Test", value, (0,1,2))
    print('str = ',str(err))
    print('repr = ',repr(err))
    assert str(err) == "Test Value 2.3 not in (0, 1, 2)"
    assert err.value == value
    assert err.allowed_values == (0,1,2)
    assert isinstance(err, galsim.GalSimError)
    assert isinstance(err, ValueError)
    check_pickle(err)


@timer
def test_galsim_key_error():
    """Test basic usage of GalSimKeyError
    """
    key = 'foo'
    err = galsim.GalSimKeyError("Test", key)
    print('str = ',str(err))
    print('repr = ',repr(err))
    assert str(err) == "Test Key foo"
    assert err.key == key
    assert isinstance(err, galsim.GalSimError)
    assert isinstance(err, KeyError)
    check_pickle(err)


@timer
def test_galsim_index_error():
    """Test basic usage of GalSimIndexError
    """
    index = 3
    err = galsim.GalSimIndexError("Test", index)
    print('str = ',str(err))
    print('repr = ',repr(err))
    assert str(err) == "Test Index 3"
    assert err.index == index
    assert isinstance(err, galsim.GalSimError)
    assert isinstance(err, IndexError)
    check_pickle(err)


@timer
def test_galsim_range_error():
    """Test basic usage of GalSimRangeError
    """
    value = 2.3
    err = galsim.GalSimRangeError("Test", value, 0, 1)
    print('str = ',str(err))
    print('repr = ',repr(err))
    assert str(err) == "Test Value 2.3 not in range [0, 1]"
    assert err.value == value
    assert err.min == 0
    assert err.max == 1
    assert isinstance(err, galsim.GalSimError)
    assert isinstance(err, ValueError)
    check_pickle(err)

    err = galsim.GalSimRangeError("Test", value, 10)
    print('str = ',str(err))
    print('repr = ',repr(err))
    assert str(err) == "Test Value 2.3 not in range [10, None]"
    assert err.value == value
    assert err.min == 10
    assert err.max == None
    assert isinstance(err, galsim.GalSimError)
    assert isinstance(err, ValueError)
    check_pickle(err)


@timer
def test_galsim_bounds_error():
    """Test basic usage of GalSimBoundsError
    """
    pos = galsim.PositionI(0,0)
    bounds = galsim.BoundsI(1,10,1,10)
    err = galsim.GalSimBoundsError("Test", pos, bounds)
    print('str = ',str(err))
    print('repr = ',repr(err))
    assert str(err) == "Test galsim.PositionI(0,0) not in galsim.BoundsI(1,10,1,10)"
    assert err.pos == pos
    assert err.bounds == bounds
    assert isinstance(err, galsim.GalSimError)
    assert isinstance(err, ValueError)
    check_pickle(err)


@timer
def test_galsim_undefined_bounds_error():
    """Test basic usage of GalSimUndefinedBoundsError
    """
    err = galsim.GalSimUndefinedBoundsError("Test")
    print('str = ',str(err))
    print('repr = ',repr(err))
    assert str(err) == "Test"
    assert isinstance(err, galsim.GalSimError)
    check_pickle(err)


@timer
def test_galsim_immutable_error():
    """Test basic usage of GalSimImmutableError
    """
    im = galsim.ImageD(np.array([[0]]), make_const=True)
    err = galsim.GalSimImmutableError("Test", im)
    print('str = ',str(err))
    print('repr = ',repr(err))
    assert str(err) == "Test Image: galsim.Image(bounds=galsim.BoundsI(1,1,1,1), wcs=None, dtype=numpy.float64)"
    assert err.image == im
    assert isinstance(err, galsim.GalSimError)
    check_pickle(err)


@timer
def test_galsim_incompatible_values_error():
    """Test basic usage of GalSimIncompatibleValuesError
    """
    err = galsim.GalSimIncompatibleValuesError("Test", a=1, b=2)
    print('str = ',str(err))
    print('repr = ',repr(err))
    # This isn't completely deterministic across python versions.
    str_possibilities = ["Test Values {'a': 1, 'b': 2}",
                         "Test Values {'b': 2, 'a': 1}"]
    assert str(err) in str_possibilities
    assert err.values == dict(a=1, b=2)
    assert isinstance(err, galsim.GalSimError)
    assert isinstance(err, ValueError)
    assert isinstance(err, TypeError)
    check_pickle(err)


@timer
def test_galsim_sed_error():
    """Test basic usage of GalSimSEDError
    """
    sed = galsim.SED('1', wave_type='nm', flux_type='fphotons')
    err = galsim.GalSimSEDError("Test", sed)
    print('str = ',str(err))
    print('repr = ',repr(err))
    assert str(err) == "Test SED: galsim.SED('1', redshift=0.0)"
    assert err.sed == sed
    assert isinstance(err, galsim.GalSimError)
    assert isinstance(err, TypeError)
    check_pickle(err)


@timer
def test_galsim_hsm_error():
    """Test basic usage of GalSimHSMError
    """
    err = galsim.GalSimHSMError("Test")
    print('str = ',str(err))
    print('repr = ',repr(err))
    assert str(err) == "Test"
    assert isinstance(err, galsim.GalSimError)
    check_pickle(err)


@timer
def test_galsim_fft_size_error():
    """Test basic usage of GalSimFFTSizeError
    """
    err = galsim.GalSimFFTSizeError("Test FFT is too big.", 10240)
    print('str = ',str(err))
    print('repr = ',repr(err))
    assert str(err) == ("Test FFT is too big.\nThe required FFT size would be 10240 x 10240, "
                        "which requires 2.34 GB of memory.\nIf you can handle "
                        "the large FFT, you may update gsparams.maximum_fft_size.")
    assert err.size == 10240
    np.testing.assert_almost_equal(err.mem, 2.34375)
    assert isinstance(err, galsim.GalSimError)
    check_pickle(err)


@timer
def test_galsim_config_error():
    """Test basic usage of GalSimConfigError
    """
    err = galsim.GalSimConfigError("Test")
    print('str = ',str(err))
    print('repr = ',repr(err))
    assert str(err) == "Test"
    assert isinstance(err, galsim.GalSimError)
    assert isinstance(err, ValueError)
    check_pickle(err)


@timer
def test_galsim_config_value_error():
    """Test basic usage of GalSimConfigValueError
    """
    value = 2.3
    err = galsim.GalSimConfigValueError("Test", value)
    print('str = ',str(err))
    print('repr = ',repr(err))
    assert str(err) == "Test Value 2.3"
    assert err.value == value
    assert err.allowed_values == None
    assert isinstance(err, galsim.GalSimError)
    assert isinstance(err, galsim.GalSimConfigError)
    assert isinstance(err, ValueError)
    check_pickle(err)

    err = galsim.GalSimConfigValueError("Test", value, (0,1,2))
    print('str = ',str(err))
    print('repr = ',repr(err))
    assert str(err) == "Test Value 2.3 not in (0, 1, 2)"
    assert err.value == value
    assert err.allowed_values == (0,1,2)
    assert isinstance(err, galsim.GalSimError)
    assert isinstance(err, galsim.GalSimConfigError)
    assert isinstance(err, ValueError)
    check_pickle(err)


@timer
def test_galsim_notimplemented_error():
    """Test basic usage of GalSimNotImplementedError
    """
    err = galsim.GalSimNotImplementedError("Test")
    print('str = ',str(err))
    print('repr = ',repr(err))
    assert str(err) == "Test"
    assert isinstance(err, galsim.GalSimError)
    assert isinstance(err, NotImplementedError)
    check_pickle(err)


@timer
def test_galsim_warning():
    """Test basic usage of GalSimWarning
    """
    err = galsim.GalSimWarning("Test")
    print('str = ',str(err))
    print('repr = ',repr(err))
    assert str(err) == "Test"
    assert isinstance(err, UserWarning)
    check_pickle(err)


@timer
def test_galsim_deprecation_warning():
    """Test basic usage of GalSimDeprecationWarning
    """
    err = galsim.GalSimDeprecationWarning("Test")
    print('str = ',str(err))
    print('repr = ',repr(err))
    assert str(err) == "Test"
    assert isinstance(err, UserWarning)
    check_pickle(err)


if __name__ == "__main__":
    runtests(__file__)
