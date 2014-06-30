# Copyright (c) 2012-2014 by the GalSim developers team on GitHub
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
import os
import numpy as np

try:
    import galsim
except ImportError:
    path, filename = os.path.split(__file__)
    sys.path.append(os.path.abspath(os.path.join(path, "..")))
    import galsim

path, filename = os.path.split(__file__)
datapath = os.path.abspath(os.path.join(path, "../examples/data/"))

def test_Bandpass_mul():
    """Check that Bandpasses multiply like I think they should...
    """
    a = galsim.Bandpass(galsim.LookupTable([1,2,3,4,5], [1,2,3,4,5]))
    b = galsim.Bandpass(galsim.LookupTable([1.1,2.2,3.0,4.4,5.5], [1.11,2.22,3.33,4.44,5.55]))
    # Bandpass * Bandpass
    c = a*b
    np.testing.assert_almost_equal(c.blue_limit, 1.1, 10,
                                   err_msg="Found wrong blue limit in Bandpass.__mul__")
    np.testing.assert_almost_equal(c.red_limit, 5.0, 10,
                                   err_msg="Found wrong red limit in Bandpass.__mul__")
    np.testing.assert_almost_equal(c(3.0), 3.0 * 3.33, 10,
                                   err_msg="Found wrong value in Bandpass.__mul__")
    np.testing.assert_almost_equal(c(1.1), a(1.1)*1.11, 10,
                                   err_msg="Found wrong value in Bandpass.__mul__")
    np.testing.assert_almost_equal(c(5.0), b(5.0)*5, 10,
                                   err_msg="Found wrong value in Bandpass.__mul__")
    np.testing.assert_array_almost_equal(c.wave_list, [1.1, 2, 2.2, 3, 4, 4.4, 5],
                                         err_msg="wrong wave_list in Bandpass.__mul__")
    # Bandpass * fn
    d = lambda w: w**2
    e = c*d
    np.testing.assert_almost_equal(e(3.0), 3.0 * 3.33 * 3.0**2, 10,
                                   err_msg="Found wrong value in Bandpass.__mul__")
    np.testing.assert_array_almost_equal(e.wave_list, [1.1, 2, 2.2, 3, 4, 4.4, 5],
                                         err_msg="wrong wave_list in Bandpass.__mul__")
    # fn * Bandpass
    e = d*c
    np.testing.assert_almost_equal(e(3.0), 3.0 * 3.33 * 3.0**2, 10,
                                   err_msg="Found wrong value in Bandpass.__mul__")
    np.testing.assert_array_almost_equal(e.wave_list, [1.1, 2, 2.2, 3, 4, 4.4, 5],
                                         err_msg="wrong wave_list in Bandpass.__mul__")
    # Bandpass * scalar
    f = e * 1.21
    np.testing.assert_almost_equal(f(3.0), 3.0 * 3.33 * 3.0**2 * 1.21, 10,
                                   err_msg="Found wrong value in Bandpass.__mul__")
    np.testing.assert_array_almost_equal(f.wave_list, [1.1, 2, 2.2, 3, 4, 4.4, 5],
                                         err_msg="wrong wave_list in Bandpass.__mul__")
    # scalar * Bandpass
    f = 1.21 * e
    np.testing.assert_almost_equal(f(3.0), 3.0 * 3.33 * 3.0**2 * 1.21, 10,
                                   err_msg="Found wrong value in Bandpass.__mul__")
    np.testing.assert_array_almost_equal(f.wave_list, [1.1, 2, 2.2, 3, 4, 4.4, 5],
                                         err_msg="wrong wave_list in Bandpass.__mul__")

def test_Bandpass_div():
    """Check that Bandpasses multiply like I think they should...
    """
    a = galsim.Bandpass(galsim.LookupTable([1,2,3,4,5], [1,2,3,4,5]))
    b = galsim.Bandpass(galsim.LookupTable([1.1,2.2,3.0,4.4,5.5], [1.11,2.22,3.33,4.44,5.55]))
    # Bandpass / Bandpass
    c = a/b
    np.testing.assert_almost_equal(c.blue_limit, 1.1, 10,
                                   err_msg="Found wrong blue limit in Bandpass.__mul__")
    np.testing.assert_almost_equal(c.red_limit, 5.0, 10,
                                   err_msg="Found wrong red limit in Bandpass.__mul__")
    np.testing.assert_almost_equal(c(3.0), 3.0 / 3.33, 10,
                                   err_msg="Found wrong value in Bandpass.__mul__")
    np.testing.assert_almost_equal(c(1.1), a(1.1)/1.11, 10,
                                   err_msg="Found wrong value in Bandpass.__mul__")
    np.testing.assert_almost_equal(c(5.0), 5/b(5.0), 10,
                                   err_msg="Found wrong value in Bandpass.__mul__")
    np.testing.assert_array_almost_equal(c.wave_list, [1.1, 2, 2.2, 3, 4, 4.4, 5],
                                         err_msg="wrong wave_list in Bandpass.__mul__")
    # Bandpass / fn
    d = lambda w: w**2
    e = c/d
    np.testing.assert_almost_equal(e(3.0), 3.0 / 3.33 / 3.0**2, 10,
                                   err_msg="Found wrong value in Bandpass.__mul__")
    np.testing.assert_array_almost_equal(e.wave_list, [1.1, 2, 2.2, 3, 4, 4.4, 5],
                                         err_msg="wrong wave_list in Bandpass.__mul__")
    # fn / Bandpass
    e = d/c
    np.testing.assert_almost_equal(e(3.0), 3.0**2 / (3.0 / 3.33), 10,
                                   err_msg="Found wrong value in Bandpass.__mul__")
    np.testing.assert_array_almost_equal(e.wave_list, [1.1, 2, 2.2, 3, 4, 4.4, 5],
                                         err_msg="wrong wave_list in Bandpass.__mul__")
    # Bandpass / scalar
    f = e / 1.21
    np.testing.assert_almost_equal(f(3.0), (3.0**2 / (3.0 / 3.33)) / 1.21, 10,
                                   err_msg="Found wrong value in Bandpass.__mul__")
    np.testing.assert_array_almost_equal(f.wave_list, [1.1, 2, 2.2, 3, 4, 4.4, 5],
                                         err_msg="wrong wave_list in Bandpass.__mul__")
    # scalar / Bandpass
    f = 1.21 / e
    np.testing.assert_almost_equal(f(3.0), 1.21 / (3.0**2 / (3.0 / 3.33)), 10,
                                   err_msg="Found wrong value in Bandpass.__mul__")
    np.testing.assert_array_almost_equal(f.wave_list, [1.1, 2, 2.2, 3, 4, 4.4, 5],
                                         err_msg="wrong wave_list in Bandpass.__mul__")

def test_Bandpass_wave_type():
    """Check that `wave_type='ang'` works in Bandpass.__init__
    """
    a0 = galsim.Bandpass(os.path.join(datapath, 'LSST_r.dat'))
    a1 = galsim.Bandpass(os.path.join(datapath, 'LSST_r.dat'), wave_type='ang')

    np.testing.assert_approx_equal(a0.red_limit, a1.red_limit*10,
                                   err_msg="Bandpass.red_limit doesn't respect wave_type")
    np.testing.assert_approx_equal(a0.blue_limit, a1.blue_limit*10,
                                   err_msg="Bandpass.blue_limit doesn't respect wave_type")
    np.testing.assert_approx_equal(a0.effective_wavelength, a1.effective_wavelength*10,
                                   err_msg="Bandpass.effective_wavelength doesn't respect"
                                           +" wave_type")

    b0 = galsim.Bandpass(galsim.LookupTable([1,2,3,4,5], [1,2,3,4,5]))
    b1 = galsim.Bandpass(galsim.LookupTable([10,20,30,40,50], [1,2,3,4,5]), wave_type='ang')
    np.testing.assert_approx_equal(b0.red_limit, b1.red_limit,
                                   err_msg="Bandpass.red_limit doesn't respect wave_type")
    np.testing.assert_approx_equal(b0.blue_limit, b1.blue_limit,
                                   err_msg="Bandpass.blue_limit doesn't respect wave_type")
    np.testing.assert_approx_equal(b0.effective_wavelength, b1.effective_wavelength,
                                   err_msg="Bandpass.effective_wavelength doesn't respect"
                                           +" wave_type")
    np.testing.assert_allclose(b0([1,2,3,4,5]), b1([1,2,3,4,5]),
                               err_msg="Bandpass.__call__ doesn't respect wave_type")

if __name__ == "__main__":
    test_Bandpass_mul()
    test_Bandpass_div()
    test_Bandpass_wave_type()
