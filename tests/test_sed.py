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
import galsim

path, filename = os.path.split(__file__)
datapath = os.path.abspath(os.path.join(path, "../examples/data/"))

def test_SED_add():
    """Check that SEDs add like I think they should...
    """
    a = galsim.SED(galsim.LookupTable([1,2,3,4,5], [1.1,2.2,3.3,4.4,5.5]),
                   flux_type='fphotons')
    b = galsim.SED(galsim.LookupTable([1.1,2.2,3.0,4.4,5.5], [1.11,2.22,3.33,4.44,5.55]),
                   flux_type='fphotons')
    c = a+b
    np.testing.assert_almost_equal(c.blue_limit, 1.1, 10,
                                   err_msg="Found wrong blue limit in SED.__add__")
    np.testing.assert_almost_equal(c.red_limit, 5.0, 10,
                                   err_msg="Found wrong red limit in SED.__add__")
    np.testing.assert_almost_equal(c(3.0), 3.3 + 3.33, 10,
                                   err_msg="Wrong sum in SED.__add__")
    np.testing.assert_almost_equal(c(1.1), a(1.1)+1.11, 10,
                                   err_msg="Wrong sum in SED.__add__")
    np.testing.assert_almost_equal(c(5.0), 5.5+b(5.0), 10,
                                   err_msg="Wrong sum in SED.__add__")

def test_SED_sub():
    """Check that SEDs subtract like I think they should...
    """
    a = galsim.SED(galsim.LookupTable([1,2,3,4,5], [1.1,2.2,3.3,4.4,5.5]),
                   flux_type='fphotons')
    b = galsim.SED(galsim.LookupTable([1.1,2.2,3.0,4.4,5.5], [1.11,2.22,3.33,4.44,5.55]),
                   flux_type='fphotons')
    c = a-b
    np.testing.assert_almost_equal(c.blue_limit, 1.1, 10,
                                   err_msg="Found wrong blue limit in SED.__add__")
    np.testing.assert_almost_equal(c.red_limit, 5.0, 10,
                                   err_msg="Found wrong red limit in SED.__add__")
    np.testing.assert_almost_equal(c(3.0), 3.3 - 3.33, 10,
                                   err_msg="Wrong sum in SED.__sub__")
    np.testing.assert_almost_equal(c(1.1), a(1.1)-1.11, 10,
                                   err_msg="Wrong sum in SED.__add__")
    np.testing.assert_almost_equal(c(5.0), 5.5-b(5.0), 10,
                                   err_msg="Wrong sum in SED.__add__")

def test_SED_mul():
    """Check that SEDs multiply like I think they should...
    """
    a = galsim.SED(galsim.LookupTable([1,2,3,4,5], [1.1,2.2,3.3,4.4,5.5]),
                   flux_type='fphotons')
    b = lambda w: w**2
    # SED multiplied by function
    c = a*b
    np.testing.assert_almost_equal(c(3.0), 3.3 * 3**2, 10,
                                   err_msg="Found wrong value in SED.__mul__")
    # function multiplied by SED
    c = b*a
    np.testing.assert_almost_equal(c(3.0), 3.3 * 3**2, 10,
                                   err_msg="Found wrong value in SED.__rmul__")
    # SED multiplied by scalar
    d = c*4.2
    np.testing.assert_almost_equal(d(3.0), 3.3 * 3**2 * 4.2, 10,
                                   err_msg="Found wrong value in SED.__mul__")
    # assignment multiplication
    d *= 2
    np.testing.assert_almost_equal(d(3.0), 3.3 * 3**2 * 4.2 * 2, 10,
                                   err_msg="Found wrong value in SED.__mul__")

def test_SED_div():
    """Check that SEDs divide like I think they should...
    """
    a = galsim.SED(galsim.LookupTable([1,2,3,4,5], [1.1,2.2,3.3,4.4,5.5]),
                   flux_type='fphotons')
    b = lambda w: w**2
    # SED divided by function
    c = a/b
    np.testing.assert_almost_equal(c(3.0), 3.3 / 3**2, 10,
                                   err_msg="Found wrong value in SED.__div__")
    # function divided by SED
    c = b/a
    np.testing.assert_almost_equal(c(3.0), 3**2 / 3.3, 10,
                                   err_msg="Found wrong value in SED.__rdiv__")
    # SED divided by scalar
    d = c/4.2
    np.testing.assert_almost_equal(d(3.0), 3**2 / 3.3 / 4.2, 10,
                                   err_msg="Found wrong value in SED.__div__")
    # assignment division
    d /= 2
    np.testing.assert_almost_equal(d(3.0), 3**2 / 3.3 / 4.2 / 2, 10,
                                   err_msg="Found wrong value in SED.__div__")

def test_SED_atRedshift():
    """Check that SEDs redshift correctly.
    """
    a = galsim.SED(os.path.join(datapath, 'CWW_E_ext.sed'))
    bolo_flux = a.calculateFlux(bandpass=None)
    for z1, z2 in zip([0.5, 1.0, 1.4], [1.0, 1.0, 1.0]):
        b = a.atRedshift(z1)
        c = b.atRedshift(z1) # same redshift, so should be no change
        d = c.atRedshift(z2) # do a relative redshifting from z1 to z2
        e = b.thin(rel_err=1.e-5)  # effectively tests that wave_list is handled correctly.
                                   # (Issue #520)
        for w in [350, 500, 650]:
            np.testing.assert_almost_equal(a(w), b(w*(1.0+z1)), 10,
                                           err_msg="error redshifting SED")
            np.testing.assert_almost_equal(a(w), c(w*(1.0+z1)), 10,
                                           err_msg="error redshifting SED")
            np.testing.assert_almost_equal(a(w), d(w*(1.0+z2)), 10,
                                           err_msg="error redshifting SED")
            np.testing.assert_almost_equal((a(w)-e(w*(1.0+z1)))/bolo_flux, 0., 5,
                                           err_msg="error redshifting and thinning SED")

def test_SED_roundoff_guard():
    """Check that SED.__init__ roundoff error guard works. (Issue #520).
    """
    a = galsim.SED(os.path.join(datapath, 'CWW_Scd_ext.sed'))
    for z in np.arange(0.0, 0.5, 0.001):
        b = a.atRedshift(z)
        w1 = b.wave_list[0]
        w2 = b.wave_list[-1]
        np.testing.assert_almost_equal(a(w1/(1.0+z)), b(w1), 10,
                                        err_msg="error using wave_list limits in redshifted SED")
        np.testing.assert_almost_equal(a(w2/(1.0+z)), b(w2), 10,
                                        err_msg="error using wave_list limits in redshifted SED")

def test_SED_init():
    """Check that certain invalid SED initializations are trapped.
    """
    try:
        # These fail.
        np.testing.assert_raises(ValueError, galsim.SED, spec='blah')
        np.testing.assert_raises(ValueError, galsim.SED, spec='wave+')
        np.testing.assert_raises(ValueError, galsim.SED, spec='somewhere/a/file')
        np.testing.assert_raises(ValueError, galsim.SED, spec='/somewhere/a/file')
    except ImportError:
        print 'The assert_raises tests require nose'
    # These should succeed.
    galsim.SED(spec='wave')
    galsim.SED(spec='wave/wave')
    galsim.SED(spec=lambda w:1.0)
    galsim.SED(spec='1./(wave-700)')

def test_SED_calculateMagnitude():
    """ Check that magnitudes work as expected.
    """
    # Test that we can create a zeropoint with an SED, and that magnitudes for that SED are
    # then 0.0
    sed = galsim.SED(spec='wave')
    bandpass = galsim.Bandpass(galsim.LookupTable([1,2,3,4,5], [1,2,3,4,5]), zeropoint=sed)
    np.testing.assert_almost_equal(sed.calculateMagnitude(bandpass), 0.0)
    # Try multiplying SED by 100 to verify that magnitude decreases by 5
    sed *= 100
    np.testing.assert_almost_equal(sed.calculateMagnitude(bandpass), -5.0)
    # Try setting zeropoint to a constant.
    bandpass = galsim.Bandpass(galsim.LookupTable([1,2,3,4,5], [1,2,3,4,5]), zeropoint=6.0)
    np.testing.assert_almost_equal(sed.calculateMagnitude(bandpass),
                                   (sed*100).calculateMagnitude(bandpass)+5.0)
    # Try setting AB zeropoint
    bandpass = galsim.Bandpass(galsim.LookupTable([1,2,3,4,5], [1,2,3,4,5]))
    np.testing.assert_almost_equal(sed.calculateMagnitude(bandpass),
                                   (sed*100).calculateMagnitude(bandpass)+5.0)

if __name__ == "__main__":
    test_SED_add()
    test_SED_sub()
    test_SED_mul()
    test_SED_div()
    test_SED_atRedshift()
    test_SED_roundoff_guard()
    test_SED_init()
    test_SED_calculateMagnitude()
