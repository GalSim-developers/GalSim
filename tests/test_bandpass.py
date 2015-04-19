# Copyright (c) 2012-2015 by the GalSim developers team on GitHub
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
from galsim_test_helpers import *
import sys

try:
    import galsim
except ImportError:
    path, filename = os.path.split(__file__)
    sys.path.append(os.path.abspath(os.path.join(path, "..")))
    import galsim

path, filename = os.path.split(__file__)
datapath = os.path.abspath(os.path.join(path, "../examples/data/"))

def test_Bandpass_basic():
    """Basic tests of Bandpass functionality
    """
    import time
    t1 = time.time()

    # All of these should be equivalent
    b_list = [
        galsim.Bandpass(throughput=lambda x: x/1000, blue_limit=400, red_limit=550),
        galsim.Bandpass(throughput='wave/1000', blue_limit=400, red_limit=550),
        galsim.Bandpass(throughput='wave/10000', blue_limit=4000, red_limit=5500, wave_type='A'),
        galsim.Bandpass('wave/1000', 400, 550, 'nanometers', 30.),
        galsim.Bandpass(galsim.LookupTable([400,550], [0.4, 0.55], interpolant='linear')),
        galsim.Bandpass(galsim.LookupTable([4000,5500], [0.4, 0.55], interpolant='linear'),
                        wave_type='ang'),
        galsim.Bandpass(galsim.LookupTable([3000,8700], [0.3, 0.87], interpolant='linear'),
                        wave_type='Angstroms', red_limit=5500, blue_limit=4000),
        galsim.Bandpass(galsim.LookupTable(np.arange(300,651,10),np.arange(0.3,0.651,0.01)),
                        400, 550),
        galsim.Bandpass('chromatic_reference_images/simple_bandpass.dat'),
        galsim.Bandpass('chromatic_reference_images/simple_bandpass.dat', 
                        blue_limit=400, red_limit=550),
        galsim.Bandpass(galsim.LookupTable([3000,8700], [0.3, 0.87], interpolant='linear'),
                          wave_type='Angstroms').truncate(400,550),
        galsim.Bandpass(galsim.LookupTable([100, 400-1.e-10, 400, 550, 550+1.e-10, 900], 
                                           [0., 0., 0.4, 0.55, 0., 0.], interpolant='linear')),
    ]
    k1 = len(b_list)
    b_list += [
        b_list[1].withZeropoint(30.),
        b_list[1].truncate(400,550),
        b_list[2].truncate(400,550),
        b_list[3].truncate(400,550),
        b_list[4].truncate(400,550),
        b_list[5].truncate(400,550),
        b_list[4].thin(),
        b_list[5].thin(),
        b_list[6].thin(),
        b_list[6].thin(preserve_range=True),
        b_list[7].thin(),
        b_list[11].thin(),
        b_list[11].thin(preserve_range=True),
    ]

    for k,b in enumerate(b_list):
        print k,' b = ',b
        if k not in [k1-1, len(b_list)-1]:
            np.testing.assert_almost_equal(b.blue_limit, 400, decimal=12)
            np.testing.assert_almost_equal(b.red_limit, 550, decimal=12)
        np.testing.assert_almost_equal(b(400), 0.4, decimal=12)
        np.testing.assert_almost_equal(b(490), 0.49, decimal=12)
        np.testing.assert_almost_equal(b(550), 0.55, decimal=12)
        np.testing.assert_almost_equal(b(399), 0., decimal=12)
        np.testing.assert_almost_equal(b(551), 0., decimal=12)
        np.testing.assert_array_almost_equal(b([410,430,450,470]), [0.41,0.43,0.45,0.47], 12)
        np.testing.assert_array_almost_equal(b((410,430,450,470)), [0.41,0.43,0.45,0.47], 12)
        np.testing.assert_array_almost_equal(b(np.array([410,430,450,470])),
                                             [0.41,0.43,0.45,0.47], 12)
        if k in [3,k1]:
            np.testing.assert_almost_equal(b.zeropoint, 30., decimal=12)

        # Default calculation isn't very accurate for widely spaced wavelengths like this 
        # example.  Only accurate to 1 digit!
        lam_eff = b.effective_wavelength
        print 'lam_eff = ',lam_eff
        true_lam_eff = (9100./19)  # analytic answer
        np.testing.assert_almost_equal(lam_eff / true_lam_eff, 1.0, 1)

        # Can get a more precise calculation with the following: (much more precise in this case)
        lam_eff = b.calculateEffectiveWavelength(precise=True)
        print 'precise lam_eff = ',lam_eff
        np.testing.assert_almost_equal(lam_eff, true_lam_eff, 12)

        # After which, the simple attribute syntax keeps the improved precision
        lam_eff = b.effective_wavelength
        np.testing.assert_almost_equal(lam_eff, true_lam_eff, 12)
        
        # Only the first one is not picklable
        if k > 0:
            do_pickle(b)
            do_pickle(b, lambda x: (x(390), x(470), x(490), x(510), x(560)) )

    t2 = time.time()
    print 'time for %s = %.2f'%(funcname(),t2-t1)

def test_Bandpass_mul():
    """Check that Bandpasses multiply like I think they should...
    """
    import time
    t1 = time.time()

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
    f = b * 1.21
    np.testing.assert_almost_equal(f(3.0), 3.33 * 1.21, 10,
                                   err_msg="Found wrong value in Bandpass.__mul__")
    np.testing.assert_array_almost_equal(f.wave_list, [1.1, 2.2, 3, 4.4, 5.5],
                                         err_msg="wrong wave_list in Bandpass.__mul__")
    do_pickle(f)

    # scalar * Bandpass
    f = 1.21 * a
    np.testing.assert_almost_equal(f(3.0), 3.0 * 1.21, 10,
                                   err_msg="Found wrong value in Bandpass.__mul__")
    np.testing.assert_array_almost_equal(f.wave_list, [1, 2, 3, 4, 5],
                                         err_msg="wrong wave_list in Bandpass.__mul__")
    do_pickle(f)

    t2 = time.time()
    print 'time for %s = %.2f'%(funcname(),t2-t1)

def test_Bandpass_div():
    """Check that Bandpasses multiply like I think they should...
    """
    import time
    t1 = time.time()

    a = galsim.Bandpass(galsim.LookupTable([1,2,3,4,5], [1,2,3,4,5]))
    b = galsim.Bandpass(galsim.LookupTable([1.1,2.2,3.0,4.4,5.5], [1.11,2.22,3.33,4.44,5.55]))

    # Bandpass / Bandpass
    c = a/b
    np.testing.assert_almost_equal(c.blue_limit, 1.1, 10,
                                   err_msg="Found wrong blue limit in Bandpass.__div__")
    np.testing.assert_almost_equal(c.red_limit, 5.0, 10,
                                   err_msg="Found wrong red limit in Bandpass.__div__")
    np.testing.assert_almost_equal(c(3.0), 3.0 / 3.33, 10,
                                   err_msg="Found wrong value in Bandpass.__div__")
    np.testing.assert_almost_equal(c(1.1), a(1.1)/1.11, 10,
                                   err_msg="Found wrong value in Bandpass.__div__")
    np.testing.assert_almost_equal(c(5.0), 5/b(5.0), 10,
                                   err_msg="Found wrong value in Bandpass.__div__")
    np.testing.assert_array_almost_equal(c.wave_list, [1.1, 2, 2.2, 3, 4, 4.4, 5],
                                         err_msg="wrong wave_list in Bandpass.__div__")

    # Bandpass / fn
    d = lambda w: w**2
    e = c/d
    np.testing.assert_almost_equal(e(3.0), c(3.0) / 3.0**2, 10,
                                   err_msg="Found wrong value in Bandpass.__div__")
    np.testing.assert_array_almost_equal(e.wave_list, [1.1, 2, 2.2, 3, 4, 4.4, 5],
                                         err_msg="wrong wave_list in Bandpass.__div__")

    # Bandpass / scalar
    f = b / 1.21
    np.testing.assert_almost_equal(f(3.0), b(3.0)/1.21, 10,
                                   err_msg="Found wrong value in Bandpass.__div__")
    np.testing.assert_array_almost_equal(f.wave_list, [1.1, 2.2, 3, 4.4, 5.5],
                                         err_msg="wrong wave_list in Bandpass.__div__")
    do_pickle(f)

    t2 = time.time()
    print 'time for %s = %.2f'%(funcname(),t2-t1)

def test_Bandpass_wave_type():
    """Check that `wave_type='ang'` works in Bandpass.__init__
    """
    import time
    t1 = time.time()

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
    np.testing.assert_array_almost_equal(b0([1,2,3,4,5]), b1([1,2,3,4,5]), decimal=7,
                               err_msg="Bandpass.__call__ doesn't respect wave_type")

    t2 = time.time()
    print 'time for %s = %.2f'%(funcname(),t2-t1)


if __name__ == "__main__":
    test_Bandpass_basic()
    test_Bandpass_mul()
    test_Bandpass_div()
    test_Bandpass_wave_type()
