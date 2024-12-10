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

import os
import numpy as np
from astropy import units

import galsim
from galsim_test_helpers import *

datapath = os.path.join(galsim.meta_data.share_dir, "bandpasses")


@timer
def test_Bandpass_basic():
    """Basic tests of Bandpass functionality
    """
    # All of these should be equivalent
    b_list = [
        galsim.Bandpass(throughput=lambda x: x/1000, wave_type='nm', blue_limit=400, red_limit=550),
        galsim.Bandpass(throughput='wave/1000', wave_type='nm', blue_limit=400, red_limit=550),
        galsim.Bandpass(throughput='wave/10000', wave_type='A', blue_limit=4000, red_limit=5500),
        galsim.Bandpass('wave/1000', 'nanometers', 400, 550, 30.),
        galsim.Bandpass('wave/np.sqrt(1.e6)', 'nm', 400, 550, 30.),
        galsim.Bandpass('wave/numpy.sqrt(1.e6)', 'nm', 400, 550, 30.),
        galsim.Bandpass('wave/math.sqrt(1.e6)', 'nm', 400, 550, 30.),
        galsim.Bandpass(galsim.LookupTable([400,550], [0.4, 0.55], interpolant='linear'),
                        wave_type=units.Unit('nm')),
        galsim.Bandpass(galsim.LookupTable([4000,5500], [0.4, 0.55], interpolant='linear'),
                        wave_type=units.Unit('Angstrom')),
        galsim.Bandpass(galsim.LookupTable([3000,8700], [0.3, 0.87], interpolant='linear'),
                        wave_type='ang', red_limit=5500, blue_limit=4000),
        galsim.Bandpass(galsim.LookupTable(np.arange(3.e-7,6.51e-7,1.e-8),
                                           np.arange(0.3,0.651,0.01)),
                        units.Unit('m'), 4.e-7, 5.5e-7),
        galsim.Bandpass('chromatic_reference_images/simple_bandpass.dat', wave_type='nm'),
        galsim.Bandpass('chromatic_reference_images/simple_bandpass.dat', wave_type='nm',
                        blue_limit=400, red_limit=550),
        galsim.Bandpass(galsim.LookupTable([3000,8700], [0.3, 0.87], interpolant='linear'),
                        wave_type='Angstroms').truncate(400,550),
        galsim.Bandpass(galsim.LookupTable([100, 400-1.e-12, 400, 550, 550+1.e-12, 900],
                                           [0., 0., 0.4, 0.55, 0., 0.], interpolant='linear'),
                        wave_type='nm'),
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
        b_list[6].thin(preserve_range=False),
        b_list[7].thin(),
        b_list[11].thin(preserve_range=False, trim_zeros=False),
        b_list[11].thin(preserve_range=False),
        b_list[11].thin()
    ]

    for k,b in enumerate(b_list):
        print(k,' b = ',b)
        if k not in [k1-1, len(b_list)-2, len(b_list)-1]:
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
        print('lam_eff = ',lam_eff)
        true_lam_eff = (9100./19)  # analytic answer
        np.testing.assert_almost_equal(lam_eff / true_lam_eff, 1.0, 1)

        # Can get a more precise calculation with the following: (much more precise in this case)
        lam_eff = b.calculateEffectiveWavelength(precise=True)
        print('precise lam_eff = ',lam_eff)
        np.testing.assert_almost_equal(lam_eff, true_lam_eff, 12)

        # After which, the simple attribute syntax keeps the improved precision
        lam_eff = b.effective_wavelength
        np.testing.assert_almost_equal(lam_eff, true_lam_eff, 12)

        # Only the first one is not picklable
        if k > 0:
            check_pickle(b)
            check_pickle(b, lambda x: (x(390), x(470), x(490), x(510), x(560)) )

    assert_raises(ValueError, galsim.Bandpass, throughput="'eggs'", wave_type='nm',
                  blue_limit=400, red_limit=700)
    assert_raises(ValueError, galsim.Bandpass, throughput="'eggs)", wave_type='nm',
                  blue_limit=400, red_limit=700)
    assert_raises(TypeError, galsim.Bandpass, throughput=lambda x:x)
    assert_raises(ValueError, galsim.Bandpass, throughput="'spam'", wave_type='A',
                  blue_limit=400, red_limit=700)
    assert_raises(TypeError, galsim.Bandpass, throughput='1', wave_type='nm')
    assert_raises(TypeError, galsim.Bandpass, throughput=lambda w: 1, wave_type='nm')
    assert_raises(ValueError, galsim.Bandpass, throughput='1', wave_type='nm',
                  blue_limit=700, red_limit=400)
    assert_raises(ValueError, galsim.Bandpass, throughput=lambda w: 1, wave_type='inches')
    assert_raises(ValueError, galsim.Bandpass, throughput=lambda w: 1, wave_type=units.Unit('Hz'))
    assert_raises(ValueError, galsim.Bandpass, galsim.LookupTable([400,550], [0.4, 0.55], 'linear'),
                  wave_type='nm', blue_limit=300, red_limit=500)
    assert_raises(ValueError, galsim.Bandpass, galsim.LookupTable([400,550], [0.4, 0.55], 'linear'),
                  wave_type='nm', blue_limit=500, red_limit=600)


@timer
def test_Bandpass_mul():
    """Check that Bandpasses multiply like I think they should...
    """
    a_lt = galsim.Bandpass(galsim.LookupTable([1,2,3,4,5], [1,2,3,4,5]), 'nm')
    a_fn = galsim.Bandpass('wave', 'nm', blue_limit=1, red_limit=5)
    b = galsim.Bandpass(galsim.LookupTable([1.1,2.2,3.0,4.4,5.5], [1.11,2.22,3.33,4.44,5.55]), 'nm')

    for a in [a_lt, a_fn]:
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
        if a is a_lt:
            combined_wave_list = [1.1, 2, 2.2, 3, 4., 4.4, 5]
        else:
            combined_wave_list = [1.1, 2.2, 3, 4.4, 5]

        np.testing.assert_array_almost_equal(c.wave_list, combined_wave_list,
                                             err_msg="wrong wave_list in Bandpass.__mul__")

        # Bandpass * fn
        d = lambda w: w**2
        e = c*d
        np.testing.assert_almost_equal(e(3.0), 3.0 * 3.33 * 3.0**2, 10,
                                       err_msg="Found wrong value in Bandpass.__mul__")
        np.testing.assert_array_almost_equal(e.wave_list, combined_wave_list,
                                             err_msg="wrong wave_list in Bandpass.__mul__")

        # fn * Bandpass
        e = d*c
        np.testing.assert_almost_equal(e(3.0), 3.0 * 3.33 * 3.0**2, 10,
                                       err_msg="Found wrong value in Bandpass.__mul__")
        np.testing.assert_array_almost_equal(e.wave_list, combined_wave_list,
                                             err_msg="wrong wave_list in Bandpass.__mul__")

        # Bandpass * scalar
        f = b * 1.21
        np.testing.assert_almost_equal(f(3.0), 3.33 * 1.21, 10,
                                       err_msg="Found wrong value in Bandpass.__mul__")
        np.testing.assert_array_almost_equal(f.wave_list, [1.1, 2.2, 3, 4.4, 5.5],
                                         err_msg="wrong wave_list in Bandpass.__mul__")
        check_pickle(f)

        # scalar * Bandpass
        f = 1.21 * a
        np.testing.assert_almost_equal(f(3.0), 3.0 * 1.21, 10,
                                       err_msg="Found wrong value in Bandpass.__mul__")
        if a is a_lt:
            np.testing.assert_array_almost_equal(f.wave_list, [1, 2, 3, 4, 5],
                                             err_msg="wrong wave_list in Bandpass.__mul__")
        else:
            np.testing.assert_array_almost_equal(f.wave_list, [],
                                             err_msg="wrong wave_list in Bandpass.__mul__")

        if a is a_lt:
            check_pickle(f)


@timer
def test_Bandpass_div():
    """Check that Bandpasses multiply like I think they should...
    """
    a_lt = galsim.Bandpass(galsim.LookupTable([1,2,3,4,5], [1,2,3,4,5]), 'nm')
    a_fn = galsim.Bandpass('wave', 'nm', blue_limit=1, red_limit=5)
    b = galsim.Bandpass(galsim.LookupTable([1.1,2.2,3.0,4.4,5.5], [1.11,2.22,3.33,4.44,5.55]), 'nm')

    for a in [a_lt, a_fn]:
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
        if a is a_lt:
            combined_wave_list = [1.1, 2, 2.2, 3, 4., 4.4, 5]
        else:
            combined_wave_list = [1.1, 2.2, 3, 4.4, 5]
        np.testing.assert_array_almost_equal(c.wave_list, combined_wave_list,
                                             err_msg="wrong wave_list in Bandpass.__div__")

        # Bandpass / fn
        d = lambda w: w**2
        e = c/d
        np.testing.assert_almost_equal(e(3.0), c(3.0) / 3.0**2, 10,
                                       err_msg="Found wrong value in Bandpass.__div__")
        np.testing.assert_array_almost_equal(e.wave_list, combined_wave_list,
                                             err_msg="wrong wave_list in Bandpass.__div__")

        # Bandpass / scalar
        f = a / 1.21
        np.testing.assert_almost_equal(f(3.0), a(3.0)/1.21, 10,
                                       err_msg="Found wrong value in Bandpass.__div__")
        if a is a_lt:
            np.testing.assert_array_almost_equal(f.wave_list, [1, 2, 3, 4, 5],
                                                 err_msg="wrong wave_list in Bandpass.__div__")
            check_pickle(f)
        else:
            np.testing.assert_array_almost_equal(f.wave_list, [],
                                                 err_msg="wrong wave_list in Bandpass.__div__")

    sed = galsim.SED('1', wave_type='nm', flux_type='1')
    assert_raises(TypeError, a_lt.__div__, sed)
    assert_raises(TypeError, a_fn.__div__, sed)


@timer
def test_Bandpass_wave_type():
    """Check that `wave_type='ang'` works in Bandpass.__init__
    """
    # Also check with and without explicit directory
    a0 = galsim.Bandpass(os.path.join(datapath, 'LSST_r.dat'), wave_type='nm')
    a1 = galsim.Bandpass('LSST_r.dat', wave_type='ang')

    np.testing.assert_approx_equal(a0.red_limit, a1.red_limit*10,
                                   err_msg="Bandpass.red_limit doesn't respect wave_type")
    np.testing.assert_approx_equal(a0.blue_limit, a1.blue_limit*10,
                                   err_msg="Bandpass.blue_limit doesn't respect wave_type")
    np.testing.assert_approx_equal(a0.effective_wavelength, a1.effective_wavelength*10,
                                   err_msg="Bandpass.effective_wavelength doesn't respect"
                                           +" wave_type")

    # Spline interpolation changes the effective wavelength slightly, but not much.
    a2 = galsim.Bandpass('LSST_r.dat', wave_type='nm', interpolant='spline')
    np.testing.assert_equal(a2.red_limit, a0.red_limit)
    np.testing.assert_equal(a2.blue_limit, a0.blue_limit)
    np.testing.assert_allclose(a2.effective_wavelength, a0.effective_wavelength, rtol=1.e-3)

    b0 = galsim.Bandpass(galsim.LookupTable([1,2,3,4,5], [1,2,3,4,5]), wave_type='nm')
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


@timer
def test_ne():
    """ Check that inequality works as expected."""
    tput = lambda x: x/1000
    lt = galsim.LookupTable([400, 550], [0.4, 0.55], interpolant='linear')
    sed = galsim.SED('3', 'nm', 'flambda')

    # These should all compare unequal.
    bps = [galsim.Bandpass(throughput=tput, wave_type='nm', blue_limit=400, red_limit=550),
           galsim.Bandpass(throughput=tput, wave_type='nm', blue_limit=400, red_limit=551),
           galsim.Bandpass(throughput=tput, wave_type='nm', blue_limit=401, red_limit=550),
           galsim.Bandpass(throughput=lt, wave_type='nm'),
           galsim.Bandpass(throughput=lt, wave_type='A'),
           galsim.Bandpass(throughput=lt, wave_type='nm', zeropoint=10.0),
           galsim.Bandpass(throughput=lt, wave_type='nm').withZeropoint('AB'),
           galsim.Bandpass(throughput=lt, wave_type='nm').withZeropoint('ST'),
           galsim.Bandpass(throughput=lt, wave_type='nm').withZeropoint('Vega'),
           galsim.Bandpass(throughput=lt, wave_type='nm').withZeropoint(100.0),
           galsim.Bandpass(throughput=lt, wave_type='nm').withZeropoint(sed)]
    check_all_diff(bps)

    with assert_raises(galsim.GalSimValueError):
           galsim.Bandpass(throughput=lt, wave_type='nm').withZeropoint('invalid')
    with assert_raises(TypeError):
           galsim.Bandpass(throughput=lt, wave_type='nm').withZeropoint(None)


@timer
def test_thin():
    """Test that bandpass thinning works with the requested accuracy."""
    s = galsim.SED('1', wave_type='nm', flux_type='fphotons')
    bp1 = galsim.Bandpass(os.path.join(datapath, 'LSST_r.dat'), 'nm')
    bp2 = galsim.Bandpass(os.path.join(datapath, 'LSST_r.dat'), 'nm', interpolant='spline')

    for bp in [bp1, bp2]:
        flux = s.calculateFlux(bp)
        print("Original number of bandpass samples = ",len(bp.wave_list))
        for err in [1.e-2, 1.e-3, 1.e-4, 1.e-5]:
            print("Test err = ",err)
            thin_bp = bp.thin(rel_err=err, preserve_range=True, fast_search=False)
            thin_flux = s.calculateFlux(thin_bp)
            thin_err = (flux-thin_flux)/flux
            print("num samples with preserve_range = True, fast_search = False: ",
                  len(thin_bp.wave_list))
            print("realized error = ",(flux-thin_flux)/flux)
            thin_bp = bp.thin(rel_err=err, preserve_range=True)
            thin_flux = s.calculateFlux(thin_bp)
            thin_err = (flux-thin_flux)/flux
            print("num samples with preserve_range = True: ",len(thin_bp.wave_list))
            print("realized error = ",(flux-thin_flux)/flux)
            assert np.abs(thin_err) < err, "Thinned bandpass failed accuracy goal, preserving range."
            thin_bp = bp.thin(rel_err=err, preserve_range=False)
            thin_flux = s.calculateFlux(thin_bp)
            thin_err = (flux-thin_flux)/flux
            print("num samples with preserve_range = False: ",len(thin_bp.wave_list))
            print("realized error = ",(flux-thin_flux)/flux)
            assert np.abs(thin_err) < err, "Thinned bandpass failed accuracy goal, w/ range shrinkage."


@timer
def test_zp():
    """Check that the zero points are maintained in an appropriate way when thinning, truncating."""
    # Make a bandpass and set an AB zeropoint.
    bp = galsim.Bandpass(os.path.join(datapath, 'LSST_r.dat'), 'nm')
    bp = bp.withZeropoint(zeropoint='AB')
    # Confirm that if we use the default thinning kwargs, then the zeropoint for the thinned
    # bandpass is the same (exactly) as the original.
    bp_th = bp.thin()
    np.testing.assert_equal(bp.zeropoint, bp_th.zeropoint,
                            "Zeropoint not preserved after thinning with defaults")
    bp_tr = bp.truncate(relative_throughput=1.e-4)
    np.testing.assert_equal(bp.zeropoint, bp_tr.zeropoint,
                            "Zeropoint not preserved after truncating with defaults")

    # Confirm that if we explicit set the kwarg to clear the zeropoint when thinning or truncating,
    # or if we truncate using blue_limit or red_limit, then the new bandpass has no zeropoint
    bp_th = bp.thin(preserve_zp = False)
    assert bp_th.zeropoint is None, \
        "Zeropoint erroneously preserved after thinning with preserve_zp=False"
    bp_tr = bp.truncate(preserve_zp = False)
    assert bp_tr.zeropoint is None, \
        "Zeropoint erroneously preserved after truncating with preserve_zp=False"
    bp_tr = bp.truncate(red_limit = 600.)
    assert bp_tr.zeropoint is None, \
        "Zeropoint erroneously preserved after truncating with explicit red_limit"
    bp_tr = bp.truncate(blue_limit = 550.)
    assert bp_tr.zeropoint is None, \
        "Zeropoint erroneously preserved after truncating with explicit blue_limit"

    with assert_raises(galsim.GalSimValueError):
        bp_tr = bp.truncate(preserve_zp = 'False')
    with assert_raises(galsim.GalSimValueError):
        bp_tr = bp.truncate(preserve_zp = 43)
    with assert_raises(galsim.GalSimIncompatibleValuesError):
        galsim.Bandpass('1', 'nm', 400, 550).truncate(relative_throughput=1.e-4)


@timer
def test_truncate_inputs():
    """Test that bandpass truncation respects certain sanity constraints on the inputs."""
    # Don't allow truncation via two different methods.
    bp = galsim.Bandpass(os.path.join(datapath, 'LSST_r.dat'), 'nm')
    assert_raises(ValueError, bp.truncate, relative_throughput=1.e-4, blue_limit=500.)

    # If blue_limit or red_limit is supplied, don't allow values that are outside the original
    # wavelength range.
    assert_raises(ValueError, bp.truncate, blue_limit=0.9*bp.blue_limit)
    assert_raises(ValueError, bp.truncate, red_limit=1.1*bp.red_limit)


if __name__ == "__main__":
    runtests(__file__)
