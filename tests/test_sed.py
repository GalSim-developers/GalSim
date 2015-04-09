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

def test_SED_basic():
    """Basic tests of SED functionality
    """
    import time
    t1 = time.time()

    c = 2.99792458e17  # speed of light in nm/s
    h = 6.62606957e-27 # Planck's constant in erg seconds
    nm_w = np.arange(10,1002,10)
    A_w = np.arange(100,10002,100)

    # All of these should be equivalent.  Flat spectrum with F_lambda = 200 erg/nm
    s_list = [
        galsim.SED(spec=lambda x: 200.),
        galsim.SED(spec='200', flux_type='flambda', wave_type='nanometers'),
        galsim.SED('200'),
        galsim.SED('200', 'nm', 'flambda'),
        # 200 erg/nm / 10 A/nm = 20 erg/A
        galsim.SED(spec='20', wave_type='Angstroms'),
        # 200 erg/nm / (hc/w erg/photon) = 200 w/hc photons/nm
        galsim.SED(spec='200 * wave / %r'%(h*c), flux_type='fphotons'),
        # 200 erg/nm / (hc/w erg/photon) / 10 A/nm = 20 (w in A)/hc photons/A
        galsim.SED(spec='20 * (wave/10) / %r'%(h*c), flux_type='fphotons', wave_type='Ang'),
        # 200 erg/nm / (c/w^2 Hz/nm) = 200 w^2/c erg/Hz
        galsim.SED(spec='200 * wave**2 / %r'%c, flux_type='fnu'),
        galsim.SED(spec='200 * (wave/10)**2 / %r'%c, flux_type='fnu', wave_type='A'),
        galsim.SED(galsim.LookupTable([1,1e3],[200,200], interpolant='linear')),
        galsim.SED(galsim.LookupTable([1,1e4],[20,20], interpolant='linear'),
                   wave_type='ang'),
        galsim.SED(galsim.LookupTable([1,1e3],[200/(h*c),2e5/(h*c)], interpolant='linear'),
                   flux_type='fphotons'),
        galsim.SED(galsim.LookupTable([1,1e4],[2/(h*c),2e4/(h*c)], interpolant='linear'),
                   flux_type='fphotons', wave_type='A'),
        galsim.SED(galsim.LookupTable([1,1e3],[200/c,2e8/c], interpolant='linear', 
                                      x_log=True, f_log=True),
                   flux_type='fnu'),
        galsim.SED(galsim.LookupTable([1,1e4],[2/c,2e8/c], interpolant='linear', 
                                      x_log=True, f_log=True),
                   flux_type='fnu', wave_type='A'),
        galsim.SED(galsim.LookupTable(nm_w, 200.*np.ones(100)), flux_type='flambda'),
        galsim.SED(galsim.LookupTable(A_w, 20.*np.ones(100)), flux_type='flambda', wave_type='A'),
        galsim.SED(galsim.LookupTable(nm_w, 200.*nm_w/(h*c)), flux_type='fphotons'),
        galsim.SED(galsim.LookupTable(A_w, 2.*A_w/(h*c)), flux_type='fphotons', wave_type='A'),
        galsim.SED(galsim.LookupTable(nm_w, 200.*nm_w**2/c), flux_type='fnu'),
        galsim.SED(galsim.LookupTable(A_w, 2.*A_w**2/c), flux_type='fnu', wave_type='A'),
        galsim.SED(galsim.LookupTable([1, 100-1.e-10, 100, 1000, 1000+1.e-10, 2000],
                                      [0., 0., 200., 200., 0., 0.], interpolant='linear'))
    ]
    s_list += [
        s_list[9].thin(),
        s_list[10].thin(),
        s_list[11].thin(),
        s_list[12].thin(),
        s_list[13].thin(),
        s_list[14].thin(),
        s_list[15].thin(),
        s_list[15].thin(preserve_range=True),
        s_list[18].thin(),
        s_list[18].thin(preserve_range=True),
        s_list[21].thin(),
        s_list[21].thin(preserve_range=True),
        galsim.SED('1000', redshift=4),
        galsim.SED('1000').atRedshift(4.0),
    ]
 
    for k,s in enumerate(s_list):
        print k,' s = ',s
        np.testing.assert_almost_equal(s(400)*h*c/400, 200, decimal=10)
        np.testing.assert_almost_equal(s(900)*h*c/900, 200, decimal=10)
        waves = np.arange(700,800,10)
        np.testing.assert_array_almost_equal(s(waves) * h*c/waves, 200, decimal=10)

        if k < len(s_list)-2:
            np.testing.assert_equal(s.redshift, 0.)
        else:
            np.testing.assert_almost_equal(s.redshift, 4.)

        # Only the first one is not picklable
        if k > 0:
            do_pickle(s, lambda x: (x(470), x(490), x(910)) )
            do_pickle(s)

    t2 = time.time()
    print 'time for %s = %.2f'%(funcname(),t2-t1)

def test_SED_add():
    """Check that SEDs add like I think they should...
    """
    import time
    t1 = time.time()

    for z in [0, 0.2, 0.4]:
        a = galsim.SED(galsim.LookupTable([1,2,3,4,5], [1.1,2.2,3.3,4.4,5.5]),
                       flux_type='fphotons')
        b = galsim.SED(galsim.LookupTable([1.1,2.2,3.0,4.4,5.5], [1.11,2.22,3.33,4.44,5.55]),
                       flux_type='fphotons')
        if z != 0:
            a = a.atRedshift(z)
            b = b.atRedshift(z)
        c = a+b
        np.testing.assert_almost_equal(c.blue_limit, np.max([a.blue_limit, b.blue_limit]), 10,
                                       err_msg="Found wrong blue limit in SED.__add__")
        np.testing.assert_almost_equal(c.red_limit, np.min([a.red_limit, b.red_limit]), 10,
                                       err_msg="Found wrong red limit in SED.__add__")
        np.testing.assert_almost_equal(c(c.blue_limit), a(c.blue_limit) + b(c.blue_limit), 10,
                                       err_msg="Wrong sum in SED.__add__")
        np.testing.assert_almost_equal(c(c.red_limit), a(c.red_limit) + b(c.red_limit), 10,
                                       err_msg="Wrong sum in SED.__add__")
        x = 0.5 * (c.blue_limit + c.red_limit)
        np.testing.assert_almost_equal(c(x), a(x) + b(x), 10,
                                       err_msg="Wrong sum in SED.__add__")
        np.testing.assert_almost_equal(c.redshift, a.redshift, 10,
                                       err_msg="Wrong redshift in SED sum")
    try:
        # Adding together two SEDs with different redshifts should fail.
        d = b.atRedshift(0.1)
        np.testing.assert_raises(ValueError, b.__add__, d)
    except ImportError:
        print 'The assert_raises tests require nose'

    t2 = time.time()
    print 'time for %s = %.2f'%(funcname(),t2-t1)

def test_SED_sub():
    """Check that SEDs subtract like I think they should...
    """
    import time
    t1 = time.time()

    for z in [0, 0.2, 0.4]:
        a = galsim.SED(galsim.LookupTable([1,2,3,4,5], [1.1,2.2,3.3,4.4,5.5]),
                       flux_type='fphotons')
        b = galsim.SED(galsim.LookupTable([1.1,2.2,3.0,4.4,5.5], [1.11,2.22,3.33,4.44,5.55]),
                       flux_type='fphotons')
        if z != 0:
            a = a.atRedshift(z)
            b = b.atRedshift(z)
        c = a-b
        np.testing.assert_almost_equal(c.blue_limit, np.max([a.blue_limit, b.blue_limit]), 10,
                                       err_msg="Found wrong blue limit in SED.__sub__")
        np.testing.assert_almost_equal(c.red_limit, np.min([a.red_limit, b.red_limit]), 10,
                                       err_msg="Found wrong red limit in SED.__sub__")
        np.testing.assert_almost_equal(c(c.blue_limit), a(c.blue_limit) - b(c.blue_limit), 10,
                                       err_msg="Wrong difference in SED.__sub__")
        np.testing.assert_almost_equal(c(c.red_limit), a(c.red_limit) - b(c.red_limit), 10,
                                       err_msg="Wrong difference in SED.__sub__")
        x = 0.5 * (c.blue_limit + c.red_limit)
        np.testing.assert_almost_equal(c(x), a(x) - b(x), 10,
                                       err_msg="Wrong difference in SED.__sub__")
        np.testing.assert_almost_equal(c.redshift, a.redshift, 10,
                                       err_msg="Wrong redshift in SED difference")

    try:
        # Subracting two SEDs with different redshifts should fail.
        d = b.atRedshift(0.1)
        np.testing.assert_raises(ValueError, b.__sub__, d)
    except ImportError:
        print 'The assert_raises tests require nose'

    t2 = time.time()
    print 'time for %s = %.2f'%(funcname(),t2-t1)

def test_SED_mul():
    """Check that SEDs multiply like I think they should...
    """
    import time
    t1 = time.time()

    for z in [0, 0.2, 0.4]:
        a = galsim.SED(galsim.LookupTable([1,2,3,4,5], [1.1,2.2,3.3,4.4,5.5]),
                       flux_type='fphotons')
        if z != 0:
            a = a.atRedshift(z)
        b = lambda w: w**2
        # SED multiplied by function
        c = a*b
        x = 3.0
        np.testing.assert_almost_equal(c(x), a(x) * b(x), 10,
                                       err_msg="Found wrong value in SED.__mul__")
        # function multiplied by SED
        c = b*a
        np.testing.assert_almost_equal(c(x), a(x) * b(x), 10,
                                       err_msg="Found wrong value in SED.__rmul__")
        # SED multiplied by scalar
        d = c*4.2
        np.testing.assert_almost_equal(d(x), c(x) * 4.2, 10,
                                       err_msg="Found wrong value in SED.__mul__")
        # assignment multiplication
        d *= 2
        np.testing.assert_almost_equal(d(x), c(x) * 4.2 * 2, 10,
                                       err_msg="Found wrong value in SED.__mul__")

    t2 = time.time()
    print 'time for %s = %.2f'%(funcname(),t2-t1)

def test_SED_div():
    """Check that SEDs divide like I think they should...
    """
    import time
    t1 = time.time()

    for z in [0, 0.2, 0.4]:
        a = galsim.SED(galsim.LookupTable([1,2,3,4,5], [1.1,2.2,3.3,4.4,5.5]),
                       flux_type='fphotons')
        if z != 0:
            a = a.atRedshift(z)
        b = lambda w: w**2
        # SED divided by function
        c = a/b
        x = 3.0
        np.testing.assert_almost_equal(c(x), a(x)/b(x), 10,
                                       err_msg="Found wrong value in SED.__div__")
        # SED divided by scalar
        d = c/4.2
        np.testing.assert_almost_equal(d(x), c(x)/4.2, 10,
                                       err_msg="Found wrong value in SED.__div__")
        # assignment division
        d /= 2
        np.testing.assert_almost_equal(d(x), c(x)/4.2/2, 10,
                                       err_msg="Found wrong value in SED.__div__")

    t2 = time.time()
    print 'time for %s = %.2f'%(funcname(),t2-t1)

def test_SED_atRedshift():
    """Check that SEDs redshift correctly.
    """
    import time
    t1 = time.time()

    a = galsim.SED(os.path.join(datapath, 'CWW_E_ext.sed'), wave_type='ang')
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

    t2 = time.time()
    print 'time for %s = %.2f'%(funcname(),t2-t1)

def test_SED_roundoff_guard():
    """Check that SED.__init__ roundoff error guard works. (Issue #520).
    """
    import time
    t1 = time.time()

    a = galsim.SED(os.path.join(datapath, 'CWW_Scd_ext.sed'))
    for z in np.arange(0.0, 0.5, 0.001):
        b = a.atRedshift(z)
        w1 = b.wave_list[0]
        w2 = b.wave_list[-1]
        np.testing.assert_almost_equal(a(w1/(1.0+z)), b(w1), 10,
                                        err_msg="error using wave_list limits in redshifted SED")
        np.testing.assert_almost_equal(a(w2/(1.0+z)), b(w2), 10,
                                        err_msg="error using wave_list limits in redshifted SED")

    t2 = time.time()
    print 'time for %s = %.2f'%(funcname(),t2-t1)

def test_SED_init():
    """Check that certain invalid SED initializations are trapped.
    """
    import time
    t1 = time.time()

    try:
        # These fail.
        np.testing.assert_raises(ValueError, galsim.SED, spec='blah')
        np.testing.assert_raises(ValueError, galsim.SED, spec='wave+')
        np.testing.assert_raises(ValueError, galsim.SED, spec='somewhere/a/file')
        np.testing.assert_raises(ValueError, galsim.SED, spec='/somewhere/a/file')
        np.testing.assert_raises(ValueError, galsim.SED, spec=lambda w:1.0, wave_type='bar')
        np.testing.assert_raises(ValueError, galsim.SED, spec=lambda w:1.0, flux_type='bar')
    except ImportError:
        print 'The assert_raises tests require nose'
    # These should succeed.
    galsim.SED(spec='wave')
    galsim.SED(spec='wave/wave')
    galsim.SED(spec=lambda w:1.0)
    galsim.SED(spec='1./(wave-700)')

    # Also check for invalid calls
    foo = np.arange(10.)+1.
    sed = galsim.SED(galsim.LookupTable(foo,foo))
    try:
        np.testing.assert_raises(ValueError, sed, 0.5)
        np.testing.assert_raises(ValueError, sed, 12.0)
    except ImportError:
        print 'The assert_raises tests require nose'

    t2 = time.time()
    print 'time for %s = %.2f'%(funcname(),t2-t1)

def test_SED_withFlux():
    """ Check that setting the flux works.
    """
    import time
    t1 = time.time()

    rband = galsim.Bandpass(os.path.join(datapath, 'LSST_r.dat'))
    for z in [0, 0.2, 0.4]:
        a = galsim.SED(os.path.join(datapath, 'CWW_E_ext.sed'), wave_type='ang')
        if z != 0:
            a = a.atRedshift(z)
        a = a.withFlux(1.0, rband)
        np.testing.assert_array_almost_equal(a.calculateFlux(rband), 1.0, 5,
                                             "Setting SED flux failed.")
    t2 = time.time()
    print 'time for %s = %.2f'%(funcname(),t2-t1)


def test_SED_withFluxDensity():
    """ Check that setting the flux density works.
    """
    import time
    t1 = time.time()

    for z in [0, 0.2, 0.4]:
        a = galsim.SED(os.path.join(datapath, 'CWW_E_ext.sed'), wave_type='ang')
        if z != 0:
            a = a.atRedshift(z)
        a = a.withFluxDensity(1.0, 500)
        np.testing.assert_array_almost_equal(a(500), 1.0, 5,
                                             "Setting SED flux density failed.")

    t2 = time.time()
    print 'time for %s = %.2f'%(funcname(),t2-t1)

def test_SED_calculateMagnitude():
    """ Check that magnitudes work as expected.
    """
    import time
    t1 = time.time()

    # Test that we can create a zeropoint with an SED, and that magnitudes for that SED are
    # then 0.0
    for z in [0, 0.2, 0.4]:
        sed = galsim.SED(spec='wave')
        if z != 0:
            sed = sed.atRedshift(z)
        bandpass = galsim.Bandpass(galsim.LookupTable([1,2,3,4,5], [1,2,3,4,5])).withZeropoint(sed)
        np.testing.assert_almost_equal(sed.calculateMagnitude(bandpass), 0.0)
        # Try multiplying SED by 100 to verify that magnitude decreases by 5
        sed *= 100
        np.testing.assert_almost_equal(sed.calculateMagnitude(bandpass), -5.0)
        # Try setting zeropoint to a constant.
        bandpass = galsim.Bandpass(galsim.LookupTable([1,2,3,4,5], [1,2,3,4,5])).withZeropoint(6.0)
        np.testing.assert_almost_equal(sed.calculateMagnitude(bandpass),
                                       (sed*100).calculateMagnitude(bandpass)+5.0)
        # Try setting AB zeropoint
        bandpass = (galsim.Bandpass(galsim.LookupTable([1,2,3,4,5], [1,2,3,4,5]))
                    .withZeropoint('AB', effective_diameter=640.0, exptime=15.0))
        np.testing.assert_almost_equal(sed.calculateMagnitude(bandpass),
                                       (sed*100).calculateMagnitude(bandpass)+5.0)

        # See if we can set a magnitude.
        sed = sed.withMagnitude(24.0, bandpass)
        np.testing.assert_almost_equal(sed.calculateMagnitude(bandpass), 24.0)

        # Test intended meaning of zeropoint.  I.e., that an object with magnitude equal to the
        # zeropoint will have a flux of 1.0.
        bandpass = galsim.Bandpass(galsim.LookupTable([1,2,3,4,5], [1,2,3,4,5])).withZeropoint(24.0)
        sed = sed.withMagnitude(bandpass.zeropoint, bandpass)
        np.testing.assert_almost_equal(sed.calculateFlux(bandpass), 1.0, 10)

    # See if Vega magnitudes work.
    # The following AB/Vega conversions are sourced from
    # http://www.astronomy.ohio-state.edu/~martini/usefuldata.html
    # Almost certainly, the LSST filters and the filters used on this website are not perfect
    # matches, but should give some idea of the expected conversion between Vega magnitudes and AB
    # magnitudes.  The results are consistent to 0.1 magnitudes, which is encouraging, but the true
    # accuracy of the get/set magnitude algorithms is probably much better than this.
    ugrizy_vega_ab_conversions = [0.91, -0.08, 0.16, 0.37, 0.54, 0.634]
    filter_names = 'ugrizy'
    sed = sed.atRedshift(0.0)
    for conversion, filter_name in zip(ugrizy_vega_ab_conversions, filter_names):
        filter_filename = os.path.join(datapath, 'LSST_{0}.dat'.format(filter_name))
        AB_bandpass = (galsim.Bandpass(filter_filename)
                       .withZeropoint('AB', effective_diameter=640, exptime=15))
        vega_bandpass = (galsim.Bandpass(filter_filename)
                         .withZeropoint('vega', effective_diameter=640, exptime=15))
        AB_mag = sed.calculateMagnitude(AB_bandpass)
        vega_mag = sed.calculateMagnitude(vega_bandpass)
        assert (abs((AB_mag - vega_mag) - conversion) < 0.1)

    t2 = time.time()
    print 'time for %s = %.2f'%(funcname(),t2-t1)

def test_SED_calculateDCRMomentShifts():
    import time
    t1 = time.time()

    # compute some moment shifts
    sed = galsim.SED(os.path.join(datapath, 'CWW_E_ext.sed'))
    bandpass = galsim.Bandpass(os.path.join(datapath, 'LSST_r.dat'))
    Rbar, V = sed.calculateDCRMomentShifts(bandpass, zenith_angle=45*galsim.degrees)
    # now rotate parallactic angle 180 degrees, and see if the output makes sense.
    Rbar2, V2 = sed.calculateDCRMomentShifts(bandpass, zenith_angle=45*galsim.degrees,
                                             parallactic_angle=180*galsim.degrees)
    np.testing.assert_array_almost_equal(Rbar, -Rbar2, 15)
    np.testing.assert_array_almost_equal(V, V2, 25)
    # now rotate parallactic angle 90 degrees.
    Rbar3, V3 = sed.calculateDCRMomentShifts(bandpass, zenith_angle=45*galsim.degrees,
                                             parallactic_angle=90*galsim.degrees)
    np.testing.assert_almost_equal(Rbar[0], Rbar3[1], 15)
    np.testing.assert_almost_equal(V[1,1], V3[0,0], 25)
    # and now do the integral right here to compare.
    # \bar{R} = \frac{\int{sed(\lambda) * bandpass(\lambda) * R(\lambda) d\lambda}}
    #                {\int{sed(\lambda) * bandpass(\lambda) d\lambda}}
    # where sed is in units of photons/nm (which is the default)
    waves = np.union1d(sed.wave_list, bandpass.wave_list)
    R = galsim.dcr.get_refraction(waves, 45.*galsim.degrees)
    Rnum = np.trapz(sed(waves) * bandpass(waves) * R, waves)
    den = np.trapz(sed(waves) * bandpass(waves), waves)
    rad2arcsec = galsim.radians / galsim.arcsec

    np.testing.assert_almost_equal(Rnum/den*rad2arcsec, Rbar[1]*rad2arcsec, 4)
    # and for the second moment, V, the numerator is:
    # \int{sed(\lambda) * bandpass(\lambda) * (R(\lambda) - Rbar)^2 d\lambda}
    Vnum = np.trapz(sed(waves) * bandpass(waves) * (R - Rnum/den)**2, waves)
    np.testing.assert_almost_equal(Vnum/den, V[1,1], 5)

    t2 = time.time()
    print 'time for %s = %.2f'%(funcname(),t2-t1)

def test_SED_calculateSeeingMomentRatio():
    import time
    t1 = time.time()

    # compute a relative moment shift and compare to externally generated known result.
    sed = galsim.SED(os.path.join(datapath, 'CWW_E_ext.sed'))
    bandpass = galsim.Bandpass(os.path.join(datapath, 'LSST_r.dat'))
    relative_size = sed.calculateSeeingMomentRatio(bandpass)

    # and now do the integral right here to compare.
    # \Delta r^2/r^2 = \frac{\int{sed(\lambda) * bandpass(\lambda) * (\lambda/500)^-0.4 d\lambda}}
    #                       {\int{sed(\lambda) * bandpass(\lambda) d\lambda}}
    waves = np.union1d(sed.wave_list, bandpass.wave_list)
    num = np.trapz(sed(waves) * bandpass(waves) * (waves/500.0)**(-0.4), waves)
    den = np.trapz(sed(waves) * bandpass(waves), waves)

    np.testing.assert_almost_equal(relative_size, num/den, 5)

    t2 = time.time()
    print 'time for %s = %.2f'%(funcname(),t2-t1)

def test_fnu_vs_flambda():
    import time
    t1 = time.time()

    c = 2.99792458e17  # speed of light in nm/s
    h = 6.62606957e-27 # Planck's constant in erg seconds
    k = 1.3806488e-16  # Boltzmann's constant ergs per Kelvin
    nm_in_cm = 1e7
    # read these straight from Wikipedia
    def rayleigh_jeans_fnu(T, w):
        nu = c/w
        return 2*nu**2*k*T/c**2 * nm_in_cm**2 # should have units of erg/s/cm^2/Hz
    def rayleigh_jeans_flambda(T, w):
        return 2*c*k*T / w**4 * nm_in_cm**2 # should have units of erg/s/cm^2/nm
    waves = np.linspace(500, 1000, 100)
    fnu = rayleigh_jeans_fnu(5800, waves)
    flambda = rayleigh_jeans_flambda(5800, waves)

    for z in [0, 0.2, 0.4]:
        sed1 = galsim.SED(galsim.LookupTable(waves, fnu), flux_type='fnu')
        sed2 = galsim.SED(galsim.LookupTable(waves, flambda), flux_type='flambda')
        if z != 0:
            sed1 = sed1.atRedshift(z)
            sed2 = sed2.atRedshift(z)
        zwaves = waves * (1.0 + z)
        np.testing.assert_array_almost_equal(sed1(zwaves)/sed2(zwaves), np.ones(len(zwaves)), 10,
                                             err_msg="Check fnu & flambda consistency.")

        # Now also check that wavelengths in Angstroms work.
        waves_ang = waves * 10
        sed3 = galsim.SED(galsim.LookupTable(waves_ang, fnu), flux_type='fnu', wave_type='Ang')
        sed4 = galsim.SED(galsim.LookupTable(waves_ang, flambda/10.),
                          flux_type='flambda', wave_type='Ang')
        if z != 0:
            sed3 = sed3.atRedshift(z)
            sed4 = sed4.atRedshift(z)
        np.testing.assert_array_almost_equal(sed1(zwaves)/sed3(zwaves), 1., 10,
                                             err_msg="Check nm and Ang SED wavelengths consistency.")
        np.testing.assert_array_almost_equal(sed2(zwaves)/sed4(zwaves), 1., 10,
                                             err_msg="Check nm and Ang SED wavelengths consistency.")

    t2 = time.time()
    print 'time for %s = %.2f'%(funcname(),t2-t1)

if __name__ == "__main__":
    test_SED_basic()
    test_SED_add()
    test_SED_sub()
    test_SED_mul()
    test_SED_div()
    test_SED_atRedshift()
    test_SED_roundoff_guard()
    test_SED_init()
    test_SED_withFlux()
    test_SED_withFluxDensity()
    test_SED_calculateMagnitude()
    test_SED_calculateDCRMomentShifts()
    test_SED_calculateSeeingMomentRatio()
    test_fnu_vs_flambda()
