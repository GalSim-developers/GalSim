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
from astropy import units, constants
import warnings
import time

import galsim
from galsim_test_helpers import *
from galsim import trapz
from pathlib import Path


bppath = os.path.join(galsim.meta_data.share_dir, "bandpasses")
sedpath = os.path.join(galsim.meta_data.share_dir, "SEDs")


@timer
def test_SED_basic():
    """Basic tests of SED functionality
    """
    c = constants.c.to('nm / s').value # speed of light
    h = constants.h.to('erg s').value # Planck's constant
    nm_w = np.arange(10,1002,10)
    A_w = np.arange(100,10002,100)

    # All of these should be equivalent.  Flat spectrum with F_lambda = 200 erg/s/cm^2/nm
    warnings.simplefilter('ignore', units.UnitsWarning)
    s_list = [
        galsim.SED(spec=lambda x: 200., flux_type='flambda', wave_type='nm'),
        galsim.SED(spec='200', flux_type='flambda', wave_type='nanometers'),
        galsim.SED('200', wave_type='nanometers', flux_type='flambda'),
        galsim.SED('200', 'nm', 'flambda', fast=False),
        galsim.SED('np.sqrt(4.e4)', units.nm, units.erg/(units.s * units.cm**2 * units.nm)),
        galsim.SED('numpy.sqrt(4.e4)', units.Unit('nm'), 'flambda'),
        galsim.SED('math.sqrt(4.e4) * 1.e9', units.Unit('m'), units.Unit('erg/s/cm^2/m')),
        # 200 erg/nm / 10 A/nm = 20 erg/A
        galsim.SED(spec='20', flux_type='flambda', wave_type='Angstroms'),
        # 200 erg/nm / (hc/w erg/photon) = 200 w/hc photons/nm
        galsim.SED(spec='200 * wave / %r'%(h*c), wave_type='NANOmeters', flux_type='fphotons'),
        # 200 erg/nm / (hc/w erg/photon) / 10 A/nm = 20 (w in A)/hc photons/A
        galsim.SED(spec='20 * (wave/10) / %r'%(h*c), flux_type='fphotons', wave_type='Ang'),
        # 200 erg/nm / (c/w^2 Hz/nm) = 200 w^2/c erg/Hz
        galsim.SED(spec='200 * wave**2 / %r'%c, flux_type='fnu', wave_type='nm'),
        galsim.SED(spec='200 * (wave/10)**2 / %r'%c, flux_type='fnu', wave_type='A'),
        galsim.SED(galsim.LookupTable([1,1e3],[200,200], interpolant='linear'),
                   wave_type='nanometers', flux_type='flambda'),
        galsim.SED(galsim.LookupTable([1,1e4],[20,20], interpolant='linear'),
                   wave_type='ang', flux_type='flambda'),
        galsim.SED(galsim.LookupTable([1,1e3],[200/(h*c),2e5/(h*c)], interpolant='linear'),
                   flux_type='fphotons', wave_type='nm'),
        galsim.SED(galsim.LookupTable([1,1e4],[2/(h*c),2e4/(h*c)], interpolant='linear'),
                   flux_type='fphotons', wave_type='A'),
        galsim.SED(galsim.LookupTable([1,1e3],[200/c,2e8/c], interpolant='linear',
                                      x_log=True, f_log=True),
                   flux_type='fnu', wave_type='nanometers'),
        galsim.SED(galsim.LookupTable([1,1e4],[2/c,2e8/c], interpolant='linear',
                                      x_log=True, f_log=True),
                   flux_type='fnu', wave_type='A'),
        galsim.SED(galsim.LookupTable(nm_w, 200.*np.ones(100)), wave_type='nanometers',
                   flux_type='flambda'),
        galsim.SED(galsim.LookupTable(A_w, 20.*np.ones(100)), wave_type=units.Unit('Angstrom'),
                flux_type=units.Unit('erg/s/cm^2/Angstrom')),
        galsim.SED(galsim.LookupTable(nm_w, 200.*nm_w/(h*c)), flux_type='fphotons', wave_type='nm'),
        galsim.SED(galsim.LookupTable(A_w, 2.*A_w/(h*c)), wave_type=units.Unit('Angstrom'),
                flux_type=units.Unit('photon/s/cm^2/Angstrom')),
        galsim.SED(galsim.LookupTable(nm_w, 200.*nm_w**2/c), flux_type='fnu',
                   wave_type='nanometers'),
        galsim.SED(galsim.LookupTable(A_w, 2.*A_w**2/c), wave_type=units.Unit('Angstrom'),
                flux_type=units.Unit('erg/s/cm^2/Hz')),
        galsim.SED('200*wave**3/%r'%(h*c**2), 'nm', units.Unit('ph/s/cm^2/Hz')),
        galsim.SED('0.2*wave**3/%r'%(h*c**2), 'A', units.Unit('ph/s/cm^2/Hz')),
        galsim.SED('2.e33*wave**3/%r'%(h*c**2), units.Unit('m'), units.Unit('ph/s/m^2/Hz')),
        galsim.SED(galsim.LookupTable([1, 100-1.e-10, 100, 1000, 1000+1.e-10, 2000],
                                      [0., 0., 200., 200., 0., 0.], interpolant='linear'),
                   wave_type='nm', flux_type='flambda'),
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
        galsim.SED('1000', 'nm', 'flambda', redshift=4),
        galsim.SED(galsim.LookupTable([1,1e4],[100,100], interpolant='linear'),
                   wave_type='ang', flux_type='flambda').atRedshift(4.0),
        galsim.SED('1000', 'nm', 'flambda').atRedshift(4.0),
    ]

    for k,s in enumerate(s_list):
        print(k,' s = ', s)
        assert s.spectral
        assert not s.dimensionless
        np.testing.assert_almost_equal(s(400)*h*c/400, 200, decimal=10)
        np.testing.assert_almost_equal(s(900)*h*c/900, 200, decimal=10)
        waves = np.arange(700,800,10)
        np.testing.assert_array_almost_equal(s(waves) * h*c/waves, 200, decimal=10)

        if k < len(s_list)-3:
            np.testing.assert_equal(s.redshift, 0.)
        else:
            np.testing.assert_almost_equal(s.redshift, 4.)

        # Not picklable when the original spec is a lambda.
        # This is just true for the first (explicit lambda) and last (atRedshift with something
        # that had to be converted into a lambda).
        if isinstance(s._orig_spec, type(lambda: None)):
            print('\nSkip pickle test for k=%d, since spec is %s\n'%(k,s._spec))
        else:
            check_pickle(s, lambda x: (x(470), x(490), x(910)) )
            check_pickle(s)

    # Check some dimensionless spectra
    d_list = [
        galsim.SED(spec=lambda x: 200., flux_type='1', wave_type='nm'),
        galsim.SED(spec='200', flux_type=units.dimensionless_unscaled, wave_type='nanometers'),
        galsim.SED(spec='200', flux_type='1', wave_type='Angstroms'),
        galsim.SED(spec='200', flux_type='1', wave_type=units.Unit('m')),
        galsim.SED(spec='200', flux_type='1', wave_type=units.Unit('km'), fast=False),
        galsim.SED(galsim.LookupTable([1,1e3],[200,200], interpolant='linear'),
                   wave_type='nanometers', flux_type='1'),
        galsim.SED(galsim.LookupTable(A_w, 200.*np.ones(100)), flux_type='1', wave_type='A'),
        galsim.SED(galsim.LookupTable([1, 100-1.e-10, 100, 1000, 1000+1.e-10, 2000],
                                      [0., 0., 200., 200., 0., 0.], interpolant='linear'),
                   wave_type='nm', flux_type='1'),
    ]
    for k,s in enumerate(d_list):
        print(k,' s = ', s)
        assert not s.spectral
        assert s.dimensionless
        np.testing.assert_almost_equal(s(400), 200, decimal=10)
        np.testing.assert_almost_equal(s(900), 200, decimal=10)
        waves = np.arange(700,800,10)
        np.testing.assert_array_almost_equal(s(waves), 200, decimal=10)

        np.testing.assert_equal(s.redshift, 0.)

        # Only the first one is not picklable
        if k > 0:
            check_pickle(s, lambda x: (x(470), x(490), x(910)) )
            check_pickle(s)


@timer
def test_SED_add():
    """Check that SEDs add like I think they should...
    """
    for z in [0, 0.2, 0.4]:
        a = galsim.SED(galsim.LookupTable([1,2,3,4,5], [1.1,2.2,3.3,4.4,5.5]),
                       wave_type='nm', flux_type='fphotons')
        b = galsim.SED(galsim.LookupTable([1.1,2.2,3.0,4.4,5.5], [1.11,2.22,3.33,4.44,5.55]),
                       wave_type='nm', flux_type='fphotons')
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

    # Adding together two SEDs with different redshifts should fail.
    d = b.atRedshift(0.1)
    with assert_raises(galsim.GalSimIncompatibleValuesError):
        b + d
    with assert_raises(galsim.GalSimIncompatibleValuesError):
        d + b

    # Can't add incompatible spectral types
    a = a.atRedshift(0)
    b = a.atRedshift(0)
    c = galsim.SED(2.0, 'nm', '1')
    with assert_raises(galsim.GalSimIncompatibleValuesError):
        a + c
    with assert_raises(galsim.GalSimIncompatibleValuesError):
        c + a
    with assert_raises(galsim.GalSimIncompatibleValuesError):
        b + c
    with assert_raises(galsim.GalSimIncompatibleValuesError):
        c + b



@timer
def test_SED_sub():
    """Check that SEDs subtract like I think they should...
    """
    for z in [0, 0.2, 0.4]:
        a = galsim.SED(galsim.LookupTable([1,2,3,4,5], [1.1,2.2,3.3,4.4,5.5]),
                       wave_type='nm', flux_type='fphotons')
        b = galsim.SED(galsim.LookupTable([1.1,2.2,3.0,4.4,5.5], [1.11,2.22,3.33,4.44,5.55]),
                       wave_type='nm', flux_type='fphotons')
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

    # Subracting two SEDs with different redshifts should fail.
    d = b.atRedshift(0.1)
    with assert_raises(ValueError):
        b.__sub__(d)


@timer
def test_SED_mul():
    """Check that SEDs multiply like I think they should...
    """
    sed0 = galsim.SED(galsim.LookupTable([1,2,3,4,5], [1.1,2.2,3.3,4.4,5.5]),
                      wave_type='nm', flux_type='fphotons')
    sed1 = galsim.SED(lambda nu: nu**2, wave_type=units.Hz, flux_type='fnu', fast=False)
    sed2 = galsim.SED(17.0, wave_type='ang', flux_type='1')

    for sed, z in zip( [sed0, sed1, sed2], [0, 0.2, 0.4] ):
        a = sed.atRedshift(z)

        # SED multiplied by function
        b = lambda w: w**2
        c = a*b
        x = 3.0
        np.testing.assert_almost_equal(c(x), a(x) * b(x), 10,
                                       err_msg="Found wrong value in SED.__mul__")

        # function multiplied by SED
        c = b*a
        np.testing.assert_almost_equal(c(x), a(x) * b(x), 10,
                                       err_msg="Found wrong value in SED.__rmul__")

        # SED multiplied by scalar
        d = a*4.2
        np.testing.assert_almost_equal(d(x), a(x) * 4.2, 10,
                                       err_msg="Found wrong value in SED.__mul__")
        if sed is sed0: check_pickle(d)

        # assignment multiplication
        d *= 2
        np.testing.assert_almost_equal(d(x), a(x) * 4.2 * 2, 10,
                                       err_msg="Found wrong value in SED.__mul__")
        if sed is sed0: check_pickle(d)

        # SED multiplied by dimensionless, constant SED
        e = galsim.SED(2.0, 'nm', '1')
        f = a*e
        np.testing.assert_almost_equal(f(x), a(x) * e(x), 10,
                                       err_msg="Found wrong value in SED.__mul__")
        f2 = e*a
        np.testing.assert_almost_equal(f2(x), e(x) * a(x), 10,
                                       err_msg="Found wrong value in SED.__mul__")
        if sed is sed0:
            check_pickle(f)
            check_pickle(f2)

        # SED multiplied by dimensionless, non-constant SED
        g = galsim.SED('wave', 'nm', '1')
        h = a*g
        np.testing.assert_almost_equal(h(x), a(x) * g(x), 10,
                                       err_msg="Found wrong value in SED.__mul__")
        h2 = g*a
        np.testing.assert_almost_equal(h2(x), g(x) * a(x), 10,
                                       err_msg="Found wrong value in SED.__mul__")
        if sed is sed0:
            check_pickle(h)
            check_pickle(h2)

        assert_raises(TypeError, a.__mul__, 'invalid')


    sed1 = galsim.SED('1', 'nm', 'fphotons', redshift=1)
    sed2 = galsim.SED('2', 'nm', 'fphotons', redshift=2)
    sed3 = galsim.SED('3', 'nm', '1')
    sed4 = galsim.SED('4', 'nm', '1')
    with assert_raises(TypeError):
        sed1.__mul__(sed2)
    np.testing.assert_almost_equal((sed1*sed3)(100), 3.0, 10, "Found wrong value in SED.__mul__")
    np.testing.assert_almost_equal((sed2*sed4)(10), 8.0, 10, "Found wrong value in SED.__mul__")
    np.testing.assert_almost_equal((sed3*sed4)(30), 12.0, 10, "Found wrong value in SED.__mul__")

@timer
def test_SED_div():
    """Check that SEDs divide like I think they should...
    """
    a0_lt = galsim.SED(galsim.LookupTable([1,2,3,4,5], [1.1,2.2,3.3,4.4,5.5]),
                       wave_type='nm', flux_type='fphotons')
    a0_fn = galsim.SED('wave', wave_type='nm', flux_type='fphotons')
    for z in [0, 0.2, 0.4]:
        for a0 in [a0_lt, a0_fn]:
            a = a0.atRedshift(z)

            # SED divided by function
            b = lambda w: w**2
            c = a/b
            x = 3.0
            np.testing.assert_almost_equal(c(x), a(x)/b(x), 10,
                                           err_msg="Found wrong value in SED.__div__")

            # SED divided by scalar
            d = a/4.2
            np.testing.assert_almost_equal(d(x), a(x)/4.2, 10,
                                           err_msg="Found wrong value in SED.__div__")

            # assignment division
            d /= 2
            np.testing.assert_almost_equal(d(x), a(x)/4.2/2, 10,
                                           err_msg="Found wrong value in SED.__div__")
            if a0 is a0_lt:
                check_pickle(d)

            # SED divided by dimensionless SED
            e = galsim.SED('wave', 'nm', '1')
            d /= e
            np.testing.assert_almost_equal(d(x), a(x)/4.2/2/e(x), 10,
                                           err_msg="Found wrong value in SED.__div__")

    # Can't divide by spectral SED
    with assert_raises(galsim.GalSimSEDError):
        a0_lt / a0_fn
    with assert_raises(galsim.GalSimSEDError):
        a0_fn / a0_lt
    with assert_raises(galsim.GalSimSEDError):
        e / a0_lt
    with assert_raises(galsim.GalSimSEDError):
        e / a0_fn


@timer
def test_SED_atRedshift():
    """Check that SEDs redshift correctly.
    """
    a = galsim.SED(os.path.join(sedpath, 'CWW_E_ext.sed'), wave_type='ang', flux_type='flambda')
    bolo_bp = galsim.Bandpass('1', blue_limit=a.blue_limit, red_limit=a.red_limit, wave_type='nm')
    bolo_flux = a.calculateFlux(bolo_bp)
    print('bolo_flux = ',bolo_flux)
    for z1, z2 in zip([-0.01, -0.02, 0.5, 1.0, 1.4], [-0.2, 0.2, 1.0, 1.0, 1.0]):
        b = a.atRedshift(z1)
        c = b.atRedshift(z1) # same redshift, so should be no change
        d = c.atRedshift(z2) # do a relative redshifting from z1 to z2
        e = b.thin(rel_err=1.e-5)  # effectively tests that wave_list is handled correctly.
                                   # (Issue #520)
        for w in [350, 500, 650]:
            print('a(w) = ',a(w))
            print('b(w(1+z)) = ',b(w*(1.+z1)))
            print('c(w(1+z)) = ',c(w*(1.+z1)))
            print('d(w(1+z)) = ',d(w*(1.+z2)))
            print('e(w(1+z)) = ',e(w*(1.+z1)))
            np.testing.assert_almost_equal(a(w)/bolo_flux, b(w*(1.0+z1))/bolo_flux, 15,
                                           err_msg="error redshifting SED")
            np.testing.assert_almost_equal(a(w)/bolo_flux, c(w*(1.0+z1))/bolo_flux, 15,
                                           err_msg="error redshifting SED")
            np.testing.assert_almost_equal(a(w)/bolo_flux, d(w*(1.0+z2))/bolo_flux, 15,
                                           err_msg="error redshifting SED")
            np.testing.assert_almost_equal(a(w)/bolo_flux, e(w*(1.0+z1))/bolo_flux, 5,
                                           err_msg="error redshifting and thinning SED")
    with assert_raises(ValueError):
        a.atRedshift(-1.1)


@timer
def test_combine_wave_list():
    class A:
        def __init__(self, wave_list):
            self.wave_list = wave_list
            if self.wave_list:
                self.blue_limit = np.min(self.wave_list)
                self.red_limit = np.max(self.wave_list)
    a = A([1, 3, 5])
    b = A([2, 4, 6])
    c = A([2, 3, 4, 5])
    d = A([7, 8, 9])

    for a1, a2 in zip([a, a, b], [b, c, c]):
        wave_list, blue_limit, red_limit = (
            galsim.utilities.combine_wave_list(a1, a2))
        np.testing.assert_equal(wave_list, c.wave_list)
        np.testing.assert_equal(blue_limit, c.blue_limit)
        np.testing.assert_equal(red_limit, c.red_limit)
    with assert_raises(galsim.GalSimError):
        galsim.utilities.combine_wave_list(a, d)

    # Degenerate case works.
    sed = galsim.SED('CWW_Scd_ext.sed', wave_type='nm', flux_type='flambda')
    wave_list, blue_limit, red_limit = galsim.utilities.combine_wave_list([sed])
    np.testing.assert_equal(wave_list, sed.wave_list)
    np.testing.assert_equal(blue_limit, sed.blue_limit)
    np.testing.assert_equal(red_limit, sed.red_limit)

    wave_list, blue_limit, red_limit = galsim.utilities.combine_wave_list([])
    np.testing.assert_equal(wave_list, [])
    np.testing.assert_equal(blue_limit, 0)
    np.testing.assert_equal(red_limit, np.inf)

    # Doesn't know about our A class though.
    assert_raises(TypeError, galsim.utilities.combine_wave_list, a)


@timer
def test_SED_roundoff_guard():
    """Check that SED.__init__ roundoff error guard works. (Issue #520).
    """
    a = galsim.SED('CWW_Scd_ext.sed', wave_type='nanometers', flux_type='flambda')
    for z in np.arange(0.0, 0.5, 0.001):
        b = a.atRedshift(z)
        w1 = b.wave_list[0]
        w2 = b.wave_list[-1]
        np.testing.assert_allclose(a(w1/(1.0+z)), b(w1), rtol=1e-10,
                                   err_msg="error using wave_list limits in redshifted SED")
        np.testing.assert_allclose(a(w2/(1.0+z)), b(w2), rtol=1e-10,
                                   err_msg="error using wave_list limits in redshifted SED")


@timer
def test_SED_init():
    """Check that certain invalid SED initializations are trapped.
    """
    # These fail.
    assert_raises(ValueError, galsim.SED, spec="'eggs'", wave_type='A', flux_type='flambda')
    assert_raises(ValueError, galsim.SED, spec='blah', wave_type='nm', flux_type='flambda')
    assert_raises(ValueError, galsim.SED, spec='wave+',wave_type='nm', flux_type='flambda')
    assert_raises(ValueError, galsim.SED, spec='somewhere/a/file', wave_type='nm',
                  flux_type='flambda')
    assert_raises(ValueError, galsim.SED, spec='/somewhere/a/file', wave_type='nm',
                  flux_type='flambda')
    assert_raises(ValueError, galsim.SED, spec=lambda w:1.0, wave_type='bar', flux_type='flambda')
    assert_raises(TypeError, galsim.SED, spec=lambda w:1.0, wave_type='nm')
    assert_raises(TypeError, galsim.SED, spec=lambda w:1.0, flux_type='bar')
    assert_raises(TypeError, galsim.SED, spec=lambda w:1.0)
    assert_raises(ValueError, galsim.SED, spec='wave', wave_type=units.Hz, flux_type='2')
    assert_raises(galsim.GalSimSEDError, galsim.SED, 1.0, 'nm', 'fphotons')
    # These should succeed.
    galsim.SED(spec='wave', wave_type='nm', flux_type='flambda')
    galsim.SED(spec='wave/wave', wave_type='nm', flux_type='flambda')
    galsim.SED(spec=lambda w:1.0, wave_type='nm', flux_type='flambda')
    galsim.SED(spec='1./(wave-700)', wave_type='nm', flux_type='flambda')
    galsim.SED(spec='wave', wave_type=units.nm, flux_type='flambda')
    galsim.SED(spec='wave', wave_type=units.Hz, flux_type='flambda')
    galsim.SED(spec='wave', wave_type=units.Hz, flux_type='fphotons')
    galsim.SED(spec='wave', wave_type=units.Hz, flux_type=units.erg/(units.s*units.nm*units.m**2))
    galsim.SED(spec='wave', wave_type=units.Hz, flux_type=units.erg/(units.s*units.Hz*units.m**2))
    galsim.SED(spec='wave', wave_type=units.Hz,
               flux_type=units.astrophys.photon/(units.s * units.Hz * units.m**2))
    galsim.SED(spec='wave', wave_type=units.Hz, flux_type='1')
    galsim.SED(spec='wave', wave_type=units.Hz, flux_type=units.dimensionless_unscaled)

    # Also check for invalid calls
    foo = np.arange(10.)+1.
    sed = galsim.SED(galsim.LookupTable(foo,foo), wave_type=units.Hz, flux_type='flambda')
    assert_raises(ValueError, sed, 0.5)
    assert_raises(ValueError, sed, 12.0)
    assert_raises(ValueError, galsim.SED, '1', 'nm', units.erg/units.s)
    assert_raises(ValueError, galsim.SED, '1', 'nm', '2')

    # Check a few valid calls for when fast=False
    sed = galsim.SED(galsim.LookupTable(foo,foo), wave_type=units.GHz,
                     flux_type=units.erg/(units.s*units.Hz*units.m**2), fast=False)
    sed(1.5*units.GHz)
    sed(3e8/1.5)  # lambda = c/nu = 3e8 m/s / 1.5e9 Hz * 1.e9 nm/m
    sed(3e8/1.5*units.nm)

    # And check the redshift kwarg.
    foo = np.arange(10.)+1.
    sed = galsim.SED(galsim.LookupTable(foo,foo), wave_type='nm', flux_type='flambda', redshift=1.0,
                     fast=False)
    # outside good range of 2->20 should raise ValueError
    assert_raises(ValueError, sed, 1.5)
    assert_raises(ValueError, sed, 24.0)

    sed(3.5)
    sed(3.5*units.nm)


@timer
def test_SED_withFlux():
    """ Check that setting the flux works.
    """
    rband = galsim.Bandpass(os.path.join(bppath, 'LSST_r.dat'), 'nm')
    rband2 = galsim.Bandpass(Path(rband._orig_tp), 'nm')
    np.testing.assert_array_equal(rband.wave_list, rband2.wave_list)
    np.testing.assert_array_equal(rband(rband.wave_list), rband2(rband2.wave_list))

    for z in [0, 0.2, 0.4]:
        for fast in [True, False]:
            for sed in [
                galsim.SED('CWW_E_ext.sed', wave_type='ang', flux_type='flambda', fast=fast),
                galsim.SED(Path('CWW_E_ext.sed'), wave_type='ang', flux_type='flambda', fast=fast),
                galsim.SED('CWW_E_ext.sed', wave_type='ang', flux_type='flambda', fast=fast,
                           interpolant='spline'),
                galsim.SED('wave', wave_type='nm', flux_type='fphotons'),
                galsim.EmissionLine(620.0, 1.0) + 2*galsim.EmissionLine(450.0, 0.5)
            ]:
                if z != 0:
                    sed = sed.atRedshift(z)
                sed = sed.withFlux(1.0, rband)
                np.testing.assert_array_almost_equal(
                    sed.calculateFlux(rband), 1.0, 5,
                    "Setting SED flux failed."
                )

                # Should be almost equivalent to multiplying an SED * Bandpass and computing the
                # "bolometric" flux.  The above is a bit more accurate, since it correctly does
                # the integration of the product of two linear segments between each tabulated point.
                ab = sed * rband
                bolo_bp = galsim.Bandpass(
                    '1', blue_limit=ab.blue_limit, red_limit=ab.red_limit,
                    wave_type='nm'
                )
                np.testing.assert_array_almost_equal(
                    ab.calculateFlux(bolo_bp), 1.0, 3,
                    "Calculating SED flux from sed * bp failed."
                )

                # If one or the other table has finer wavelength gridding, then the agreement
                # will be better.  Check with finer gridding for rband.
                fine_wave = np.linspace(ab.blue_limit, ab.red_limit, 169101)
                rband_fine = galsim.Bandpass(
                    galsim.LookupTable(fine_wave, rband(fine_wave), 'linear'),
                    'nm'
                )
                ab = sed * rband_fine

                np.testing.assert_array_almost_equal(
                    ab.calculateFlux(bolo_bp), 1.0, 5,
                    "Calculating SED flux from sed * bp failed."
                )

                # Multiplying in the other order also works.
                ba = rband_fine * sed
                np.testing.assert_array_almost_equal(
                    ba.calculateFlux(bolo_bp), 1.0, 5,
                    "Calculating SED flux from sed * bp failed."
                )

    # Invalid for dimensionless SED
    flat = galsim.SED(2.0, 'nm', '1')
    with assert_raises(galsim.GalSimSEDError):
        flat.withFlux(1.0, rband)
    with assert_raises(galsim.GalSimSEDError):
        flat.calculateFlux(rband)


@timer
def test_SED_withFluxDensity():
    """ Check that setting the flux density works.
    """

    a0 = galsim.SED('CWW_E_ext.sed', wave_type='ang', flux_type='flambda')
    for z in [0, 0.2, 0.4]:
        a = a0.atRedshift(z)
        a = a.withFluxDensity(1.0, 500)
        np.testing.assert_array_almost_equal(
                a(500), 1.0, 5, "Setting SED flux density failed.")
        a = a.withFluxDensity(2.0, 5000*units.AA)
        np.testing.assert_array_almost_equal(
                a(500), 2.0, 5, "Setting SED flux density failed.")
        a = a.withFluxDensity(0.3*units.astrophys.photon/(units.s*units.cm**2*units.AA), 500)
        np.testing.assert_array_almost_equal(
                a(500), 3.0, 5, "Setting SED flux density failed.")

    # Invalid for dimensionless SED
    flat = galsim.SED(2.0, 'nm', '1')
    with assert_raises(galsim.GalSimSEDError):
        flat.withFluxDensity(1.0, 500)


@timer
def test_SED_calculateMagnitude():
    """ Check that magnitudes work as expected.
    """
    # Test that we can create a zeropoint with an SED, and that magnitudes for that SED are
    # then 0.0
    for z in [0, 0.2, 0.4]:
        sed = galsim.SED(spec='wave', wave_type='nm', flux_type='flambda')
        if z != 0:
            sed = sed.atRedshift(z)
        bandpass = galsim.Bandpass(galsim.LookupTable([1,2,3,4,5], [1,2,3,4,5]),
                                   'nm').withZeropoint(sed)
        np.testing.assert_almost_equal(sed.calculateMagnitude(bandpass), 0.0)
        # Try multiplying SED by 100 to verify that magnitude decreases by 5
        sed *= 100
        np.testing.assert_almost_equal(sed.calculateMagnitude(bandpass), -5.0)
        # Try setting zeropoint to a constant.
        bandpass = galsim.Bandpass(galsim.LookupTable([1,2,3,4,5], [1,2,3,4,5]),
                                   'nm').withZeropoint(6.0)
        np.testing.assert_almost_equal(sed.calculateMagnitude(bandpass),
                                       (sed*100).calculateMagnitude(bandpass)+5.0)
        # Try setting AB zeropoint
        bandpass = (galsim.Bandpass(galsim.LookupTable([1,2,3,4,5], [1,2,3,4,5]), 'nm')
                    .withZeropoint('AB'))
        np.testing.assert_almost_equal(sed.calculateMagnitude(bandpass),
                                       (sed*100).calculateMagnitude(bandpass)+5.0)

        # See if we can set a magnitude.
        sed = sed.withMagnitude(24.0, bandpass)
        np.testing.assert_almost_equal(sed.calculateMagnitude(bandpass), 24.0)

        # Test intended meaning of zeropoint.  I.e., that an object with magnitude equal to the
        # zeropoint will have a flux of 1.0.
        bandpass = galsim.Bandpass(galsim.LookupTable([1,2,3,4,5], [1,2,3,4,5]),
                                   'nm').withZeropoint(24.0)
        sed = sed.withMagnitude(bandpass.zeropoint, bandpass)
        np.testing.assert_almost_equal(sed.calculateFlux(bandpass), 1.0, 10)

    # See if Vega magnitudes work.
    # The following AB/Vega conversions are sourced from
    # http://www.astronomy.ohio-state.edu/~martini/usefuldata.html
    # Almost certainly, the LSST filters and the filters used on this website are not perfect
    # matches, but should give some idea of the expected conversion between Vega magnitudes and AB
    # magnitudes.  Except for u-band, the results are consistent to 0.1 magnitudes, which is
    # encouraging, but the true accuracy of the get/set magnitude algorithms is probably much better
    # than this.
    ugrizy_vega_ab_conversions = [0.91, -0.08, 0.16, 0.37, 0.54, 0.634]
    filter_names = 'ugrizy'
    sed = sed.atRedshift(0.0)
    for conversion, filter_name in zip(ugrizy_vega_ab_conversions, filter_names):
        filter_filename = os.path.join(bppath, 'LSST_{0}.dat'.format(filter_name))
        AB_bandpass = (galsim.Bandpass(filter_filename, 'nm')
                       .withZeropoint('AB'))
        vega_bandpass = (galsim.Bandpass(filter_filename, 'nm')
                         .withZeropoint('vega'))
        AB_mag = sed.calculateMagnitude(AB_bandpass)
        vega_mag = sed.calculateMagnitude(vega_bandpass)
        thresh = 0.3 if filter_name == 'u' else 0.1
        assert (abs((AB_mag - vega_mag) - conversion) < thresh)

    # Invalid for dimensionless SED
    flat = galsim.SED(2.0, 'nm', '1')
    with assert_raises(galsim.GalSimSEDError):
        flat.withMagnitude(24.0, bandpass)

    # Zeropoint needs to be set.
    bp = galsim.Bandpass(galsim.LookupTable([1,2,3,4,5], [1,2,3,4,5]), 'nm')
    with assert_raises(galsim.GalSimError):
        sed.withMagnitude(24.0, bp)
    with assert_raises(galsim.GalSimError):
        sed.calculateMagnitude(bp)


@timer
def test_redshift_calculateFlux():
    sed1 = galsim.SED(galsim.LookupTable([1,2,3,4,5], [1.1, 1.9, 1.4, 1.8, 2.0], 'linear'),
                      wave_type='nm', flux_type='fphotons')
    sed2 = galsim.SED(galsim.LookupTable([1,2,3,4,5], [1.1, 3.9, 1.4, 3.8, 2.0], 'spline'),
                      wave_type='nm', flux_type='fphotons')
    bp1 = galsim.Bandpass(galsim.LookupTable([4,6], [1,2]), wave_type='nm')
    bp2 = galsim.Bandpass(galsim.LookupTable([40,60], [1,2]), wave_type='Ang')

    for sed in [sed1, sed2]:
        for z in [0, 0.19, 0.2, 0.21, 2.5, 2.99, 3, 3.01, 4]:
            sedz = sed.atRedshift(z)
            if sedz.blue_limit > bp1.blue_limit or sedz.red_limit < bp1.red_limit:
                with assert_raises(ValueError):
                    sedz.calculateFlux(bp1)
                with assert_raises(ValueError):
                    sedz.calculateFlux(bp2)
            else:
                flux1 = sedz.calculateFlux(bp1)
                flux2 = sedz.calculateFlux(bp2)
                print('z = {} flux = {}, {}'.format(z, flux1, flux2))
                wave = np.linspace(bp1.blue_limit, bp1.red_limit, 10000)
                f = sedz(wave) * bp1(wave)
                flux3 = trapz(f, wave)
                np.testing.assert_allclose(flux1, flux3)
                np.testing.assert_allclose(flux2, flux3)

    # All analytic has easy to check answers
    sed = galsim.SED('(wave/500)**2', wave_type='nm', flux_type='fphotons')
    bp1 = galsim.Bandpass('1', blue_limit=500, red_limit=1000, wave_type='nm')
    bp2 = galsim.Bandpass(galsim.LookupTable(np.arange(500,1001),np.ones(501)), wave_type='nm')
    bp3 = galsim.Bandpass(galsim.LookupTable(np.arange(5000,10001),np.ones(5001)), wave_type='Ang')

    for z in [0, 0.19, 0.2, 0.21, 2.5, 2.99, 3, 3.01, 4]:
        sedz = sed.atRedshift(z)
        f1 = sedz.calculateFlux(bp1)
        f2 = sedz.calculateFlux(bp2)
        f3 = sedz.calculateFlux(bp3)
        print('z = {} flux = {}, {}'.format(z, f1, f2))
        np.testing.assert_allclose(f1, 7./3. * 500 / (1.+z)**2)
        np.testing.assert_allclose(f2, 7./3. * 500 / (1.+z)**2, rtol=1.e-6)
        np.testing.assert_allclose(f3, 7./3. * 500 / (1.+z)**2, rtol=1.e-7)

    # Time the flux calculation for a real SED through a non-trivial bandpass
    sed = galsim.SED('CWW_Sbc_ext.sed', 'nm', 'flambda')
    bp = galsim.Bandpass('ACS_wfc_F606W.dat', 'nm')
    t0 = time.time()
    flux0 = sed.calculateFlux(bp)
    t1 = time.time()
    flux1 = sed.atRedshift(1).calculateFlux(bp)
    t2 = time.time()
    print('z=0 disk in HST V band: flux = ',flux0, t1-t0)
    print('z=1 disk in HST V band: flux = ',flux1, t2-t1)

    # Regression tests
    np.testing.assert_allclose(flux0, 2.993792e+15, rtol=1.e-4)
    np.testing.assert_allclose(flux1, 4.395954e+14, rtol=1.e-4)

    # With spline, it's almost the same, but slightly different of course.
    sed = galsim.SED('CWW_Sbc_ext.sed', 'nm', 'flambda', interpolant='spline')
    bp = galsim.Bandpass('ACS_wfc_F606W.dat', 'nm', interpolant='spline')
    t0 = time.time()
    flux0 = sed.calculateFlux(bp)
    t1 = time.time()
    flux1 = sed.atRedshift(1).calculateFlux(bp)
    t2 = time.time()
    print('z=0 disk in HST V band: flux = ',flux0, t1-t0)
    print('z=1 disk in HST V band: flux = ',flux1, t2-t1)

    # Regression tests
    np.testing.assert_allclose(flux0, 3.023368e+15, rtol=1.e-4)
    np.testing.assert_allclose(flux1, 4.303569e+14, rtol=1.e-4)


@timer
def test_SED_calculateDCRMomentShifts():
    # compute some moment shifts
    sed = galsim.SED(os.path.join(sedpath, 'CWW_E_ext.sed'), 'nm', 'flambda')
    bandpass = galsim.Bandpass(os.path.join(bppath, 'LSST_r.dat'), 'nm')
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
    waves = np.linspace(bandpass.blue_limit, bandpass.red_limit, 1000)
    R = galsim.dcr.get_refraction(waves, 45.*galsim.degrees)
    Rnum = trapz(sed(waves) * bandpass(waves) * R, waves)
    den = trapz(sed(waves) * bandpass(waves), waves)
    rad2arcsec = galsim.radians / galsim.arcsec

    np.testing.assert_almost_equal(Rnum/den*rad2arcsec, Rbar[1]*rad2arcsec, 4)
    # and for the second moment, V, the numerator is:
    # \int{sed(\lambda) * bandpass(\lambda) * (R(\lambda) - Rbar)^2 d\lambda}
    Vnum = trapz(sed(waves) * bandpass(waves) * (R - Rnum/den)**2, waves)
    np.testing.assert_almost_equal(Vnum/den, V[1,1], 5)

    # Repeat with a function sed and bandpass, since different path in code
    sed2 = galsim.SED(spec=lambda x: 20.+5.*np.sin(x/400), flux_type='flambda', wave_type='nm')
    bp2 = galsim.Bandpass('1', 'nm', blue_limit=bandpass.blue_limit, red_limit=bandpass.red_limit)
    Rbar, V = sed2.calculateDCRMomentShifts(bp2, zenith_angle=45*galsim.degrees)
    Rbar2, V2 = sed2.calculateDCRMomentShifts(bp2, zenith_angle=45*galsim.degrees,
                                              parallactic_angle=180*galsim.degrees)
    np.testing.assert_array_almost_equal(Rbar, -Rbar2, 15)
    np.testing.assert_array_almost_equal(V, V2, 25)
    Rbar3, V3 = sed2.calculateDCRMomentShifts(bp2, zenith_angle=45*galsim.degrees,
                                              parallactic_angle=90*galsim.degrees)
    np.testing.assert_almost_equal(Rbar[0], Rbar3[1], 15)
    np.testing.assert_almost_equal(V[1,1], V3[0,0], 25)
    R = galsim.dcr.get_refraction(waves, 45.*galsim.degrees)
    Rnum = trapz(sed2(waves) * R, waves)
    den = trapz(sed2(waves), waves)
    np.testing.assert_almost_equal(Rnum/den, Rbar[1], 4)
    Vnum = trapz(sed2(waves) * (R - Rnum/den)**2, waves)
    np.testing.assert_almost_equal(Vnum/den, V[1,1], 5)

    dim = galsim.SED('200', 'nm', '1')
    assert_raises(TypeError, dim.calculateDCRMomentShifts, bandpass,
                  zenith_angle=0*galsim.degrees, parallactic_angle=0*galsim.degrees)
    assert_raises(TypeError, sed.calculateDCRMomentShifts, bandpass,
                  zenith_angle=0*galsim.degrees, parallactic_angle=0*galsim.degrees,
                  temperature=280, pressure=70, H2O_pressure=1.1,
                  invalid=True)
    # Works with temperature, pressure, and H2O_pressure
    sed.calculateDCRMomentShifts(bandpass,
                                 zenith_angle=0*galsim.degrees, parallactic_angle=0*galsim.degrees,
                                 temperature=280, pressure=70, H2O_pressure=1.1)


@timer
def test_SED_calculateSeeingMomentRatio():
    # compute a relative moment shift and compare to externally generated known result.
    sed = galsim.SED(os.path.join(sedpath, 'CWW_E_ext.sed'), 'nm', 'flambda')
    bandpass = galsim.Bandpass(os.path.join(bppath, 'LSST_r.dat'), 'nm')
    relative_size = sed.calculateSeeingMomentRatio(bandpass)

    # and now do the integral right here to compare.
    # \Delta r^2/r^2 = \frac{\int{sed(\lambda) * bandpass(\lambda) * (\lambda/500)^-0.4 d\lambda}}
    #                       {\int{sed(\lambda) * bandpass(\lambda) d\lambda}}
    waves = np.linspace(bandpass.blue_limit, bandpass.red_limit, 1000)
    num = trapz(sed(waves) * bandpass(waves) * (waves/500.0)**(-0.4), waves)
    den = trapz(sed(waves) * bandpass(waves), waves)
    np.testing.assert_almost_equal(relative_size, num/den, 5)

    # Repeat with a function sed and bandpass, since different path in code
    sed2 = galsim.SED(spec=lambda x: 20.+5.*np.sin(x/400), flux_type='flambda', wave_type='nm')
    bp2 = galsim.Bandpass('1', 'nm', blue_limit=bandpass.blue_limit, red_limit=bandpass.red_limit)
    relative_size = sed2.calculateSeeingMomentRatio(bp2)
    num = trapz(sed2(waves) * (waves/500.0)**(-0.4), waves)
    den = trapz(sed2(waves), waves)
    np.testing.assert_almost_equal(relative_size, num/den, 4)

    # Invalid for dimensionless SED
    flat = galsim.SED(2.0, 'nm', '1')
    with assert_raises(galsim.GalSimSEDError):
        flat.calculateSeeingMomentRatio(bandpass)

@timer
def test_SED_sampleWavelength():
    seed = 12345
    rng = galsim.UniformDeviate(seed)

    sed  = galsim.SED(galsim.LookupTable([1,2,3,4,5], [0.,1.,0.5,1.,0.]),
                      wave_type='nm', flux_type='fphotons')
    bandpass = galsim.Bandpass(galsim.LookupTable([1,2,3,4,5], [0,0,1,1,0], interpolant='linear'),
                               'nm')
    sedbp = sed*bandpass

    # Need to use 256 here to get the regression test to work, since that used to be the default.
    out = sed.sampleWavelength(3,None,npoints=256)
    print('out = ',out)
    np.testing.assert_equal(hasattr(sed,'_cache_deviate'),True,"Creating SED deviate cache failed.")
    np.testing.assert_equal(len(sed._cache_deviate),1,"Creating SED deviate failed.")

    out = sed.sampleWavelength(3,None,rng=seed, npoints=256)
    print('out = ',out)
    np.testing.assert_equal(len(sed._cache_deviate),1,"Accessing existing SED deviate failed.")

    test0 = np.array([ 4.15562438,  4.737775  ,  1.93594078])
    print('test0 = ',test0)
    np.testing.assert_array_almost_equal(out,test0,8,"Unexpected SED sample values.")

    out = sed.sampleWavelength(3,None,rng=rng, npoints=256)
    np.testing.assert_array_almost_equal(out,test0,8,"Failed to pass 'UniformDeviate'.")

    out = sed.sampleWavelength(3,bandpass,rng=seed, npoints=256)
    np.testing.assert_equal(len(sed._cache_deviate),2,"Creating new SED deviate failed.")

    test1 = np.array([ 4.16227593,  4.6166918 ,  2.95075946])
    np.testing.assert_array_almost_equal(out,test1,5,"Unexpected SED sample values.")

    out = sed.sampleWavelength(1e3,bandpass,rng=seed,npoints=256)
    np.testing.assert_equal(len(sed._cache_deviate),2,"Unexpected number of SED deviates.")
    np.testing.assert_equal(len(out),1e3,"Unexpected number of SED samples.")

    np.testing.assert_equal(np.sum(out > sedbp.red_limit),0,
                            "SED sample outside of function bounds.")
    np.testing.assert_equal(np.sum(out < sedbp.blue_limit),0,
                            "SED sample outside of function bounds.")

    out2 = sed.sampleWavelength(1e3,bandpass,rng=seed,npoints=512)
    np.testing.assert_equal(len(sed._cache_deviate),3,"Unexpected number of SED deviates.")
    np.testing.assert_almost_equal(out,out2,0,"SED samples using different npoints don't match "
                                   "to the nearest integer.")

    out2 = sed.sampleWavelength(1e3,bandpass,rng=seed)
    np.testing.assert_equal(len(sed._cache_deviate),4,"Unexpected number of SED deviates.")
    np.testing.assert_almost_equal(out,out2,0,"SED samples using different npoints don't match "
                                   "to the nearest integer.")

    def create_cdfs(sed,out,nbins=100):
        bins,step = np.linspace(sed.blue_limit,sed.red_limit,100000,retstep=True)
        centers = bins[:-1] + step/2.
        cdf = np.cumsum(sed(centers)*step)
        cdf /= cdf.max()

        test = np.linspace(sed.blue_limit,sed.red_limit,nbins)
        cdf1 = np.interp(test,centers,cdf)
        cdf2 = np.array([(out <= w).sum() for w in test],dtype=float)/len(out)
        return centers,(cdf1,cdf2)

    def create_counts(sed,out,nbins=100):
        bins,step = np.linspace(sed.blue_limit,sed.red_limit,nbins+1,retstep=True)
        centers = bins[:-1] + step/2.
        cts1,_ = np.histogram(out,nbins)

        cts2 = np.array([galsim.integ.int1d(sed,low,low+step,1e-3,1e-6) for low in bins[:-1]])
        cts2 *= (float(len(out))/cts2.sum())
        return centers,(cts1, cts2)

    # Test the output distribution
    out = sed.sampleWavelength(1e5,None,rng=seed,npoints=256)

    _,(cts1,cts2) = create_counts(sed,out)
    chisq = np.sum( (cts1 - cts2)**2 / cts1 )/len(cts1)
    np.testing.assert_almost_equal(chisq,1.0,1,"Sampled counts do not match input SED.")

    _,(cdf1,cdf2) = create_cdfs(sed,out)
    np.testing.assert_almost_equal(cdf1,cdf2,2,"Sampled CDF does not match input SED.")

    # Test redshift dependence
    z = 2.0
    sedz = sed.atRedshift(z)
    outz = sedz.sampleWavelength(1e5,None,rng=seed,npoints=256)
    np.testing.assert_almost_equal(out, outz/(1 + z), 8,
                                   "Redshifted wavelengths are not shifted by 1/(1+z).")

    _,(cts1,cts2) = create_counts(sedz,outz)
    chisq = np.sum( (cts1 - cts2)**2 / cts1 )/len(cts1)
    np.testing.assert_almost_equal(chisq, 1.0, 1,
                                   "Sampled counts do not match input redshifted SED.")

    _,(cdf1,cdf2) = create_cdfs(sedz,outz)
    np.testing.assert_almost_equal(cdf1, cdf2, 2,
                                   "Sampled CDF does not match input redshifted SED.")

@timer
def test_sampleWavelength_limits():
    # Troxel ran across a rare bug where the sampleWavelength could sometimes produce values
    # that were just below the blue limit of a bandpass.  This would then cause a RangeError
    # when drawing.
    #
    # e.g. galsim.errors.GalSimRangeError: Shooting photons outside the interpolated wave_list
    #      Value [1604.9999999999998] not in range [1605.0, 2090.0]
    #
    # Clearly this is a floating point accuracy issue.  The difference is of order 1.e-16.
    #
    # The problem sees to be that sampleWavelength samples in the rest frame, and then multiplies
    # by (1+z) at the end to get back to the intended range.  That divide and multiply
    # is not guaranteed to roundtrip properly, and you can get errors of order epsilon.
    import galsim.roman

    # N.B. I tried a bunch of random values for z, seed until one of them failed the assert below.
    # Even with the extrme SED concentration, this failure mode was still pretty rare.
    # But with these values, this test failed on commit e4e2c5d32ec4f925791f7cb8
    z = 2.0655996529385448
    seed = 1579718864

    bp = galsim.roman.getBandpasses()['F184']
    blue = bp.blue_limit
    red = bp.red_limit
    # Concentrate the SED right near the blue limit at the observed redshift.
    w1 = blue / (1.+z) - 1.e-12
    w2 = blue / (1.+z) + 1.e-12
    w3 = blue / (1.+z) + 2.e-12
    w4 = red / (1.+z)
    sed = galsim.SED(galsim.LookupTable([w1,w2,w3,w4], [1,1,1.e-200,1.e-200], 'linear'),
                     wave_type='nm', flux_type='flambda', redshift=z)

    rng = galsim.BaseDeviate(seed)
    waves = sed.sampleWavelength(10**6, bp, rng=rng)
    print('min waves = ',waves.min())
    print('max waves = ',waves.max())
    print('blue = ',blue)
    assert waves.min() >= blue

@timer
def test_fnu_vs_flambda():
    c = 2.99792458e17  # speed of light in nm/s
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
        sed1 = galsim.SED(galsim.LookupTable(waves, fnu), wave_type='nm', flux_type='fnu')
        sed2 = galsim.SED(galsim.LookupTable(waves, flambda), wave_type='nm', flux_type='flambda')
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


@timer
def test_ne():
    """ Check that inequality works as expected."""
    spec1 = lambda x: x/1000
    spec2 = galsim.LookupTable([400, 550], [0.4, 0.55], interpolant='linear')
    spec3 = '3'

    # These should all compare unequal.
    seds = [galsim.SED(spec1, wave_type='nm', flux_type='flambda'),
            galsim.SED(spec1, wave_type='A', flux_type='flambda'),
            galsim.SED(spec1, wave_type='nm', flux_type='fnu'),
            galsim.SED(spec1, 'nm', 'flambda', redshift=1.0),
            galsim.SED(spec2, 'nm', 'flambda'),
            galsim.SED(spec3, 'nm', 'flambda'),
            galsim.EmissionLine(500.0, 1.0),
        ]
    check_all_diff(seds)


@timer
def test_thin():
    for interpolant in ['linear', 'nearest', 'spline']:
        s = galsim.SED('CWW_E_ext.sed', wave_type='ang', flux_type='flambda',
                       fast=False, interpolant=interpolant)
        bp = galsim.Bandpass('1', 'nm', blue_limit=s.blue_limit, red_limit=s.red_limit)
        flux = s.calculateFlux(bp)
        print("Original number of SED samples = ",len(s.wave_list))
        for err in [1.e-2, 1.e-3, 1.e-4, 1.e-5]:
            print("Test err = ",err)
            thin_s = s.thin(rel_err=err, preserve_range=True, fast_search=False)
            thin_flux = thin_s.calculateFlux(bp)
            thin_err = (flux-thin_flux)/flux
            print("num samples with preserve_range = True, fast_search = False: ",
                  len(thin_s.wave_list))
            print("realized error = ",(flux-thin_flux)/flux)
            thin_s = s.thin(rel_err=err, preserve_range=True)
            thin_flux = thin_s.calculateFlux(bp)
            thin_err = (flux-thin_flux)/flux
            print("num samples with preserve_range = True: ",len(thin_s.wave_list))
            print("realized error = ",(flux-thin_flux)/flux)
            print('true flux = ',flux)
            print('thinned flux = ',thin_flux)
            print('err = ',thin_err)
            # The thinning algorithm guarantees a relative error of err for bolometric flux,
            # but not for any arbitrary bandpass.  When the target error is very small, it can
            # miss by a bit, especially for spline.  So test it with a little looser tolerance
            # than the target.
            test_err = err*4 if err <= 1.e-5 else err
            assert np.abs(thin_err) < test_err,\
                "Thinned SED failed accuracy goal, preserving range."
            thin_s = s.thin(rel_err=err, preserve_range=False)
            thin_flux = thin_s.calculateFlux(bp.truncate(thin_s.blue_limit, thin_s.red_limit))
            thin_err = (flux-thin_flux)/flux
            print("num samples with preserve_range = False: ",len(thin_s.wave_list))
            print("realized error = ",(flux-thin_flux)/flux)
            assert np.abs(thin_err) < test_err,\
                "Thinned SED failed accuracy goal, w/ range shrinkage."

    assert_raises(ValueError, s.thin, rel_err=-0.5)
    assert_raises(ValueError, s.thin, rel_err=1.5)
    # These errors aren't accessible from the SED or Bandpass calls.
    assert_raises(ValueError, galsim.utilities.thin_tabulated_values,
                  s.wave_list[3:], s._spec.getVals())
    assert_raises(ValueError, galsim.utilities.thin_tabulated_values,
                  s.wave_list[-1::-1], s._spec.getVals())

    # Check some pathalogical spectra to stress the thinning algorithm
    s = galsim.SED(galsim.LookupTable(range(6), [0,0,1,1,0,0]),'nm','1').thin()
    print('s = ',s)
    np.testing.assert_equal(s.wave_list, range(1,5))

    s = galsim.SED(galsim.LookupTable(range(6), [0,0,1,1,0,0]),'nm','1').thin(trim_zeros=False)
    print('s = ',s)
    np.testing.assert_equal(s.wave_list, range(6))

    s = galsim.SED(galsim.LookupTable(range(8), [1.e-8,1.e-6,1,1,1,1.e-6,1.e-10,1.e-100]),
            'nm','1').thin(preserve_range=False)
    print('s = ',s)
    np.testing.assert_equal(s.wave_list, range(1,6))

    s = galsim.SED(galsim.LookupTable(range(8), np.zeros(8)),'nm','1').thin()
    print('s = ',s)
    np.testing.assert_equal(s.wave_list, [0,7])

    s = galsim.SED(galsim.LookupTable(range(2), [1,1], interpolant='linear'),'nm','1').thin()
    print('s = ',s)
    np.testing.assert_equal(s.wave_list, [0,1])

    s = galsim.SED(galsim.LookupTable(range(3), [1, 1.e-20, 0], interpolant='linear'),
            'nm','1').thin(preserve_range=False)
    print('s = ',s)
    np.testing.assert_equal(s.wave_list, [0,1])

@timer
def test_broadcast():
    """ Check that constand SED broadcasts over waves.
    """
    # In response to issue #1228
    sed = galsim.SED(1, 'nm', '1')
    waves = [1, 2, 3]
    print(sed(waves))
    assert np.array_equal(sed(waves), np.ones(3))

    sed = galsim.SED('1', 'nm', 'fphotons')
    print(sed(waves))
    assert np.array_equal(sed(waves), np.ones(3))

    sed = galsim.SED('1', 'nm', 'flambda')
    print(sed(waves))
    assert np.array_equal(sed(waves), np.ones(3) * waves / (galsim.SED._h * galsim.SED._c))

    # Repeat with fast=False
    sed = galsim.SED(1, 'nm', '1', fast=False)
    waves = [1, 2, 3]
    print(sed(waves))
    assert np.array_equal(sed(waves), np.ones(3))

    sed = galsim.SED('1', 'nm', 'fphotons', fast=False)
    print(sed(waves))
    assert np.array_equal(sed(waves), np.ones(3))

    sed = galsim.SED('1', 'nm', 'flambda', fast=False)
    print(sed(waves))
    assert np.array_equal(sed(waves), np.ones(3) * waves / (galsim.SED._h * galsim.SED._c))


@timer
def test_SED_calculateFlux_inf():
    """ Check that calculateFlux works properly if the endpoint is inf.
    """
    # These two SEDs are functionally the same, but it didn't use to work with np.inf.
    sed1 = galsim.SED(
        galsim.LookupTable(
            [0, 621, 622, 623, 10000],
            [0, 0, 1, 0, 0],
            interpolant='linear'
        ),
        wave_type='nm',
        flux_type='fphotons'
    )
    sed2 = galsim.SED(
        galsim.LookupTable(
            [0, 621, 622, 623, np.inf],
            [0, 0, 1, 0, 0],
            interpolant='linear'
        ),
        wave_type='nm',
        flux_type='fphotons'
    )
    sed3 = galsim.EmissionLine(622)

    bp = galsim.Bandpass("LSST_r.dat", 'nm')
    flux1 = sed1.calculateFlux(bp)
    flux2 = sed2.calculateFlux(bp)
    flux3 = sed3.calculateFlux(bp)
    print('flux = ', flux1, flux2, flux3)
    assert flux1 == flux2
    assert flux1 == flux3


@timer
def test_emission_line():
    spectral = galsim.SED("vega.txt", wave_type='nm', flux_type='flambda')
    dimensionless = galsim.SED("1", wave_type='nm', flux_type='1')

    for wavelength, fwhm in [
        (500.0, 1.0),
        (650.0, 0.3),
        (700.0, 4.3)
    ]:
        for sed in [
            galsim.EmissionLine(wavelength, fwhm),
            galsim.EmissionLine(wavelength*10, fwhm*10, wave_type='ang'),
            galsim.EmissionLine(wavelength*1.e-9, fwhm*1.e-9, wave_type=units.m),
        ]:
            print(sed)
            np.testing.assert_allclose(sed.calculateFlux(None), 1.0)
            np.testing.assert_allclose((sed*2).calculateFlux(None), 2.0)
            np.testing.assert_allclose((3*sed).calculateFlux(None), 3.0)
            np.testing.assert_allclose((sed/2).calculateFlux(None), 0.5)
            np.testing.assert_allclose(sed(wavelength), 1./fwhm)
            np.testing.assert_allclose(sed(wavelength+fwhm), 0.0, rtol=0, atol=1e-11)
            np.testing.assert_allclose(sed(wavelength-fwhm), 0.0, rtol=0, atol=1e-11)
            np.testing.assert_allclose(sed(wavelength+fwhm/2), 0.5/fwhm, rtol=0, atol=1e-11)
            np.testing.assert_allclose(sed(wavelength-fwhm/2), 0.5/fwhm, rtol=0, atol=1e-11)
            np.testing.assert_allclose((sed*dimensionless).calculateFlux(None), 1.0)

            check_pickle(sed)
            check_pickle(sed*2)
            check_pickle(sed/3)
            check_pickle(sed.atRedshift(0.3))

            with np.testing.assert_raises(galsim.GalSimIncompatibleValuesError):
                sed * spectral
            with np.testing.assert_raises(galsim.GalSimSEDError):
                sed / spectral
            with np.testing.assert_raises(galsim.GalSimSEDError):
                spectral / sed

            z = 1.1
            sed = sed.atRedshift(z)
            assert sed is sed.atRedshift(z)
            np.testing.assert_allclose(sed.calculateFlux(None), (1+z))
            np.testing.assert_allclose(sed(wavelength*(1+z)), 1/fwhm)

            with np.testing.assert_raises(galsim.GalSimRangeError):
                sed.atRedshift(-2.0)
        with np.testing.assert_raises(galsim.GalSimValueError):
            galsim.EmissionLine(wavelength, fwhm, wave_type=units.Hz),


@timer
def test_flux_type_calculateFlux():
    sed1 = galsim.SED(
        galsim.LookupTable([1,2,3,4,5], [1.1, 1.9, 1.4, 1.8, 2.0]),
        wave_type='nm', flux_type='flambda'
    )
    sed2 = galsim.SED(
        galsim.LookupTable([1,2,3,4,5], [1.1, 1.9, 1.4, 1.8, 2.0]),
        wave_type='nm', flux_type='fnu'
    )
    sed3 = galsim.SED(
        galsim.LookupTable([1,2,3,4,5], [1.1, 1.9, 1.4, 1.8, 2.0]),
        wave_type='nm', flux_type='fphotons'
    )
    sed4 = galsim.SED(
        galsim.LookupTable([1,2,3,4,5], [1.1, 1.9, 1.4, 1.8, 2.0]),
        wave_type="nm", flux_type=units.Lsun / units.Hz / units.Mpc**2
    )
    flat_sed = galsim.SED(
        '1',
        wave_type="nm", flux_type="1"
    )
    bp = galsim.Bandpass(galsim.LookupTable([2,4], [1,2]), wave_type='nm')
    exp_gal = galsim.Exponential(half_light_radius=0.5, flux=1)

    for sed in [sed1, sed2, sed3, sed4]:
        flux1 = sed.calculateFlux(bp)
        flux2 = (1 * sed).calculateFlux(bp)
        flux3 = (exp_gal * sed).calculateFlux(bp)
        flux4 = (flat_sed * sed).calculateFlux(bp)
        print('flux = {}, {}, {}'.format(flux1, flux2, flux3))
        wave = np.linspace(bp.blue_limit, bp.red_limit, 10000)
        f = sed(wave) * bp(wave)
        flux5 = trapz(f, wave)
        np.testing.assert_allclose(flux1, flux2)
        np.testing.assert_allclose(flux1, flux3)
        np.testing.assert_allclose(flux1, flux4)
        np.testing.assert_allclose(flux1, flux5)


if __name__ == "__main__":
    runtests(__file__)
