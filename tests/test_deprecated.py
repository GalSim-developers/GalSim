# Copyright (c) 2012-2017 by the GalSim developers team on GitHub
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

from __future__ import print_function
import os
import sys
import numpy as np

from galsim_test_helpers import *
from test_draw import CalculateScale

try:
    import galsim
except ImportError:
    path, filename = os.path.split(__file__)
    sys.path.append(os.path.abspath(os.path.join(path, "..")))
    import galsim

test_hlr = 1.8
test_fwhm = 1.8
test_sersic_n = [1.5, 2.5, 4, -4]  # -4 means use explicit DeVauc rather than n=4
test_scale = [1.8, 0.05, 0.002, 0.002]
test_spergel_nu = [-0.85, -0.5, 0.0, 0.85, 4.0]
test_spergel_scale = [20.0, 1.0, 1.0, 0.5, 0.5]
if __name__ == "__main__":
    # If doing a pytest run, we don't actually need to do all 4 sersic n values.
    # Two should be enough to notice if there is a problem, and the full list will be tested
    # when running python test_base.py to try to diagnose the problem.
    test_sersic_n = [1.5, -4]
    test_scale = [1.8, 0.002]

# some helper functions
def ellip_to_moments(e1, e2, sigma):
    a_val = (1.0 + e1) / (1.0 - e1)
    b_val = np.sqrt(a_val - (0.5*(1.0+a_val)*e2)**2)
    mxx = a_val * (sigma**2) / b_val
    myy = (sigma**2) / b_val
    mxy = 0.5 * e2 * (mxx + myy)
    return mxx, myy, mxy

def moments_to_ellip(mxx, myy, mxy):
    e1 = (mxx - myy) / (mxx + myy)
    e2 = 2*mxy / (mxx + myy)
    sig = (mxx*myy - mxy**2)**(0.25)
    return e1, e2, sig


def check_dep(f, *args, **kwargs):
    """Check that some function raises a GalSimDeprecationWarning as a warning, but not an error.
    """
    import warnings
    # Cause all warnings to always be triggered.
    # Important in case we want to trigger the same one twice in the test suite.
    warnings.simplefilter("always")

    # Check that f() raises a warning, but not an error.
    with warnings.catch_warnings(record=True) as w:
        res = f(*args, **kwargs)
    assert len(w) >= 1, "Calling %s did not raise a warning"%str(f)
    #print([ str(wk.message) for wk in w ])
    assert issubclass(w[0].category, galsim.GalSimDeprecationWarning)
    return res

def check_dep_tuple2(rhs):
    """Check that (x,y) = rhs raises a GalSimDeprecationWarning as a warning, but not an error.
    """
    #print('check dep tuple2: ',rhs)
    import warnings
    # Cause all warnings to always be triggered.
    # Important in case we want to trigger the same one twice in the test suite.
    warnings.simplefilter("always")

    with warnings.catch_warnings(record=True) as w:
        x,y = rhs
    #print('x,y = ',x,y)
    #print('w = ',w)
    assert len(w) >= 1, "Converting %s to a tuple did not raise a warning"%str(rhs)
    #print([ str(wk.message) for wk in w ])
    assert issubclass(w[0].category, galsim.GalSimDeprecationWarning)
    return x,y


@timer
def test_dep_bandpass():
    """Test the deprecated methods in galsim/deprecated/bandpass.py.
    """
    b = galsim.Bandpass(galsim.LookupTable([1.1,2.2,3.0,4.4,5.5], [1.11,2.22,3.33,4.44,5.55]), 'nm')
    d = lambda w: w**2

    check_dep(b.copy)

    # fn / Bandpass
    #e = d/b
    e = check_dep(b.__rdiv__, d)
    np.testing.assert_almost_equal(e(3.0), 3.0**2 / 3.33, 10,
                                   err_msg="Found wrong value in Bandpass.__rdiv__")
    np.testing.assert_array_almost_equal(e.wave_list, [1.1, 2.2, 3.0, 4.4, 5.5],
                                         err_msg="wrong wave_list in Bandpass.__rdiv__")

    # scalar / Bandpass
    #f = 1.21 / b
    f = check_dep(b.__rdiv__, 1.21)
    np.testing.assert_almost_equal(f(3.0), 1.21 / 3.33, 10,
                                   err_msg="Found wrong value in Bandpass.__rdiv__")
    np.testing.assert_array_almost_equal(f.wave_list, [1.1, 2.2, 3.0, 4.4, 5.5],
                                         err_msg="wrong wave_list in Bandpass.__rdiv__")

    check_dep(galsim.Bandpass, os.path.join(galsim.meta_data.share_dir, 'LSST_r.dat'), 'nm')


@timer
def test_dep_base():
    """Test the deprecated methods in galsim/deprecated/base.py
    """
    g = galsim.Gaussian(sigma=0.34)

    np.testing.assert_almost_equal(check_dep(g.nyquistDx), g.nyquistScale())

    check_dep(g.setFlux,flux=1.7)
    np.testing.assert_almost_equal(g.getFlux(), 1.7)

    check_dep(g.scaleFlux,flux_ratio=1.9)
    np.testing.assert_almost_equal(g.getFlux(), 1.7 * 1.9)

    g2 = g.expand(4.3)
    g3 = check_dep(g.createExpanded,scale=4.3)
    gsobject_compare(g3,g2)

    check_dep(g.applyExpansion,scale=4.3)
    gsobject_compare(g,g2)

    g2 = g.shear(g1=0.3, g2=-0.13)
    g3 = check_dep(g.createSheared,g1=0.3, g2=-0.13)
    gsobject_compare(g3,g2)

    check_dep(g.applyShear,g1=0.3, g2=-0.13)
    gsobject_compare(g,g2)

    g2 = g.dilate(0.54)
    g3 = check_dep(g.createDilated,scale=0.54)
    gsobject_compare(g3,g2)

    check_dep(g.applyDilation,scale=0.54)
    gsobject_compare(g,g2)

    g2 = g.magnify(1.1)
    g3 = check_dep(g.createMagnified,mu=1.1)
    gsobject_compare(g3,g2)

    check_dep(g.applyMagnification,mu=1.1)
    gsobject_compare(g,g2)

    g2 = g.lens(g1=0.31, g2=0.48, mu=0.22)
    g3 = check_dep(g.createLensed,g1=0.31, g2=0.48, mu=0.22)
    gsobject_compare(g3,g2)

    check_dep(g.applyLensing,g1=0.31, g2=0.48, mu=0.22)
    gsobject_compare(g,g2)

    g2 = g.rotate(38.4 * galsim.degrees)
    g3 = check_dep(g.createRotated,38.4 * galsim.degrees)
    gsobject_compare(g3,g2)

    check_dep(g.applyRotation,theta=38.4 * galsim.degrees)
    gsobject_compare(g,g2)

    g2 = g.transform(0.1, 1.09, -1.32, -0.09)
    g3 = check_dep(g.createTransformed, dudx=0.1, dudy=1.09, dvdx=-1.32, dvdy=-0.09)
    gsobject_compare(g3,g2)

    check_dep(g.applyTransformation, dudx=0.1, dudy=1.09, dvdx=-1.32, dvdy=-0.09)
    gsobject_compare(g,g2)

    g2 = g.shift(0.16, -0.79)
    g3 = check_dep(g.createShifted,dx=0.16, dy=-0.79)
    gsobject_compare(g3,g2)

    check_dep(g.applyShift,dx=0.16, dy=-0.79)
    gsobject_compare(g,g2)

    im1 = check_dep(g.draw)
    im2 = g.drawImage(method='no_pixel')
    np.testing.assert_almost_equal(im1.scale, im2.scale)
    np.testing.assert_equal(im1.bounds, im2.bounds)
    np.testing.assert_array_almost_equal(im1.array, im2.array)

    im1 = check_dep(g.draw,normalization='sb')
    im2 = g.drawImage(method='sb')
    np.testing.assert_almost_equal(im1.scale, im2.scale)
    np.testing.assert_equal(im1.bounds, im2.bounds)
    np.testing.assert_array_almost_equal(im1.array, im2.array)

    rng = galsim.BaseDeviate(123)
    im1 = check_dep(g.drawShoot,n_photons=1000, rng=rng.duplicate())
    im2 = g.drawImage(method='phot', n_photons=1000, rng=rng.duplicate())
    np.testing.assert_almost_equal(im1.scale, im2.scale)
    np.testing.assert_equal(im1.bounds, im2.bounds)
    np.testing.assert_array_almost_equal(im1.array, im2.array)

    im1, im1b = check_dep_tuple2(check_dep(g.drawK))
    im2, im2b = check_dep_tuple2(g.drawKImage())
    np.testing.assert_almost_equal(im1.scale, im2.scale)
    np.testing.assert_almost_equal(im1b.scale, im2b.scale)
    np.testing.assert_equal(im1.bounds, im2.bounds)
    np.testing.assert_equal(im1b.bounds, im2b.bounds)
    np.testing.assert_array_almost_equal(im1.array, im2.array)
    np.testing.assert_array_almost_equal(im1b.array, im2b.array)

    gsp1 = check_dep(galsim.GSParams, alias_threshold=0.1)
    gsp2 = galsim.GSParams(folding_threshold=0.1)
    np.testing.assert_equal(gsp1.folding_threshold, gsp2.folding_threshold)
    np.testing.assert_equal(gsp1.folding_threshold, check_dep(getattr, gsp2, 'alias_threshold'))

    test_gal = galsim.Gaussian(flux = 1., half_light_radius = test_hlr)
    test_gal_copy = check_dep(test_gal.copy)
    test_gal = galsim.Exponential(flux = 1., scale_radius = test_scale[0])
    test_gal_copy = check_dep(test_gal.copy)

    for n, scale in zip(test_sersic_n, test_scale) :
        if n == -4:
            test_gal1 = galsim.DeVaucouleurs(half_light_radius=test_hlr, flux=1.)
        else:
            test_gal1 = galsim.Sersic(n=n, half_light_radius=test_hlr, flux=1.)
        test_gal_copy = check_dep(test_gal1.copy)

    test_gal = galsim.Airy(lam_over_diam= 1./0.8, flux=1.)
    test_gal_copy = check_dep(test_gal.copy)

    test_beta = 2.
    test_gal = galsim.Moffat(flux=1, beta=test_beta, trunc=2.*test_fwhm,
                             fwhm = test_fwhm)
    test_gal_copy = check_dep(test_gal.copy)
    test_gal = galsim.Kolmogorov(flux=1., fwhm = test_fwhm)
    test_gal_copy = check_dep(test_gal.copy)

    for nu, scale in zip(test_spergel_nu, test_spergel_scale) :
        test_gal = galsim.Spergel(nu=nu, half_light_radius=test_hlr, flux=1.)
        test_gal_copy = check_dep(test_gal.copy)

@timer
def test_dep_bounds():
    """Test the deprecated methods in galsim/deprecated/bounds.py
    """
    bi = galsim.BoundsI(123,345,234,567)
    bf = galsim.BoundsD(123.,345.,234.,567.)

    for b in [bi, bf]:

        check_dep(b.setXMin,101)
        np.testing.assert_almost_equal(b.xmin, 101)
        np.testing.assert_almost_equal(b.xmax, 345)
        np.testing.assert_almost_equal(b.ymin, 234)
        np.testing.assert_almost_equal(b.ymax, 567)

        check_dep(b.setXMax,401)
        np.testing.assert_almost_equal(b.xmin, 101)
        np.testing.assert_almost_equal(b.xmax, 401)
        np.testing.assert_almost_equal(b.ymin, 234)
        np.testing.assert_almost_equal(b.ymax, 567)

        check_dep(b.setYMin,201)
        np.testing.assert_almost_equal(b.xmin, 101)
        np.testing.assert_almost_equal(b.xmax, 401)
        np.testing.assert_almost_equal(b.ymin, 201)
        np.testing.assert_almost_equal(b.ymax, 567)

        check_dep(b.setYMax,501)
        np.testing.assert_almost_equal(b.xmin, 101)
        np.testing.assert_almost_equal(b.xmax, 401)
        np.testing.assert_almost_equal(b.ymin, 201)
        np.testing.assert_almost_equal(b.ymax, 501)

        b2 = check_dep(b.addBorder,2)
        np.testing.assert_almost_equal(b.xmin, 101)
        np.testing.assert_almost_equal(b.xmax, 401)
        np.testing.assert_almost_equal(b.ymin, 201)
        np.testing.assert_almost_equal(b.ymax, 501)
        np.testing.assert_almost_equal(b2.xmin,  99)
        np.testing.assert_almost_equal(b2.xmax, 403)
        np.testing.assert_almost_equal(b2.ymin, 199)
        np.testing.assert_almost_equal(b2.ymax, 503)


@timer
def test_dep_chromatic():
    """Test the deprecated methods in galsim/deprecated/chromatic.py
    """
    g = galsim.Gaussian(sigma=0.34)
    sed = galsim.SED('wave**3', 'nm', 'flambda')
    obj = g * sed
    check_dep(obj.copy)
    band = galsim.Bandpass('1-((wave-700)/100)**2', 'nm', blue_limit=600., red_limit=800.)

    im1 = check_dep(obj.draw, bandpass=band)
    im2 = obj.drawImage(band, method='no_pixel')
    np.testing.assert_almost_equal(im1.scale, im2.scale)
    np.testing.assert_equal(im1.bounds, im2.bounds)
    np.testing.assert_array_almost_equal(im1.array, im2.array)

    im1 = check_dep(obj.draw, bandpass=band, normalization='sb')
    im2 = obj.drawImage(band, method='sb')
    np.testing.assert_almost_equal(im1.scale, im2.scale)
    np.testing.assert_equal(im1.bounds, im2.bounds)
    np.testing.assert_array_almost_equal(im1.array, im2.array)


@timer
def test_dep_correlatednoise():
    """Test the deprecated methods in galsim/deprecated/correlatednoise.py
    """
    rng = galsim.BaseDeviate(123)
    n1 = galsim.UncorrelatedNoise(variance=0.01, scale=1.3, rng=rng.duplicate())
    n2 = galsim.UncorrelatedNoise(variance=0.01, scale=1.3, rng=rng.duplicate())

    b = galsim.BoundsI(1,3,1,3)
    mat = check_dep(n1.calculateCovarianceMatrix,bounds=b, scale=1.9)
    # No replacement, so nothing to compare with.

    check_dep(n1.setVariance,variance=1.7)
    np.testing.assert_almost_equal(n1.getVariance(), 1.7)

    check_dep(n1.scaleVariance,variance_ratio=1.9)
    np.testing.assert_almost_equal(n1.getVariance(), 1.7 * 1.9)

    n2 = n2.withVariance(1.7 * 1.9)

    n2 = n2.expand(4.3)
    n3 = check_dep(n1.createExpanded,scale=4.3)
    gsobject_compare(n3,n2)

    check_dep(n1.applyExpansion,scale=4.3)
    gsobject_compare(n1,n2)

    n2 = n2.shear(g1=0.3, g2=-0.13)
    n3 = check_dep(n1.createSheared,g1=0.3, g2=-0.13)
    gsobject_compare(n3,n2)

    check_dep(n1.applyShear,g1=0.3, g2=-0.13)
    gsobject_compare(n1,n2)

    n2 = n2.dilate(0.54)
    n3 = check_dep(n1.createDilated,scale=0.54)
    gsobject_compare(n3,n2)

    check_dep(n1.applyDilation,scale=0.54)
    gsobject_compare(n1,n2)

    n2 = n2.magnify(1.1)
    n3 = check_dep(n1.createMagnified,mu=1.1)
    gsobject_compare(n3,n2)

    check_dep(n1.applyMagnification,mu=1.1)
    gsobject_compare(n1,n2)

    n2 = n2.lens(g1=0.31, g2=0.48, mu=0.22)
    n3 = check_dep(n1.createLensed,g1=0.31, g2=0.48, mu=0.22)
    gsobject_compare(n3,n2)

    check_dep(n1.applyLensing,g1=0.31, g2=0.48, mu=0.22)
    gsobject_compare(n1,n2)

    n2 = n2.rotate(38.4 * galsim.degrees)
    n3 = check_dep(n1.createRotated,38.4 * galsim.degrees)
    gsobject_compare(n3,n2)

    check_dep(n1.applyRotation,theta=38.4 * galsim.degrees)
    gsobject_compare(n1,n2)

    n2 = n2.transform(0.1, 1.09, -1.32, -0.09)
    n3 = check_dep(n1.createTransformed, dudx=0.1, dudy=1.09, dvdx=-1.32, dvdy=-0.09)
    gsobject_compare(n3,n2)

    check_dep(n1.applyTransformation, dudx=0.1, dudy=1.09, dvdx=-1.32, dvdy=-0.09)
    gsobject_compare(n1,n2)

    g = galsim.Gaussian(sigma=0.34)
    n2 = n2.convolvedWith(g)
    check_dep(n1.convolveWith,g)
    gsobject_compare(n1,n2)

    im1 = n1.drawImage()
    im2 = check_dep(n2.draw)
    np.testing.assert_almost_equal(im1.scale, im2.scale)
    np.testing.assert_equal(im1.bounds, im2.bounds)
    np.testing.assert_array_almost_equal(im1.array, im2.array)

    n1.whitenImage(im1)
    check_dep(n2.applyWhiteningTo,im2)
    np.testing.assert_array_almost_equal(im1.array, im2.array)


@timer
def test_dep_gsobject_ring():
    """Test building a GSObject from a ring test:
    """
    config = {
        'gal' : {
            'type' : 'Ring' ,
            'num' : 2,
            'first' : {
                'type' : 'Gaussian' ,
                'sigma' : 2 ,
                'ellip' : {
                    'type' : 'E1E2',
                    'e1' : { 'type' : 'List' ,
                             'items' : [ 0.3, 0.2, 0.8 ],
                             'index' : { 'type' : 'Sequence', 'repeat' : 2 }
                           },
                    'e2' : 0.1
                }
            }
        }
    }

    gauss = galsim.Gaussian(sigma=2)
    e1_list = [ 0.3, -0.3, 0.2, -0.2, 0.8, -0.8 ]
    e2_list = [ 0.1, -0.1, 0.1, -0.1, 0.1, -0.1 ]

    for k in range(6):
        config['obj_num'] = k
        gal1a = check_dep(galsim.config.BuildGSObject, config, 'gal')[0]
        gal1b = gauss.shear(e1=e1_list[k], e2=e2_list[k])
        gsobject_compare(gal1a, gal1b)

    config = {
        'gal' : {
            'type' : 'Ring' ,
            'num' : 10,
            'first' : { 'type' : 'Exponential', 'half_light_radius' : 2,
                        'ellip' : galsim.Shear(e2=0.3)
                      },
        }
    }

    disk = galsim.Exponential(half_light_radius=2).shear(e2=0.3)

    for k in range(25):
        config['obj_num'] = k
        gal2a = check_dep(galsim.config.BuildGSObject, config, 'gal')[0]
        gal2b = disk.rotate(theta = k * 18 * galsim.degrees)
        gsobject_compare(gal2a, gal2b)

    config = {
        'gal' : {
            'type' : 'Ring' ,
            'num' : 5,
            'full_rotation' : 360. * galsim.degrees,
            'first' : {
                'type' : 'Sum',
                'items' : [
                    { 'type' : 'Exponential', 'half_light_radius' : 2,
                      'ellip' : galsim.Shear(e2=0.3)
                    },
                    { 'type' : 'Sersic', 'n' : 3, 'half_light_radius' : 1.3,
                      'ellip' : galsim.Shear(e1=0.12,e2=-0.08)
                    }
                ]
            },
            'index' : { 'type' : 'Sequence', 'repeat' : 4 }
        }
    }

    disk = galsim.Exponential(half_light_radius=2).shear(e2=0.3)
    bulge = galsim.Sersic(n=3, half_light_radius=1.3).shear(e1=0.12,e2=-0.08)
    sum = disk + bulge

    for k in range(25):
        config['obj_num'] = k
        index = k // 4  # make sure we use integer division
        gal3a = check_dep(galsim.config.BuildGSObject, config, 'gal')[0]
        gal3b = sum.rotate(theta = index * 72 * galsim.degrees)
        gsobject_compare(gal3a, gal3b)

    # Check that the ring items correctly inherit their gsparams from the top level
    config = {
        'gal' : {
            'type' : 'Ring' ,
            'num' : 20,
            'full_rotation' : 360. * galsim.degrees,
            'first' : {
                'type' : 'Sum',
                'items' : [
                    { 'type' : 'Exponential', 'half_light_radius' : 2,
                      'ellip' : galsim.Shear(e2=0.3)
                    },
                    { 'type' : 'Sersic', 'n' : 3, 'half_light_radius' : 1.3,
                      'ellip' : galsim.Shear(e1=0.12,e2=-0.08)
                    }
                ]
            },
            'gsparams' : { 'maxk_threshold' : 1.e-2,
                           'folding_threshold' : 1.e-2,
                           'stepk_minimum_hlr' : 3 }
        }
    }

    config['obj_num'] = 0
    gal4a = check_dep(galsim.config.BuildGSObject, config, 'gal')[0]
    gsparams = galsim.GSParams(maxk_threshold=1.e-2, folding_threshold=1.e-2, stepk_minimum_hlr=3)
    disk = galsim.Exponential(half_light_radius=2, gsparams=gsparams).shear(e2=0.3)
    bulge = galsim.Sersic(n=3,half_light_radius=1.3, gsparams=gsparams).shear(e1=0.12,e2=-0.08)
    gal4b = disk + bulge
    gsobject_compare(gal4a, gal4b, conv=galsim.Gaussian(sigma=1))

    try:
        # Make sure they don't match when using the default GSParams
        disk = galsim.Exponential(half_light_radius=2).shear(e2=0.3)
        bulge = galsim.Sersic(n=3,half_light_radius=1.3).shear(e1=0.12,e2=-0.08)
        gal4c = disk + bulge
        np.testing.assert_raises(AssertionError,gsobject_compare, gal4a, gal4c,
                                 conv=galsim.Gaussian(sigma=1))
    except ImportError:
        print('The assert_raises tests require nose')


@timer
def test_dep_image():
    """Test that the old obsolete syntax still works (for now)
    """
    # This is the old version of the test_Image_basic function from version 1.0

    ntypes = 4  # Note: Most tests below only run through the first 4 types.
                # test_Image_basic tests all 6 types including the aliases.
    types = [np.int16, np.int32, np.float32, np.float64, int, float]
    tchar = ['S', 'I', 'F', 'D', 'I', 'D']

    ncol = 7
    nrow = 5
    test_shape = (ncol, nrow)  # shape of image arrays for all tests
    ref_array = np.array([
        [11, 21, 31, 41, 51, 61, 71],
        [12, 22, 32, 42, 52, 62, 72],
        [13, 23, 33, 43, 53, 63, 73],
        [14, 24, 34, 44, 54, 64, 74],
        [15, 25, 35, 45, 55, 65, 75] ]).astype(np.int16)

    check_dep(galsim.ImageViewS, ref_array.astype(np.int16))
    check_dep(galsim.ImageViewI, ref_array.astype(np.int32))
    check_dep(galsim.ImageViewF, ref_array.astype(np.float32))
    check_dep(galsim.ImageViewD, ref_array.astype(np.float64))
    check_dep(galsim.ConstImageViewS, ref_array.astype(np.int16))
    check_dep(galsim.ConstImageViewI, ref_array.astype(np.int32))
    check_dep(galsim.ConstImageViewF, ref_array.astype(np.float32))
    check_dep(galsim.ConstImageViewD, ref_array.astype(np.float64))

    for i in range(ntypes):
        array_type = types[i]
        check_dep(galsim.ImageView[array_type], ref_array.astype(array_type))
        check_dep(galsim.ConstImageView[array_type], ref_array.astype(array_type))
        # This next one is normally executed as im = galsim.Image[type]
        check_dep(galsim.image.MetaImage.__getitem__, galsim.Image, array_type)

    # The rest of this is taken from an older version of the Image class test suite that
    # tests the old syntax.  Might as well keep it.
    for i in range(ntypes):
        # Check basic constructor from ncol, nrow
        array_type = types[i]
        im1 = check_dep(galsim.image.MetaImage.__getitem__, galsim.Image, array_type)(ncol,nrow)
        bounds = galsim.BoundsI(1,ncol,1,nrow)

        assert im1.getXMin() == 1
        assert im1.getXMax() == ncol
        assert im1.getYMin() == 1
        assert im1.getYMax() == nrow
        assert im1.getBounds() == bounds
        assert im1.bounds == bounds

        # Check basic constructor from ncol, nrow
        # Also test alternate name of image type: ImageD, ImageF, etc.
        image_type = eval("galsim.Image"+tchar[i]) # Use handy eval() mimics use of ImageSIFD
        im2 = image_type(bounds)
        im2_view = im2.view()

        assert im2_view.getXMin() == 1
        assert im2_view.getXMax() == ncol
        assert im2_view.getYMin() == 1
        assert im2_view.getYMax() == nrow
        assert im2_view.bounds == bounds

        # Check various ways to set and get values
        for y in range(1,nrow):
            for x in range(1,ncol):
                im1.setValue(x,y, 100 + 10*x + y)
                im2_view.setValue(x,y, 100 + 10*x + y)

        for y in range(1,nrow):
            for x in range(1,ncol):
                assert check_dep(im1.at,x,y) == 100+10*x+y
                assert check_dep(im1.view().at,x,y) == 100+10*x+y
                assert check_dep(im2.at,x,y) == 100+10*x+y
                assert check_dep(im2_view.at,x,y) == 100+10*x+y
                im1.setValue(x,y, 10*x + y)
                im2_view.setValue(x,y, 10*x + y)
                assert im1(x,y) == 10*x+y
                assert im1.view()(x,y) == 10*x+y
                assert im2(x,y) == 10*x+y
                assert im2_view(x,y) == 10*x+y

        # Check view of given data
        im3_view = check_dep(galsim.ImageView[array_type], ref_array.astype(array_type))
        for y in range(1,nrow):
            for x in range(1,ncol):
                assert im3_view(x,y) == 10*x+y

        # Check shift ops
        im1_view = im1.view() # View with old bounds
        dx = 31
        dy = 16
        im1.shift(dx,dy)
        im2_view.setOrigin( 1+dx , 1+dy )
        im3_view.setCenter( (ncol+1)/2+dx , (nrow+1)/2+dy )
        shifted_bounds = galsim.BoundsI(1+dx, ncol+dx, 1+dy, nrow+dy)

        assert im1.bounds == shifted_bounds
        assert im2_view.bounds == shifted_bounds
        assert im3_view.bounds == shifted_bounds
        # Others shouldn't have changed
        assert im1_view.bounds == bounds
        assert im2.bounds == bounds
        for y in range(1,nrow):
            for x in range(1,ncol):
                assert im1(x+dx,y+dy) == 10*x+y
                assert im1_view(x,y) == 10*x+y
                assert im2(x,y) == 10*x+y
                assert im2_view(x+dx,y+dy) == 10*x+y
                assert im3_view(x+dx,y+dy) == 10*x+y


@timer
def test_dep_noise():
    """Test the deprecated methods in galsim/deprecated/noise.py
    """
    rng = galsim.BaseDeviate(123)
    gn = galsim.GaussianNoise(rng=rng, sigma=0.3)

    rng2 = galsim.BaseDeviate(999)
    check_dep(gn.setRNG, rng2)
    assert gn.rng is rng2

    check_dep(gn.setVariance, 1.7)
    np.testing.assert_almost_equal(gn.getVariance(), 1.7)
    check_dep(gn.scaleVariance, 1.9)
    np.testing.assert_almost_equal(gn.getVariance(), 1.7 * 1.9)

    check_dep(gn.setSigma, 2.3)
    np.testing.assert_almost_equal(gn.getSigma(), 2.3)

    pn = galsim.PoissonNoise(rng=rng, sky_level=0.3)
    check_dep(pn.setSkyLevel, 2.3)
    np.testing.assert_almost_equal(pn.getSkyLevel(), 2.3)

    cn = galsim.CCDNoise(rng=rng, gain=1.7, read_noise=0.5, sky_level=0.3)
    np.testing.assert_almost_equal(cn.getSkyLevel(), 0.3)
    np.testing.assert_almost_equal(cn.getGain(), 1.7)
    np.testing.assert_almost_equal(cn.getReadNoise(), 0.5)

    check_dep(cn.setSkyLevel, 2.3)
    np.testing.assert_almost_equal(cn.getSkyLevel(), 2.3)
    np.testing.assert_almost_equal(cn.getGain(), 1.7)
    np.testing.assert_almost_equal(cn.getReadNoise(), 0.5)

    check_dep(cn.setGain, 0.9)
    np.testing.assert_almost_equal(cn.getSkyLevel(), 2.3)
    np.testing.assert_almost_equal(cn.getGain(), 0.9)
    np.testing.assert_almost_equal(cn.getReadNoise(), 0.5)

    check_dep(cn.setReadNoise, 11)
    np.testing.assert_almost_equal(cn.getSkyLevel(), 2.3)
    np.testing.assert_almost_equal(cn.getGain(), 0.9)
    np.testing.assert_almost_equal(cn.getReadNoise(), 11)


@timer
def test_dep_random():
    """Test the deprecated methods in galsim/deprecated/random.py
    """
    rng = galsim.BaseDeviate(123)

    gd = galsim.GaussianDeviate(rng, mean=0.5, sigma=1.7)
    np.testing.assert_almost_equal(gd.getMean(), 0.5)
    np.testing.assert_almost_equal(gd.getSigma(), 1.7)

    check_dep(gd.setMean, 0.9)
    np.testing.assert_almost_equal(gd.getMean(), 0.9)
    np.testing.assert_almost_equal(gd.getSigma(), 1.7)

    check_dep(gd.setSigma, 2.3)
    np.testing.assert_almost_equal(gd.getMean(), 0.9)
    np.testing.assert_almost_equal(gd.getSigma(), 2.3)


    bd = galsim.BinomialDeviate(rng, N=7, p=0.7)
    np.testing.assert_almost_equal(bd.getN(), 7)
    np.testing.assert_almost_equal(bd.getP(), 0.7)

    check_dep(bd.setN, 9)
    np.testing.assert_almost_equal(bd.getN(), 9)
    np.testing.assert_almost_equal(bd.getP(), 0.7)

    check_dep(bd.setP, 0.3)
    np.testing.assert_almost_equal(bd.getN(), 9)
    np.testing.assert_almost_equal(bd.getP(), 0.3)


    pd = galsim.PoissonDeviate(rng, mean=0.5)
    np.testing.assert_almost_equal(pd.getMean(), 0.5)

    check_dep(pd.setMean, 0.9)
    np.testing.assert_almost_equal(pd.getMean(), 0.9)


    wd = galsim.WeibullDeviate(rng, a=0.5, b=1.7)
    np.testing.assert_almost_equal(wd.getA(), 0.5)
    np.testing.assert_almost_equal(wd.getB(), 1.7)

    check_dep(wd.setA, 0.9)
    np.testing.assert_almost_equal(wd.getA(), 0.9)
    np.testing.assert_almost_equal(wd.getB(), 1.7)

    check_dep(wd.setB, 2.3)
    np.testing.assert_almost_equal(wd.getA(), 0.9)
    np.testing.assert_almost_equal(wd.getB(), 2.3)


    gd = galsim.GammaDeviate(rng, k=0.5, theta=1.7)
    np.testing.assert_almost_equal(gd.getK(), 0.5)
    np.testing.assert_almost_equal(gd.getTheta(), 1.7)

    check_dep(gd.setK, 0.9)
    np.testing.assert_almost_equal(gd.getK(), 0.9)
    np.testing.assert_almost_equal(gd.getTheta(), 1.7)

    check_dep(gd.setTheta, 2.3)
    np.testing.assert_almost_equal(gd.getK(), 0.9)
    np.testing.assert_almost_equal(gd.getTheta(), 2.3)


    cd = galsim.Chi2Deviate(rng, n=5)
    np.testing.assert_almost_equal(cd.getN(), 5)

    check_dep(cd.setN, 9)
    np.testing.assert_almost_equal(cd.getN(), 9)


@timer
def test_dep_scene():
    """Test the deprecated exclude_bad and exclude_fail args to COSMOSCatalog
    """
    path, filename = os.path.split(__file__)
    datapath = os.path.abspath(os.path.join(path, "../examples/data/"))

    # Initialize one that doesn't exclude failures.  It should be >= the previous one in length.
    cat2 = check_dep(galsim.COSMOSCatalog,
                     file_name='real_galaxy_catalog_23.5_example.fits',
                     dir=datapath, exclude_fail=False, exclude_bad=False)
    # Initialize a COSMOSCatalog with all defaults.
    cat = galsim.COSMOSCatalog(file_name='real_galaxy_catalog_23.5_example.fits',
                               dir=datapath)
    assert cat2.nobjects>=cat.nobjects
    # Equivalent to current exclusion_level='none'
    cat3 = galsim.COSMOSCatalog(file_name='real_galaxy_catalog_23.5_example.fits',
                                dir=datapath, exclusion_level='none')
    assert cat2.nobjects==cat3.nobjects

    # Just exclude_bad=True
    cat2 = check_dep(galsim.COSMOSCatalog,
                     file_name='real_galaxy_catalog_23.5_example.fits',
                     dir=datapath, exclude_fail=False)  # i.e. leave exclude_bad=True
    assert cat2.nobjects>=cat.nobjects
    # Equivalent to current exclusion_level='bad_stamp'
    cat3 = galsim.COSMOSCatalog(file_name='real_galaxy_catalog_23.5_example.fits',
                                dir=datapath, exclusion_level='bad_stamp')
    assert cat2.nobjects==cat3.nobjects

    # Just exclude_fail=True
    cat2 = check_dep(galsim.COSMOSCatalog,
                     file_name='real_galaxy_catalog_23.5_example.fits',
                     dir=datapath, exclude_bad=False)  # i.e. leave exclude_fail=True
    assert cat2.nobjects>=cat.nobjects
    # Equivalent to current exclusion_level='bad_fits'
    cat3 = galsim.COSMOSCatalog(file_name='real_galaxy_catalog_23.5_example.fits',
                                dir=datapath, exclusion_level='bad_fits')
    assert cat2.nobjects==cat3.nobjects

    # Both=True
    cat2 = check_dep(galsim.COSMOSCatalog,
                     file_name='real_galaxy_catalog_23.5_example.fits',
                     dir=datapath, exclude_fail=True, exclude_bad=True)
    assert cat2.nobjects>=cat.nobjects
    # Equivalent to current exclusion_level='marginal'
    cat3 = galsim.COSMOSCatalog(file_name='real_galaxy_catalog_23.5_example.fits',
                                dir=datapath, exclusion_level='marginal')
    assert cat2.nobjects==cat3.nobjects


@timer
def test_dep_sed():
    """Test the deprecated methods in galsim/deprecated/sed.py.
    """
    z = 0.4
    a = galsim.SED(galsim.LookupTable([1,2,3,4,5], [1.1,2.2,3.3,4.4,5.5]),
                   wave_type='nm', flux_type='fphotons', redshift=0.4)
    b = lambda w: w**2

    check_dep(a.copy)

    # function divided by SED
    #c = b/a
    c = check_dep(a.__rdiv__, b)
    x = 3.0
    np.testing.assert_almost_equal(c(x), b(x)/a(x), 10,
                                   err_msg="Found wrong value in SED.__rdiv__")

    # number divided by SED
    #d = x/a
    d = check_dep(a.__rdiv__, x)
    np.testing.assert_almost_equal(d(x), x/a(x), 10,
                                   err_msg="Found wrong value in SED.__rdiv__")


@timer
def test_dep_shapelet():
    """Test the deprecated methods in galsim/deprecated/shapelet.py
    """
    np.testing.assert_almost_equal(check_dep(galsim.LVectorSize,12), galsim.ShapeletSize(12))

    # The next bit is from the old test_shapelet_adjustments() test

    ftypes = [np.float32, np.float64]

    nx = 128
    ny = 128
    scale = 0.2
    im = galsim.ImageF(nx,ny, scale=scale)

    sigma = 1.8
    order = 6
    bvec = [1.3,                                            # n = 0
            0.02, 0.03,                                     # n = 1
            0.23, -0.19, 0.08,                              # n = 2
            0.01, 0.02, 0.04, -0.03,                        # n = 3
            -0.09, 0.07, -0.11, -0.08, 0.11,                # n = 4
            -0.03, -0.02, -0.08, 0.01, -0.06, -0.03,        # n = 5
            0.06, -0.02, 0.00, -0.05, -0.04, 0.01, 0.09 ]   # n = 6

    ref_shapelet = galsim.Shapelet(sigma=sigma, order=order, bvec=bvec)
    ref_im = galsim.ImageF(nx,ny)
    ref_shapelet.drawImage(ref_im, scale=scale, method='no_pixel')

    # test setsigma
    shapelet = galsim.Shapelet(sigma=1., order=order, bvec=bvec)
    check_dep(shapelet.setSigma,sigma)
    shapelet.drawImage(im, method='no_pixel')
    np.testing.assert_array_almost_equal(
        im.array, ref_im.array, 6,
        err_msg="Shapelet set with setSigma disagrees with reference Shapelet")

    # Test setBVec
    shapelet = galsim.Shapelet(sigma=sigma, order=order)
    check_dep(shapelet.setBVec,bvec)
    shapelet.drawImage(im, method='no_pixel')
    np.testing.assert_array_almost_equal(
        im.array, ref_im.array, 6,
        err_msg="Shapelet set with setBVec disagrees with reference Shapelet")

    # Test setOrder
    shapelet = galsim.Shapelet(sigma=sigma, order=2)
    check_dep(shapelet.setOrder,order)
    check_dep(shapelet.setBVec,bvec)
    shapelet.drawImage(im, method='no_pixel')
    np.testing.assert_array_almost_equal(
        im.array, ref_im.array, 6,
        err_msg="Shapelet set with setOrder disagrees with reference Shapelet")

    # Test that changing the order preserves the values to the extent possible.
    shapelet = galsim.Shapelet(sigma=sigma, order=order, bvec=bvec)
    check_dep(shapelet.setOrder,10)
    np.testing.assert_array_equal(
        shapelet.getBVec()[0:28], bvec,
        err_msg="Shapelet setOrder to larger doesn't preserve existing values.")
    np.testing.assert_array_equal(
        shapelet.getBVec()[28:66], np.zeros(66-28),
        err_msg="Shapelet setOrder to larger doesn't fill with zeros.")
    check_dep(shapelet.setOrder,6)
    np.testing.assert_array_equal(
        shapelet.getBVec(), bvec,
        err_msg="Shapelet setOrder back to original from larger doesn't preserve existing values.")
    check_dep(shapelet.setOrder,3)
    np.testing.assert_array_equal(
        shapelet.getBVec()[0:10], bvec[0:10],
        err_msg="Shapelet setOrder to smaller doesn't preserve existing values.")
    check_dep(shapelet.setOrder,6)
    np.testing.assert_array_equal(
        shapelet.getBVec()[0:10], bvec[0:10],
        err_msg="Shapelet setOrder back to original from smaller doesn't preserve existing values.")
    check_dep(shapelet.setOrder,6)
    np.testing.assert_array_equal(
        shapelet.getBVec()[10:28], np.zeros(28-10),
        err_msg="Shapelet setOrder back to original from smaller doesn't fill with zeros.")

    # Test that setting a Shapelet with setNM gives the right profile
    shapelet = galsim.Shapelet(sigma=sigma, order=order)
    i = 0
    for n in range(order+1):
        for m in range(n,-1,-2):
            if m == 0:
                check_dep(shapelet.setNM,n,m,bvec[i])
                i = i+1
            else:
                check_dep(shapelet.setNM,n,m,bvec[i],bvec[i+1])
                i = i+2
    shapelet.drawImage(im, method='no_pixel')
    np.testing.assert_array_almost_equal(
        im.array, ref_im.array, 6,
        err_msg="Shapelet set with setNM disagrees with reference Shapelet")

    # Test that setting a Shapelet with setPQ gives the right profile
    shapelet = galsim.Shapelet(sigma=sigma, order=order)
    i = 0
    for n in range(order+1):
        for m in range(n,-1,-2):
            p = (n+m)//2
            q = (n-m)//2
            if m == 0:
                check_dep(shapelet.setPQ,p,q,bvec[i])
                i = i+1
            else:
                check_dep(shapelet.setPQ,p,q,bvec[i],bvec[i+1])
                i = i+2
    shapelet.drawImage(im, method='no_pixel')
    np.testing.assert_array_almost_equal(
        im.array, ref_im.array, 6,
        err_msg="Shapelet set with setPQ disagrees with reference Shapelet")

    # Check fitImage
    s1 = galsim.Shapelet(sigma=sigma, order=10)
    check_dep(s1.fitImage, image=im)
    s2 = galsim.FitShapelet(sigma=sigma, order=10, image=im)
    np.testing.assert_array_almost_equal(s1.getBVec(), s2.getBVec())


@timer
def test_dep_shear():
    """Test the deprecated methods in galsim/deprecated/shear.py
    """
    s = galsim.Shear(g1=0.17, g2=0.23)

    np.testing.assert_almost_equal(s.g1, 0.17)
    np.testing.assert_almost_equal(s.g2, 0.23)

    check_dep(s.setE1E2,e1=0.4, e2=0.1)
    np.testing.assert_almost_equal(s.e1, 0.4)
    np.testing.assert_almost_equal(s.e2, 0.1)

    check_dep(s.setEBeta,e=0.17, beta=39 * galsim.degrees)
    np.testing.assert_almost_equal(s.e, 0.17)
    np.testing.assert_almost_equal(s.beta / galsim.degrees, 39)

    check_dep(s.setG1G2,g1=-0.23, g2=0.87)
    np.testing.assert_almost_equal(s.g1, -0.23)
    np.testing.assert_almost_equal(s.g2, 0.87)

    check_dep(s.setEta1Eta2,eta1=1.8, eta2=-1.1)
    np.testing.assert_almost_equal(s.e1 * s.eta/s.e, 1.8)
    np.testing.assert_almost_equal(s.e2 * s.eta/s.e, -1.1)

    check_dep(s.setEtaBeta,eta=0.19, beta=52 * galsim.degrees)
    np.testing.assert_almost_equal(s.eta, 0.19)
    np.testing.assert_almost_equal(s.beta / galsim.degrees, 52)



@timer
def test_dep_optics():
    """Test the deprecated module galsim/deprecated/optics.py
    """
    testshape = (512, 512)  # shape of image arrays for all tests
    decimal = 6     # Last decimal place used for checking equality of float arrays, see
                    # np.testing.assert_array_almost_equal(), low since many are ImageF

    decimal_dft = 3  # Last decimal place used for checking near equality of DFT product matrices to
                     # continuous-result derived check values... note this is not as stringent as
                     # decimal, because this is tough, because the DFT representation of a function is
                     # not precisely equivalent to its continuous counterpart.

    # def test_check_all_contiguous():
    """Test all galsim.optics outputs are C-contiguous as required by the galsim.Image class.
    """
    # Check basic outputs from wavefront, psf and mtf (array contents won't matter, so we'll use
    # a pure circular pupil)
    test_obj, _ = check_dep(galsim.optics.wavefront, array_shape=testshape)
    assert test_obj.flags.c_contiguous
    test_obj, _ = check_dep(galsim.optics.psf, array_shape=testshape)
    assert test_obj.flags.c_contiguous
    assert check_dep(galsim.optics.otf, array_shape=testshape).flags.c_contiguous
    assert check_dep(galsim.optics.mtf, array_shape=testshape).flags.c_contiguous
    assert check_dep(galsim.optics.ptf, array_shape=testshape).flags.c_contiguous


    # def test_simple_wavefront():
    """Test the wavefront of a pure circular pupil against the known result.
    """
    kx, ky = galsim.utilities.kxky(testshape)
    dx_test = 3.  # } choose some properly-sampled, yet non-unit / trival, input params
    lod_test = 8. # }
    kmax_test = 2. * np.pi * dx_test / lod_test  # corresponding INTERNAL kmax used in optics code
    kmag = np.sqrt(kx**2 + ky**2) / kmax_test # Set up array of |k| in units of kmax_test
    # Simple pupil wavefront should merely be unit ordinate tophat of radius kmax / 2:
    in_pupil = kmag < .5
    wf_true = np.zeros(kmag.shape)
    wf_true[in_pupil] = 1.
    # Compare
    wf, _ = check_dep(galsim.optics.wavefront,
                      array_shape=testshape, scale=dx_test, lam_over_diam=lod_test)
    np.testing.assert_array_almost_equal(wf, wf_true, decimal=decimal)

    # def test_simple_mtf():
    """Test the MTF of a pure circular pupil against the known result.
    """
    kx, ky = galsim.utilities.kxky(testshape)
    dx_test = 3.  # } choose some properly-sampled, yet non-unit / trival, input params
    lod_test = 8. # }
    kmax_test = 2. * np.pi * dx_test / lod_test  # corresponding INTERNAL kmax used in optics code
    kmag = np.sqrt(kx**2 + ky**2) / kmax_test # Set up array of |k| in units of kmax_test
    in_pupil = kmag < 1.
    # Then use analytic formula for MTF of circ pupil (fun to derive)
    mtf_true = np.zeros(kmag.shape)
    mtf_true[in_pupil] = (np.arccos(kmag[in_pupil]) - kmag[in_pupil] *
                          np.sqrt(1. - kmag[in_pupil]**2)) * 2. / np.pi
    # Compare
    mtf = check_dep(galsim.optics.mtf, array_shape=testshape, scale=dx_test, lam_over_diam=lod_test)
    np.testing.assert_array_almost_equal(mtf, mtf_true, decimal=decimal_dft)

    # def test_simple_ptf():
    """Test the PTF of a pure circular pupil against the known result (zero).
    """
    ptf_true = np.zeros(testshape)
    # Compare
    ptf = check_dep(galsim.optics.ptf, array_shape=testshape)
    # Test via median absolute deviation, since occasionally things around the edge of the OTF get
    # hairy when dividing a small number by another small number
    nmad_ptfdiff = np.median(np.abs(ptf - np.median(ptf_true)))
    assert nmad_ptfdiff <= 10.**(-decimal)

    # def test_consistency_psf_mtf():
    """Test that the MTF of a pure circular pupil is |FT{PSF}|.
    """
    kx, ky = galsim.utilities.kxky(testshape)
    dx_test = 3.  # } choose some properly-sampled, yet non-unit / trival, input params
    lod_test = 8. # }
    kmax_test = 2. * np.pi * dx_test / lod_test  # corresponding INTERNAL kmax used in optics code
    psf, _ = check_dep(galsim.optics.psf,
                       array_shape=testshape, scale=dx_test, lam_over_diam=lod_test)
    psf *= dx_test**2 # put the PSF into flux units rather than SB for comparison
    mtf_test = np.abs(np.fft.fft2(psf))
    # Compare
    mtf = check_dep(galsim.optics.mtf, array_shape=testshape, scale=dx_test, lam_over_diam=lod_test)
    np.testing.assert_array_almost_equal(mtf, mtf_test, decimal=decimal_dft)

    # def test_wavefront_image_view():
    """Test that the ImageF.array view of the wavefront is consistent with the wavefront array.
    """
    array, _ = check_dep(galsim.optics.wavefront, array_shape=testshape)
    (real, imag), _ = check_dep(galsim.optics.wavefront_image, array_shape=testshape)
    np.testing.assert_array_almost_equal(array.real.astype(np.float32), real.array, decimal)
    np.testing.assert_array_almost_equal(array.imag.astype(np.float32), imag.array, decimal)

    # def test_psf_image_view():
    """Test that the ImageF.array view of the PSF is consistent with the PSF array.
    """
    array, _ = check_dep(galsim.optics.psf, array_shape=testshape)
    image = check_dep(galsim.optics.psf_image, array_shape=testshape)
    np.testing.assert_array_almost_equal(array.astype(np.float32), image.array, decimal)

    # def test_otf_image_view():
    """Test that the ImageF.array view of the OTF is consistent with the OTF array.
    """
    array = check_dep(galsim.optics.otf, array_shape=testshape)
    (real, imag) = check_dep(galsim.optics.otf_image, array_shape=testshape)
    np.testing.assert_array_almost_equal(array.real.astype(np.float32), real.array, decimal)
    np.testing.assert_array_almost_equal(array.imag.astype(np.float32), imag.array, decimal)

    # def test_mtf_image_view():
    """Test that the ImageF.array view of the MTF is consistent with the MTF array.
    """
    array = check_dep(galsim.optics.mtf, array_shape=testshape)
    image = check_dep(galsim.optics.mtf_image, array_shape=testshape)
    np.testing.assert_array_almost_equal(array.astype(np.float32), image.array)

    # def test_ptf_image_view():
    """Test that the ImageF.array view of the OTF is consistent with the OTF array.
    """
    array = check_dep(galsim.optics.ptf, array_shape=testshape)
    image = check_dep(galsim.optics.ptf_image, array_shape=testshape)
    np.testing.assert_array_almost_equal(array.astype(np.float32), image.array)

@timer
def test_dep_phase_psf():
    """Test deprecated input in PhaseScreenPSF"""
    import time
    NPSFs = 10
    exptime = 0.3
    rng = galsim.BaseDeviate(1234)
    atm = galsim.Atmosphere(screen_size=10.0, altitude=10.0, alpha=0.997, time_step=0.01, rng=rng)
    theta = [(i*galsim.arcsec, i*galsim.arcsec) for i in range(NPSFs)]

    kwargs = dict(lam=1000.0, exptime=exptime, diam=1.0)

    t1 = time.time()
    psfs = check_dep(atm.makePSF, theta=theta, **kwargs)
    imgs = [psf.drawImage() for psf in psfs]
    print('time for {0} PSFs in batch: {1:.2f} s'.format(NPSFs, time.time() - t1))

    t2 = time.time()
    more_imgs = []
    for th in theta:
        psf = atm.makePSF(theta=th, **kwargs)
        more_imgs.append(psf.drawImage())

    print('time for {0} PSFs in serial: {1:.2f} s'.format(NPSFs, time.time() - t2))

    for img1, img2 in zip(imgs, more_imgs):
        np.testing.assert_array_equal(
            img1, img2,
            "Individually generated AtmosphericPSF differs from AtmosphericPSF generated in batch")

@timer
def test_dep_wmult():
    """Test drawImage with wmult parameter.

    (A subset of the test_drawImage function in test_draw.py.)
    """
    test_flux = 1.8
    obj = galsim.Exponential(flux=test_flux, scale_radius=2)
    im1 = obj.drawImage(method='no_pixel')
    obj2 = galsim.Convolve([ obj, galsim.Pixel(im1.scale) ])
    nyq_scale = obj2.nyquistScale()
    scale = 0.51   # Just something different from 1 or dx_nyq
    im3 = galsim.ImageD(56,56)
    im5 = galsim.ImageD()
    obj.drawImage(im5)

    # Test if we provide wmult.  It should:
    #   - create a new image that is wmult times larger in each direction.
    #   - return the new image
    #   - set the scale to obj2.nyquistScale()
    im6 = check_dep(obj.drawImage, wmult=3.)
    np.testing.assert_almost_equal(im6.scale, nyq_scale, 9,
                                   "obj.drawImage(wmult) produced image with wrong scale")
    # Can assert accuracy to 4 decimal places now, since we're capturing much more
    # of the flux on the image.
    np.testing.assert_almost_equal(im6.array.astype(float).sum(), test_flux, 4,
                                   "obj.drawImage(wmult) produced image with wrong flux")
    np.testing.assert_almost_equal(CalculateScale(im6), 2, 2,
                                   "Measured wrong scale after obj.drawImage(wmult)")
    assert im6.bounds == galsim.BoundsI(1,166,1,166),(
            "obj.drawImage(wmult) produced image with wrong bounds")

    # Test if we provide an image argument and wmult.  It should:
    #   - write to the existing image
    #   - also return that image
    #   - set the scale to obj2.nyquistScale()
    #   - zero out any existing data
    #   - the calculation of the convolution should be slightly more accurate than for im3
    im3.setZero()
    im5.setZero()
    check_dep(obj.drawImage, im3, wmult=4.)
    obj.drawImage(im5)
    np.testing.assert_almost_equal(im3.scale, nyq_scale, 9,
                                   "obj.drawImage(im3) produced image with wrong scale")
    np.testing.assert_almost_equal(im3.array.sum(), test_flux, 2,
                                   "obj.drawImage(im3,wmult) produced image with wrong flux")
    np.testing.assert_almost_equal(CalculateScale(im3), 2, 1,
                                   "Measured wrong scale after obj.drawImage(im3,wmult)")
    assert ((im3.array-im5.array)**2).sum() > 0, (
            "obj.drawImage(im3,wmult) produced the same image as without wmult")

    # Test with dx and wmult.  It should:
    #   - create a new image using that dx for the scale
    #   - set the size a factor of wmult times larger in each direction.
    #   - return the new image
    im8 = check_dep(obj.drawImage, scale=scale, wmult=4.)
    np.testing.assert_almost_equal(im8.scale, scale, 9,
                                   "obj.drawImage(dx,wmult) produced image with wrong scale")
    np.testing.assert_almost_equal(im8.array.astype(float).sum(), test_flux, 4,
                                   "obj.drawImage(dx,wmult) produced image with wrong flux")
    np.testing.assert_almost_equal(CalculateScale(im8), 2, 2,
                                   "Measured wrong scale after obj.drawImage(dx,wmult)")
    assert im8.bounds == galsim.BoundsI(1,270,1,270),(
            "obj.drawImage(dx,wmult) produced image with wrong bounds")

@timer
def test_dep_drawKImage():
    """Test the various optional parameters to the drawKImage function.
       In particular test the parameters image, and scale in various combinations.
    """
    # We use a Moffat profile with beta = 1.5, since its real-space profile is
    #    flux / (2 pi rD^2) * (1 + (r/rD)^2)^3/2
    # and the 2-d Fourier transform of that is
    #    flux * exp(-rD k)
    # So this should draw in Fourier space the same image as the Exponential drawn in
    # test_drawImage().
    test_flux = 1.8
    obj = galsim.Moffat(flux=test_flux, beta=1.5, scale_radius=0.5)

    # First test drawKImage() with no kwargs.  It should:
    #   - create new images
    #   - return the new images
    #   - set the scale to 2pi/(N*obj.nyquistScale())
    re1, im1 = check_dep_tuple2(obj.drawKImage())
    re1.setOrigin(1,1)  # Go back to old convention on bounds
    im1.setOrigin(1,1)
    N = 1163
    assert re1.bounds == galsim.BoundsI(1,N,1,N),(
            "obj.drawKImage() produced image with wrong bounds")
    assert im1.bounds == galsim.BoundsI(1,N,1,N),(
            "obj.drawKImage() produced image with wrong bounds")
    nyq_scale = obj.nyquistScale()
    stepk = obj.stepK()
    np.testing.assert_almost_equal(re1.scale, stepk, 9,
                                   "obj.drawKImage() produced real image with wrong scale")
    np.testing.assert_almost_equal(im1.scale, stepk, 9,
                                   "obj.drawKImage() produced imag image with wrong scale")
    np.testing.assert_almost_equal(CalculateScale(re1), 2, 1,
                                   "Measured wrong scale after obj.drawKImage()")

    # The flux in Fourier space is just the value at k=0
    np.testing.assert_almost_equal(re1(re1.bounds.center()), test_flux, 2,
                                   "obj.drawKImage() produced real image with wrong flux")
    # Imaginary component should all be 0.
    np.testing.assert_almost_equal(im1.array.sum(), 0., 3,
                                   "obj.drawKImage() produced non-zero imaginary image")

    # Test if we provide an image argument.  It should:
    #   - write to the existing image
    #   - also return that image
    #   - set the scale to obj.stepK()
    #   - zero out any existing data
    re3 = galsim.ImageD(1149,1149)
    im3 = galsim.ImageD(1149,1149)
    re4, im4 = check_dep_tuple2(check_dep(obj.drawKImage, re3, im3))
    np.testing.assert_almost_equal(re3.scale, stepk, 9,
                                   "obj.drawKImage(re3,im3) produced real image with wrong scale")
    np.testing.assert_almost_equal(im3.scale, stepk, 9,
                                   "obj.drawKImage(re3,im3) produced imag image with wrong scale")
    np.testing.assert_almost_equal(re3(re3.bounds.center()), test_flux, 2,
                                   "obj.drawKImage(re3,im3) produced real image with wrong flux")
    np.testing.assert_almost_equal(im3.array.sum(), 0., 3,
                                   "obj.drawKImage(re3,im3) produced non-zero imaginary image")
    np.testing.assert_almost_equal(CalculateScale(re3), 2, 1,
                                   "Measured wrong scale after obj.drawKImage(re3,im3)")
    np.testing.assert_array_equal(re3.array, re4.array,
                                  "re4, im4 = obj.drawKImage(re3,im3) produced re4 != re3")
    np.testing.assert_array_equal(im3.array, im4.array,
                                  "re4, im4 = obj.drawKImage(re3,im3) produced im4 != im3")
    re3.fill(9.8)
    im3.fill(9.8)
    np.testing.assert_array_equal(re3.array, re4.array,
                                  "re4, im4 = obj.drawKImage(re3,im3) produced re4 is not re3")
    np.testing.assert_array_equal(im3.array, im4.array,
                                  "re4, im4 = obj.drawKImage(re3,im3) produced im4 is not im3")

    # Test if we provide an image with undefined bounds.  It should:
    #   - resize the provided image
    #   - also return that image
    #   - set the scale to obj.stepK()
    re5 = galsim.ImageD()
    im5 = galsim.ImageD()
    check_dep(obj.drawKImage, re5, im5)
    np.testing.assert_almost_equal(re5.scale, stepk, 9,
                                   "obj.drawKImage(re5,im5) produced real image with wrong scale")
    np.testing.assert_almost_equal(im5.scale, stepk, 9,
                                   "obj.drawKImage(re5,im5) produced imag image with wrong scale")
    np.testing.assert_almost_equal(re5(re5.bounds.center()), test_flux, 2,
                                   "obj.drawKImage(re5,im5) produced real image with wrong flux")
    np.testing.assert_almost_equal(im5.array.sum(), 0., 3,
                                   "obj.drawKImage(re5,im5) produced non-zero imaginary image")
    np.testing.assert_almost_equal(CalculateScale(re5), 2, 1,
                                   "Measured wrong scale after obj.drawKImage(re5,im5)")
    im5.setOrigin(1,1)
    assert im5.bounds == galsim.BoundsI(1,N,1,N),(
            "obj.drawKImage(re5,im5) produced image with wrong bounds")

    # Test if we provide a scale to use.  It should:
    #   - create a new image using that scale for the scale
    #   - return the new image
    #   - set the size large enough to contain 99.5% of the flux
    scale = 0.51   # Just something different from 1 or dx_nyq
    re7, im7 = check_dep_tuple2(obj.drawKImage(scale=scale))
    np.testing.assert_almost_equal(re7.scale, scale, 9,
                                   "obj.drawKImage(dx) produced real image with wrong scale")
    np.testing.assert_almost_equal(im7.scale, scale, 9,
                                   "obj.drawKImage(dx) produced imag image with wrong scale")
    np.testing.assert_almost_equal(re7(re7.bounds.center()), test_flux, 2,
                                   "obj.drawKImage(dx) produced real image with wrong flux")
    np.testing.assert_almost_equal(im7.array.astype(float).sum(), 0., 2,
                                   "obj.drawKImage(dx) produced non-zero imaginary image")
    np.testing.assert_almost_equal(CalculateScale(re7), 2, 1,
                                   "Measured wrong scale after obj.drawKImage(dx)")
    # This image is smaller because not using nyquist scale for stepk
    im7.setOrigin(1,1)
    assert im7.bounds == galsim.BoundsI(1,73,1,73),(
            "obj.drawKImage(dx) produced image with wrong bounds")

    # Test if we provide an image with a defined scale.  It should:
    #   - write to the existing image
    #   - use the image's scale
    nx = 401
    re9 = galsim.ImageD(nx,nx, scale=scale)
    im9 = galsim.ImageD(nx,nx, scale=scale)
    check_dep(obj.drawKImage, re9, im9)
    np.testing.assert_almost_equal(re9.scale, scale, 9,
                                   "obj.drawKImage(re9,im9) produced real image with wrong scale")
    np.testing.assert_almost_equal(im9.scale, scale, 9,
                                   "obj.drawKImage(re9,im9) produced imag image with wrong scale")
    np.testing.assert_almost_equal(re9(re9.bounds.center()), test_flux, 4,
                                   "obj.drawKImage(re9,im9) produced real image with wrong flux")
    np.testing.assert_almost_equal(im9.array.sum(), 0., 5,
                                   "obj.drawKImage(re9,im9) produced non-zero imaginary image")
    np.testing.assert_almost_equal(CalculateScale(re9), 2, 1,
                                   "Measured wrong scale after obj.drawKImage(re9,im9)")

    # Test if we provide an image with a defined scale <= 0.  It should:
    #   - write to the existing image
    #   - set the scale to obj.stepK()
    re3.scale = -scale
    im3.scale = -scale
    re3.setZero()
    check_dep(obj.drawKImage, re3, im3)
    np.testing.assert_almost_equal(re3.scale, stepk, 9,
                                   "obj.drawKImage(re3,im3) produced real image with wrong scale")
    np.testing.assert_almost_equal(im3.scale, stepk, 9,
                                   "obj.drawKImage(re3,im3) produced imag image with wrong scale")
    np.testing.assert_almost_equal(re3(re3.bounds.center()), test_flux, 4,
                                   "obj.drawKImage(re3,im3) produced real image with wrong flux")
    np.testing.assert_almost_equal(im3.array.sum(), 0., 5,
                                   "obj.drawKImage(re3,im3) produced non-zero imaginary image")
    np.testing.assert_almost_equal(CalculateScale(re3), 2, 1,
                                   "Measured wrong scale after obj.drawKImage(re3,im3)")
    re3.scale = 0
    im3.scale = 0
    re3.setZero()
    check_dep(obj.drawKImage, re3, im3)
    np.testing.assert_almost_equal(re3.scale, stepk, 9,
                                   "obj.drawKImage(re3,im3) produced real image with wrong scale")
    np.testing.assert_almost_equal(im3.scale, stepk, 9,
                                   "obj.drawKImage(re3,im3) produced imag image with wrong scale")
    np.testing.assert_almost_equal(re3(re3.bounds.center()), test_flux, 4,
                                   "obj.drawKImage(re3,im3) produced real image with wrong flux")
    np.testing.assert_almost_equal(im3.array.sum(), 0., 5,
                                   "obj.drawKImage(re3,im3) produced non-zero imaginary image")
    np.testing.assert_almost_equal(CalculateScale(re3), 2, 1,
                                   "Measured wrong scale after obj.drawKImage(re3,im3)")

    # Test if we provide an image and dx.  It should:
    #   - write to the existing image
    #   - use the provided dx
    #   - write the new dx value to the image's scale
    re9.scale = scale + 0.3  # Just something other than scale
    im9.scale = scale + 0.3
    re9.setZero()
    check_dep(obj.drawKImage, re9, im9, scale=scale)
    np.testing.assert_almost_equal(
            re9.scale, scale, 9,
            "obj.drawKImage(re9,im9,scale) produced real image with wrong scale")
    np.testing.assert_almost_equal(
            im9.scale, scale, 9,
            "obj.drawKImage(re9,im9,scale) produced imag image with wrong scale")
    np.testing.assert_almost_equal(
            re9(re9.bounds.center()), test_flux, 4,
            "obj.drawKImage(re9,im9,scale) produced real image with wrong flux")
    np.testing.assert_almost_equal(
            im9.array.sum(), 0., 5,
            "obj.drawKImage(re9,im9,scale) produced non-zero imaginary image")
    np.testing.assert_almost_equal(CalculateScale(re9), 2, 1,
                                   "Measured wrong scale after obj.drawKImage(re9,im9,scale)")

    # Test if we provide an image and scale <= 0.  It should:
    #   - write to the existing image
    #   - set the scale to obj.stepK()
    re3.scale = scale + 0.3
    im3.scale = scale + 0.3
    re3.setZero()
    check_dep(obj.drawKImage, re3, im3, scale=-scale)
    np.testing.assert_almost_equal(
            re3.scale, stepk, 9,
            "obj.drawKImage(re3,im3,scale<0) produced real image with wrong scale")
    np.testing.assert_almost_equal(
            im3.scale, stepk, 9,
            "obj.drawKImage(re3,im3,scale<0) produced imag image with wrong scale")
    np.testing.assert_almost_equal(
            re3(re3.bounds.center()), test_flux, 4,
            "obj.drawKImage(re3,im3,scale<0) produced real image with wrong flux")
    np.testing.assert_almost_equal(
            im3.array.sum(), 0., 5,
            "obj.drawKImage(re3,im3,scale<0) produced non-zero imaginary image")
    np.testing.assert_almost_equal(CalculateScale(re3), 2, 1,
                                   "Measured wrong scale after obj.drawKImage(re3,im3,scale<0)")
    re3.scale = scale + 0.3
    im3.scale = scale + 0.3
    re3.setZero()
    check_dep(obj.drawKImage, re3, im3, scale=0)
    np.testing.assert_almost_equal(
        re3.scale, stepk, 9,
        "obj.drawKImage(re3,im3,scale=0) produced real image with wrong scale")
    np.testing.assert_almost_equal(
        im3.scale, stepk, 9,
        "obj.drawKImage(re3,im3,scale=0) produced imag image with wrong scale")
    np.testing.assert_almost_equal(
        re3(re3.bounds.center()), test_flux, 4,
        "obj.drawKImage(re3,im3,scale=0) produced real image with wrong flux")
    np.testing.assert_almost_equal(
        im3.array.sum(), 0., 5,
        "obj.drawKImage(re3,im3,scale=0) produced non-zero imaginary image")
    np.testing.assert_almost_equal(CalculateScale(re3), 2, 1,
                                   "Measured wrong scale after obj.drawKImage(re3,im3,scale=0)")

    # Test if we provide nx, ny, and scale.  It should:
    #   - create a new image with the right size
    #   - set the scale
    nx = 200  # Some randome non-square size
    ny = 100
    re4, im4 = check_dep_tuple2(obj.drawKImage(nx=nx, ny=ny, scale=scale))
    np.testing.assert_almost_equal(
        re4.scale, scale, 9,
        "obj.drawKImage(nx,ny,scale) produced real image with wrong scale")
    np.testing.assert_almost_equal(
        im4.scale, scale, 9,
        "obj.drawKImage(nx,ny,scale) produced imag image with wrong scale")
    np.testing.assert_almost_equal(
        re4.array.shape, (ny, nx), 9,
        "obj.drawKImage(nx,ny,scale) produced real image with wrong shape")
    np.testing.assert_almost_equal(
        im4.array.shape, (ny, nx), 9,
        "obj.drawKImage(nx,ny,scale) produced imag image with wrong shape")

    # Test if we provide bounds and scale.  It should:
    #   - create a new image with the right size
    #   - set the scale
    bounds = galsim.BoundsI(1,nx,1,ny)
    re4, im4 = check_dep_tuple2(obj.drawKImage(bounds=bounds, scale=stepk))
    np.testing.assert_almost_equal(
        re4.scale, stepk, 9,
        "obj.drawKImage(bounds,scale) produced real image with wrong scale")
    np.testing.assert_almost_equal(
        im4.scale, stepk, 9,
        "obj.drawKImage(bounds,scale) produced imag image with wrong scale")
    np.testing.assert_almost_equal(
        re4.array.shape, (ny, nx), 9,
        "obj.drawKImage(bounds,scale) produced real image with wrong shape")
    np.testing.assert_almost_equal(
        im4.array.shape, (ny, nx), 9,
        "obj.drawKImage(bounds,scale) produced imag image with wrong shape")

@timer
def test_dep_drawKImage_Gaussian():
    """Test the drawKImage function using known symmetries of the Gaussian Hankel transform.

    See http://en.wikipedia.org/wiki/Hankel_transform.
    """
    test_flux = 2.3     # Choose a non-unity flux
    test_sigma = 17.    # ...likewise for sigma
    test_imsize = 45    # Dimensions of comparison image, doesn't need to be large

    # Define a Gaussian GSObject
    gal = galsim.Gaussian(sigma=test_sigma, flux=test_flux)
    # Then define a related object which is in fact the opposite number in the Hankel transform pair
    # For the Gaussian this is straightforward in our definition of the Fourier transform notation,
    # and has sigma -> 1/sigma and flux -> flux * 2 pi / sigma**2
    gal_hankel = galsim.Gaussian(sigma=1./test_sigma, flux=test_flux*2.*np.pi/test_sigma**2)

    # Do a basic flux test: the total flux of the gal should equal gal_Hankel(k=(0, 0))
    np.testing.assert_almost_equal(
        gal.getFlux(), gal_hankel.xValue(galsim.PositionD(0., 0.)), decimal=12,
        err_msg="Test object flux does not equal k=(0, 0) mode of its Hankel transform conjugate.")

    image_test = galsim.ImageD(test_imsize, test_imsize)
    rekimage_test = galsim.ImageD(test_imsize, test_imsize)
    imkimage_test = galsim.ImageD(test_imsize, test_imsize)

    # Then compare these two objects at a couple of different scale (reasonably matched for size)
    for scale_test in (0.03 / test_sigma, 0.4 / test_sigma):
        check_dep(gal.drawKImage,re=rekimage_test, im=imkimage_test, scale=scale_test)
        gal_hankel.drawImage(image_test, scale=scale_test, use_true_center=False, method='sb')
        np.testing.assert_array_almost_equal(
            rekimage_test.array, image_test.array, decimal=12,
            err_msg="Test object drawKImage() and drawImage() from Hankel conjugate do not match "
            "for grid spacing scale = "+str(scale_test))
        np.testing.assert_array_almost_equal(
            imkimage_test.array, np.zeros_like(imkimage_test.array), decimal=12,
            err_msg="Non-zero imaginary part for drawKImage from test object that is purely "
            "centred on the origin.")

@timer
def test_dep_kroundtrip():
    KXVALS = np.array((1.30, 0.71, -4.30)) * np.pi / 2.
    KYVALS = np.array((0.80, -0.02, -0.31,)) * np.pi / 2.
    g1 = galsim.Gaussian(sigma = 3.1, flux=2.4).shear(g1=0.2,g2=0.1)
    g2 = galsim.Gaussian(sigma = 1.9, flux=3.1).shear(g1=-0.4,g2=0.3).shift(-0.3,0.5)
    g3 = galsim.Gaussian(sigma = 4.1, flux=1.6).shear(g1=0.1,g2=-0.1).shift(0.7,-0.2)
    final = g1 + g2 + g3
    a = final
    real_a, imag_a = check_dep_tuple2(a.drawKImage())
    b = check_dep(galsim.InterpolatedKImage, real_a, imag_a)

    # Check picklability
    do_pickle(b)
    do_pickle(b, lambda x: x.drawImage())
    do_pickle(b.SBProfile)
    do_pickle(b.SBProfile, lambda x: repr(x))

    for kx, ky in zip(KXVALS, KYVALS):
        np.testing.assert_almost_equal(a.kValue(kx, ky), b.kValue(kx, ky), 3,
            err_msg=("InterpolatedKImage evaluated incorrectly at ({0:},{1:})"
                     .format(kx, ky)))

    np.testing.assert_almost_equal(a.getFlux(), b.getFlux(), 6) #Fails at 7th decimal

    real_b, imag_b = check_dep_tuple2(check_dep(b.drawKImage, real_a.copy(), imag_a.copy()))
    # Fails at 4th decimal
    np.testing.assert_array_almost_equal(real_a.array, real_b.array, 3,
                                         "InterpolatedKImage kimage drawn incorrectly.")
    # Fails at 4th decimal
    np.testing.assert_array_almost_equal(imag_a.array, imag_b.array, 3,
                                         "InterpolatedKImage kimage drawn incorrectly.")

    img_a = a.drawImage()
    img_b = b.drawImage(img_a.copy())
    # This is the one that matters though; fails at 6th decimal
    np.testing.assert_array_almost_equal(img_a.array, img_b.array, 5,
                                         "InterpolatedKImage image drawn incorrectly.")

    # Try some (slightly larger maxk) non-even kimages:
    for dx, dy in zip((2,3,3), (3,2,3)):
        shape = real_a.array.shape
        real_a, imag_a = check_dep_tuple2(a.drawKImage(nx=shape[1]+dx, ny=shape[0]+dy,
                                          scale=real_a.scale))
        b = check_dep(galsim.InterpolatedKImage, real_a, imag_a)

        np.testing.assert_almost_equal(a.getFlux(), b.getFlux(), 6) #Fails at 7th decimal
        img_b = b.drawImage(img_a.copy())
        # One of these fails at 6th decimal
        np.testing.assert_array_almost_equal(img_a.array, img_b.array, 5)

    # Try some additional transformations:
    a = a.shear(g1=0.2, g2=-0.2).shift(1.1, -0.2).dilate(0.7)
    b = b.shear(g1=0.2, g2=-0.2).shift(1.1, -0.2).dilate(0.7)
    img_a = a.drawImage()
    img_b = b.drawImage(img_a.copy())
    # Fails at 6th decimal
    np.testing.assert_array_almost_equal(img_a.array, img_b.array, 5,
                                         "Transformed InterpolatedKImage image drawn incorrectly.")

    # Does the stepk parameter do anything?
    a = final
    b = check_dep(galsim.InterpolatedKImage, *check_dep_tuple2(a.drawKImage()))
    c = check_dep(galsim.InterpolatedKImage, *check_dep_tuple2(a.drawKImage()), stepk=2*b.stepK())
    np.testing.assert_almost_equal(2*b.stepK(), c.stepK())
    np.testing.assert_almost_equal(b.maxK(), c.maxK())

    # Test centroid
    for dx, dy in zip(KXVALS, KYVALS):
        a = final.shift(dx, dy)
        b = check_dep(galsim.InterpolatedKImage, *check_dep_tuple2(a.drawKImage()))
        np.testing.assert_almost_equal(a.centroid().x, b.centroid().x, 4) #Fails at 5th decimal
        np.testing.assert_almost_equal(a.centroid().y, b.centroid().y, 4)

    # Test convolution with another object.
    a = final
    b = check_dep(galsim.InterpolatedKImage, *check_dep_tuple2(a.drawKImage()))
    c = galsim.Kolmogorov(fwhm=0.8).shear(e1=0.01, e2=0.02).shift(0.01, 0.02)
    a_conv_c = galsim.Convolve(a, c)
    b_conv_c = galsim.Convolve(b, c)
    a_conv_c_img = a_conv_c.drawImage()
    b_conv_c_img = b_conv_c.drawImage(image=a_conv_c_img.copy())
    # Fails at 6th decimal.
    np.testing.assert_array_almost_equal(a_conv_c_img.array, b_conv_c_img.array, 5,
                                         "Convolution of InterpolatedKImage drawn incorrectly.")

@timer
def test_dep_simreal():
    """Test accuracy of various calculations with fake Gaussian RealGalaxy vs. ideal expectations
    and stored results"""
    image_dir = './real_comparison_images'
    catalog_file = 'test_catalog.fits'

    ind_fake = 1 # index of mock galaxy (Gaussian) in catalog
    fake_gal_fwhm = 0.7 # arcsec
    fake_gal_shear1 = 0.29 # shear representing intrinsic shape component 1
    fake_gal_shear2 = -0.21 # shear representing intrinsic shape component 2
    fake_gal_flux = 1000.0
    fake_gal_orig_PSF_fwhm = 0.1 # arcsec
    fake_gal_orig_PSF_shear1 = 0.0
    fake_gal_orig_PSF_shear2 = -0.07

    targ_pixel_scale = [0.18, 0.25] # arcsec
    targ_PSF_fwhm = [0.7, 1.0] # arcsec
    targ_PSF_shear1 = [-0.03, 0.0]
    targ_PSF_shear2 = [0.05, -0.08]
    targ_applied_shear1 = 0.06
    targ_applied_shear2 = -0.04

    sigma_to_fwhm = 2.0*np.sqrt(2.0*np.log(2.0)) # multiply sigma by this to get FWHM for Gaussian
    fwhm_to_sigma = 1.0/sigma_to_fwhm

    ind_real = 0 # index of real galaxy in catalog
    shera_file = 'real_comparison_images/shera_result.fits'
    shera_target_PSF_file = 'real_comparison_images/shera_target_PSF.fits'
    shera_target_pixel_scale = 0.24
    shera_target_flux = 1000.0

    # read in faked Gaussian RealGalaxy from file
    rgc = galsim.RealGalaxyCatalog(catalog_file, dir=image_dir)
    rg = galsim.RealGalaxy(rgc, index=ind_fake)

    ## for the generation of the ideal right answer, we need to add the intrinsic shape of the
    ## galaxy and the lensing shear using the rule for addition of distortions which is ugly, but oh
    ## well:
    (d1, d2) = galsim.utilities.g1g2_to_e1e2(fake_gal_shear1, fake_gal_shear2)
    (d1app, d2app) = galsim.utilities.g1g2_to_e1e2(targ_applied_shear1, targ_applied_shear2)
    denom = 1.0 + d1*d1app + d2*d2app
    dapp_sq = d1app**2 + d2app**2
    d1tot = (d1 + d1app + d2app/dapp_sq*(1.0 - np.sqrt(1.0-dapp_sq))*(d2*d1app - d1*d2app))/denom
    d2tot = (d2 + d2app + d1app/dapp_sq*(1.0 - np.sqrt(1.0-dapp_sq))*(d1*d2app - d2*d1app))/denom

    # convolve with a range of Gaussians, with and without shear (note, for this test all the
    # original and target ePSFs are Gaussian - there's no separate pixel response so that everything
    # can be calculated analytically)
    for tps in targ_pixel_scale:
        for tpf in targ_PSF_fwhm:
            for tps1 in targ_PSF_shear1:
                for tps2 in targ_PSF_shear2:
                    print('tps,tpf,tps1,tps2 = ',tps,tpf,tps1,tps2)
                    # make target PSF
                    targ_PSF = galsim.Gaussian(fwhm = tpf).shear(g1=tps1, g2=tps2)
                    # simulate image
                    sim_image = check_dep(galsim.simReal,
                                          rg, targ_PSF, tps,
                                          g1 = targ_applied_shear1,
                                          g2 = targ_applied_shear2,
                                          rand_rotate = False,
                                          target_flux = fake_gal_flux)
                    # galaxy sigma, in units of pixels on the final image
                    sigma_ideal = (fake_gal_fwhm/tps)*fwhm_to_sigma
                    # compute analytically the expected galaxy moments:
                    mxx_gal, myy_gal, mxy_gal = ellip_to_moments(d1tot, d2tot, sigma_ideal)
                    # compute analytically the expected PSF moments:
                    targ_PSF_e1, targ_PSF_e2 = galsim.utilities.g1g2_to_e1e2(tps1, tps2)
                    targ_PSF_sigma = (tpf/tps)*fwhm_to_sigma
                    mxx_PSF, myy_PSF, mxy_PSF = ellip_to_moments(
                            targ_PSF_e1, targ_PSF_e2, targ_PSF_sigma)
                    # get expected e1, e2, sigma for the PSF-convolved image
                    tot_e1, tot_e2, tot_sigma = moments_to_ellip(
                            mxx_gal+mxx_PSF, myy_gal+myy_PSF, mxy_gal+mxy_PSF)

                    # compare with images that are expected
                    expected_gaussian = galsim.Gaussian(
                            flux = fake_gal_flux, sigma = tps*tot_sigma)
                    expected_gaussian = expected_gaussian.shear(e1 = tot_e1, e2 = tot_e2)
                    expected_image = galsim.ImageD(
                            sim_image.array.shape[0], sim_image.array.shape[1])
                    expected_gaussian.drawImage(expected_image, scale=tps, method='no_pixel')
                    printval(expected_image,sim_image)
                    np.testing.assert_array_almost_equal(
                        sim_image.array, expected_image.array, decimal = 3,
                        err_msg = "Error in comparison of ideal Gaussian RealGalaxy calculations")

    full_catalog_file = os.path.join(image_dir,catalog_file)
    rgc = galsim.RealGalaxyCatalog(full_catalog_file)
    rg = galsim.RealGalaxy(rgc, index=ind_real)

    # read in expected result for some shear
    shera_image = galsim.fits.read(shera_file)
    shera_target_PSF_image = galsim.fits.read(shera_target_PSF_file)

    # simulate the same galaxy with galsim.simReal
    sim_image = check_dep(galsim.simReal, rg, shera_target_PSF_image,
                          shera_target_pixel_scale, g1=targ_applied_shear1,
                          g2=targ_applied_shear2, rand_rotate=False,
                          target_flux=shera_target_flux)

    # there are centroid issues when comparing Shera vs. SBProfile outputs, so compare 2nd moments
    # instead of images
    sbp_res = sim_image.FindAdaptiveMom()
    shera_res = shera_image.FindAdaptiveMom()

    np.testing.assert_almost_equal(sbp_res.observed_shape.e1,
                                   shera_res.observed_shape.e1, 2,
                                   err_msg = "Error in comparison with SHERA result: e1")
    np.testing.assert_almost_equal(sbp_res.observed_shape.e2,
                                   shera_res.observed_shape.e2, 2,
                                   err_msg = "Error in comparison with SHERA result: e2")
    np.testing.assert_almost_equal(sbp_res.moments_sigma, shera_res.moments_sigma, 2,
                                   err_msg = "Error in comparison with SHERA result: sigma")

if __name__ == "__main__":
    test_dep_bandpass()
    test_dep_base()
    test_dep_bounds()
    test_dep_chromatic()
    test_dep_correlatednoise()
    test_dep_gsobject_ring()
    test_dep_image()
    test_dep_noise()
    test_dep_random()
    test_dep_scene()
    test_dep_sed()
    test_dep_shapelet()
    test_dep_shear()
    test_dep_optics()
    test_dep_phase_psf()
    test_dep_wmult()
    test_dep_drawKImage()
    test_dep_drawKImage_Gaussian()
    test_dep_kroundtrip()
    test_dep_simreal()
