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
import sys
import numpy as np

from galsim_test_helpers import *

try:
    import galsim
except ImportError:
    path, filename = os.path.split(__file__)
    sys.path.append(os.path.abspath(os.path.join(path, "..")))
    import galsim

def check_dep(f, *args, **kwargs):
    """Check that some function raises a GalSimDeprecationWarning as a warning, but not an error.
    """
    #print 'Check dep: ',f,args,kwargs
    import warnings
    # Cause all warnings to always be triggered.
    # Important in case we want to trigger the same one twice in the test suite.
    warnings.simplefilter("always")

    # Check that f() raises a warning, but not an error.
    with warnings.catch_warnings(galsim.GalSimDeprecationWarning) as w:
        res = f(*args, **kwargs)
    #print 'w = ',w
    assert len(w) >= 1, "Calling %s did not raise a warning"%str(f)
    print [ str(wk.message) for wk in w ]
    return res


def test_dep_bandpass():
    """Test the deprecated methods in galsim/deprecated/bandpass.py.
    """
    import time
    t1 = time.time()

    b = galsim.Bandpass(galsim.LookupTable([1.1,2.2,3.0,4.4,5.5], [1.11,2.22,3.33,4.44,5.55]))
    d = lambda w: w**2

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

    t2 = time.time()
    print 'time for %s = %.2f'%(funcname(),t2-t1)


def test_dep_base():
    """Test the deprecated methods in galsim/deprecated/base.py
    """
    import time
    t1 = time.time()

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

    im1, im1b = check_dep(g.drawK)
    im2, im2b = g.drawKImage()
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

    t2 = time.time()
    print 'time for %s = %.2f'%(funcname(),t2-t1)


def test_dep_bounds():
    """Test the deprecated methods in galsim/deprecated/bounds.py
    """
    import time
    t1 = time.time()

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

    t2 = time.time()
    print 'time for %s = %.2f'%(funcname(),t2-t1)
 

def test_dep_chromatic():
    """Test the deprecated methods in galsim/deprecated/chromatic.py
    """
    import time
    t1 = time.time()

    g = galsim.Gaussian(sigma=0.34)
    sed = galsim.SED('wave**3')
    obj = g * sed
    band = galsim.Bandpass('1-((wave-700)/100)**2', blue_limit=600., red_limit=800.)

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

    t2 = time.time()
    print 'time for %s = %.2f'%(funcname(),t2-t1)
 

def test_dep_correlatednoise():
    """Test the deprecated methods in galsim/deprecated/correlatednoise.py
    """
    import time
    t1 = time.time()

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

    t2 = time.time()
    print 'time for %s = %.2f'%(funcname(),t2-t1)


def test_dep_gsobject_ring():
    """Test building a GSObject from a ring test:
    """
    import time
    t1 = time.time()

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
        print 'The assert_raises tests require nose'

    t2 = time.time()
    print 'time for %s = %.2f'%(funcname(),t2-t1)


def test_dep_image():
    """Test that the old obsolete syntax still works (for now)
    """
    # This is the old version of the test_Image_basic function from version 1.0
    import time
    t1 = time.time()

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

    for i in xrange(ntypes):
        array_type = types[i]
        check_dep(galsim.ImageView[array_type], ref_array.astype(array_type))
        check_dep(galsim.ConstImageView[array_type], ref_array.astype(array_type))
        # This next one is normally executed as im = galsim.Image[type]
        check_dep(galsim.image.MetaImage.__getitem__, galsim.Image, array_type)

    # The rest of this is taken from an older version of the Image class test suite that 
    # tests the old syntax.  Might as well keep it.
    import warnings
    with warnings.catch_warnings(galsim.GalSimDeprecationWarning):
        warnings.simplefilter("ignore")

        for i in xrange(ntypes):
            # Check basic constructor from ncol, nrow
            array_type = types[i]
            im1 = galsim.Image[array_type](ncol,nrow)
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
                    assert im1.at(x,y) == 100+10*x+y
                    assert im1.view().at(x,y) == 100+10*x+y
                    assert im2.at(x,y) == 100+10*x+y
                    assert im2_view.at(x,y) == 100+10*x+y
                    im1.setValue(x,y, 10*x + y)
                    im2_view.setValue(x,y, 10*x + y)
                    assert im1(x,y) == 10*x+y
                    assert im1.view()(x,y) == 10*x+y
                    assert im2(x,y) == 10*x+y
                    assert im2_view(x,y) == 10*x+y

            # Check view of given data
            im3_view = galsim.ImageView[array_type](ref_array.astype(array_type))
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

    t2 = time.time()
    print 'time for %s = %.2f'%(funcname(),t2-t1)


def test_dep_noise():
    """Test the deprecated methods in galsim/deprecated/noise.py
    """
    import time
    t1 = time.time()

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

    t2 = time.time()
    print 'time for %s = %.2f'%(funcname(),t2-t1)


def test_dep_random():
    """Test the deprecated methods in galsim/deprecated/random.py
    """
    import time
    t1 = time.time()

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


    t2 = time.time()
    print 'time for %s = %.2f'%(funcname(),t2-t1)


def test_dep_scene():
    """Test the deprecated exclude_bad and exclude_fail args to COSMOSCatalog
    """
    import time
    t1 = time.time()

    path, filename = os.path.split(__file__)
    datapath = os.path.abspath(os.path.join(path, "../examples/data/"))

    # Initialize one that doesn't exclude failures.  It should be >= the previous one in length.
    cat2 = check_dep(galsim.COSMOSCatalog,
                     file_name='real_galaxy_catalog_example.fits',
                     dir=datapath, exclude_fail=False, exclude_bad=False)
    # Initialize a COSMOSCatalog with all defaults.
    cat = galsim.COSMOSCatalog(file_name='real_galaxy_catalog_example.fits',
                               dir=datapath)
    assert cat2.nobjects>=cat.nobjects
    # Equivalent to current exclusion_level='none'
    cat3 = galsim.COSMOSCatalog(file_name='real_galaxy_catalog_example.fits',
                                dir=datapath, exclusion_level='none')
    assert cat2.nobjects==cat3.nobjects

    # Just exclude_bad=True
    cat2 = check_dep(galsim.COSMOSCatalog,
                     file_name='real_galaxy_catalog_example.fits',
                     dir=datapath, exclude_fail=False)  # i.e. leave exclude_bad=True
    assert cat2.nobjects>=cat.nobjects
    # Equivalent to current exclusion_level='bad_ps'
    cat3 = galsim.COSMOSCatalog(file_name='real_galaxy_catalog_example.fits',
                                dir=datapath, exclusion_level='bad_ps')
    assert cat2.nobjects==cat3.nobjects

    # Just exclude_fail=True
    cat2 = check_dep(galsim.COSMOSCatalog,
                     file_name='real_galaxy_catalog_example.fits',
                     dir=datapath, exclude_bad=False)  # i.e. leave exclude_fail=True
    assert cat2.nobjects>=cat.nobjects
    # Equivalent to current exclusion_level='bad_fits'
    cat3 = galsim.COSMOSCatalog(file_name='real_galaxy_catalog_example.fits',
                                dir=datapath, exclusion_level='bad_fits')
    assert cat2.nobjects==cat3.nobjects

    # Both=True
    cat2 = check_dep(galsim.COSMOSCatalog,
                     file_name='real_galaxy_catalog_example.fits',
                     dir=datapath, exclude_fail=True, exclude_bad=True)
    assert cat2.nobjects>=cat.nobjects
    # Equivalent to current exclusion_level='marginal'
    cat3 = galsim.COSMOSCatalog(file_name='real_galaxy_catalog_example.fits',
                                dir=datapath, exclusion_level='marginal')
    assert cat2.nobjects==cat3.nobjects

    t2 = time.time()
    print 'time for %s = %.2f'%(funcname(),t2-t1)


def test_dep_sed():
    """Test the deprecated methods in galsim/deprecated/sed.py.
    """
    import time
    t1 = time.time()

    z = 0.4
    a = galsim.SED(galsim.LookupTable([1,2,3,4,5], [1.1,2.2,3.3,4.4,5.5]),
                   flux_type='fphotons', redshift=0.4)
    b = lambda w: w**2

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

    t2 = time.time()
    print 'time for %s = %.2f'%(funcname(),t2-t1)


def test_dep_shapelet():
    """Test the deprecated methods in galsim/deprecated/shapelet.py
    """
    import time
    t1 = time.time()

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
            p = (n+m)/2
            q = (n-m)/2
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

    t2 = time.time()
    print 'time for %s = %.2f'%(funcname(),t2-t1)


def test_dep_shear():
    """Test the deprecated methods in galsim/deprecated/shear.py
    """
    import time
    t1 = time.time()

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

    t2 = time.time()
    print 'time for %s = %.2f'%(funcname(),t2-t1)

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
