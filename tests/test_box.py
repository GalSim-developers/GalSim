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

import numpy as np
import os

import galsim
from galsim_test_helpers import *

path, filename = os.path.split(__file__)
imgdir = os.path.join(path, "SBProfile_comparison_images") # Directory containing the reference
                                                           # images.

@timer
def test_box(run_slow):
    """Test the generation of a specific box profile against a known result.
    """
    savedImg = galsim.fits.read(os.path.join(imgdir, "box_1.fits"))
    myImg = galsim.ImageF(savedImg.bounds, scale=0.2)
    myImg.setCenter(0,0)
    test_flux = 1.8

    pixel = galsim.Pixel(scale=1, flux=1)
    pixel.drawImage(myImg, method="sb", use_true_center=False)
    np.testing.assert_array_almost_equal(
            myImg.array, savedImg.array, 5,
            err_msg="Using GSObject Pixel disagrees with expected result")
    np.testing.assert_array_equal(
            pixel.scale, 1,
            err_msg="Pixel scale returned wrong value")

    # Check with default_params
    pixel = galsim.Pixel(scale=1, flux=1, gsparams=default_params)
    pixel.drawImage(myImg, method="sb", use_true_center=False)
    np.testing.assert_array_almost_equal(
            myImg.array, savedImg.array, 5,
            err_msg="Using GSObject Pixel with default_params disagrees with expected result")
    pixel = galsim.Pixel(scale=1, flux=1, gsparams=galsim.GSParams())
    pixel.drawImage(myImg, method="sb", use_true_center=False)
    np.testing.assert_array_almost_equal(
            myImg.array, savedImg.array, 5,
            err_msg="Using GSObject Pixel with GSParams() disagrees with expected result")

    # Use non-unity values.
    pixel = galsim.Pixel(flux=1.7, scale=2.3)
    gsp = galsim.GSParams(xvalue_accuracy=1.e-8, kvalue_accuracy=1.e-8)
    pixel2 = galsim.Pixel(flux=1.7, scale=2.3, gsparams=gsp)
    assert pixel2 != pixel
    assert pixel2 == pixel.withGSParams(xvalue_accuracy=1.e-8, kvalue_accuracy=1.e-8)

    # Test photon shooting.
    do_shoot(pixel,myImg,"Pixel")

    # Check picklability
    check_pickle(pixel, lambda x: x.drawImage(method='no_pixel'))
    check_pickle(pixel)
    check_pickle(galsim.Pixel(1))

    # Check that non-square Box profiles work correctly
    scale = 0.2939  # Use a strange scale here to make sure that the centers of the pixels
                    # never fall on the box edge, otherwise it gets a bit weird to know what
                    # the correct SB value is for that pixel.
    im = galsim.ImageF(16,16, scale=scale)
    gsp = galsim.GSParams(maximum_fft_size = 30000)
    for (width,height) in [ (3,2), (1.7, 2.7), (2.2222, 3.1415) ]:
        box = galsim.Box(width=width, height=height, flux=test_flux, gsparams=gsp)
        check_basic(box, "Box with width,height = %f,%f"%(width,height))
        do_shoot(box,im,"Box with width,height = %f,%f"%(width,height))
        if run_slow:
            # These are slow because they require a pretty huge fft.
            # So only do them if running as main.
            do_kvalue(box,im,"Box with width,height = %f,%f"%(width,height))
        cen = galsim.PositionD(0, 0)
        np.testing.assert_equal(box.centroid, cen)
        np.testing.assert_almost_equal(box.kValue(cen), (1+0j) * test_flux)
        np.testing.assert_almost_equal(box.flux, test_flux)
        np.testing.assert_almost_equal(box.xValue(cen), box.max_sb)
        np.testing.assert_almost_equal(box.xValue(width/2.-0.001, height/2.-0.001), box.max_sb)
        np.testing.assert_almost_equal(box.xValue(width/2.-0.001, height/2.+0.001), 0.)
        np.testing.assert_almost_equal(box.xValue(width/2.+0.001, height/2.-0.001), 0.)
        np.testing.assert_almost_equal(box.xValue(width/2.+0.001, height/2.+0.001), 0.)
        np.testing.assert_array_equal(
                box.width, width,
                err_msg="Box width returned wrong value")
        np.testing.assert_array_equal(
                box.height, height,
                err_msg="Box height returned wrong value")

        gsp2 = galsim.GSParams(xvalue_accuracy=1.e-8, kvalue_accuracy=1.e-8)
        box2 = galsim.Box(width=width, height=height, flux=test_flux, gsparams=gsp2)
        assert box2 != box
        assert box2 == box.withGSParams(gsp2)
        assert box2 != box.withGSParams(xvalue_accuracy=1.e-8, kvalue_accuracy=1.e-8)
        assert box2.withGSParams(maximum_fft_size=30000) == box.withGSParams(xvalue_accuracy=1.e-8, kvalue_accuracy=1.e-8)

    # Check picklability
    check_pickle(box, lambda x: x.drawImage(method='no_pixel'))
    check_pickle(box)
    check_pickle(galsim.Box(1,1))

    # Check sheared boxes the same way
    box = galsim.Box(width=3, height=2, flux=test_flux, gsparams=gsp)
    box = box.shear(galsim.Shear(g1=0.2, g2=-0.3))
    check_basic(box, "Sheared Box", approx_maxsb=True)
    do_shoot(box,im, "Sheared Box")
    if run_slow:
        do_kvalue(box,im, "Sheared Box")
        check_pickle(box, lambda x: x.drawImage(method='no_pixel'))
        check_pickle(box)
    cen = galsim.PositionD(0, 0)
    np.testing.assert_equal(box.centroid, cen)
    np.testing.assert_almost_equal(box.kValue(cen), (1+0j) * test_flux)
    np.testing.assert_almost_equal(box.flux, test_flux)
    np.testing.assert_almost_equal(box.xValue(cen), box.max_sb)

    # This is also a profile that may be convolved using real space convolution, so test that.
    if run_slow:
        conv = galsim.Convolve(box, galsim.Pixel(scale=scale), real_space=True)
        check_basic(conv, "Sheared Box convolved with pixel in real space",
                    approx_maxsb=True, scale=0.2)
        do_kvalue(conv,im, "Sheared Box convolved with pixel in real space")
        check_pickle(conv, lambda x: x.xValue(0.123,-0.456))
        check_pickle(conv)


@timer
def test_tophat():
    """Test the generation of a specific tophat profile against a known result.
    """
    savedImg = galsim.fits.read(os.path.join(imgdir, "tophat_101.fits"))
    myImg = galsim.ImageF(savedImg.bounds, scale=0.2)
    myImg.setCenter(0,0)
    test_flux = 1.8

    # There are numerical issues with using radius = 1, since many points are right on the edge
    # of the circle.  e.g. (+-1,0), (0,+-1), (+-0.6,+-0.8), (+-0.8,+-0.6).  And in practice, some
    # of these end up getting drawn and not others, which means it's not a good choice for a unit
    # test since it wouldn't be any less correct for a different subset of these points to be
    # drawn. Using r = 1.01 solves this problem and makes the result symmetric.
    tophat = galsim.TopHat(radius=1.01, flux=1)
    tophat.drawImage(myImg, method="sb", use_true_center=False)
    np.testing.assert_array_almost_equal(
            myImg.array, savedImg.array, 5,
            err_msg="Using GSObject TopHat disagrees with expected result")
    np.testing.assert_array_equal(
            tophat.radius, 1.01,
            err_msg="TopHat radius returned wrong value")

    # Check with default_params
    tophat = galsim.TopHat(radius=1.01, flux=1, gsparams=default_params)
    tophat.drawImage(myImg, method="sb", use_true_center=False)
    np.testing.assert_array_almost_equal(
            myImg.array, savedImg.array, 5,
            err_msg="Using GSObject TopHat with default_params disagrees with expected result")
    tophat = galsim.TopHat(radius=1.01, flux=1, gsparams=galsim.GSParams())
    tophat.drawImage(myImg, method="sb", use_true_center=False)
    np.testing.assert_array_almost_equal(
            myImg.array, savedImg.array, 5,
            err_msg="Using GSObject TopHat with GSParams() disagrees with expected result")

    # Use non-unity values.
    tophat = galsim.TopHat(flux=1.7, radius=2.3)
    gsp = galsim.GSParams(xvalue_accuracy=1.e-8, kvalue_accuracy=1.e-8)
    tophat2 = galsim.TopHat(flux=1.7, radius=2.3, gsparams=gsp)
    assert tophat2 != tophat
    assert tophat2 == tophat.withGSParams(xvalue_accuracy=1.e-8, kvalue_accuracy=1.e-8)

    # Test photon shooting.
    do_shoot(tophat,myImg,"TopHat")

    # Test shoot and kvalue
    scale = 0.2939
    im = galsim.ImageF(16,16, scale=scale)
    # The choices of radius here are fairly specific.  If the edge of the circle comes too close
    # to the center of one of the pixels, then the test will fail, since the Fourier draw method
    # will blur the edge a bit and give some flux to that pixel.
    for radius in [ 1.2, 0.93, 2.11 ]:
        tophat = galsim.TopHat(radius=radius, flux=test_flux)
        check_basic(tophat, "TopHat with radius = %f"%radius)
        do_shoot(tophat,im,"TopHat with radius = %f"%radius)
        do_kvalue(tophat,im,"TopHat with radius = %f"%radius)

        # This is also a profile that may be convolved using real space convolution, so test that.
        conv = galsim.Convolve(tophat, galsim.Pixel(scale=scale), real_space=True)
        check_basic(conv, "TopHat convolved with pixel in real space",
                    approx_maxsb=True, scale=0.2)
        do_kvalue(conv,im, "TopHat convolved with pixel in real space")

        cen = galsim.PositionD(0, 0)
        np.testing.assert_equal(tophat.centroid, cen)
        np.testing.assert_almost_equal(tophat.kValue(cen), (1+0j) * test_flux)
        np.testing.assert_almost_equal(tophat.flux, test_flux)
        np.testing.assert_almost_equal(tophat.xValue(cen), tophat.max_sb)
        np.testing.assert_almost_equal(tophat.xValue(radius-0.001, 0.), tophat.max_sb)
        np.testing.assert_almost_equal(tophat.xValue(0., radius-0.001), tophat.max_sb)
        np.testing.assert_almost_equal(tophat.xValue(radius+0.001, 0.), 0.)
        np.testing.assert_almost_equal(tophat.xValue(0., radius+0.001), 0.)

    # Check picklability
    check_pickle(tophat, lambda x: x.drawImage(method='no_pixel'))
    check_pickle(tophat)
    check_pickle(galsim.TopHat(1))

    # Check sheared tophat the same way
    tophat = galsim.TopHat(radius=1.2, flux=test_flux)
    # Again, the test is very sensitive to the choice of shear here.  Most values fail because
    # some pixel center gets too close to the resulting ellipse for the fourier draw to match
    # the real-space draw at the required accuracy.
    tophat = tophat.shear(galsim.Shear(g1=0.15, g2=-0.33))
    check_basic(tophat, "Sheared TopHat")
    do_shoot(tophat,im, "Sheared TopHat")
    do_kvalue(tophat,im, "Sheared TopHat")
    cen = galsim.PositionD(0, 0)
    np.testing.assert_equal(tophat.centroid, cen)
    np.testing.assert_almost_equal(tophat.kValue(cen), (1+0j) * test_flux)
    np.testing.assert_almost_equal(tophat.flux, test_flux)
    np.testing.assert_almost_equal(tophat.xValue(cen), tophat.max_sb)

    # Check picklability
    check_pickle(tophat, lambda x: x.drawImage(method='no_pixel'))
    check_pickle(tophat)

    # Check real-space convolution of the sheared tophat.
    conv = galsim.Convolve(tophat, galsim.Pixel(scale=scale), real_space=True)
    check_basic(conv, "Sheared TopHat convolved with pixel in real space",
                approx_maxsb=True, scale=0.2)
    do_kvalue(conv,im, "Sheared TopHat convolved with pixel in real space")


@timer
def test_box_shoot():
    """Test Box with photon shooting.  Particularly the flux of the final image.
    """
    rng = galsim.BaseDeviate(1234)
    obj = galsim.Box(width=1.3, height=2.4, flux=1.e4)
    im = galsim.Image(100,100, scale=1)
    im.setCenter(0,0)
    added_flux, photons = obj.drawPhot(im, poisson_flux=False, rng=rng.duplicate())
    print('obj.flux = ',obj.flux)
    print('added_flux = ',added_flux)
    print('photon fluxes = ',photons.flux.min(),'..',photons.flux.max())
    print('image flux = ',im.array.sum())
    assert np.isclose(added_flux, obj.flux)
    assert np.isclose(im.array.sum(), obj.flux)
    photons2 = obj.makePhot(poisson_flux=False, rng=rng)
    assert photons2 == photons, "Box makePhot not equivalent to drawPhot"

    obj = galsim.Pixel(scale=9.3, flux=1.e4)
    added_flux, photons = obj.drawPhot(im, poisson_flux=False, rng=rng.duplicate())
    print('obj.flux = ',obj.flux)
    print('added_flux = ',added_flux)
    print('photon fluxes = ',photons.flux.min(),'..',photons.flux.max())
    print('image flux = ',im.array.sum())
    assert np.isclose(added_flux, obj.flux)
    assert np.isclose(im.array.sum(), obj.flux)
    photons2 = obj.makePhot(poisson_flux=False, rng=rng)
    assert photons2 == photons, "Pixel makePhot not equivalent to drawPhot"

    obj = galsim.TopHat(radius=4.7, flux=1.e4)
    added_flux, photons = obj.drawPhot(im, poisson_flux=False, rng=rng.duplicate())
    print('obj.flux = ',obj.flux)
    print('added_flux = ',added_flux)
    print('photon fluxes = ',photons.flux.min(),'..',photons.flux.max())
    print('image flux = ',im.array.sum())
    assert np.isclose(added_flux, obj.flux)
    assert np.isclose(im.array.sum(), obj.flux)
    photons2 = obj.makePhot(poisson_flux=False, rng=rng)
    assert photons2 == photons, "TopHat makePhot not equivalent to drawPhot"


@timer
def test_ne():
    """Test base.py GSObjects for not-equals."""
    # Define some universal gsps
    gsp = galsim.GSParams(maxk_threshold=1.1e-3, folding_threshold=5.1e-3)

    # Pixel.  Params include scale, flux, gsparams.
    # gsparams.
    # The following should all test unequal:
    gals = [galsim.Pixel(scale=1.0),
            galsim.Pixel(scale=1.1),
            galsim.Pixel(scale=1.0, flux=1.1),
            galsim.Pixel(scale=1.0, gsparams=gsp)]
    check_all_diff(gals)

    # Box.  Params include width, height, flux, gsparams.
    # gsparams.
    # The following should all test unequal:
    gals = [galsim.Box(width=1.0, height=1.0),
            galsim.Box(width=1.1, height=1.0),
            galsim.Box(width=1.0, height=1.1),
            galsim.Box(width=1.0, height=1.0, flux=1.1),
            galsim.Box(width=1.0, height=1.0, gsparams=gsp)]
    check_all_diff(gals)

    # TopHat.  Params include radius, flux, gsparams.
    # gsparams.
    # The following should all test unequal:
    gals = [galsim.TopHat(radius=1.0),
            galsim.TopHat(radius=1.1),
            galsim.TopHat(radius=1.0, flux=1.1),
            galsim.TopHat(radius=1.0, gsparams=gsp)]
    check_all_diff(gals)


if __name__ == "__main__":
    runtests(__file__)
