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
def test_shapelet_gaussian():
    """Test that the simplest Shapelet profile is equivalent to a Gaussian
    """
    ftypes = [np.float32, np.float64]
    scale = 0.2
    test_flux = 23.

    # First, a Shapelet with only b_00 = 1 should be identically a Gaussian
    im1 = galsim.ImageF(64,64, scale=scale)
    im2 = galsim.ImageF(64,64, scale=scale)
    for sigma in [1., 0.6, 2.4]:
        gauss = galsim.Gaussian(flux=test_flux, sigma=sigma)
        gauss.drawImage(im1, method='no_pixel')
        for order in [0, 2, 8]:
            bvec = np.zeros(galsim.Shapelet.size(order))
            bvec[0] = test_flux
            shapelet = galsim.Shapelet(sigma=sigma, order=order, bvec=bvec)
            shapelet.drawImage(im2, method='no_pixel')
            printval(im2,im1)
            np.testing.assert_array_almost_equal(
                    im1.array, im2.array, 5,
                    err_msg="Shapelet with (only) b00=1 disagrees with Gaussian result"
                            "for flux=%f, sigma=%f, order=%d"%(test_flux,sigma,order))
            np.testing.assert_almost_equal(
                    gauss.max_sb, shapelet.max_sb, 5,
                    err_msg="Shapelet max_sb did not match Gaussian max_sb")
            np.testing.assert_almost_equal(
                    gauss.flux, shapelet.flux, 5,
                    err_msg="Shapelet flux did not match Gaussian flux")


@timer
def test_shapelet_drawImage():
    """Test some measured properties of a drawn shapelet against the supposed true values
    """
    ftypes = [np.float32, np.float64]
    scale = 0.2
    test_flux = 23.

    im = galsim.ImageF(129,129, scale=scale)
    for sigma in [1., 0.3, 2.4]:
        for order in [0, 2, 8]:
            bvec = np.zeros(galsim.Shapelet.size(order))
            bvec[0] = 1.  # N,m = 0,0
            k = 0
            for n in range(1,order+1):
                k += n+1
                if n%2 == 0:  # even n
                    bvec[k] = 0.23/(n*n)        # N,m = n,0  or p,q = n/2,n/2
                    if n >= 2:
                        bvec[k-2] = 0.14/n      # N,m = n,2  real part
                        bvec[k-1] = -0.08/n     # N,m = n,2  imag part
                else:  # odd n
                    if n >= 1:
                        bvec[k-1] = -0.08/n**3.2    # N,m = n,1  real part
                        bvec[k] = 0.05/n**2.1       # N,m = n,1  imag part
                    if n >= 3:
                        bvec[k-3] = 0.31/n**4.2    # N,m = n,3  real part
                        bvec[k-2] = -0.18/n**3.9       # N,m = n,3  imag part
            print('shapelet vector = ',bvec)
            shapelet = galsim.Shapelet(sigma=sigma, order=order, bvec=bvec)

            gsp = galsim.GSParams(xvalue_accuracy=1.e-8, kvalue_accuracy=1.e-8)
            shapelet2 = galsim.Shapelet(sigma=sigma, order=order, bvec=bvec, gsparams=gsp)
            assert shapelet2 != shapelet
            assert shapelet2 == shapelet.withGSParams(gsp)
            assert shapelet2 == shapelet.withGSParams(xvalue_accuracy=1.e-8, kvalue_accuracy=1.e-8)

            check_basic(shapelet, "Shapelet", approx_maxsb=True)

            # Test normalization  (This is normally part of do_shoot.  When we eventually
            # implement photon shooting, we should go back to the normal do_shoot call,
            # and remove this section.)
            shapelet = shapelet.withFlux(test_flux)
            shapelet.drawImage(im)
            flux = im.array.sum()
            print('im.sum = ',flux,'  cf. ',test_flux)
            np.testing.assert_almost_equal(flux / test_flux, 1., 4,
                    err_msg="Flux normalization for Shapelet disagrees with expected result")
            np.testing.assert_allclose(
                    im.array.max(), shapelet.max_sb * im.scale**2, rtol=0.1,
                    err_msg="Shapelet max_sb did not match maximum pixel")

            # Test centroid
            # Note: this only works if the image has odd sizes.  If they are even, then
            # setCenter doesn't actually set the center to the true center of the image
            # (since it falls between pixels).
            im.setCenter(0,0)
            x,y = np.meshgrid(np.arange(im.array.shape[0]).astype(float) + im.xmin,
                              np.arange(im.array.shape[1]).astype(float) + im.ymin)
            x *= scale
            y *= scale
            flux = im.array.sum()
            mx = (x*im.array).sum() / flux
            my = (y*im.array).sum() / flux
            conv = galsim.Convolve([shapelet, galsim.Pixel(scale)])
            print('centroid = ',mx,my,' cf. ',conv.centroid)
            np.testing.assert_almost_equal(mx, shapelet.centroid.x, 3,
                    err_msg="Measured centroid (x) for Shapelet disagrees with expected result")
            np.testing.assert_almost_equal(my, shapelet.centroid.y, 3,
                    err_msg="Measured centroid (y) for Shapelet disagrees with expected result")


@timer
def test_shapelet_properties():
    """Test some specific numbers for a particular Shapelet profile.
    """
    # A semi-random particular vector of coefficients.
    sigma = 1.8
    order = 4
    bvec = [1.3,                               # n = 0
            0.02, 0.03,                        # n = 1
            0.23, -0.19, 0.08,                 # n = 2
            0.01, 0.02, 0.04, -0.03,           # n = 3
            -0.09, 0.07, -0.11, -0.08, 0.11]   # n = 4

    shapelet = galsim.Shapelet(sigma=sigma, order=order, bvec=bvec)

    assert shapelet.sigma == sigma
    assert shapelet.order == order
    np.testing.assert_array_equal(shapelet.bvec, bvec)

    check_basic(shapelet, "Shapelet", approx_maxsb=True)

    # Check flux
    flux = bvec[0] + bvec[5] + bvec[14]
    np.testing.assert_almost_equal(shapelet.flux, flux, 10)
    # The max_sb is not very accurate for Shapelet, but in this case it is still ok (matching
    # xValue(0,0), which isn't actually the maximum) to 2 digits.
    np.testing.assert_almost_equal(
            shapelet.xValue(0,0), shapelet.max_sb, 2,
            err_msg="Shapelet max_sb did not match maximum pixel value")
    # Check centroid
    cen = galsim.PositionD(bvec[1],-bvec[2]) + np.sqrt(2.) * galsim.PositionD(bvec[8],-bvec[9])
    cen *= 2. * sigma / flux
    np.testing.assert_almost_equal(shapelet.centroid.x, cen.x, 10)
    np.testing.assert_almost_equal(shapelet.centroid.y, cen.y, 10)
    # Check Fourier properties
    np.testing.assert_almost_equal(shapelet.maxk, 4.61738371186, 10)
    np.testing.assert_almost_equal(shapelet.stepk, 0.195133742529, 10)
    # Check image values in real and Fourier space
    zero = galsim.PositionD(0., 0.)
    np.testing.assert_almost_equal(shapelet.kValue(zero), flux+0j, 10)
    np.testing.assert_almost_equal(shapelet.xValue(zero), 0.0653321217013, 10)

    # Check picklability
    check_pickle(shapelet)

    assert_raises(TypeError, galsim.Shapelet, sigma=sigma)
    assert_raises(TypeError, galsim.Shapelet, sigma=sigma, bvec=bvec)
    assert_raises(TypeError, galsim.Shapelet, order=order, bvec=bvec)
    assert_raises(ValueError, galsim.Shapelet, sigma=sigma, order=5, bvec=bvec)


@timer
def test_shapelet_fit():
    """Test fitting a Shapelet decomposition of an image
    """
    for method, norm in [('no_pixel','f'), ('sb','sb')]:
        # We fit a shapelet approximation of a distorted Moffat profile:
        flux = 20
        psf = galsim.Moffat(beta=3.4, half_light_radius=1.2, flux=flux)
        psf = psf.shear(g1=0.11,g2=0.07).shift(0.03,0.04)
        scale = 0.2
        pixel = galsim.Pixel(scale)
        conv = galsim.Convolve([psf,pixel])
        im1 = conv.drawImage(scale=scale, method=method)

        sigma = 1.2  # Match half-light-radius as a decent first approximation.
        shapelet = galsim.Shapelet.fit(sigma, 10, im1, normalization=norm)
        print('fitted shapelet coefficients = ',shapelet.bvec)

        # Check flux
        print('flux = ',shapelet.flux,'  cf. ',flux)
        np.testing.assert_almost_equal(shapelet.flux / flux, 1., 1,
                err_msg="Fitted shapelet has the wrong flux")

        # Test centroid
        print('centroid = ',shapelet.centroid,'  cf. ',conv.centroid)
        np.testing.assert_almost_equal(shapelet.centroid.x, conv.centroid.x, 2,
                err_msg="Fitted shapelet has the wrong centroid (x)")
        np.testing.assert_almost_equal(shapelet.centroid.y, conv.centroid.y, 2,
                err_msg="Fitted shapelet has the wrong centroid (y)")

        # Test drawing image from shapelet
        im2 = im1.copy()
        shapelet.drawImage(im2, method=method)
        # Check that images are close to the same:
        print('norm(diff) = ',np.sum((im1.array-im2.array)**2))
        print('norm(im) = ',np.sum(im1.array**2))
        print('max(diff) = ',np.max(np.abs(im1.array-im2.array)))
        print('max(im) = ',np.max(np.abs(im1.array)))
        peak_scale = np.max(np.abs(im1.array))*3  # Check agreement to within 3% of peak value.
        np.testing.assert_almost_equal(im2.array/peak_scale, im1.array/peak_scale, decimal=2,
                err_msg="Shapelet version not a good match to original")

        # Remeasure -- should now be very close to the same.
        shapelet2 = galsim.Shapelet.fit(sigma, 10, im2, normalization=norm)
        np.testing.assert_equal(shapelet.sigma, shapelet2.sigma,
                err_msg="Second fitted shapelet has the wrong sigma")
        np.testing.assert_equal(shapelet.order, shapelet2.order,
                err_msg="Second fitted shapelet has the wrong order")
        np.testing.assert_almost_equal(shapelet.bvec, shapelet2.bvec, 6,
                err_msg="Second fitted shapelet coefficients do not match original")

        # Test drawing off center
        im2 = im1.copy()
        offset = galsim.PositionD(0.3,1.4)
        shapelet.drawImage(im2, method=method, offset=offset)
        shapelet2 = galsim.Shapelet.fit(sigma, 10, im2, normalization=norm,
                                        center=im2.true_center + offset)
        np.testing.assert_equal(shapelet.sigma, shapelet2.sigma,
                err_msg="Second fitted shapelet has the wrong sigma")
        np.testing.assert_equal(shapelet.order, shapelet2.order,
                err_msg="Second fitted shapelet has the wrong order")
        np.testing.assert_almost_equal(shapelet.bvec, shapelet2.bvec, 6,
                err_msg="Second fitted shapelet coefficients do not match original")

    assert_raises(ValueError, galsim.Shapelet.fit, sigma, 10, im1, normalization='invalid')

    # Haven't gotten around to implementing this yet...
    im2.wcs = galsim.JacobianWCS(0.2,0.01,0.01,0.2)
    with assert_raises(NotImplementedError):
        galsim.Shapelet.fit(sigma, 10, im2)


@timer
def test_shapelet_adjustments():
    """Test that adjusting the Shapelet profile in various ways does the right thing
    """
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
    ref_shapelet.drawImage(ref_im, scale=scale)

    # Test PQ and NM access
    np.testing.assert_equal(ref_shapelet.getPQ(0,0), (bvec[0],0))
    np.testing.assert_equal(ref_shapelet.getPQ(1,1), (bvec[5],0))
    np.testing.assert_equal(ref_shapelet.getPQ(1,2), (bvec[8],-bvec[9]))
    np.testing.assert_equal(ref_shapelet.getPQ(3,2), (bvec[19],bvec[20]))
    np.testing.assert_equal(ref_shapelet.getNM(0,0), (bvec[0],0))
    np.testing.assert_equal(ref_shapelet.getNM(2,0), (bvec[5],0))
    np.testing.assert_equal(ref_shapelet.getNM(3,-1), (bvec[8],-bvec[9]))
    np.testing.assert_equal(ref_shapelet.getNM(5,1), (bvec[19],bvec[20]))

    # Test that the Shapelet withFlux does the same thing as the GSObject withFlux
    # Make it opaque to the Shapelet versions
    alt_shapelet = ref_shapelet + 0. * galsim.Gaussian(sigma=1)
    alt_shapelet.withFlux(23.).drawImage(ref_im, method='no_pixel')
    shapelet = galsim.Shapelet(sigma=sigma, order=order, bvec=bvec)
    shapelet.withFlux(23.).drawImage(im, method='no_pixel')
    np.testing.assert_array_almost_equal(
        im.array, ref_im.array, 6,
        err_msg="Shapelet withFlux disagrees with GSObject withFlux")

    # Test that scaling the Shapelet flux does the same thing as the GSObject scaling
    (alt_shapelet * 0.23).drawImage(ref_im, method='no_pixel')
    (shapelet * 0.23).drawImage(im, method='no_pixel')
    np.testing.assert_array_almost_equal(
        im.array, ref_im.array, 6,
        err_msg="Shapelet *= 0.23 disagrees with GSObject *= 0.23")

    # Test that the Shapelet rotate does the same thing as the GSObject rotate
    alt_shapelet.rotate(23. * galsim.degrees).drawImage(ref_im, method='no_pixel')
    shapelet.rotate(23. * galsim.degrees).drawImage(im, method='no_pixel')
    np.testing.assert_array_almost_equal(
        im.array, ref_im.array, 6,
        err_msg="Shapelet rotate disagrees with GSObject rotate")

    # Test that the Shapelet dilate does the same thing as the GSObject dilate
    alt_shapelet.dilate(1.3).drawImage(ref_im, method='no_pixel')
    shapelet.dilate(1.3).drawImage(im, method='no_pixel')
    np.testing.assert_array_almost_equal(
        im.array, ref_im.array, 6,
        err_msg="Shapelet dilate disagrees with GSObject dilate")

    # Test that the Shapelet expand does the same thing as the GSObject expand
    alt_shapelet.expand(1.7).drawImage(ref_im, method='no_pixel')
    shapelet.expand(1.7).drawImage(im, method='no_pixel')
    np.testing.assert_array_almost_equal(
        im.array, ref_im.array, 6,
        err_msg="Shapelet expand disagrees with GSObject expand")

    # Test that the Shapelet magnify does the same thing as the GSObject magnify
    alt_shapelet.magnify(0.8).drawImage(ref_im, method='no_pixel')
    shapelet.magnify(0.8).drawImage(im, method='no_pixel')
    np.testing.assert_array_almost_equal(
        im.array, ref_im.array, 6,
        err_msg="Shapelet magnify disagrees with GSObject magnify")

    # Test that lens works on Shapelet
    alt_shapelet.lens(-0.05, 0.15, 1.1).drawImage(ref_im, method='no_pixel')
    shapelet.lens(-0.05, 0.15, 1.1).drawImage(im, method='no_pixel')
    np.testing.assert_array_almost_equal(
        im.array, ref_im.array, 6,
        err_msg="Shapelet lens disagrees with GSObject lens")


@timer
def test_ne():
    """ Check that inequality works as expected."""
    gsp = galsim.GSParams(maxk_threshold=5.1e-3, folding_threshold=1.1e-3)
    objs = [galsim.Shapelet(1., 2),
            galsim.Shapelet(1., 3),
            galsim.Shapelet(2., 2),
            galsim.Shapelet(1., 2, bvec=[1, 0, 0, 0.2, 0.3, -0.1]),
            galsim.Shapelet(1., 2, gsparams=gsp)]
    check_all_diff(objs)


if __name__ == "__main__":
    runtests(__file__)
