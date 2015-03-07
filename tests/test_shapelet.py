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
import numpy as np
import os
import sys

from galsim_test_helpers import *

imgdir = os.path.join(".", "SBProfile_comparison_images") # Directory containing the reference
                                                          # images. 

try:
    import galsim
except ImportError:
    path, filename = os.path.split(__file__)
    sys.path.append(os.path.abspath(os.path.join(path, "..")))
    import galsim

# define a series of tests

def test_shapelet_gaussian():
    """Test that the simplest Shapelet profile is equivalent to a Gaussian
    """
    import time
    t1 = time.time()

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
            bvec = np.zeros(galsim.ShapeletSize(order))
            bvec[0] = test_flux
            shapelet = galsim.Shapelet(sigma=sigma, order=order, bvec=bvec)
            shapelet.drawImage(im2, method='no_pixel')
            printval(im2,im1)
            np.testing.assert_array_almost_equal(
                    im1.array, im2.array, 5,
                    err_msg="Shapelet with (only) b00=1 disagrees with Gaussian result"
                    "for flux=%f, sigma=%f, order=%d"%(test_flux,sigma,order))

    t2 = time.time()
    print 'time for %s = %.2f'%(funcname(),t2-t1)


def test_shapelet_drawImage():
    """Test some measured properties of a drawn shapelet against the supposed true values
    """
    import time
    t1 = time.time()

    ftypes = [np.float32, np.float64]
    scale = 0.2
    test_flux = 23.

    im = galsim.ImageF(129,129, scale=scale)
    for sigma in [1., 0.3, 2.4]:
        for order in [0, 2, 8]:
            bvec = np.zeros(galsim.ShapeletSize(order))
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
            print 'shapelet vector = ',bvec
            shapelet = galsim.Shapelet(sigma=sigma, order=order, bvec=bvec)

            # Test normalization  (This is normally part of do_shoot.  When we eventually 
            # implement photon shooting, we should go back to the normal do_shoot call, 
            # and remove this section.)
            shapelet = shapelet.withFlux(test_flux)
            shapelet.drawImage(im)
            flux = im.array.sum()
            print 'im.sum = ',flux,'  cf. ',test_flux
            np.testing.assert_almost_equal(flux / test_flux, 1., 4,
                    err_msg="Flux normalization for Shapelet disagrees with expected result")

            # Test centroid
            # Note: this only works if the image has odd sizes.  If they are even, then
            # setCenter doesn't actually set the center to the true center of the image 
            # (since it falls between pixels).
            im.setCenter(0,0)
            x,y = np.meshgrid(np.arange(im.array.shape[0]).astype(float) + im.getXMin(), 
                              np.arange(im.array.shape[1]).astype(float) + im.getYMin())
            x *= scale
            y *= scale
            flux = im.array.sum()
            mx = (x*im.array).sum() / flux
            my = (y*im.array).sum() / flux
            conv = galsim.Convolve([shapelet, galsim.Pixel(scale)])
            print 'centroid = ',mx,my,' cf. ',conv.centroid()
            np.testing.assert_almost_equal(mx, shapelet.centroid().x, 3,
                    err_msg="Measured centroid (x) for Shapelet disagrees with expected result")
            np.testing.assert_almost_equal(my, shapelet.centroid().y, 3,
                    err_msg="Measured centroid (y) for Shapelet disagrees with expected result")

    t2 = time.time()
    print 'time for %s = %.2f'%(funcname(),t2-t1)


def test_shapelet_properties():
    """Test some specific numbers for a particular Shapelet profile.
    """
    import time
    t1 = time.time()

    # A semi-random particular vector of coefficients.
    sigma = 1.8
    order = 4
    bvec = [1.3,                               # n = 0
            0.02, 0.03,                        # n = 1
            0.23, -0.19, 0.08,                 # n = 2
            0.01, 0.02, 0.04, -0.03,           # n = 3
            -0.09, 0.07, -0.11, -0.08, 0.11]   # n = 4

    shapelet = galsim.Shapelet(sigma=sigma, order=order, bvec=bvec)

    # Check flux
    flux = bvec[0] + bvec[5] + bvec[14]
    np.testing.assert_almost_equal(shapelet.getFlux(), flux, 10)
    # Check centroid
    cen = galsim.PositionD(bvec[1],-bvec[2]) + np.sqrt(2.) * galsim.PositionD(bvec[8],-bvec[9])
    cen *= 2. * sigma / flux
    np.testing.assert_almost_equal(shapelet.centroid().x, cen.x, 10)
    np.testing.assert_almost_equal(shapelet.centroid().y, cen.y, 10)
    # Check Fourier properties
    np.testing.assert_almost_equal(shapelet.maxK(), 4.61738371186, 10)
    np.testing.assert_almost_equal(shapelet.stepK(), 0.195133742529, 10)
    # Check image values in real and Fourier space
    zero = galsim.PositionD(0., 0.)
    np.testing.assert_almost_equal(shapelet.kValue(zero), flux+0j, 10)
    np.testing.assert_almost_equal(shapelet.xValue(zero), 0.0653321217013, 10)

    # Check picklability
    #do_pickle(shapelet)

    t2 = time.time()
    print 'time for %s = %.2f'%(funcname(),t2-t1)

def test_shapelet_fit():
    """Test fitting a Shapelet decomposition of an image
    """
    import time
    t1 = time.time()

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
        shapelet = galsim.FitShapelet(sigma, 10, im1, normalization=norm)
        print 'fitted shapelet coefficients = ',shapelet.bvec

        # Check flux
        print 'flux = ',shapelet.getFlux(),'  cf. ',flux
        np.testing.assert_almost_equal(shapelet.getFlux() / flux, 1., 1,
                err_msg="Fitted shapelet has the wrong flux")

        # Test centroid
        print 'centroid = ',shapelet.centroid(),'  cf. ',conv.centroid()
        np.testing.assert_almost_equal(shapelet.centroid().x, conv.centroid().x, 2,
                err_msg="Fitted shapelet has the wrong centroid (x)")
        np.testing.assert_almost_equal(shapelet.centroid().y, conv.centroid().y, 2,
                err_msg="Fitted shapelet has the wrong centroid (y)")

        # Test drawing image from shapelet
        im2 = im1.copy()
        shapelet.drawImage(im2, method=method)
        # Check that images are close to the same:
        print 'norm(diff) = ',np.sum((im1.array-im2.array)**2)
        print 'norm(im) = ',np.sum(im1.array**2)
        assert np.sum((im1.array-im2.array)**2) < 1.e-3 * np.sum(im1.array**2)

        # Remeasure -- should now be very close to the same.
        shapelet2 = galsim.FitShapelet(sigma, 10, im2, normalization=norm)
        np.testing.assert_equal(shapelet.sigma, shapelet2.sigma,
                err_msg="Second fitted shapelet has the wrong sigma")
        np.testing.assert_equal(shapelet.order, shapelet2.order,
                err_msg="Second fitted shapelet has the wrong order")
        np.testing.assert_almost_equal(shapelet.bvec, shapelet2.bvec, 6,
                err_msg="Second fitted shapelet coefficients do not match original")

    t2 = time.time()
    print 'time for %s = %.2f'%(funcname(),t2-t1)

def test_shapelet_adjustments():
    """Test that adjusting the Shapelet profile in various ways does the right thing
    """
    import time
    t1 = time.time()

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

    # Test that the Shapelet withFlux does the same thing as the GSObject withFlux
    gsref_shapelet = galsim.GSObject(ref_shapelet)  # Make it opaque to the Shapelet versions
    gsref_shapelet.withFlux(23.).drawImage(ref_im, method='no_pixel')
    shapelet = galsim.Shapelet(sigma=sigma, order=order, bvec=bvec)
    shapelet.withFlux(23.).drawImage(im, method='no_pixel')
    np.testing.assert_array_almost_equal(
        im.array, ref_im.array, 6,
        err_msg="Shapelet withFlux disagrees with GSObject withFlux")

    # Test that scaling the Shapelet flux does the same thing as the GSObject scaling
    gsref_shapelet *= 0.23
    gsref_shapelet.drawImage(ref_im, method='no_pixel')
    shapelet *= 0.23
    shapelet.drawImage(im, method='no_pixel')
    np.testing.assert_array_almost_equal(
        im.array, ref_im.array, 6,
        err_msg="Shapelet *= 0.23 disagrees with GSObject *= 0.23")

    # Test that the Shapelet rotate does the same thing as the GSObject rotate
    gsref_shapelet.rotate(23. * galsim.degrees).drawImage(ref_im, method='no_pixel')
    shapelet.rotate(23. * galsim.degrees).drawImage(im, method='no_pixel')
    np.testing.assert_array_almost_equal(
        im.array, ref_im.array, 6,
        err_msg="Shapelet rotate disagrees with GSObject rotate")

    # Test that the Shapelet dilate does the same thing as the GSObject dilate
    gsref_shapelet.dilate(1.3).drawImage(ref_im, method='no_pixel')
    shapelet.dilate(1.3).drawImage(im, method='no_pixel')
    np.testing.assert_array_almost_equal(
        im.array, ref_im.array, 6,
        err_msg="Shapelet dilate disagrees with GSObject dilate")

    # Test that the Shapelet magnify does the same thing as the GSObject magnify
    gsref_shapelet.magnify(0.8).drawImage(ref_im, method='no_pixel')
    shapelet.magnify(0.8).drawImage(im, method='no_pixel')
    np.testing.assert_array_almost_equal(
        im.array, ref_im.array, 6,
        err_msg="Shapelet magnify disagrees with GSObject magnify")

    # Test that lens works on Shapelet
    gsref_shapelet.lens(-0.05, 0.15, 1.1).drawImage(ref_im, method='no_pixel')
    shapelet.lens(-0.05, 0.15, 1.1).drawImage(im, method='no_pixel')
    np.testing.assert_array_almost_equal(
        im.array, ref_im.array, 6,
        err_msg="Shapelet lens disagrees with GSObject lens")

    t2 = time.time()
    print 'time for %s = %.2f'%(funcname(),t2-t1)



if __name__ == "__main__":
    test_shapelet_gaussian()
    test_shapelet_drawImage()
    test_shapelet_properties()
    test_shapelet_fit()
    test_shapelet_adjustments()
