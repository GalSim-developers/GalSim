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
import warnings

from galsim_test_helpers import *

path, filename = os.path.split(__file__)
imgdir = os.path.join(path, "SBProfile_comparison_images") # Directory containing the reference

def test_spergelet():
    """ Generate some spergelets in python to check the c++ layer.
    """
    for nu in [-0.5, 0.5]:
        for (j,q) in [(0,0), (5,-5), (5,0), (5,5)]:
            filename = "spergelet_nu{:.2f}_j{}_q{}.fits".format(nu, j, q)
            savedImg = galsim.fits.read(os.path.join(imgdir, filename))
            dx = 0.2
            spergelet = galsim.Spergelet(nu=nu, scale_radius=1.0, j=j, q=q)
            myImg = spergelet.drawKImage(nx=63, ny=63, scale=0.2)[0]

            np.testing.assert_array_almost_equal(
                myImg.array, savedImg.array, 5,
                err_msg="GSObject Spergelet disagrees with expected result")

            # The nu = -0.5 case requires maximum_fft_size larger than 8150
            if not nu < 0:
                test_im = galsim.Image(16, 16, scale=0.2)
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    # This complains about a divide-by-zero, which is a consequence of some
                    # Spergelets having equal amounts of positive and negative surface brightness
                    # values.
                    do_kvalue(spergelet, test_im, "Spergelet")


def test_series_draw():
    """ Test that we can draw Series objects
    """
    a = galsim.SpergelSeries(nu=0.0, scale_radius=1.0, jmax=4)
    im = a.drawImage(nx=15, ny=15, scale=0.2)
    re, im = a.drawKImage(nx=15, ny=15, scale=0.2)

    b = galsim.SeriesConvolution(a, a)
    im = b.drawImage(nx=15, ny=15, scale=0.2)
    re, im = b.drawKImage(nx=15, ny=15, scale=0.2)

def test_series_gsobject_convolution():
    """ Test that we can convolve a Series and a GSObject.
    """
    a = galsim.SpergelSeries(nu=0.0, scale_radius=1.0, jmax=4)
    b = galsim.SeriesConvolution(a, galsim.Gaussian(fwhm=1))
    im = b.drawImage(nx=15, ny=15, scale=0.2)
    re, im = b.drawKImage(nx=15, ny=15, scale=0.2)

def test_spergelseries_decomposeA():
    """ Test that the SpergelSeries decomposition of the A matrix works.
    """
    pass

def test_spergelseries_kValue():
    """ Test that SpergelSeries.kValue() agrees with some Mathematica computations.
    This is essentially testing the handling of the internal .Delta, .ellip, and .phi0 parameters
    directly, without relying on the _decomposeA method.
    """

    #                  inputs                      output
    #       [   nu,    D, eps, phi,    x,    y,      kval]
    vals = [
            [-0.85,  0.1, 0.1, 1.0,  1.1, -3.4, 0.686564 ],
            [ -0.5, -0.1, 0.2, 1.1, -2.3,  3.3, 0.243808 ],
            [  0.0, 0.23, 0.2, 0.2,  0.3, 10.0, 0.015526 ],
            [  1.1, 0.01, 0.5, 0.1,  5.2, -2.7, 0.0003938]
    ]


    for nu, Delta, epsilon, phi0, x, y, kval in vals:
        gal = galsim.SpergelSeries(nu=nu, scale_radius=1.0, jmax=5)
        # Set these by hand instead of through .dilate, .shear, etc...
        gal.epsilon = epsilon
        gal.phi0 = phi0 * galsim.radians
        gal.Delta = Delta
        gal.ri = 1.0
        np.testing.assert_almost_equal(gal.kValue(x,y), kval, 6)

def test_spergelseries():
    """Test that SpergelSeries converges to Spergel.
    """
    # Need a PSF to handle super peaky nu < 0 Spergel profiles.  Otherwise the required fft size
    # explodes.
    psf = galsim.Gaussian(fwhm=0.5)
    # Higher |e| profiles require more terms to converge, so include jmax as a param to vary.
    #       [   nu,    e1, e2,  mu, jmax]
    vals = [
            [-0.85,  0.1, 0.1, 0.9, 5],
            [ -0.5, -0.1, 0.2, 1.1, 6],
            [  0.0, 0.23, 0.2, 0.7, 7],
            [  1.1, 0.01, 0.5, 0.2, 9]
    ]
    for nu, e1, e2, mu, jmax in vals:
        gal_exact = galsim.Spergel(nu=nu, half_light_radius=1.0).shear(e1=e1, e2=e2).dilate(mu)
        gal_series = (galsim.SpergelSeries(nu=nu, half_light_radius=1.0, jmax=jmax)
                      .shear(e1=e1, e2=e2).dilate(mu))
        obj_exact = galsim.Convolve(gal_exact, psf)
        obj_series = galsim.SeriesConvolution(gal_series, psf)
        im_exact = obj_exact.drawImage(nx=32, ny=32, scale=0.2)
        im_series = obj_series.drawImage(nx=32, ny=32, scale=0.2)
        if False:
            import matplotlib.pyplot as plt
            plt.subplot(121)
            plt.imshow(im_exact.array)
            plt.subplot(122)
            plt.imshow(im_series.array)
            plt.show()
        mx = im_exact.array.max()
        np.testing.assert_array_almost_equal(im_exact.array/mx, im_series.array/mx, 5)


def test_spergelseries_dilate():
    """ Check that SpergelSeries with .scale_radius = mu gives same image as SpergelSeries with
    .scale_radius = 1.0 and .Delta = (1-mu**2).
    """
    # Need a PSF to handle super peaky nu < 0 Spergel profiles.  Otherwise the required fft size
    # explodes.
    psf = galsim.Gaussian(fwhm=0.5)
    #                  inputs
    #       [   nu,    D, eps, phi, jmax]
    vals = [
            [-0.85,  0.2, 0.1, 0.9, 9],
            [ -0.5, -0.2, 0.2, 1.1, 9],
            [  0.0, 0.23, 0.2, 0.7, 9],
            [  1.1,  0.2, 0.5, 0.2, 9]
    ]
    for nu, Delta, eps, phi, jmax in vals:
        mu = np.sqrt(1-Delta)
        gal_direct = galsim.SpergelSeries(nu=nu, half_light_radius=1.0, jmax=jmax)
        # Set appropriate attributes directly, instead of through .decomposeA()
        gal_direct.epsilon = eps
        gal_direct.phi0 = phi * galsim.radians
        gal_direct.ri = mu
        gal_direct.Delta = 0.0
        gal_dilate = galsim.SpergelSeries(nu=nu, half_light_radius=1.0, jmax=jmax)
        gal_dilate.epsilon = eps
        gal_dilate.phi0 = phi * galsim.radians
        gal_dilate.ri = 1.0
        gal_dilate.Delta = Delta
        obj_direct = galsim.SeriesConvolution(gal_direct, psf)
        obj_dilate = galsim.SeriesConvolution(gal_dilate, psf)
        im_direct = obj_direct.drawImage(nx=32, ny=32, scale=0.2)
        im_dilate = obj_dilate.drawImage(nx=32, ny=32, scale=0.2)
        mx = im_direct.array.max()
        np.testing.assert_almost_equal(im_direct.array/mx, im_dilate.array/mx, 3)

def test_moffatlet():
    betas = [2,3,4,5]
    srs = [1,1.1,1.2,1.3]
    js = [0,1,2,3]
    qs = [0,1,2,-2]
    for beta, sr, j, q in zip(betas, srs, js, qs):
        moffatlet = galsim.Moffatlet(beta=beta, scale_radius=sr, j=j, q=q)
        test_im = galsim.Image(16, 16, scale=0.2)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            # This complains about a divide-by-zero, which is a consequence of some Moffatlets
            # having equal amounts of positive and negative surface brightness values.
            do_kvalue(moffatlet, test_im, "Moffatlet")

def test_moffatseries():
    """Test that MoffatSeries converges to Moffat.
    """
    # Higher |e| profiles require more terms to converge, so include jmax as a param to vary.
    #       [ beta,    e1,   e2,  mu, jmax]
    vals = [
            [  2.6,  0.01, 0.01, 0.9, 2], # Misses 24% of pixels for jmax=1
            [  3.2, -0.01, 0.02, 1.1, 1], # Misses 32% of pixels for jmax=0
            [  4.4,  0.03, 0.02, 0.7, 2], # Misses 12% of pixels for jmax=1
            [  5.1,  0.01, 0.02, 0.2, 3]  # Misses 0.4% of pixels for jmax=2
    ]
    for beta, e1, e2, mu, jmax in vals:
        psf_exact = galsim.Moffat(beta=beta, half_light_radius=1.0).shear(e1=e1, e2=e2).dilate(mu)
        psf_series = (galsim.MoffatSeries(beta=beta, half_light_radius=1.0, jmax=jmax)
                      .shear(e1=e1, e2=e2).dilate(mu))
        im_exact = psf_exact.drawImage(nx=32, ny=32, scale=0.2)
        im_series = psf_series.drawImage(nx=32, ny=32, scale=0.2)
        if False:
            import matplotlib.pyplot as plt
            plt.subplot(121)
            plt.imshow(im_exact.array)
            plt.subplot(122)
            plt.imshow(im_series.array)
            plt.show()
        np.testing.assert_array_almost_equal(im_exact.array, im_series.array, 4,
            "MoffatSeries failed to converge for beta={:0}".format(beta))

if __name__ == "__main__":
    test_spergelet()
    test_series_draw()
    test_series_gsobject_convolution()
    test_spergelseries_decomposeA()
    test_spergelseries_kValue()
    test_spergelseries()
    test_spergelseries_dilate()
    test_moffatlet()
    test_moffatseries()