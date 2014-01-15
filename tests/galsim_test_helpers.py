# Copyright 2012, 2013 The GalSim developers:
# https://github.com/GalSim-developers
#
# This file is part of GalSim: The modular galaxy image simulation toolkit.
#
# GalSim is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# GalSim is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with GalSim.  If not, see <http://www.gnu.org/licenses/>
#
import numpy as np
import os
import sys

try:
    import galsim
except ImportError:
    path, filename = os.path.split(__file__)
    sys.path.append(os.path.abspath(os.path.join(path, "..")))
    import galsim


# This file has some helper functions that are used by tests from multiple files to help
# avoid code duplication.

def printval(image1, image2):
    print "New, saved array sizes: ", np.shape(image1.array), np.shape(image2.array)
    print "Sum of values: ", np.sum(image1.array), np.sum(image2.array)
    print "Minimum image value: ", np.min(image1.array), np.min(image2.array)
    print "Maximum image value: ", np.max(image1.array), np.max(image2.array)
    print "Peak location: ", image1.array.argmax(), image2.array.argmax()
    print "Moments Mx, My, Mxx, Myy, Mxy for new array: "
    getmoments(image1)
    print "Moments Mx, My, Mxx, Myy, Mxy for saved array: "
    getmoments(image2)
    #xcen = image2.array.shape[0]/2
    #ycen = image2.array.shape[1]/2
    #print "new image.center = ",image1.array[xcen-3:xcen+4,ycen-3:ycen+4]
    #print "saved image.center = ",image2.array[xcen-3:xcen+4,ycen-3:ycen+4]

def getmoments(image1):
    print 'shape = ',image1.array.shape
    print 'bounds = ',image1.bounds
    xgrid, ygrid = np.meshgrid(np.arange(image1.array.shape[1]) + image1.getXMin(), 
                               np.arange(image1.array.shape[0]) + image1.getYMin())
    mx = np.sum(xgrid * image1.array) / np.sum(image1.array)
    my = np.sum(ygrid * image1.array) / np.sum(image1.array)
    mxx = np.sum(((xgrid-mx)**2) * image1.array) / np.sum(image1.array)
    myy = np.sum(((ygrid-my)**2) * image1.array) / np.sum(image1.array)
    mxy = np.sum((xgrid-mx) * (ygrid-my) * image1.array) / np.sum(image1.array)
    print "    ", mx-image1.getXMin(), my-image1.getYMin(), mxx, myy, mxy
    return mx, my, mxx, myy, mxy

def convertToShear(e1,e2):
    # Convert a distortion (e1,e2) to a shear (g1,g2)
    import math
    e = math.sqrt(e1*e1 + e2*e2)
    g = math.tanh( 0.5 * math.atanh(e) )
    g1 = e1 * (g/e)
    g2 = e2 * (g/e)
    return (g1,g2)

def do_shoot(prof, img, name):
    # For photon shooting, we calculate the number of photons to use based on the target
    # accuracy we are shooting for.  (Pun intended.)
    # For each pixel,
    # uncertainty = sqrt(N_pix) * flux_photon = sqrt(N_tot * flux_pix / flux_tot) * flux_tot / N_tot
    #             = sqrt(flux_pix) * sqrt(flux_tot) / sqrt(N_tot)
    # This is largest for the brightest pixel.  So we use:
    # N = flux_max * flux_tot / photon_shoot_accuracy^2
    photon_shoot_accuracy = 2.e-3
    # The number of decimal places at which to test the photon shooting
    photon_decimal_test = 2

    test_flux = 1.8

    print 'Start do_shoot'
    # Test photon shooting for a particular profile (given as prof). 
    # Since shooting implicitly convolves with the pixel, we need to compare it to 
    # the given profile convolved with a pixel.
    pix = galsim.Pixel(scale=img.scale)
    compar = galsim.Convolve(prof,pix)
    compar.draw(img)
    flux_max = img.array.max()
    print 'prof.getFlux = ',prof.getFlux()
    print 'compar.getFlux = ',compar.getFlux()
    print 'flux_max = ',flux_max
    flux_tot = img.array.sum()
    print 'flux_tot = ',flux_tot
    if flux_max > 1.:
        # Since the number of photons required for a given accuracy level (in terms of 
        # number of decimal places), we rescale the comparison by the flux of the 
        # brightest pixel.
        compar /= flux_max
        img /= flux_max
        prof /= flux_max
        # The formula for number of photons needed is:
        # nphot = flux_max * flux_tot / photon_shoot_accuracy**2
        # But since we rescaled the image by 1/flux_max, it becomes
        nphot = flux_tot / flux_max / photon_shoot_accuracy**2
    elif flux_max < 0.1:
        # If the max is very small, at least bring it up to 0.1, so we are testing something.
        scale = 0.1 / flux_max;
        print 'scale = ',scale
        compar *= scale
        img *= scale
        prof *= scale
        nphot = flux_max * flux_tot * scale * scale / photon_shoot_accuracy**2
    else:
        nphot = flux_max * flux_tot / photon_shoot_accuracy**2
    print 'prof.getFlux => ',prof.getFlux()
    print 'compar.getFlux => ',compar.getFlux()
    print 'img.sum => ',img.array.sum()
    print 'img.max => ',img.array.max()
    print 'nphot = ',nphot
    img2 = img.copy()

    # Use a deterministic random number generator so we don't fail tests because of rare flukes
    # in the random numbers.
    rng = galsim.UniformDeviate(12345)

    prof.drawShoot(img2, n_photons=nphot, poisson_flux=False, rng=rng)
    print 'img2.sum => ',img2.array.sum()
    np.testing.assert_array_almost_equal(
            img2.array, img.array, photon_decimal_test,
            err_msg="Photon shooting for %s disagrees with expected result"%name)

    # Test normalization
    dx = img.scale
    # Test with a large image to make sure we capture enough of the flux
    # even for slow convergers like Airy (which needs a _very_ large image) or Sersic.
    if 'Airy' in name:
        img = galsim.ImageD(2048,2048, scale=dx)
    elif 'Sersic' in name or 'DeVauc' in name:
        img = galsim.ImageD(512,512, scale=dx)
    else:
        img = galsim.ImageD(128,128, scale=dx)
    compar.setFlux(test_flux)
    compar.draw(img, normalization="surface brightness")
    print 'img.sum = ',img.array.sum(),'  cf. ',test_flux/(dx*dx)
    np.testing.assert_almost_equal(img.array.sum() * dx*dx, test_flux, 4,
            err_msg="Surface brightness normalization for %s disagrees with expected result"%name)
    compar.draw(img, normalization="flux")
    print 'img.sum = ',img.array.sum(),'  cf. ',test_flux
    np.testing.assert_almost_equal(img.array.sum(), test_flux, 4,
            err_msg="Flux normalization for %s disagrees with expected result"%name)

    prof.setFlux(test_flux)
    scale = test_flux / flux_tot # from above
    nphot *= scale * scale
    print 'nphot -> ',nphot
    if 'InterpolatedImage' in name:
        nphot *= 10
        print 'nphot -> ',nphot
    prof.drawShoot(img, n_photons=nphot, normalization="surface brightness", poisson_flux=False,
                   rng=rng)
    print 'img.sum = ',img.array.sum(),'  cf. ',test_flux/(dx*dx)
    np.testing.assert_almost_equal(img.array.sum() * dx*dx, test_flux, photon_decimal_test,
            err_msg="Photon shooting SB normalization for %s disagrees with expected result"%name)
    prof.drawShoot(img, n_photons=nphot, normalization="flux", poisson_flux=False, rng=rng)
    print 'img.sum = ',img.array.sum(),'  cf. ',test_flux
    np.testing.assert_almost_equal(img.array.sum(), test_flux, photon_decimal_test,
            err_msg="Photon shooting flux normalization for %s disagrees with expected result"%name)


def do_kvalue(prof, name):
    """Test that the k-space values are consistent with the real-space values by drawing the 
    profile directly (without any convolution, so using fillXValues) and convolved by a tiny
    Gaussian (effectively a delta function).
    """

    im1 = galsim.ImageF(16,16, scale=0.2)
    prof.draw(im1)

    delta = galsim.Gaussian(sigma = 1.e-8)
    conv = galsim.Convolve([prof,delta])
    im2 = galsim.ImageF(16,16, scale=0.2)
    conv.draw(im2)
    printval(im1,im2)
    np.testing.assert_array_almost_equal(
            im2.array, im1.array, 3,
            err_msg = name + 
            " convolved with a delta function is inconsistent with real-space image.")


def radial_integrate(prof, minr, maxr, dr):
    """A simple helper that calculates int 2pi r f(r) dr, from rmin to rmax
       for an axially symmetric profile.
    """
    import math
    assert prof.isAxisymmetric()
    r = minr
    sum = 0.
    while r < maxr:
        sum += r * prof.xValue(galsim.PositionD(r,0)) 
        r += dr
    sum *= 2. * math.pi * dr
    return sum
 
def funcname():
    import inspect
    return inspect.stack()[1][3]

