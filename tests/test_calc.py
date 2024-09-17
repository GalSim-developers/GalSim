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

import galsim
from galsim_test_helpers import *


@timer
def test_hlr():
    """Test the calculateHLR method.
    """
    # Compare the calculation for a simple Gaussian.
    g1 = galsim.Gaussian(sigma=5, flux=1.7)

    print('g1 native hlr = ',g1.half_light_radius)
    print('g1.calculateHLR = ',g1.calculateHLR())
    print('nyquist scale = ',g1.nyquist_scale)
    # These should be exactly equal.
    np.testing.assert_equal(g1.half_light_radius, g1.calculateHLR(),
                            err_msg="Gaussian.calculateHLR() returned wrong value.")

    # Check for a convolution of two Gaussians.  Should be equivalent, but now will need to
    # do the calculation.
    g2 = galsim.Convolve(galsim.Gaussian(sigma=3, flux=1.3), galsim.Gaussian(sigma=4, flux=23))
    test_hlr = g2.calculateHLR()
    print('g2.calculateHLR = ',test_hlr)
    print('ratio - 1 = ',test_hlr/g1.half_light_radius-1)
    np.testing.assert_almost_equal(test_hlr/g1.half_light_radius, 1.0, decimal=1,
                                   err_msg="Gaussian.calculateHLR() is not accurate.")

    # The default scale is only accurate to around 1 dp.  Using scale = 0.1 is accurate to 3 dp.
    # Note: Nyquist scale is about 4.23 for this profile.
    test_hlr = g2.calculateHLR(scale=0.1)
    print('g2.calculateHLR(scale=0.1) = ',test_hlr)
    print('ratio - 1 = ',test_hlr/g1.half_light_radius-1)
    np.testing.assert_almost_equal(test_hlr/g1.half_light_radius, 1.0, decimal=3,
                                   err_msg="Gaussian.calculateHLR(scale=0.1) is not accurate.")

    # Finally, we don't expect this to be accurate, but make sure the code can handle having
    # more than half the flux in the central pixel.
    test_hlr = g2.calculateHLR(scale=15)
    print('g2.calculateHLR(scale=15) = ',test_hlr)
    print('ratio - 1 = ',test_hlr/g1.half_light_radius-1)
    np.testing.assert_almost_equal(test_hlr/g1.half_light_radius/10, 0.1, decimal=1,
                                   err_msg="Gaussian.calculateHLR(scale=15) is not accurate.")

    # Next, use an Exponential profile
    e1 = galsim.Exponential(scale_radius=5, flux=1.7)

    print('e1 native hlr = ',e1.half_light_radius)
    print('e1.calculateHLR = ',e1.calculateHLR())
    print('nyquist scale = ',e1.nyquist_scale)
    # These should be exactly equal.
    np.testing.assert_equal(e1.half_light_radius, e1.calculateHLR(),
                            err_msg="Exponential.calculateHLR() returned wrong value.")

    # Check for a convolution with a delta function.  Should be equivalent, but now will need to
    # do the calculation.
    e2 = galsim.Convolve(galsim.Exponential(scale_radius=5, flux=1.3),
                         galsim.Gaussian(sigma=1.e-4, flux=23))
    test_hlr = e2.calculateHLR()
    print('e2.calculateHLR = ',test_hlr)
    print('ratio - 1 = ',test_hlr/e1.half_light_radius-1)
    np.testing.assert_almost_equal(test_hlr/e1.half_light_radius, 1.0, decimal=1,
                                   err_msg="Exponential.calculateHLR() is not accurate.")

    # The default scale is only accurate to around 1 dp.  Using scale = 0.1 is accurate to 3 dp.
    # Note: Nyquist scale is about 1.57 for this profile.
    # We can also decrease the size, which should still be accurate, but maybe a little faster.
    # Go a bit more that 2*hlr in units of the pixels.
    size = int(2.2 * e1.half_light_radius / 0.1)
    test_hlr = e2.calculateHLR(scale=0.1, size=size)
    print('e2.calculateHLR(scale=0.1) = ',test_hlr)
    print('ratio - 1 = ',test_hlr/e1.half_light_radius-1)
    np.testing.assert_almost_equal(test_hlr/e1.half_light_radius, 1.0, decimal=3,
                                   err_msg="Exponential.calculateHLR(scale=0.1) is not accurate.")

    # Check that it works if the centroid is not at the origin
    e3 = e2.shift(2,3)
    test_hlr = e3.calculateHLR(scale=0.1)
    print('e3.calculateHLR(scale=0.1) = ',test_hlr)
    print('ratio - 1 = ',test_hlr/e1.half_light_radius-1)
    np.testing.assert_almost_equal(test_hlr/e1.half_light_radius, 1.0, decimal=3,
                                   err_msg="shifted Exponential HLR is not accurate.")

    # Can set a centroid manually.  This should be equivalent to the default.
    print('e3.centroid = ',e3.centroid)
    test_hlr = e3.calculateHLR(scale=0.1, centroid=e3.centroid)
    np.testing.assert_almost_equal(test_hlr/e1.half_light_radius, 1.0, decimal=3,
                                   err_msg="shifted HLR with explicit centroid is not accurate.")

    # The calculateHLR method can also return other radii like r90, rather than r50 using the
    # parameter flux_fraction.  This is also analytic for Exponential
    r90 = 3.889720170 * e1.scale_radius
    test_r90 = e2.calculateHLR(scale=0.1, flux_frac=0.9)
    print('r90 = ',r90)
    print('e2.calculateHLR(scale=0.1, flux_frac=0.9) = ',test_r90)
    print('ratio - 1 = ',test_r90/r90-1)
    np.testing.assert_almost_equal(test_r90/r90, 1.0, decimal=3,
                                   err_msg="Exponential r90 calculation is not accurate.")

    # Check the image version.
    im = e1.drawImage(scale=0.1, nx=2048, ny=2048)  # Needs to be large to get enough flux for the
                                                    # image to get it to 3 digits of accuracy.
    test_hlr = im.calculateHLR()
    print('im.calculateHLR() = ',test_hlr)
    print('ratio - 1 = ',test_hlr/e1.half_light_radius-1)
    np.testing.assert_almost_equal(test_hlr/e1.half_light_radius, 1.0, decimal=3,
                                   err_msg="image.calculateHLR is not accurate.")

    # Check that a non-square image works correctly.  Also, not centered anywhere in particular.
    bounds = galsim.BoundsI(-1234, -1234+2048, 8234, 8234+2099)
    offset = galsim.PositionD(29,1)
    im = e1.drawImage(scale=0.1, bounds=bounds, offset=offset)
    test_hlr = im.calculateHLR(center=im.true_center+offset)
    print('im.calculateHLR() = ',test_hlr)
    print('ratio - 1 = ',test_hlr/e1.half_light_radius-1)
    np.testing.assert_almost_equal(test_hlr/e1.half_light_radius, 1.0, decimal=3,
                                   err_msg="non-square image.calculateHLR is not accurate.")


@timer
def test_sigma():
    """Test the calculateMomentRadius method.
    """
    # Compare the calculation for a simple Gaussian.
    g1 = galsim.Gaussian(sigma=5, flux=1.7)

    print('g1 native sigma = ',g1.sigma)
    print('g1.calculateMomentRadius = ',g1.calculateMomentRadius())
    # These should be exactly equal.
    np.testing.assert_equal(
            g1.sigma, g1.calculateMomentRadius(),
            err_msg="Gaussian.calculateMomentRadius() returned wrong value.")
    np.testing.assert_equal(
            g1.sigma, g1.calculateMomentRadius(rtype='trace'),
            err_msg="Gaussian.calculateMomentRadius(trace) returned wrong value.")
    np.testing.assert_equal(
            g1.sigma, g1.calculateMomentRadius(rtype='det'),
            err_msg="Gaussian.calculateMomentRadius(det) returned wrong value.")
    np.testing.assert_equal(
            (g1.sigma, g1.sigma), g1.calculateMomentRadius(rtype='both'),
            err_msg="Gaussian.calculateMomentRadius(both) returned wrong value.")
    with assert_raises(galsim.GalSimValueError):
        g1.calculateMomentRadius(rtype='invalid')

    # Check for a convolution of two Gaussians.  Should be equivalent, but now will need to
    # do the calculation.
    g2 = galsim.Convolve(galsim.Gaussian(sigma=3, flux=1.3), galsim.Gaussian(sigma=4, flux=23))
    test_sigma = g2.calculateMomentRadius()
    print('g2.calculateMomentRadius = ',test_sigma)
    print('ratio - 1 = ',test_sigma/g1.sigma-1)
    np.testing.assert_almost_equal(
            test_sigma/g1.sigma, 1.0, decimal=1,
            err_msg="Gaussian.calculateMomentRadius() is not accurate.")

    # The default scale and size is only accurate to around 1 dp.  Using scale = 0.1 is accurate
    # to 4 dp.
    test_sigma = g2.calculateMomentRadius(scale=0.1)
    print('g2.calculateMomentRadius(scale=0.1) = ',test_sigma)
    print('ratio - 1 = ',test_sigma/g1.sigma-1)
    np.testing.assert_almost_equal(
            test_sigma/g1.sigma, 1.0, decimal=4,
            err_msg="Gaussian.calculateMomentRadius(scale=0.1) is not accurate.")

    # In this case, the different calculations are eqivalent:
    np.testing.assert_almost_equal(
            test_sigma, g2.calculateMomentRadius(scale=0.1, rtype='trace'),
            err_msg="Gaussian.calculateMomentRadius(trace) is not accurate.")
    np.testing.assert_almost_equal(
            test_sigma, g2.calculateMomentRadius(scale=0.1, rtype='det'),
            err_msg="Gaussian.calculateMomentRadius(trace) is not accurate.")
    np.testing.assert_almost_equal(
            (test_sigma, test_sigma), g2.calculateMomentRadius(scale=0.1, rtype='both'),
            err_msg="Gaussian.calculateMomentRadius(trace) is not accurate.")

    # However, when we shear it, the default (det) measure stays equal to the original sigma, but
    # the trace measure increases by a factor of (1-e^2)^0.25
    g3 = g2.shear(e1=0.4, e2=0.3)
    esq = 0.4**2 + 0.3**2
    sheared_sigma = g3.calculateMomentRadius(scale=0.1)
    print('g3.calculateMomentRadius(scale=0.1) = ',sheared_sigma)
    print('ratio - 1 = ',sheared_sigma/g1.sigma-1)
    sheared_sigma2 = g3.calculateMomentRadius(scale=0.1, rtype='trace')
    print('g3.calculateMomentRadius(scale=0.1,trace) = ',sheared_sigma2)
    print('ratio = ',sheared_sigma2 / g1.sigma)
    print('(1-e^2)^-0.25 = ',(1-esq)**-0.25)
    print('ratio - 1 = ',sheared_sigma2/(g1.sigma*(1.-esq)**-0.25)-1)
    np.testing.assert_almost_equal(
            sheared_sigma/g1.sigma, 1.0, decimal=4,
            err_msg="sheared Gaussian.calculateMomentRadius(scale=0.1) is not accurate.")
    np.testing.assert_almost_equal(
            sheared_sigma2/(g1.sigma*(1.-esq)**-0.25), 1.0, decimal=4,
            err_msg="sheared Gaussian.calculateMomentRadius(scale=0.1,trace) is not accurate.")


    # Next, use an Exponential profile
    e1 = galsim.Exponential(scale_radius=5, flux=1.7)

    # The true "sigma" for this is analytic, but not an attribute.
    e1_sigma = np.sqrt(3.0) * e1.scale_radius
    print('true e1 sigma = sqrt(3) * e1.scale_radius = ',e1_sigma)

    # Test with the default scale and size.
    test_sigma = e1.calculateMomentRadius()
    print('e1.calculateMomentRadius = ',test_sigma)
    print('ratio - 1 = ',test_sigma/e1_sigma-1)
    np.testing.assert_almost_equal(
            test_sigma/e1_sigma, 1.0, decimal=1,
            err_msg="Exponential.calculateMomentRadius() is not accurate.")

    # The default scale and size is only accurate to around 1 dp.  This time we have to both
    # decrease the scale and also increase the size to get 4 dp of precision.
    test_sigma = e1.calculateMomentRadius(scale=0.1, size=2000)
    print('e1.calculateMomentRadius(scale=0.1) = ',test_sigma)
    print('ratio - 1 = ',test_sigma/e1_sigma-1)
    np.testing.assert_almost_equal(
            test_sigma/e1_sigma, 1.0, decimal=4,
            err_msg="Exponential.calculateMomentRadius(scale=0.1) is not accurate.")

    # Check that it works if the centroid is not at the origin
    e3 = e1.shift(2,3)
    test_sigma = e3.calculateMomentRadius(scale=0.1, size=2000)
    print('e1.calculateMomentRadius(scale=0.1) = ',test_sigma)
    print('ratio - 1 = ',test_sigma/e1_sigma-1)
    np.testing.assert_almost_equal(
            test_sigma/e1_sigma, 1.0, decimal=4,
            err_msg="shifted Exponential MomentRadius is not accurate.")

    # Can set a centroid manually.  This should be equivalent to the default.
    print('e3.centroid = ',e3.centroid)
    test_sigma = e3.calculateMomentRadius(scale=0.1, size=2000, centroid=e3.centroid)
    np.testing.assert_almost_equal(
            test_sigma/e1_sigma, 1.0, decimal=4,
            err_msg="shifted MomentRadius with explicit centroid is not accurate.")

    # Check the image version.
    size = 2000
    im = e1.drawImage(scale=0.1, nx=size, ny=size)
    test_sigma = im.calculateMomentRadius()
    print('im.calculateMomentRadius() = ',test_sigma)
    print('ratio - 1 = ',test_sigma/e1_sigma-1)
    np.testing.assert_almost_equal(
            test_sigma/e1_sigma, 1.0, decimal=4,
            err_msg="image.calculateMomentRadius is not accurate.")
    with assert_raises(galsim.GalSimValueError):
        im.calculateMomentRadius(rtype='invalid')

    # Check that a non-square image works correctly.  Also, not centered anywhere in particular.
    bounds = galsim.BoundsI(-1234, -1234+size*2, 8234, 8234+size)
    offset = galsim.PositionD(29,1)
    im = e1.drawImage(scale=0.1, bounds=bounds, offset=offset)
    test_hlr = im.calculateMomentRadius(center=im.true_center+offset)
    print('im.calculateMomentRadius() = ',test_sigma)
    print('ratio - 1 = ',test_sigma/e1_sigma-1)
    np.testing.assert_almost_equal(
            test_sigma/e1_sigma, 1.0, decimal=4,
            err_msg="non-square image.calculateMomentRadius is not accurate.")


@timer
def test_fwhm():
    """Test the calculateFWHM method.
    """
    # Compare the calculation for a simple Gaussian.
    g1 = galsim.Gaussian(sigma=5, flux=1.7)

    print('g1 native fwhm = ',g1.fwhm)
    print('g1.calculateFWHM = ',g1.calculateFWHM())
    # These should be exactly equal.
    np.testing.assert_equal(g1.fwhm, g1.calculateFWHM(),
                            err_msg="Gaussian.calculateFWHM() returned wrong value.")

    # Check for a convolution of two Gaussians.  Should be equivalent, but now will need to
    # do the calculation.
    g2 = galsim.Convolve(galsim.Gaussian(sigma=3, flux=1.3), galsim.Gaussian(sigma=4, flux=23))
    test_fwhm = g2.calculateFWHM()
    print('g2.calculateFWHM = ',test_fwhm)
    print('ratio - 1 = ',test_fwhm/g1.fwhm-1)
    np.testing.assert_almost_equal(test_fwhm/g1.fwhm, 1.0, decimal=3,
                                   err_msg="Gaussian.calculateFWHM() is not accurate.")

    # The default scale already accurate to around 3 dp.  Using scale = 0.1 is accurate to 8 dp.
    test_fwhm = g2.calculateFWHM(scale=0.1)
    print('g2.calculateFWHM(scale=0.1) = ',test_fwhm)
    print('ratio - 1 = ',test_fwhm/g1.fwhm-1)
    np.testing.assert_almost_equal(test_fwhm/g1.fwhm, 1.0, decimal=8,
                                   err_msg="Gaussian.calculateFWHM(scale=0.1) is not accurate.")

    # Finally, we don't expect this to be accurate, but make sure the code can handle having
    # only the central pixel higher than half-maximum.
    test_fwhm = g2.calculateFWHM(scale=20)
    print('g2.calculateFWHM(scale=20) = ',test_fwhm)
    print('ratio - 1 = ',test_fwhm/g1.fwhm-1)
    np.testing.assert_almost_equal(test_fwhm/g1.fwhm/10, 0.1, decimal=1,
                                   err_msg="Gaussian.calculateFWHM(scale=20) is not accurate.")

    # Next, use an Exponential profile
    e1 = galsim.Exponential(scale_radius=5, flux=1.7)

    # The true fwhm for this is analytic, but not an attribute.
    e1_fwhm = 2. * np.log(2.0) * e1.scale_radius
    print('true e1 fwhm = ',e1_fwhm)

    # Test with the default scale and size.
    test_fwhm = e1.calculateFWHM()
    print('e1.calculateFWHM = ',test_fwhm)
    print('ratio - 1 = ',test_fwhm/e1_fwhm-1)
    np.testing.assert_almost_equal(test_fwhm/e1_fwhm, 1.0, decimal=3,
                                   err_msg="Exponential.calculateFWHM() is not accurate.")

    # The default scale already accurate to around 3 dp.  Using scale = 0.1 is accurate to 7 dp.
    # We can also decrease the size, which should still be accurate, but maybe a little faster.
    # Go a bit more that fwhm in units of the pixels.
    size = int(1.2 * e1_fwhm / 0.1)
    test_fwhm = e1.calculateFWHM(scale=0.1, size=size)
    print('e1.calculateFWHM(scale=0.1) = ',test_fwhm)
    print('ratio - 1 = ',test_fwhm/e1_fwhm-1)
    np.testing.assert_almost_equal(test_fwhm/e1_fwhm, 1.0, decimal=7,
                                   err_msg="Exponential.calculateFWHM(scale=0.1) is not accurate.")

    # Check that it works if the centroid is not at the origin
    e3 = e1.shift(2,3)
    test_fwhm = e3.calculateFWHM(scale=0.1)
    print('e3.calculateFWHM(scale=0.1) = ',test_fwhm)
    print('ratio - 1 = ',test_fwhm/e1_fwhm-1)
    np.testing.assert_almost_equal(test_fwhm/e1_fwhm, 1.0, decimal=6,
                                   err_msg="shifted Exponential FWHM is not accurate.")

    # Can set a centroid manually.  This should be equivalent to the default.
    print('e3.centroid = ',e3.centroid)
    test_fwhm = e3.calculateFWHM(scale=0.1, centroid=e3.centroid)
    np.testing.assert_almost_equal(test_fwhm/e1_fwhm, 1.0, decimal=6,
                                   err_msg="shifted FWHM with explicit centroid is not accurate.")

    # Check the image version.
    im = e1.drawImage(scale=0.1, method='sb')
    test_fwhm = im.calculateFWHM(Imax=e1.xValue(0,0))
    print('im.calculateFWHM() = ',test_fwhm)
    print('ratio - 1 = ',test_fwhm/e1_fwhm-1)
    np.testing.assert_almost_equal(test_fwhm/e1_fwhm, 1.0, decimal=6,
                                   err_msg="image.calculateFWHM is not accurate.")

    # Check that a non-square image works correctly.  Also, not centered anywhere in particular.
    bounds = galsim.BoundsI(-1234, -1234+size*2, 8234, 8234+size)
    offset = galsim.PositionD(29,1)
    im = e1.drawImage(scale=0.1, bounds=bounds, offset=offset, method='sb')
    test_fwhm = im.calculateFWHM(Imax=e1.xValue(0,0), center=im.true_center+offset)
    print('im.calculateFWHM() = ',test_fwhm)
    print('ratio - 1 = ',test_fwhm/e1_fwhm-1)
    np.testing.assert_almost_equal(test_fwhm/e1_fwhm, 1.0, decimal=6,
                                   err_msg="non-square image.calculateFWHM is not accurate.")


if __name__ == "__main__":
    runtests(__file__)
