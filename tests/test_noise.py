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

testseed = 1000

precision = 10
# decimal point at which agreement is required for all double precision tests

precisionD = precision
precisionF = 5  # precision=10 does not make sense at single precision
precisionS = 1  # "precision" also a silly concept for ints, but allows all 4 tests to run in one go
precisionI = 1

@timer
def test_deviate_noise():
    """Test basic functionality of the DeviateNoise class
    """
    u = galsim.UniformDeviate(testseed)
    uResult = np.empty((10,10))
    u.generate(uResult)

    noise = galsim.DeviateNoise(galsim.UniformDeviate(testseed))

    # Test filling an image with random values
    testimage = galsim.ImageD(10,10)
    testimage.addNoise(noise)
    np.testing.assert_array_almost_equal(
            testimage.array, uResult, precision,
            err_msg='Wrong uniform random number sequence generated when applied to image.')

    # Test filling a single-precision image
    noise.rng.seed(testseed)
    testimage = galsim.ImageF(10,10)
    testimage.addNoise(noise)
    np.testing.assert_array_almost_equal(
            testimage.array, uResult, precisionF,
            err_msg='Wrong uniform random number sequence generated when applied to ImageF.')

    # Test filling an image with Fortran ordering
    noise.rng.seed(testseed)
    testimage = galsim.ImageD(np.zeros((10,10)).T)
    testimage.addNoise(noise)
    np.testing.assert_array_almost_equal(
            testimage.array, uResult, precision,
            err_msg="Wrong uniform randoms generated for Fortran-ordered Image")

    # Check picklability
    check_pickle(noise, drawNoise)
    check_pickle(noise)

    # Check copy, eq and ne
    noise2 = galsim.DeviateNoise(noise.rng.duplicate())  # Separate but equivalent rng chain.
    noise3 = noise.copy()  # Always has exactly the same rng as noise.
    noise4 = noise.copy(rng=galsim.BaseDeviate(11))  # Always has a different rng than noise
    assert noise == noise2
    assert noise == noise3
    assert noise != noise4
    assert noise.rng() == noise2.rng()
    assert noise == noise2  # Still equal because both chains incremented one place.
    assert noise == noise3  # Still equal because noise 3's rng is always equal to noise's rng.
    noise.rng()
    assert noise2 != noise3 # This is no longer equal, since only noise.rng is incremented.
    assert noise == noise3

    assert_raises(TypeError, galsim.DeviateNoise, 53)
    assert_raises(NotImplementedError, galsim.BaseNoise().getVariance)
    assert_raises(NotImplementedError, galsim.BaseNoise().withVariance, 23)
    assert_raises(NotImplementedError, galsim.BaseNoise().withScaledVariance, 23)
    assert_raises(TypeError, noise.applyTo, 23)
    assert_raises(NotImplementedError, galsim.BaseNoise().applyTo, testimage)
    assert_raises(galsim.GalSimError, noise.getVariance)
    assert_raises(galsim.GalSimError, noise.withVariance, 23)
    assert_raises(galsim.GalSimError, noise.withScaledVariance, 23)


@timer
def test_gaussian_noise():
    """Test Gaussian random number generator
    """
    gSigma = 17.23
    g = galsim.GaussianDeviate(testseed, sigma=gSigma)
    gResult = np.empty((10,10))
    g.generate(gResult)
    noise = galsim.DeviateNoise(g)

    # Test filling an image
    testimage = galsim.ImageD(10,10)
    noise.rng.seed(testseed)
    testimage.addNoise(noise)
    np.testing.assert_array_almost_equal(
            testimage.array, gResult, precision,
            err_msg='Wrong Gaussian random number sequence generated when applied to image.')

    # Test filling a single-precision image
    noise.rng.seed(testseed)
    testimage = galsim.ImageF(10,10)
    testimage.addNoise(noise)
    np.testing.assert_array_almost_equal(
            testimage.array, gResult, precisionF,
            err_msg='Wrong Gaussian random number sequence generated when applied to ImageF.')

    # GaussianNoise is equivalent, but no mean allowed.
    gn = galsim.GaussianNoise(galsim.BaseDeviate(testseed), sigma=gSigma)
    testimage = galsim.ImageD(10,10)
    testimage.addNoise(gn)
    np.testing.assert_array_almost_equal(
            testimage.array, gResult, precision,
            err_msg="GaussianNoise applied to Images does not reproduce expected sequence")

    # Test filling an image with Fortran ordering
    gn.rng.seed(testseed)
    testimage = galsim.ImageD(np.zeros((10,10)).T)
    testimage.addNoise(gn)
    np.testing.assert_array_almost_equal(
            testimage.array, gResult, precision,
            err_msg="Wrong Gaussian noise generated for Fortran-ordered Image")

    # Check GaussianNoise variance:
    np.testing.assert_almost_equal(
            gn.getVariance(), gSigma**2, precision,
            err_msg="GaussianNoise getVariance returns wrong variance")
    np.testing.assert_almost_equal(
            gn.sigma, gSigma, precision,
            err_msg="GaussianNoise sigma returns wrong value")

    # Check that the noise model really does produce this variance.
    big_im = galsim.Image(2048,2048,dtype=float)
    gn.rng.seed(testseed)
    big_im.addNoise(gn)
    var = np.var(big_im.array)
    print('variance = ',var)
    print('getVar = ',gn.getVariance())
    np.testing.assert_almost_equal(
            var, gn.getVariance(), 1,
            err_msg='Realized variance for GaussianNoise did not match getVariance()')

    # Check that GaussianNoise adds to the image, not overwrites the image.
    gal = galsim.Exponential(half_light_radius=2.3, flux=1.e4)
    gal.drawImage(image=big_im)
    gn.rng.seed(testseed)
    big_im.addNoise(gn)
    gal.withFlux(-1.e4).drawImage(image=big_im, add_to_image=True)
    var = np.var(big_im.array)
    np.testing.assert_almost_equal(
            var, gn.getVariance(), 1,
            err_msg='GaussianNoise wrong when already an object drawn on the image')

    # Check that DeviateNoise adds to the image, not overwrites the image.
    gal.drawImage(image=big_im)
    gn.rng.seed(testseed)
    big_im.addNoise(gn)
    gal.withFlux(-1.e4).drawImage(image=big_im, add_to_image=True)
    var = np.var(big_im.array)
    np.testing.assert_almost_equal(
            var, gn.getVariance(), 1,
            err_msg='DeviateNoise wrong when already an object drawn on the image')

    # Check withVariance
    gn = gn.withVariance(9.)
    np.testing.assert_almost_equal(
            gn.getVariance(), 9, precision,
            err_msg="GaussianNoise withVariance results in wrong variance")
    np.testing.assert_almost_equal(
            gn.sigma, 3., precision,
            err_msg="GaussianNoise withVariance results in wrong sigma")

    # Check withScaledVariance
    gn = gn.withScaledVariance(4.)
    np.testing.assert_almost_equal(
            gn.getVariance(), 36., precision,
            err_msg="GaussianNoise withScaledVariance results in wrong variance")
    np.testing.assert_almost_equal(
            gn.sigma, 6., precision,
            err_msg="GaussianNoise withScaledVariance results in wrong sigma")

    # Check arithmetic
    gn = gn.withVariance(0.5)
    gn2 = gn * 3
    np.testing.assert_almost_equal(
            gn2.getVariance(), 1.5, precision,
            err_msg="GaussianNoise gn*3 results in wrong variance")
    np.testing.assert_almost_equal(
            gn.getVariance(), 0.5, precision,
            err_msg="GaussianNoise gn*3 results in wrong variance for original gn")
    gn2 = 5 * gn
    np.testing.assert_almost_equal(
            gn2.getVariance(), 2.5, precision,
            err_msg="GaussianNoise 5*gn results in wrong variance")
    np.testing.assert_almost_equal(
            gn.getVariance(), 0.5, precision,
            err_msg="GaussianNoise 5*gn results in wrong variance for original gn")
    gn2 = gn/2
    np.testing.assert_almost_equal(
            gn2.getVariance(), 0.25, precision,
            err_msg="GaussianNoise gn/2 results in wrong variance")
    np.testing.assert_almost_equal(
            gn.getVariance(), 0.5, precision,
            err_msg="GaussianNoise 5*gn results in wrong variance for original gn")
    gn *= 3
    np.testing.assert_almost_equal(
            gn.getVariance(), 1.5, precision,
            err_msg="GaussianNoise gn*=3 results in wrong variance")
    gn /= 2
    np.testing.assert_almost_equal(
            gn.getVariance(), 0.75, precision,
            err_msg="GaussianNoise gn/=2 results in wrong variance")

    # Check starting with GaussianNoise()
    gn2 = galsim.GaussianNoise()
    gn2 = gn2.withVariance(9.)
    np.testing.assert_almost_equal(
            gn2.getVariance(), 9, precision,
            err_msg="GaussianNoise().withVariance results in wrong variance")
    np.testing.assert_almost_equal(
            gn2.sigma, 3., precision,
            err_msg="GaussianNoise().withVariance results in wrong sigma")

    gn2 = galsim.GaussianNoise()
    gn2 = gn2.withScaledVariance(4.)
    np.testing.assert_almost_equal(
            gn2.getVariance(), 4., precision,
            err_msg="GaussianNoise().withScaledVariance results in wrong variance")
    np.testing.assert_almost_equal(
            gn2.sigma, 2., precision,
            err_msg="GaussianNoise().withScaledVariance results in wrong sigma")

    # Check picklability
    check_pickle(gn, lambda x: (x.rng.serialize(), x.sigma))
    check_pickle(gn, drawNoise)
    check_pickle(gn)

    # Check copy, eq and ne
    gn = gn.withVariance(gSigma**2)
    gn2 = galsim.GaussianNoise(gn.rng.duplicate(), gSigma)
    gn3 = gn.copy()
    gn4 = gn.copy(rng=galsim.BaseDeviate(11))
    gn5 = galsim.GaussianNoise(gn.rng, 2.*gSigma)
    assert gn == gn2
    assert gn == gn3
    assert gn != gn4
    assert gn != gn5
    assert gn.rng.raw() == gn2.rng.raw()
    assert gn == gn2
    assert gn == gn3
    gn.rng.raw()
    assert gn != gn2
    assert gn == gn3


@timer
def test_variable_gaussian_noise():
    """Test VariableGaussian random number generator
    """
    # Make a checkerboard image with two values for the variance
    gSigma1 = 17.23
    gSigma2 = 28.55
    var_image = galsim.ImageD(galsim.BoundsI(0,9,0,9))
    coords = np.ogrid[0:10, 0:10]
    var_image.array[ (coords[0] + coords[1]) % 2 == 1 ] = gSigma1**2
    var_image.array[ (coords[0] + coords[1]) % 2 == 0 ] = gSigma2**2
    print('var_image.array = ',var_image.array)

    g = galsim.GaussianDeviate(testseed, sigma=1.)
    vgResult = np.empty((10,10))
    g.generate(vgResult)
    vgResult *= np.sqrt(var_image.array)

    # Test filling an image
    vgn = galsim.VariableGaussianNoise(galsim.BaseDeviate(testseed), var_image)
    testimage = galsim.ImageD(10,10)
    testimage.addNoise(vgn)
    np.testing.assert_array_almost_equal(
            testimage.array, vgResult, precision,
            err_msg="VariableGaussianNoise applied to Images does not reproduce expected sequence")

    # Test filling an image with Fortran ordering
    vgn.rng.seed(testseed)
    testimage = galsim.ImageD(np.zeros((10,10)).T)
    testimage.addNoise(vgn)
    np.testing.assert_array_almost_equal(
            testimage.array, vgResult, precision,
            err_msg="Wrong VariableGaussian noise generated for Fortran-ordered Image")

    # Check var_image property
    np.testing.assert_almost_equal(
            vgn.var_image.array, var_image.array, precision,
            err_msg="VariableGaussianNoise var_image returns wrong var_image")

    # Check that the noise model really does produce this variance.
    big_var_image = galsim.ImageD(galsim.BoundsI(0,2047,0,2047))
    big_coords = np.ogrid[0:2048, 0:2048]
    mask1 = (big_coords[0] + big_coords[1]) % 2 == 0
    mask2 = (big_coords[0] + big_coords[1]) % 2 == 1
    big_var_image.array[mask1] = gSigma1**2
    big_var_image.array[mask2] = gSigma2**2
    big_vgn = galsim.VariableGaussianNoise(galsim.BaseDeviate(testseed), big_var_image)

    big_im = galsim.Image(2048,2048,dtype=float)
    big_im.addNoise(big_vgn)
    var = np.var(big_im.array)
    print('variance = ',var)
    print('getVar = ',big_vgn.var_image.array.mean())
    np.testing.assert_almost_equal(
            var, big_vgn.var_image.array.mean(), 1,
            err_msg='Realized variance for VariableGaussianNoise did not match var_image')

    # Check realized variance in each mask
    print('rms1 = ',np.std(big_im.array[mask1]))
    print('rms2 = ',np.std(big_im.array[mask2]))
    np.testing.assert_almost_equal(np.std(big_im.array[mask1]), gSigma1, decimal=1)
    np.testing.assert_almost_equal(np.std(big_im.array[mask2]), gSigma2, decimal=1)

    # Check that VariableGaussianNoise adds to the image, not overwrites the image.
    gal = galsim.Exponential(half_light_radius=2.3, flux=1.e4)
    gal.drawImage(image=big_im)
    big_vgn.rng.seed(testseed)
    big_im.addNoise(big_vgn)
    gal.withFlux(-1.e4).drawImage(image=big_im, add_to_image=True)
    var = np.var(big_im.array)
    np.testing.assert_almost_equal(
            var, big_vgn.var_image.array.mean(), 1,
            err_msg='VariableGaussianNoise wrong when already an object drawn on the image')

    # Check picklability
    check_pickle(vgn, lambda x: (x.rng.serialize(), x.var_image))
    check_pickle(vgn, drawNoise)
    check_pickle(vgn)

    # Check copy, eq and ne
    vgn2 = galsim.VariableGaussianNoise(vgn.rng.duplicate(), var_image)
    vgn3 = vgn.copy()
    vgn4 = vgn.copy(rng=galsim.BaseDeviate(11))
    vgn5 = galsim.VariableGaussianNoise(vgn.rng, 2.*var_image)
    assert vgn == vgn2
    assert vgn == vgn3
    assert vgn != vgn4
    assert vgn != vgn5
    assert vgn.rng.raw() == vgn2.rng.raw()
    assert vgn == vgn2
    assert vgn == vgn3
    vgn.rng.raw()
    assert vgn != vgn2
    assert vgn == vgn3

    assert_raises(TypeError, vgn.applyTo, 23)
    assert_raises(ValueError, vgn.applyTo, galsim.ImageF(3,3))
    assert_raises(galsim.GalSimError, vgn.getVariance)
    assert_raises(galsim.GalSimError, vgn.withVariance, 23)
    assert_raises(galsim.GalSimError, vgn.withScaledVariance, 23)


@timer
def test_poisson_noise():
    """Test Poisson random number generator
    """
    pMean = 17
    p = galsim.PoissonDeviate(testseed, mean=pMean)
    pResult = np.empty((10,10))
    p.generate(pResult)
    noise = galsim.DeviateNoise(p)

    # Test filling an image
    noise.rng.seed(testseed)
    testimage = galsim.ImageI(10, 10)
    testimage.addNoise(galsim.DeviateNoise(p))
    np.testing.assert_array_equal(
            testimage.array, pResult,
            err_msg='Wrong poisson random number sequence generated when applied to image.')

    # The PoissonNoise version also subtracts off the mean value
    pn = galsim.PoissonNoise(galsim.BaseDeviate(testseed), sky_level=pMean)
    testimage.fill(0)
    testimage.addNoise(pn)
    np.testing.assert_array_equal(
            testimage.array, pResult-pMean,
            err_msg='Wrong poisson random number sequence generated using PoissonNoise')

    # Test filling a single-precision image
    pn.rng.seed(testseed)
    testimage = galsim.ImageF(10,10)
    testimage.addNoise(pn)
    np.testing.assert_array_almost_equal(
            testimage.array, pResult-pMean, precisionF,
            err_msg='Wrong Poisson random number sequence generated when applied to ImageF.')

    # Test filling an image with Fortran ordering
    pn.rng.seed(testseed)
    testimage = galsim.ImageD(10,10)
    testimage.addNoise(pn)
    np.testing.assert_array_almost_equal(
            testimage.array, pResult-pMean,
            err_msg="Wrong Poisson noise generated for Fortran-ordered Image")

    # Check PoissonNoise variance:
    np.testing.assert_almost_equal(
            pn.getVariance(), pMean, precision,
            err_msg="PoissonNoise getVariance returns wrong variance")
    np.testing.assert_almost_equal(
            pn.sky_level, pMean, precision,
            err_msg="PoissonNoise sky_level returns wrong value")

    # Check that the noise model really does produce this variance.
    big_im = galsim.Image(2048,2048,dtype=float)
    big_im.addNoise(pn)
    var = np.var(big_im.array)
    print('variance = ',var)
    print('getVar = ',pn.getVariance())
    np.testing.assert_almost_equal(
            var, pn.getVariance(), 1,
            err_msg='Realized variance for PoissonNoise did not match getVariance()')

    # Check that PoissonNoise adds to the image, not overwrites the image.
    gal = galsim.Exponential(half_light_radius=2.3, flux=0.3)
    # Note: in this case, flux/size^2 needs to be << sky_level or it will mess up the statistics.
    gal.drawImage(image=big_im)
    big_im.addNoise(pn)
    gal.withFlux(-0.3).drawImage(image=big_im, add_to_image=True)
    var = np.var(big_im.array)
    np.testing.assert_almost_equal(
            var, pn.getVariance(), 1,
            err_msg='PoissonNoise wrong when already an object drawn on the image')

    # Check withVariance
    pn = pn.withVariance(9.)
    np.testing.assert_almost_equal(
            pn.getVariance(), 9., precision,
            err_msg="PoissonNoise withVariance results in wrong variance")
    np.testing.assert_almost_equal(
            pn.sky_level, 9., precision,
            err_msg="PoissonNoise withVariance results in wrong sky_level")

    # Check withScaledVariance
    pn = pn.withScaledVariance(4.)
    np.testing.assert_almost_equal(
            pn.getVariance(), 36, precision,
            err_msg="PoissonNoise withScaledVariance results in wrong variance")
    np.testing.assert_almost_equal(
            pn.sky_level, 36., precision,
            err_msg="PoissonNoise withScaledVariance results in wrong sky_level")

    # Check arithmetic
    pn = pn.withVariance(0.5)
    pn2 = pn * 3
    np.testing.assert_almost_equal(
            pn2.getVariance(), 1.5, precision,
            err_msg="PoissonNoise pn*3 results in wrong variance")
    np.testing.assert_almost_equal(
            pn.getVariance(), 0.5, precision,
            err_msg="PoissonNoise pn*3 results in wrong variance for original pn")
    pn2 = 5 * pn
    np.testing.assert_almost_equal(
            pn2.getVariance(), 2.5, precision,
            err_msg="PoissonNoise 5*pn results in wrong variance")
    np.testing.assert_almost_equal(
            pn.getVariance(), 0.5, precision,
            err_msg="PoissonNoise 5*pn results in wrong variance for original pn")
    pn2 = pn/2
    np.testing.assert_almost_equal(
            pn2.getVariance(), 0.25, precision,
            err_msg="PoissonNoise pn/2 results in wrong variance")
    np.testing.assert_almost_equal(
            pn.getVariance(), 0.5, precision,
            err_msg="PoissonNoise 5*pn results in wrong variance for original pn")
    pn *= 3
    np.testing.assert_almost_equal(
            pn.getVariance(), 1.5, precision,
            err_msg="PoissonNoise pn*=3 results in wrong variance")
    pn /= 2
    np.testing.assert_almost_equal(
            pn.getVariance(), 0.75, precision,
            err_msg="PoissonNoise pn/=2 results in wrong variance")

    # Check starting with PoissonNoise()
    pn = galsim.PoissonNoise()
    pn = pn.withVariance(9.)
    np.testing.assert_almost_equal(
            pn.getVariance(), 9., precision,
            err_msg="PoissonNoise().withVariance results in wrong variance")
    np.testing.assert_almost_equal(
            pn.sky_level, 9., precision,
            err_msg="PoissonNoise().withVariance results in wrong sky_level")
    pn = pn.withScaledVariance(4.)
    np.testing.assert_almost_equal(
            pn.getVariance(), 36, precision,
            err_msg="PoissonNoise().withScaledVariance results in wrong variance")
    np.testing.assert_almost_equal(
            pn.sky_level, 36., precision,
            err_msg="PoissonNoise().withScaledVariance results in wrong sky_level")

    # Check picklability
    check_pickle(pn, lambda x: (x.rng.serialize(), x.sky_level))
    check_pickle(pn, drawNoise)
    check_pickle(pn)

    # Check copy, eq and ne
    pn = pn.withVariance(pMean)
    pn2 = galsim.PoissonNoise(pn.rng.duplicate(), pMean)
    pn3 = pn.copy()
    pn4 = pn.copy(rng=galsim.BaseDeviate(11))
    pn5 = galsim.PoissonNoise(pn.rng, 2*pMean)
    assert pn == pn2
    assert pn == pn3
    assert pn != pn4
    assert pn != pn5
    assert pn.rng.raw() == pn2.rng.raw()
    assert pn == pn2
    assert pn == pn3
    pn.rng.raw()
    assert pn != pn2
    assert pn == pn3


@timer
def test_ccdnoise():
    """Test CCD Noise generator
    """
    # Start with some regression tests where we have known values that we expect to generate:

    types = (np.int16, np.int32, np.float32, np.float64)
    typestrings = ("S", "I", "F", "D")

    testseed = 1000
    gain = 3.
    read_noise = 5.
    sky = 50

    # Tabulated results for the above settings and testseed value.
    cResultS = np.array([[44, 47], [50, 49]], dtype=np.int16)
    cResultI = np.array([[44, 47], [50, 49]], dtype=np.int32)
    cResultF = np.array([[44.45332718, 47.79725266], [50.67744064, 49.58272934]], dtype=np.float32)
    cResultD = np.array([[44.453328440057618, 47.797254142519577],
                        [50.677442088335162, 49.582730949808081]],dtype=np.float64)

    for i in range(4):
        prec = eval("precision"+typestrings[i])
        cResult = eval("cResult"+typestrings[i])

        rng = galsim.BaseDeviate(testseed)
        ccdnoise = galsim.CCDNoise(rng, gain=gain, read_noise=read_noise)
        testImage = galsim.Image((np.zeros((2, 2))+sky).astype(types[i]))
        ccdnoise.applyTo(testImage)
        np.testing.assert_array_almost_equal(
                testImage.array, cResult, prec,
                err_msg="Wrong CCD noise random sequence generated for Image"+typestrings[i]+".")

        # Check that reseeding the rng reseeds the internal deviate in CCDNoise
        rng.seed(testseed)
        testImage.fill(sky)
        ccdnoise.applyTo(testImage)
        np.testing.assert_array_almost_equal(
                testImage.array, cResult, prec,
                err_msg="Wrong CCD noise random sequence generated for Image"+typestrings[i]+
                " after seed")

        # Check using addNoise
        rng.seed(testseed)
        testImage.fill(sky)
        testImage.addNoise(ccdnoise)
        np.testing.assert_array_almost_equal(
                testImage.array, cResult, prec,
                err_msg="Wrong CCD noise random sequence generated for Image"+typestrings[i]+
                " using addNoise")

        # Test filling an image with Fortran ordering
        rng.seed(testseed)
        testImageF = galsim.Image(np.zeros((2, 2)).T, dtype=types[i])
        testImageF.fill(sky)
        testImageF.addNoise(ccdnoise)
        np.testing.assert_array_almost_equal(
                testImageF.array, cResult, prec,
                err_msg="Wrong CCD noise generated for Fortran-ordered Image"+typestrings[i])

        # Now include sky_level in ccdnoise
        rng.seed(testseed)
        ccdnoise = galsim.CCDNoise(rng, sky_level=sky, gain=gain, read_noise=read_noise)
        testImage.fill(0)
        ccdnoise.applyTo(testImage)
        np.testing.assert_array_almost_equal(
                testImage.array, cResult-sky, prec,
                err_msg="Wrong CCD noise random sequence generated for Image"+typestrings[i]+
                " with sky_level included in noise")

        rng.seed(testseed)
        testImage.fill(0)
        testImage.addNoise(ccdnoise)
        np.testing.assert_array_almost_equal(
                testImage.array, cResult-sky, prec,
                err_msg="Wrong CCD noise random sequence generated for Image"+typestrings[i]+
                " using addNoise with sky_level included in noise")

    # Check CCDNoise variance:
    var1 = sky/gain + (read_noise/gain)**2
    np.testing.assert_almost_equal(
            ccdnoise.getVariance(), var1, precision,
            err_msg="CCDNoise getVariance returns wrong variance")
    np.testing.assert_almost_equal(
            ccdnoise.sky_level, sky, precision,
            err_msg="CCDNoise sky_level returns wrong value")
    np.testing.assert_almost_equal(
            ccdnoise.gain, gain, precision,
            err_msg="CCDNoise gain returns wrong value")
    np.testing.assert_almost_equal(
            ccdnoise.read_noise, read_noise, precision,
            err_msg="CCDNoise read_noise returns wrong value")

    # Check that the noise model really does produce this variance.
    # NB. If default float32 is used here, older versions of numpy will compute the variance
    # in single precision, and with 2048^2 values, the final answer comes out significantly
    # wrong (19.33 instead of 19.42, which gets compared to the nominal value of 19.44).
    big_im = galsim.Image(2048,2048,dtype=float)
    big_im.addNoise(ccdnoise)
    var = np.var(big_im.array)
    print('variance = ',var)
    print('getVar = ',ccdnoise.getVariance())
    np.testing.assert_almost_equal(
            var, ccdnoise.getVariance(), 1,
            err_msg='Realized variance for CCDNoise did not match getVariance()')

    # Check that CCDNoise adds to the image, not overwrites the image.
    gal = galsim.Exponential(half_light_radius=2.3, flux=0.3)
    # Note: again, flux/size^2 needs to be << sky_level or it will mess up the statistics.
    gal.drawImage(image=big_im)
    big_im.addNoise(ccdnoise)
    gal.withFlux(-0.3).drawImage(image=big_im, add_to_image=True)
    var = np.var(big_im.array)
    np.testing.assert_almost_equal(
            var, ccdnoise.getVariance(), 1,
            err_msg='CCDNoise wrong when already an object drawn on the image')

    # Check using a non-integer sky level which does some slightly different calculations.
    rng.seed(testseed)
    big_im_int = galsim.Image(2048,2048,dtype=int)
    ccdnoise = galsim.CCDNoise(rng, sky_level=34.42, gain=1.6, read_noise=11.2)
    big_im_int.fill(0)
    big_im_int.addNoise(ccdnoise)
    var = np.var(big_im_int.array)
    np.testing.assert_almost_equal(var/ccdnoise.getVariance(), 1., decimal=2,
                                   err_msg='CCDNoise wrong when sky_level is not an integer')

    # Using gain=0 means the read_noise is in ADU, not e-
    rng.seed(testseed)
    ccdnoise = galsim.CCDNoise(rng, gain=0., read_noise=read_noise)
    var2 = read_noise**2
    np.testing.assert_almost_equal(
            ccdnoise.getVariance(), var2, precision,
            err_msg="CCDNoise getVariance returns wrong variance with gain=0")
    np.testing.assert_almost_equal(
            ccdnoise.sky_level, 0., precision,
            err_msg="CCDNoise sky_level returns wrong value with gain=0")
    np.testing.assert_almost_equal(
            ccdnoise.gain, 0., precision,
            err_msg="CCDNoise gain returns wrong value with gain=0")
    np.testing.assert_almost_equal(
            ccdnoise.read_noise, read_noise, precision,
            err_msg="CCDNoise read_noise returns wrong value with gain=0")
    big_im.fill(0)
    big_im.addNoise(ccdnoise)
    var = np.var(big_im.array)
    np.testing.assert_almost_equal(var, ccdnoise.getVariance(), 1,
                                   err_msg='CCDNoise wrong when gain=0')

    # Check withVariance
    ccdnoise = galsim.CCDNoise(rng, sky_level=sky, gain=gain, read_noise=read_noise)
    ccdnoise = ccdnoise.withVariance(9.)
    np.testing.assert_almost_equal(
            ccdnoise.getVariance(), 9., precision,
            err_msg="CCDNoise withVariance results in wrong variance")
    np.testing.assert_almost_equal(
            ccdnoise.sky_level, (9./var1)*sky, precision,
            err_msg="CCDNoise withVariance results in wrong sky_level")
    np.testing.assert_almost_equal(
            ccdnoise.gain, gain, precision,
            err_msg="CCDNoise withVariance results in wrong gain")
    np.testing.assert_almost_equal(
            ccdnoise.read_noise, np.sqrt(9./var1) * read_noise, precision,
            err_msg="CCDNoise withVariance results in wrong ReadNoise")

    # Check withScaledVariance
    ccdnoise = ccdnoise.withScaledVariance(4.)
    np.testing.assert_almost_equal(
            ccdnoise.getVariance(), 36., precision,
            err_msg="CCDNoise withVariance results in wrong variance")
    np.testing.assert_almost_equal(
            ccdnoise.sky_level, (36./var1)*sky, precision,
            err_msg="CCDNoise withVariance results in wrong sky_level")
    np.testing.assert_almost_equal(
            ccdnoise.gain, gain, precision,
            err_msg="CCDNoise withVariance results in wrong gain")
    np.testing.assert_almost_equal(
            ccdnoise.read_noise, np.sqrt(36./var1) * read_noise, precision,
            err_msg="CCDNoise withVariance results in wrong ReadNoise")

    # Check arithmetic
    ccdnoise = ccdnoise.withVariance(0.5)
    ccdnoise2 = ccdnoise * 3
    np.testing.assert_almost_equal(
            ccdnoise2.getVariance(), 1.5, precision,
            err_msg="CCDNoise ccdnoise*3 results in wrong variance")
    np.testing.assert_almost_equal(
            ccdnoise.getVariance(), 0.5, precision,
            err_msg="CCDNoise ccdnoise*3 results in wrong variance for original ccdnoise")
    ccdnoise2 = 5 * ccdnoise
    np.testing.assert_almost_equal(
            ccdnoise2.getVariance(), 2.5, precision,
            err_msg="CCDNoise 5*ccdnoise results in wrong variance")
    np.testing.assert_almost_equal(
            ccdnoise.getVariance(), 0.5, precision,
            err_msg="CCDNoise 5*ccdnoise results in wrong variance for original ccdnoise")
    ccdnoise2 = ccdnoise/2
    np.testing.assert_almost_equal(
            ccdnoise2.getVariance(), 0.25, precision,
            err_msg="CCDNoise ccdnoise/2 results in wrong variance")
    np.testing.assert_almost_equal(
            ccdnoise.getVariance(), 0.5, precision,
            err_msg="CCDNoise 5*ccdnoise results in wrong variance for original ccdnoise")
    ccdnoise *= 3
    np.testing.assert_almost_equal(
            ccdnoise.getVariance(), 1.5, precision,
            err_msg="CCDNoise ccdnoise*=3 results in wrong variance")
    ccdnoise /= 2
    np.testing.assert_almost_equal(
            ccdnoise.getVariance(), 0.75, precision,
            err_msg="CCDNoise ccdnoise/=2 results in wrong variance")

    # Check starting with CCDNoise()
    ccdnoise = galsim.CCDNoise()
    ccdnoise = ccdnoise.withVariance(9.)
    np.testing.assert_almost_equal(
            ccdnoise.getVariance(), 9., precision,
            err_msg="CCDNoise().withVariance results in wrong variance")
    np.testing.assert_almost_equal(
            ccdnoise.sky_level, 9., precision,
            err_msg="CCDNoise().withVariance results in wrong sky_level")
    np.testing.assert_almost_equal(
            ccdnoise.gain, 1., precision,
            err_msg="CCDNoise().withVariance results in wrong gain")
    np.testing.assert_almost_equal(
            ccdnoise.read_noise, 0., precision,
            err_msg="CCDNoise().withVariance results in wrong ReadNoise")
    ccdnoise = ccdnoise.withScaledVariance(4.)
    np.testing.assert_almost_equal(
            ccdnoise.getVariance(), 36., precision,
            err_msg="CCDNoise().withScaledVariance results in wrong variance")
    np.testing.assert_almost_equal(
            ccdnoise.sky_level, 36., precision,
            err_msg="CCDNoise().withScaledVariance results in wrong sky_level")
    np.testing.assert_almost_equal(
            ccdnoise.gain, 1., precision,
            err_msg="CCDNoise().withScaledVariance results in wrong gain")
    np.testing.assert_almost_equal(
            ccdnoise.read_noise, 0., precision,
            err_msg="CCDNoise().withScaledVariance results in wrong ReadNoise")

    # Check picklability
    check_pickle(ccdnoise, lambda x: (x.rng.serialize(), x.sky_level, x.gain, x.read_noise))
    check_pickle(ccdnoise, drawNoise)
    check_pickle(ccdnoise)

    # Check copy, eq and ne
    ccdnoise = galsim.CCDNoise(rng, sky, gain, read_noise)
    ccdnoise2 = galsim.CCDNoise(ccdnoise.rng.duplicate(), gain=gain, read_noise=read_noise,
                                sky_level=sky)
    ccdnoise3 = ccdnoise.copy()
    ccdnoise4 = ccdnoise.copy(rng=galsim.BaseDeviate(11))
    ccdnoise5 = galsim.CCDNoise(ccdnoise.rng, gain=2*gain, read_noise=read_noise, sky_level=sky)
    ccdnoise6 = galsim.CCDNoise(ccdnoise.rng, gain=gain, read_noise=2*read_noise, sky_level=sky)
    ccdnoise7 = galsim.CCDNoise(ccdnoise.rng, gain=gain, read_noise=read_noise, sky_level=2*sky)
    assert ccdnoise == ccdnoise2
    assert ccdnoise == ccdnoise3
    assert ccdnoise != ccdnoise4
    assert ccdnoise != ccdnoise5
    assert ccdnoise != ccdnoise6
    assert ccdnoise != ccdnoise7
    assert ccdnoise.rng.raw() == ccdnoise2.rng.raw()
    assert ccdnoise == ccdnoise2
    assert ccdnoise == ccdnoise3
    ccdnoise.rng.raw()
    assert ccdnoise != ccdnoise2
    assert ccdnoise == ccdnoise3


@timer
def test_addnoisesnr():
    """Test that addNoiseSNR is behaving sensibly.
    """
    # Rather than reproducing the S/N calculation in addNoiseSNR(), we'll just check for
    # self-consistency of the behavior with / without flux preservation.
    # Begin by making some object that we draw into an Image.
    gal_sigma = 3.7
    pix_scale = 0.6
    test_snr = 73.
    gauss = galsim.Gaussian(sigma=gal_sigma)
    im = gauss.drawImage(scale=pix_scale, dtype=np.float64)

    # Now make the noise object to use.
    # Use a default-constructed rng (i.e. rng=None) since we had initially had trouble
    # with that.  And use the duplicate feature to get a second copy of this rng.
    gn = galsim.GaussianNoise()
    rng2 = gn.rng.duplicate()

    # Try addNoiseSNR with preserve_flux=True, so the RNG needs a different variance.
    # Check what variance was added for this SNR, and that the RNG still has its original variance
    # after this call.
    var_out = im.addNoiseSNR(gn, test_snr, preserve_flux=True)
    assert gn.getVariance()==1.0
    max_val = im.array.max()

    # Now apply addNoiseSNR to another (clean) image with preserve_flux=False, so we use the noise
    # variance in the original RNG, i.e., 1.  Check that the returned variance is 1, and that the
    # value of the maximum pixel (presumably the peak of the galaxy light profile) is scaled as we
    # expect for this SNR.
    im2 = gauss.drawImage(scale=pix_scale, dtype=np.float64)
    gn2 = galsim.GaussianNoise(rng=rng2)
    var_out2 = im2.addNoiseSNR(gn2, test_snr, preserve_flux=False)
    assert var_out2==1.0
    expect_max_val2 = max_val*np.sqrt(var_out2/var_out)
    np.testing.assert_almost_equal(
            im2.array.max(), expect_max_val2, decimal=8,
            err_msg='addNoiseSNR with preserve_flux = True and False give inconsistent results')


if __name__ == "__main__":
    runtests(__file__)
