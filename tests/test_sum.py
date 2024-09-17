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

imgdir = os.path.join(".", "SBProfile_comparison_images") # Directory containing the reference
                                                          # images.

@timer
def test_add():
    """Test the addition of two rescaled Gaussian profiles against a known double Gaussian result.
    """
    savedImg = galsim.fits.read(os.path.join(imgdir, "double_gaussian.fits"))
    dx = 0.2
    myImg = galsim.ImageF(savedImg.bounds, scale=dx)
    myImg.setCenter(0,0)

    gauss1 = galsim.Gaussian(flux=0.75, sigma=1)
    gauss2 = galsim.Gaussian(flux=0.25, sigma=3)
    sum_gauss = galsim.Add(gauss1,gauss2)
    sum_gauss.drawImage(myImg,scale=dx, method="sb", use_true_center=False)
    printval(myImg, savedImg)
    np.testing.assert_array_almost_equal(
            myImg.array, savedImg.array, 5,
            err_msg="Using GSObject Add(gauss1,gauss2) disagrees with expected result")

    cen = galsim.PositionD(0,0)
    np.testing.assert_equal(sum_gauss.centroid, cen)
    np.testing.assert_almost_equal(sum_gauss.flux, gauss1.flux + gauss2.flux)
    np.testing.assert_almost_equal(sum_gauss.xValue(cen), sum_gauss.max_sb)

    # Check with default_params
    sum_gauss = galsim.Add(gauss1,gauss2,gsparams=default_params)
    sum_gauss.drawImage(myImg,scale=dx, method="sb", use_true_center=False)
    np.testing.assert_array_almost_equal(
            myImg.array, savedImg.array, 5,
            err_msg="Using GSObject Add(gauss1,gauss2) with default_params disagrees with "
            "expected result")
    sum_gauss = galsim.Add(gauss1,gauss2,gsparams=galsim.GSParams())
    sum_gauss.drawImage(myImg,scale=dx, method="sb", use_true_center=False)
    np.testing.assert_array_almost_equal(
            myImg.array, savedImg.array, 5,
            err_msg="Using GSObject Add(gauss1,gauss2) with GSParams() disagrees with "
            "expected result")

    # Other ways to do the sum:
    sum_gauss = gauss1 + gauss2
    sum_gauss.drawImage(myImg,scale=dx, method="sb", use_true_center=False)
    printval(myImg, savedImg)
    np.testing.assert_array_almost_equal(
            myImg.array, savedImg.array, 5,
            err_msg="Using GSObject gauss1 + gauss2 disagrees with expected result")
    sum_gauss = gauss1
    sum_gauss += gauss2
    sum_gauss.drawImage(myImg,scale=dx, method="sb", use_true_center=False)
    printval(myImg, savedImg)
    np.testing.assert_array_almost_equal(
            myImg.array, savedImg.array, 5,
            err_msg="Using GSObject sum = gauss1; sum += gauss2 disagrees with expected result")
    sum_gauss = galsim.Add([gauss1,gauss2])
    sum_gauss.drawImage(myImg,scale=dx, method="sb", use_true_center=False)
    printval(myImg, savedImg)
    np.testing.assert_array_almost_equal(
            myImg.array, savedImg.array, 5,
            err_msg="Using GSObject Add([gauss1,gauss2]) disagrees with expected result")
    gauss1 = galsim.Gaussian(flux=1, sigma=1)
    gauss2 = galsim.Gaussian(flux=1, sigma=3)
    sum_gauss = 0.75 * gauss1 + 0.25 * gauss2
    sum_gauss.drawImage(myImg,scale=dx, method="sb", use_true_center=False)
    printval(myImg, savedImg)
    np.testing.assert_array_almost_equal(
            myImg.array, savedImg.array, 5,
            err_msg="Using GSObject 0.75 * gauss1 + 0.25 * gauss2 disagrees with expected result")
    sum_gauss = 0.75 * gauss1
    sum_gauss += 0.25 * gauss2
    sum_gauss.drawImage(myImg,scale=dx, method="sb", use_true_center=False)
    printval(myImg, savedImg)
    np.testing.assert_array_almost_equal(
            myImg.array, savedImg.array, 5,
            err_msg="Using GSObject sum += 0.25 * gauss2 disagrees with expected result")

    check_basic(sum_gauss, "sum of 2 Gaussians")

    # Test photon shooting.
    do_shoot(sum_gauss,myImg,"sum of 2 Gaussians")

    # Test kvalues
    do_kvalue(sum_gauss,myImg,"sum of 2 Gaussians")

    # Check picklability
    check_pickle(sum_gauss, lambda x: x.drawImage(method='sb'))
    check_pickle(sum_gauss)

    # Sum of just one argument should be equivalent to that argument.
    single = galsim.Add(gauss1)
    gsobject_compare(single, gauss1)
    check_basic(single, "`sum' of 1 Gaussian")
    check_pickle(single)
    do_shoot(single, myImg, "Single Sum")

    single = galsim.Add([gauss1])
    gsobject_compare(single, gauss1)
    check_basic(single, "`sum' of 1 Gaussian")
    check_pickle(single)

    # Should raise an exception for invalid arguments
    assert_raises(TypeError, galsim.Add)
    assert_raises(TypeError, galsim.Add, myImg)
    assert_raises(TypeError, galsim.Add, [myImg])
    assert_raises(TypeError, galsim.Add, [gauss1, myImg])
    assert_raises(TypeError, galsim.Add, [gauss1, gauss1, myImg])
    assert_raises(TypeError, galsim.Add, [gauss1, gauss1], real_space=False)
    assert_raises(TypeError, galsim.Sum)
    assert_raises(TypeError, galsim.Sum, myImg)
    assert_raises(TypeError, galsim.Sum, [myImg])
    assert_raises(TypeError, galsim.Sum, [gauss1, myImg])
    assert_raises(TypeError, galsim.Sum, [gauss1, gauss1, myImg])
    assert_raises(TypeError, galsim.Sum, [gauss1, gauss1], real_space=False)


@timer
def test_sub_neg():
    """Test that a - b is the same as a + (-b)."""
    a = galsim.Gaussian(fwhm=1)
    b = galsim.Kolmogorov(fwhm=1)

    c = a - b
    d = a + (-b)

    assert c == d

    im1 = c.drawImage()
    im2 = d.drawImage(im1.copy())

    np.testing.assert_equal(im1.array, im2.array)

@timer
def test_add_flux_scaling():
    """Test flux scaling for Add.
    """
    # decimal point to go to for parameter value comparisons
    param_decimal = 12
    test_flux = 17.9
    test_sigma = 1.8
    test_scale = 1.9

    # init with Gaussian and Exponential only (should be ok given last tests)
    obj = galsim.Add([galsim.Gaussian(sigma=test_sigma, flux=test_flux * .5),
                      galsim.Exponential(scale_radius=test_scale, flux=test_flux * .5)])
    obj *= 2.
    np.testing.assert_almost_equal(
        obj.flux, test_flux * 2., decimal=param_decimal,
        err_msg="Flux param inconsistent after __imul__.")
    obj = galsim.Add([galsim.Gaussian(sigma=test_sigma, flux=test_flux * .5),
                      galsim.Exponential(scale_radius=test_scale, flux=test_flux * .5)])
    obj /= 2.
    np.testing.assert_almost_equal(
        obj.flux, test_flux / 2., decimal=param_decimal,
        err_msg="Flux param inconsistent after __idiv__.")
    obj = galsim.Add([galsim.Gaussian(sigma=test_sigma, flux=test_flux * .5),
                      galsim.Exponential(scale_radius=test_scale, flux=test_flux * .5)])
    obj2 = obj * 2.
    # First test that original obj is unharmed...
    np.testing.assert_almost_equal(
        obj.flux, test_flux, decimal=param_decimal,
        err_msg="Flux param inconsistent after __rmul__ (original).")
    # Then test new obj2 flux
    np.testing.assert_almost_equal(
        obj2.flux, test_flux * 2., decimal=param_decimal,
        err_msg="Flux param inconsistent after __rmul__ (result).")
    obj = galsim.Add([galsim.Gaussian(sigma=test_sigma, flux=test_flux * .5),
                      galsim.Exponential(scale_radius=test_scale, flux=test_flux * .5)])
    obj2 = 2. * obj
    # First test that original obj is unharmed...
    np.testing.assert_almost_equal(
        obj.flux, test_flux, decimal=param_decimal,
        err_msg="Flux param inconsistent after __mul__ (original).")
    # Then test new obj2 flux
    np.testing.assert_almost_equal(
        obj2.flux, test_flux * 2., decimal=param_decimal,
        err_msg="Flux param inconsistent after __mul__ (result).")
    obj = galsim.Add([galsim.Gaussian(sigma=test_sigma, flux=test_flux * .5),
                      galsim.Exponential(scale_radius=test_scale, flux=test_flux * .5)])
    obj2 = obj / 2.
    # First test that original obj is unharmed...
    np.testing.assert_almost_equal(
        obj.flux, test_flux, decimal=param_decimal,
        err_msg="Flux param inconsistent after __div__ (original).")
    # Then test new obj2 flux
    np.testing.assert_almost_equal(
        obj2.flux, test_flux / 2., decimal=param_decimal,
        err_msg="Flux param inconsistent after __div__ (result).")


@timer
def test_ne():
    """ Check that inequality works as expected."""
    gsp = galsim.GSParams(maxk_threshold=1.1e-3, folding_threshold=5.1e-3)
    gal1 = galsim.Gaussian(fwhm=1)
    gal2 = galsim.Gaussian(fwhm=2)

    # Sum.  Params are objs to add and potentially gsparams.
    gals = [galsim.Sum(gal1),
            galsim.Sum(gal1, gal2),
            galsim.Sum(gal2, gal1),  # Not! commutative.  (but is associative)
            galsim.Sum(galsim.Sum(gal1, gal2), gal2),
            galsim.Sum(gal1, gsparams=gsp),
            galsim.Sum(gal1, gsparams=gsp, propagate_gsparams=False)]
    check_all_diff(gals)


@timer
def test_sum_transform():
    """This test addresses a bug found by Ismael Serrano, #763, wherein some attributes
    got messed up for a Transform(Sum(Transform())) object.

    The bug was that we didn't bother to make a new SBProfile for a Sum (or Convolve) of
    a single object.  But if that was an SBTransform, then the next Transform operation
    combined the two Transforms, which messed up the repr.

    The fix is to always make an SBAdd or SBConvolve object even if the list of things to add
    or convolve only has one element.
    """
    gal0 = galsim.Exponential(scale_radius=0.34, flux=105.).shear(g1=-0.56,g2=0.15)

    for gal1 in [ galsim.Sum(gal0), galsim.Convolve(gal0) ]:
        gal2 = gal1.dilate(1)

        sgal1 = eval(str(gal1))
        rgal1 = eval(repr(gal1))
        sgal2 = eval(str(gal2))
        rgal2 = eval(repr(gal2))

        gal1_im = gal1.drawImage(nx=64, ny=64, scale=0.2)
        sgal1_im = sgal1.drawImage(nx=64, ny=64, scale=0.2)
        rgal1_im = rgal1.drawImage(nx=64, ny=64, scale=0.2)

        gal2_im = gal2.drawImage(nx=64, ny=64, scale=0.2)
        sgal2_im = sgal2.drawImage(nx=64, ny=64, scale=0.2)
        rgal2_im = rgal2.drawImage(nx=64, ny=64, scale=0.2)

        # Check that the objects are equivalent, even if they may be written differently.
        np.testing.assert_almost_equal(gal1_im.array, sgal1_im.array, decimal=8)
        np.testing.assert_almost_equal(gal1_im.array, rgal1_im.array, decimal=8)

        # These two used to fail.
        np.testing.assert_almost_equal(gal2_im.array, sgal2_im.array, decimal=8)
        np.testing.assert_almost_equal(gal2_im.array, rgal2_im.array, decimal=8)

        check_pickle(gal0)
        check_pickle(gal1)
        check_pickle(gal2)

@timer
def test_sum_noise():
    """Test that noise propagation works properly for compound objects.
    """
    obj1 = galsim.Gaussian(sigma=1.7)
    obj2 = galsim.Gaussian(sigma=2.3)
    obj1.noise = galsim.UncorrelatedNoise(variance=0.3, scale=0.2)
    obj2.noise = galsim.UncorrelatedNoise(variance=0.5, scale=0.2)
    obj3 = galsim.Gaussian(sigma=2.9)

    # Sum adds the variance of the components
    sum2 = galsim.Sum([obj1,obj2])
    np.testing.assert_almost_equal(sum2.noise.getVariance(), 0.8,
            err_msg = "Sum of two objects did not add noise varinace")
    sum2 = galsim.Sum([obj1,obj3])
    np.testing.assert_almost_equal(sum2.noise.getVariance(), 0.3,
            err_msg = "Sum of two objects did not add noise varinace")
    sum2 = galsim.Sum([obj2,obj3])
    np.testing.assert_almost_equal(sum2.noise.getVariance(), 0.5,
            err_msg = "Sum of two objects did not add noise varinace")
    sum3 = galsim.Sum([obj1,obj2,obj3])
    np.testing.assert_almost_equal(sum3.noise.getVariance(), 0.8,
            err_msg = "Sum of three objects did not add noise varinace")
    sum3 = obj1 + obj2 + obj3
    np.testing.assert_almost_equal(sum3.noise.getVariance(), 0.8,
            err_msg = "Sum of three objects did not add noise varinace")

    # Adding noise objects with different WCSs will raise a warning.
    obj4 = galsim.Gaussian(sigma=3.3)
    obj4.noise = galsim.UncorrelatedNoise(variance=0.3, scale=0.8)
    try:
        np.testing.assert_warns(galsim.GalSimWarning, galsim.Sum, [obj1, obj4])
    except:
        pass

@timer
def test_gsparams():
    """Test withGSParams with some non-default gsparams
    """
    obj1 = galsim.Exponential(half_light_radius=1.7)
    obj2 = galsim.Pixel(scale=0.2)
    gsp = galsim.GSParams(folding_threshold=1.e-4, maxk_threshold=1.e-4, maximum_fft_size=1.e4)
    gsp2 = galsim.GSParams(folding_threshold=1.e-2, maxk_threshold=1.e-2)

    sum = galsim.Sum(obj1, obj2)
    sum1 = sum.withGSParams(gsp)
    assert sum.gsparams == galsim.GSParams()
    assert sum1.gsparams == gsp
    assert sum1.obj_list[0].gsparams == gsp
    assert sum1.obj_list[1].gsparams == gsp

    sum2 = galsim.Sum(obj1.withGSParams(gsp), obj2.withGSParams(gsp))
    sum3 = galsim.Sum(galsim.Exponential(half_light_radius=1.7, gsparams=gsp),
                       galsim.Pixel(scale=0.2))
    sum4 = galsim.Add(obj1, obj2, gsparams=gsp)
    assert sum != sum1
    assert sum1 == sum2
    assert sum1 == sum3
    assert sum1 == sum4
    print('stepk = ',sum.stepk, sum1.stepk)
    assert sum1.stepk < sum.stepk
    print('maxk = ',sum.maxk, sum1.maxk)
    assert sum1.maxk > sum.maxk

    sum5 = galsim.Add(obj1, obj2, gsparams=gsp, propagate_gsparams=False)
    assert sum5 != sum4
    assert sum5.gsparams == gsp
    assert sum5.obj_list[0].gsparams == galsim.GSParams()
    assert sum5.obj_list[1].gsparams == galsim.GSParams()

    sum6 = sum5.withGSParams(gsp2)
    assert sum6 != sum5
    assert sum6.gsparams == gsp2
    assert sum6.obj_list[0].gsparams == galsim.GSParams()
    assert sum6.obj_list[1].gsparams == galsim.GSParams()



if __name__ == "__main__":
    runtests(__file__)
