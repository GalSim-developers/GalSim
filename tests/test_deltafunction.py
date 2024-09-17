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
def test_deltaFunction():
    """Test the generation of a Delta function profile
    """
    # Check construction with no arguments gives expected result
    delta = galsim.DeltaFunction()
    np.testing.assert_almost_equal(delta.flux, 1.0)
    check_basic(delta, "DeltaFunction")
    check_pickle(delta)

    # Check with default_params
    delta = galsim.DeltaFunction(flux=1, gsparams=default_params)
    np.testing.assert_almost_equal(delta.flux, 1.0)

    test_flux = 17.9
    delta = galsim.DeltaFunction(flux=test_flux)
    np.testing.assert_almost_equal(delta.flux, test_flux)
    check_basic(delta, "DeltaFunction")
    check_pickle(delta)

    gsp = galsim.GSParams(xvalue_accuracy=1.e-8, kvalue_accuracy=1.e-8)
    delta2 = galsim.DeltaFunction(flux=test_flux, gsparams=gsp)
    assert delta2 != delta
    assert delta2 == delta.withGSParams(gsp)

    # Test operations with no-ops on DeltaFunction
    delta_shr = delta.shear(g1=0.3, g2=0.1)
    np.testing.assert_almost_equal(delta_shr.flux, test_flux)

    delta_dil = delta.dilate(2.0)
    np.testing.assert_almost_equal(delta_dil.flux, test_flux)

    delta_rot = delta.rotate(45 * galsim.radians)
    np.testing.assert_almost_equal(delta_rot.flux, test_flux)

    delta_tfm = delta.transform(dudx=1.25, dudy=0., dvdx=0., dvdy=0.8)
    np.testing.assert_almost_equal(delta_tfm.flux, test_flux)

    delta_shift = delta.shift(1.,2.)
    np.testing.assert_almost_equal(delta_shift.flux, test_flux)

    # These aren't no ops, since they do in fact alter the flux.
    delta_exp = delta.expand(2.0)
    np.testing.assert_almost_equal(delta_exp.flux, test_flux * 4)

    delta_mag = delta.magnify(2.0)
    np.testing.assert_almost_equal(delta_mag.flux, test_flux * 2)

    delta_tfm = delta.transform(dudx=1.4, dudy=0.2, dvdx=0.4, dvdy=1.2)
    np.testing.assert_almost_equal(delta_tfm.flux, test_flux * (1.4*1.2-0.2*0.4))

    # Test simple translation of DeltaFunction
    delta2 = delta.shift(1.,2.)
    offcen = galsim.PositionD(1, 2)
    np.testing.assert_equal(delta2.centroid, offcen)
    assert delta2.xValue(offcen) > 1.e10
    np.testing.assert_almost_equal(delta2.xValue(galsim.PositionD(0,0)), 0)

    # Test photon shooting.
    gauss = galsim.Gaussian(sigma = 1.0)
    delta_conv = galsim.Convolve(gauss,delta)
    myImg = galsim.ImageF()
    do_shoot(delta_conv,myImg,"Delta Function")

    # Test kvalues
    do_kvalue(delta_conv,myImg,"Delta Function")


@timer
def test_deltaFunction_properties():
    """Test some basic properties of the Delta function profile
    """
    test_flux = 17.9
    delta = galsim.DeltaFunction(flux=test_flux)
    # Check that we are centered on (0, 0)
    cen = galsim.PositionD(0, 0)
    np.testing.assert_equal(delta.centroid, cen)
    offcen = galsim.PositionD(1,1)
    # Check Fourier properties
    np.testing.assert_equal(delta.kValue(cen), (1+0j) * test_flux)
    np.testing.assert_equal(delta.kValue(offcen), (1+0j) * test_flux)
    import math
    assert delta.xValue(cen) > 1.e10
    np.testing.assert_almost_equal(delta.xValue(offcen), 0)
    assert delta.maxk > 1.e10
    assert delta.stepk > 1.e10
    # Check input flux vs output flux
    for inFlux in np.logspace(-2, 2, 10):
        delta = galsim.DeltaFunction(flux=inFlux)
        outFlux = delta.flux
        np.testing.assert_almost_equal(outFlux, inFlux)

@timer
def test_deltaFunction_flux_scaling():
    """Test flux scaling for Delta function.
    """
    # decimal point to go to for parameter value comparisons
    param_decimal = 12
    test_flux = 17.9

    # init with flux only (should be ok given last tests)
    obj = galsim.DeltaFunction(flux=test_flux)
    obj *= 2.
    np.testing.assert_almost_equal(
        obj.flux, test_flux * 2., decimal=param_decimal,
        err_msg="Flux param inconsistent after __imul__.")
    obj = galsim.DeltaFunction(flux=test_flux)
    obj /= 2.
    np.testing.assert_almost_equal(
        obj.flux, test_flux / 2., decimal=param_decimal,
        err_msg="Flux param inconsistent after __idiv__.")
    obj = galsim.DeltaFunction(flux=test_flux)
    obj2 = obj * 2.
    # First test that original obj is unharmed...
    np.testing.assert_almost_equal(
        obj.flux, test_flux, decimal=param_decimal,
        err_msg="Flux param inconsistent after __rmul__ (original).")
    # Then test new obj2 flux
    np.testing.assert_almost_equal(
        obj2.flux, test_flux * 2., decimal=param_decimal,
        err_msg="Flux param inconsistent after __rmul__ (result).")
    obj = galsim.DeltaFunction(flux=test_flux)
    obj2 = 2. * obj
    # First test that original obj is unharmed...
    np.testing.assert_almost_equal(
        obj.flux, test_flux, decimal=param_decimal,
        err_msg="Flux param inconsistent after __mul__ (original).")
    # Then test new obj2 flux
    np.testing.assert_almost_equal(
        obj2.flux, test_flux * 2., decimal=param_decimal,
        err_msg="Flux param inconsistent after __mul__ (result).")
    obj = galsim.DeltaFunction(flux=test_flux)
    obj2 = obj / 2.
    # First test that original obj is unharmed...
    np.testing.assert_almost_equal(
        obj.flux, test_flux, decimal=param_decimal,
        err_msg="Flux param inconsistent after __div__ (original).")
    # Then test new obj2 flux
    np.testing.assert_almost_equal(
        obj2.flux, test_flux / 2., decimal=param_decimal,
        err_msg="Flux param inconsistent after __div__ (result).")

    obj = galsim.DeltaFunction(flux=test_flux)
    obj2 = obj.withFlux(test_flux*2.)
    # First test that original obj is unharmed...
    np.testing.assert_almost_equal(
        obj.flux, test_flux, decimal=param_decimal,
        err_msg="Flux param inconsistent after obj.withFlux(flux).")
    # Then test new obj2 flux
    np.testing.assert_almost_equal(
        obj2.flux, test_flux * 2., decimal=param_decimal,
        err_msg="Flux param inconsistent after obj.withFlux(flux).")

@timer
def test_deltaFunction_convolution():
    """Test convolutions using the Delta function.
    """
    # Convolve two delta functions
    delta = galsim.DeltaFunction(flux=2.0)
    delta2 = galsim.DeltaFunction(flux=3.0)
    delta_delta = galsim.Convolve(delta,delta2)
    np.testing.assert_almost_equal(delta_delta.flux,6.0)

    # Test that no-ops on gaussians dont affect convolution
    gauss = galsim.Gaussian(sigma = 1.0)

    delta_shr = delta.shear(g1=0.3, g2=0.1)
    delta_conv = galsim.Convolve(gauss,delta_shr)
    np.testing.assert_almost_equal(delta_conv.flux,2.0)

    delta_dil = delta.dilate(2.0)
    delta_conv = galsim.Convolve(gauss,delta_dil)
    np.testing.assert_almost_equal(delta_conv.flux,2.0)

    delta_tfm = delta.transform(dudx=1.25, dudy=0.0, dvdx=0.0, dvdy=0.8)
    delta_conv = galsim.Convolve(gauss,delta_tfm)
    np.testing.assert_almost_equal(delta_conv.flux,2.0)

    delta_rot = delta.rotate(45 * galsim.radians)
    delta_conv = galsim.Convolve(gauss,delta_rot)
    np.testing.assert_almost_equal(delta_conv.flux,2.0)


if __name__ == "__main__":
    runtests(__file__)
