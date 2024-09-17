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


# Use a deterministic random number generator so we don't fail tests because of rare flukes in the
# random numbers.
rseed=12345


@timer
def test_simplegeometry():
    """Test charge deflection model for image with charges in only the central pixel(s).
    """
    size = 50
    center = 25
    shiftcoeff = 1.e-7
    # shift coefficients in DECam are of that order
    # note that this is fully degenerate with the gain, i.e. the flux level in the simulations

    level = 1.e5

    # create otherwise empty image with central pixel at one
    i0 = galsim.Image(size,size, dtype=np.float64, init_value=0)
    i0.setValue(center,center,level)

    # create otherwise empty image with three central pixels at one
    # central row
    ir = galsim.Image(size,size, dtype=np.float64, init_value=0)
    ir.setValue(center-1,center,level)
    ir.setValue(center  ,center,level)
    ir.setValue(center+1,center,level)
    # central column
    it = galsim.Image(size,size, dtype=np.float64, init_value=0)
    it.setValue(center,center-1,level)
    it.setValue(center,center  ,level)
    it.setValue(center,center+1,level)

    # set up models, images
    cdr0   = galsim.cdmodel.PowerLawCD(2,shiftcoeff,0,0,0,0,0,0)
    i0cdr0 = cdr0.applyForward(i0)

    cdt0   = galsim.cdmodel.PowerLawCD(2,0,shiftcoeff,0,0,0,0,0)
    i0cdt0 = cdt0.applyForward(i0)
    cdrx   = galsim.cdmodel.PowerLawCD(2,0,0,shiftcoeff,0,0,0,0)
    cdtx   = galsim.cdmodel.PowerLawCD(2,0,0,0,shiftcoeff,0,0,0)

    # these should do something
    ircdtx = cdtx.applyForward(ir)
    itcdrx = cdrx.applyForward(it)

    # these shouldn't do anything
    itcdtx = cdtx.applyForward(it)
    ircdrx = cdrx.applyForward(ir)

    # R0, T0
    np.testing.assert_almost_equal(i0cdr0(center,center), level*(1.-level*shiftcoeff),
                                   13-int(np.log10(level)),
                                   "Central pixel wrong in test_onepixel R0")
    np.testing.assert_almost_equal(i0cdt0(center,center), level*(1.-level*shiftcoeff),
                                   13-int(np.log10(level)),
                                   "Central pixel wrong in test_onepixel T0")

    np.testing.assert_almost_equal(i0cdr0(center+1,center), level*(level*shiftcoeff/2.),
                                   13-int(np.log10(level)),
                                   "Off-center pixel wrong in test_onepixel R0")
    np.testing.assert_almost_equal(i0cdr0(center-1,center), level*(level*shiftcoeff/2.),
                                   13-int(np.log10(level)),
                                   "Off-center pixel wrong in test_onepixel R0")

    np.testing.assert_almost_equal(i0cdt0(center,center+1), level*(level*shiftcoeff/2.),
                                   13-int(np.log10(level)),
                                   "Off-center pixel wrong in test_onepixel T0")
    np.testing.assert_almost_equal(i0cdt0(center,center-1), level*(level*shiftcoeff/2.),
                                   13-int(np.log10(level)),
                                   "Off-center pixel wrong in test_onepixel T0")

    # Tx
    np.testing.assert_almost_equal(ircdtx(center,center), level*(1.-2.*level*shiftcoeff),
                                   13-int(np.log10(level)),
                                   "Central pixel wrong in test_onepixel TX")
    np.testing.assert_almost_equal(ircdtx(center-1,center), level*(1.-level*shiftcoeff),
                                   13-int(np.log10(level)),
                                   "Off-center pixel wrong in test_onepixel TX")
    np.testing.assert_almost_equal(ircdtx(center+1,center), level*(1.-level*shiftcoeff),
                                   13-int(np.log10(level)),
                                   "Off-center pixel wrong in test_onepixel TX")

    np.testing.assert_almost_equal(ircdtx(center,center+1), level*level*shiftcoeff,
                                   13-int(np.log10(level)),
                                   "Central pixel wrong in test_onepixel TX")
    np.testing.assert_almost_equal(ircdtx(center-1,center+1), level*level*shiftcoeff/2.,
                                   13-int(np.log10(level)),
                                   "Off-center pixel wrong in test_onepixel TX")
    np.testing.assert_almost_equal(ircdtx(center+1,center+1), level*level*shiftcoeff/2.,
                                   13-int(np.log10(level)),
                                   "Off-center pixel wrong in test_onepixel TX")

    np.testing.assert_almost_equal(ircdtx(center,center-1), level*level*shiftcoeff,
                                   13-int(np.log10(level)),
                                   "Central pixel wrong in test_onepixel TX")
    np.testing.assert_almost_equal(ircdtx(center-1,center-1), level*level*shiftcoeff/2.,
                                   13-int(np.log10(level)),
                                   "Off-center pixel wrong in test_onepixel TX")
    np.testing.assert_almost_equal(ircdtx(center+1,center-1), level*level*shiftcoeff/2.,
                                   13-int(np.log10(level)),
                                   "Off-center pixel wrong in test_onepixel TX")

    # Rx
    np.testing.assert_almost_equal(itcdrx(center,center), level*(1.-2.*level*shiftcoeff),
                                   13-int(np.log10(level)),
                                   "Central pixel wrong in test_onepixel RX")
    np.testing.assert_almost_equal(itcdrx(center,center-1), level*(1.-level*shiftcoeff),
                                   13-int(np.log10(level)),
                                   "Off-center pixel wrong in test_onepixel RX")
    np.testing.assert_almost_equal(itcdrx(center,center+1), level*(1.-level*shiftcoeff),
                                   13-int(np.log10(level)),
                                   "Off-center pixel wrong in test_onepixel RX")

    np.testing.assert_almost_equal(itcdrx(center+1,center), level*level*shiftcoeff,
                                   13-int(np.log10(level)),
                                   "Central pixel wrong in test_onepixel RX")
    np.testing.assert_almost_equal(itcdrx(center+1,center-1), level*level*shiftcoeff/2.,
                                   13-int(np.log10(level)),
                                   "Off-center pixel wrong in test_onepixel RX")
    np.testing.assert_almost_equal(itcdrx(center+1,center+1), level*level*shiftcoeff/2.,
                                   13-int(np.log10(level)),
                                   "Off-center pixel wrong in test_onepixel RX")

    np.testing.assert_almost_equal(itcdrx(center-1,center), level*level*shiftcoeff,
                                   13-int(np.log10(level)),
                                   "Central pixel wrong in test_onepixel RX")
    np.testing.assert_almost_equal(itcdrx(center-1,center-1), level*level*shiftcoeff/2.,
                                   13-int(np.log10(level)),
                                   "Off-center pixel wrong in test_onepixel RX")
    np.testing.assert_almost_equal(itcdrx(center-1,center+1), level*level*shiftcoeff/2.,
                                   13-int(np.log10(level)),
                                   "Off-center pixel wrong in test_onepixel RX")

    # a model that should not change anything here
    u = galsim.UniformDeviate(rseed)

    cdnull = galsim.cdmodel.PowerLawCD(
        2, 0, 0, shiftcoeff*u(), shiftcoeff*u(), shiftcoeff*u(), shiftcoeff*u(), 0)
    i0cdnull = cdnull.applyForward(i0)

    # setting all pixels to 0 that we expect to be not 0...
    i0.setValue(center,center,0)
    i0cdnull.setValue(center,center,0)
    i0cdr0.setValue(center,center,0)
    i0cdr0.setValue(center+1,center,0)
    i0cdr0.setValue(center-1,center,0)
    i0cdt0.setValue(center,center,0)
    i0cdt0.setValue(center,center+1,0)
    i0cdt0.setValue(center,center-1,0)

    ircdtx.subImage(galsim.BoundsI(center-1,center+1,center-1,center+1)).fill(0)
    itcdrx.subImage(galsim.BoundsI(center-1,center+1,center-1,center+1)).fill(0)

    ircdrx.subImage(galsim.BoundsI(center-1,center+1,center,center)).fill(0)
    itcdtx.subImage(galsim.BoundsI(center,center,center-1,center+1)).fill(0)

    # ... and comparing
    np.testing.assert_array_almost_equal(i0cdnull.array, i0.array, 10,
                                   "i0cdnull array is not 0 where it should be")
    np.testing.assert_array_almost_equal(i0cdr0.array, i0.array, 10,
                                   "i0cdr0 array is not 0 where it should be")
    np.testing.assert_array_almost_equal(i0cdt0.array, i0.array, 10,
                                   "i0cdr0 array is not 0 where it should be")
    np.testing.assert_array_almost_equal(ircdtx.array, i0.array, 10,
                                   "ircdtx array is not 0 where it should be")
    np.testing.assert_array_almost_equal(ircdrx.array, i0.array, 10,
                                   "ircdrx array is not 0 where it should be")
    np.testing.assert_array_almost_equal(itcdtx.array, i0.array, 10,
                                   "itcdtx array is not 0 where it should be")
    np.testing.assert_array_almost_equal(itcdrx.array, i0.array, 10,
                                   "itcdrx array is not 0 where it should be")


@timer
def test_cdmodel_errors():
    """Test some invalid usage of CDModel"""

    # I don't think these errors are possible from the PowerLawCD constructor, so test
    # them directly in the base class.
    with assert_raises(galsim.GalSimValueError):
        # Must be odd x odd
        galsim.cdmodel.BaseCDModel(
            np.zeros((4,4)), np.zeros((4,4)), np.zeros((4,4)), np.zeros((4,4)) )
    with assert_raises(galsim.GalSimValueError):
        # Must be square
        galsim.cdmodel.BaseCDModel(
            np.zeros((5,3)), np.zeros((5,3)), np.zeros((5,3)), np.zeros((5,3)) )
    with assert_raises(galsim.GalSimValueError):
        # Must be same shape
        galsim.cdmodel.BaseCDModel(
            np.zeros((3,3)), np.zeros((3,3)), np.zeros((3,3)), np.zeros((5,5)) )
    with assert_raises(galsim.GalSimValueError):
        # Must be >= 3x3
        galsim.cdmodel.BaseCDModel(
            np.zeros((1,1)), np.zeros((1,1)), np.zeros((1,1)), np.zeros((1,1)) )

@timer
def test_fluxconservation():
    """Test flux conservation of charge deflection model for galaxy and flat image.
    """
    galflux = 3.e4
    galsigma = 3.
    noise = 10.
    shiftcoeff = 1.e-7
    alpha = 0.3
    size = 50

    # Define a consistent rng for repeatability
    urng = galsim.UniformDeviate(rseed)

    gal = galsim.Gaussian(flux=galflux, sigma=galsigma)
    image = gal.drawImage(scale=1.,dtype=np.float64)
    image.addNoise(galsim.GaussianNoise(sigma=noise, rng=urng))

    flat = galsim.Image(size, size, dtype=np.float64, init_value=1.)
    cd = galsim.cdmodel.PowerLawCD(
        2, shiftcoeff, 0.94 * shiftcoeff, shiftcoeff/2.4, shiftcoeff/5., shiftcoeff/3.7,
        shiftcoeff/1.8, alpha)
    imagecd = cd.applyForward(image)
    flatcd  = cd.applyForward(flat)

    # Then test
    np.testing.assert_almost_equal(
        image.array.sum(), imagecd.array.sum(), 13-int(np.log10(galflux)),
        "Galaxy image flux is not left invariant by charge deflection")
    np.testing.assert_almost_equal(
        flat.array.sum(), flatcd.array.sum(), 13-int(np.log10(galflux)),
        "Flat image flux is not left invariant by charge deflection")

    # Check picklability
    check_pickle(cd, lambda x: x.applyForward(image))
    check_pickle(cd)


@timer
def test_forwardbackward():
    """Test invariance (to first order) under forward-backward transformation.
    """
    galflux = 3000.
    galsigma = 3.
    noise = 1.
    shiftcoeff = 1.e-7
    alpha = 0.3
    size = 50

    gal = galsim.Gaussian(flux=galflux, sigma=galsigma)
    maxflux = gal.xValue(0,0)
    image = gal.drawImage(scale=1., dtype=np.float64)

    cimage = galsim.Image(image.bounds, dtype=np.float64)
    # used for normalization later, we expect residual to be of this order
    cimage.fill(1.e-3)
    cimage = cimage+image
    cimage = cimage*maxflux*maxflux*shiftcoeff*shiftcoeff

    # Define a consistent rng for repeatability
    urng = galsim.UniformDeviate(rseed)
    image.addNoise(galsim.GaussianNoise(sigma=noise, rng=urng))
    cd = galsim.cdmodel.PowerLawCD(
        2, shiftcoeff * 0.0234, shiftcoeff * 0.05234, shiftcoeff * 0.01312, shiftcoeff * 0.00823,
        shiftcoeff * 0.07216, shiftcoeff * 0.01934, alpha)

    imagecd = cd.applyForward(image)
    imagecddc = cd.applyBackward(imagecd)

    # residual after forward-backward should be of order a^2 q qmax^2
    imageres = (imagecddc - image) / cimage
    maxres = imageres.array.max()
    minres = imageres.array.min()
    assert maxres<10, ("maximum positive residual of forward-backward transformation is too large")
    assert minres>-10, ("maximum negative residual of forward-backward transformation is too large")


@timer
def test_gainratio():
    """Test gain ratio functionality
    """
    galflux = 3000.
    galsigma = 3.
    noise = 1.
    shiftcoeff = 1.e-7
    alpha = 0.3
    size = 50

    # image with fiducial gain
    gal    = galsim.Gaussian(flux=galflux, sigma=galsigma)
    image  = gal.drawImage(scale=1.,dtype=np.float64)

    # image with twice the gain, i.e. half the level
    gal2   = galsim.Gaussian(flux=0.5*galflux, sigma=galsigma)
    image2 = gal2.drawImage(scale=1.,dtype=np.float64)

    cd = galsim.cdmodel.PowerLawCD(
        2, shiftcoeff, 1.389*shiftcoeff, shiftcoeff/7.23, 2.*shiftcoeff/2.4323,
        shiftcoeff/1.8934, shiftcoeff/3.1, alpha)

    image_cd  = cd.applyForward(image)
    image2_cd = cd.applyForward(image2,gain_ratio=2.)

    imageres = (2.*image2_cd - image_cd)
    np.testing.assert_array_almost_equal(2.*image2_cd.array, image_cd.array,
                                         13-int(np.log10(galflux)),
                                         "images with different gain not transformed equally")


@timer
def test_exampleimage():
    """Test application of model compared to an independent implementation that was run on the
    example image.
    """
    shiftcoeff = 1.e-7

    #n, r0, t0, rx, tx, r, t, alpha
    cd = galsim.cdmodel.PowerLawCD(
        5, 2. * shiftcoeff, shiftcoeff, 1.25 * shiftcoeff, 1.25 * shiftcoeff, 0.75 * shiftcoeff,
        0.5 * shiftcoeff, 0.3)
    # model used externally to bring cdtest1 to cdtest2
    image_orig  = galsim.fits.read("fits_files/cdtest1.fits") # unprocessed image
    image_proc  = galsim.fits.read("fits_files/cdtest2.fits") # image with cd model applied with
                                                              # other library
    # Calculate the test image
    image_plcd  = cd.applyForward(image_orig)

    # These images have a large flux per pixel, so make the typical flux per pixel in each image
    # closer to O(1) for a more transparently meaningful decimal order in the test
    norm = 2.64 / np.std(image_orig.array)
    image_proc *= norm
    image_plcd *= norm
    # Compare
    np.testing.assert_array_almost_equal(
        image_proc.array, image_plcd.array, 4, "Externally and internally processed image unequal")
        # DG checked that the remaining differences appear to be numerical noise - BR agrees
        # that the difference images do not show coherent structure other than a border feature
        # which is expected


if __name__ == "__main__":
    runtests(__file__)
