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

try:
    import galsim
    from galsim.cdmodel import *
except ImportError:
    path, filename = os.path.split(__file__)
    sys.path.append(os.path.abspath(os.path.join(path, "..")))
    import galsim
    from galsim.cdmodel import *
    
# Use a deterministic random number generator so we don't fail tests because of rare flukes in the
# random numbers.
rseed=12345


def test_simplegeometry():
    """Test charge deflection model for image with charges in only the central pixel(s).
    """
    import time
    t1 = time.time()
    size = 50
    center = 25
    shiftcoeff = 0.1
    
    # create otherwise empty image with central pixel at one
    i0 = galsim.Image(size,size,dtype=np.float64,init_value=0)
    i0.setValue(center,center,1)
    
    # create otherwise empty image with three central pixels at one
    # central row
    ir = galsim.Image(size,size,dtype=np.float64,init_value=0)
    ir.setValue(center-1,center,1)
    ir.setValue(center  ,center,1)
    ir.setValue(center+1,center,1)
    # central column
    it = galsim.Image(size,size,dtype=np.float64,init_value=0)
    it.setValue(center,center-1,1)
    it.setValue(center,center  ,1)
    it.setValue(center,center+1,1)

    # set up models, images
    cdr0   = PowerLawCD(2,shiftcoeff,0,0,0,0,0,0)
    i0cdr0 = cdr0.applyForward(i0)
    cdt0   = PowerLawCD(2,0,shiftcoeff,0,0,0,0,0)
    i0cdt0 = cdt0.applyForward(i0)
    cdrx   = PowerLawCD(2,0,0,shiftcoeff,0,0,0,0)
    cdtx   = PowerLawCD(2,0,0,0,shiftcoeff,0,0,0)

    # these should do something
    ircdtx = cdtx.applyForward(ir)
    itcdrx = cdrx.applyForward(it)
    
    # these shouldn't do anything
    itcdtx = cdtx.applyForward(it)
    ircdrx = cdrx.applyForward(ir)
    
    # R0, T0
    np.testing.assert_almost_equal(i0cdr0.at(center,center), 1.-shiftcoeff, 10,
                                   "Central pixel wrong in test_onepixel R0")
    np.testing.assert_almost_equal(i0cdt0.at(center,center), 1.-shiftcoeff, 10,
                                   "Central pixel wrong in test_onepixel T0")
    
    np.testing.assert_almost_equal(i0cdr0.at(center+1,center), shiftcoeff/2., 10,
                                   "Off-center pixel wrong in test_onepixel R0")
    np.testing.assert_almost_equal(i0cdr0.at(center-1,center), shiftcoeff/2., 10,
                                   "Off-center pixel wrong in test_onepixel R0")
    
    np.testing.assert_almost_equal(i0cdt0.at(center,center+1), shiftcoeff/2., 10,
                                   "Off-center pixel wrong in test_onepixel T0")
    np.testing.assert_almost_equal(i0cdt0.at(center,center-1), shiftcoeff/2., 10,
                                   "Off-center pixel wrong in test_onepixel T0")
    
    # Tx
    np.testing.assert_almost_equal(ircdtx.at(center,center), 1.-2.*shiftcoeff, 10,
                                   "Central pixel wrong in test_onepixel TX")
    np.testing.assert_almost_equal(ircdtx.at(center-1,center), 1.-shiftcoeff, 10,
                                   "Off-center pixel wrong in test_onepixel TX")
    np.testing.assert_almost_equal(ircdtx.at(center+1,center), 1.-shiftcoeff, 10,
                                   "Off-center pixel wrong in test_onepixel TX")
                                       
    np.testing.assert_almost_equal(ircdtx.at(center,center+1), shiftcoeff, 10,
                                   "Central pixel wrong in test_onepixel TX")
    np.testing.assert_almost_equal(ircdtx.at(center-1,center+1), shiftcoeff/2., 10,
                                   "Off-center pixel wrong in test_onepixel TX")
    np.testing.assert_almost_equal(ircdtx.at(center+1,center+1), shiftcoeff/2., 10,
                                   "Off-center pixel wrong in test_onepixel TX")
                                   
    np.testing.assert_almost_equal(ircdtx.at(center,center-1), shiftcoeff, 10,
                                   "Central pixel wrong in test_onepixel TX")
    np.testing.assert_almost_equal(ircdtx.at(center-1,center-1), shiftcoeff/2., 10,
                                   "Off-center pixel wrong in test_onepixel TX")
    np.testing.assert_almost_equal(ircdtx.at(center+1,center-1), shiftcoeff/2., 10,
                                   "Off-center pixel wrong in test_onepixel TX")
                                   
    # Rx
    np.testing.assert_almost_equal(itcdrx.at(center,center), 1.-2.*shiftcoeff, 10,
                                   "Central pixel wrong in test_onepixel RX")
    np.testing.assert_almost_equal(itcdrx.at(center,center-1), 1.-shiftcoeff, 10,
                                   "Off-center pixel wrong in test_onepixel RX")
    np.testing.assert_almost_equal(itcdrx.at(center,center+1), 1.-shiftcoeff, 10,
                                   "Off-center pixel wrong in test_onepixel RX")
                                       
    np.testing.assert_almost_equal(itcdrx.at(center+1,center), shiftcoeff, 10,
                                   "Central pixel wrong in test_onepixel RX")
    np.testing.assert_almost_equal(itcdrx.at(center+1,center-1), shiftcoeff/2., 10,
                                   "Off-center pixel wrong in test_onepixel RX")
    np.testing.assert_almost_equal(itcdrx.at(center+1,center+1), shiftcoeff/2., 10,
                                   "Off-center pixel wrong in test_onepixel RX")
                                   
    np.testing.assert_almost_equal(itcdrx.at(center-1,center), shiftcoeff, 10,
                                   "Central pixel wrong in test_onepixel RX")
    np.testing.assert_almost_equal(itcdrx.at(center-1,center-1), shiftcoeff/2., 10,
                                   "Off-center pixel wrong in test_onepixel RX")
    np.testing.assert_almost_equal(itcdrx.at(center-1,center+1), shiftcoeff/2., 10,
                                   "Off-center pixel wrong in test_onepixel RX")
    
    # a model that should not change anything here
    u = galsim.UniformDeviate(rseed)
    
    cdnull = PowerLawCD(2, 0, 0, shiftcoeff*u(), shiftcoeff*u(), shiftcoeff*u(), shiftcoeff*u(), 0)
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
    t2 = time.time()
    print 'time for %s = %.2f' % (funcname(), t2 - t1)

def test_fluxconservation():
    """Test flux conservation of charge deflection model for galaxy and flat image.
    """
    import time
    t1 = time.time()
    galflux = 30.
    galsigma = 3.
    noise = 0.01
    shiftcoeff = 5.e-2
    alpha = 0.3
    size = 50

    gal = galsim.Gaussian(flux=galflux, sigma=galsigma)
    image = gal.drawImage(scale=1.,dtype=np.float64)
    image.addNoise(galsim.GaussianNoise(sigma=noise))

    flat = galsim.Image(size, size, dtype=np.float64, init_value=0)
    flat.fill(1.)
    cd = PowerLawCD(
        2, shiftcoeff, shiftcoeff, shiftcoeff/2., shiftcoeff/2., shiftcoeff/2., shiftcoeff/2.,
        alpha)
    imagecd = cd.applyForward(image)
    flatcd  = cd.applyForward(flat)
    
    # Then test
    np.testing.assert_almost_equal(
        image.array.sum(), imagecd.array.sum(), 10,
        "Galaxy image flux is not left invariant by charge deflection")
    np.testing.assert_almost_equal(
        flat.array.sum(), flatcd.array.sum(), 10,
        "Flat image flux is not left invariant by charge deflection")
    t2 = time.time()
    print 'time for %s = %.2f' % (funcname(), t2 - t1)
    
def test_forwardbackward():
    """Test invariance (to first order) under forward-backward transformation.
    """
    import time
    t1 = time.time()
    galflux = 30.
    galsigma = 3.
    noise = 0.01
    shiftcoeff = 1.e-5
    alpha = 0.3
    size = 50

    gal = galsim.Gaussian(flux=galflux, sigma=galsigma)
    maxflux = gal.xValue(0,0)
    image = gal.drawImage(scale=1.,dtype=np.float64)
    
    cimage = galsim.Image(image.getBounds(),dtype=np.float64) 
    # used for normalization later, we expect residual to be of this order
    cimage.fill(1.e-3)
    cimage = cimage+image
    cimage = cimage*maxflux*maxflux*shiftcoeff*shiftcoeff

    image.addNoise(galsim.GaussianNoise(sigma=noise))    
    cd = PowerLawCD(
        2, shiftcoeff, 2.*shiftcoeff, shiftcoeff/2., 2.*shiftcoeff/3., shiftcoeff/2.,
        shiftcoeff/3., alpha)
    
    imagecd = cd.applyForward(image)
    imagecddc = cd.applyBackward(imagecd)

    # residual after forward-backward should be of order a^2 q qmax^2
    imageres = (imagecddc - image) / cimage    
    maxres = imageres.array.max()
    minres = imageres.array.min()
    assert maxres<10, ("maximum positive residual of forward-backward transformation is too large")
    assert minres>-10, ("maximum negative residual of forward-backward transformation is too large")
    t2 = time.time()
    print 'time for %s = %.2f' % (funcname(), t2 - t1)
    
    
def test_gainratio():
    """Test gain ratio functionality
    """
    import time
    t1 = time.time()
    galflux = 30.
    galsigma = 3.
    noise = 0.01
    shiftcoeff = 1.e-5
    alpha = 0.3
    size = 50

    # image with fiducial gain
    gal    = galsim.Gaussian(flux=galflux, sigma=galsigma)
    image  = gal.drawImage(scale=1.,dtype=np.float64)    
    
    # image with twice the gain, i.e. half the level
    gal2   = galsim.Gaussian(flux=0.5*galflux, sigma=galsigma)    
    image2 = gal2.drawImage(scale=1.,dtype=np.float64)   
    
    cd = PowerLawCD(2, shiftcoeff, 2.*shiftcoeff, shiftcoeff/2., 2.*shiftcoeff/3., shiftcoeff/2.,
        shiftcoeff/3., alpha)
        
    image_cd  = cd.applyForward(image)
    image2_cd = cd.applyForward(image2,gain_ratio=2.)
    
    imageres = (2.*image2_cd - image_cd)
    np.testing.assert_array_almost_equal(2.*image2_cd.array, image_cd.array, 10,
                                   "images with different gain not transformed equally")
    t2 = time.time()
    print 'time for %s = %.2f' % (funcname(), t2 - t1)
    
    
    
def test_exampleimage():
    """Test application of model compared to an independent implementation that was run on the
    example image.
    """
    import time
    t1 = time.time()
    #n, r0, t0, rx, tx, r, t, alpha
    cd = PowerLawCD(5, 2.e-7, 1.e-7, 1.25e-7, 1.25e-7, 0.75e-7, 0.5e-7, 0.3)
    # model used externally to bring cdtest1 to cdtest2
    image_orig  = galsim.fits.read("fits_files/cdtest1.fits") # unprocessed image
    image_proc  = galsim.fits.read("fits_files/cdtest2.fits") # image with cd model applied with
                                                              # other library
    # Calculate the test image
    image_plcd  = cd.applyForward(image_orig)
    # For debugging (remove at end of PR?): make if True in block below to output difference image.
    # Compare to fits_files/cdtest[1-2].fits above
    if False:
        import pyfits
        pyfits.writeto(
            "junk_test_cdmodel_exampleimage_difference.fits", (image_proc - image_plcd).array,
            clobber=True)
    # These images have a large flux per pixel, so make the typical flux per pixel in each image
    # closer to O(1) for a more transparently meaningful decimal order in the test
    norm = 2.64 / np.std(image_orig.array)
    image_proc *= norm
    image_plcd *= norm
    # Compare
    np.testing.assert_array_almost_equal(
        image_proc.array, image_plcd.array, 4, "Externally and internally processed image unequal")
        # DG checked that the remaining differences appear to be numerical noise - BR agrees the
        # that the difference images do not show coherent structure other than a border feature
        # which is expected
    t2 = time.time()
    print 'time for %s = %.2f' % (funcname(), t2 - t1)


if __name__ == "__main__":
    test_simplegeometry()
    test_fluxconservation()
    test_forwardbackward()
    test_gainratio()
    test_exampleimage()
