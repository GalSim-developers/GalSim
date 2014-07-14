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
    

def test_simplegeometry():
    """Test charge deflection model for image with charges in only the central pixel(s)
    """
    size=50
    center=25
    shiftcoeff=0.1
    
    # create otherwise empty image with central pixel at one
    i0 = galsim.Image(size,size,dtype=np.float32,init_value=0)
    i0.setValue(center,center,1)
    
    # create otherwise empty image with three central pixels at one
    # central row
    ir = galsim.Image(size,size,dtype=np.float32,init_value=0)
    ir.setValue(center-1,center,1)
    ir.setValue(center  ,center,1)
    ir.setValue(center+1,center,1)
    # central column
    it = galsim.Image(size,size,dtype=np.float32,init_value=0)
    it.setValue(center,center-1,1)
    it.setValue(center,center  ,1)
    it.setValue(center,center+1,1)
        
      
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
    np.testing.assert_almost_equal(i0cdr0.at(center,center), 1.-shiftcoeff, 4,
                                   "Central pixel wrong in test_onepixel R0")
    np.testing.assert_almost_equal(i0cdt0.at(center,center), 1.-shiftcoeff, 4,
                                   "Central pixel wrong in test_onepixel T0")
    
    np.testing.assert_almost_equal(i0cdr0.at(center+1,center), shiftcoeff/2., 4,
                                   "Off-center pixel wrong in test_onepixel R0")
    np.testing.assert_almost_equal(i0cdr0.at(center-1,center), shiftcoeff/2., 4,
                                   "Off-center pixel wrong in test_onepixel R0")
    
    np.testing.assert_almost_equal(i0cdt0.at(center,center+1), shiftcoeff/2., 4,
                                   "Off-center pixel wrong in test_onepixel T0")
    np.testing.assert_almost_equal(i0cdt0.at(center,center-1), shiftcoeff/2., 4,
                                   "Off-center pixel wrong in test_onepixel T0")
    
    # Tx
    np.testing.assert_almost_equal(ircdtx.at(center,center), 1.-2.*shiftcoeff, 4,
                                   "Central pixel wrong in test_onepixel TX")
    np.testing.assert_almost_equal(ircdtx.at(center-1,center), 1.-shiftcoeff, 4,
                                   "Off-center pixel wrong in test_onepixel TX")
    np.testing.assert_almost_equal(ircdtx.at(center+1,center), 1.-shiftcoeff, 4,
                                   "Off-center pixel wrong in test_onepixel TX")
                                       
    np.testing.assert_almost_equal(ircdtx.at(center,center+1), shiftcoeff, 4,
                                   "Central pixel wrong in test_onepixel TX")
    np.testing.assert_almost_equal(ircdtx.at(center-1,center+1), shiftcoeff/2., 4,
                                   "Off-center pixel wrong in test_onepixel TX")
    np.testing.assert_almost_equal(ircdtx.at(center+1,center+1), shiftcoeff/2., 4,
                                   "Off-center pixel wrong in test_onepixel TX")
                                   
    np.testing.assert_almost_equal(ircdtx.at(center,center-1), shiftcoeff, 4,
                                   "Central pixel wrong in test_onepixel TX")
    np.testing.assert_almost_equal(ircdtx.at(center-1,center-1), shiftcoeff/2., 4,
                                   "Off-center pixel wrong in test_onepixel TX")
    np.testing.assert_almost_equal(ircdtx.at(center+1,center-1), shiftcoeff/2., 4,
                                   "Off-center pixel wrong in test_onepixel TX")
                                   
    # Rx
    np.testing.assert_almost_equal(itcdrx.at(center,center), 1.-2.*shiftcoeff, 4,
                                   "Central pixel wrong in test_onepixel RX")
    np.testing.assert_almost_equal(itcdrx.at(center,center-1), 1.-shiftcoeff, 4,
                                   "Off-center pixel wrong in test_onepixel RX")
    np.testing.assert_almost_equal(itcdrx.at(center,center+1), 1.-shiftcoeff, 4,
                                   "Off-center pixel wrong in test_onepixel RX")
                                       
    np.testing.assert_almost_equal(itcdrx.at(center+1,center), shiftcoeff, 4,
                                   "Central pixel wrong in test_onepixel RX")
    np.testing.assert_almost_equal(itcdrx.at(center+1,center-1), shiftcoeff/2., 4,
                                   "Off-center pixel wrong in test_onepixel RX")
    np.testing.assert_almost_equal(itcdrx.at(center+1,center+1), shiftcoeff/2., 4,
                                   "Off-center pixel wrong in test_onepixel RX")
                                   
    np.testing.assert_almost_equal(itcdrx.at(center-1,center), shiftcoeff, 4,
                                   "Central pixel wrong in test_onepixel RX")
    np.testing.assert_almost_equal(itcdrx.at(center-1,center-1), shiftcoeff/2., 4,
                                   "Off-center pixel wrong in test_onepixel RX")
    np.testing.assert_almost_equal(itcdrx.at(center-1,center+1), shiftcoeff/2., 4,
                                   "Off-center pixel wrong in test_onepixel RX")
    
    # a model that should not change anything here
    
    import time
    u = galsim.UniformDeviate(int(time.time()))
    
    cdnull   = PowerLawCD(2,0,0,shiftcoeff*u(),shiftcoeff*u(),shiftcoeff*u(),shiftcoeff*u(),0)
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
    np.testing.assert_array_almost_equal(i0cdnull.array, i0.array, 6,
                                   "i0cdnull array is not 0 where it should be")
    np.testing.assert_array_almost_equal(i0cdr0.array, i0.array, 6,
                                   "i0cdr0 array is not 0 where it should be")
    np.testing.assert_array_almost_equal(i0cdt0.array, i0.array, 6,
                                   "i0cdr0 array is not 0 where it should be")
    np.testing.assert_array_almost_equal(ircdtx.array, i0.array, 6,
                                   "ircdtx array is not 0 where it should be")
    np.testing.assert_array_almost_equal(ircdrx.array, i0.array, 6,
                                   "ircdrx array is not 0 where it should be")
    np.testing.assert_array_almost_equal(itcdtx.array, i0.array, 6,
                                   "itcdtx array is not 0 where it should be")
    np.testing.assert_array_almost_equal(itcdrx.array, i0.array, 6,
                                   "itcdrx array is not 0 where it should be")


def test_fluxconservation():
    """Test flux conservation of charge deflection model for galaxy and flat image
    """

    galflux=30000.
    galsigma=3.
    noise=1.
    shiftcoeff=5.e-5
    alpha=0.3
    
    size=50

    gal = galsim.Gaussian(flux=galflux, sigma=galsigma)
    image = gal.drawImage(scale=1.)
    image.addNoise(galsim.GaussianNoise(sigma=noise))
    
    flat = galsim.Image(size,size,dtype=np.float32,init_value=0)
    flat.fill(100)
    
    cd   = PowerLawCD(2,shiftcoeff,shiftcoeff,shiftcoeff/2.,shiftcoeff/2.,shiftcoeff/2.,shiftcoeff/2.,alpha)
    
    imagecd = cd.applyForward(image)
    flatcd  = cd.applyForward(flat)
    
    np.testing.assert_almost_equal(image.array.sum(), imagecd.array.sum(), 2, "Galaxy image flux is not left invariant by charge deflection")
    np.testing.assert_almost_equal(flat.array.sum(), flatcd.array.sum(), 2, "Flat image flux is not left invariant by charge deflection")
    
    
def test_forwardbackward():
    """Test invariance (to first order) under forward-backward transformation
    """
    galflux=30000.
    galsigma=3.
    noise=1.
    shiftcoeff=1.e-5
    alpha=0.3
    
    size=50

    gal = galsim.Gaussian(flux=galflux, sigma=galsigma)
    maxflux = gal.xValue(0,0)
    #print maxflux
    image = gal.drawImage(scale=1.)
    
    cimage = galsim.Image(image.getBounds()) # used for normalization later   
    cimage.fill(1.)
    cimage = cimage+image
    cimage = cimage*maxflux*maxflux*shiftcoeff*shiftcoeff
    
    image.addNoise(galsim.GaussianNoise(sigma=noise))    
    
    #image.write("image.fits")
    #cimage.write("imagec.fits")
    
    cd   = PowerLawCD(2,shiftcoeff,2.*shiftcoeff,shiftcoeff/2.,2.*shiftcoeff/3.,shiftcoeff/2.,shiftcoeff/3.,alpha)
    
    imagecd = cd.applyForward(image)
    imagecddc = cd.applyBackward(imagecd)
    
    #imagecd.write("imagecd.fits")
    #imagecddc.write("imagecddc.fits")
    
    # residual after forward-backward should be of order a^2 q qmax^2
    
    imageres = (imagecddc-image)/cimage
    #imageres.write("imageres.fits")
    
    maxres = imageres.array.max()
    minres = imageres.array.min()
    
    assert maxres<10, ("maximum positive residual of forward-backward transformation is too large")
    assert minres>-10, ("maximum negative residual of forward-backward transformation is too large")
    
def test_exampleimage():
    """Test application of model compared to an independent implementation that was run on example 
       image
    """
    
    #n, r0, t0, rx, tx, r, t, alpha
    cd = PowerLawCD(5, 2.e-7, 1.e-7, 1.25e-7, 1.25e-7, 0.75e-7, 0.5e-7, 0.3)
      # model used externally to bring cdtest1 to cdtest2

    image_orig  = galsim.fits.read("fits_files/cdtest1.fits") # unprocessed image
    image_proc  = galsim.fits.read("fits_files/cdtest2.fits") # image with cd model applied with other library
    
    image_plcd  = cd.applyForward(image_orig)
    
    #image_plcd.write("cdtest_intern.fits")
    #image_proc.write("cdtest_extern.fits")

    np.testing.assert_array_almost_equal(image_proc.array, image_plcd.array, 1,
                                   "externally and internally processed image not equal") 
                                   # I checked that the remaining differences are numerical noise

    
if __name__ == "__main__":
    test_simplegeometry()
    test_fluxconservation()
    test_forwardbackward()
    test_exampleimage()
