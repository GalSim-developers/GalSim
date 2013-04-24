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

def funcname():
    import inspect
    return inspect.stack()[1][3]

def gsobject_compare(obj1, obj2, conv=None):
    """Helper function to check that two GSObjects are equivalent
    """
    if conv:
        obj1 = galsim.Convolve([obj1,conv])
        obj2 = galsim.Convolve([obj2,conv])

    im1 = galsim.ImageD(16,16)
    im2 = galsim.ImageD(16,16)
    obj1.draw(dx=0.2, image=im1, normalization='sb')
    obj2.draw(dx=0.2, image=im2, normalization='sb')
    np.testing.assert_array_almost_equal(im1.array, im2.array, 10)


def test_gaussian():
    """Test various ways to build a Gaussian
    """
    import time
    t1 = time.time()

    config = {
        'gal1' : { 'type' : 'Gaussian' , 'sigma' : 2 },
        'gal2' : { 'type' : 'Gaussian' , 'fwhm' : 2, 'flux' : 100 },
        'gal3' : { 'type' : 'Gaussian' , 'half_light_radius' : 2, 'flux' : 1.e6,
                   'ellip' : { 'type' : 'QBeta' , 'q' : 0.6, 'beta' : 0.39 * galsim.radians } 
                 },
        'gal4' : { 'type' : 'Gaussian' , 'sigma' : 1, 'flux' : 50,
                   'dilate' : 3, 'ellip' : galsim.Shear(e1=0.3),
                   'rotate' : 12 * galsim.degrees, 
                   'magnify' : 1.03, 'shear' : galsim.Shear(g1=0.03, g2=-0.05),
                   'shift' : { 'type' : 'XY', 'x' : 0.7, 'y' : -1.2 }  
                 },
        'gal5' : { 'type' : 'Gaussian' , 'sigma' : 1.5, 'flux' : 72.5,
                   'rotate' : -34. * galsim.degrees,
                   'magnify' : 0.93,
                   'shear' : galsim.Shear(g1=-0.15, g2=0.2) 
                 },
    }

    gal1a = galsim.config.BuildGSObject(config, 'gal1')[0]
    gal1b = galsim.Gaussian(sigma = 2)
    gsobject_compare(gal1a, gal1b)

    gal2a = galsim.config.BuildGSObject(config, 'gal2')[0]
    gal2b = galsim.Gaussian(fwhm = 2, flux = 100)
    gsobject_compare(gal2a, gal2b)

    gal3a = galsim.config.BuildGSObject(config, 'gal3')[0]
    gal3b = galsim.Gaussian(half_light_radius = 2, flux = 1.e6)
    gal3b.applyShear(q = 0.6, beta = 0.39 * galsim.radians)
    gsobject_compare(gal3a, gal3b)

    gal4a = galsim.config.BuildGSObject(config, 'gal4')[0]
    gal4b = galsim.Gaussian(sigma = 1, flux = 50)
    gal4b.applyDilation(3)
    gal4b.applyShear(e1 = 0.3)
    gal4b.applyRotation(12 * galsim.degrees)
    gal4b.applyMagnification(1.03)
    gal4b.applyShear(g1 = 0.03, g2 = -0.05)
    gal4b.applyShift(dx = 0.7, dy = -1.2) 
    gsobject_compare(gal4a, gal4b)

    gal5a = galsim.config.BuildGSObject(config, 'gal5')[0]
    gal5b = galsim.Gaussian(sigma = 1.5, flux = 72.5)
    gal5b.applyRotation(-34 * galsim.degrees)
    gal5b.applyLensing(-0.15, 0.2, 0.93)
    gsobject_compare(gal5a, gal5b)

    t2 = time.time()
    print 'time for %s = %.2f'%(funcname(),t2-t1)

def test_moffat():
    """Test various ways to build a Moffat
    """
    import time
    t1 = time.time()

    config = {
        'gal1' : { 'type' : 'Moffat' , 'beta' : 1.4, 'scale_radius' : 2 },
        'gal2' : { 'type' : 'Moffat' , 'beta' : 3.5, 'fwhm' : 2, 'trunc' : 5, 'flux' : 100 },
        'gal3' : { 'type' : 'Moffat' , 'beta' : 2.2, 'half_light_radius' : 2, 'flux' : 1.e6,
                   'ellip' : { 'type' : 'QBeta' , 'q' : 0.6, 'beta' : 0.39 * galsim.radians } 
                 },
        'gal4' : { 'type' : 'Moffat' , 'beta' : 1.7, 'fwhm' : 1, 'flux' : 50,
                   'dilate' : 3, 'ellip' : galsim.Shear(e1=0.3),
                   'rotate' : 12 * galsim.degrees, 
                   'magnify' : 1.03, 'shear' : galsim.Shear(g1=0.03, g2=-0.05),
                   'shift' : { 'type' : 'XY', 'x' : 0.7, 'y' : -1.2 } 
                 },
        'gal5' : { 'type' : 'Moffat' , 'beta' : 2.8, 'flux' : 22, 'fwhm' : 0.3, 'trunc' : 0.7,
                   'shear' : galsim.Shear(g1=-0.15, g2=0.2),
                   'gsparams' : { 'maxk_threshold' : 1.e-2 }
                 },
    }

    gal1a = galsim.config.BuildGSObject(config, 'gal1')[0]
    gal1b = galsim.Moffat(beta = 1.4, scale_radius = 2)
    gsobject_compare(gal1a, gal1b)

    gal2a = galsim.config.BuildGSObject(config, 'gal2')[0]
    gal2b = galsim.Moffat(beta = 3.5, fwhm = 2, trunc = 5, flux = 100)
    gsobject_compare(gal2a, gal2b)

    gal3a = galsim.config.BuildGSObject(config, 'gal3')[0]
    gal3b = galsim.Moffat(beta = 2.2, half_light_radius = 2, flux = 1.e6)
    gal3b.applyShear(q = 0.6, beta = 0.39 * galsim.radians)
    gsobject_compare(gal3a, gal3b)

    gal4a = galsim.config.BuildGSObject(config, 'gal4')[0]
    gal4b = galsim.Moffat(beta = 1.7, fwhm = 1, flux = 50)
    gal4b.applyDilation(3)
    gal4b.applyShear(e1 = 0.3)
    gal4b.applyRotation(12 * galsim.degrees)
    gal4b.applyMagnification(1.03)
    gal4b.applyShear(g1 = 0.03, g2 = -0.05)
    gal4b.applyShift(dx = 0.7, dy = -1.2) 
    gsobject_compare(gal4a, gal4b)

    gal5a = galsim.config.BuildGSObject(config, 'gal5')[0]
    # Note: this needs to be rather small otherwise maxk_threshold is obviated by other 
    # adjustments we make to the parameters in SBProfile.cpp
    gsparams = galsim.GSParams(maxk_threshold=1.e-2)
    gal5b = galsim.Moffat(beta=2.8, fwhm=0.3, flux=22, trunc=0.7, gsparams=gsparams)
    gal5b.applyShear(g1=-0.15, g2=0.2)
    # convolve to test the k-space gsparams (with an even smaller profile)
    gsobject_compare(gal5a, gal5b, conv=galsim.Gaussian(sigma=0.01))

    try:
        # Make sure they don't match when using the default GSParams
        gal5c = galsim.Moffat(beta=2.8, fwhm=0.3, flux=22, trunc=0.7)
        gal5c.applyShear(g1=-0.15, g2=0.2)
        np.testing.assert_raises(AssertionError,gsobject_compare, gal5a, gal5c,
                                 conv=galsim.Gaussian(sigma=0.01))
    except ImportError:
        print 'The assert_raises tests require nose'

    t2 = time.time()
    print 'time for %s = %.2f'%(funcname(),t2-t1)

def test_airy():
    """Test various ways to build a Airy
    """
    import time
    t1 = time.time()

    config = {
        'gal1' : { 'type' : 'Airy' , 'lam_over_diam' : 2 },
        'gal2' : { 'type' : 'Airy' , 'lam_over_diam' : 0.4, 'obscuration' : 0.3, 'flux' : 100 },
        'gal3' : { 'type' : 'Airy' , 'lam_over_diam' : 1.3, 'obscuration' : 0, 'flux' : 1.e6,
                   'ellip' : { 'type' : 'QBeta' , 'q' : 0.6, 'beta' : 0.39 * galsim.radians } 
                 },
        'gal4' : { 'type' : 'Airy' , 'lam_over_diam' : 1, 'flux' : 50,
                   'dilate' : 3, 'ellip' : galsim.Shear(e1=0.3),
                   'rotate' : 12 * galsim.degrees, 
                   'magnify' : 1.03, 'shear' : galsim.Shear(g1=0.03, g2=-0.05),
                   'shift' : { 'type' : 'XY', 'x' : 0.7, 'y' : -1.2 } 
                 },
        'gal5' : { 'type' : 'Airy' , 'lam_over_diam' : 45, 
                   'gsparams' : { 'xvalue_accuracy' : 1.e-2 }
                 }
    }

    gal1a = galsim.config.BuildGSObject(config, 'gal1')[0]
    gal1b = galsim.Airy(lam_over_diam = 2)
    gsobject_compare(gal1a, gal1b)

    gal2a = galsim.config.BuildGSObject(config, 'gal2')[0]
    gal2b = galsim.Airy(lam_over_diam = 0.4, obscuration = 0.3, flux = 100)
    gsobject_compare(gal2a, gal2b)

    gal3a = galsim.config.BuildGSObject(config, 'gal3')[0]
    gal3b = galsim.Airy(lam_over_diam = 1.3, flux = 1.e6)
    gal3b.applyShear(q = 0.6, beta = 0.39 * galsim.radians)
    gsobject_compare(gal3a, gal3b)

    gal4a = galsim.config.BuildGSObject(config, 'gal4')[0]
    gal4b = galsim.Airy(lam_over_diam = 1, flux = 50)
    gal4b.applyDilation(3)
    gal4b.applyShear(e1 = 0.3)
    gal4b.applyRotation(12 * galsim.degrees)
    gal4b.applyMagnification(1.03)
    gal4b.applyShear(g1 = 0.03, g2 = -0.05)
    gal4b.applyShift(dx = 0.7, dy = -1.2) 
    gsobject_compare(gal4a, gal4b)

    # The approximation from xvalue_accuracy here happens at the core, so you need a very
    # large size to notice.  (Which tells me this isn't that useful an approximation, but 
    # so be it.)
    gal5a = galsim.config.BuildGSObject(config, 'gal5')[0]
    gsparams = galsim.GSParams(xvalue_accuracy=1.e-2)
    gal5b = galsim.Airy(lam_over_diam=45, gsparams=gsparams)
    gsobject_compare(gal5a, gal5b)

    try:
        # Make sure they don't match when using the default GSParams
        gal5c = galsim.Airy(lam_over_diam=45)
        np.testing.assert_raises(AssertionError,gsobject_compare, gal5a, gal5c)
    except ImportError:
        print 'The assert_raises tests require nose'


    t2 = time.time()
    print 'time for %s = %.2f'%(funcname(),t2-t1)

def test_kolmogorov():
    """Test various ways to build a Kolmogorov
    """
    import time
    t1 = time.time()

    config = {
        'gal1' : { 'type' : 'Kolmogorov' , 'lam_over_r0' : 2 },
        'gal2' : { 'type' : 'Kolmogorov' , 'fwhm' : 2, 'flux' : 100 },
        'gal3' : { 'type' : 'Kolmogorov' , 'half_light_radius' : 2, 'flux' : 1.e6,
                   'ellip' : { 'type' : 'QBeta' , 'q' : 0.6, 'beta' : 0.39 * galsim.radians } 
                 },
        'gal4' : { 'type' : 'Kolmogorov' , 'lam_over_r0' : 1, 'flux' : 50,
                   'dilate' : 3, 'ellip' : galsim.Shear(e1=0.3),
                   'rotate' : 12 * galsim.degrees, 
                   'magnify' : 1.03, 'shear' : galsim.Shear(g1=0.03, g2=-0.05),
                   'shift' : { 'type' : 'XY', 'x' : 0.7, 'y' : -1.2 } 
                 },
        'gal5' : { 'type' : 'Kolmogorov' , 'lam_over_r0' : 1, 'flux' : 50,
                   'gsparams' : { 'integration_relerr' : 1.e-2, 'integration_abserr' : 1.e-4 }
                 }
    }

    gal1a = galsim.config.BuildGSObject(config, 'gal1')[0]
    gal1b = galsim.Kolmogorov(lam_over_r0 = 2)
    gsobject_compare(gal1a, gal1b)

    gal2a = galsim.config.BuildGSObject(config, 'gal2')[0]
    gal2b = galsim.Kolmogorov(fwhm = 2, flux = 100)
    gsobject_compare(gal2a, gal2b)

    gal3a = galsim.config.BuildGSObject(config, 'gal3')[0]
    gal3b = galsim.Kolmogorov(half_light_radius = 2, flux = 1.e6)
    gal3b.applyShear(q = 0.6, beta = 0.39 * galsim.radians)
    gsobject_compare(gal3a, gal3b)

    gal4a = galsim.config.BuildGSObject(config, 'gal4')[0]
    gal4b = galsim.Kolmogorov(lam_over_r0 = 1, flux = 50)
    gal4b.applyDilation(3)
    gal4b.applyShear(e1 = 0.3)
    gal4b.applyRotation(12 * galsim.degrees)
    gal4b.applyLensing(0.03, -0.05, 1.03)
    gal4b.applyShift(dx = 0.7, dy = -1.2) 
    gsobject_compare(gal4a, gal4b)

    gal5a = galsim.config.BuildGSObject(config, 'gal5')[0]
    gsparams = galsim.GSParams(integration_relerr=1.e-2, integration_abserr=1.e-4)
    gal5b = galsim.Kolmogorov(lam_over_r0=1, flux=50, gsparams=gsparams)
    gsobject_compare(gal5a, gal5b)

    try:
        # Make sure they don't match when using the default GSParams
        gal5c = galsim.Kolmogorov(lam_over_r0=1, flux=50)
        np.testing.assert_raises(AssertionError,gsobject_compare, gal5a, gal5c)
    except ImportError:
        print 'The assert_raises tests require nose'


    t2 = time.time()
    print 'time for %s = %.2f'%(funcname(),t2-t1)

def test_opticalpsf():
    """Test various ways to build a OpticalPSF
    """
    import time
    t1 = time.time()

    config = {
        'gal1' : { 'type' : 'OpticalPSF' , 'lam_over_diam' : 2 },
        'gal2' : { 'type' : 'OpticalPSF' , 'lam_over_diam' : 2, 'flux' : 100,
                   'defocus' : 0.23, 'astig1' : -0.12, 'astig2' : 0.11,
                   'coma1' : -0.09, 'coma2' : 0.03, 'spher' : 0.19,
                   'pad_factor' : 1.0, 'oversampling' : 1.0 
                 },
        'gal3' : { 'type' : 'OpticalPSF' , 'lam_over_diam' : 2, 'flux' : 1.e6,
                   'defocus' : 0.23, 'astig1' : -0.12, 'astig2' : 0.11,
                   'circular_pupil' : False, 'obscuration' : 0.3,
                   'pad_factor' : 1.0, 'oversampling' : 1.0,
                   'ellip' : { 'type' : 'QBeta' , 'q' : 0.6, 'beta' : 0.39 * galsim.radians } 
                 },
        'gal4' : { 'type' : 'OpticalPSF' , 'lam_over_diam' : 1, 'flux' : 50,
                   'defocus' : 0.23, 'astig1' : -0.12, 'astig2' : 0.11,
                   'coma1' : -0.09, 'coma2' : 0.03, 'spher' : 0.19,
                   'circular_pupil' : True, 'obscuration' : 0.2,
                   'pad_factor' : 1.0, 'oversampling' : 1.0,
                   'dilate' : 3, 'ellip' : galsim.Shear(e1=0.3),
                   'rotate' : 12 * galsim.degrees, 
                   'magnify' : 1.03, 'shear' : galsim.Shear(g1=0.03, g2=-0.05),
                   'shift' : { 'type' : 'XY', 'x' : 0.7, 'y' : -1.2 } 
                 }
    }

    gal1a = galsim.config.BuildGSObject(config, 'gal1')[0]
    gal1b = galsim.OpticalPSF(lam_over_diam = 2)
    gsobject_compare(gal1a, gal1b)

    gal2a = galsim.config.BuildGSObject(config, 'gal2')[0]
    gal2b = galsim.OpticalPSF(lam_over_diam = 2, flux = 100,
                              defocus = 0.23, astig1 = -0.12, astig2 = 0.11,
                              coma1 = -0.09, coma2 = 0.03, spher = 0.19,
                              pad_factor = 1, oversampling = 1)
    gsobject_compare(gal2a, gal2b)

    gal3a = galsim.config.BuildGSObject(config, 'gal3')[0]
    gal3b = galsim.OpticalPSF(lam_over_diam = 2, flux = 1.e6, 
                              defocus = 0.23, astig1 = -0.12, astig2 = 0.11,
                              circular_pupil = False, obscuration = 0.3,
                              pad_factor = 1, oversampling = 1)
    gal3b.applyShear(q = 0.6, beta = 0.39 * galsim.radians)
    gsobject_compare(gal3a, gal3b)

    gal4a = galsim.config.BuildGSObject(config, 'gal4')[0]
    gal4b = galsim.OpticalPSF(lam_over_diam = 1, flux = 50,
                              defocus = 0.23, astig1 = -0.12, astig2 = 0.11,
                              coma1 = -0.09, coma2 = 0.03, spher = 0.19,
                              circular_pupil = True, obscuration = 0.2,
                              pad_factor = 1, oversampling = 1)
    gal4b.applyDilation(3)
    gal4b.applyShear(e1 = 0.3)
    gal4b.applyRotation(12 * galsim.degrees)
    gal4b.applyMagnification(1.03)
    gal4b.applyShear(g1 = 0.03, g2 = -0.05)
    gal4b.applyShift(dx = 0.7, dy = -1.2) 
    gsobject_compare(gal4a, gal4b)


    t2 = time.time()
    print 'time for %s = %.2f'%(funcname(),t2-t1)

def test_exponential():
    """Test various ways to build a Exponential
    """
    import time
    t1 = time.time()

    config = {
        'gal1' : { 'type' : 'Exponential' , 'scale_radius' : 2 },
        'gal2' : { 'type' : 'Exponential' , 'scale_radius' : 1.7, 'flux' : 100 },
        'gal3' : { 'type' : 'Exponential' , 'half_light_radius' : 2, 'flux' : 1.e6,
                   'ellip' : { 'type' : 'QBeta' , 'q' : 0.6, 'beta' : 0.39 * galsim.radians } 
                 },
        'gal4' : { 'type' : 'Exponential' , 'scale_radius' : 1, 'flux' : 50,
                   'dilate' : 3, 'ellip' : galsim.Shear(e1=0.3),
                   'rotate' : 12 * galsim.degrees, 
                   'magnify' : 1.03, 'shear' : galsim.Shear(g1=0.03, g2=-0.05),
                   'shift' : { 'type' : 'XY', 'x' : 0.7, 'y' : -1.2 } 
                 },
        'gal5' : { 'type' : 'Exponential' , 'scale_radius' : 1, 'flux' : 50,
                   'gsparams' : { 'kvalue_accuracy' : 1.e-2 }
                 }
    }

    gal1a = galsim.config.BuildGSObject(config, 'gal1')[0]
    gal1b = galsim.Exponential(scale_radius = 2)
    gsobject_compare(gal1a, gal1b)

    gal2a = galsim.config.BuildGSObject(config, 'gal2')[0]
    gal2b = galsim.Exponential(scale_radius = 1.7, flux = 100)
    gsobject_compare(gal2a, gal2b)

    gal3a = galsim.config.BuildGSObject(config, 'gal3')[0]
    gal3b = galsim.Exponential(half_light_radius = 2, flux = 1.e6)
    gal3b.applyShear(q = 0.6, beta = 0.39 * galsim.radians)
    gsobject_compare(gal3a, gal3b)

    gal4a = galsim.config.BuildGSObject(config, 'gal4')[0]
    gal4b = galsim.Exponential(scale_radius = 1, flux = 50)
    gal4b.applyDilation(3)
    gal4b.applyShear(e1 = 0.3)
    gal4b.applyRotation(12 * galsim.degrees)
    gal4b.applyMagnification(1.03)
    gal4b.applyShear(g1 = 0.03, g2 = -0.05)
    gal4b.applyShift(dx = 0.7, dy = -1.2) 
    gsobject_compare(gal4a, gal4b)

    gal5a = galsim.config.BuildGSObject(config, 'gal5')[0]
    gsparams = galsim.GSParams(kvalue_accuracy=1.e-2)
    gal5b = galsim.Exponential(scale_radius=1, flux=50, gsparams=gsparams)
    gsobject_compare(gal5a, gal5b, conv=galsim.Gaussian(sigma=1))

    try:
        # Make sure they don't match when using the default GSParams
        gal5c = galsim.Exponential(scale_radius=1, flux=50)
        np.testing.assert_raises(AssertionError,gsobject_compare, gal5a, gal5c, 
                                 conv=galsim.Gaussian(sigma=1))
    except ImportError:
        print 'The assert_raises tests require nose'


    t2 = time.time()
    print 'time for %s = %.2f'%(funcname(),t2-t1)

def test_sersic():
    """Test various ways to build a Sersic
    """
    import time
    t1 = time.time()

    config = {
        'gal1' : { 'type' : 'Sersic' , 'n' : 1.2,  'half_light_radius' : 2 },
        'gal2' : { 'type' : 'Sersic' , 'n' : 3.5,  'half_light_radius' : 1.7, 'flux' : 100 },
        'gal3' : { 'type' : 'Sersic' , 'n' : 2.2,  'half_light_radius' : 3.5, 'flux' : 1.e6,
                   'ellip' : { 'type' : 'QBeta' , 'q' : 0.6, 'beta' : 0.39 * galsim.radians } 
                 },
        'gal4' : { 'type' : 'Sersic' , 'n' : 0.7,  'half_light_radius' : 1, 'flux' : 50,
                   'dilate' : 3, 'ellip' : galsim.Shear(e1=0.3),
                   'rotate' : 12 * galsim.degrees, 
                   'magnify' : 1.03, 'shear' : galsim.Shear(g1=0.03, g2=-0.05),
                   'shift' : { 'type' : 'XY', 'x' : 0.7, 'y' : -1.2 } 
                 },
        'gal5' : { 'type' : 'Sersic' , 'n' : 0.7,  'half_light_radius' : 1, 'flux' : 50,
                   'gsparams' : { 'minimum_fft_size' : 256 }
                 },
        'gal6' : { 'type' : 'Sersic' , 'n' : 0.7,  'half_light_radius' : 1, 'flux' : 50,
                   'gsparams' : { 'maximum_fft_size' : 64 }
                 }

    }

    gal1a = galsim.config.BuildGSObject(config, 'gal1')[0]
    gal1b = galsim.Sersic(n = 1.2, half_light_radius = 2)
    gsobject_compare(gal1a, gal1b)

    gal2a = galsim.config.BuildGSObject(config, 'gal2')[0]
    gal2b = galsim.Sersic(n = 3.5, half_light_radius = 1.7, flux = 100)
    gsobject_compare(gal2a, gal2b)

    gal3a = galsim.config.BuildGSObject(config, 'gal3')[0]
    gal3b = galsim.Sersic(n = 2.2, half_light_radius = 3.5, flux = 1.e6)
    gal3b.applyShear(q = 0.6, beta = 0.39 * galsim.radians)
    gsobject_compare(gal3a, gal3b)

    gal4a = galsim.config.BuildGSObject(config, 'gal4')[0]
    gal4b = galsim.Sersic(n = 0.7, half_light_radius = 1, flux = 50)
    gal4b.applyDilation(3)
    gal4b.applyShear(e1 = 0.3)
    gal4b.applyRotation(12 * galsim.degrees)
    gal4b.applyLensing(0.03, -0.05, 1.03)
    gal4b.applyShift(dx = 0.7, dy = -1.2) 
    gsobject_compare(gal4a, gal4b)

    gal5a = galsim.config.BuildGSObject(config, 'gal5')[0]
    gsparams = galsim.GSParams(minimum_fft_size=256)
    gal5b = galsim.Sersic(n=0.7, half_light_radius=1, flux=50, gsparams=gsparams)
    gsobject_compare(gal5a, gal5b, conv=galsim.Gaussian(sigma=1))

    try:
        # Make sure they don't match when using the default GSParams
        gal5c = galsim.Sersic(n=0.7, half_light_radius=1, flux=50)
        np.testing.assert_raises(AssertionError,gsobject_compare, gal5a, gal5c, 
                                 conv=galsim.Gaussian(sigma=1))

        # For the maximum_fft_size test, we need to do things a little differently
        # We lower maximum_fft_size below the size that SBProfile wants this to be, 
        # and we check to make sure an exception is thrown.  Of course, this isn't how you
        # would normally use maximum_fft_size.  Normally, you would raise it when the default
        # is too small.  But to construct the test that way would require a lot of memory
        # and would be rather slow.
        gal6a = galsim.config.BuildGSObject(config, 'gal6')[0]
        gal6b = galsim.Sersic(n=0.7, half_light_radius=1, flux=50)
        np.testing.assert_raises(RuntimeError,gsobject_compare, gal6a, gal6b, 
                                 conv=galsim.Gaussian(sigma=1))

    except ImportError:
        print 'The assert_raises tests require nose'


    t2 = time.time()
    print 'time for %s = %.2f'%(funcname(),t2-t1)

def test_devaucouleurs():
    """Test various ways to build a DeVaucouleurs
    """
    import time
    t1 = time.time()

    config = {
        'gal1' : { 'type' : 'DeVaucouleurs' , 'half_light_radius' : 2 },
        'gal2' : { 'type' : 'DeVaucouleurs' , 'half_light_radius' : 1.7, 'flux' : 100 },
        'gal3' : { 'type' : 'DeVaucouleurs' , 'half_light_radius' : 3.5, 'flux' : 1.e6,
                   'ellip' : { 'type' : 'QBeta' , 'q' : 0.6, 'beta' : 0.39 * galsim.radians } 
                 },
        'gal4' : { 'type' : 'DeVaucouleurs' , 'half_light_radius' : 1, 'flux' : 50,
                   'dilate' : 3, 'ellip' : galsim.Shear(e1=0.3),
                   'rotate' : 12 * galsim.degrees, 
                   'magnify' : 1.03, 'shear' : galsim.Shear(g1=0.03, g2=-0.05),
                   'shift' : { 'type' : 'XY', 'x' : 0.7, 'y' : -1.2 } 
                 },
        'gal5' : { 'type' : 'DeVaucouleurs' , 'half_light_radius' : 1, 'flux' : 50,
                   'gsparams' : { 'alias_threshold' : 1.e-4 }
                 }
    }

    gal1a = galsim.config.BuildGSObject(config, 'gal1')[0]
    gal1b = galsim.DeVaucouleurs(half_light_radius = 2)
    gsobject_compare(gal1a, gal1b)

    gal2a = galsim.config.BuildGSObject(config, 'gal2')[0]
    gal2b = galsim.DeVaucouleurs(half_light_radius = 1.7, flux = 100)
    gsobject_compare(gal2a, gal2b)

    gal3a = galsim.config.BuildGSObject(config, 'gal3')[0]
    gal3b = galsim.DeVaucouleurs(half_light_radius = 3.5, flux = 1.e6)
    gal3b.applyShear(q = 0.6, beta = 0.39 * galsim.radians)
    gsobject_compare(gal3a, gal3b)

    gal4a = galsim.config.BuildGSObject(config, 'gal4')[0]
    gal4b = galsim.DeVaucouleurs(half_light_radius = 1, flux = 50)
    gal4b.applyDilation(3)
    gal4b.applyShear(e1 = 0.3)
    gal4b.applyRotation(12 * galsim.degrees)
    gal4b.applyMagnification(1.03)
    gal4b.applyShear(g1 = 0.03, g2 = -0.05)
    gal4b.applyShift(dx = 0.7, dy = -1.2) 
    gsobject_compare(gal4a, gal4b)

    gal5a = galsim.config.BuildGSObject(config, 'gal5')[0]
    gsparams = galsim.GSParams(alias_threshold=1.e-4)
    gal5b = galsim.DeVaucouleurs(half_light_radius=1, flux=50, gsparams=gsparams)
    gsobject_compare(gal5a, gal5b, conv=galsim.Gaussian(sigma=1))

    try:
        # Make sure they don't match when using the default GSParams
        gal5c = galsim.DeVaucouleurs(half_light_radius=1, flux=50)
        np.testing.assert_raises(AssertionError,gsobject_compare, gal5a, gal5c,
                                 conv=galsim.Gaussian(sigma=1))
    except ImportError:
        print 'The assert_raises tests require nose'



    t2 = time.time()
    print 'time for %s = %.2f'%(funcname(),t2-t1)

def test_pixel():
    """Test various ways to build a Pixel
    """
    import time
    t1 = time.time()

    config = {
        'gal1' : { 'type' : 'Pixel' , 'xw' : 2 },
        'gal2' : { 'type' : 'Pixel' , 'xw' : 1.7, 'yw' : 1.7, 'flux' : 100 },
        'gal3' : { 'type' : 'Pixel' , 'xw' : 2, 'yw' : 2.1, 'flux' : 1.e6,
                   'ellip' : { 'type' : 'QBeta' , 'q' : 0.6, 'beta' : 0.39 * galsim.radians } 
                 },
        'gal4' : { 'type' : 'Pixel' , 'xw' : 1, 'yw' : 1.2, 'flux' : 50,
                   'dilate' : 3, 'ellip' : galsim.Shear(e1=0.3),
                   'rotate' : 12 * galsim.degrees, 
                   'magnify' : 1.03, 'shear' : galsim.Shear(g1=0.03, g2=-0.05),
                   'shift' : { 'type' : 'XY', 'x' : 0.7, 'y' : -1.2 } 
                 },
        'gal5' : { 'type' : 'Pixel' , 'xw' : 2, 'flux' : 50,
                   'gsparams' : { 'realspace_relerr' : 1.e-2 , 'realspace_abserr' : 1.e-4 } 
                 }
    }

    gal1a = galsim.config.BuildGSObject(config, 'gal1')[0]
    gal1b = galsim.Pixel(xw = 2)
    gsobject_compare(gal1a, gal1b)

    gal2a = galsim.config.BuildGSObject(config, 'gal2')[0]
    gal2b = galsim.Pixel(xw = 1.7, yw = 1.7, flux = 100)
    gsobject_compare(gal2a, gal2b)

    # The config stuff emits a warning about the rectangular pixel.
    # We suppress that here, since we're doing it on purpose.
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        gal3a = galsim.config.BuildGSObject(config, 'gal3')[0]
        gal3b = galsim.Pixel(xw = 2, yw = 2.1, flux = 1.e6)
        gal3b.applyShear(q = 0.6, beta = 0.39 * galsim.radians)
        # Drawing sheared Pixel without convolution doesn't work, so we need to 
        # do the extra convolution by a Gaussian here 
        gsobject_compare(gal3a, gal3b, conv=galsim.Gaussian(0.1))

        gal4a = galsim.config.BuildGSObject(config, 'gal4')[0]
        gal4b = galsim.Pixel(xw = 1, yw = 1.2, flux = 50)
        gal4b.applyDilation(3)
        gal4b.applyShear(e1 = 0.3)
        gal4b.applyRotation(12 * galsim.degrees)
        gal4b.applyMagnification(1.03)
        gal4b.applyShear(g1 = 0.03, g2 = -0.05)
        gal4b.applyShift(dx = 0.7, dy = -1.2) 
        gsobject_compare(gal4a, gal4b, conv=galsim.Gaussian(0.1))

    gal5a = galsim.config.BuildGSObject(config, 'gal5')[0]
    gsparams = galsim.GSParams(realspace_relerr=1.e-2, realspace_abserr=1.e-4)
    gal5b = galsim.Pixel(xw=2, flux=50, gsparams=gsparams)
    # Convolution of a pixel with a truncated Moffat will use realspace convolution
    conv = galsim.Moffat(beta=2.8, fwhm=1.3, trunc=3.7)
    gsobject_compare(gal5a, gal5b, conv=conv)

    try:
        # Make sure they don't match when using the default GSParams
        gal5c = galsim.Pixel(xw=2, flux=50)
        np.testing.assert_raises(AssertionError,gsobject_compare, gal5a, gal5c, conv=conv)
    except ImportError:
        print 'The assert_raises tests require nose'

    t2 = time.time()
    print 'time for %s = %.2f'%(funcname(),t2-t1)

def test_realgalaxy():
    """Test various ways to build a RealGalaxy
    """
    import time
    t1 = time.time()

    # I don't want to gratuitously copy the real_catalog catalog, so use the 
    # version in the examples directory.
    real_gal_dir = os.path.join('..','examples','data')
    real_gal_cat = os.path.join(real_gal_dir,'real_galaxy_catalog_example.fits')
    config = {
        'input' : { 'real_catalog' : 
                        { 'image_dir' : real_gal_dir , 
                          'file_name' : real_gal_cat ,
                          'preload' : True } 
                  },

        'gal1' : { 'type' : 'RealGalaxy' },
        'gal2' : { 'type' : 'RealGalaxy' , 'index' : 23, 'flux' : 100 },
        'gal3' : { 'type' : 'RealGalaxy' , 'id' : 103176, 'flux' : 1.e6,
                   'ellip' : { 'type' : 'QBeta' , 'q' : 0.6, 'beta' : 0.39 * galsim.radians } 
                 },
        'gal4' : { 'type' : 'RealGalaxy' , 'index' : 5, 'flux' : 50,
                   'dilate' : 3, 'ellip' : galsim.Shear(e1=0.3),
                   'rotate' : 12 * galsim.degrees, 
                   'magnify' : 1.03, 'shear' : galsim.Shear(g1=0.03, g2=-0.05),
                   'shift' : { 'type' : 'XY', 'x' : 0.7, 'y' : -1.2 } 
                 },
        'gal5' : { 'type' : 'RealGalaxy' , 'index' : 41, 'noise_pad' : 'True' },
        'gal6' : { 'type' : 'RealGalaxy' , 'index' : 41, 'noise_pad' : 'blankimg.fits' },
        'gal7' : { 'type' : 'RealGalaxy' , 'index' : 32, 'pad_image' : 'blankimg.fits' },
        'gal8' : { 'type' : 'RealGalaxy' ,
                   'index' : 87, 'pad_image' : 'blankimg.fits', 'noise_pad' : 'blankimg.fits'
                 }
    }
    rng = galsim.UniformDeviate(1234)
    config['rng'] = galsim.UniformDeviate(1234) # A second copy starting with the same seed.

    galsim.config.ProcessInput(config)

    real_cat = galsim.RealGalaxyCatalog(
        image_dir=real_gal_dir, file_name=real_gal_cat, preload=True)

    # For these profiles, we convolve by a gaussian to smooth out the profile.
    # This makes the comparison much faster without changing the validity of the test.
    conv = galsim.Gaussian(sigma = 1)

    config['seq_index'] = 0
    gal1a = galsim.config.BuildGSObject(config, 'gal1')[0]
    gal1b = galsim.RealGalaxy(real_cat, index=0)
    # The convolution here 
    gsobject_compare(gal1a, gal1b, conv=conv)

    config['seq_index'] = 1
    gal2a = galsim.config.BuildGSObject(config, 'gal2')[0]
    gal2b = galsim.RealGalaxy(real_cat, index = 23)
    gal2b.setFlux(100)
    gsobject_compare(gal2a, gal2b, conv=conv)

    config['seq_index'] = 2
    gal3a = galsim.config.BuildGSObject(config, 'gal3')[0]
    gal3b = galsim.RealGalaxy(real_cat, index = 17)
    gal3b.setFlux(1.e6)
    gal3b.applyShear(q = 0.6, beta = 0.39 * galsim.radians)
    gsobject_compare(gal3a, gal3b, conv=conv)

    config['seq_index'] = 3
    gal4a = galsim.config.BuildGSObject(config, 'gal4')[0]
    gal4b = galsim.RealGalaxy(real_cat, index = 5)
    gal4b.setFlux(50)
    gal4b.applyDilation(3)
    gal4b.applyShear(e1 = 0.3)
    gal4b.applyRotation(12 * galsim.degrees)
    gal4b.applyMagnification(1.03)
    gal4b.applyShear(g1 = 0.03, g2 = -0.05)
    gal4b.applyShift(dx = 0.7, dy = -1.2) 
    gsobject_compare(gal4a, gal4b, conv=conv)

    config['seq_index'] = 4
    gal5a = galsim.config.BuildGSObject(config, 'gal5')[0]
    gal5b = galsim.RealGalaxy(real_cat, index = 41, rng = rng, noise_pad = True)
    gsobject_compare(gal5a, gal5b, conv=conv)

    config['seq_index'] = 5
    gal6a = galsim.config.BuildGSObject(config, 'gal6')[0]
    gal6b = galsim.RealGalaxy(real_cat, index = 41, rng = rng, noise_pad = 'blankimg.fits')
    gsobject_compare(gal6a, gal6b, conv=conv)

    config['seq_index'] = 6
    gal7a = galsim.config.BuildGSObject(config, 'gal7')[0]
    gal7b = galsim.RealGalaxy(real_cat, index = 32, pad_image = 'blankimg.fits')
    gsobject_compare(gal7a, gal7b, conv=conv)
    
    config['seq_index'] = 7
    gal8a = galsim.config.BuildGSObject(config, 'gal8')[0]
    gal8b = galsim.RealGalaxy(real_cat, index = 87, rng = rng, noise_pad = 'blankimg.fits',
                              pad_image = 'blankimg.fits')
    gsobject_compare(gal8a, gal8b, conv=conv)

    t2 = time.time()
    print 'time for %s = %.2f'%(funcname(),t2-t1)


def test_interpolated_image():
    """Test various ways to build an InterpolatedImage
    """
    import time
    t1 = time.time()

    imgdir = 'SBProfile_comparison_images'
    file_name = os.path.join(imgdir,'gauss_smallshear.fits')
    config = {
        'gal1' : { 'type' : 'InterpolatedImage', 'image' : file_name },
        'gal2' : { 'type' : 'InterpolatedImage', 'image' : file_name,
                   'x_interpolant' : 'linear' },
        'gal3' : { 'type' : 'InterpolatedImage', 'image' : file_name,
                   'x_interpolant' : 'cubic', 'normalization' : 'sb', 'flux' : 1.e4 
                 },
        'gal4' : { 'type' : 'InterpolatedImage', 'image' : file_name,
                   'x_interpolant' : 'lanczos5', 'dx' : 0.7, 'flux' : 1.e5 
                 },
        'gal5' : { 'type' : 'InterpolatedImage', 'image' : file_name,
                   'noise_pad' : 0.001 
                 },
        'gal6' : { 'type' : 'InterpolatedImage', 'image' : file_name,
                   'noise_pad' : 'blankimg.fits' 
                 },
        'gal7' : { 'type' : 'InterpolatedImage', 'image' : file_name,
                   'pad_image' : 'blankimg.fits' 
                 }
    }
    rng = galsim.UniformDeviate(1234)
    config['rng'] = galsim.UniformDeviate(1234) # A second copy starting with the same seed.

    gal1a = galsim.config.BuildGSObject(config, 'gal1')[0]
    im = galsim.fits.read(file_name)
    gal1b = galsim.InterpolatedImage(im)
    gsobject_compare(gal1a, gal1b)

    gal2a = galsim.config.BuildGSObject(config, 'gal2')[0]
    interp = galsim.InterpolantXY(galsim.Linear())
    gal2b = galsim.InterpolatedImage(im, x_interpolant=interp)
    gsobject_compare(gal2a, gal2b)

    gal3a = galsim.config.BuildGSObject(config, 'gal3')[0]
    interp = galsim.InterpolantXY(galsim.Cubic())
    gal3b = galsim.InterpolatedImage(im, x_interpolant=interp, normalization='surface brightness')
    gal3b.setFlux(1.e4)
    gsobject_compare(gal3a, gal3b)

    gal4a = galsim.config.BuildGSObject(config, 'gal4')[0]
    interp = galsim.InterpolantXY(galsim.Lanczos(n=5,conserve_flux=True))
    gal4b = galsim.InterpolatedImage(im, x_interpolant=interp, dx=0.7)
    gal4b.setFlux(1.e5)
    gsobject_compare(gal4a, gal4b)

    gal5a = galsim.config.BuildGSObject(config, 'gal5')[0]
    gal5b = galsim.InterpolatedImage(im, rng=rng, noise_pad=0.001)
    gsobject_compare(gal5a, gal5b)

    gal6a = galsim.config.BuildGSObject(config, 'gal6')[0]
    gal6b = galsim.InterpolatedImage(im, rng=rng, noise_pad='blankimg.fits')
    gsobject_compare(gal6a, gal6b)

    gal7a = galsim.config.BuildGSObject(config, 'gal7')[0]
    gal7b = galsim.InterpolatedImage(im, pad_image = 'blankimg.fits')
    gsobject_compare(gal7a, gal7b)

    t2 = time.time()
    print 'time for %s = %.2f'%(funcname(),t2-t1)


def test_add():
    """Test various ways to build a Add
    """
    import time
    t1 = time.time()

    config = {
        'gal1' : { 
            'type' : 'Add' , 
            'items' : [
                { 'type' : 'Gaussian' , 'sigma' : 2 },
                { 'type' : 'Exponential' , 'half_light_radius' : 2.3 } 
            ] 
        },
        'gal2' : { 
            'type' : 'Sum' ,
            'items' : [
                { 'type' : 'Gaussian' , 'half_light_radius' : 2 , 'flux' : 30 },
                { 'type' : 'Sersic' , 'n' : 2.5 , 'half_light_radius' : 1.7 , 'flux' : 15 },
                { 'type' : 'Exponential' , 'half_light_radius' : 2.3 , 'flux' : 60 } 
            ] 
        },
        'gal3' : { 
            'type' : 'Add' ,
            'items' : [
                { 'type' : 'Sersic' , 'n' : 3.4 , 'half_light_radius' : 1.1, 
                  'flux' : 0.3 , 'ellip' : galsim.Shear(e1=0.2,e2=0.3),
                  'shift' : { 'type' : 'XY' , 'x' : 0.4 , 'y' : 0.9 } 
                },
                { 'type' : 'Sersic' , 'n' : 1.1 , 'half_light_radius' : 2.5, 'flux' : 0.7 } 
            ],
            'flux' : 1.e6,
            'ellip' : { 'type' : 'QBeta' , 'q' : 0.6, 'beta' : 0.39 * galsim.radians } 
        },
        'gal4' : { 
            'type' : 'Add' , 
            'items' : [
                { 'type' : 'Gaussian' , 'half_light_radius' : 2 , 'flux' : 8 },
                { 'type' : 'Exponential' , 'half_light_radius' : 2.3 , 'flux' : 2 } 
            ],
            'flux' : 50,
            'dilate' : 3, 'ellip' : galsim.Shear(e1=0.3),
            'rotate' : 12 * galsim.degrees, 
            'magnify' : 1.03, 'shear' : galsim.Shear(g1=0.03, g2=-0.05),
            'shift' : { 'type' : 'XY' , 'x' : 0.7 , 'y' : -1.2 } 
        },
        'gal5' : { 
            'type' : 'Add',
            'items' : [
                { 'type' : 'Exponential' , 'scale_radius' : 1.7, 'flux' : 100 },
                { 'type' : 'Gaussian' , 'sigma' : 1, 'flux' : 50 }
            ],
            'gsparams' : { 'maxk_threshold' : 1.e-2, 'alias_threshold' : 1.e-2 }
        }
    }

    gal1a = galsim.config.BuildGSObject(config, 'gal1')[0]
    gal1b_1 = galsim.Gaussian(sigma = 2)
    gal1b_2 = galsim.Exponential(half_light_radius = 2.3)
    gal1b = galsim.Add([gal1b_1, gal1b_2])
    gsobject_compare(gal1a, gal1b)

    gal2a = galsim.config.BuildGSObject(config, 'gal2')[0]
    gal2b_1 = galsim.Gaussian(half_light_radius = 2, flux = 30)
    gal2b_2 = galsim.Sersic(n = 2.5, half_light_radius = 1.7, flux = 15)
    gal2b_3 = galsim.Exponential(half_light_radius = 2.3, flux = 60)
    gal2b = galsim.Add([gal2b_1, gal2b_2, gal2b_3])
    gsobject_compare(gal2a, gal2b)

    gal3a = galsim.config.BuildGSObject(config, 'gal3')[0]
    gal3b_1 = galsim.Sersic(n = 3.4, half_light_radius = 1.1, flux = 0.3)
    gal3b_1.applyShear(e1=0.2, e2=0.3)
    gal3b_1.applyShift(0.4,0.9)
    gal3b_2 = galsim.Sersic(n = 1.1, half_light_radius = 2.5, flux = 0.7)
    gal3b = galsim.Add([gal3b_1, gal3b_2])
    gal3b.setFlux(1.e6)
    gal3b.applyShear(q = 0.6, beta = 0.39 * galsim.radians)
    gsobject_compare(gal3a, gal3b)

    gal4a = galsim.config.BuildGSObject(config, 'gal4')[0]
    gal4b_1 = galsim.Gaussian(half_light_radius = 2, flux = 8)
    gal4b_2 = galsim.Exponential(half_light_radius = 2.3, flux = 2)
    gal4b = galsim.Add([gal4b_1, gal4b_2])
    gal4b.setFlux(50)
    gal4b.applyDilation(3)
    gal4b.applyShear(e1 = 0.3)
    gal4b.applyRotation(12 * galsim.degrees)
    gal4b.applyMagnification(1.03)
    gal4b.applyShear(g1 = 0.03, g2 = -0.05)
    gal4b.applyShift(dx = 0.7, dy = -1.2) 
    gsobject_compare(gal4a, gal4b)

    # Check that the Add items correctly inherit their gsparams from the top level
    gal5a = galsim.config.BuildGSObject(config, 'gal5')[0]
    gsparams = galsim.GSParams(maxk_threshold=1.e-2, alias_threshold=1.e-2)
    gal5b_1 = galsim.Exponential(scale_radius=1.7, flux=100, gsparams=gsparams)
    gal5b_2 = galsim.Gaussian(sigma=1, flux=50, gsparams=gsparams)
    gal5b = galsim.Add([gal5b_1, gal5b_2])
    gsobject_compare(gal5a, gal5b, conv=galsim.Gaussian(sigma=1))

    try:
        # Make sure they don't match when using the default GSParams
        gal5c_1 = galsim.Exponential(scale_radius=1.7, flux=100)
        gal5c_2 = galsim.Gaussian(sigma=1, flux=50)
        gal5c = galsim.Add([gal5c_1, gal5c_2])
        np.testing.assert_raises(AssertionError,gsobject_compare, gal5a, gal5c,
                                 conv=galsim.Gaussian(sigma=1))
    except ImportError:
        print 'The assert_raises tests require nose'


    t2 = time.time()
    print 'time for %s = %.2f'%(funcname(),t2-t1)


def test_convolve():
    """Test various ways to build a Convolve
    """
    import time
    t1 = time.time()

    config = {
        'gal1' : { 
            'type' : 'Convolve' , 
            'items' : [
                { 'type' : 'Gaussian' , 'sigma' : 2 },
                { 'type' : 'Exponential' , 'half_light_radius' : 2.3 } 
            ] 
        },
        'gal2' : { 
            'type' : 'Convolution' ,
            'items' : [
                { 'type' : 'Gaussian' , 'half_light_radius' : 2 , 'flux' : 30 },
                { 'type' : 'Sersic' , 'n' : 2.5 , 'half_light_radius' : 1.7 , 'flux' : 15 },
                { 'type' : 'Exponential' , 'half_light_radius' : 2.3 , 'flux' : 60 } 
            ] 
        },
        'gal3' : { 
            'type' : 'Convolve' ,
            'items' : [
                { 'type' : 'Sersic' , 'n' : 3.4 , 'half_light_radius' : 1.1, 
                  'flux' : 0.3 , 'ellip' : galsim.Shear(e1=0.2,e2=0.3),
                  'shift' : { 'type' : 'XY' , 'x' : 0.4 , 'y' : 0.9 }
                },
                { 'type' : 'Sersic' , 'n' : 1.1 , 'half_light_radius' : 2.5, 
                  'flux' : 0.7 
                } 
            ],
            'flux' : 1.e6,
            'ellip' : { 'type' : 'QBeta' , 'q' : 0.6, 'beta' : 0.39 * galsim.radians } 
        },
        'gal4' : { 
            'type' : 'Convolve' , 
            'items' : [
                { 'type' : 'Gaussian' , 'half_light_radius' : 2 , 'flux' : 8 },
                { 'type' : 'Exponential' , 'half_light_radius' : 2.3 , 'flux' : 2 } 
            ],
            'flux' : 50,
            'dilate' : 3, 'ellip' : galsim.Shear(e1=0.3),
            'rotate' : 12 * galsim.degrees, 
            'magnify' : 1.03, 'shear' : galsim.Shear(g1=0.03, g2=-0.05),
            'shift' : { 'type' : 'XY' , 'x' : 0.7 , 'y' : -1.2 } 
        },
        'gal5' : { 
            'type' : 'Convolution' ,
            'items' : [
                { 'type' : 'Exponential' , 'scale_radius' : 1.7, 'flux' : 100 },
                { 'type' : 'Gaussian' , 'sigma' : 1 }
            ],
            'gsparams' : { 'maxk_threshold' : 1.e-2, 'alias_threshold' : 1.e-2 }
        }
    }

    gal1a = galsim.config.BuildGSObject(config, 'gal1')[0]
    gal1b_1 = galsim.Gaussian(sigma = 2)
    gal1b_2 = galsim.Exponential(half_light_radius = 2.3)
    gal1b = galsim.Convolve([gal1b_1, gal1b_2])
    gsobject_compare(gal1a, gal1b)

    gal2a = galsim.config.BuildGSObject(config, 'gal2')[0]
    gal2b_1 = galsim.Gaussian(half_light_radius = 2, flux = 30)
    gal2b_2 = galsim.Sersic(n = 2.5, half_light_radius = 1.7, flux = 15)
    gal2b_3 = galsim.Exponential(half_light_radius = 2.3, flux = 60)
    gal2b = galsim.Convolve([gal2b_1, gal2b_2, gal2b_3])
    gsobject_compare(gal2a, gal2b)

    gal3a = galsim.config.BuildGSObject(config, 'gal3')[0]
    gal3b_1 = galsim.Sersic(n = 3.4, half_light_radius = 1.1, flux = 0.3)
    gal3b_1.applyShear(e1=0.2, e2=0.3)
    gal3b_1.applyShift(0.4,0.9)
    gal3b_2 = galsim.Sersic(n = 1.1, half_light_radius = 2.5, flux = 0.7)
    gal3b = galsim.Convolve([gal3b_1, gal3b_2])
    gal3b.setFlux(1.e6)
    gal3b.applyShear(q = 0.6, beta = 0.39 * galsim.radians)
    gsobject_compare(gal3a, gal3b)

    gal4a = galsim.config.BuildGSObject(config, 'gal4')[0]
    gal4b_1 = galsim.Gaussian(half_light_radius = 2, flux = 8)
    gal4b_2 = galsim.Exponential(half_light_radius = 2.3, flux = 2)
    gal4b = galsim.Convolve([gal4b_1, gal4b_2])
    gal4b.setFlux(50)
    gal4b.applyDilation(3)
    gal4b.applyShear(e1 = 0.3)
    gal4b.applyRotation(12 * galsim.degrees)
    gal4b.applyMagnification(1.03)
    gal4b.applyShear(g1 = 0.03, g2 = -0.05)
    gal4b.applyShift(dx = 0.7, dy = -1.2) 
    gsobject_compare(gal4a, gal4b)

    # Check that the Convolve items correctly inherit their gsparams from the top level
    gal5a = galsim.config.BuildGSObject(config, 'gal5')[0]
    gsparams = galsim.GSParams(maxk_threshold=1.e-2, alias_threshold=1.e-2)
    gal5b_1 = galsim.Exponential(scale_radius=1.7, flux=100, gsparams=gsparams)
    gal5b_2 = galsim.Gaussian(sigma=1, gsparams=gsparams)
    gal5b = galsim.Convolve([gal5b_1, gal5b_2])
    gsobject_compare(gal5a, gal5b)

    try:
        # Make sure they don't match when using the default GSParams
        gal5c_1 = galsim.Exponential(scale_radius=1.7, flux=100)
        gal5c_2 = galsim.Gaussian(sigma=1)
        gal5c = galsim.Convolve([gal5c_1, gal5c_2])
        np.testing.assert_raises(AssertionError,gsobject_compare, gal5a, gal5c)
    except ImportError:
        print 'The assert_raises tests require nose'


    t2 = time.time()
    print 'time for %s = %.2f'%(funcname(),t2-t1)


def test_list():
    """Test building a GSObject from a list:
    """
    import time
    t1 = time.time()

    config = {
        'gal' : { 
            'type' : 'List' ,
            'items' : [
                { 'type' : 'Gaussian' , 'sigma' : 2 },
                { 'type' : 'Gaussian' , 'fwhm' : 2, 'flux' : 100 },
                { 'type' : 'Gaussian' , 'half_light_radius' : 2, 'flux' : 1.e6,
                  'ellip' : { 'type' : 'QBeta' , 'q' : 0.6, 'beta' : 0.39 * galsim.radians } 
                },
                { 'type' : 'Gaussian' , 'sigma' : 1, 'flux' : 50,
                  'dilate' : 3, 'ellip' : galsim.Shear(e1=0.3),
                  'rotate' : 12 * galsim.degrees, 
                  'magnify' : 1.03, 'shear' : galsim.Shear(g1=0.03, g2=-0.05),
                  'shift' : { 'type' : 'XY', 'x' : 0.7, 'y' : -1.2 } 
                }
            ]
        }
    }

    config['seq_index'] = 0
    gal1a = galsim.config.BuildGSObject(config, 'gal')[0]
    gal1b = galsim.Gaussian(sigma = 2)
    gsobject_compare(gal1a, gal1b)

    config['seq_index'] = 1
    gal2a = galsim.config.BuildGSObject(config, 'gal')[0]
    gal2b = galsim.Gaussian(fwhm = 2, flux = 100)
    gsobject_compare(gal2a, gal2b)

    config['seq_index'] = 2
    gal3a = galsim.config.BuildGSObject(config, 'gal')[0]
    gal3b = galsim.Gaussian(half_light_radius = 2, flux = 1.e6)
    gal3b.applyShear(q = 0.6, beta = 0.39 * galsim.radians)
    gsobject_compare(gal3a, gal3b)

    config['seq_index'] = 3
    gal4a = galsim.config.BuildGSObject(config, 'gal')[0]
    gal4b = galsim.Gaussian(sigma = 1, flux = 50)
    gal4b.applyDilation(3)
    gal4b.applyShear(e1 = 0.3)
    gal4b.applyRotation(12 * galsim.degrees)
    gal4b.applyMagnification(1.03)
    gal4b.applyShear(g1 = 0.03, g2 = -0.05)
    gal4b.applyShift(dx = 0.7, dy = -1.2) 
    gsobject_compare(gal4a, gal4b)

    # Check that the list items correctly inherit their gsparams from the top level
    config = {
        'gal' : { 
            'type' : 'List' ,
            'items' : [
                { 'type' : 'Exponential' , 'scale_radius' : 1.7, 'flux' : 100 },
                { 'type' : 'Exponential' , 'scale_radius' : 3, 'flux' : 10 }
            ],
            'gsparams' : { 'maxk_threshold' : 1.e-2, 'alias_threshold' : 1.e-2 }
        }
    }

    config['seq_index'] = 0
    gal5a = galsim.config.BuildGSObject(config, 'gal')[0]
    gsparams = galsim.GSParams(maxk_threshold=1.e-2, alias_threshold=1.e-2)
    gal5b = galsim.Exponential(scale_radius=1.7, flux=100, gsparams=gsparams)
    gsobject_compare(gal5a, gal5b, conv=galsim.Gaussian(sigma=1))

    try:
        # Make sure they don't match when using the default GSParams
        gal5c = galsim.Exponential(scale_radius=1.7, flux=100)
        np.testing.assert_raises(AssertionError,gsobject_compare, gal5a, gal5c,
                                 conv=galsim.Gaussian(sigma=1))
    except ImportError:
        print 'The assert_raises tests require nose'

    t2 = time.time()
    print 'time for %s = %.2f'%(funcname(),t2-t1)

def test_ring():
    """Test building a GSObject from a ring test:
    """
    import time
    t1 = time.time()

    config = {
        'gal' : { 
            'type' : 'Ring' ,
            'num' : 2,
            'first' : { 
                'type' : 'Gaussian' ,
                'sigma' : 2 , 
                'ellip' : {
                    'type' : 'E1E2',
                    'e1' : { 'type' : 'List' ,
                             'items' : [ 0.3, 0.2, 0.8 ],
                             'index' : { 'type' : 'Sequence', 'repeat' : 2 } 
                           },
                    'e2' : 0.1
                }
            }
        }
    }

    gauss = galsim.Gaussian(sigma=2)
    e1_list = [ 0.3, -0.3, 0.2, -0.2, 0.8, -0.8 ]
    e2_list = [ 0.1, -0.1, 0.1, -0.1, 0.1, -0.1 ]

    for k in range(6):
        config['seq_index'] = k
        gal1a = galsim.config.BuildGSObject(config, 'gal')[0]
        gal1b = gauss.createSheared(e1=e1_list[k], e2=e2_list[k])
        gsobject_compare(gal1a, gal1b)

    config = {
        'gal' : {
            'type' : 'Ring' ,
            'num' : 10,
            'first' : { 'type' : 'Exponential', 'half_light_radius' : 2,
                        'ellip' : galsim.Shear(e2=0.3) 
                      },
        }
    }

    disk = galsim.Exponential(half_light_radius=2)
    disk.applyShear(e2=0.3)

    for k in range(25):
        config['seq_index'] = k
        gal2a = galsim.config.BuildGSObject(config, 'gal')[0]
        gal2b = disk.createRotated(theta = k * 18 * galsim.degrees)
        gsobject_compare(gal2a, gal2b)

    config = {
        'gal' : {
            'type' : 'Ring' ,
            'num' : 20,
            'full_rotation' : 360. * galsim.degrees,
            'first' : { 
                'type' : 'Sum',
                'items' : [
                    { 'type' : 'Exponential', 'half_light_radius' : 2,
                      'ellip' : galsim.Shear(e2=0.3) 
                    },
                    { 'type' : 'Sersic', 'n' : 3, 'half_light_radius' : 1.3, 
                      'ellip' : galsim.Shear(e1=0.12,e2=-0.08) 
                    } 
                ]
            }
        }
    }

    disk = galsim.Exponential(half_light_radius=2)
    disk.applyShear(e2=0.3)
    bulge = galsim.Sersic(n=3, half_light_radius=1.3)
    bulge.applyShear(e1=0.12,e2=-0.08)
    sum = disk + bulge

    for k in range(25):
        config['seq_index'] = k
        gal3a = galsim.config.BuildGSObject(config, 'gal')[0]
        gal3b = sum.createRotated(theta = k * 18 * galsim.degrees)
        gsobject_compare(gal3a, gal3b)

    # Check that the ring items correctly inherit their gsparams from the top level
    config = {
        'gal' : {
            'type' : 'Ring' ,
            'num' : 20,
            'full_rotation' : 360. * galsim.degrees,
            'first' : { 
                'type' : 'Sum',
                'items' : [
                    { 'type' : 'Exponential', 'half_light_radius' : 2,
                      'ellip' : galsim.Shear(e2=0.3) 
                    },
                    { 'type' : 'Sersic', 'n' : 3, 'half_light_radius' : 1.3, 
                      'ellip' : galsim.Shear(e1=0.12,e2=-0.08) 
                    } 
                ]
            },
            'gsparams' : { 'maxk_threshold' : 1.e-2, 'alias_threshold' : 1.e-2 }
        }
    }

    config['seq_index'] = 0
    gal4a = galsim.config.BuildGSObject(config, 'gal')[0]
    gsparams = galsim.GSParams(maxk_threshold=1.e-2, alias_threshold=1.e-2)
    disk = galsim.Exponential(half_light_radius=2, gsparams=gsparams)
    disk.applyShear(e2=0.3)
    bulge = galsim.Sersic(n=3,half_light_radius=1.3, gsparams=gsparams)
    bulge.applyShear(e1=0.12,e2=-0.08)
    gal4b = disk + bulge
    gsobject_compare(gal4a, gal4b, conv=galsim.Gaussian(sigma=1))

    try:
        # Make sure they don't match when using the default GSParams
        disk = galsim.Exponential(half_light_radius=2)
        disk.applyShear(e2=0.3)
        bulge = galsim.Sersic(n=3,half_light_radius=1.3)
        bulge.applyShear(e1=0.12,e2=-0.08)
        gal4c = disk + bulge
        np.testing.assert_raises(AssertionError,gsobject_compare, gal4a, gal4c,
                                 conv=galsim.Gaussian(sigma=1))
    except ImportError:
        print 'The assert_raises tests require nose'


    t2 = time.time()
    print 'time for %s = %.2f'%(funcname(),t2-t1)


if __name__ == "__main__":
    test_gaussian()
    test_moffat()
    test_airy()
    test_kolmogorov()
    test_opticalpsf()
    test_exponential()
    test_sersic()
    test_devaucouleurs()
    test_pixel()
    test_realgalaxy()
    test_interpolated_image()
    test_add()
    test_convolve()
    test_list()
    test_ring()


