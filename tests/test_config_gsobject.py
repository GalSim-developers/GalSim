from __future__ import print_function
# Copyright (c) 2012-2015 by the GalSim developers team on GitHub
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
except ImportError:
    path, filename = os.path.split(__file__)
    sys.path.append(os.path.abspath(os.path.join(path, "..")))
    import galsim


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
    gal3b = gal3b.shear(q = 0.6, beta = 0.39 * galsim.radians)
    gsobject_compare(gal3a, gal3b)

    gal4a = galsim.config.BuildGSObject(config, 'gal4')[0]
    gal4b = galsim.Gaussian(sigma = 1, flux = 50)
    gal4b = gal4b.dilate(3).shear(e1 = 0.3).rotate(12 * galsim.degrees).magnify(1.03)
    gal4b = gal4b.shear(g1 = 0.03, g2 = -0.05).shift(dx = 0.7, dy = -1.2) 
    gsobject_compare(gal4a, gal4b)

    gal5a = galsim.config.BuildGSObject(config, 'gal5')[0]
    gal5b = galsim.Gaussian(sigma = 1.5, flux = 72.5)
    gal5b = gal5b.rotate(-34 * galsim.degrees).lens(-0.15, 0.2, 0.93)
    gsobject_compare(gal5a, gal5b)

    t2 = time.time()
    print('time for %s = %.2f'%(funcname(),t2-t1))


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
    gal3b = gal3b.shear(q = 0.6, beta = 0.39 * galsim.radians)
    gsobject_compare(gal3a, gal3b)

    gal4a = galsim.config.BuildGSObject(config, 'gal4')[0]
    gal4b = galsim.Moffat(beta = 1.7, fwhm = 1, flux = 50)
    gal4b = gal4b.dilate(3).shear(e1 = 0.3).rotate(12 * galsim.degrees).magnify(1.03)
    gal4b = gal4b.shear(g1 = 0.03, g2 = -0.05).shift(dx = 0.7, dy = -1.2) 
    gsobject_compare(gal4a, gal4b)

    gal5a = galsim.config.BuildGSObject(config, 'gal5')[0]
    # Note: this needs to be rather small otherwise maxk_threshold is obviated by other 
    # adjustments we make to the parameters in SBProfile.cpp
    gsparams = galsim.GSParams(maxk_threshold=1.e-2)
    gal5b = galsim.Moffat(beta=2.8, fwhm=0.3, flux=22, trunc=0.7, gsparams=gsparams)
    gal5b = gal5b.shear(g1=-0.15, g2=0.2)
    # convolve to test the k-space gsparams (with an even smaller profile)
    gsobject_compare(gal5a, gal5b, conv=galsim.Gaussian(sigma=0.01))

    try:
        # Make sure they don't match when using the default GSParams
        gal5c = galsim.Moffat(beta=2.8, fwhm=0.3, flux=22, trunc=0.7)
        gal5c = gal5c.shear(g1=-0.15, g2=0.2)
        np.testing.assert_raises(AssertionError,gsobject_compare, gal5a, gal5c,
                                 conv=galsim.Gaussian(sigma=0.01))
    except ImportError:
        print('The assert_raises tests require nose')

    t2 = time.time()
    print('time for %s = %.2f'%(funcname(),t2-t1))


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
                 },
        'gal6' : { 'type' : 'Airy' , 'lam' : 400., 'diam' : 4.0, 'scale_unit' : 'arcmin' }
    }

    gal1a = galsim.config.BuildGSObject(config, 'gal1')[0]
    gal1b = galsim.Airy(lam_over_diam = 2)
    gsobject_compare(gal1a, gal1b)

    gal2a = galsim.config.BuildGSObject(config, 'gal2')[0]
    gal2b = galsim.Airy(lam_over_diam = 0.4, obscuration = 0.3, flux = 100)
    gsobject_compare(gal2a, gal2b)

    gal3a = galsim.config.BuildGSObject(config, 'gal3')[0]
    gal3b = galsim.Airy(lam_over_diam = 1.3, flux = 1.e6)
    gal3b = gal3b.shear(q = 0.6, beta = 0.39 * galsim.radians)
    gsobject_compare(gal3a, gal3b)

    gal4a = galsim.config.BuildGSObject(config, 'gal4')[0]
    gal4b = galsim.Airy(lam_over_diam = 1, flux = 50)
    gal4b = gal4b.dilate(3).shear(e1 = 0.3).rotate(12 * galsim.degrees).magnify(1.03)
    gal4b = gal4b.shear(g1 = 0.03, g2 = -0.05).shift(dx = 0.7, dy = -1.2) 
    gsobject_compare(gal4a, gal4b)

    # The approximation from xvalue_accuracy here happens at the core, so you need a very
    # large size to notice.  (Which tells me this isn't that useful an approximation, but 
    # so be it.)
    gal5a = galsim.config.BuildGSObject(config, 'gal5')[0]
    gsparams = galsim.GSParams(xvalue_accuracy=1.e-2)
    gal5b = galsim.Airy(lam_over_diam=45, gsparams=gsparams)
    gsobject_compare(gal5a, gal5b)

    gal6a = galsim.config.BuildGSObject(config, 'gal6')[0]
    gal6b = galsim.Airy(lam=400., diam=4., scale_unit=galsim.arcmin)
    gsobject_compare(gal6a, gal6b)

    try:
        # Make sure they don't match when using the default GSParams
        gal5c = galsim.Airy(lam_over_diam=45)
        np.testing.assert_raises(AssertionError,gsobject_compare, gal5a, gal5c)
    except ImportError:
        print('The assert_raises tests require nose')

    t2 = time.time()
    print('time for %s = %.2f'%(funcname(),t2-t1))


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
    gal3b = gal3b.shear(q = 0.6, beta = 0.39 * galsim.radians)
    gsobject_compare(gal3a, gal3b)

    gal4a = galsim.config.BuildGSObject(config, 'gal4')[0]
    gal4b = galsim.Kolmogorov(lam_over_r0 = 1, flux = 50)
    gal4b = gal4b.dilate(3).shear(e1 = 0.3).rotate(12 * galsim.degrees)
    gal4b = gal4b.lens(0.03, -0.05, 1.03).shift(dx = 0.7, dy = -1.2) 
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
        print('The assert_raises tests require nose')

    t2 = time.time()
    print('time for %s = %.2f'%(funcname(),t2-t1))


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
                   'pad_factor' : 1.4, 'oversampling' : 1.2 
                 },
        'gal3' : { 'type' : 'OpticalPSF' , 'lam_over_diam' : 2, 'flux' : 1.e6,
                   'defocus' : 0.23, 'astig1' : -0.12, 'astig2' : 0.11,
                   'circular_pupil' : False, 'obscuration' : 0.3,
                   'ellip' : { 'type' : 'QBeta' , 'q' : 0.6, 'beta' : 0.39 * galsim.radians } 
                 },
        'gal4' : { 'type' : 'OpticalPSF' , 'lam_over_diam' : 0.05, 'flux' : 50,
                   'defocus' : 0.03, 'astig1' : -0.04, 'astig2' : 0.07,
                   'coma1' : -0.09, 'coma2' : 0.03, 'spher' : -0.09,
                   'circular_pupil' : True, 'obscuration' : 0.2,
                   'dilate' : 3, 'ellip' : galsim.Shear(e1=0.3),
                   'rotate' : 12 * galsim.degrees, 
                   'magnify' : 1.03, 'shear' : galsim.Shear(g1=0.03, g2=-0.05),
                   'shift' : { 'type' : 'XY', 'x' : 0.7, 'y' : -1.2 } 
                 },
        'gal5' : { 'type': 'OpticalPSF' , 'lam_over_diam' : 0.12, 'flux' : 1.8,
                   'defocus' : 0.1, 'obscuration' : 0.18,
                   'pupil_plane_im' : \
                       os.path.join(".","Optics_comparison_images","sample_pupil_rolled.fits"),
                   'pupil_angle' : 27.*galsim.degrees },
        'gal6' : {'type' : 'OpticalPSF' , 'lam' : 874.0, 'diam' : 7.4, 'flux' : 70.,
                  'obscuration' : 0.1 }
    }

    gal1a = galsim.config.BuildGSObject(config, 'gal1')[0]
    gal1b = galsim.OpticalPSF(lam_over_diam = 2)
    gsobject_compare(gal1a, gal1b)

    gal2a = galsim.config.BuildGSObject(config, 'gal2')[0]
    gal2b = galsim.OpticalPSF(lam_over_diam = 2, flux = 100,
                              defocus = 0.23, astig1 = -0.12, astig2 = 0.11,
                              coma1 = -0.09, coma2 = 0.03, spher = 0.19,
                              pad_factor = 1.4, oversampling = 1.2)
    gsobject_compare(gal2a, gal2b)

    gal3a = galsim.config.BuildGSObject(config, 'gal3')[0]
    gal3b = galsim.OpticalPSF(lam_over_diam = 2, flux = 1.e6, 
                              defocus = 0.23, astig1 = -0.12, astig2 = 0.11,
                              circular_pupil = False, obscuration = 0.3)
    gal3b = gal3b.shear(q = 0.6, beta = 0.39 * galsim.radians)
    gsobject_compare(gal3a, gal3b)

    gal4a = galsim.config.BuildGSObject(config, 'gal4')[0]
    gal4b = galsim.OpticalPSF(lam_over_diam = 0.05, flux = 50,
                              defocus = 0.03, astig1 = -0.04, astig2 = 0.07,
                              coma1 = -0.09, coma2 = 0.03, spher = -0.09,
                              circular_pupil = True, obscuration = 0.2)
    gal4b = gal4b.dilate(3).shear(e1 = 0.3).rotate(12 * galsim.degrees).magnify(1.03)
    gal4b = gal4b.shear(g1 = 0.03, g2 = -0.05).shift(dx = 0.7, dy = -1.2) 
    gsobject_compare(gal4a, gal4b)

    gal5a = galsim.config.BuildGSObject(config, 'gal5')[0]
    gal5b = galsim.OpticalPSF(
        lam_over_diam=0.12, flux=1.8, defocus=0.1, obscuration=0.18,
        pupil_plane_im=os.path.join(".","Optics_comparison_images","sample_pupil_rolled.fits"),
        pupil_angle=27.*galsim.degrees)
    gsobject_compare(gal5a, gal5b)

    gal6a = galsim.config.BuildGSObject(config, 'gal6')[0]
    gal6b = galsim.OpticalPSF(lam=874., diam=7.4, flux=70., obscuration=0.1)
    gsobject_compare(gal6a, gal6b)

    t2 = time.time()
    print('time for %s = %.2f'%(funcname(),t2-t1))


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
    gal3b = gal3b.shear(q = 0.6, beta = 0.39 * galsim.radians)
    gsobject_compare(gal3a, gal3b)

    gal4a = galsim.config.BuildGSObject(config, 'gal4')[0]
    gal4b = galsim.Exponential(scale_radius = 1, flux = 50)
    gal4b = gal4b.dilate(3).shear(e1 = 0.3).rotate(12 * galsim.degrees).magnify(1.03)
    gal4b = gal4b.shear(g1 = 0.03, g2 = -0.05).shift(dx = 0.7, dy = -1.2) 
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
        print('The assert_raises tests require nose')

    t2 = time.time()
    print('time for %s = %.2f'%(funcname(),t2-t1))


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
                 },
        'gal7' : { 'type' : 'Sersic' , 'n' : 3.2,  'half_light_radius' : 1.7, 'flux' : 50,
                   'trunc' : 4.3,
                   'gsparams' : { 'realspace_relerr' : 1.e-2 , 'realspace_abserr' : 1.e-4 }
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
    gal3b = gal3b.shear(q = 0.6, beta = 0.39 * galsim.radians)
    gsobject_compare(gal3a, gal3b)

    gal4a = galsim.config.BuildGSObject(config, 'gal4')[0]
    gal4b = galsim.Sersic(n = 0.7, half_light_radius = 1, flux = 50)
    gal4b = gal4b.dilate(3).shear(e1 = 0.3).rotate(12 * galsim.degrees)
    gal4b = gal4b.lens(0.03, -0.05, 1.03).shift(dx = 0.7, dy = -1.2) 
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
        print('The assert_raises tests require nose')

    gal7a = galsim.config.BuildGSObject(config, 'gal7')[0]
    gsparams = galsim.GSParams(realspace_relerr=1.e-2, realspace_abserr=1.e-4)
    gal7b = galsim.Sersic(n=3.2, half_light_radius=1.7, flux=50, trunc=4.3, gsparams=gsparams)
    # Convolution with a truncated Moffat will use realspace convolution
    conv = galsim.Moffat(beta=2.8, fwhm=1.3, trunc=3.7)
    gsobject_compare(gal7a, gal7b, conv=conv)

    try:
        # Make sure they don't match when using the default GSParams
        gal7c = galsim.Sersic(n=3.2, half_light_radius=1.7, flux=50, trunc=4.3)
        np.testing.assert_raises(AssertionError,gsobject_compare, gal7a, gal7c, conv=conv)
    except ImportError:
        print('The assert_raises tests require nose')


    t2 = time.time()
    print('time for %s = %.2f'%(funcname(),t2-t1))


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
                   'gsparams' : { 'folding_threshold' : 1.e-4 }
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
    gal3b = gal3b.shear(q = 0.6, beta = 0.39 * galsim.radians)
    gsobject_compare(gal3a, gal3b)

    gal4a = galsim.config.BuildGSObject(config, 'gal4')[0]
    gal4b = galsim.DeVaucouleurs(half_light_radius = 1, flux = 50)
    gal4b = gal4b.dilate(3).shear(e1 = 0.3).rotate(12 * galsim.degrees).magnify(1.03)
    gal4b = gal4b.shear(g1 = 0.03, g2 = -0.05).shift(dx = 0.7, dy = -1.2) 
    gsobject_compare(gal4a, gal4b)

    gal5a = galsim.config.BuildGSObject(config, 'gal5')[0]
    gsparams = galsim.GSParams(folding_threshold=1.e-4)
    gal5b = galsim.DeVaucouleurs(half_light_radius=1, flux=50, gsparams=gsparams)
    gsobject_compare(gal5a, gal5b, conv=galsim.Gaussian(sigma=1))

    try:
        # Make sure they don't match when using the default GSParams
        gal5c = galsim.DeVaucouleurs(half_light_radius=1, flux=50)
        np.testing.assert_raises(AssertionError,gsobject_compare, gal5a, gal5c,
                                 conv=galsim.Gaussian(sigma=1))
    except ImportError:
        print('The assert_raises tests require nose')

    t2 = time.time()
    print('time for %s = %.2f'%(funcname(),t2-t1))


def test_pixel():
    """Test various ways to build a Pixel
    """
    import time
    t1 = time.time()

    config = {
        'gal1' : { 'type' : 'Pixel' , 'scale' : 2 },
        'gal2' : { 'type' : 'Pixel' , 'scale' : 1.7, 'flux' : 100 },
        'gal3' : { 'type' : 'Box' , 'width' : 2, 'height' : 2.1, 'flux' : 1.e6,
                   'ellip' : { 'type' : 'QBeta' , 'q' : 0.6, 'beta' : 0.39 * galsim.radians } 
                 },
        'gal4' : { 'type' : 'Box' , 'width' : 1, 'height' : 1.2, 'flux' : 50,
                   'dilate' : 3, 'ellip' : galsim.Shear(e1=0.3),
                   'rotate' : 12 * galsim.degrees, 
                   'magnify' : 1.03, 'shear' : galsim.Shear(g1=0.03, g2=-0.05),
                   'shift' : { 'type' : 'XY', 'x' : 0.7, 'y' : -1.2 } 
                 },
    }

    gal1a = galsim.config.BuildGSObject(config, 'gal1')[0]
    gal1b = galsim.Pixel(scale = 2)
    gsobject_compare(gal1a, gal1b)

    gal2a = galsim.config.BuildGSObject(config, 'gal2')[0]
    gal2b = galsim.Pixel(scale = 1.7, flux = 100)
    gsobject_compare(gal2a, gal2b)

    # The config stuff emits a warning about the rectangular pixel.
    # We suppress that here, since we're doing it on purpose.
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        gal3a = galsim.config.BuildGSObject(config, 'gal3')[0]
        gal3b = galsim.Box(width = 2, height = 2.1, flux = 1.e6)
        gal3b = gal3b.shear(q = 0.6, beta = 0.39 * galsim.radians)
        # Drawing sheared Pixel without convolution doesn't work, so we need to 
        # do the extra convolution by a Gaussian here 
        gsobject_compare(gal3a, gal3b, conv=galsim.Gaussian(0.1))

        gal4a = galsim.config.BuildGSObject(config, 'gal4')[0]
        gal4b = galsim.Box(width = 1, height = 1.2, flux = 50)
        gal4b = gal4b.dilate(3).shear(e1 = 0.3).rotate(12 * galsim.degrees).magnify(1.03)
        gal4b = gal4b.shear(g1 = 0.03, g2 = -0.05).shift(dx = 0.7, dy = -1.2) 
        gsobject_compare(gal4a, gal4b, conv=galsim.Gaussian(0.1))

    t2 = time.time()
    print('time for %s = %.2f'%(funcname(),t2-t1))


def test_realgalaxy():
    """Test various ways to build a RealGalaxy
    """
    import time
    t1 = time.time()

    # I don't want to gratuitously copy the real_catalog catalog, so use the 
    # version in the examples directory.
    real_gal_dir = os.path.join('..','examples','data')
    real_gal_cat = 'real_galaxy_catalog_example.fits'
    config = {
        'input' : { 'real_catalog' : 
                        { 'dir' : real_gal_dir , 
                          'file_name' : real_gal_cat ,
                          'preload' : True } 
                  },

        'gal1' : { 'type' : 'RealGalaxy' },
        'gal2' : { 'type' : 'RealGalaxy' , 'index' : 23, 'flux' : 100 },
        'gal3' : { 'type' : 'RealGalaxy' , 'id' : 103176, 'flux' : 1.e6,
                   'ellip' : { 'type' : 'QBeta' , 'q' : 0.6, 'beta' : 0.39 * galsim.radians } 
                 },
        'gal4' : { 'type' : 'RealGalaxy' , 'index' : 5, 'scale_flux' : 50,
                   'dilate' : 3, 'ellip' : galsim.Shear(e1=0.3),
                   'rotate' : 12 * galsim.degrees, 
                   'magnify' : 1.03, 'shear' : galsim.Shear(g1=0.03, g2=-0.05),
                   'shift' : { 'type' : 'XY', 'x' : 0.7, 'y' : -1.2 } 
                 },
        'gal5' : { 'type' : 'RealGalaxy' , 'index' : 23, 'noise_pad_size' : 10 }
    }
    rng = galsim.UniformDeviate(1234)
    config['rng'] = galsim.UniformDeviate(1234) # A second copy starting with the same seed.

    galsim.config.ProcessInput(config)

    real_cat = galsim.RealGalaxyCatalog(
        dir=real_gal_dir, file_name=real_gal_cat, preload=True)

    # For these profiles, we convolve by a gaussian to smooth out the profile.
    # This makes the comparison much faster without changing the validity of the test.
    conv = galsim.Gaussian(sigma = 1)

    config['obj_num'] = 0
    gal1a = galsim.config.BuildGSObject(config, 'gal1')[0]
    gal1b = galsim.RealGalaxy(real_cat, index=0)
    # The convolution here 
    gsobject_compare(gal1a, gal1b, conv=conv)

    config['obj_num'] = 1
    gal2a = galsim.config.BuildGSObject(config, 'gal2')[0]
    gal2b = galsim.RealGalaxy(real_cat, index = 23, flux=100)
    gsobject_compare(gal2a, gal2b, conv=conv)

    config['obj_num'] = 2
    gal3a = galsim.config.BuildGSObject(config, 'gal3')[0]
    gal3b = galsim.RealGalaxy(real_cat, index = 17, flux=1.e6)
    gal3b = gal3b.shear(q = 0.6, beta = 0.39 * galsim.radians)
    gsobject_compare(gal3a, gal3b, conv=conv)

    config['obj_num'] = 3
    gal4a = galsim.config.BuildGSObject(config, 'gal4')[0]
    gal4b = galsim.RealGalaxy(real_cat, index = 5)
    gal4b *= 50
    gal4b = gal4b.dilate(3).shear(e1 = 0.3).rotate(12 * galsim.degrees).magnify(1.03)
    gal4b = gal4b.shear(g1 = 0.03, g2 = -0.05).shift(dx = 0.7, dy = -1.2) 
    gsobject_compare(gal4a, gal4b, conv=conv)

    config['obj_num'] = 4
    gal5a = galsim.config.BuildGSObject(config, 'gal5')[0]
    gal5b = galsim.RealGalaxy(real_cat, index = 23, rng = rng, noise_pad_size = 10)
    gsobject_compare(gal5a, gal5b, conv=conv)
    # Also check that the noise attribute is correct.
    gsobject_compare(gal5a.noise._profile, gal5b.noise._profile, conv=conv)

    t2 = time.time()
    print('time for %s = %.2f'%(funcname(),t2-t1))


def test_interpolated_image():
    """Test various ways to build an InterpolatedImage
    """
    import time
    t1 = time.time()

    imgdir = 'SBProfile_comparison_images'
    file_name = os.path.join(imgdir,'gauss_smallshear.fits')
    imgdir2 = 'fits_files'
    file_name2 = os.path.join(imgdir2,'interpim_hdu_test.fits')
    config = {
        'gal1' : { 'type' : 'InterpolatedImage', 'image' : file_name },
        'gal2' : { 'type' : 'InterpolatedImage', 'image' : file_name,
                   'x_interpolant' : 'linear' },
        'gal3' : { 'type' : 'InterpolatedImage', 'image' : file_name,
                   'x_interpolant' : 'cubic', 'normalization' : 'sb', 'flux' : 1.e4 
                 },
        'gal4' : { 'type' : 'InterpolatedImage', 'image' : file_name,
                   'x_interpolant' : 'lanczos5', 'scale' : 0.7, 'flux' : 1.e5 
                 },
        'gal5' : { 'type' : 'InterpolatedImage', 'image' : file_name,
                   'noise_pad' : 0.001 
                 },
        'gal6' : { 'type' : 'InterpolatedImage', 'image' : file_name,
                   'noise_pad' : 'fits_files/blankimg.fits' 
                 },
        'gal7' : { 'type' : 'InterpolatedImage', 'image' : file_name,
                   'pad_image' : 'fits_files/blankimg.fits' 
                 },
        'galmulti' : { 'type' : 'InterpolatedImage', 'image' : file_name2,
                       'hdu' : 2 }
    }
    rng = galsim.UniformDeviate(1234)
    config['rng'] = galsim.UniformDeviate(1234) # A second copy starting with the same seed.

    gal1a = galsim.config.BuildGSObject(config, 'gal1')[0]
    im = galsim.fits.read(file_name)
    gal1b = galsim.InterpolatedImage(im)
    gsobject_compare(gal1a, gal1b)

    gal2a = galsim.config.BuildGSObject(config, 'gal2')[0]
    gal2b = galsim.InterpolatedImage(im, x_interpolant=galsim.Linear())
    gsobject_compare(gal2a, gal2b)

    gal3a = galsim.config.BuildGSObject(config, 'gal3')[0]
    gal3b = galsim.InterpolatedImage(im, x_interpolant=galsim.Cubic(), normalization='sb',
                                     flux=1.e4)
    gsobject_compare(gal3a, gal3b)

    gal4a = galsim.config.BuildGSObject(config, 'gal4')[0]
    interp = galsim.Lanczos(n=5, conserve_dc=True)
    gal4b = galsim.InterpolatedImage(im, x_interpolant=interp, scale=0.7, flux=1.e5)
    gsobject_compare(gal4a, gal4b)

    gal5a = galsim.config.BuildGSObject(config, 'gal5')[0]
    gal5b = galsim.InterpolatedImage(im, rng=rng, noise_pad=0.001)
    gsobject_compare(gal5a, gal5b)

    gal6a = galsim.config.BuildGSObject(config, 'gal6')[0]
    gal6b = galsim.InterpolatedImage(im, rng=rng, noise_pad='fits_files/blankimg.fits')
    gsobject_compare(gal6a, gal6b)

    gal7a = galsim.config.BuildGSObject(config, 'gal7')[0]
    gal7b = galsim.InterpolatedImage(im, pad_image = 'fits_files/blankimg.fits')
    gsobject_compare(gal7a, gal7b)

    # Now test the reading from some particular HDU
    galmulti = galsim.config.BuildGSObject(config, 'galmulti')[0]
    im = galmulti.drawImage(scale=0.2, method='no_pixel')
    test_g2 = im.FindAdaptiveMom().observed_shape.g2
    np.testing.assert_almost_equal(
        test_g2, 0.7, decimal=3,
        err_msg='Did not get right shape image after reading InterpolatedImage from HDU')

    t2 = time.time()
    print('time for %s = %.2f'%(funcname(),t2-t1))


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
                { 'type' : 'Exponential' , 'scale_radius' : 3.4, 'flux' : 100 },
                { 'type' : 'Gaussian' , 'sigma' : 1, 'flux' : 50 }
            ],
            'gsparams' : { 'maxk_threshold' : 1.e-2,
                           'folding_threshold' : 1.e-2,
                           'stepk_minimum_hlr' : 3 }
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
    gal3b_1 = gal3b_1.shear(e1=0.2, e2=0.3).shift(0.4,0.9)
    gal3b_2 = galsim.Sersic(n = 1.1, half_light_radius = 2.5, flux = 0.7)
    gal3b = galsim.Add([gal3b_1, gal3b_2])
    gal3b = gal3b.withFlux(1.e6).shear(q = 0.6, beta = 0.39 * galsim.radians)
    gsobject_compare(gal3a, gal3b)

    gal4a = galsim.config.BuildGSObject(config, 'gal4')[0]
    gal4b_1 = galsim.Gaussian(half_light_radius = 2, flux = 8)
    gal4b_2 = galsim.Exponential(half_light_radius = 2.3, flux = 2)
    gal4b = galsim.Add([gal4b_1, gal4b_2])
    gal4b = gal4b.withFlux(50).dilate(3).shear(e1 = 0.3).rotate(12 * galsim.degrees)
    gal4b = gal4b.magnify(1.03).shear(g1 = 0.03, g2 = -0.05).shift(dx = 0.7, dy = -1.2) 
    gsobject_compare(gal4a, gal4b)

    # Check that the Add items correctly inherit their gsparams from the top level
    gal5a = galsim.config.BuildGSObject(config, 'gal5')[0]
    gsparams = galsim.GSParams(maxk_threshold=1.e-2, folding_threshold=1.e-2, stepk_minimum_hlr=3)
    gal5b_1 = galsim.Exponential(scale_radius=3.4, flux=100, gsparams=gsparams)
    gal5b_2 = galsim.Gaussian(sigma=1, flux=50, gsparams=gsparams)
    gal5b = galsim.Add([gal5b_1, gal5b_2])
    gsobject_compare(gal5a, gal5b, conv=galsim.Gaussian(sigma=1))

    try:
        # Make sure they don't match when using the default GSParams
        gal5c_1 = galsim.Exponential(scale_radius=3.4, flux=100)
        gal5c_2 = galsim.Gaussian(sigma=1, flux=50)
        gal5c = galsim.Add([gal5c_1, gal5c_2])
        np.testing.assert_raises(AssertionError,gsobject_compare, gal5a, gal5c,
                                 conv=galsim.Gaussian(sigma=1))
    except ImportError:
        print('The assert_raises tests require nose')

    t2 = time.time()
    print('time for %s = %.2f'%(funcname(),t2-t1))


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
            'gsparams' : { 'maxk_threshold' : 1.e-2,
                           'folding_threshold' : 1.e-2,
                           'stepk_minimum_hlr' : 3 }
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
    gal3b_1 = gal3b_1.shear(e1=0.2, e2=0.3).shift(0.4,0.9)
    gal3b_2 = galsim.Sersic(n = 1.1, half_light_radius = 2.5, flux = 0.7)
    gal3b = galsim.Convolve([gal3b_1, gal3b_2])
    gal3b = gal3b.withFlux(1.e6).shear(q = 0.6, beta = 0.39 * galsim.radians)
    gsobject_compare(gal3a, gal3b)

    gal4a = galsim.config.BuildGSObject(config, 'gal4')[0]
    gal4b_1 = galsim.Gaussian(half_light_radius = 2, flux = 8)
    gal4b_2 = galsim.Exponential(half_light_radius = 2.3, flux = 2)
    gal4b = galsim.Convolve([gal4b_1, gal4b_2])
    gal4b = gal4b.withFlux(50).dilate(3).shear(e1 = 0.3).rotate(12 * galsim.degrees)
    gal4b = gal4b.magnify(1.03).shear(g1 = 0.03, g2 = -0.05).shift(dx = 0.7, dy = -1.2) 
    gsobject_compare(gal4a, gal4b)

    # Check that the Convolve items correctly inherit their gsparams from the top level
    gal5a = galsim.config.BuildGSObject(config, 'gal5')[0]
    gsparams = galsim.GSParams(maxk_threshold=1.e-2, folding_threshold=1.e-2, stepk_minimum_hlr=3)
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
        print('The assert_raises tests require nose')

    t2 = time.time()
    print('time for %s = %.2f'%(funcname(),t2-t1))


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

    config['obj_num'] = 0
    gal1a = galsim.config.BuildGSObject(config, 'gal')[0]
    gal1b = galsim.Gaussian(sigma = 2)
    gsobject_compare(gal1a, gal1b)

    config['obj_num'] = 1
    gal2a = galsim.config.BuildGSObject(config, 'gal')[0]
    gal2b = galsim.Gaussian(fwhm = 2, flux = 100)
    gsobject_compare(gal2a, gal2b)

    config['obj_num'] = 2
    gal3a = galsim.config.BuildGSObject(config, 'gal')[0]
    gal3b = galsim.Gaussian(half_light_radius = 2, flux = 1.e6)
    gal3b = gal3b.shear(q = 0.6, beta = 0.39 * galsim.radians)
    gsobject_compare(gal3a, gal3b)

    config['obj_num'] = 3
    gal4a = galsim.config.BuildGSObject(config, 'gal')[0]
    gal4b = galsim.Gaussian(sigma = 1, flux = 50)
    gal4b = gal4b.dilate(3).shear(e1 = 0.3).rotate(12 * galsim.degrees).magnify(1.03)
    gal4b = gal4b.shear(g1 = 0.03, g2 = -0.05).shift(dx = 0.7, dy = -1.2) 
    gsobject_compare(gal4a, gal4b)

    # Check that the list items correctly inherit their gsparams from the top level
    config = {
        'gal' : { 
            'type' : 'List' ,
            'items' : [
                { 'type' : 'Exponential' , 'scale_radius' : 3.4, 'flux' : 100 },
                { 'type' : 'Exponential' , 'scale_radius' : 3, 'flux' : 10 }
            ],
            'gsparams' : { 'maxk_threshold' : 1.e-2,
                           'folding_threshold' : 1.e-2,
                           'stepk_minimum_hlr' : 3 }
        }
    }

    config['obj_num'] = 0
    gal5a = galsim.config.BuildGSObject(config, 'gal')[0]
    gsparams = galsim.GSParams(maxk_threshold=1.e-2, folding_threshold=1.e-2, stepk_minimum_hlr=3)
    gal5b = galsim.Exponential(scale_radius=3.4, flux=100, gsparams=gsparams)
    gsobject_compare(gal5a, gal5b, conv=galsim.Gaussian(sigma=1))

    try:
        # Make sure they don't match when using the default GSParams
        gal5c = galsim.Exponential(scale_radius=3.4, flux=100)
        np.testing.assert_raises(AssertionError,gsobject_compare, gal5a, gal5c,
                                 conv=galsim.Gaussian(sigma=1))
    except ImportError:
        print('The assert_raises tests require nose')

    t2 = time.time()
    print('time for %s = %.2f'%(funcname(),t2-t1))


def test_ring():
    """Test building a GSObject from a ring test:
    """
    import time
    t1 = time.time()

    config = {
        'stamp' : {
            'type' : 'Ring' ,
            'num' : 2,
        },
        'gal' : {
            'type' : 'Gaussian' ,
            'sigma' : 2,
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

    gauss = galsim.Gaussian(sigma=2)
    e1_list = [ 0.3, -0.3, 0.2, -0.2, 0.8, -0.8 ]
    e2_list = [ 0.1, -0.1, 0.1, -0.1, 0.1, -0.1 ]

    galsim.config.SetupConfigImageNum(config, 0, 0)
    ignore = galsim.config.stamp_ignore
    ring_builder = galsim.config.stamp_ring.RingBuilder()
    for k in range(6):
        galsim.config.SetupConfigObjNum(config, k)
        ring_builder.setup(config['stamp'], config, None, None, ignore, None)
        gal1a = ring_builder.buildProfile(config['stamp'], config, None, {}, None)
        gal1b = gauss.shear(e1=e1_list[k], e2=e2_list[k])
        print('gal1a = ',gal1a)
        print('gal1b = ',gal1b)
        gsobject_compare(gal1a, gal1b)

    config = {
        'stamp' : {
            'type' : 'Ring' ,
            'num' : 10,
        },
        'gal' : {
            'type' : 'Exponential', 'half_light_radius' : 2,
            'ellip' : galsim.Shear(e2=0.3)
        },
    }

    disk = galsim.Exponential(half_light_radius=2).shear(e2=0.3)

    galsim.config.SetupConfigImageNum(config, 0, 0)
    for k in range(25):
        galsim.config.SetupConfigObjNum(config, k)
        ring_builder.setup(config['stamp'], config, None, None, ignore, None)
        gal2a = ring_builder.buildProfile(config['stamp'], config, None, {}, None)
        gal2b = disk.rotate(theta = k * 18 * galsim.degrees)
        gsobject_compare(gal2a, gal2b)

    config = {
        'stamp' : {
            'type' : 'Ring' ,
            'num' : 5,
            'full_rotation' : 360. * galsim.degrees,
            'index' : { 'type' : 'Sequence', 'repeat' : 4 }
        },
        'gal' : {
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
    }

    disk = galsim.Exponential(half_light_radius=2).shear(e2=0.3)
    bulge = galsim.Sersic(n=3, half_light_radius=1.3).shear(e1=0.12,e2=-0.08)
    sum = disk + bulge

    galsim.config.SetupConfigImageNum(config, 0, 0)
    for k in range(25):
        galsim.config.SetupConfigObjNum(config, k)
        index = k // 4  # make sure we use integer division
        ring_builder.setup(config['stamp'], config, None, None, ignore, None)
        gal3a = ring_builder.buildProfile(config['stamp'], config, None, {}, None)
        gal3b = sum.rotate(theta = index * 72 * galsim.degrees)
        gsobject_compare(gal3a, gal3b)

    # Check that the ring items correctly inherit their gsparams from the top level
    config = {
        'stamp' : {
            'type' : 'Ring' ,
            'num' : 20,
            'full_rotation' : 360. * galsim.degrees,
            'gsparams' : { 'maxk_threshold' : 1.e-2,
                           'folding_threshold' : 1.e-2,
                           'stepk_minimum_hlr' : 3 }
        },
        'gal' : {
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
    }

    galsim.config.SetupConfigImageNum(config, 0, 0)
    galsim.config.SetupConfigObjNum(config, 0)
    ring_builder.setup(config['stamp'], config, None, None, ignore, None)
    gal4a = ring_builder.buildProfile(config['stamp'], config, None, config['stamp']['gsparams'],
                                      None)
    gsparams = galsim.GSParams(maxk_threshold=1.e-2, folding_threshold=1.e-2, stepk_minimum_hlr=3)
    disk = galsim.Exponential(half_light_radius=2, gsparams=gsparams).shear(e2=0.3)
    bulge = galsim.Sersic(n=3,half_light_radius=1.3, gsparams=gsparams).shear(e1=0.12,e2=-0.08)
    gal4b = disk + bulge
    gsobject_compare(gal4a, gal4b, conv=galsim.Gaussian(sigma=1))

    try:
        # Make sure they don't match when using the default GSParams
        disk = galsim.Exponential(half_light_radius=2).shear(e2=0.3)
        bulge = galsim.Sersic(n=3,half_light_radius=1.3).shear(e1=0.12,e2=-0.08)
        gal4c = disk + bulge
        np.testing.assert_raises(AssertionError,gsobject_compare, gal4a, gal4c,
                                 conv=galsim.Gaussian(sigma=1))
    except ImportError:
        print('The assert_raises tests require nose')

    t2 = time.time()
    print('time for %s = %.2f'%(funcname(),t2-t1))


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


