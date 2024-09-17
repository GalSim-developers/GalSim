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

import logging
import numpy as np
import astropy.units as u
import os
import sys

import galsim
from galsim_test_helpers import *


@timer
def test_gaussian():
    """Test various ways to build a Gaussian
    """
    config = {
        'gal1' : { 'type' : 'Gaussian' , 'sigma' : 2 },
        'gal2' : { 'type' : 'Gaussian' , 'fwhm' : 2, 'flux' : 100 },
        'gal2c' : { 'type' : 'Eval', 'str': 'galsim.Gaussian(fwhm=2, flux=100)' },
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
                   'lens' : {
                       'mu' : 0.93,
                       'shear' : galsim.Shear(g1=-0.15, g2=0.2),
                    },
                 },
        'gal6' : { 'type' : 'DeltaFunction' , 'flux' : 72.5 },
        'bad1' : { 'type' : 'Gaussian' , 'fwhm' : 2, 'sigma' : 3, 'flux' : 100 },
        'bad2' : { 'type' : 'Gaussian' },
        'bad3' : { 'type' : 'Gaussian', 'sig' : 4 },
        'bad4' : { 'sigma' : 2 },
        'bad5' : { 'type' : 'Gauss', 'sigma' : 2 },
        'bad6' : { 'type' : 'Gaussian', 'resolution' : 1.5 },  # requires psf field.
        'bad7' : { 'type' : 'Gaussian', 'sigma' : 2, 'resolution' : 1.5 }, # can't give sigma
        'bad8' : { 'type' : 'Gaussian', 'half_light_radius' : 2, 'resolution' : 1.5 }, # or hlr
     }

    gal1a = galsim.config.BuildGSObject(config, 'gal1')[0]
    gal1b = galsim.Gaussian(sigma = 2)
    gsobject_compare(gal1a, gal1b)

    gal2a = galsim.config.BuildGSObject(config, 'gal2')[0]
    gal2b = galsim.Gaussian(fwhm = 2, flux = 100)
    gal2c = galsim.config.BuildGSObject(config, 'gal2c')[0]
    gsobject_compare(gal2a, gal2b)
    gsobject_compare(gal2a, gal2c)

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

    gal6a = galsim.config.BuildGSObject(config, 'gal6')[0]
    gal6b = galsim.DeltaFunction(flux = 72.5)
    gsobject_compare(gal6a, gal6b, conv=galsim.Gaussian(sigma=0.01))
    # DeltaFunction is functionally equivalent to an extremely narrow Gaussian.
    gal6c = galsim.Gaussian(sigma = 1.e-10, flux = 72.5)
    gsobject_compare(gal6a, gal6c, conv=galsim.Gaussian(sigma=0.01))

    with assert_raises(galsim.GalSimConfigError):
        galsim.config.BuildGSObject(config, 'bad1')
    with assert_raises(galsim.GalSimConfigError):
        galsim.config.BuildGSObject(config, 'bad2')
    with assert_raises(galsim.GalSimConfigError):
        galsim.config.BuildGSObject(config, 'bad3')
    with assert_raises(galsim.GalSimConfigError):
        galsim.config.BuildGSObject(config, 'bad4')
    with assert_raises(galsim.GalSimConfigError):
        galsim.config.BuildGSObject(config, 'bad5')
    with assert_raises(galsim.GalSimConfigError):
        galsim.config.BuildGSObject(config, 'bad6')

    # Test various invalid ways to use resolution.
    # This psf cannot be used for resolution, since no half_light_radius field.
    psf_file = os.path.join('SBProfile_comparison_images','gauss_smallshear.fits')
    config['psf'] = { 'type' : 'InterpolatedImage', 'image' : psf_file }
    psf = galsim.config.BuildGSObject(config, 'psf')
    with assert_raises(galsim.GalSimConfigError):
        galsim.config.BuildGSObject(config, 'bad6')
    # This has half_light_radius, but it raises an exception for obscuration != 1
    config['psf'] = { 'type' : 'Airy' , 'lam_over_diam' : 0.4, 'obscuration' : 0.3 }
    psf = galsim.config.BuildGSObject(config, 'psf')
    with assert_raises(galsim.GalSimConfigError):
        galsim.config.BuildGSObject(config, 'bad6')
    # This finally works.
    config['psf'] = { 'type' : 'Airy' , 'lam_over_diam' : 0.4 }
    psf = galsim.config.BuildGSObject(config, 'psf')
    gal = galsim.config.BuildGSObject(config, 'bad6')
    # Can't give a different size along with resolution.
    with assert_raises(galsim.GalSimConfigError):
        gal = galsim.config.BuildGSObject(config, 'bad7')
    with assert_raises(galsim.GalSimConfigError):
        gal = galsim.config.BuildGSObject(config, 'bad8')

    # Next time it gets called, it doesn't raise despite half_light_radius now being there.
    galsim.config.RemoveCurrent(config)
    gal = galsim.config.BuildGSObject(config, 'bad6')

@timer
def test_moffat():
    """Test various ways to build a Moffat
    """
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
        'gal5' : { 'type' : 'Moffat' , 'beta' : 2.8, 'flux' : 22, 'fwhm' : 0.2, 'trunc' : 0.7,
                   'shear' : galsim.Shear(g1=-0.15, g2=0.2),
                   'gsparams' : { 'maxk_threshold' : 1.e-2 }
                 },
        'bad1' : { 'type' : 'Moffat' , 'beta' : 1.4, 'scale_radius' : 2, 'fwhm' : 3 },
        'bad2' : { 'type' : 'Moffat' , 'beta' : 1.4 },
        'bad3' : { 'type' : 'Moffat' , 'beth' : 1.4, 'fwhm' : 8 },
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
    gal5b = galsim.Moffat(beta=2.8, fwhm=0.2, flux=22, trunc=0.7, gsparams=gsparams)
    gal5b = gal5b.shear(g1=-0.15, g2=0.2)
    # convolve to test the k-space gsparams (with an even smaller profile)
    gsobject_compare(gal5a, gal5b, conv=galsim.Gaussian(sigma=0.01))

    # Make sure they don't match when using the default GSParams
    gal5c = galsim.Moffat(beta=2.8, fwhm=0.3, flux=22, trunc=0.7)
    gal5c = gal5c.shear(g1=-0.15, g2=0.2)
    with assert_raises(AssertionError):
        gsobject_compare(gal5a, gal5c, conv=galsim.Gaussian(sigma=0.01))

    with assert_raises(galsim.GalSimConfigError):
        galsim.config.BuildGSObject(config, 'bad1')
    with assert_raises(galsim.GalSimConfigError):
        galsim.config.BuildGSObject(config, 'bad2')
    with assert_raises(galsim.GalSimConfigError):
        galsim.config.BuildGSObject(config, 'bad3')


@timer
def test_airy():
    """Test various ways to build a Airy
    """
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
        'gal5c' : { 'type' : 'Eval', 'str': 'galsim.Airy(lam_over_diam=45)',
                    'gsparams' : { 'xvalue_accuracy' : 1.e-2 }
                 },
        'gal6' : { 'type' : 'Airy' , 'lam' : 400., 'diam' : 4.0, 'scale_unit' : 'arcmin' },
        'gal7' : { 'type' : 'Airy' , 'lam' : 400*u.nm, 'diam' : '4000 mm', 'scale_unit' : 'arcmin' },
        'bad1' : { 'type' : 'Airy' , 'lam_over_diam' : 0.4, 'lam' : 400, 'diam' : 10 },
        'bad2' : { 'type' : 'Airy' , 'flux' : 1.3 },
        'bad3' : { 'type' : 'Airy' , 'lam_over_diam' : 0.4, 'obsc' : 0.3, 'flux' : 100 },
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
    gal5c = galsim.config.BuildGSObject(config, 'gal5c')[0]
    gsobject_compare(gal5a, gal5b)
    gsobject_compare(gal5a, gal5c)

    gal6a = galsim.config.BuildGSObject(config, 'gal6')[0]
    gal6b = galsim.Airy(lam=400., diam=4., scale_unit=galsim.arcmin)
    gsobject_compare(gal6a, gal6b)
    gal7a = galsim.config.BuildGSObject(config, 'gal7')[0]
    gsobject_compare(gal6a, gal7a)

    # Make sure they don't match when using the default GSParams
    gal5c = galsim.Airy(lam_over_diam=45)
    with assert_raises(AssertionError):
        gsobject_compare(gal5a, gal5c)

    with assert_raises(galsim.GalSimConfigError):
        galsim.config.BuildGSObject(config, 'bad1')
    with assert_raises(galsim.GalSimConfigError):
        galsim.config.BuildGSObject(config, 'bad2')
    with assert_raises(galsim.GalSimConfigError):
        galsim.config.BuildGSObject(config, 'bad3')


@timer
def test_kolmogorov():
    """Test various ways to build a Kolmogorov
    """
    config = {
        'gal1' : { 'type' : 'Kolmogorov' , 'lam_over_r0' : 2 },
        'gal2' : { 'type' : 'Kolmogorov' , 'fwhm' : 2, 'flux' : 100 },
        'gal3' : { 'type' : 'Kolmogorov' , 'half_light_radius' : 2, 'flux' : 1.e6,
                   'ellip' : { 'type' : 'QBeta' , 'q' : 0.6, 'beta' : 0.39 * galsim.radians }
                 },
        'gal3c' : { 'type': 'Eval',
                    'str': "galsim.Kolmogorov(half_light_radius=2, flux=1.e6).shear(q=0.6, beta=0.39*galsim.radians)"
                  },
        'gal4' : { 'type' : 'Kolmogorov' , 'lam' : 400, 'r0_500' : 0.15, 'flux' : 50,
                   'dilate' : 3, 'ellip' : galsim.Shear(e1=0.3),
                   'rotate' : 12 * galsim.degrees,
                   'lens' : {
                       'shear' : galsim.Shear(g1=0.03, g2=-0.05),
                       'mu' : 1.03,
                   },
                   'shift' : { 'type' : 'XY', 'x' : 0.7, 'y' : -1.2 },
                 },
        'gal5' : { 'type' : 'Kolmogorov' , 'lam_over_r0' : 1, 'flux' : 50,
                   'gsparams' : { 'integration_relerr' : 1.e-2, 'integration_abserr' : 1.e-4 }
                 },
        'gal6' : { 'type' : 'Kolmogorov' , 'lam' : '400 nm', 'r0_500' : '15 cm' },
        'gal7' : { 'type' : 'Kolmogorov' , 'lam' : '$4000*u.Angstrom', 'r0' : 0.18*u.m },
        'bad1' : { 'type' : 'Kolmogorov' , 'fwhm' : 2, 'lam_over_r0' : 3, 'flux' : 100 },
        'bad2' : { 'type' : 'Kolmogorov', 'flux' : 100 },
        'bad3' : { 'type' : 'Kolmogorov' , 'lam_over_r0' : 2, 'lam' : 400, 'r0' : 0.15 },
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
    gal3c = galsim.config.BuildGSObject(config, 'gal3c')[0]
    gsobject_compare(gal3a, gal3b)
    gsobject_compare(gal3a, gal3c)

    gal4a = galsim.config.BuildGSObject(config, 'gal4')[0]
    gal4b = galsim.Kolmogorov(lam=400, r0_500=0.15, flux = 50)
    gal4b = gal4b.dilate(3).shear(e1 = 0.3).rotate(12 * galsim.degrees)
    gal4b = gal4b.lens(0.03, -0.05, 1.03).shift(dx = 0.7, dy = -1.2)
    gsobject_compare(gal4a, gal4b)

    gal5a = galsim.config.BuildGSObject(config, 'gal5')[0]
    gsparams = galsim.GSParams(integration_relerr=1.e-2, integration_abserr=1.e-4)
    gal5b = galsim.Kolmogorov(lam_over_r0=1, flux=50, gsparams=gsparams)
    gsobject_compare(gal5a, gal5b)

    # Make sure they don't match when using the default GSParams
    gal5c = galsim.Kolmogorov(lam_over_r0=1, flux=50)
    with assert_raises(AssertionError):
        gsobject_compare(gal5a, gal5c)

    gal6a = galsim.config.BuildGSObject(config, 'gal6')[0]
    gal6b = galsim.Kolmogorov(lam=400*u.nm, r0_500=15*u.cm)
    gsobject_compare(gal6a, gal6b)

    gal7a = galsim.config.BuildGSObject(config, 'gal7')[0]
    gal7b = galsim.Kolmogorov(lam=4000*u.Angstrom, r0=0.18)
    gsobject_compare(gal7a, gal7b)

    with assert_raises(galsim.GalSimConfigError):
        galsim.config.BuildGSObject(config, 'bad1')
    with assert_raises(galsim.GalSimConfigError):
        galsim.config.BuildGSObject(config, 'bad2')
    with assert_raises(galsim.GalSimConfigError):
        galsim.config.BuildGSObject(config, 'bad3')


@timer
def test_VonKarman():
    """Test various ways to build a VonKarman
    """
    config = {
        'gal1' : { 'type' : 'VonKarman' , 'lam' : 500, 'r0' : 0.2 },
        'gal2' : { 'type' : 'VonKarman' , 'lam' : 760, 'r0_500' : 0.2, 'L0' : 24.0 },
        'gal3' : { 'type' : 'VonKarman' , 'lam' : 450*u.nm, 'r0_500' : '20 cm', 'L0' : '$80*u.imperial.ft',
                   'flux' : 1.e6,
                   'ellip' : { 'type' : 'QBeta' , 'q' : 0.6, 'beta' : 0.39 * galsim.radians }
                 },
        'gal4' : { 'type' : 'VonKarman' , 'lam' : 500, 'r0' : 0.2*u.m,
                   'dilate' : 3, 'ellip' : galsim.Shear(e1=0.3),
                   'rotate' : 12 * galsim.degrees,
                   'lens' : {
                       'shear' : galsim.Shear(g1=0.03, g2=-0.05),
                       'mu' : 1.03,
                   },
                   'shift' : { 'type' : 'XY', 'x' : 0.7, 'y' : -1.2 },
                 },
        'gal5' : { 'type' : 'VonKarman' , 'lam' : 500, 'r0' : 0.2, 'do_delta' : True,
                   'gsparams' : { 'integration_relerr' : 1.e-2, 'integration_abserr' : 1.e-4 }
                 },
        'bad1' : { 'type' : 'VonKarman' , 'fwhm' : 2, 'lam_over_r0' : 3, 'flux' : 100 },
        'bad2' : { 'type' : 'VonKarman', 'flux' : 100 },
        'bad3' : { 'type' : 'VonKarman' , 'lam' : 400, 'r0' : 0.15, 'r0_500' : 0.12 },
        'bad4' : { 'type' : 'VonKarman' , 'lam' : 'not_a_quantity_or_float', 'r0' : 0.2},
    }

    gal1a = galsim.config.BuildGSObject(config, 'gal1')[0]
    gal1b = galsim.VonKarman(lam = 500, r0 = 0.2)
    gsobject_compare(gal1a, gal1b)

    gal2a = galsim.config.BuildGSObject(config, 'gal2')[0]
    gal2b = galsim.VonKarman(lam = 760, r0_500 = 0.2, L0 = 24.0)
    gsobject_compare(gal2a, gal2b)

    gal3a = galsim.config.BuildGSObject(config, 'gal3')[0]
    gal3b = galsim.VonKarman(lam = 450*u.nm, r0_500 = 20*u.cm, L0 = 80*u.imperial.ft, flux = 1.e6)
    gal3b = gal3b.shear(q = 0.6, beta = 0.39 * galsim.radians)
    gsobject_compare(gal3a, gal3b)


    gal4a = galsim.config.BuildGSObject(config, 'gal4')[0]
    gal4b = galsim.VonKarman(lam = 500, r0 = 0.2)
    gal4b = gal4b.dilate(3).shear(e1 = 0.3).rotate(12 * galsim.degrees)
    gal4b = gal4b.lens(0.03, -0.05, 1.03).shift(dx = 0.7, dy = -1.2)
    gsobject_compare(gal4a, gal4b)

    gal5a = galsim.config.BuildGSObject(config, 'gal5')[0]
    gsparams = galsim.GSParams(integration_relerr=1.e-2, integration_abserr=1.e-4)
    gal5b = galsim.VonKarman(lam=500, r0=0.2, do_delta=True, gsparams=gsparams)
    gsobject_compare(gal5a, gal5b)

    with assert_raises(galsim.GalSimConfigError):
        galsim.config.BuildGSObject(config, 'bad1')
    with assert_raises(galsim.GalSimConfigError):
        galsim.config.BuildGSObject(config, 'bad2')
    with assert_raises(galsim.GalSimConfigError):
        galsim.config.BuildGSObject(config, 'bad3')
    with assert_raises(galsim.GalSimConfigError):
        galsim.config.BuildGSObject(config, 'bad4')


@timer
def test_secondKick():
    config = {
        'gal1' : { 'type' : 'SecondKick' , 'lam' : 500, 'r0' : 0.2, 'diam': 4.0 },
        'gal2' : { 'type' : 'SecondKick' , 'lam' : '0.5 micron', 'r0' : '10 cm', 'diam': 8.2*u.m, 'flux' : 100 },
        'gal3' : { 'type' : 'SecondKick' , 'lam' : '$2*450*u.nm', 'r0' : 0.2, 'diam': 4.0, 'obscuration' : 0.2, 'kcrit' : 0.1 },
    }

    gal1a = galsim.config.BuildGSObject(config, 'gal1')[0]
    gal1b = galsim.SecondKick(lam = 500, r0 = 0.2, diam=4.0)
    gsobject_compare(gal1a, gal1b)

    gal2a = galsim.config.BuildGSObject(config, 'gal2')[0]
    gal2b = galsim.SecondKick(lam = 0.5*u.micron, r0 = 10*u.cm, diam=8.2*u.m, flux = 100)
    gsobject_compare(gal2a, gal2b)

    gal3a = galsim.config.BuildGSObject(config, 'gal3')[0]
    gal3b = galsim.SecondKick(lam = 2*450, r0 = 0.2, diam=4.0, obscuration=0.2, kcrit=0.1)
    gsobject_compare(gal3a, gal3b)


@timer
def test_opticalpsf():
    """Test various ways to build a OpticalPSF
    """
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
        'gal5' : { 'type': 'OpticalPSF' , 'lam' : 900, 'diam' : 2.4, 'flux' : 1.8,
                   'defocus' : 0.1, 'obscuration' : 0.18,
                   'pupil_plane_im' :
                       os.path.join(".","Optics_comparison_images","sample_pupil_rolled.fits"),
                   'pupil_angle' : 27.*galsim.degrees },
        'gal6' : {'type' : 'OpticalPSF' , 'lam' : '874 nm', 'diam' : '7.4 m', 'flux' : 70.,
                  'aberrations' : [0.06, 0.12, -0.08, 0.07, 0.04, 0.0, 0.0, -0.13],
                  'obscuration' : 0.1 },
        'gal7' : {'type' : 'OpticalPSF' , 'lam' : 0.874*u.micron, 'diam' : '$740*u.cm', 'aberrations' : []},
        'bad1' : {'type' : 'OpticalPSF' , 'lam' : 874.0, 'diam' : 7.4, 'lam_over_diam' : 0.2},
        'bad2' : {'type' : 'OpticalPSF' , 'lam_over_diam' : 0.2,
                  'aberrations' : "0.06, 0.12, -0.08, 0.07, 0.04, 0.0, 0.0, -0.13"},
        'bad3' : {'type' : 'OpticalPSF' , 'lam_over_diam' : 0.2, 'aberr' : []},
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
        lam=900, diam=2.4, flux=1.8, defocus=0.1, obscuration=0.18,
        pupil_plane_im=os.path.join(".","Optics_comparison_images","sample_pupil_rolled.fits"),
        pupil_angle=27.*galsim.degrees)
    gsobject_compare(gal5a, gal5b)

    gal6a = galsim.config.BuildGSObject(config, 'gal6')[0]
    aberrations = np.zeros(12, dtype=float)
    aberrations[4:] = [0.06, 0.12, -0.08, 0.07, 0.04, 0.0, 0.0, -0.13]
    gal6b = galsim.OpticalPSF(lam=874., diam=7.4, flux=70., obscuration=0.1,
                              aberrations=aberrations)
    gsobject_compare(gal6a, gal6b)

    gal7a = galsim.config.BuildGSObject(config, 'gal7')[0]
    gal7b = galsim.OpticalPSF(lam=874., diam=7.4)
    gsobject_compare(gal7a, gal7b)

    with assert_raises(galsim.GalSimConfigError):
        galsim.config.BuildGSObject(config, 'bad1')
    with assert_raises(galsim.GalSimConfigError):
        galsim.config.BuildGSObject(config, 'bad2')
    with assert_raises(galsim.GalSimConfigError):
        galsim.config.BuildGSObject(config, 'bad3')


@timer
def test_exponential():
    """Test various ways to build a Exponential
    """
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
                 },
        'bad1' : { 'type' : 'Exponential' , 'scale_radius' : 2, 'half_light_radius' : 3 },
        'bad2' : { 'type' : 'Exponential' },
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

    # Make sure they don't match when using the default GSParams
    gal5c = galsim.Exponential(scale_radius=1, flux=50)
    with assert_raises(AssertionError):
        gsobject_compare(gal5a, gal5c, conv=galsim.Gaussian(sigma=1))

    with assert_raises(galsim.GalSimConfigError):
        galsim.config.BuildGSObject(config, 'bad1')
    with assert_raises(galsim.GalSimConfigError):
        galsim.config.BuildGSObject(config, 'bad2')

@timer
def test_sersic():
    """Test various ways to build a Sersic
    """
    config = {
        'gal1' : { 'type' : 'Sersic' , 'n' : 1.2,  'half_light_radius' : 2 },
        'gal2' : { 'type' : 'Sersic' , 'n' : 3.5,  'half_light_radius' : 1.7, 'flux' : 100 },
        'gal3' : { 'type' : 'Sersic' , 'n' : 2.2,  'half_light_radius' : 3.5, 'flux' : 1.e6,
                   'ellip' : { 'type' : 'QBeta' , 'q' : 0.6, 'beta' : 0.39 * galsim.radians }
                 },
        'gal3c' : { 'type': 'Eval',
                    'str': "galsim.Sersic(n=2.2, half_light_radius=3.5, flux=1.e6)",
                    'ellip' : { 'type' : 'QBeta' , 'q' : 0.6, 'beta' : 0.39 * galsim.radians }
                  },
        'gal4' : { 'type' : 'Sersic' , 'n' : 0.7,  'half_light_radius' : 1, 'flux' : 50,
                   'dilate' : 3, 'ellip' : galsim.Shear(e1=0.3),
                   'rotate' : 12 * galsim.degrees,
                   'lens' : {
                       'mu' : 1.03,
                       'shear' : galsim.Shear(g1=0.03, g2=-0.05),
                   },
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
                 },
        'bad1' : { 'type' : 'Sersic' , 'n' : 0.1,  'half_light_radius' : 3.5 },
        'bad2' : { 'type' : 'Sersic' , 'n' : 11.1,  'half_light_radius' : 3.5 },
        'bad3' : { 'type' : 'Sersic' , 'n' : 1.1 },
        'bad4' : { 'type' : 'Sersic' , 'n' : 1.1,  'half_light_radius' : 3.5, 'scale_radius' : 2 },
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
    gal3c = galsim.config.BuildGSObject(config, 'gal3c')[0]
    gsobject_compare(gal3a, gal3b)
    gsobject_compare(gal3a, gal3c)

    gal4a = galsim.config.BuildGSObject(config, 'gal4')[0]
    gal4b = galsim.Sersic(n = 0.7, half_light_radius = 1, flux = 50)
    gal4b = gal4b.dilate(3).shear(e1 = 0.3).rotate(12 * galsim.degrees)
    gal4b = gal4b.lens(0.03, -0.05, 1.03).shift(dx = 0.7, dy = -1.2)
    gsobject_compare(gal4a, gal4b)

    gal5a = galsim.config.BuildGSObject(config, 'gal5')[0]
    gsparams = galsim.GSParams(minimum_fft_size=256)
    gal5b = galsim.Sersic(n=0.7, half_light_radius=1, flux=50, gsparams=gsparams)
    gsobject_compare(gal5a, gal5b, conv=galsim.Gaussian(sigma=1))

    # Make sure they don't match when using the default GSParams
    gal5c = galsim.Sersic(n=0.7, half_light_radius=1, flux=50)
    with assert_raises(AssertionError):
        gsobject_compare(gal5a, gal5c, conv=galsim.Gaussian(sigma=1))

    # For the maximum_fft_size test, we need to do things a little differently
    # We lower maximum_fft_size below the size that it normally wants this to be,
    # and we check to make sure an exception is thrown.  Of course, this isn't how you
    # would normally use maximum_fft_size.  Normally, you would raise it when the default
    # is too small.  But to construct the test that way would require a lot of memory
    # and would be rather slow.
    gal6a = galsim.config.BuildGSObject(config, 'gal6')[0]
    gal6b = galsim.Sersic(n=0.7, half_light_radius=1, flux=50)
    with assert_raises(galsim.GalSimFFTSizeError):
        gsobject_compare(gal6a, gal6b, conv=galsim.Gaussian(sigma=1))

    gal7a = galsim.config.BuildGSObject(config, 'gal7')[0]
    gsparams = galsim.GSParams(realspace_relerr=1.e-2, realspace_abserr=1.e-4)
    gal7b = galsim.Sersic(n=3.2, half_light_radius=1.7, flux=50, trunc=4.3, gsparams=gsparams)
    # Convolution with a truncated Moffat will use realspace convolution
    conv = galsim.Moffat(beta=2.8, fwhm=1.3, trunc=3.7)
    gsobject_compare(gal7a, gal7b, conv=conv)

    # Make sure they don't match when using the default GSParams
    gal7c = galsim.Sersic(n=3.2, half_light_radius=1.7, flux=50, trunc=4.3)
    with assert_raises(AssertionError):
        gsobject_compare(gal7a, gal7c, conv=conv)

    with assert_raises(galsim.GalSimRangeError):
        galsim.config.BuildGSObject(config, 'bad1')
    with assert_raises(galsim.GalSimRangeError):
        galsim.config.BuildGSObject(config, 'bad2')
    with assert_raises(galsim.GalSimConfigError):
        galsim.config.BuildGSObject(config, 'bad3')
    with assert_raises(galsim.GalSimConfigError):
        galsim.config.BuildGSObject(config, 'bad4')


@timer
def test_devaucouleurs():
    """Test various ways to build a DeVaucouleurs
    """
    config = {
        'gal1' : { 'type' : 'DeVaucouleurs' , 'half_light_radius' : 2 },
        'gal2' : { 'type' : 'DeVaucouleurs' , 'half_light_radius' : 1.7, 'flux' : 100 },
        'gal3' : { 'type' : 'DeVaucouleurs' , 'half_light_radius' : 3.5, 'flux' : 1.e6,
                   'ellip' : { 'type' : 'QBeta' , 'q' : 0.6, 'beta' : 0.39 * galsim.radians }
                 },
        'gal3c': { 'type': 'Eval',
                   'str': "galsim.DeVaucouleurs(half_light_radius=@gal3.half_light_radius,\
                                                flux=@gal2.flux**3)",
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
    gal3c = galsim.config.BuildGSObject(config, 'gal3c')[0]
    gsobject_compare(gal3a, gal3b)
    gsobject_compare(gal3a, gal3c)

    gal4a = galsim.config.BuildGSObject(config, 'gal4')[0]
    gal4b = galsim.DeVaucouleurs(half_light_radius = 1, flux = 50)
    gal4b = gal4b.dilate(3).shear(e1 = 0.3).rotate(12 * galsim.degrees).magnify(1.03)
    gal4b = gal4b.shear(g1 = 0.03, g2 = -0.05).shift(dx = 0.7, dy = -1.2)
    gsobject_compare(gal4a, gal4b)

    gal5a = galsim.config.BuildGSObject(config, 'gal5')[0]
    gsparams = galsim.GSParams(folding_threshold=1.e-4)
    gal5b = galsim.DeVaucouleurs(half_light_radius=1, flux=50, gsparams=gsparams)
    gsobject_compare(gal5a, gal5b, conv=galsim.Gaussian(sigma=1))

    # Make sure they don't match when using the default GSParams
    gal5c = galsim.DeVaucouleurs(half_light_radius=1, flux=50)
    with assert_raises(AssertionError):
        gsobject_compare(gal5a, gal5c, conv=galsim.Gaussian(sigma=1))

@timer
def test_inclined_exponential():
    """Test various ways to build an InclinedExponential
    """
    config = {
        'gal1' : { 'type' : 'InclinedExponential' , 'inclination' : 0.1 * galsim.radians,
                   'half_light_radius' : 2 },
        'gal2' : { 'type' : 'InclinedExponential' , 'inclination' : 21 * galsim.degrees,
                   'scale_radius' : 0.7, 'flux' : 100 },
        'gal3' : { 'type' : 'InclinedExponential' , 'inclination' : 0.3 * galsim.radians,
                   'scale_radius' : 0.35, 'scale_height' : 0.23, 'flux' : 1.e6,
                   'ellip' : { 'type' : 'QBeta' , 'q' : 0.6, 'beta' : 0.39 * galsim.radians }
                 },
        'gal4' : { 'type' : 'InclinedExponential' , 'inclination' : 0.7 * galsim.radians,
                   'half_light_radius' : 1, 'scale_h_over_r' : 0.2, 'flux' : 50,
                   'dilate' : 3, 'ellip' : galsim.Shear(e1=0.3),
                   'rotate' : 12 * galsim.degrees,
                   # Note: Unlike some others, here we use magnification and shear, just to
                   # show that this is also equivalent to using lens.
                   'magnify' : 1.03, 'shear' : galsim.Shear(g1=0.03, g2=-0.05),
                   'shift' : { 'type' : 'XY', 'x' : 0.7, 'y' :-1.2 }
                 },
        'gal5' : { 'type' : 'InclinedExponential' , 'inclination' : 0.7 * galsim.radians,
                   'half_light_radius' : 1, 'flux' : 50,
                   'gsparams' : { 'minimum_fft_size' : 256 }
                 },
        'bad1' : { 'type' : 'InclinedExponential' , 'inclination' : 0.7 * galsim.radians,
                   'half_light_radius' : 1, 'scale_radius' : 2 },
        'bad2' : { 'type' : 'InclinedExponential' , 'inclination' : 0.7 * galsim.radians,
                   'scale_h_over_r' : 0.2 },
        'bad3' : { 'type' : 'InclinedExponential' , 'inclination' : 0.7 * galsim.radians,
                   'scale_radius' : 1, 'scale_h_over_r' : 0.2, 'scale_height' : 0.1 },
    }

    gal1a = galsim.config.BuildGSObject(config, 'gal1')[0]
    gal1b = galsim.InclinedExponential(inclination=0.1 * galsim.radians, half_light_radius=2)
    gsobject_compare(gal1a, gal1b)

    gal2a = galsim.config.BuildGSObject(config, 'gal2')[0]
    gal2b = galsim.InclinedExponential(inclination=21 * galsim.degrees, scale_radius=0.7, flux=100)
    gsobject_compare(gal2a, gal2b)

    gal3a = galsim.config.BuildGSObject(config, 'gal3')[0]
    gal3b = galsim.InclinedExponential(inclination=0.3 * galsim.radians, scale_radius=0.35,
                                  scale_height=0.23, flux=1.e6)
    gal3b = gal3b.shear(q=0.6, beta=0.39 * galsim.radians)
    gsobject_compare(gal3a, gal3b)

    gal4a = galsim.config.BuildGSObject(config, 'gal4')[0]
    gal4b = galsim.InclinedExponential(inclination=0.7 * galsim.radians, half_light_radius=1,
                                  scale_h_over_r=0.2, flux=50)
    gal4b = gal4b.dilate(3).shear(e1=0.3).rotate(12 * galsim.degrees)
    gal4b = gal4b.lens(0.03, -0.05, 1.03).shift(dx=0.7, dy=-1.2)
    gsobject_compare(gal4a, gal4b)

    gal5a = galsim.config.BuildGSObject(config, 'gal5')[0]
    gsparams = galsim.GSParams(minimum_fft_size=256)
    gal5b = galsim.InclinedExponential(inclination=0.7 * galsim.radians, half_light_radius=1,
                                       flux=50, gsparams=gsparams)
    gsobject_compare(gal5a, gal5b, conv=galsim.Gaussian(sigma=1))

    # Make sure they don't match when using the default GSParams
    gal5c = galsim.InclinedExponential(inclination=0.7 * galsim.radians, half_light_radius=1,
                                       flux=50)
    with assert_raises(AssertionError):
        gsobject_compare(gal5a, gal5c, conv=galsim.Gaussian(sigma=1))

    with assert_raises(galsim.GalSimConfigError):
        galsim.config.BuildGSObject(config, 'bad1')
    with assert_raises(galsim.GalSimConfigError):
        galsim.config.BuildGSObject(config, 'bad2')
    with assert_raises(galsim.GalSimError):
        galsim.config.BuildGSObject(config, 'bad3')

@timer
def test_inclined_sersic():
    """Test various ways to build an InclinedSersic
    """
    config = {
        'gal1' : { 'type' : 'InclinedSersic' , 'n' : 1.2, 'inclination' : 0.1 * galsim.radians,
                   'half_light_radius' : 2 },
        'gal2' : { 'type' : 'InclinedSersic' , 'n' : 3.5, 'inclination' : 21 * galsim.degrees,
                   'scale_radius' : 0.007, 'flux' : 100 },
        'gal3' : { 'type' : 'InclinedSersic' , 'n' : 2.2, 'inclination' : 0.3 * galsim.radians,
                   'scale_radius' : 0.35, 'scale_height' : 0.23, 'flux' : 1.e6,
                   'ellip' : { 'type' : 'QBeta' , 'q' : 0.6, 'beta' : 0.39 * galsim.radians }
                 },
        'gal4' : { 'type' : 'InclinedSersic' , 'n' : 0.7, 'inclination' : 0.7 * galsim.radians,
                   'half_light_radius' : 1, 'scale_h_over_r' : 0.2, 'flux' : 50,
                   'dilate' : 3, 'ellip' : galsim.Shear(e1=0.3),
                   'rotate' : 12 * galsim.degrees,
                   'magnify' : 1.03, 'shear' : galsim.Shear(g1=0.03, g2=-0.05),
                   'shift' : { 'type' : 'XY', 'x' : 0.7, 'y' :-1.2 }
                 },
        'gal5' : { 'type' : 'InclinedSersic' , 'n' : 0.7, 'inclination' : 0.7 * galsim.radians,
                   'half_light_radius' : 1, 'flux' : 50,
                   'gsparams' : { 'minimum_fft_size' : 256 }
                 },
    }

    gal1a = galsim.config.BuildGSObject(config, 'gal1')[0]
    gal1b = galsim.InclinedSersic(n=1.2, inclination=0.1 * galsim.radians, half_light_radius=2)
    gsobject_compare(gal1a, gal1b)

    gal2a = galsim.config.BuildGSObject(config, 'gal2')[0]
    gal2b = galsim.InclinedSersic(n=3.5, inclination=21 * galsim.degrees, scale_radius=0.007,
                                  flux=100)
    gsobject_compare(gal2a, gal2b)

    gal3a = galsim.config.BuildGSObject(config, 'gal3')[0]
    gal3b = galsim.InclinedSersic(n=2.2, inclination=0.3 * galsim.radians, scale_radius=0.35,
                                  scale_height=0.23, flux=1.e6)
    gal3b = gal3b.shear(q=0.6, beta=0.39 * galsim.radians)
    gsobject_compare(gal3a, gal3b)

    gal4a = galsim.config.BuildGSObject(config, 'gal4')[0]
    gal4b = galsim.InclinedSersic(n=0.7, inclination=0.7 * galsim.radians, half_light_radius=1,
                                  scale_h_over_r=0.2, flux=50)
    gal4b = gal4b.dilate(3).shear(e1=0.3).rotate(12 * galsim.degrees)
    gal4b = gal4b.lens(0.03, -0.05, 1.03).shift(dx=0.7, dy=-1.2)
    gsobject_compare(gal4a, gal4b)

    gal5a = galsim.config.BuildGSObject(config, 'gal5')[0]
    gsparams = galsim.GSParams(minimum_fft_size=256)
    gal5b = galsim.InclinedSersic(n=0.7, inclination=0.7 * galsim.radians, half_light_radius=1,
                                  flux=50, gsparams=gsparams)
    gsobject_compare(gal5a, gal5b, conv=galsim.Gaussian(sigma=1))

    # Make sure they don't match when using the default GSParams
    gal5c = galsim.InclinedSersic(n=0.7, inclination=0.7 * galsim.radians, half_light_radius=1,
                                  flux=50)
    with assert_raises(AssertionError):
        gsobject_compare(gal5a, gal5c, conv=galsim.Gaussian(sigma=1))


@timer
def test_pixel():
    """Test various ways to build a Pixel
    """
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

@timer
def test_realgalaxy():
    """Test various ways to build a RealGalaxy
    """
    # I don't want to gratuitously copy the real_catalog catalog, so use the
    # version in the examples directory.
    real_gal_dir = os.path.join('..','examples','data')
    real_gal_cat = 'real_galaxy_catalog_23.5_example.fits'
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
        'gal5' : { 'type' : 'RealGalaxy' , 'index' : 23, 'noise_pad_size' : 10 },
        'gal6' : { 'type' : 'RealGalaxyOriginal' },
        'gal7' : { 'type' : 'RealGalaxy' , 'random' : True},
        # I admit the one below is odd (why would you specify "random" and have it be False?) but
        # one could imagine setting it based on some probabilistic process...
        'gal8' : { 'type' : 'RealGalaxy' , 'random' : False},
        'bad1' : { 'type' : 'RealGalaxy' , 'index' : -3 },
        'bad2' : { 'type' : 'RealGalaxy' , 'index' : 3000 },
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

    # Should work if rng not in base config dict.
    del config['rng']
    galsim.config.RemoveCurrent(config)   # Clear the cached values, so it rebuilds.
    galsim.config.BuildGSObject(config, 'gal5')

    # If there is a logger, there should be a warning message emitted.
    with CaptureLog() as cl:
        galsim.config.RemoveCurrent(config)
        galsim.config.BuildGSObject(config, 'gal5', logger=cl.logger)
    assert "No base['rng'] available" in cl.output

    config['obj_num'] = 5
    gal6a = galsim.config.BuildGSObject(config, 'gal6')[0]
    gal6b = galsim.RealGalaxy(real_cat, index=5).original_gal
    gsobject_compare(gal6a, gal6b, conv=conv)

    config['obj_num'] = 6
    # Since we are comparing the random functionality, we need to reset the RNG.
    config['rng'] = galsim.UniformDeviate(1234)
    gal7a = galsim.config.BuildGSObject(config, 'gal7')[0]
    gal7b = galsim.RealGalaxy(real_cat, random=True, rng=galsim.BaseDeviate(1234))
    gsobject_compare(gal7a, gal7b, conv=conv)

    config['obj_num'] = 7
    gal8a = galsim.config.BuildGSObject(config, 'gal8')[0]
    gal8b = galsim.RealGalaxy(real_cat, index=7)
    gsobject_compare(gal8a, gal8b, conv=conv)

    with assert_raises(galsim.GalSimConfigError):
        galsim.config.BuildGSObject(config, 'bad1')
    with assert_raises(galsim.GalSimConfigError):
        galsim.config.BuildGSObject(config, 'bad2')


@timer
def test_cosmosgalaxy():
    """Test various ways to build a COSMOSGalaxy
    """
    # I don't want to gratuitously copy the real_catalog catalog, so use the
    # version in the examples directory.
    real_gal_dir = os.path.join('..','examples','data')
    real_gal_cat = 'real_galaxy_catalog_23.5_example.fits'
    config = {

        'input' : {
            'cosmos_catalog' :
                { 'dir' : real_gal_dir ,
                  'file_name' : real_gal_cat,
                  'preload' : True
                },
            'galaxy_sample' :
                { 'dir' : real_gal_dir ,
                  'file_name' : real_gal_cat,
                  'preload' : True
                },
        },

        # First one uses defaults for gal_type (real, since we used the actual catalog and not the
        # parametric one) and selects a random galaxy using internal routines
        # (the default if index is unspecified).
        'gal1' : { 'type' : 'COSMOSGalaxy', 'scale_flux' : 3.14 },

        # Second uses parametric gal_type and selects a random galaxy using the config sequence
        # option.  Includes flux modifications and rotation.
        'gal2' : { 'type' : 'COSMOSGalaxy', 'gal_type' : 'parametric',
                   'index' : { 'type' : 'Sequence', 'nitems' : 1},
                   'scale_flux' : 0.3,
                   'rotate' : 30 * galsim.degrees },

        # Third uses parametric gal_type and a specific galaxy index.  Includes flux modifications,
        # shear and magnification.
        'gal3' : {'type' : 'COSMOSGalaxy', 'gal_type' : 'parametric',
                  'index' : 27, 'scale_flux' : 1.e6,
                  'magnify' : 0.9, 'shear' : galsim.Shear(g1=0.01, g2=-0.07)},

        # Fourth tries to select outside the catalog; make sure the exception is caught.
        'gal4' : {'type' : 'COSMOSGalaxy', 'gal_type' : 'parametric',
                  'index' : 1001},

        # Same as gal2, but using SampleGalaxy
        'gal2s' : { 'type' : 'SampleGalaxy', 'gal_type' : 'parametric',
                    'index' : { 'type' : 'Sequence', 'nitems' : 1},
                    'scale_flux' : 0.3,
                    'rotate' : 30 * galsim.degrees },

        # Same as gal3, but using SampleGalaxy
        'gal3s' : {'type' : 'SampleGalaxy', 'gal_type' : 'parametric',
                   'index' : 27, 'scale_flux' : 1.e6,
                   'magnify' : 0.9, 'shear' : galsim.Shear(g1=0.01, g2=-0.07)},
    }
    rng = galsim.UniformDeviate(1234)
    config['rng'] = galsim.UniformDeviate(1234) # A second copy starting with the same seed.

    galsim.config.ProcessInput(config)

    cosmos_cat = galsim.COSMOSCatalog(
        dir=real_gal_dir, file_name=real_gal_cat, preload=True)

    # For these profiles, we convolve by a gaussian to smooth out the profile.
    # This makes the comparison much faster without changing the validity of the test.
    conv = galsim.Gaussian(sigma = 1)

    config['obj_num'] = 0
    gal1a = galsim.config.BuildGSObject(config, 'gal1')[0]
    gal1b = 3.14*cosmos_cat.makeGalaxy(rng=rng)
    gsobject_compare(gal1a, gal1b, conv=conv)
    # N.B. Don't test SampleGalaxy here, since the index is random, so we'd have to muck around
    # with the random numbers to make it match.

    config['obj_num'] = 1
    gal2a = galsim.config.BuildGSObject(config, 'gal2')[0]
    gal2b = cosmos_cat.makeGalaxy(index=0, gal_type='parametric', rng=rng)
    gal2b = gal2b.withScaledFlux(0.3).rotate(30*galsim.degrees)
    gsobject_compare(gal2a, gal2b, conv=conv)
    gal2s = galsim.config.BuildGSObject(config, 'gal2s')[0]
    gsobject_compare(gal2s, gal2b, conv=conv)

    config['obj_num'] = 2
    gal3a = galsim.config.BuildGSObject(config, 'gal3')[0]
    gal3b = cosmos_cat.makeGalaxy(index=27, gal_type='parametric', rng=rng)
    gal3b = gal3b.withScaledFlux(1.e6).magnify(0.9).shear(g1=0.01, g2=-0.07)
    gsobject_compare(gal3a, gal3b, conv=conv)
    gal3s = galsim.config.BuildGSObject(config, 'gal3s')[0]
    gsobject_compare(gal3s, gal3b, conv=conv)

    config['obj_num'] = 3
    with assert_raises(galsim.GalSimConfigError):
        galsim.config.BuildGSObject(config, 'gal4')

    # Check some subtleties about using COSMOSGalaxy in a multiprocessing context
    config = {
        'input' : { 'cosmos_catalog' : { 'dir' : real_gal_dir , 'file_name' : real_gal_cat, } },
        'gal' : { 'type' : 'COSMOSGalaxy', 'scale_flux' : 3.14 },
        'psf' : { 'type' : 'Moffat', 'fwhm' : 0.1, 'beta' : 3.5 },
        'image' : {
            'type' : 'Tiled',
            'nx_tiles' : 3,
            'ny_tiles' : 3,
            'stamp_size' : 64,
            'pixel_scale' : 0.04,
            'random_seed' : 1234,
        },
        'output' : { 'file_name' : 'test_cosmosgalaxy.fits', 'dir': 'output' }
    }
    im1 = galsim.config.BuildImage(config)
    config['image']['nproc'] = 3
    del config['_input_objs']
    im2 = galsim.config.BuildImage(config)
    np.testing.assert_array_equal(im1.array, im2.array)

    # Make sure that if we specified from the start not to use real galaxies, that
    # failure to specify gal_type is treated properly (should default to parametric).
    real_gal_cat = 'real_galaxy_catalog_23.5_example_fits.fits'
    config = {

        'input' : { 'cosmos_catalog' :
                    { 'dir' : real_gal_dir ,
                      'file_name' : real_gal_cat,
                      'use_real' : False,
                      'preload' : True}
                    },

        # Use defaults for gal_type (parametric, since we used the actual catalog and not the
        # parametric one) and select a random galaxy using internal routines.
        'gal' : { 'type' : 'COSMOSGalaxy' },
    }
    rng = galsim.UniformDeviate(1234)
    config['rng'] = galsim.UniformDeviate(1234) # A second copy starting with the same seed.

    galsim.config.ProcessInput(config)

    cosmos_cat = galsim.COSMOSCatalog(
        dir=real_gal_dir, file_name=real_gal_cat, use_real=False, preload=True)

    config['obj_num'] = 0
    # It is going to complain that it doesn't have weight factors.  We want to ignore this.
    with assert_warns(galsim.GalSimWarning):
        gal1a = galsim.config.BuildGSObject(config, 'gal')[0]
        gal1b = cosmos_cat.makeGalaxy(rng=rng)
    gsobject_compare(gal1a, gal1b, conv=conv)

    # Check some information passed to the logger about the kind of catalog being built.
    galsim.config.SetupConfigImageNum(config, 0, 0)
    with CaptureLog() as cl:
        galsim.config.SetupInputsForImage(config, logger=cl.logger)
    assert "Using user-specified COSMOSCatalog" in cl.output
    config['input']['cosmos_catalog']['sample'] = 25.2
    with CaptureLog() as cl:
        galsim.config.SetupInputsForImage(config, logger=cl.logger)
    assert "sample = 25.2" in cl.output
    config['input']['cosmos_catalog'] = {}
    with CaptureLog() as cl:
        galsim.config.SetupInputsForImage(config, logger=cl.logger)
    assert "Using user-specified" not in cl.output
    config['gal']['gal_type'] = 'parametric'
    with CaptureLog() as cl:
        galsim.config.SetupInputsForImage(config, logger=cl.logger)
    assert "Using parametric galaxies" in cl.output
    config['gal']['gal_type'] = 'real'
    with CaptureLog() as cl:
        galsim.config.SetupInputsForImage(config, logger=cl.logger)
    assert "Using real galaxies" in cl.output


@timer
def test_cosmos_redshift():
    """Test accessing the redshift of a COSMOSGalaxy.  Useful e.g. when using NFWHalo.
    """

    # First check with a custom value type specialized to just zphot.
    def COSMOS_ZPhot(config, base, value_type):
        cosmos_cat = galsim.config.GetInputObj('cosmos_catalog', config, base, 'COSMOS_ZPhot')
        # Requires that galaxy uses an explicit index.
        # This will raise an exception if index is not set.
        index = galsim.config.GetCurrentValue('gal.index', base, value_type=int)
        record = cosmos_cat.getParametricRecord(index)
        zphot = record['zphot']
        # Note: this intentionally omits safe, which is a mistake that is likely to be
        # common for use-defined generators.  We handle it by assuming safe=False.
        return zphot

    # Gratuitously use a list for the input_type to test that that works.
    galsim.config.RegisterValueType('COSMOS_ZPhot', COSMOS_ZPhot, [float], ['cosmos_catalog'])

    real_gal_dir = os.path.join('..','examples','data')
    real_gal_cat = 'real_galaxy_catalog_23.5_example.fits'

    halo_mass = 1.e14  # Solar masses
    halo_conc = 4
    halo_z = 0.3
    config = {

        'input' : {
            'cosmos_catalog' : {
                'dir' : real_gal_dir ,
                'file_name' : real_gal_cat,
                'use_real' : False,
                'preload' : True
            },
            'galaxy_sample' : {
                'dir' : real_gal_dir ,
                'file_name' : real_gal_cat,
                'use_real' : False,
                'preload' : True
            },
            'nfw_halo' : {
                'mass' : halo_mass,
                'conc' : halo_conc,
                'redshift' : halo_z,
            },
         },

        'gal' : {
            'type' : 'COSMOSGalaxy',
            # Note: not all redshifts are valid with NFWHalo.  It requires z>0.2.
            # This index has z>0.2.
            'index' : 64,
            'shear' : { 'type' : 'NFWHaloShear' },
            'magnify' : { 'type' : 'NFWHaloMagnification' },
            'redshift' :  { 'type' : 'COSMOS_ZPhot' },
        },

        'gal2' : {
            'type' : 'COSMOSGalaxy',
            'index' : 64,
            'shear' : { 'type' : 'NFWHaloShear' },
            'magnify' : { 'type' : 'NFWHaloMagnification' },
            'redshift' :  { 'type' : 'COSMOSValue', 'key' : 'zphot', 'index': '@gal2.index' },
        },

        'gal3' : {
            'type' : 'SampleGalaxy',
            'index' : 64,
            'shear' : { 'type' : 'NFWHaloShear' },
            'magnify' : { 'type' : 'NFWHaloMagnification' },
            'redshift' :  { 'type' : 'SampleValue', 'key' : 'zphot', 'index': '@gal3.index' },
        },

        'image' : {
            'type' : 'Scattered',
            'size' : 256,
            'stamp_size' : 32,
            'nobjects' : 1,
            'random_seed' : 12345,
            'image_pos' : {
                'type' : 'XY' ,
                'x' : { 'type' : 'Random' , 'min' : 0 , 'max' : 256 },
                'y' : { 'type' : 'Random' , 'min' : 0 , 'max' : 256 },
            }

        }
    }

    logger = logging.getLogger('test_cosmos_redshift')
    logger.addHandler(logging.StreamHandler(sys.stdout))
    logger.setLevel(logging.DEBUG)

    galsim.config.ProcessInput(config, logger=logger)
    galsim.config.SetupConfigImageNum(config, 0, 0)
    galsim.config.SetupInputsForImage(config, logger=logger)

    cosmos_cat = galsim.COSMOSCatalog(
        dir=real_gal_dir, file_name=real_gal_cat, use_real=False, preload=True)
    sample_cat = galsim.GalaxySample(
        dir=real_gal_dir, file_name=real_gal_cat, use_real=False, preload=True)

    galsim.config.SetupConfigObjNum(config, 0, logger=logger)
    galsim.config.SetupConfigRNG(config, seed_offset=1, logger=logger)
    image_pos = galsim.config.ParseValue(config['image'], 'image_pos', config, galsim.PositionD)[0]
    galsim.config.SetupConfigStampSize(config,32,32,image_pos,None, logger=logger)
    gal1a = galsim.config.BuildGSObject(config, 'gal', logger=logger)[0]
    print('gal1a = ',gal1a)
    first_seed = galsim.BaseDeviate(12345).raw()
    rng = galsim.UniformDeviate(first_seed+1)
    x = float(rng() * 256)
    y = float(rng() * 256)
    assert x == image_pos.x
    assert y == image_pos.y
    index = 64
    gal1b = cosmos_cat.makeGalaxy(index)
    redshift = cosmos_cat.getParametricRecord(index)['zphot']
    nfw_halo = galsim.NFWHalo(mass=halo_mass, conc=halo_conc, redshift=halo_z)
    g1,g2 = nfw_halo.getShear(image_pos, redshift)
    mag = nfw_halo.getMagnification(image_pos, redshift)
    gal1b = gal1b.shear(g1=g1, g2=g2).magnify(mag)
    print('gal1b = ',gal1b)
    gsobject_compare(gal1a, gal1b)

    gal2 = galsim.config.BuildGSObject(config, 'gal2', logger=logger)[0]
    gsobject_compare(gal2, gal1b)
    gal3 = galsim.config.BuildGSObject(config, 'gal3', logger=logger)[0]
    gsobject_compare(gal3, gal1b)

@timer
def test_interpolated_image():
    """Test various ways to build an InterpolatedImage
    """
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
                   'noise_pad' : 0.001, 'noise_pad_size' : 64,
                 },
        'gal6' : { 'type' : 'InterpolatedImage', 'image' : file_name,
                   'noise_pad' : 'fits_files/blankimg.fits', 'noise_pad_size' : 64,
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
    gal5b = galsim.InterpolatedImage(im, rng=rng, noise_pad=0.001, noise_pad_size=64)
    gsobject_compare(gal5a, gal5b)

    gal6a = galsim.config.BuildGSObject(config, 'gal6')[0]
    gal6b = galsim.InterpolatedImage(im, rng=rng, noise_pad='fits_files/blankimg.fits',
                                     noise_pad_size=64)
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

    # gal5, gal6 should work with default rngs
    del config['rng']
    galsim.config.RemoveCurrent(config)   # Clear the cached values, so it rebuilds.
    galsim.config.BuildGSObject(config, 'gal5')
    galsim.config.BuildGSObject(config, 'gal6')

    # If there is a logger, there should be a warning message emitted, but only the first time.
    with CaptureLog() as cl:
        galsim.config.RemoveCurrent(config)
        galsim.config.BuildGSObject(config, 'gal5', logger=cl.logger)
    assert "No base['rng'] available" in cl.output
    with CaptureLog(level=1) as cl:
        galsim.config.RemoveCurrent(config)
        galsim.config.BuildGSObject(config, 'gal6', logger=cl.logger)
    assert cl.output == ''

@timer
def test_add():
    """Test various ways to build a Add
    """
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
        },
        'gal6' : {
            'type' : 'Add' ,
            'items' : [
                { 'type' : 'Gaussian' , 'sigma' : 2 },
            ]
        },
        'gal7' : {
            'type' : 'Add' ,
            'items' : [
                { 'type' : 'Gaussian' , 'sigma' : 2 },
                { 'type' : 'Exponential' , 'half_light_radius' : 2.3, 'flux' : 0 }
            ]
        },
        'gal8' : {
            'type' : 'Add' ,
            'items' : [
                { 'type' : 'Gaussian', 'sigma' : 2, 'flux' : 0.3 },
                { 'type' : 'Exponential', 'half_light_radius' : 2.3, 'flux' : 0.5 },
                { 'type' : 'Sersic', 'n': 3, 'half_light_radius' : 1.2 }
            ],
            'flux' : 170.
        },
        'bad1' : {
            'type' : 'Add' ,
            'items' : { 'type' : 'Gaussian', 'sigma' : 2, 'flux' : 0.3 },
        },
        'bad2' : {
            'type' : 'Add' ,
            'items' :  [],
        },
        'bad3' : {
            'type' : 'Add',
            'items' :  'invalid',
        },
        'bad4' : {
            'type' : 'Add',
        },
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

    # Make sure they don't match when using the default GSParams
    gal5c_1 = galsim.Exponential(scale_radius=3.4, flux=100)
    gal5c_2 = galsim.Gaussian(sigma=1, flux=50)
    gal5c = galsim.Add([gal5c_1, gal5c_2])
    with assert_raises(AssertionError):
        gsobject_compare(gal5a, gal5c, conv=galsim.Gaussian(sigma=1))

    # "Adding" 1 item is equivalent to just that item alone
    gal6a = galsim.config.BuildGSObject(config, 'gal6')[0]
    gal6b = galsim.Gaussian(sigma = 2)
    gsobject_compare(gal6a, gal6b)

    # Also if an item has 0 flux, it is ignored (for efficiency)
    gal7a = galsim.config.BuildGSObject(config, 'gal7')[0]
    gal7b = galsim.Gaussian(sigma = 2)
    gsobject_compare(gal7a, gal7b)

    # If the last flux is omitted, then it is set to make the toal = 1.
    gal8a = galsim.config.BuildGSObject(config, 'gal8')[0]
    gal8b_1 = galsim.Gaussian(sigma = 2, flux = 0.3)
    gal8b_2 = galsim.Exponential(half_light_radius = 2.3, flux = 0.5)
    gal8b_3 = galsim.Sersic(n = 3, half_light_radius = 1.2, flux = 0.2)
    gal8b = galsim.Add([gal8b_1, gal8b_2, gal8b_3])
    gal8b = gal8b.withFlux(170)
    gsobject_compare(gal8a, gal8b)

    # If the sum comes out larger than 1, emit a warning
    config['gal8']['items'][1]['flux'] = 0.9
    galsim.config.RemoveCurrent(config)
    with CaptureLog() as cl:
        galsim.config.BuildGSObject(config, 'gal8', logger=cl.logger)
    assert ("Warning: Automatic flux for the last item in Sum (to make the total flux=1) " +
            "resulted in negative flux = -0.200000 for that item") in cl.output

    with assert_raises(galsim.GalSimConfigError):
        galsim.config.BuildGSObject(config, 'bad1')
    with assert_raises(galsim.GalSimConfigError):
        galsim.config.BuildGSObject(config, 'bad2')
    with assert_raises(galsim.GalSimConfigError):
        galsim.config.BuildGSObject(config, 'bad3')
    with assert_raises(galsim.GalSimConfigError):
        galsim.config.BuildGSObject(config, 'bad4')


@timer
def test_convolve():
    """Test various ways to build a Convolve
    """
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
        },
        'gal6' : {
            'type' : 'Convolve' ,
            'items' : [
                { 'type' : 'Gaussian' , 'sigma' : 2 },
            ]
        },
        'bad1' : {
            'type' : 'Convolve' ,
            'items' : { 'type' : 'Gaussian', 'sigma' : 2, 'flux' : 0.3 },
        },
        'bad2' : {
            'type' : 'Convolve' ,
            'items' :  [],
        },
        'bad3' : {
            'type' : 'Convolve' ,
            'items' :  'invalid',
        },
        'bad4' : {
            'type' : 'Convolve' ,
        },
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

    # Make sure they don't match when using the default GSParams
    gal5c_1 = galsim.Exponential(scale_radius=1.7, flux=100)
    gal5c_2 = galsim.Gaussian(sigma=1)
    gal5c = galsim.Convolve([gal5c_1, gal5c_2])
    with assert_raises(AssertionError):
        gsobject_compare(gal5a, gal5c)

    # "Convolving" 1 item is equivalent to just that item alone
    gal6a = galsim.config.BuildGSObject(config, 'gal6')[0]
    gal6b = galsim.Gaussian(sigma = 2)
    gsobject_compare(gal6a, gal6b)

    with assert_raises(galsim.GalSimConfigError):
        galsim.config.BuildGSObject(config, 'bad1')
    with assert_raises(galsim.GalSimConfigError):
        galsim.config.BuildGSObject(config, 'bad2')
    with assert_raises(galsim.GalSimConfigError):
        galsim.config.BuildGSObject(config, 'bad3')
    with assert_raises(galsim.GalSimConfigError):
        galsim.config.BuildGSObject(config, 'bad4')


@timer
def test_list():
    """Test building a GSObject from a list:
    """
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

    # Check that the list items correctly inherit their gsparams and flux from the top level
    config = {
        'gal' : {
            'type' : 'List' ,
            'flux' : 100,
            'items' : [
                { 'type' : 'Exponential' , 'scale_radius' : 3.4 },
                { 'type' : 'Exponential' , 'scale_radius' : 3 }
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

    # Make sure they don't match when using the default GSParams
    gal5c = galsim.Exponential(scale_radius=3.4, flux=100)
    with assert_raises(AssertionError):
        gsobject_compare(gal5a, gal5c, conv=galsim.Gaussian(sigma=1))

    config = {
        'bad1' : { 'type' : 'List',
                   'items' : { 'type' : 'Exponential' , 'scale_radius' : 3.4, 'flux' : 100 } },
        'bad2' : { 'type' : 'List', 'items' : [], },
        'bad3' : { 'type' : 'List', 'items' : 'invalid', },
        'bad4' : { 'type' : 'List', },
        'bad5' : {
            'type' : 'List' ,
            'items' : [ { 'type' : 'Gaussian' , 'sigma' : 2 },
                        { 'type' : 'Gaussian' , 'fwhm' : 2, 'flux' : 100 }, ],
            'index' : -1,
        },
        'bad6' : {
            'type' : 'List' ,
            'items' : [ { 'type' : 'Gaussian' , 'sigma' : 2 },
                        { 'type' : 'Gaussian' , 'fwhm' : 2, 'flux' : 100 }, ],
            'index' : 2,
        },
    }
    with assert_raises(galsim.GalSimConfigError):
        galsim.config.BuildGSObject(config, 'bad1')
    with assert_raises(galsim.GalSimConfigError):
        galsim.config.BuildGSObject(config, 'bad2')
    with assert_raises(galsim.GalSimConfigError):
        galsim.config.BuildGSObject(config, 'bad3')
    with assert_raises(galsim.GalSimConfigError):
        galsim.config.BuildGSObject(config, 'bad4')
    with assert_raises(galsim.GalSimConfigError):
        galsim.config.BuildGSObject(config, 'bad5')
    with assert_raises(galsim.GalSimConfigError):
        galsim.config.BuildGSObject(config, 'bad6')


@timer
def test_repeat():
    """Test use of the repeat option for an object
    """
    config = {
        'rng' : galsim.BaseDeviate(1234),
        'gal' : {
            'repeat' : 3,
            'type' : 'Gaussian',
            'sigma' : { 'type' : 'Random', 'min' : 1, 'max' : 2 },
            'flux' : '$(obj_num + 1) * 100'
        }
    }

    ud = galsim.UniformDeviate(1234)
    config['obj_num'] = 0
    gal1a = galsim.config.BuildGSObject(config, 'gal')[0]
    gal1b = galsim.Gaussian(sigma=ud()+1, flux=100)
    gsobject_compare(gal1a, gal1b)

    # Next 2 should be the same.
    config['obj_num'] = 1
    gal1a = galsim.config.BuildGSObject(config, 'gal')[0]
    gsobject_compare(gal1a, gal1b)
    config['obj_num'] = 2
    gal1a = galsim.config.BuildGSObject(config, 'gal')[0]
    gsobject_compare(gal1a, gal1b)

    # Then next 3 should be a new object.
    config['obj_num'] = 3
    gal2a = galsim.config.BuildGSObject(config, 'gal')[0]
    gal2b = galsim.Gaussian(sigma=ud()+1, flux=400)
    gsobject_compare(gal2a, gal2b)
    config['obj_num'] = 4
    gal2a = galsim.config.BuildGSObject(config, 'gal')[0]
    gsobject_compare(gal2a, gal2b)

    # Also check that the logger reports why it is using the current object
    config['obj_num'] = 5
    with CaptureLog() as cl:
        gal2a = galsim.config.BuildGSObject(config, 'gal', logger=cl.logger)[0]
    gsobject_compare(gal2a, gal2b)
    assert "repeat = 3, index = 5, use current object" in cl.output


@timer
def test_usertype():
    """Test a user-defined type
    """
    # A custom GSObject class that will use BuildSimple
    class PseudoDelta(galsim.Gaussian):
        _req_params = {}
        _opt_params = { "flux" : float }
        _single_params = []
        _takes_rng = False
        def __init__(self, flux=1., gsparams=None):
            super(PseudoDelta, self).__init__(sigma=1.e-8, flux=flux, gsparams=gsparams)

    galsim.config.RegisterObjectType('PseudoDelta', PseudoDelta)

    config = {
        'gal1' : { 'type' : 'PseudoDelta' },
        'gal2' : { 'type' : 'PseudoDelta', 'flux' : 100 },
        'gal3' : { 'type' : 'PseudoDelta', 'flux' : 1.e5,
                   'shift' : { 'type' : 'XY', 'x' : 0.7, 'y' : -1.2 }
                 },
    }

    psf = galsim.Gaussian(sigma=2.3)

    gal1a = galsim.config.BuildGSObject(config, 'gal1')[0]
    gal1b = galsim.Gaussian(sigma=1.e-8, flux=1)
    gsobject_compare(gal1a, gal1b, conv=psf)

    gal2a = galsim.config.BuildGSObject(config, 'gal2')[0]
    gal2b = galsim.Gaussian(sigma=1.e-8, flux = 100)
    gsobject_compare(gal2a, gal2b, conv=psf)

    gal3a = galsim.config.BuildGSObject(config, 'gal3')[0]
    gal3b = galsim.Gaussian(sigma=1.e-8, flux = 1.e5)
    gal3b = gal3b.shift(dx = 0.7, dy = -1.2)
    gsobject_compare(gal3a, gal3b, conv=psf)

    # Now an equivalent thing, but implemented with a builder rather than a class.
    def BuildPseudoDelta(config, base, ignore, gsparams, logger):
        opt = { 'flux' : float }
        kwargs, safe = galsim.config.GetAllParams(config, base, opt=opt, ignore=ignore)
        gsparams = galsim.GSParams(**gsparams)  # within config, it is passed around as a dict
        return galsim.Gaussian(sigma=1.e-8, gsparams=gsparams, **kwargs), safe

    galsim.config.RegisterObjectType('PseudoDelta', BuildPseudoDelta)

    galsim.config.RemoveCurrent(config)   # Clear the cached values, so it rebuilds.
    gal1a = galsim.config.BuildGSObject(config, 'gal1')[0]
    gsobject_compare(gal1a, gal1b, conv=psf)

    gal2a = galsim.config.BuildGSObject(config, 'gal2')[0]
    gsobject_compare(gal2a, gal2b, conv=psf)

    gal3a = galsim.config.BuildGSObject(config, 'gal3')[0]
    gsobject_compare(gal3a, gal3b, conv=psf)

    # This one is potentially more useful.
    # This is now equivalent to the Eval type, but it started its life as a user type,
    # so keep it here with a different name.
    def EvalGSObject(config, base, ignore, gsparams, logger):
        req = { 'str': str }
        params, safe = galsim.config.GetAllParams(config, base, req=req, ignore=ignore)
        return galsim.utilities.math_eval(params['str']).withGSParams(**gsparams), safe

    galsim.config.RegisterObjectType('EvalX', EvalGSObject)

    config = {
        'gal1' : { 'type': 'EvalX', 'str': 'galsim.Gaussian(fwhm=0.3, flux=10)'  },
        'gal2' : { 'type': 'EvalX', 'str': 'galsim.DeltaFunction(432)',
                   'gsparams' : { 'folding_threshold' : 1.e-4 } },
        # Also test the now-standard Eval type.
        'gal1c' : { 'type': 'Eval', 'str': 'galsim.Gaussian(fwhm=0.3, flux=10)'  },
        'gal2c' : { 'type': 'Eval', 'str': 'galsim.DeltaFunction(432)',
                    'gsparams' : { 'folding_threshold' : 1.e-4 } },
    }

    gal1a = galsim.config.BuildGSObject(config, 'gal1')[0]
    gal1b = galsim.Gaussian(fwhm=0.3, flux=10)
    gal1c = galsim.config.BuildGSObject(config, 'gal1c')[0]
    gsobject_compare(gal1a, gal1b)
    gsobject_compare(gal1a, gal1c)

    gal2a = galsim.config.BuildGSObject(config, 'gal2')[0]
    gal2b = galsim.DeltaFunction(432, gsparams=galsim.GSParams(folding_threshold=1.e-4))
    gal2c = galsim.config.BuildGSObject(config, 'gal2c')[0]
    gsobject_compare(gal2a, gal2b, conv=galsim.Gaussian(fwhm=0.3))
    gsobject_compare(gal2a, gal2c, conv=galsim.Gaussian(fwhm=0.3))


@timer
def test_modules():
    """Test using modules in the config dict to get extra types
    """
    # This is basically a duplicate of test_psf_config in test_des.py,
    # except we dont import galsim.des in this file, so this wouldn't work without
    # the modules directive.
    data_dir = 'des_data'
    psfex_file = "DECam_00154912_12_psfcat.psf"
    fitpsf_file = "DECam_00154912_12_fitpsf.fits"
    wcs_file = "DECam_00154912_12_header.fits"

    image_pos = galsim.PositionD(123.45, 543.21)

    config = {
        'modules' : ['galsim.des', 'datetime'],
        'input' : {
            'des_shapelet' : { 'dir' : data_dir, 'file_name' : fitpsf_file },
            'des_psfex' : [
                { 'dir' : data_dir, 'file_name' : psfex_file },
                { 'dir' : data_dir, 'file_name' : psfex_file, 'image_file_name' : wcs_file },
            ]
        },

        'psf1' : { 'type' : 'DES_Shapelet' },
        'psf2' : { 'type' : 'DES_PSFEx', 'num' : 0 },
        'psf3' : { 'type' : 'DES_PSFEx', 'num' : 1 },
        'psf4' : { 'type' : 'DES_Shapelet', 'image_pos' : galsim.PositionD(567,789), 'flux' : 179,
                   'gsparams' : { 'folding_threshold' : 1.e-4 } },
        'psf5' : { 'type' : 'DES_PSFEx', 'image_pos' : galsim.PositionD(789,567),
                   # Check that imported modules can be used in an eval.
                   'flux' : '$datetime.datetime(2021, 5, 17).day',
                   'gsparams' : { 'folding_threshold' : 1.e-4 } },

        # This would normally be set by the config processing.  Set it manually here.
        'image_pos' : image_pos,
    }

    galsim.config.ImportModules(config)
    galsim.config.ProcessInput(config)

    psf1a = galsim.config.BuildGSObject(config, 'psf1')[0]
    fitpsf = galsim.des.DES_Shapelet(fitpsf_file, dir=data_dir)
    psf1b = fitpsf.getPSF(image_pos)
    gsobject_compare(psf1a, psf1b)

    psf2a = galsim.config.BuildGSObject(config, 'psf2')[0]
    psfex0 = galsim.des.DES_PSFEx(psfex_file, dir=data_dir)
    psf2b = psfex0.getPSF(image_pos)
    gsobject_compare(psf2a, psf2b)

    psf3a = galsim.config.BuildGSObject(config, 'psf3')[0]
    psfex1 = galsim.des.DES_PSFEx(psfex_file, wcs_file, dir=data_dir)
    psf3b = psfex1.getPSF(image_pos)
    gsobject_compare(psf3a, psf3b)

    gsparams = galsim.GSParams(folding_threshold=1.e-4)
    psf4a = galsim.config.BuildGSObject(config, 'psf4')[0]
    psf4b = fitpsf.getPSF(galsim.PositionD(567,789),gsparams=gsparams).withFlux(179)
    gsobject_compare(psf4a, psf4b)

    # Insert a wcs for thes last one.
    config['wcs'] = galsim.FitsWCS(os.path.join(data_dir,wcs_file))
    config = galsim.config.CleanConfig(config)
    galsim.config.ProcessInput(config)
    psfex2 = galsim.des.DES_PSFEx(psfex_file, dir=data_dir, wcs=config['wcs'])
    psf5a = galsim.config.BuildGSObject(config, 'psf5')[0]
    psf5b = psfex2.getPSF(galsim.PositionD(789,567),gsparams=gsparams).withFlux(17)
    gsobject_compare(psf5a, psf5b)

    # The first item in sys.path is the current directory.
    # However, when running galsim as an executable, it's the python/bin directory,
    # So anything in the current directory isn't found.
    # Mock this up by removing the current first item from sys.path.
    save_sys_path = sys.path
    sys.path = sys.path[1:]
    config['modules'] = ['hsm_shape']
    galsim.config.ImportModules(config)
    assert 'hsm_shape' in sys.modules

    # Check that invalid modules raises an appropriate error
    config['modules'] = ['invalid']
    with assert_raises(ImportError):
        galsim.config.ImportModules(config)

    # Put it back how we found it.
    sys.path = save_sys_path

@timer
def test_sed():
    """Test various sed options"""
    config = {
        'sed1' : {
            'type' : 'FileSED',
            'file_name' : 'vega.txt',
            'wave_type' : 'nm',
            'flux_type' : 'fnu',
        },
        'sed2' : {
            'file_name' : 'CWW_E_ext.sed',
            'wave_type' : 'Ang',
            'flux_type' : 'flambda',
            'redshift' : 0.8,
        },
        'sed3' : galsim.SED('CWW_Sbc_ext.sed', 'Ang', 'flambda'),
        'sed4' : {
            'file_name' : 'CWW_Scd_ext.sed',
            'wave_type' : 'Ang',
            'flux_type' : 'flambda',
            'norm_flux_density' : 1.0,
            'norm_wavelength' : [500, 700, 900],
        },
        'sed5' : {
            'file_name' : 'CWW_Im_ext.sed',
            'wave_type' : 'Ang',
            'flux_type' : 'flambda',
            'norm_flux' : 50.0,
            'norm_bandpass' : {'file_name' : 'LSST_g.dat', 'wave_type' : 'nm'},
        },

        'current1' : '@sed1',
        'current2' : {
            'type' : 'Current',
            'key' : 'sed3',
        },

        'eval1' : '$@sed2 * 0.5',
        'eval2' : {
            'type' : 'Eval',
            'str' : "galsim.SED('1', 'nm', 'fphotons')",
        },

        'bad1' : 34,
        'bad2' : { 'type' : 'Invalid' },
    }
    config['index_key'] = 'obj_num'
    config['obj_num'] = 0

    sed1 = galsim.config.BuildSED(config, 'sed1', config)[0]
    assert sed1 == galsim.SED('vega.txt', 'nm', 'fnu')

    sed2 = galsim.config.BuildSED(config, 'sed2', config)[0]
    assert sed2 == galsim.SED('CWW_E_ext.sed', 'Ang', 'flambda').atRedshift(0.8)

    sed3 = galsim.config.BuildSED(config, 'sed3', config)[0]
    assert sed3 == galsim.SED('CWW_Sbc_ext.sed', 'Ang', 'flambda')

    sed4 = galsim.config.BuildSED(config, 'sed4', config)[0]
    sed4b = galsim.SED('CWW_Scd_ext.sed', 'Ang', 'flambda')
    sed4c = sed4b.withFluxDensity(1.0, wavelength=500)
    assert sed4 == sed4c

    sed5 = galsim.config.BuildSED(config, 'sed5', config)[0]
    sed5b = galsim.SED('CWW_Im_ext.sed', 'Ang', 'flambda')
    sed5c = sed4b.withFlux(50., bandpass=galsim.Bandpass('LSST_g.dat', 'nm'))

    sed6 = galsim.config.BuildSED(config, 'current1', config)[0]
    assert sed6 == sed1

    sed7 = galsim.config.BuildSED(config, 'current2', config)[0]
    assert sed7 == sed3

    sed8 = galsim.config.BuildSED(config, 'eval1', config)[0]
    assert sed8 == sed2 * 0.5

    sed9 = galsim.config.BuildSED(config, 'eval2', config)[0]
    assert sed9 == galsim.SED('1', 'nm', 'fphotons')

    # If we build something again with the same index, it should get the current value
    sed10 = galsim.config.BuildSED(config, 'sed4', config)[0]
    assert sed10 is sed4

    # But not if the index has changed.
    config['obj_num'] = 1
    sed11 = galsim.config.BuildSED(config, 'sed4', config)[0]
    assert sed11 is not sed4
    assert sed11 == sed4b.withFluxDensity(1.0, wavelength=700)

    for bad in ['bad1', 'bad2']:
        with assert_raises(galsim.GalSimConfigError):
            galsim.config.BuildSED(config, bad, config)

    # Base class usage is invalid
    builder = galsim.config.sed.SEDBuilder()
    assert_raises(NotImplementedError, builder.buildSED, config, config, logger=None)



if __name__ == "__main__":
    runtests(__file__)
