# Copyright (c) 2012-2018 by the GalSim developers team on GitHub
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

# A simple exponential fitter that we can use to time the k-space drawing function

from __future__ import print_function
import galsim
import scipy.optimize
import numpy

import time
import cProfile, pstats
pr = cProfile.Profile()

def draw_exp(params, image=None, gsparams=None, dtype=None):
    """Draw an Exponential profile in k-space onto the given image

    params = [ half_light_radius, flux, e1, e2, x0, y0 ]
    image is optional, but if provided will be used as is.
    """
    exp = galsim.Exponential(params[0], flux=params[1], gsparams=gsparams)
    exp = exp._shear(galsim._Shear(params[2] + 1j * params[3]))
    exp = exp._shift(galsim.PositionD(params[4],params[5]))
    if image is None:
        image = exp.drawKImage(dtype=dtype, nx=256, ny=256)
    else:
        image = exp._drawKImage(image)
    return image

def draw_spergel(params, image=None, gsparams=None, dtype=None):
    """Draw a Spergel profile in k-space onto the given image

    params = [ half_light_radius, flux, e1, e2, x0, y0 ]
    image is optional, but if provided will be used as is.
    """
    nu = 0.5
    gal = galsim.Spergel(nu, params[0], flux=params[1], gsparams=gsparams)
    gal = gal._shear(galsim._Shear(params[2] + 1j * params[3]))
    gal = gal._shift(galsim.PositionD(params[4],params[5]))
    if image is None:
        image = gal.drawKImage(dtype=dtype, nx=256, ny=256)
    else:
        image = gal._drawKImage(image)
    return image

def draw_sersic(params, image=None, gsparams=None, dtype=None):
    """Draw a Sersic profile in k-space onto the given image

    params = [ half_light_radius, flux, e1, e2, x0, y0 ]
    image is optional, but if provided will be used as is.
    """
    n = 1
    gal = galsim.Sersic(n, params[0], flux=params[1], gsparams=gsparams)
    gal = gal._shear(galsim._Shear(params[2] + 1j * params[3]))
    gal = gal._shift(galsim.PositionD(params[4],params[5]))
    if image is None:
        image = gal.drawKImage(dtype=dtype, nx=256, ny=256)
    else:
        image = gal._drawKImage(image)
    return image

draw = draw_spergel

def fit(image, guess=(1.,1.,0.,0.,0.,0.), tol=1.e-6):
    """Find the best fitting profile to the given k-space image
    """

    class resid(object):
        def __init__(self, image):
            self._gsp = galsim.GSParams(kvalue_accuracy=1.e-5)
            self._target_image = galsim.ImageCF(image)
            self._scratch_image = galsim.ImageCF(image)
            #self._target_image = galsim.ImageCD(image)
            #self._scratch_image = galsim.ImageCD(image)
        def __call__(self, params):
            if params[0] < 0. or abs(params[2]) > 1. or abs(params[3]) > 1.:
                return 1.e500
            try:
                draw(params, self._scratch_image, gsparams=self._gsp)
            except ValueError:
                return 1.e500
            a = self._scratch_image.array
            a -= self._target_image.array
            a **= 2
            chisq = numpy.sum(numpy.abs(a))
            #print(params,'  ',chisq)
            return chisq

    guess = numpy.array(guess)
    print('guess = ',guess)
    # With float images, the numerical derivatives fail for the default method.
    result = scipy.optimize.minimize(resid(image), guess, tol=tol, method='Nelder-Mead')
    print('result = ',result.x)
    print('number of iterations = ',result.nit)
    print('number of function evals = ',result.nfev)
    return result.x

true_params = [3.49, 99.123, 0.0812, -0.2345, 0.1, -0.5]
true_image_cd = draw(true_params, dtype=numpy.complex128)  # Do truth at double precision
if False:
    # Check that I didn't mess up the SSE stuff.
    true_image_cf = draw(true_params, dtype=numpy.complex64)
    print('cf = ',true_image_cf[galsim.BoundsI(-5,6,-5,6)].array)
    print('cd = ',true_image_cd[galsim.BoundsI(-5,6,-5,6)].array)
    print('diff = ',(true_image_cd-true_image_cf)[galsim.BoundsI(-5,6,-5,6)].array)
    print('max diff = ',numpy.max(numpy.abs(true_image_cd.array-true_image_cf.array)/numpy.sum(true_image_cd.array)))
    quit()

pr.enable()
t0 = time.time()
for n in range(1):
    fit_params = fit(true_image_cd, tol=1.e-6)
t1 = time.time()
pr.disable()

ps = pstats.Stats(pr).sort_stats('time')
#ps = pstats.Stats(pr).sort_stats('cumtime')
ps.print_stats(20)

print('True params = ',true_params)
print('Fitted params = ',fit_params.tolist())
print('time = ',t1-t0)
