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
"""@file integ.py
Includes a Python layer version of the C++ int1d() function in galim::integ,
and python image integrators for use in galsim.chromatic
"""

import numpy as np
from functools import reduce

from . import _galsim
from .errors import GalSimError, GalSimRangeError, GalSimValueError, convert_cpp_errors

def int1d(func, min, max, rel_err=1.e-6, abs_err=1.e-12):
    """Integrate a 1-dimensional function from min to max.

    Example usage:

        >>> def func(x): return x**2
        >>> galsim.integ.int1d(func, 0, 1)
        0.33333333333333337
        >>> galsim.integ.int1d(func, 0, 2)
        2.666666666666667
        >>> galsim.integ.int1d(func, -1, 1)
        0.66666666666666674

    @param func     The function to be integrated.  y = func(x) should be valid.
    @param min      The lower end of the integration bounds (anything < -1.e10 is treated as
                    negative infinity).
    @param max      The upper end of the integration bounds (anything > 1.e10 is treated as positive
                    infinity).
    @param rel_err  The desired relative error [default: 1.e-6]
    @param abs_err  The desired absolute error [default: 1.e-12]

    @returns the value of the integral.
    """
    min = float(min)
    max = float(max)
    rel_err = float(rel_err)
    abs_err = float(abs_err)
    with convert_cpp_errors():
        success, result = _galsim.PyInt1d(func,min,max,rel_err,abs_err)
    if success:
        return result
    else:
        raise GalSimError(result)

def midpt(fvals, x):
    """Midpoint rule for integration.

    @param fvals  Samples of the integrand
    @param x      Locations at which the integrand was sampled.

    @returns midpoint rule approximation of the integral.
    """
    dx = [x[1]-x[0]]
    dx.extend(0.5*(x[2:]-x[0:-2]))
    dx.append(x[-1]-x[-2])
    weighted_fvals = [w*f for w,f in zip(dx, fvals)]
    return reduce(lambda y,z:y+z, weighted_fvals)

def trapz(func, min, max, points=10000):
    """Simple wrapper around 'numpy.trapz' to take function and limits as inputs.

    Example usage:

        >>> def func(x): return x**2
        >>> galsim.integ.trapz(func, 0, 1)
        0.33333333500033341
        >>> galsim.integ.trapz(func, 0, 1, 1e6)
        0.33333333333349996
        >>> galsim.integ.trapz(func, 0, 1, np.linspace(0, 1, 1e3))
        0.33333350033383402

    @param func     The function to be integrated.  y = func(x) should be valid.
    @param min      The lower end of the integration bounds.
    @param max      The upper end of the integration bounds.
    @param points   If integer, the number of points to sample the integrand. If array-like, then
                    the points to sample the integrand at. [default: 1000].
    """
    if not np.isscalar(points):
        if (np.max(points) > max) or (np.min(points) < min):
            raise GalSimRangeError("Points outside of specified range", points, min, max)
    elif int(points) != points:
        raise TypeError("npoints must be integer type or array")
    else:
        points = np.linspace(min, max, points)

    return np.trapz(func(points),points)


def midptRule(f, xs):
    """Midpoint rule for integration.

    @param f   Function to integrate.
    @param xs  Locations at which to evaluate f.

    @returns  Midpoint rule approximation to the integral.
    """
    if len(xs) < 2:
        raise GalSimValueError("Not enough points for midptRule integration", xs)
    x, xp = xs[:2]
    result = f(x)*(xp-x)
    for x, xp, xpp in zip(xs[0:-2], xs[1:-1], xs[2:]):
        result += 0.5*f(xp)*(xpp-x)
    result += f(xpp)*(xpp-xp)
    return result


def trapzRule(f, xs):
    """Trapezoidal rule for integration.

    @param f   Function to integrate.
    @param xs  Locations at which to evaluate f.

    @returns  Trapezoidal rule approximation to the integral.
    """
    if len(xs) < 2:
        raise GalSimValueError("Not enough points for trapzRule integration", xs)
    x, xp = xs[:2]
    result = 0.5*f(x)*(xp-x)
    for x, xp, xpp in zip(xs[0:-2], xs[1:-1], xs[2:]):
        result += 0.5*f(xp)*(xpp-x)
    result += 0.5*f(xpp)*(xpp-xp)
    return result


class ImageIntegrator(object):
    def __init__(self):
        raise NotImplementedError("Must instantiate subclass of ImageIntegrator")
    # subclasses must define
    # 1) a method `.calculateWaves(bandpass)` which will return the wavelengths at which to
    #    evaluate the integrand
    # 2) an function attribute `.rule` which takes an integrand function as its first
    #    argument, and a list of evaluation wavelengths as its second argument, and returns
    #    an approximation to the integral.  (E.g., the function midptRule above)

    def __call__(self, evaluateAtWavelength, bandpass, image, drawImageKwargs, doK=False):
        """
        @param evaluateAtWavelength Function that returns a monochromatic surface brightness
                                    profile as a function of wavelength.
        @param bandpass             Bandpass object representing the filter being imaged through.
        @param image                Image used to set size and scale of output
        @param drawImageKwargs      dict with other kwargs to send to drawImage function.
        @param doK                  Integrate up results of drawKImage instead of results of
                                    drawImage.  [default: False]

        @returns the result of integral as an Image
        """
        waves = self.calculateWaves(bandpass)
        self.last_n_eval = len(waves)
        drawImageKwargs.pop('add_to_image', None) # Make sure add_to_image isn't in kwargs

        def integrand(w):
            prof = evaluateAtWavelength(w) * bandpass(w)
            if not doK:
                return prof.drawImage(image=image.copy(), **drawImageKwargs)
            else:
                return prof.drawKImage(image=image.copy(), **drawImageKwargs)
        return self.rule(integrand, waves)


class SampleIntegrator(ImageIntegrator):
    """Create a chromatic surface brightness profile integrator, which will integrate over
    wavelength using a Bandpass as a weight function.

    This integrator will evaluate the integrand only at the wavelengths in `bandpass.wave_list`.
    See ContinuousIntegrator for an integrator that evaluates the integrand at a given number of
    points equally spaced apart.

    @param rule         Which integration rule to apply to the wavelength and monochromatic surface
                        brightness samples.  Options include:
                            galsim.integ.midptRule  --  Use the midpoint integration rule
                            galsim.integ.trapzRule  --  Use the trapezoidal integration rule
    """
    def __init__(self, rule):
        self.rule = rule

    def calculateWaves(self, bandpass):
        return bandpass.wave_list


class ContinuousIntegrator(ImageIntegrator):
    """Create a chromatic surface brightness profile integrator, which will integrate over
    wavelength using a Bandpass as a weight function.

    This integrator will evaluate the integrand at a given number `N` of equally spaced
    wavelengths over the interval defined by bandpass.blue_limit and bandpass.red_limit.  See
    SampleIntegrator for an integrator that only evaluates the integrand at the predefined set of
    wavelengths in `bandpass.wave_list`.

    @param rule         Which integration rule to apply to the wavelength and monochromatic
                        surface brightness samples.  Options include:
                            galsim.integ.midptRule  --  Use the midpoint integration rule
                            galsim.integ.trapzRule  --  Use the trapezoidal integration rule
    @param N            Number of equally-wavelength-spaced monochromatic surface brightness
                        samples to evaluate. [default: 250]
    @param use_endpoints  Whether to sample the endpoints `bandpass.blue_limit` and
                        `bandpass.red_limit`.  This should probably be True for a rule like
                        numpy.trapz, which explicitly samples the integration limits.  For a
                        rule like the midpoint rule, however, the integration limits are not
                        generally sampled, (only the midpoint between each integration limit and
                        its nearest interior point is sampled), thus `use_endpoints` should be
                        set to False in this case.  [default: True]
    """
    def __init__(self, rule, N=250, use_endpoints=True):
        self.rule = rule
        self.N = N
        self.use_endpoints = use_endpoints

    def calculateWaves(self, bandpass):
        h = (bandpass.red_limit*1.0 - bandpass.blue_limit)/self.N
        if self.use_endpoints:
            return [bandpass.blue_limit + h * i for i in range(self.N+1)]
        else:
            return [bandpass.blue_limit + h * (i+0.5) for i in range(self.N)]
