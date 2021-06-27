# Copyright (c) 2012-2021 by the GalSim developers team on GitHub
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

from . import _galsim
from .errors import GalSimError, GalSimValueError, convert_cpp_errors

def int1d(func, min, max, rel_err=1.e-6, abs_err=1.e-12):
    """Integrate a 1-dimensional function from min to max.

    Example usage::

        >>> def func(x): return x**2
        >>> galsim.integ.int1d(func, 0, 1)
        0.33333333333333337
        >>> galsim.integ.int1d(func, 0, 2)
        2.666666666666667
        >>> galsim.integ.int1d(func, -1, 1)
        0.66666666666666674

    .. note::

        This uses an adaptive Gauss-Kronrod-Patterson method for doing the integration.

        cf. https://www.jstor.org/stable/2004583

        If one or both endpoints are infinite, it will automatically use an appropriate
        transformation to turn it into a finite integral.

    Parameters:
        func:       The function to be integrated.  y = func(x) should be valid.
        min:        The lower end of the integration bounds (anything < -1.e10 is treated as
                    negative infinity).
        max:        The upper end of the integration bounds (anything > 1.e10 is treated as positive
                    infinity).
        rel_err:    The desired relative error [default: 1.e-6]
        abs_err:    The desired absolute error [default: 1.e-12]

    Returns:
        the value of the integral.
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

def hankel(func, k, nu=0, rmax=None, rel_err=1.e-6, abs_err=1.e-12):
    r"""Perform an order nu Hankel transform of the given function f(r) at a specific k value.

    .. math::

        F(k) = \int_0^\infty f(r) J_\nu(k r) r dr

    .. note::

        For non-truncated Hankel integrals, this uses the method outlined in Ogata, 2005:
        http://www.kurims.kyoto-u.ac.jp/~prims/pdf/41-4/41-4-40.pdf

        For truncated integrals (and k=0), it uses the same adaptive Gauss-Kronrod-Patterson
        method used for `int1d`.

    Parameters:

        func:       The function f(r)
        k:          (float or numpy array) The value(s) of k for which to calculate F(k).
        nu:         The order of the Bessel function to use for the transform. [default: 0]
        rmax:       An optional truncation radius at which to have f(r) drop to 0. [default: None]
        rel_err:    The desired relative accuracy [default: 1.e-6]
        abs_err:    The desired absolute accuracy [default: 1.e-12]

    Returns:

        An estimate of F(k)
    """
    rel_err = float(rel_err)
    abs_err = float(abs_err)
    nu = float(nu)
    rmax = float(rmax) if rmax is not None else 0.

    k = np.ascontiguousarray(k, dtype=float)
    res = np.empty_like(k, dtype=float)
    N = 1 if k.shape == () else len(k)

    if np.any(k < 0):
        raise GalSimValueError("k must be >= 0",k)
    if nu < 0:
        raise GalSimValueError("nu must be >= 0",k)
    _k = k.__array_interface__['data'][0]
    _res = res.__array_interface__['data'][0]
    with convert_cpp_errors():
        _galsim.PyHankel(func, _k, _res, N, nu, rmax, rel_err, abs_err)
    return res

class IntegrationRule(object):
    """A class that can be used to integrate something more complicated than a normal
    scalar function.

    In GalSim, we use it to do the integration of chromatic images over a bandpass.
    Typically f is some kind of draw function, xs are the wavelengths, and w is the
    bandpass throughput.  But this class is abstracted away from all of that and can be used
    for anything where the function returns something complicated, but which can be added
    together to compute the quadrature.

    Specifically the return value from f must be closed under both addition and multiplication
    by a scalar (a float value).
    """
    def __init__(self):
        pass

    def __call__(self, f, xs, w=None):
        """Calculate the integral int(f(x) w(x) dx) using the appropriate Rule.

        Parameters:
            f:      Function to integrate.
            xs:     Locations at which to evaluate f.
            w:      Weight function if desired [default: None]

        Returns:
            The approximation to the integral.
        """
        gs = self.calculateWeights(xs, w)
        return sum(g * f(x) for g,x in zip(gs, xs))

class MidptRule(IntegrationRule):
    """Midpoint rule for integration.
    """
    def calculateWeights(self, xs, w):
        """Calculate the apporpriate weights for the midpoint rule integration

        Parameters:
            xs:     Locations at which to evaluate f.
            w:      Weight function if desired [default: None]

        Returns:
            The net weights to use at each location.
        """
        if len(xs) < 2:
            raise GalSimValueError("Not enough points for midptRule integration", xs)
        xs = np.asarray(xs)
        gs = np.empty_like(xs)
        gs[0] = (xs[1] - xs[0])
        gs[1:-1] = 0.5 * (xs[2:] - xs[:-2])
        gs[-1] = (xs[-1] - xs[-2])
        if w is not None:
            gs *= w(xs)
        return gs

class TrapzRule(IntegrationRule):
    """Trapezoidal rule for integration.
    """
    def calculateWeights(self, xs, w):
        """Calculate the apporpriate weights for the trapezoidal rule integration

        Parameters:
            xs:     Locations at which to evaluate f.
            w:      Weight function if desired [default: None]

        Returns:
            The net weights to use at each location.
        """
        if len(xs) < 2:
            raise GalSimValueError("Not enough points for trapzRule integration", xs)
        xs = np.asarray(xs)
        gs = np.empty_like(xs)
        gs[0] = 0.5 * (xs[1] - xs[0])
        gs[1:-1] = 0.5 * (xs[2:] - xs[:-2])
        gs[-1] = 0.5 * (xs[-1] - xs[-2])
        if w is not None:
            gs *= w(xs)
        return gs

class QuadRule(IntegrationRule):
    """Quadratic rule for integration

    This models both f and w as linear between the evaluation points, so the product is
    quadratic.
    """
    def calculateWeights(self, xs, w):
        """Calculate the apporpriate weights for the quadratic rule integration

        Parameters:
            xs:     Locations at which to evaluate f.
            w:      Weight function if desired [default: None]

        Returns:
            The net weights to use at each location.
        """
        if len(xs) < 2:
            raise GalSimValueError("Not enough points for quadRule integration", xs)
        if w is None:
            return TrapzRule().calculateWeights(xs,w)
        xs = np.asarray(xs)
        ws = w(xs)
        gs = np.empty_like(xs)
        gs[0] = (xs[1] - xs[0]) * (2*ws[0] + ws[1])
        gs[1:-1] = (xs[1:-1] - xs[:-2]) * (ws[:-2] + 2*ws[1:-1])
        gs[1:-1] += (xs[2:] - xs[1:-1]) * (2*ws[1:-1] + ws[2:])
        gs[-1] = (xs[-1] - xs[-2]) * (ws[-2] + 2*ws[-1])
        gs /= 6.
        return gs

# To ease backwards compatibility, these are an instantiated object of the above classes
midptRule = MidptRule()  #: For convenience, an instance of `MidptRule`
trapzRule = TrapzRule()  #: For convenience, an instance of `TrapzRule`
quadRule = QuadRule()    #: For convenience, an instance of `QuadRule`


class ImageIntegrator(object):
    """A base class for integrators used by `ChromaticObject` to integrate the drawn images
    over wavelengthh using a `Bandpass` as a weight function.
    """
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
        Parameters:
            evaluateAtWavelength:   Function that returns a monochromatic surface brightness
                                    profile as a function of wavelength.
            bandpass:               `Bandpass` object representing the filter being imaged through.
            image:                  `Image` used to set size and scale of output
            drawImageKwargs:        dict with other kwargs to send to `ChromaticObject.drawImage`
                                    function.
            doK:                    Integrate up results of `ChromaticObject.drawKImage` instead of
                                    results of `ChromaticObject.drawImage`.  [default: False]

        Returns:
            the result of integral as an `Image`
        """
        waves = self.calculateWaves(bandpass)
        self.last_n_eval = len(waves)
        drawImageKwargs.pop('add_to_image', None) # Make sure add_to_image isn't in kwargs

        def integrand(w):
            prof = evaluateAtWavelength(w)
            if not doK:
                return prof.drawImage(image=image.copy(), **drawImageKwargs)
            else:
                return prof.drawKImage(image=image.copy(), **drawImageKwargs)
        return self.rule(integrand, waves, bandpass)


class SampleIntegrator(ImageIntegrator):
    """Create a chromatic surface brightness profile integrator, which will integrate over
    wavelength using a `Bandpass` as a weight function.

    This integrator will evaluate the integrand only at the wavelengths in ``bandpass.wave_list``.
    See ContinuousIntegrator for an integrator that evaluates the integrand at a given number of
    points equally spaced apart.

    Parameters:
        rule:       Which integration rule to apply to the wavelength and monochromatic surface
                    brightness samples.  Options include:

                    - galsim.integ.midptRule: Use the midpoint integration rule
                    - galsim.integ.trapzRule: Use the trapezoidal integration rule
                    - galsim.integ.quadRule: Use the quadratic integration rule

    """
    def __init__(self, rule):
        self.rule = rule

    def calculateWaves(self, bandpass):
        return bandpass.wave_list


class ContinuousIntegrator(ImageIntegrator):
    """Create a chromatic surface brightness profile integrator, which will integrate over
    wavelength using a `Bandpass` as a weight function.

    This integrator will evaluate the integrand at a given number ``N`` of equally spaced
    wavelengths over the interval defined by bandpass.blue_limit and bandpass.red_limit.  See
    SampleIntegrator for an integrator that only evaluates the integrand at the predefined set of
    wavelengths in ``bandpass.wave_list``.

    Parameters:
        rule:           Which integration rule to apply to the wavelength and monochromatic
                        surface brightness samples.  Options include:

                        - galsim.integ.midptRule: Use the midpoint integration rule
                        - galsim.integ.trapzRule: Use the trapezoidal integration rule
                        - galsim.integ.quadRule: Use the quadratic integration rule

        N:              Number of equally-wavelength-spaced monochromatic surface brightness
                        samples to evaluate. [default: 250]
        use_endpoints:  Whether to sample the endpoints ``bandpass.blue_limit`` and
                        ``bandpass.red_limit``.  This should probably be True for a rule like
                        numpy.trapz, which explicitly samples the integration limits.  For a
                        rule like the midpoint rule, however, the integration limits are not
                        generally sampled, (only the midpoint between each integration limit and
                        its nearest interior point is sampled), thus ``use_endpoints`` should be
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
