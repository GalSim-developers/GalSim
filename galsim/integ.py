# Copyright 2012-2014 The GalSim developers:
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
"""@file integ.py
Includes a Python layer version of the C++ int1d() function in galim::integ,
and python image integrators for use in galsim.chromatic
"""

from . import _galsim
import numpy as np

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
    success, result = _galsim.PyInt1d(func,min,max,rel_err,abs_err)
    if success:
        return result
    else:
        raise RuntimeError(result)

def midpt(fvals, x):
    """Midpoint rule for integration.

    @param fvals  Samples of the integrand
    @param x      Locations at which the integrand was sampled.

    @returns midpoint rule approximation of the integral.
    """
    x = np.array(x)
    dx = [x[1]-x[0]]
    dx.extend(0.5*(x[2:]-x[0:-2]))
    dx.append(x[-1]-x[-2])
    weighted_fvals = [w*f for w,f in zip(dx, fvals)]
    return reduce(lambda y,z:y+z, weighted_fvals)

class ImageIntegrator(object):
    def __init__(self):
        raise NotImplementedError("Must instantiate subclass of ImageIntegrator")
    # subclasses must define
    # 1) a method `.calculateWaves(bandpass)` which will return the wavelengths at which to
    #    evaluate the integrand
    # 2) an function attribute `.rule` which takes a list of integrand evaluations as its first
    #    argument, and a list of evaluation wavelengths as its second argument, and returns
    #    an approximation to the integral.  (E.g., the function midpt above, or numpy.trapz)

    def __call__(self, evaluateAtWavelength, bandpass, image, gain=1.0, wmult=1.0,
                 use_true_center=True, offset=None):
        """
        @param evaluateAtWavelength Function that returns a monochromatic surface brightness
                                    profile as a function of wavelength.
        @param bandpass             Bandpass object representing the filter being imaged through.
        @param image                Image used to set size and scale of output
        @param gain                 See GSObject.draw()
        @param wmult                See GSObject.draw()
        @param use_true_center      See GSObject.draw()
        @param offset               See GSObject.draw()

        @returns the result of integral as an Image
        """
        images = []
        waves = self.calculateWaves(bandpass)
        self.last_n_eval = len(waves)
        for w in waves:
            prof = evaluateAtWavelength(w) * bandpass(w)
            tmpimage = image.copy()
            tmpimage.setZero()
            prof.draw(image=tmpimage, gain=gain, wmult=wmult,
                      use_true_center=use_true_center, offset=offset)
            images.append(tmpimage)
        return self.rule(images, waves)

class SampleIntegrator(ImageIntegrator):
    """Create a chromatic surface brightness profile integrator, which will integrate over
    wavelength using a Bandpass as a weight function.

    This integrator will evaluate the integrand only at the wavelengths in `bandpass.wave_list`.
    See ContinuousIntegrator for an integrator that evaluates the integrand at a given number of
    points equally spaced apart.

    @param rule         Which integration rule to apply to the wavelength and monochromatic surface
                        brightness samples.  Options include:
                            galsim.integ.midpt  --  Use the midpoint integration rule
                            numpy.trapz         --  Use the trapezoidal integration rule
    """
    def __init__(self, rule):
        self.rule = rule
    def calculateWaves(self, bandpass):
        if len(bandpass.wave_list) < 0:
            raise AttributeError("Bandpass does not have attribute `wave_list` needed by " +
                                 "midpt_sample_integrator.")
        return bandpass.wave_list

class ContinuousIntegrator(ImageIntegrator):
    """Create a chromatic surface brightness profile integrator, which will integrate over
    wavelength using a Bandpass as a weight function.

    This integrator will evaluate the integrand only at the wavelengths in `bandpass.wave_list`.
    See ContinuousIntegrator for an integrator that evaluates the integrand at a given number of
    points equally spaced apart.

    @param rule         Which integration rule to apply to the wavelength and monochromatic
                        surface brightness samples.  Options include:
                            galsim.integ.midpt  --  Use the midpoint integration rule
                            numpy.trapz         --  Use the trapezoidal integration rule
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
        self.N = N
        self.rule = rule
        self.use_endpoints = use_endpoints
    def calculateWaves(self, bandpass):
        h = (bandpass.red_limit*1.0 - bandpass.blue_limit)/self.N
        if self.use_endpoints:
            return [bandpass.blue_limit + h * i for i in range(self.N+1)]
        else:
            return [bandpass.blue_limit + h * (i+0.5) for i in range(self.N)]
