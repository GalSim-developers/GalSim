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
Includes a Python layer version of the C++ int1d function in galim::integ,
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
    @param rel_err  The desired relative error (default `rel_err = 1.e-6`)
    @param abs_err  The desired absolute error (default `abs_err = 1.e-12`)
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

def trapz_sample_integrator(evaluateAtWavelength, bandpass, image, gain=1.0, wmult=1.0,
                            use_true_center=True, offset=None):
    """Integrate a chromatic surface brightness profile over wavelength, using `bandpass` throughput
    as a weight function, and the trapezoidal rule.

    This function evaluates the integrand (= evaluateAtWavelength * bandpass) exactly at the
    wavelengths stored in `bandpass.wave_list`.  See `trapz_continuous_integrator` for a more
    flexible integrator.

    @param evaluateAtWavelength  Function that returns a monochromatic surface brightness profile as
                                 a function of wavelength.
    @param bandpass              galsim.Bandpass object representing the filter being imaged through.
    @param image                 galsim.Image used to set size and scale of output
    @param N                     Number of subintervals in to which to divide the integration
                                 interval.
    @param gain                  See GSObject.draw()
    @param wmult                 See GSObject.draw()
    @param use_true_center       See GSObject.draw()
    @param offset                See GSObject.draw()
    @returns                     result of integral as a galsim.Image
    """
    images = []
    if not hasattr(bandpass, 'wave_list'):
        raise AttributeError("Bandpass does not have attribute `wave_list` needed by " +
                             "trapz_sample_integrator.")
    for w in bandpass.wave_list:
        prof = evaluateAtWavelength(w) * bandpass(w)
        tmpimage = image.copy()
        tmpimage.setZero()
        prof.draw(image=tmpimage, gain=gain, wmult=wmult,
                  use_true_center=use_true_center, offset=offset)
        images.append(tmpimage)
    return np.trapz(images, bandpass.wave_list)

def trapz_continuous_integrator(evaluateAtWavelength, bandpass, image, gain=1.0, wmult=1.0,
                                use_true_center=True, offset=None, N=250):
    """Integrate a chromatic surface brightness profile over wavelength, using `bandpass` throughput
    as a weight function, and the trapezoidal rule.

    This function evaluates the integrand (= evaluateAtWavelength * bandpass) at the wavelengths at
    the midpoints of `N` equally sized subintervals between `bandpass.blue_limit` and
    `bandpass.red_limit`.  See `trapz_sample_integrator` for an integrator that only evaluates the
    integrand at pre-specified wavelengths.

    @param evaluateAtWavelength  Function that returns a monochromatic surface brightness profile as
                                 a function of wavelength.
    @param bandpass              galsim.Bandpass object representing the filter being imaged through.
    @param image                 galsim.Image used to set size and scale of output
    @param gain                  See GSObject.draw()
    @param wmult                 See GSObject.draw()
    @param use_true_center       See GSObject.draw()
    @param offset                See GSObject.draw()
    @returns                     result of integral as a galsim.Image
    """
    h = (bandpass.red_limit*1.0 - bandpass.blue_limit)/N
    waves = [bandpass.blue_limit + h * i for i in range(N+1)]
    images = []
    for w in waves:
        prof = evaluateAtWavelength(w) * bandpass(w)
        tmpimage = image.copy()
        tmpimage.setZero()
        prof.draw(image=tmpimage, gain=gain, wmult=wmult,
                  use_true_center=use_true_center, offset=offset)
        images.append(tmpimage)
    return np.trapz(images, waves)

def midpt_sample_integrator(evaluateAtWavelength, bandpass, image, gain=1.0, wmult=1.0,
                            use_true_center=True, offset=None):
    """Integrate a chromatic surface brightness profile over wavelength, using `bandpass` throughput
    as a weight function, and the midpoint rule.

    This function evaluates the integrand (= evaluateAtWavelength * bandpass) exactly at the
    wavelengths stored in `bandpass.wave_list`.  See `midpt_continuous_integrator` for a more
    flexible integrator.

    @param evaluateAtWavelength  Function that returns a monochromatic surface brightness profile as
                                 a function of wavelength.
    @param bandpass              galsim.Bandpass object representing the filter being imaged through.
    @param image                 galsim.Image used to set size and scale of output
    @param gain                  See GSObject.draw()
    @param wmult                 See GSObject.draw()
    @param use_true_center       See GSObject.draw()
    @param offset                See GSObject.draw()
    @returns                     result of integral as a galsim.Image
    """
    images = []
    for w in bandpass.wave_list:
        prof = evaluateAtWavelength(w) * bandpass(w)
        tmpimage = image.copy()
        tmpimage.setZero()
        prof.draw(image=tmpimage, gain=gain, wmult=wmult,
                  use_true_center=use_true_center, offset=offset)
        images.append(tmpimage)
    return midpt(images, bandpass.wave_list)

def midpt_continuous_integrator(evaluateAtWavelength, bandpass, image, gain=1.0, wmult=1.0,
                                use_true_center=True, offset=None, N=250):
    """Integrate a chromatic surface brightness profile over wavelength, using `bandpass` throughput
    as a weight function.

    This function evaluates the integrand (= evaluateAtWavelength * bandpass) at the wavelengths at
    the midpoints of `N` equally sized subintervals between `bandpass.blue_limit` and
    `bandpass.red_limit`.  See `trapz_sample_integrator` for an integrator that only evaluates the
    integrand at pre-specified wavelengths.

    @param evaluateAtWavelength  Function that returns a monochromatic surface brightness profile as
                                 a function of wavelength.
    @param bandpass              galsim.Bandpass object representing the filter being imaged through.
    @param image                 galsim.Image used to set size and scale of output
    @param gain                  See GSObject.draw()
    @param wmult                 See GSObject.draw()
    @param use_true_center       See GSObject.draw()
    @param offset                See GSObject.draw()
    @param N                     Number of subintervals in to which to divide the integration
                                 interval.
    @returns                     result of integral as a galsim.Image
    """
    h = (bandpass.red_limit*1.0 - bandpass.blue_limit)/N
    waves = [bandpass.blue_limit + h * (i+0.5) for i in range(N)]
    images = []
    for w in waves:
        prof = evaluateAtWavelength(w) * bandpass(w)
        tmpimage = image.copy()
        tmpimage.setZero()
        prof.draw(image=tmpimage, gain=gain, wmult=wmult,
                  use_true_center=use_true_center, offset=offset)
        images.append(tmpimage)
    return midpt(images, waves)

def midpt(fvals, x):
    """Midpoint rule for integration.

    @param fvals  Samples of the integrand
    @param x      Locations at which the integrand was sampled.
    @returns      Midpoint rule approximation of the integral.
    """
    x = np.array(x)
    dx = [x[1]-x[0]]
    dx.extend(0.5*(x[2:]-x[0:-2]))
    dx.append(x[-1]-x[-2])
    weighted_fvals = [w*f for w,f in zip(dx, fvals)]
    return reduce(lambda y,z:y+z, weighted_fvals)
