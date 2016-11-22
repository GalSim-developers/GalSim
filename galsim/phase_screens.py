# Copyright (c) 2012-2016 by the GalSim developers team on GitHub
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

from builtins import range, zip

import numpy as np
import galsim
from . import utilities

class AtmosphericScreen(object):
    """ An atmospheric phase screen that can drift in the wind and evolves ("boils") over time.  The
    initial phases and fractional phase updates are drawn from a von Karman power spectrum, which is
    defined by a Fried parameter that effectively sets the amplitude of the turbulence, and an outer
    scale beyond which the turbulence power flattens.

    @param screen_size   Physical extent of square phase screen in meters.  This should be large
                         enough to accommodate the desired field-of-view of the telescope as well as
                         the meta-pupil defined by the wind speed and exposure time.  Note that
                         the screen will have periodic boundary conditions, so while the code will
                         still run with a small screen, this may introduce artifacts into PSFs or
                         PSF correlations functions.  Also note that screen_size may be tweaked by
                         the initializer to ensure `screen_size` is a multiple of `screen_scale`.
    @param screen_scale  Physical pixel scale of phase screen in meters.  An order unity multiple of
                         the Fried parameter is usually sufficiently small, but users should test
                         the effects of varying this parameter to ensure robust results.
                         [default: r0_500]
    @param altitude      Altitude of phase screen in km.  This is with respect to the telescope, not
                         sea-level.  [default: 0.0]
    @param time_step     Interval to use when advancing the screen in time in seconds.
                         [default: 0.03]
    @param r0_500        Fried parameter setting the amplitude of turbulence; contributes to "size"
                         of the resulting atmospheric PSF.  Specified at wavelength 500 nm, in units
                         of meters.  [default: 0.2]
    @param L0            Outer scale in meters.  The turbulence power spectrum will smoothly
                         approach a constant at scales larger than L0.  Set to `None` or `np.inf`
                         for a power spectrum without an outer scale.  [default: 25.0]
    @param vx            x-component wind velocity in meters/second.  [default: 0.]
    @param vy            y-component wind velocity in meters/second.  [default: 0.]
    @param alpha         Square root of fraction of phase that is "remembered" between time_steps
                         (i.e., alpha**2 is the fraction remembered). The fraction sqrt(1-alpha**2)
                         is then the amount of turbulence freshly generated in each step.  Setting
                         alpha=1.0 results in a frozen-flow atmosphere.  Note that computing PSFs
                         from frozen-flow atmospheres may be significantly faster than computing
                         PSFs with non-frozen-flow atmospheres.  [default: 1.0]
    @param rng           Random number generator as a galsim.BaseDeviate().  If None, then use the
                         clock time or system entropy to seed a new generator.  [default: None]

    Relevant SPIE paper:
    "Remembrance of phases past: An autoregressive method for generating realistic atmospheres in
    simulations"
    Srikar Srinath, Univ. of California, Santa Cruz;
    Lisa A. Poyneer, Lawrence Livermore National Lab.;
    Alexander R. Rudy, UCSC; S. Mark Ammons, LLNL
    Published in Proceedings Volume 9148: Adaptive Optics Systems IV
    September 2014
    """
    def __init__(self, screen_size, screen_scale=None, altitude=0.0, time_step=0.03,
                 r0_500=0.2, L0=25.0, vx=0.0, vy=0.0, alpha=1.0, rng=None,
                 _orig_rng=None, _tab2d=None, _psi=None, _screen=None, _origin=None):

        if screen_scale is None:
            # We copy Jee+Tyson(2011) and (arbitrarily) set the screen scale equal to r0 by default.
            screen_scale = r0_500
        self.npix = galsim.Image.good_fft_size(int(np.ceil(screen_size/screen_scale)))
        self.screen_scale = screen_scale
        self.screen_size = screen_scale * self.npix
        self.altitude = altitude
        self.time_step = time_step
        self.r0_500 = r0_500
        if L0 == np.inf:  # Allow np.inf as synonym for None.
            L0 = None
        self.L0 = L0
        self.vx = vx
        self.vy = vy
        self.alpha = alpha

        if rng is None:
            rng = galsim.BaseDeviate()

        # Should only be using private constructor variables when reconstituting from
        # eval(repr(obj)).
        if _orig_rng is not None:
            self.orig_rng = _orig_rng
            self.rng = rng
            self.tab2d = _tab2d
            # Last two might get quickly deleted if alpha==1, but that's okay.
            self.psi = _psi
            self.screen = _screen
            self.origin = _origin
        else:
            self.orig_rng = rng
            self._init_psi()
            self.reset()

        # Free some RAM for frozen-flow screen.
        if self.alpha == 1.0:
            del self.psi, self.screen

    def __str__(self):
        return "galsim.AtmosphericScreen(altitude=%s)" % self.altitude

    def __repr__(self):
        s = ("galsim.AtmosphericScreen(%r, %r, altitude=%r, time_step=%r, r0_500=%r, L0=%r, " +
             "vx=%r, vy=%r, alpha=%r, rng=%r, _origin=array(%r), _orig_rng=%r, _tab2d=%r") % (
                self.screen_size, self.screen_scale, self.altitude, self.time_step, self.r0_500,
                self.L0, self.vx, self.vy, self.alpha, self.rng, self.origin, self.orig_rng,
                self.tab2d)
        if self.alpha != 1.0:
            s += ", _screen=array(%r, dtype=%s)" % (self.screen.tolist(), self.screen.dtype)
            s += ", _psi=array(%r, dtype=%s)" % (self.screen.tolist(), self.screen.dtype)
        s += ")"
        return s

    def __eq__(self, other):
        # This is a bit draconian since two phase screens with different `time_step`s but otherwise
        # equivalent attributes at least start out equal.  However, I like the idea of comparing
        # rng, orig_rng, and time_step better than comparing self.tab2d, so I'm going with that.
        return (isinstance(other, galsim.AtmosphericScreen) and
                self.screen_size == other.screen_size and
                self.screen_scale == other.screen_scale and
                self.altitude == other.altitude and
                self.r0_500 == other.r0_500 and
                self.L0 == other.L0 and
                self.vx == other.vx and
                self.vy == other.vy and
                self.alpha == other.alpha and
                self.orig_rng == other.orig_rng and
                self.rng == other.rng and
                self.time_step == other.time_step and
                np.array_equal(self.origin, other.origin))

    # No hash since this is a mutable class
    __hash__ = None

    def __ne__(self, other): return not self == other

    # Note the magic number 0.00058 is actually ... wait for it ...
    # (5 * (24/5 * gamma(6/5))**(5/6) * gamma(11/6)) / (6 * pi**(8/3) * gamma(1/6)) / (2 pi)**2
    # It nearly impossible to figure this out from a single source, but it can be derived from a
    # combination of Roddier (1981), Sasiela (1994), and Noll (1976).  (These atmosphere people
    # sure like to work alone... )
    kolmogorov_constant = np.sqrt(0.00058)

    def _init_psi(self):
        """Assemble 2D von Karman sqrt power spectrum.
        """
        fx = np.fft.fftfreq(self.npix, self.screen_scale)
        fx, fy = np.meshgrid(fx, fx)

        L0_inv = 1./self.L0 if self.L0 is not None else 0.0
        old_settings = np.seterr(all='ignore')
        self.psi = (1./self.screen_size*self.kolmogorov_constant*(self.r0_500**(-5.0/6.0)) *
                    (fx*fx + fy*fy + L0_inv*L0_inv)**(-11.0/12.0) *
                    self.npix * np.sqrt(np.sqrt(2.0)))
        np.seterr(**old_settings)
        self.psi *= 500.0  # Multiply by 500 here so we can divide by arbitrary lam later.
        self.psi[0, 0] = 0.0

    def _random_screen(self):
        """Generate a random phase screen with power spectrum given by self.psi**2"""
        gd = galsim.GaussianDeviate(self.rng)
        noise = utilities.rand_arr(self.psi.shape, gd)
        return np.fft.ifft2(np.fft.fft2(noise)*self.psi).real

    def advance(self):
        """Advance phase screen realization by self.time_step."""
        # Moving the origin of the aperture in the opposite direction of the wind is equivalent to
        # moving the screen with the wind.
        self.origin -= (self.vx*self.time_step, self.vy*self.time_step)
        # "Boil" the atmsopheric screen if alpha not 1.
        if self.alpha != 1.0:
            self.screen = self.alpha*self.screen + np.sqrt(1.-self.alpha**2)*self._random_screen()
            self.tab2d = galsim.LookupTable2D(self._xs, self._ys, self.screen, edge_mode='wrap')

    def advance_by(self, dt):
        """Advance phase screen by specified amount of time.

        @param dt  Amount of time in seconds by which to update the screen.
        @returns   The actual amount of time updated, which will differ from `dt` when `dt` is not a
                   multiple of self.time_step.
        """
        if dt < 0:
            raise ValueError("Cannot advance phase screen backwards in time.")
        _nstep = int(np.round(dt/self.time_step))
        if _nstep == 0:
            _nstep = 1
        for i in range(_nstep):
            self.advance()
        return _nstep*self.time_step  # return the time *actually* advanced

    # Note -- use **kwargs here so that AtmosphericScreen.stepK and OpticalScreen.stepK
    # can use the same signature, even though they depend on different parameters.
    def stepK(self, **kwargs):
        """Return an appropriate stepk for this atmospheric layer.

        @param lam         Wavelength in nanometers.
        @param scale_unit  Sky coordinate units of output profile. [default: galsim.arcsec]
        @param gsparams    An optional GSParams argument.  See the docstring for GSParams for
                           details. [default: None]
        @returns  Good pupil scale size in meters.
        """
        lam = kwargs['lam']
        gsparams = kwargs.pop('gsparams', None)
        obj = galsim.Kolmogorov(lam=lam, r0_500=self.r0_500, gsparams=gsparams)
        return obj.stepK()

    def wavefront(self, aper, theta=(0.0*galsim.arcmin, 0.0*galsim.arcmin), compact=True):
        """ Compute wavefront due to atmospheric phase screen.

        Wavefront here indicates the distance by which the physical wavefront lags or leads the
        ideal plane wave.

        @param aper     `galsim.Aperture` over which to compute wavefront.
        @param theta    Field angle of center of output array, as a 2-tuple of `galsim.Angle`s.
                        [default: (0.0*galsim.arcmin, 0.0*galsim.arcmin)]
        @param compact  If true, then only return wavefront for illuminated pixels in a
                        single-dimensional array congruent with array[aper.illuminated].  Otherwise,
                        return wavefront as a 2d array for the full Aperture pupil plane.
                        [default: True]
        @returns        Wavefront lag or lead in nanometers over aperture.
        """
        if compact:
            u, v = aper.u[aper.illuminated], aper.v[aper.illuminated]
        else:
            u, v = aper.u, aper.v
        return self.tab2d(u + self.origin[0] + 1000*self.altitude*theta[0].tan(),
                          v + self.origin[1] + 1000*self.altitude*theta[1].tan())

    def reset(self):
        """Reset phase screen back to time=0."""
        self.rng = self.orig_rng.duplicate()
        self.origin = np.array([0.0, 0.0])

        # Only need to reset/create tab2d if not frozen or doesn't already exist
        if self.alpha != 1.0 or not hasattr(self, 'tab2d'):
            self.screen = self._random_screen()
            self._xs = np.linspace(-0.5*self.screen_size, 0.5*self.screen_size, self.npix,
                                   endpoint=False)
            self._ys = self._xs
            self.tab2d = galsim.LookupTable2D(self._xs, self._ys, self.screen, edge_mode='wrap')


def Atmosphere(screen_size, rng=None, **kwargs):
    """Create an atmosphere as a list of turbulent phase screens at different altitudes.  The
    atmosphere model can then be used to simulate atmospheric PSFs.

    Simulating an atmospheric PSF is typically accomplished by first representing the 3-dimensional
    turbulence in the atmosphere as a series of discrete 2-dimensional phase screens.  These screens
    may blow around in the wind, and may or may not also evolve in time.  This function allows one
    to quickly assemble a list of atmospheric phase screens into a galsim.PhaseScreenList object,
    which can then be used to evaluate PSFs through various columns of atmosphere at different field
    angles.

    The atmospheric screens currently available both produce turbulence that follows a von Karman
    power spectrum.  Specifically, the phase power spectrum in each screen can be written

    psi(nu) = 0.023 r0^(-5/3) (nu^2 + 1/L0^2)^(11/6)

    where psi(nu) is the power spectral density at spatial frequency nu, r0 is the Fried parameter
    (which has dimensions of length) and sets the amplitude of the turbulence, and L0 is the outer
    scale (also dimensions of length) beyond which the power asymptotically flattens.  Typical
    values for r0 are ~0.1 to 0.2 meters, which corresponds roughly to PSF FWHMs of ~0.5 to 1.0
    arcsec for optical wavelengths.  Note that r0 is a function of wavelength, scaling like
    r0 ~ wavelength^(6/5).  To reduce confusion, the input parameter here is named r0_500 and refers
    explicitly to the Fried parameter at a wavelength of 500 nm.  The outer scale is typically in
    the 10s of meters and does not vary with wavelength.

    To create multiple layers, simply specify keyword arguments as length-N lists instead of scalars
    (works for all arguments except `time_step` and `rng`).  If, for any of these keyword arguments,
    you want to use the same value for each layer, then you can just specify the argument as a
    scalar and the function will automatically broadcast it into a list with length equal to the
    longest found keyword argument list.  Note that it is an error to specify keywords with lists of
    different lengths (unless only one of them has length > 1).

    The one exception to the above is the keyword `r0_500`.  The effective Fried parameter for a set
    of atmospheric layers is r0_500_effective = (sum(r**(-5./3) for r in r0_500s))**(-3./5).
    Providing `r0_500` as a scalar or single-element list will result in broadcasting such that the
    effective Fried parameter for the whole set of layers equals the input argument.

    As an example, the following code approximately creates the atmosphere used by Jee+Tyson(2011)
    for their study of atmospheric PSFs for LSST.  Note this code takes about ~3 minutes to run on
    a fast laptop, and will consume about (8192**2 pixels) * (8 bytes) * (6 screens) ~ 3 GB of
    RAM in its final state, and more at intermediate states.

        >>> altitude = [0, 2.58, 5.16, 7.73, 12.89, 15.46]  # km
        >>> r0_500_effective = 0.16  # m
        >>> weights = [0.652, 0.172, 0.055, 0.025, 0.074, 0.022]
        >>> r0_500 = [r0_500_effective * w**(-3./5) for w in weights]
        >>> speed = np.random.uniform(0, 20, size=6)  # m/s
        >>> direction = [np.random.uniform(0, 360)*galsim.degrees for i in xrange(6)]
        >>> npix = 8192
        >>> screen_scale = r0_500_effective
        >>> atm = galsim.Atmosphere(r0_500=r0_500, screen_size=screen_scale*npix, time_step=0.005,
                                    altitude=altitude, L0=25.0, speed=speed,
                                    direction=direction, screen_scale=screen_scale)

    Once the atmosphere is constructed, a 15-sec exposure length monochromatic PSF at 700nm (using
    an 8.4 meter aperture, 0.6 fractional obscuration and otherwise default settings) takes about
    7 minutes to generate on a fast laptop.

        >>> psf = atm.makePSF(lam=700.0, exptime=15.0, diam=8.4, obscuration=0.6)

    Many factors will affect the timing of results, of course, including aperture diameter, gsparams
    settings, pad_factor and oversampling options to makePSF, time_step and exposure time, frozen
    vs. non-frozen atmospheric layers, and so on.  We recommend that users try varying these
    settings to find a balance of speed and fidelity.

    @param r0_500        Fried parameter setting the amplitude of turbulence; contributes to "size"
                         of the resulting atmospheric PSF.  Specified at wavelength 500 nm, in units
                         of meters.  [default: 0.2]
    @param screen_size   Physical extent of square phase screen in meters.  This should be large
                         enough to accommodate the desired field-of-view of the telescope as well as
                         the meta-pupil defined by the wind speed and exposure time.  Note that
                         the screen will have periodic boundary conditions, so the code will run
                         with a smaller sized screen, though this may introduce artifacts into PSFs
                         or PSF correlations functions. Note that screen_size may be tweaked by the
                         initializer to ensure screen_size is a multiple of screen_scale.
    @param time_step     Interval to use when advancing the screen in time in seconds.
                         [default: 0.03]
    @param altitude      Altitude of phase screen in km.  This is with respect to the telescope, not
                         sea-level.  [default: 0.0]
    @param L0            Outer scale in meters.  The turbulence power spectrum will smoothly
                         approach a constant at scales larger than L0.  Set to `None` or `np.inf`
                         for a power spectrum without an outer scale.  [default: 25.0]
    @param speed         Wind speed in meters/second.  [default: 0.0]
    @param direction     Wind direction as galsim.Angle [default: 0.0 * galsim.degrees]
    @param alpha         Fraction of phase that is "remembered" between time_steps.  The fraction
                         1-alpha is then the amount of turbulence freshly generated in each step.
                         [default: 1.0]
    @param screen_scale  Physical pixel scale of phase screen in meters.  A fraction of the Fried
                         parameter is usually sufficiently small, but users should test the effects
                         of this parameter to ensure robust results.
                         [default: same as each screen's r0_500]
    @param rng           Random number generator as a galsim.BaseDeviate().  If None, then use the
                         clock time or system entropy to seed a new generator.  [default: None]
    """
    # Fill in screen_size here, since there isn't a default in AtmosphericScreen
    kwargs['screen_size'] = utilities.listify(screen_size)

    # Set default r0_500 here, so that by default it gets broadcasted below such that the
    # _total_ r0_500 from _all_ screens is 0.2 m.
    if 'r0_500' not in kwargs:
        kwargs['r0_500'] = [0.2]
    kwargs['r0_500'] = utilities.listify(kwargs['r0_500'])

    # Turn speed, direction into vx, vy
    if 'speed' in kwargs:
        kwargs['speed'] = utilities.listify(kwargs['speed'])
        if 'direction' not in kwargs:
            kwargs['direction'] = [0*galsim.degrees]*len(kwargs['speed'])
        kwargs['vx'], kwargs['vy'] = zip(*[v*d.sincos()
                                         for v, d in zip(kwargs['speed'],
                                                         kwargs['direction'])])
        del kwargs['speed']
        del kwargs['direction']

    # Determine broadcast size
    nmax = max(len(v) for v in kwargs.values() if hasattr(v, '__len__'))

    # Broadcast r0_500 here, since logical combination of indiv layers' r0s is complex:
    if len(kwargs['r0_500']) == 1:
        kwargs['r0_500'] = [nmax**(3./5) * kwargs['r0_500'][0]] * nmax

    if rng is None:
        rng = galsim.BaseDeviate()
    kwargs['rng'] = [galsim.BaseDeviate(rng.raw()) for i in range(nmax)]
    return galsim.PhaseScreenList(AtmosphericScreen(**kw)
                                  for kw in utilities.lod_to_dol(kwargs, nmax))


# Some utilities for working with Zernike polynomials
# Combinations.  n choose r.
# See http://stackoverflow.com/questions/3025162/statistics-combinations-in-python
# This is J. F. Sebastian's answer.
def _nCr(n, r):
    if 0 <= r <= n:
        ntok = 1
        rtok = 1
        for t in range(1, min(r, n - r) + 1):
            ntok *= n
            rtok *= t
            n -= 1
        return ntok // rtok
    else:
        return 0


# This function stolen from https://github.com/tvwerkhoven/libtim-py/blob/master/libtim/zern.py
def _noll_to_zern(j):
    """
    Convert linear Noll index to tuple of Zernike indices.
    j is the linear Noll coordinate, n is the radial Zernike index and m is the azimuthal Zernike
    index.
    @param [in] j Zernike mode Noll index
    @return (n, m) tuple of Zernike indices
    @see <https://oeis.org/A176988>.
    """
    if (j == 0):
        raise ValueError("Noll indices start at 1. 0 is invalid.")

    n = 0
    j1 = j-1
    while (j1 > n):
        n += 1
        j1 -= n
    m = (-1)**j * ((n % 2) + 2 * int((j1+((n+1) % 2)) / 2.0))
    return (n, m)


def _zern_norm(n, m):
    """Normalization coefficient for zernike (n, m).

    Defined such that \int_{unit disc} Z(n1, m1) Z(n2, m2) dA = \pi if n1==n2 and m1==m2 else 0.0
    """
    if m == 0:
        return np.sqrt(1./(n+1))
    else:
        return np.sqrt(1./(2.*n+2))


def _zern_rho_coefs(n, m):
    """Compute coefficients of radial part of Zernike (n, m).
    """
    kmax = (n-abs(m)) // 2
    A = [0]*(n+1)
    val = _nCr(n,kmax) # The value for k = 0 in the equation below.
    for k in range(kmax):
        # val = (-1)**k * _nCr(n-k, k) * _nCr(n-2*k, kmax-k) / _zern_norm(n, m)
        # The above formula is faster as a recurrence relation:
        A[n-2*k] = val
        # Don't use *= since the factor is not an integer, but the result is.
        val = -val * (kmax-k)*(n-kmax-k) // ((n-k)*(k+1))
    A[n-2*kmax] = val
    return A


def __zern_coef_array(n, m, eps, shape, annular):
    """Assemble coefficient array for evaluating Zernike (n, m) as the real part of a
    bivariate polynomial in abs(rho)^2 and rho, where rho is a complex array indicating position on
    a unit disc.

    @param n        Zernike radial coefficient
    @param m        Zernike azimuthal coefficient
    @param eps      Linear obscuration fraction.
    @param shape    Output array shape
    @param annular  Boolean indicating polynomials are orthogonal on a disk or an annulus.
    @returns        2D array of coefficients in |r|^2 and r, where r = u + 1j * v, and u, v are unit
                    disk coordinates.
    """
    if shape is None:
        shape = ((n//2)+1, abs(m)+1)
    out = np.zeros(shape, dtype=np.complex128)
    if annular:
        coefs = np.array(_annular_zern_rho_coefs(n, m, eps), dtype=np.complex128)
    else:
        coefs = np.array(_zern_rho_coefs(n, m), dtype=np.complex128)
    coefs /= _zern_norm(n, m)
    if m < 0:
        coefs *= -1j
    for i, c in enumerate(coefs[abs(m)::2]):
        out[i, abs(m)] = c
    return out
_zern_coef_array = utilities.LRU_Cache(__zern_coef_array)

# Following 3 functions from
#
# "Zernike annular polynomials for imaging systems with annular pupils"
# Mahajan (1981) JOSA Vol. 71, No. 1.

# Mahajan's h-function normalization for annular Zernike coefficients.
def __h(m, j, eps):
    if m == 0:  # Equation (A5)
        return (1-eps**2)/(2*(2*j+1))
    else:  # Equation (A14)
        num = -(2*(2*j+2*m-1)) * _Q(m-1, j+1, eps)[0]
        den = (j+m)*(1-eps**2) * _Q(m-1, j, eps)[0]
        return num/den * _h(m-1, j, eps)
_h = utilities.LRU_Cache(__h)

# Mahajan's Q-function for annular Zernikes.
def __Q(m, j, eps):
    if m == 0:  # Equation (A4)
        return _annular_zern_rho_coefs(2*j, 0, eps)[::2]
    else:  # Equation (A13)
        num = 2*(2*j+2*m-1) * _h(m-1, j, eps)
        den = (j+m)*(1-eps**2)*_Q(m-1, j, eps)[0]
        summation = np.zeros((j+1,), dtype=float)
        for i in range(j+1):
            qq = _Q(m-1, i, eps)
            qq = qq*qq[0]  # Don't use *= here since it modifies the cache!
            summation[:i+1] += qq/_h(m-1, i, eps)
        return summation * num / den
_Q = utilities.LRU_Cache(__Q)

def __annular_zern_rho_coefs(n, m, eps):
    """Compute coefficients of radial part of annular Zernike (n, m), with fractional linear
    obscuration eps.
    """
    out = np.zeros((n+1,), dtype=float)
    m = abs(m)
    if m == 0:  # Equation (18)
        norm = 1./(1-eps**2)
        # R[n, m=0, eps](r^2) = R[n, m=0, eps=0]((r^2 - eps^2)/(1 - eps^2))
        # Implement this by retrieving R[n, 0] coefficients of (r^2)^k and
        # multiplying in the binomial (in r^2) expansion of ((r^2 - eps^2)/(1 - eps^2))^k
        coefs = _zern_rho_coefs(n, 0)
        for i, coef in enumerate(coefs):
            if i % 2 == 1: continue
            j = i // 2
            more_coefs = (norm**j) * utilities.binomial(-eps**2, 1, j)
            out[0:i+1:2] += coef*more_coefs
    elif m == n:  # Equation (25)
        norm = 1./np.sqrt(np.sum((eps**2)**np.arange(n+1)))
        out[n] = norm
    else:  # Equation (A1)
        j = (n-m)//2
        norm = np.sqrt((1-eps**2)/(2*(2*j+m+1) * _h(m,j,eps)))
        out[m::2] = norm * _Q(m, j, eps)
    return out
_annular_zern_rho_coefs = utilities.LRU_Cache(__annular_zern_rho_coefs)

def horner(x, coef):
    """Evaluate univariate polynomial using Horner's method.

    I.e., take A + Bx + Cx^2 + Dx^3 and evaluate it as
    A + x(B + x(C + x(D)))

    @param x     Where to evaluate polynomial.
    @param coef  Polynomial coefficients of increasing powers of x.
    @returns     Polynomial evaluation.  Will take on the shape of x if x is an ndarray.
    """
    result = 0
    for c in coef[::-1]:
        result = result*x + c
    return result


def horner2d(x, y, coefs):
    """Evaluate bivariate polynomial using nested Horner's method.

    @param x      Where to evaluate polynomial (first covariate).  Must be same shape as y.
    @param y      Where to evaluate polynomial (second covariate).  Must be same shape as x.
    @param coefs  2D array-like of coefficients in increasing powers of x and y.
                  The first axis corresponds to increasing the power of y, and the second to
                  increasing the power of x.
    @returns      Polynomial evaluation.  Will take on the shape of x and y if these are ndarrays.
    """
    result = 0
    for coef in coefs[::-1]:
        result = result*x + horner(y, coef)
    return result


class OpticalScreen(object):
    """
    Class to describe optical aberrations in terms of Zernike polynomial coefficients.

    Input aberration coefficients are assumed to be supplied in units of wavelength, and correspond
    to the Zernike polynomials in the Noll convention defined in
    Noll, J. Opt. Soc. Am. 66, 207-211(1976).  For a brief summary of the polynomials, refer to
    http://en.wikipedia.org/wiki/Zernike_polynomials#Zernike_polynomials.

    @param tip              Tip aberration in units of reference wavelength.  [default: 0]
    @param tilt             Tilt aberration in units of reference wavelength.  [default: 0]
    @param defocus          Defocus in units of reference wavelength. [default: 0]
    @param astig1           Astigmatism (like e2) in units of reference wavelength.
                            [default: 0]
    @param astig2           Astigmatism (like e1) in units of reference wavelength.
                            [default: 0]
    @param coma1            Coma along y in units of reference wavelength. [default: 0]
    @param coma2            Coma along x in units of reference wavelength. [default: 0]
    @param trefoil1         Trefoil (one of the arrows along y) in units of reference wavelength.
                            [default: 0]
    @param trefoil2         Trefoil (one of the arrows along x) in units of reference wavelength.
                            [default: 0]
    @param spher            Spherical aberration in units of reference wavelength.
                            [default: 0]
    @param aberrations      Optional keyword, to pass in a list, tuple, or NumPy array of
                            aberrations in units of reference wavelength (ordered according to
                            the Noll convention), rather than passing in individual values for each
                            individual aberration.  Note that aberrations[1] is piston (and not
                            aberrations[0], which is unused.)  This list can be arbitrarily long to
                            handle Zernike polynomial aberrations of arbitrary order.
    @param annular_zernike  Boolean indicating that aberrations specify the amplitudes of annular
                            Zernike polynomials instead of circular Zernike polynomials.
                            [default: False]
    @param obscuration      Linear dimension of central obscuration as fraction of aperture linear
                            dimension. [0., 1.).  Note it is the user's responsibility to ensure
                            consistency of OpticalScreen obscuration and Aperture obscuration.
                            [default: 0.0]
    @param lam_0            Reference wavelength in nanometers at which Zernike aberrations are
                            being specified.  [default: 500]
    """
    def __init__(self, tip=0.0, tilt=0.0, defocus=0.0, astig1=0.0, astig2=0.0, coma1=0.0, coma2=0.0,
                 trefoil1=0.0, trefoil2=0.0, spher=0.0, aberrations=None, annular_zernike=False,
                 obscuration=0.0, lam_0=500.0):
        if aberrations is None:
            aberrations = np.zeros(12)
            aberrations[2] = tip
            aberrations[3] = tilt
            aberrations[4] = defocus
            aberrations[5] = astig1
            aberrations[6] = astig2
            aberrations[7] = coma1
            aberrations[8] = coma2
            aberrations[9] = trefoil1
            aberrations[10] = trefoil2
            aberrations[11] = spher
        else:
            # Make sure no individual aberrations were passed in, since they will be ignored.
            if any([tip, tilt, defocus, astig1, astig2, coma1, coma2, trefoil1, trefoil2, spher]):
                raise TypeError("Cannot pass in individual aberrations and array!")
            # Aberrations were passed in, so check for right number of entries.
            if len(aberrations) <= 2:
                raise ValueError("Aberrations keyword must have length > 2")
            # Check for non-zero value in first two places.  Probably a mistake.
            if aberrations[0] != 0.0:
                import warnings
                warnings.warn(
                    "Detected non-zero value in aberrations[0] -- this value is ignored!")

        self.aberrations = np.array(aberrations)
        # strip any trailing zeros.
        self.aberrations = np.trim_zeros(self.aberrations, trim='b')
        self.annular_zernike = annular_zernike
        self.obscuration = obscuration
        self.lam_0 = lam_0
        try:
            maxn = max(_noll_to_zern(j)[0] for j in range(1, len(self.aberrations)))
        except:
            maxn = 0
        shape = (maxn//2+1, maxn+1)  # (max power of |rho|^2,  max power of rho)
        self.coef_array = np.zeros(shape, dtype=np.complex128)

        for j, ab in enumerate(self.aberrations):
            if j == 0: continue
            n, m = _noll_to_zern(j)
            self.coef_array += ab * _zern_coef_array(n, m, self.obscuration, shape,
                                                     self.annular_zernike)

    def __str__(self):
        return "galsim.OpticalScreen(lam_0=%s)" % self.lam_0

    def __repr__(self):
        s = "galsim.OpticalScreen(lam_0=%r" % self.lam_0
        if any(self.aberrations):
            s += ", aberrations=%r"%self.aberrations
        if self.annular_zernike:
            s += ", annular_zernike=True"
            s += ", obscuration=%r"%self.obscuration
        s += ")"
        return s

    def __eq__(self, other):
        return (isinstance(other, galsim.OpticalScreen)
                and np.array_equal(self.aberrations*self.lam_0, other.aberrations*other.lam_0)
                and self.annular_zernike == other.annular_zernike)

    def __ne__(self, other): return not self == other

    # This screen is immutable, so make a hash for it.
    def __hash__(self):
        return hash(("galsim.AtmosphericScreen", tuple((self.aberrations*self.lam_0).ravel())))

    # Note -- use **kwargs here so that AtmosphericScreen.stepK and OpticalScreen.stepK
    # can use the same signature, even though they depend on different parameters.
    def stepK(self, **kwargs):
        """Return an appropriate stepK for this phase screen.

        @param lam         Wavelength in nanometers.
        @param diam        Aperture diameter in meters.
        @param obscuration Fractional linear aperture obscuration. [default: 0.0]
        @param gsparams    An optional GSParams argument.  See the docstring for GSParams for
                           details. [default: None]
        @returns  stepK in inverse arcsec.
        """
        lam = kwargs['lam']
        diam = kwargs['diam']
        obscuration = kwargs.get('obscuration', 0.0)
        gsparams = kwargs.get('gsparams', None)
        # Use an Airy for get appropriate stepK.
        obj = galsim.Airy(lam=lam, diam=diam, obscuration=obscuration, gsparams=gsparams)
        return obj.stepK()

    def wavefront(self, aper, theta=(0.0*galsim.arcmin, 0.0*galsim.arcmin), compact=True):
        """ Compute wavefront due to optical phase screen.

        Wavefront here indicates the distance by which the physical wavefront lags or leads the
        ideal converging Gaussian reference spherical wave.

        @param aper     `galsim.Aperture` over which to compute wavefront.
        @param theta    Field angle of center of output array, as a 2-tuple of `galsim.Angle`s.
                        [default: (0.0*galsim.arcmin, 0.0*galsim.arcmin)]
        @param compact  If true, then only return wavefront for illuminated pixels in a
                        single-dimensional array congruent with array[aper.illuminated].  Otherwise,
                        return wavefront as a 2d array for the full Aperture pupil plane.
                        [default: True]
        @returns        Wavefront lag or lead in nanometers over aperture.
        """
        # ignore theta
        if compact:
            r = aper.rho[aper.illuminated]
        else:
            r = aper.rho
        rsqr = np.abs(r)**2
        return horner2d(rsqr, r, self.coef_array).real * self.lam_0
