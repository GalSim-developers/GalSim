# Copyright (c) 2012-2022 by the GalSim developers team on GitHub
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

import math
import numpy as np

from . import _galsim
from .gsparams import GSParams
from .utilities import lazy_property
from .errors import GalSimValueError

class Interpolant:
    """A base class that defines how interpolation should be done.

    An Interpolant is needed for an `InterpolatedImage` to define how interpolation should be done
    an locations in between the integer pixel centers.
    """
    def __init__(self):
        raise NotImplementedError(
            "The Interpolant base class should not be instantiated directly. "
            "Use one of the subclasses instead, or use the `from_name` factory function.")

    @staticmethod
    def from_name(name, tol=None, gsparams=None):
        """A factory function to create an `Interpolant` of the correct type according to
        the (string) name of the `Interpolant`.

        This is mostly used to simplify how config files specify the `Interpolant` to use.

        Valid names are:

            - 'delta' = `Delta`
            - 'nearest' = `Nearest`
            - 'sinc' = `SincInterpolant`
            - 'linear' = `Linear`
            - 'cubic' = `Cubic`
            - 'quintic' = `Quintic`
            - 'lanczosN' = `Lanczos`  (where N is an integer, given the ``n`` parameter)

        In addition, if you want to specify the ``conserve_dc`` option for `Lanczos`, you can
        append either T or F to represent ``conserve_dc = True/False`` (respectively).  Otherwise,
        the default ``conserve_dc=True`` is used.

        Parameters:
            name:       The name of the interpolant to create.
            tol:        [deprecated]
            gsparams:   An optional `GSParams` argument. [default: None]
        """
        if tol is not None:
            from galsim.deprecated import depr
            depr('tol', 2.2, 'gsparams=GSParams(kvalue_accuracy=tol)')
            gsparams = GSParams(kvalue_accuracy=tol)
        gsparams = GSParams.check(gsparams)

        # Do these in rough order of likelihood (most to least)
        if name.lower() == 'quintic':
            return Quintic(gsparams=gsparams)
        if name.lower().startswith('lanczos'):
            conserve_dc = True
            if name[-1].upper() in ('T', 'F'):
                conserve_dc = (name[-1].upper() == 'T')
                name = name[:-1]
            try:
                n = int(name[7:])
            except Exception:
                raise GalSimValueError("Invalid Lanczos specification. Should look like "
                                       "lanczosN, where N is an integer", name) from None
            return Lanczos(n, conserve_dc, gsparams=gsparams)
        elif name.lower() == 'linear':
            return Linear(gsparams=gsparams)
        elif name.lower() == 'cubic' :
            return Cubic(gsparams=gsparams)
        elif name.lower() == 'nearest':
            return Nearest(gsparams=gsparams)
        elif name.lower() == 'delta':
            return Delta(gsparams=gsparams)
        elif name.lower() == 'sinc':
            return SincInterpolant(gsparams=gsparams)
        else:
            raise GalSimValueError("Invalid Interpolant name %s.",name,
                                   ('linear', 'cubic', 'quintic', 'lanczosN', 'nearest', 'delta',
                                    'sinc'))

    @property
    def gsparams(self):
        """The `GSParams` of the `Interpolant`
        """
        return self._gsparams

    @property
    def positive_flux(self):
        """The positive-flux fraction of the interpolation kernel."""
        return self._i.getPositiveFlux();

    @property
    def negative_flux(self):
        """The negative-flux fraction of the interpolation kernel."""
        return self._i.getNegativeFlux();

    @property
    def tol(self):
        from galsim.deprecated import depr
        depr('interpolant.tol', 2.2, 'interpolant.gsparams.kvalue_accuracy')
        return self._gsparams.kvalue_accuracy

    def withGSParams(self, gsparams=None, **kwargs):
        """Create a version of the current interpolant with the given gsparams
        """
        if gsparams == self.gsparams: return self
        from copy import copy
        ret = copy(self)
        ret._gsparams = GSParams.check(gsparams, self.gsparams, **kwargs)
        return ret

    def __getstate__(self):
        d = self.__dict__.copy()
        d.pop('_i', None)
        return d

    def __setstate__(self, d):
        self.__dict__ = d

    def __eq__(self, other):
        return (self is other or (isinstance(other, self.__class__) and repr(self) == repr(other)))

    def __ne__(self, other):
        return not self.__eq__(other)

    def __hash__(self):
        return hash(repr(self))

    def xval(self, x):
        """Calculate the value of the interpolant kernel at one or more x values

        Parameters:
            x:      The value (as a float) or values (as a np.array) at which to compute the
                    amplitude of the Interpolant kernel.

        Returns:
            xval:   The value(s) at the x location(s).  If x was an array, then this is also
                    an array.
        """
        xx = np.array(x, dtype=float, copy=True)
        if xx.shape == ():
            return self._i.xval(float(xx))
        else:
            dimen = len(x.shape)
            if dimen > 1:
                raise GalSimValueError("Input x must be 1-dimensional", x)
            _xx = xx.__array_interface__['data'][0]
            self._i.xvalMany(_xx, len(xx))
            return xx

    def kval(self, k):
        """Calculate the value of the interpolant kernel in Fourier space at one or more k values.

        Parameters:
            k:      The value (as a float) or values (as a np.array) at which to compute the
                    amplitude of the Interpolant kernel in Fourier space.

        Returns:
            kval:   The k-value(s) at the k location(s).  If k was an array, then this is also
                    an array.
        """
        # Note: the C++ layer uses u = k/2pi rather than k.
        u = np.array(k, dtype=float, copy=True) / (2.*np.pi)
        if u.shape == ():
            return self._i.uval(float(u))
        else:
            dimen = len(k.shape)
            if dimen > 1:
                raise GalSimValueError("Input k must be 1-dimensional", k)
            _u = u.__array_interface__['data'][0]
            self._i.uvalMany(_u, len(u))
            return u

    def unit_integrals(self, max_len=None):
        """Compute the unit integrals of the real-space kernel.

        integrals[i] = int(xval(x), i-0.5, i+0.5)

        Parameters:
            max_len:    The maximum length of the returned array.  This is usually only relevant
                        for SincInterpolant, where xrange = inf.

        Returns:
            integrals:  An array of unit integrals of length max_len or smaller.
        """
        from .integ import int1d
        n = self.ixrange//2 + 1
        n = n if max_len is None else min(n, max_len)
        if not hasattr(self, '_unit_integrals') or n > len(self._unit_integrals):
            # Note: This is pretty fast, but subclasses may choose to override this if there
            # is an analytic integral to avoid the int1d call.
            self._unit_integrals = np.zeros(n)
            for i in range(n):
                self._unit_integrals[i] = int1d(self.xval, i-0.5, i+0.5)
        return self._unit_integrals[:n]

    # Sub-classes should define _i property, repr, and str


class Delta(Interpolant):
    """Delta-function interpolation.

    The interpolant for when you do not want to interpolate between samples.  It is not really
    intended to be used for any analytic drawing because it is infinite in the x domain at the
    location of samples, and it extends to infinity in the u domain.  But it could be useful for
    photon-shooting, where it is trivially implemented as no displacements.

    Parameters:
        tol:        [deprecated]
        gsparams:   An optional `GSParams` argument. [default: None]
    """
    def __init__(self, tol=None, gsparams=None):
        if tol is not None:
            from galsim.deprecated import depr
            depr('tol', 2.2, 'gsparams=GSParams(kvalue_accuracy=tol)')
            gsparams = GSParams(kvalue_accuracy=tol)
        self._gsparams = GSParams.check(gsparams)

    @lazy_property
    def _i(self):
        return _galsim.Delta(self._gsparams._gsp)

    def __repr__(self):
        return "galsim.Delta(gsparams=%r)"%(self._gsparams)

    def __str__(self):
        return "galsim.Delta()"

    @property
    def xrange(self):
        """The maximum extent of the interpolant from the origin (in pixels).
        """
        return 0.

    @property
    def ixrange(self):
        """The total integral range of the interpolant.  Typically 2 * xrange.
        """
        return 0

    @property
    def krange(self):
        """The maximum extent of the interpolant in Fourier space (in 1/pixels).
        """
        return 2. * math.pi / self._gsparams.kvalue_accuracy

    _unit_integrals = np.array([1], dtype=float)

    def unit_integrals(self, max_len=None):
        """Compute the unit integrals of the real-space kernel.

        integrals[i] = int(xval(x), i-0.5, i+0.5)

        Parameters:
            max_len:    The maximum length of the returned array. (ignored)

        Returns:
            integrals:  An array of unit integrals of length max_len or smaller.
        """
        return self._unit_integrals


class Nearest(Interpolant):
    """Nearest-neighbor interpolation (boxcar).

    The nearest-neighbor interpolant performs poorly as a k-space or x-space interpolant for
    interpolated images.  (See paper by "Bernstein & Gruen, http://arxiv.org/abs/1401.2636.)
    The objection to its use in Fourier space does not apply when shooting photons to generate
    an image; in that case, the nearest-neighbor interpolant is quite efficient (but not
    necessarily the best choice in terms of accuracy).

    Parameters:
        tol:        [deprecated]
        gsparams:   An optional `GSParams` argument. [default: None]
    """
    def __init__(self, tol=None, gsparams=None):
        if tol is not None:
            from galsim.deprecated import depr
            depr('tol', 2.2, 'gsparams=GSParams(kvalue_accuracy=tol)')
            gsparams = GSParams(kvalue_accuracy=tol)
        self._gsparams = GSParams.check(gsparams)

    @lazy_property
    def _i(self):
        return _galsim.Nearest(self._gsparams._gsp)

    def __repr__(self):
        return "galsim.Nearest(gsparams=%r)"%(self._gsparams)

    def __str__(self):
        return "galsim.Nearest()"

    @property
    def xrange(self):
        """The maximum extent of the interpolant from the origin (in pixels).
        """
        return 0.5

    @property
    def ixrange(self):
        """The total integral range of the interpolant.  Typically 2 * xrange.
        """
        return 1

    @property
    def krange(self):
        """The maximum extent of the interpolant in Fourier space (in 1/pixels).
        """
        return 2. / self._gsparams.kvalue_accuracy

    _unit_integrals = np.array([1], dtype=float)

    def unit_integrals(self, max_len=None):
        """Compute the unit integrals of the real-space kernel.

        integrals[i] = int(xval(x), i-0.5, i+0.5)

        Parameters:
            max_len:    The maximum length of the returned array. (ignored)

        Returns:
            integrals:  An array of unit integrals of length max_len or smaller.
        """
        return self._unit_integrals


class SincInterpolant(Interpolant):
    """Sinc interpolation (inverse of nearest-neighbor).

    The Sinc interpolant (K(x) = sin(pi x)/(pi x)) is mathematically perfect for band-limited
    data, introducing no spurious frequency content beyond kmax = pi/dx for input data with pixel
    scale dx.  However, it is formally infinite in extent and, even with reasonable trunction, is
    still quite large.  It will give exact results in `GSObject.kValue` for `InterpolatedImage`
    when it is used as a k-space interpolant, but is extremely slow.  The usual compromise between
    sinc accuracy vs. speed is the `Lanczos` interpolant (see its documentation for details).

    Parameters:
        tol:        [deprecated]
        gsparams:   An optional `GSParams` argument. [default: None]
    """
    def __init__(self, tol=None, gsparams=None):
        if tol is not None:
            from galsim.deprecated import depr
            depr('tol', 2.2, 'gsparams=GSParams(kvalue_accuracy=tol)')
            gsparams = GSParams(kvalue_accuracy=tol)
        self._gsparams = GSParams.check(gsparams)

    @lazy_property
    def _i(self):
        return _galsim.SincInterpolant(self._gsparams._gsp)

    def __repr__(self):
        return "galsim.SincInterpolant(gsparams=%r)"%(self._gsparams)

    def __str__(self):
        return "galsim.SincInterpolant()"

    @property
    def xrange(self):
        """The maximum extent of the interpolant from the origin (in pixels).
        """
        # Technically infinity, but truncated by the tolerance.
        return 1./(math.pi * self._gsparams.kvalue_accuracy)

    @property
    def ixrange(self):
        """The total integral range of the interpolant.  Typically 2 * xrange.
        """
        return 2*int(np.ceil(self.xrange))

    @property
    def krange(self):
        """The maximum extent of the interpolant in Fourier space (in 1/pixels).
        """
        return math.pi

    def unit_integrals(self, max_len=None):
        """Compute the unit integrals of the real-space kernel.

        integrals[i] = int(xval(x), i-0.5, i+0.5)

        Parameters:
            max_len:    The maximum length of the returned array.

        Returns:
            integrals:  An array of unit integrals of length max_len or smaller.
        """
        from .bessel import si
        n = self.ixrange//2 + 1
        n = n if max_len is None else min(n, max_len)
        if not hasattr(self, '_unit_integrals') or n > len(self._unit_integrals):
            self._unit_integrals = np.zeros(n)
            for i in range(n):
                self._unit_integrals[i] = (si(np.pi * (i+0.5)) - si(np.pi * (i-0.5))) / np.pi
        return self._unit_integrals[:n]


class Linear(Interpolant):
    """Linear interpolation

    The linear interpolant is a poor choice for FFT-based operations on interpolated images, as
    it rings to high frequencies.  (See Bernstein & Gruen, http://arxiv.org/abs/1401.2636.)
    This objection does not apply when shooting photons, in which case the linear interpolant is
    quite efficient (but not necessarily the best choice in terms of accuracy).

    Parameters:
        tol:        [deprecated]
        gsparams:   An optional `GSParams` argument. [default: None]
    """
    def __init__(self, tol=None, gsparams=None):
        if tol is not None:
            from galsim.deprecated import depr
            depr('tol', 2.2, 'gsparams=GSParams(kvalue_accuracy=tol)')
            gsparams = GSParams(kvalue_accuracy=tol)
        self._gsparams = GSParams.check(gsparams)

    @lazy_property
    def _i(self):
        return _galsim.Linear(self._gsparams._gsp)

    def __repr__(self):
        return "galsim.Linear(gsparams=%r)"%(self._gsparams)

    def __str__(self):
        return "galsim.Linear()"

    @property
    def xrange(self):
        """The maximum extent of the interpolant from the origin (in pixels).
        """
        # Reduce range slightly so not including points with zero weight.
        return 1.

    @property
    def ixrange(self):
        """The total integral range of the interpolant.  Typically 2 * xrange.
        """
        return 2

    @property
    def krange(self):
        """The maximum extent of the interpolant in Fourier space (in 1/pixels).
        """
        return 2. / self._gsparams.kvalue_accuracy**0.5

    _unit_integrals = np.array([0.75, 0.125], dtype=float)

    def unit_integrals(self, max_len=None):
        """Compute the unit integrals of the real-space kernel.

        integrals[i] = int(xval(x), i-0.5, i+0.5)

        Parameters:
            max_len:    The maximum length of the returned array.  This is usually only relevant
                        for SincInterpolant, where xrange = inf.
        Returns:
            integrals:  An array of unit integrals of length max_len or smaller.
        """
        # i0 = 2*int(1-x, x=0..0.5)
        #    = 3/4
        # i1 = int(1-x, x=0.5..1)
        #    = 1/8
        return self._unit_integrals


class Cubic(Interpolant):
    """Cubic interpolation

    The cubic interpolant is exact to 3rd order Taylor expansion (from R. G. Keys, IEEE Trans.
    Acoustics, Speech, & Signal Proc 29, p 1153, 1981).  It is a reasonable choice for a four-point
    interpolant for interpolated images.  (See Bernstein & Gruen, http://arxiv.org/abs/1401.2636.)

    Parameters:
        tol:        [deprecated]
        gsparams:   An optional `GSParams` argument. [default: None]
    """
    def __init__(self, tol=None, gsparams=None):
        if tol is not None:
            from galsim.deprecated import depr
            depr('tol', 2.2, 'gsparams=GSParams(kvalue_accuracy=tol)')
            gsparams = GSParams(kvalue_accuracy=tol)
        self._gsparams = GSParams.check(gsparams)

    @lazy_property
    def _i(self):
        return _galsim.Cubic(self._gsparams._gsp)

    def __repr__(self):
        return "galsim.Cubic(gsparams=%r)"%(self._gsparams)

    def __str__(self):
        return "galsim.Cubic()"

    @property
    def xrange(self):
        """The maximum extent of the interpolant from the origin (in pixels).
        """
        return 2.

    @property
    def ixrange(self):
        """The total integral range of the interpolant.  Typically 2 * xrange.
        """
        return 4

    @property
    def krange(self):
        """The maximum extent of the interpolant in Fourier space (in 1/pixels).
        """
        # kmax = 2 * (3sqrt(3)/8 tol)^1/3
        return 1.7320508075688774 / self._gsparams.kvalue_accuracy**(1./3.)

    _unit_integrals = np.array([161./192, 3./32, -5./384], dtype=float)

    def unit_integrals(self, max_len=None):
        """Compute the unit integrals of the real-space kernel.

        integrals[i] = int(xval(x), i-0.5, i+0.5)

        Parameters:
            max_len:    The maximum length of the returned array.  This is usually only relevant
                        for SincInterpolant, where xrange = inf.
        Returns:
            integrals:  An array of unit integrals of length max_len or smaller.
        """
        # i0 = 2*int(1 + 1/2 x^2(3x-5), x=0..0.5)
        #    = 161/192
        # i1 = int(1 + 1/2 x^2(3x-5), x=0.5..1) - int(1/2 (x-1)*(x-2)^2, x=1..1.5)
        #    = 47/384 - 11/384 = 3/32
        # i2 = -int(1/2 (x-1)*(x-2)^2, x=1.5..2)
        #    = -5/384
        return self._unit_integrals


class Quintic(Interpolant):
    """Fifth order interpolation

    The quintic interpolant is exact to 5th order in the Taylor expansion and was found by
    Bernstein & Gruen (http://arxiv.org/abs/1401.2636) to give optimal results as a k-space
    interpolant.

    Parameters:
        tol:        [deprecated]
        gsparams:   An optional `GSParams` argument. [default: None]
    """
    def __init__(self, tol=None, gsparams=None):
        if tol is not None:
            from galsim.deprecated import depr
            depr('tol', 2.2, 'gsparams=GSParams(kvalue_accuracy=tol)')
            gsparams = GSParams(kvalue_accuracy=tol)
        self._gsparams = GSParams.check(gsparams)

    @lazy_property
    def _i(self):
        return _galsim.Quintic(self._gsparams._gsp)

    def __repr__(self):
        return "galsim.Quintic(gsparams=%r)"%(self._gsparams)

    def __str__(self):
        return "galsim.Quintic()"

    @property
    def xrange(self):
        """The maximum extent of the interpolant from the origin (in pixels).
        """
        return 3.

    @property
    def ixrange(self):
        """The total integral range of the interpolant.  Typically 2 * xrange.
        """
        return 6

    @property
    def krange(self):
        """The maximum extent of the interpolant in Fourier space (in 1/pixels).
        """
        # kmax = 2 * (25sqrt(5)/108 tol)^1/3
        return 1.6058208066649935 / self._gsparams.kvalue_accuracy**(1./3.)


class Lanczos(Interpolant):
    """The Lanczos interpolation filter, nominally sinc(x)*sinc(x/n)

    The Lanczos filter is an approximation to the band-limiting sinc filter with a smooth cutoff
    at high x.  Order n Lanczos has a range of +/- n pixels.  It typically is a good compromise
    between kernel size and accuracy.

    Note that pure Lanczos, when interpolating a set of constant-valued samples, does not return
    this constant.  Setting ``conserve_dc`` in the constructor tweaks the function so that it
    approximately conserves the value of constant (DC) input data (accurate to better than 1.e-5
    when used in two dimensions).

    Parameters:
        n:              The order of the Lanczos function
        conserve_dc:    Whether to add the first order correction to flatten out the flux response
                        to a constant input. [default: True, see above]
        tol:            [deprecated]
        gsparams:       An optional `GSParams` argument. [default: None]
    """
    def __init__(self, n, conserve_dc=True, tol=None, gsparams=None):
        if tol is not None:
            from galsim.deprecated import depr
            depr('tol', 2.2, 'gsparams=GSParams(kvalue_accuracy=tol)')
            gsparams = GSParams(kvalue_accuracy=tol)
        self._n = int(n)
        self._conserve_dc = bool(conserve_dc)
        self._gsparams = GSParams.check(gsparams)

    @lazy_property
    def _i(self):
        return _galsim.Lanczos(self._n, self._conserve_dc, self._gsparams._gsp)

    def __repr__(self):
        return "galsim.Lanczos(%r, %r, gsparams=%r)"%(self._n, self._conserve_dc, self._gsparams)

    def __str__(self):
        return "galsim.Lanczos(%s)"%(self._n)

    @property
    def n(self):
        """The order of the Lanczos function.
        """
        return self._n

    @property
    def conserve_dc(self):
        """Whether this interpolant is modified to improve flux conservation.
        """
        return self._conserve_dc

    @property
    def xrange(self):
        """The maximum extent of the interpolant from the origin (in pixels).
        """
        return self._n

    @property
    def ixrange(self):
        """The total integral range of the interpolant.  Typically 2 * xrange.
        """
        return 2*self._n

    @property
    def krange(self):
        """The maximum extent of the interpolant in Fourier space (in 1/pixels).
        """
        return 2. * math.pi * self._i.urange()
