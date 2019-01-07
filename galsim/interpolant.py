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
"""@file interpolant.py

Definitions of the various interpolants used by InterpolatedImage and InterpolatedKImage
"""

import math
from past.builtins import basestring

from . import _galsim
from .gsparams import GSParams
from .utilities import lazy_property
from .errors import GalSimValueError, convert_cpp_errors

class Interpolant(object):
    """A base class that defines how interpolation should be done.

    An Interpolant is needed for an InterpolatedImage to define how interpolation should be done
    an locations in between the integer pixel centers.
    """
    def __init__(self):
        raise NotImplementedError(
            "The Interpolant base class should not be instantiated directly. "
            "Use one of the subclasses instead, or use the `from_name` factory function.")

    @staticmethod
    def from_name(name, tol=1.e-4, gsparams=None):
        """A factory function to create an Interpolant of the correct type according to
        the (string) name of the Interpolant.

        This is mostly used to simplify how config files specify the Interpolant to use.

        Valid names are:

            'delta' = Delta()
            'nearest' = Nearest()
            'sinc' = SincInterpolant()
            'linear' = Linear()
            'cubic' = Cubic()
            'quintic' = Quintic()
            'lanczosN' = Lanczos(N)  (where N is an integer)

        In addition, if you want to specify the conserve_dc option for Lanczos, you can append
        either T or F to represent conserve_dc = True/False (respectively).  Otherwise, the
        default conserve_dc=True is used.

        @param name         The name of the interpolant to create.
        @param tol          The requested accuracy tolerance [default: 1.e-4]
        @param gsparams     An optional GSParams instance [default: None]
        """
        tol = float(tol)
        gsparams = GSParams.check(gsparams)

        # Do these in rough order of likelihood (most to least)
        if name.lower() == 'quintic':
            return Quintic(tol, gsparams)
        if name.lower().startswith('lanczos'):
            conserve_dc = True
            if name[-1].upper() in ('T', 'F'):
                conserve_dc = (name[-1].upper() == 'T')
                name = name[:-1]
            try:
                n = int(name[7:])
            except:
                raise GalSimValueError("Invalid Lanczos specification. Should look like "
                                       "lanczosN, where N is an integer", name)
            return Lanczos(n, conserve_dc, tol, gsparams)
        elif name.lower() == 'linear':
            return Linear(tol, gsparams)
        elif name.lower() == 'cubic' :
            return Cubic(tol, gsparams)
        elif name.lower() == 'nearest':
            return Nearest(tol, gsparams)
        elif name.lower() == 'delta':
            return Delta(tol, gsparams)
        elif name.lower() == 'sinc':
            return SincInterpolant(tol, gsparams)
        else:
            raise GalSimValueError("Invalid Interpolant name %s.",name,
                                   ('linear', 'cubic', 'quintic', 'lanczosN', 'nearest', 'delta',
                                    'sinc'))

    @property
    def gsparams(self):
        return self._gsparams

    def withGSParams(self, gsparams):
        """Create a version of the current interpolant with the given gsparams
        """
        if gsparams == self.gsparams: return self
        from copy import copy
        ret = copy(self)
        ret._gsparams = GSParams.check(gsparams)
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

    # Sub-classes should define _i property, repr, and str


class Delta(Interpolant):
    """Delta-function interpolation.

    The interpolant for when you do not want to interpolate between samples.  It is not really
    intended to be used for any analytic drawing because it is infinite in the x domain at the
    location of samples, and it extends to infinity in the u domain.  But it could be useful for
    photon-shooting, where it is trivially implemented as no displacements.

    @param tol          This defines a crude box approximation to the x-space delta function and to
                        give a large but finite range in k space. [default: 1.e-4]
    @param gsparams     An optional GSParams instance.  [default: None]
    """
    def __init__(self, tol=1.e-4, gsparams=None):
        self._tol = float(tol)
        self._gsparams = GSParams.check(gsparams)

    @lazy_property
    def _i(self):
        with convert_cpp_errors():
            return _galsim.Delta(self._tol, self._gsparams._gsp)

    def __repr__(self):
        return "galsim.Delta(%r, %r)"%(self._tol, self._gsparams)

    def __str__(self):
        return "galsim.Delta(%s)"%(self._tol)

    @property
    def xrange(self):
        return 0.

    @property
    def krange(self):
        return 2. * math.pi / self._tol


class Nearest(Interpolant):
    """Nearest-neighbor interpolation (boxcar).

    The nearest-neighbor interpolant performs poorly as a k-space or x-space interpolant for
    interpolated images.  (See paper by "Bernstein & Gruen, http://arxiv.org/abs/1401.2636.)
    The objection to its use in Fourier space does not apply when shooting photons to generate
    an image; in that case, the nearest-neighbor interpolant is quite efficient (but not
    necessarily the best choice in terms of accuracy).

    @param tol          This determines how far onto sinc wiggles the uval will go.  (Very far, by
                        default!) [default: 1.e-4]
    @param gsparams     An optional GSParams instance.  [default: None]
    """
    def __init__(self, tol=1.e-4, gsparams=None):
        self._tol = float(tol)
        self._gsparams = GSParams.check(gsparams)

    @lazy_property
    def _i(self):
        with convert_cpp_errors():
            return _galsim.Nearest(self._tol, self._gsparams._gsp)

    def __repr__(self):
        return "galsim.Nearest(%r, %r)"%(self._tol, self._gsparams)

    def __str__(self):
        return "galsim.Nearest(%s)"%(self._tol)

    @property
    def xrange(self):
        return 0.5

    @property
    def krange(self):
        return 2. / self._tol


class SincInterpolant(Interpolant):
    """Sinc interpolation (inverse of nearest-neighbor).

    The Sinc interpolant (K(x) = sin(pi x)/(pi x)) is mathematically perfect for band-limited
    data, introducing no spurious frequency content beyond kmax = pi/dx for input data with pixel
    scale dx.  However, it is formally infinite in extent and, even with reasonable trunction, is
    still quite large.  It will give exact results in SBInterpolatedImage::kValue() when it is
    used as a k-space interpolant, but is extremely slow.  The usual compromise between sinc
    accuracy vs. speed is the Lanczos interpolant (see its documentation for details).

    @param tol          This determines how far onto sinc wiggles the xval will go.  (Very far, by
                        default!) [default 1.e-4]
    @param gsparams     An optional GSParams instance.  [default: None]
    """
    def __init__(self, tol=1.e-4, gsparams=None):
        self._tol = float(tol)
        self._gsparams = GSParams.check(gsparams)

    @lazy_property
    def _i(self):
        with convert_cpp_errors():
            return _galsim.SincInterpolant(self._tol, self._gsparams._gsp)

    def __repr__(self):
        return "galsim.SincInterpolant(%r, %r)"%(self._tol, self._gsparams)

    def __str__(self):
        return "galsim.SincInterpolant(%s)"%(self._tol)

    @property
    def xrange(self):
        # Technically infinity, but truncated by the tolerance.
        return 1./(math.pi * self._tol)

    @property
    def krange(self):
        return math.pi


class Linear(Interpolant):
    """Linear interpolation

    The linear interpolant is a poor choice for FFT-based operations on interpolated images, as
    it rings to high frequencies.  (See Bernstein & Gruen, http://arxiv.org/abs/1401.2636.)
    This objection does not apply when shooting photons, in which case the linear interpolant is
    quite efficient (but not necessarily the best choice in terms of accuracy).

    @param tol          This determines how far onto sinc^2 wiggles the uval will go. (Very far,
                        by default!) [default 1.e-4]
    @param gsparams     An optional GSParams instance.  [default: None]
    """
    def __init__(self, tol=1.e-4, gsparams=None):
        self._tol = float(tol)
        self._gsparams = GSParams.check(gsparams)

    @lazy_property
    def _i(self):
        with convert_cpp_errors():
            return _galsim.Linear(self._tol, self._gsparams._gsp)

    def __repr__(self):
        return "galsim.Linear(%r, %r)"%(self._tol, self._gsparams)

    def __str__(self):
        return "galsim.Linear(%s)"%(self._tol)

    @property
    def xrange(self):
        # Reduce range slightly so not including points with zero weight.
        return 1. - 0.1*self._tol

    @property
    def krange(self):
        return 2. / self._tol**0.5


class Cubic(Interpolant):
    """Cubic interpolation

    The cubic interpolant is exact to 3rd order Taylor expansion (from R. G. Keys, IEEE Trans.
    Acoustics, Speech, & Signal Proc 29, p 1153, 1981).  It is a reasonable choice for a four-point
    interpolant for interpolated images.  (See Bernstein & Gruen, http://arxiv.org/abs/1401.2636.)

    @param tol          This sets the accuracy and extent of the Fourier transform. [default 1.e-4]
    @param gsparams     An optional GSParams instance.  [default: None]
    """
    def __init__(self, tol=1.e-4, gsparams=None):
        self._tol = float(tol)
        self._gsparams = GSParams.check(gsparams)

    @lazy_property
    def _i(self):
        with convert_cpp_errors():
            return _galsim.Cubic(self._tol, self._gsparams._gsp)

    def __repr__(self):
        return "galsim.Cubic(%r, %r)"%(self._tol, self._gsparams)

    def __str__(self):
        return "galsim.Cubic(%s)"%(self._tol)

    @property
    def xrange(self):
        return 2. - 0.1*self._tol

    @property
    def krange(self):
        # kmax = 2 * (3sqrt(3)/8 tol)^1/3
        return 1.7320508075688774 / self._tol**(1./3.)


class Quintic(Interpolant):
    """Fifth order interpolation

    The quintic interpolant is exact to 5th order in the Taylor expansion and was found by
    Bernstein & Gruen (http://arxiv.org/abs/1401.2636) to give optimal results as a k-space
    interpolant.

    @param tol          This sets the accuracy and extent of the Fourier transform. [default 1.e-4]
    @param gsparams     An optional GSParams instance.  [default: None]
    """
    def __init__(self, tol=1.e-4, gsparams=None):
        self._tol = float(tol)
        self._gsparams = GSParams.check(gsparams)

    @lazy_property
    def _i(self):
        with convert_cpp_errors():
            return _galsim.Quintic(self._tol, self._gsparams._gsp)

    def __repr__(self):
        return "galsim.Quintic(%r, %r)"%(self._tol, self._gsparams)

    def __str__(self):
        return "galsim.Quintic(%s)"%(self._tol)

    @property
    def xrange(self):
        return 3. - 0.1*self._tol

    @property
    def krange(self):
        # kmax = 2 * (25sqrt(5)/108 tol)^1/3
        return 1.6058208066649935 / self._tol**(1./3.)


class Lanczos(Interpolant):
    """The Lanczos interpolation filter, nominally sinc(x)*sinc(x/n)

    The Lanczos filter is an approximation to the band-limiting sinc filter with a smooth cutoff
    at high x.  Order n Lanczos has a range of +/- n pixels.  It typically is a good compromise
    between kernel size and accuracy.

    Note that pure Lanczos, when interpolating a set of constant-valued samples, does not return
    this constant.  Setting `conserve_dc` in the constructor tweaks the function so that it
    approximately conserves the value of constant (DC) input data (accurate to better than 1.e-5
    when used in two dimensions).

    @param n            The order of the Lanczos function
    @param conserve_dc  Whether to add the first order correction to flatten out the flux response
                        to a constant input. [default: True, see above]
    @param tol          This sets the accuracy and extent of the Fourier transform. [default 1.e-4]
    @param gsparams     An optional GSParams instance.  [default: None]
    """
    def __init__(self, n, conserve_dc=True, tol=1.e-4, gsparams=None):
        self._n = int(n)
        self._conserve_dc = bool(conserve_dc)
        self._tol = float(tol)
        self._gsparams = GSParams.check(gsparams)

    @lazy_property
    def _i(self):
        with convert_cpp_errors():
            return _galsim.Lanczos(self._n, self._conserve_dc, self._tol, self._gsparams._gsp)

    def __repr__(self):
        return "galsim.Lanczos(%r, %r, %r, %r)"%(self._n, self._conserve_dc, self._tol,
                                                 self._gsparams)

    def __str__(self):
        return "galsim.Lanczos(%s, %s)"%(self._n, self._tol)

    @property
    def xrange(self):
        return self._n - 0.1*self._tol

    @property
    def krange(self):
        return 2. * math.pi * self._i.urange()
