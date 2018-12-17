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

import numpy as np

from . import _galsim
from .gsparams import GSParams
from .gsobject import GSObject
from .chromatic import ChromaticObject
from .utilities import lazy_property, doc_inherit
from .errors import convert_cpp_errors, galsim_warn


def FourierSqrt(obj, gsparams=None, propagate_gsparams=True):
    """A function for computing the Fourier-space square root of either a GSObject or
    ChromaticObject.

    The FourierSqrt function is principally used for doing an optimal coaddition algorithm
    originally developed by Nick Kaiser (but unpublished) and also described by Zackay & Ofek 2015
    (http://adsabs.harvard.edu/abs/2015arXiv151206879Z).  See the script make_coadd.py in the
    GalSim/examples directory for an example of how it works.

    This function will inspect its input argument to decide if a FourierSqrtProfile object or a
    ChromaticFourierSqrtProfile object is required to represent the operation applied to a surface
    brightness profile.

    @param obj              The object to compute the Fourier-space square root of.
    @param gsparams         An optional GSParams argument.  See the docstring for GSParams for
                            details. [default: None]
    @param propagate_gsparams   Whether to propagate gsparams to the transformed object.  This
                                is normally a good idea, but there may be use cases where one
                                would not want to do this. [default: True]

    @returns a FourierSqrtProfile or ChromaticFourierSqrtProfile instance as appropriate.
    """
    from .chromatic import ChromaticFourierSqrtProfile
    if isinstance(obj, ChromaticObject):
        return ChromaticFourierSqrtProfile(obj, gsparams=gsparams,
                                           propagate_gsparams=propagate_gsparams)
    elif isinstance(obj, GSObject):
        return FourierSqrtProfile(obj, gsparams=gsparams, propagate_gsparams=propagate_gsparams)
    else:
        raise TypeError("Argument to FourierSqrt must be either a GSObject or a ChromaticObject.")


class FourierSqrtProfile(GSObject):
    """A class for computing the Fourier-space sqrt of a GSObject.

    The FourierSqrtProfile class represents the Fourier-space square root of another profile.
    Note that the FourierSqrtProfile class, or compound objects (Sum, Convolution) that include a
    FourierSqrtProfile as one of the components cannot be photon-shot using the 'phot' method of
    drawImage() method.

    You may also specify a `gsparams` argument.  See the docstring for GSParams using
    `help(galsim.GSParams)` for more information about this option.  Note: if `gsparams` is
    unspecified (or None), then the FourierSqrtProfile instance inherits the same GSParams as the
    object being operated on.

    Initialization
    --------------

    The normal way to use this class is to use the FourierSqrt() factory function:

        >>> fourier_sqrt = galsim.FourierSqrt(obj)

    @param obj              The object to compute Fourier-space square root of.
    @param gsparams         An optional GSParams argument.  See the docstring for GSParams for
                            details. [default: None]
    @param propagate_gsparams   Whether to propagate gsparams to the transformed object.  This
                                is normally a good idea, but there may be use cases where one
                                would not want to do this. [default: True]

    Methods
    -------

    There are no additional methods for FourierSqrtProfile beyond the usual GSObject methods.
    """
    _sqrt2 = 1.4142135623730951

    _has_hard_edges = False
    _is_analytic_x = False

    def __init__(self, obj, gsparams=None, propagate_gsparams=True):
        if not isinstance(obj, GSObject):
            raise TypeError("Argument to FourierSqrtProfile must be a GSObject.")

        # Save the original object as an attribute, so it can be inspected later if necessary.
        self._gsparams = GSParams.check(gsparams, obj.gsparams)
        self._propagate_gsparams = propagate_gsparams
        if self._propagate_gsparams:
            self._orig_obj = obj.withGSParams(self._gsparams)
        else:
            self._orig_obj = obj

    @property
    def orig_obj(self): return self._orig_obj

    @property
    def _sbp(self):
        with convert_cpp_errors():
            return _galsim.SBFourierSqrt(self.orig_obj._sbp, self.gsparams._gsp)

    @property
    def _noise(self):
        if self.orig_obj.noise is not None:
            galsim_warn("Unable to propagate noise in galsim.FourierSqrtProfile")
        return None

    @doc_inherit
    def withGSParams(self, gsparams):
        if gsparams == self.gsparams: return self
        from copy import copy
        ret = copy(self)
        ret._gsparams = GSParams.check(gsparams)
        if self._propagate_gsparams:
            ret._orig_obj = self._orig_obj.withGSParams(gsparams)
        return ret

    def __eq__(self, other):
        return (self is other or
                (isinstance(other, FourierSqrtProfile) and
                 self.orig_obj == other.orig_obj and
                 self.gsparams == other.gsparams and
                 self._propagate_gsparams == other._propagate_gsparams))

    def __hash__(self):
        return hash(("galsim.FourierSqrtProfile", self.orig_obj, self.gsparams,
                     self._propagate_gsparams))

    def __repr__(self):
        return 'galsim.FourierSqrtProfile(%r, gsparams=%r, propagate_gsparams=%r)'%(
                self.orig_obj, self.gsparams, self._propagate_gsparams)

    def __str__(self):
        return 'galsim.FourierSqrt(%s)'%self.orig_obj

    def _prepareDraw(self):
        self.orig_obj._prepareDraw()

    @property
    def _maxk(self):
        return self.orig_obj.maxk

    @property
    def _stepk(self):
        return self.orig_obj.stepk * self._sqrt2

    @property
    def _is_axisymmetric(self):
        return self.orig_obj.is_axisymmetric

    @property
    def _is_analytic_k(self):
        return self.orig_obj.is_analytic_k

    @property
    def _centroid(self):
        return 0.5 * self.orig_obj.centroid

    @property
    def _flux(self):
        return np.sqrt(self.orig_obj.flux)

    @property
    def _positive_flux(self):
        return np.sqrt(self.orig_obj.positive_flux)

    @property
    def _negative_flux(self):
        return np.sqrt(self.orig_obj.negative_flux)

    @property
    def _max_sb(self):
        # In this case, we want the autoconvolution of this object to get back to the
        # maxSB value of the original obj
        # flux * maxSB / 2 = maxSB_orig
        # maxSB = 2 * maxSB_orig / flux
        return 2. * self.orig_obj.max_sb / self.flux

    @doc_inherit
    def _kValue(self, pos):
        return np.sqrt(self.orig_obj._kValue(pos))

    @doc_inherit
    def _drawKImage(self, image):
        self.orig_obj._drawKImage(image)
        image.array[:,:] = np.sqrt(image.array)
