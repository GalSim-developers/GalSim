# Copyright (c) 2012-2017 by the GalSim developers team on GitHub
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

import galsim
from . import _galsim
from .utilities import lazy_property


def FourierSqrt(obj, gsparams=None):
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

    @returns a FourierSqrtProfile or ChromaticFourierSqrtProfile instance as appropriate.
    """
    if isinstance(obj, galsim.ChromaticObject):
        return galsim.ChromaticFourierSqrtProfile(obj, gsparams=gsparams)
    elif isinstance(obj, galsim.GSObject):
        return FourierSqrtProfile(obj, gsparams=gsparams)
    else:
        raise TypeError("Argument to FourierSqrt must be either a GSObject or a ChromaticObject.")


class FourierSqrtProfile(galsim.GSObject):
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

    Methods
    -------

    There are no additional methods for FourierSqrtProfile beyond the usual GSObject methods.
    """
    def __init__(self, obj, gsparams=None):
        if not isinstance(obj, galsim.GSObject):
            raise TypeError("Argument to FourierSqrtProfile must be a GSObject.")

        # Save the original object as an attribute, so it can be inspected later if necessary.
        self._orig_obj = obj
        self._gsparams = galsim.GSParams.check(gsparams, self._orig_obj.gsparams)

        self._sbp = galsim._galsim.SBFourierSqrt(obj._sbp, self.gsparams._gsp)
        if obj.noise is not None:
            import warnings
            warnings.warn("Unable to propagate noise in galsim.FourierSqrtProfile")

    @property
    def orig_obj(self): return self._orig_obj

    def __eq__(self, other):
        return (isinstance(other, galsim.FourierSqrtProfile) and
                self.orig_obj == other.orig_obj and
                self.gsparams == other.gsparams)

    def __hash__(self):
        return hash(("galsim.FourierSqrtProfile", self.orig_obj, self.gsparams))

    def __repr__(self):
        return 'galsim.FourierSqrtProfile(%r, gsparams=%r)'%(self.orig_obj, self.gsparams)

    def __str__(self):
        return 'galsim.FourierSqrt(%s)'%self.orig_obj

    def _prepareDraw(self):
        self._orig_obj._prepareDraw()
        self._sbp = galsim._galsim.SBFourierSqrt(self._orig_obj._sbp, self.gsparams._gsp)

    def __getstate__(self):
        d = self.__dict__.copy()
        del d['_sbp']
        return d

    def __setstate__(self, d):
        self.__dict__ = d
        self.__init__(self._orig_obj, self._gsparams)
