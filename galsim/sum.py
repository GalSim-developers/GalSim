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

def Add(*args, **kwargs):
    """A function for adding 2 or more GSObject or ChromaticObject instances.

    This function will inspect its input arguments to decide if a Sum object or a
    ChromaticSum object is required to represent the sum of surface brightness profiles.

    Typically, you do not need to call Add() explicitly.  Normally, you would just use the +
    operator, which returns a Sum:

        >>> bulge = galsim.Sersic(n=3, half_light_radius=0.8)
        >>> disk = galsim.Exponential(half_light_radius=1.4)
        >>> gal = bulge + disk
        >>> psf = galsim.Gaussian(sigma=0.3, flux=0.3) + galsim.Gaussian(sigma=0.8, flux=0.7)

    If one of the items is chromatic, it will return a ChromaticSum

        >>> disk = galsim.Exponential(half_light_radius=1.4) * galsim.SED(sed_file)
        >>> gal = bulge + disk

    @param args             Unnamed args should be a list of objects to add.
    @param gsparams         An optional GSParams argument.  See the docstring for GSParams for
                            details. [default: None]

    @returns a Sum or ChromaticSum instance as appropriate.
    """
    if len(args) == 0:
        raise TypeError("At least one ChromaticObject or GSObject must be provided.")
    elif len(args) == 1:
        # 1 argument.  Should be either a GSObject or a list of GSObjects
        if isinstance(args[0], (galsim.GSObject, galsim.ChromaticObject)):
            args = [args[0]]
        elif isinstance(args[0], list) or isinstance(args[0], tuple):
            args = args[0]
        else:
            raise TypeError("Single input argument must be a GSObject, ChromaticObject or "
                            + "a (possibly mixed) list of them.")
    # else args is already the list of objects

    if any([isinstance(a, galsim.ChromaticObject) for a in args]):
        return galsim.ChromaticSum(*args, **kwargs)
    else:
        return Sum(*args, **kwargs)


class Sum(galsim.GSObject):
    """A class for adding 2 or more GSObject instances.

    The Sum class is used to represent the sum of multiple GSObject instances.  For example, it
    might be used to represent a multiple-component galaxy as the sum of an Exponential and a
    DeVaucouleurs, or to represent a PSF as the sum of multiple Gaussian objects.

    Initialization
    --------------

    Typically, you do not need to construct a Sum object explicitly.  Normally, you would just
    use the + operator, which returns a Sum:

        >>> bulge = galsim.Sersic(n=3, half_light_radius=0.8)
        >>> disk = galsim.Exponential(half_light_radius=1.4)
        >>> gal = bulge + disk
        >>> psf = galsim.Gaussian(sigma=0.3, flux=0.3) + galsim.Gaussian(sigma=0.8, flux=0.7)

    You can also use the Add() factory function, which returns a Sum object if none of the
    individual objects are chromatic:

        >>> gal = galsim.Add([bulge,disk])

    @param args             Unnamed args should be a list of objects to add.
    @param gsparams         An optional GSParams argument.  See the docstring for GSParams for
                            details. [default: None]

    Note: if `gsparams` is unspecified (or None), then the Sum instance inherits the same GSParams
    as the first item in the list.  Also, note that parameters related to the Fourier-space
    calculations must be set when initializing the individual GSObject instances that go into the
    Sum, NOT when creating the Sum (at which point the accuracy and threshold parameters will simply
    be ignored).

    Methods
    -------

    There are no additional methods for Sum beyond the usual GSObject methods.
    """
    def __init__(self, *args, **kwargs):

        # Check kwargs first:
        gsparams = kwargs.pop("gsparams", None)

        # Make sure there is nothing left in the dict.
        if kwargs:
            raise TypeError(
                "Sum constructor got unexpected keyword argument(s): %s"%kwargs.keys())

        if len(args) == 0:
            raise TypeError("At least one ChromaticObject or GSObject must be provided.")
        elif len(args) == 1:
            # 1 argument.  Should be either a GSObject or a list of GSObjects
            if isinstance(args[0], galsim.GSObject):
                args = [args[0]]
            elif isinstance(args[0], list) or isinstance(args[0], tuple):
                args = args[0]
            else:
                raise TypeError("Single input argument must be a GSObject or list of them.")
        # else args is already the list of objects

        # Consolidate args for Sums of Sums...
        new_args = []
        for a in args:
            if isinstance(a, Sum):
                new_args.extend(a._obj_list)
            else:
                new_args.append(a)
        args = new_args

        # Save the list as an attribute, so it can be inspected later if necessary.
        self._obj_list = args

        for obj in args:
            if not isinstance(obj, galsim.GSObject):
                raise TypeError("Arguments to Sum must be GSObjects, not %s"%obj)
        self._gsparams = galsim.GSParams.check(gsparams, self._obj_list[0].gsparams)
        SBList = [obj._sbp for obj in args]
        self._sbp = galsim._galsim.SBAdd(SBList, self.gsparams._gsp)

    @lazy_property
    def noise(self):
        # If any of the objects have a noise attribute, then we propagate the sum of the
        # noises (they add like variances) to the final sum.
        _noise = None
        for obj in self._obj_list:
            if obj.noise is not None:
                if _noise is None:
                    _noise = obj.noise
                else:
                    _noise += obj.noise
        return _noise

    @property
    def obj_list(self): return self._obj_list

    def __eq__(self, other):
        return (isinstance(other, galsim.Sum) and
                self.obj_list == other.obj_list and
                self.gsparams == other.gsparams)

    def __hash__(self):
        return hash(("galsim.Sum", tuple(self.obj_list), self.gsparams))

    def __repr__(self):
        return 'galsim.Sum(%r, gsparams=%r)'%(self.obj_list, self.gsparams)

    def __str__(self):
        str_list = [ str(obj) for obj in self.obj_list ]
        return '(' + ' + '.join(str_list) + ')'
        #return 'galsim.Sum([%s])'%', '.join(str_list)

    def _prepareDraw(self):
        for obj in self._obj_list:
            obj._prepareDraw()
        SBList = [obj._sbp for obj in self._obj_list]
        self._sbp = galsim._galsim.SBAdd(SBList, self.gsparams._gsp)

    def shoot(self, n_photons, rng=None):
        """Shoot photons into a PhotonArray.

        @param n_photons    The number of photons to use for photon shooting.
        @param rng          If provided, a random number generator to use for photon shooting,
                            which may be any kind of BaseDeviate object.  If `rng` is None, one
                            will be automatically created, using the time as a seed.
                            [default: None]
        @returns PhotonArray.
        """
        if n_photons == 0:
            return galsim.PhotonArray(0)
        ud = galsim.UniformDeviate(rng)

        remainingAbsoluteFlux = self.positive_flux + self.negative_flux
        fluxPerPhoton = remainingAbsoluteFlux / n_photons

        # Initialize the output array
        photons = galsim.PhotonArray(n_photons)

        remainingN = n_photons
        istart = 0  # The location in the photons array where we assign the component arrays.

        # Get photons from each summand, using BinomialDeviate to randomize
        # the distribution of photons among summands
        for i, obj in enumerate(self.obj_list):
            thisAbsoluteFlux = obj.positive_flux + obj.negative_flux

            # How many photons to shoot from this summand?
            thisN = remainingN  # All of what's left, if this is the last summand...
            if i < len(self.obj_list)-1:
                # otherwise, allocate a randomized fraction of the remaining photons to summand.
                bd = galsim.BinomialDeviate(ud, remainingN, thisAbsoluteFlux/remainingAbsoluteFlux)
                thisN = int(bd())
            if thisN > 0:
                thisPA = obj.shoot(thisN, ud)
                # Now rescale the photon fluxes so that they are each nominally fluxPerPhoton
                # whereas the shoot() routine would have made them each nominally
                # thisAbsoluteFlux/thisN
                thisPA.scaleFlux(fluxPerPhoton*thisN/thisAbsoluteFlux)
                photons.assignAt(istart, thisPA)
                istart += thisN
            remainingN -= thisN
            remainingAbsoluteFlux -= thisAbsoluteFlux
        assert remainingN == 0
        assert np.isclose(remainingAbsoluteFlux, 0.0)

        # This process produces correlated photons, so mark the resulting array as such.
        if len(self.obj_list) > 1:
            photons.setCorrelated()
        return photons

    def __getstate__(self):
        d = self.__dict__.copy()
        del d['_sbp']
        return d

    def __setstate__(self, d):
        self.__dict__ = d
        self.__init__(self._obj_list, gsparams=self.gsparams)
