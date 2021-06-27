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

from .gsparams import GSParams
from .gsobject import GSObject
from .chromatic import ChromaticObject, ChromaticSum
from .utilities import lazy_property
from . import _galsim

def Add(*args, **kwargs):
    """A function for adding 2 or more `GSObject` or `ChromaticObject` instances.

    This function will inspect its input arguments to decide if a `Sum` object or a
    `ChromaticSum` object is required to represent the sum of surface brightness profiles.

    Typically, you do not need to call Add() explicitly.  Normally, you would just use the +
    operator, which returns a Sum::

        >>> bulge = galsim.Sersic(n=3, half_light_radius=0.8)
        >>> disk = galsim.Exponential(half_light_radius=1.4)
        >>> gal = bulge + disk
        >>> psf = galsim.Gaussian(sigma=0.3, flux=0.3) + galsim.Gaussian(sigma=0.8, flux=0.7)

    If one of the items is chromatic, it will return a `ChromaticSum`::

        >>> disk = galsim.Exponential(half_light_radius=1.4) * galsim.SED(sed_file)
        >>> gal = bulge + disk

    Parameters:
        args:               Unnamed args should be a list of objects to add.
        gsparams:           An optional `GSParams` argument. [default: None]
        propagate_gsparams: Whether to propagate gsparams to each of the components.  This
                            is normally a good idea, but there may be use cases where one
                            would not want to do this. [default: True]

    Returns:
        a `Sum` or `ChromaticSum` instance as appropriate.
    """
    if len(args) == 0:
        raise TypeError("At least one ChromaticObject or GSObject must be provided.")
    elif len(args) == 1:
        # 1 argument.  Should be either a GSObject or a list of GSObjects
        if isinstance(args[0], (GSObject, ChromaticObject)):
            args = [args[0]]
        elif isinstance(args[0], list) or isinstance(args[0], tuple):
            args = args[0]
        else:
            raise TypeError("Single input argument must be a GSObject, ChromaticObject or "
                            + "a (possibly mixed) list of them.")
    # else args is already the list of objects

    if any([isinstance(a, ChromaticObject) for a in args]):
        return ChromaticSum(*args, **kwargs)
    else:
        return Sum(*args, **kwargs)


class Sum(GSObject):
    """A class for adding 2 or more `GSObject` instances.

    The Sum class is used to represent the sum of multiple `GSObject` instances.  For example, it
    might be used to represent a multiple-component galaxy as the sum of an Exponential and a
    DeVaucouleurs, or to represent a PSF as the sum of multiple Gaussian objects.

    Typically, you do not need to construct a Sum object explicitly.  Normally, you would just
    use the + operator, which returns a Sum::

        >>> bulge = galsim.Sersic(n=3, half_light_radius=0.8)
        >>> disk = galsim.Exponential(half_light_radius=1.4)
        >>> gal = bulge + disk
        >>> psf = galsim.Gaussian(sigma=0.3, flux=0.3) + galsim.Gaussian(sigma=0.8, flux=0.7)

    You can also use the Add() factory function, which returns a Sum object if none of the
    individual objects are chromatic::

        >>> gal = galsim.Add([bulge,disk])

    Parameters:
        args:               Unnamed args should be a list of objects to add.
        gsparams:           An optional `GSParams` argument. [default: None]
        propagate_gsparams: Whether to propagate gsparams to each of the components.  This
                            is normally a good idea, but there may be use cases where one
                            would not want to do this. [default: True]

    Note: if ``gsparams`` is unspecified (or None), then the Sum instance will use the most
    restrictive combination of parameters from each of the component objects. Normally, this means
    the smallest numerical value (e.g. folding_threshold, xvalue_accuracy, etc.), but for a few
    parameters, the largest numerical value is used.  See `GSParams.combine` for details.

    Furthermore, the gsparams used for the Sum (either given explicitly or derived from the
    components) will normally be applied to each of the components.  It doesn't usually make much
    sense to apply stricter-than-normal accuracy or threshold values to one component but not
    another in a Sum, so this ensures that they all have consistent rendering behavior.
    However, if you want to keep the existing gsparams of the component objects (e.g. because
    one object is much fainter and can thus use looser accuracy tolerances), then you may
    set ``propagate_gsparams=False``.
    """
    def __init__(self, *args, **kwargs):

        # Check kwargs first:
        gsparams = kwargs.pop("gsparams", None)
        self._propagate_gsparams = kwargs.pop('propagate_gsparams', True)

        # Make sure there is nothing left in the dict.
        if kwargs:
            raise TypeError(
                "Sum constructor got unexpected keyword argument(s): %s"%kwargs.keys())

        if len(args) == 0:
            raise TypeError("At least one ChromaticObject or GSObject must be provided.")
        elif len(args) == 1:
            # 1 argument.  Should be either a GSObject or a list of GSObjects
            if isinstance(args[0], GSObject):
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
            if not isinstance(obj, GSObject):
                raise TypeError("Arguments to Sum must be GSObjects, not %s"%obj)

        # Figure out what gsparams to use
        if gsparams is None:
            # If none is given, take the most restrictive combination from the obj_list.
            self._gsparams = GSParams.combine([obj.gsparams for obj in args])
        else:
            # If something explicitly given, then use that.
            self._gsparams = GSParams.check(gsparams)

        # Apply gsparams to all in obj_list.
        if self._propagate_gsparams:
            self._obj_list = [obj.withGSParams(self._gsparams) for obj in args]
        else:
            self._obj_list = args


    @property
    def obj_list(self):
        """The list of objects being added.
        """
        return self._obj_list

    @lazy_property
    def _sbp(self):
        # NB. I only need this until compound and transform are reimplemented in Python...
        sb_list = [obj._sbp for obj in self.obj_list]
        return _galsim.SBAdd(sb_list, self.gsparams._gsp)

    @lazy_property
    def _flux(self):
        flux_list = [obj.flux for obj in self.obj_list]
        return np.sum(flux_list)

    @lazy_property
    def _noise(self):
        # If any of the objects have a noise attribute, then we propagate the sum of the
        # noises (they add like variances) to the final sum.
        _noise = None
        for obj in self.obj_list:
            if obj.noise is not None:
                if _noise is None:
                    _noise = obj.noise
                else:
                    _noise += obj.noise
        return _noise

    def withGSParams(self, gsparams=None, **kwargs):
        """Create a version of the current object with the given gsparams

        .. note::

            Unless you set ``propagate_gsparams=False``, this method will also update the gsparams
            of each object being added.
        """
        if gsparams == self.gsparams: return self
        from copy import copy
        ret = copy(self)
        ret._gsparams = GSParams.check(gsparams, self.gsparams, **kwargs)
        if self._propagate_gsparams:
            ret._obj_list = [ obj.withGSParams(ret._gsparams) for obj in self.obj_list ]
        return ret

    def __eq__(self, other):
        return (self is other or
                (isinstance(other, Sum) and
                 self.obj_list == other.obj_list and
                 self.gsparams == other.gsparams and
                 self._propagate_gsparams == other._propagate_gsparams))

    def __hash__(self):
        return hash(("galsim.Sum", tuple(self.obj_list), self.gsparams, self._propagate_gsparams))

    def __repr__(self):
        return 'galsim.Sum(%r, gsparams=%r, propagate_gsparams=%r)'%(self.obj_list, self.gsparams,
                self._propagate_gsparams)

    def __str__(self):
        str_list = [ str(obj) for obj in self.obj_list ]
        return '(' + ' + '.join(str_list) + ')'

    def _prepareDraw(self):
        for obj in self.obj_list:
            obj._prepareDraw()

    @lazy_property
    def _maxk(self):
        maxk_list = [obj.maxk for obj in self.obj_list]
        return np.max(maxk_list)

    @lazy_property
    def _stepk(self):
        stepk_list = [obj.stepk for obj in self.obj_list]
        return np.min(stepk_list)

    @lazy_property
    def _has_hard_edges(self):
        hard_list = [obj.has_hard_edges for obj in self.obj_list]
        return bool(np.any(hard_list))

    @lazy_property
    def _is_axisymmetric(self):
        axi_list = [obj.is_axisymmetric for obj in self.obj_list]
        return bool(np.all(axi_list))

    @lazy_property
    def _is_analytic_x(self):
        ax_list = [obj.is_analytic_x for obj in self.obj_list]
        return bool(np.all(ax_list))

    @lazy_property
    def _is_analytic_k(self):
        ak_list = [obj.is_analytic_k for obj in self.obj_list]
        return bool(np.all(ak_list))

    @lazy_property
    def _centroid(self):
        cen_list = [obj.centroid * obj.flux for obj in self.obj_list]
        return sum(cen_list[1:], cen_list[0]) / self.flux

    @lazy_property
    def _positive_flux(self):
        pflux_list = [obj.positive_flux for obj in self.obj_list]
        return np.sum(pflux_list)

    @lazy_property
    def _negative_flux(self):
        nflux_list = [obj.negative_flux for obj in self.obj_list]
        return np.sum(nflux_list)

    @lazy_property
    def _flux_per_photon(self):
        return self._calculate_flux_per_photon()

    @lazy_property
    def _max_sb(self):
        sb_list = [obj.max_sb for obj in self.obj_list]
        return np.sum(sb_list)

    def _xValue(self, pos):
        xv_list = [obj.xValue(pos) for obj in self.obj_list]
        return np.sum(xv_list)

    def _kValue(self, pos):
        kv_list = [obj.kValue(pos) for obj in self.obj_list]
        return np.sum(kv_list)

    def _drawReal(self, image, jac=None, offset=(0.,0.), flux_scaling=1.):
        self.obj_list[0]._drawReal(image, jac, offset, flux_scaling)
        if len(self.obj_list) > 1:
            im1 = image.copy()
            for obj in self.obj_list[1:]:
                obj._drawReal(im1, jac, offset, flux_scaling)
                image += im1

    def _shoot(self, photons, rng):
        from .photon_array import PhotonArray
        from .random import BinomialDeviate

        remainingAbsoluteFlux = self.positive_flux + self.negative_flux
        fluxPerPhoton = remainingAbsoluteFlux / len(photons)

        remainingN = len(photons)
        istart = 0  # The location in the photons array where we assign the component arrays.

        # Get photons from each summand, using BinomialDeviate to randomize
        # the distribution of photons among summands
        for i, obj in enumerate(self.obj_list):
            thisAbsoluteFlux = obj.positive_flux + obj.negative_flux

            # How many photons to shoot from this summand?
            thisN = remainingN  # All of what's left, if this is the last summand...
            if i < len(self.obj_list)-1:
                # otherwise, allocate a randomized fraction of the remaining photons to summand.
                bd = BinomialDeviate(rng, remainingN, thisAbsoluteFlux/remainingAbsoluteFlux)
                thisN = int(bd())
            if thisN > 0:
                thisPA = obj.shoot(thisN, rng)
                # Now rescale the photon fluxes so that they are each nominally fluxPerPhoton
                # whereas the shoot() routine would have made them each nominally
                # thisAbsoluteFlux/thisN
                thisPA.scaleFlux(fluxPerPhoton*thisN/thisAbsoluteFlux)
                photons.assignAt(istart, thisPA)
                istart += thisN
            remainingN -= thisN
            remainingAbsoluteFlux -= thisAbsoluteFlux
        #assert remainingN == 0
        #assert np.isclose(remainingAbsoluteFlux, 0.0)

        # This process produces correlated photons, so mark the resulting array as such.
        if len(self.obj_list) > 1:
            photons.setCorrelated()

    def _drawKImage(self, image, jac=None):
        self.obj_list[0]._drawKImage(image, jac)
        if len(self.obj_list) > 1:
            im1 = image.copy()
            for obj in self.obj_list[1:]:
                obj._drawKImage(im1, jac)
                image += im1

    def __getstate__(self):
        d = self.__dict__.copy()
        d.pop('_sbp',None)
        return d

    def __setstate__(self, d):
        self.__dict__ = d
