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
from .gsparams import GSParams
from .gsobject import GSObject
from .position import PositionD
from .utilities import lazy_property, doc_inherit
from .errors import GalSimRangeError, GalSimValueError, GalSimIncompatibleValuesError
from .gaussian import Gaussian

class RandomKnots(GSObject):
    """
    A class for generating a set of point sources, following either a `Gaussian` profile or a
    specified input profile.

    Uses of this profile include representing an "irregular" galaxy, or
    adding this profile to an Exponential to represent knots of star formation.

    RandomKnots profiles have "shape noise" that depends on the number of point
    sources used.  For example, using the default Gaussian distribution, with
    100 points, the shape noise is g~0.05, and this will decrease as more
    points are added.  The profile can be sheared to give additional
    ellipticity, for example to follow that of an associated disk.

    The requested half light radius (hlr) should be thought of as a rough
    value.  With a finite number point sources the actual realized hlr will be
    noisy.

    .. note::

        If providing an input ``profile`` object, it must be "shoot-able".  Objects that
        cannot be drawn with ``method='phot'`` cannot be used as the ``profile`` parameter here.

    Parameters:
         npoints:           Number of point sources to generate.
         half_light_radius: Optional half light radius of the distribution of points.  This value
                            is used for a Gaussian distribution if an explicit profile is not sent.
                            This is the mean half light radius produced by an infinite number of
                            points.  A single instance will be noisy.  [default None]
         flux:              Optional total flux in all point sources.  This value is used for a
                            Gaussian distribution if an explicit profile is not sent. Defaults to
                            None if profile is sent, otherwise 1.  [default: None]
         profile:           Optional profile to use for drawing points.  If a profile is sent, the
                            half_light_radius and flux keywords are invalid.  [default: None]
         rng:               Optional random number generator. Can be any `galsim.BaseDeviate`.  If
                            None, the rng is created internally.  [default: None]
         gsparams:          Optional `GSParams` for the objects representing each point source.
                            [default: None]

    Attributes:
        npoints:                    The number of points to use as knots
        input_half_light_radius:    The input half_light_radius
        flux:                       The flux
        points:                     The array of x,y offsets used to create the point sources

    .. note::

        The algorithm was originally a modified version of that presented in
        https://arxiv.org/abs/1312.5514v3.  However, we now use the GalSim photon shooting
        mechanism, which allows the knots to trace any profile, not just a `Gaussian`.
    """
    # these allow use in a galsim configuration context

    _req_params = { "npoints" : int }
    _opt_params = {
        "flux" : float ,
        "half_light_radius": float,
        "profile": GSObject,
    }
    _takes_rng = True

    _has_hard_edges = False
    _is_axisymmetric = False
    _is_analytic_x = False
    _is_analytic_k = True

    def __init__(self, npoints, half_light_radius=None, flux=None, profile=None, rng=None,
                 gsparams=None):
        from .random import BaseDeviate

        self._npoints=npoints
        self._half_light_radius = half_light_radius
        self._flux = flux
        self._profile=profile

        self._verify()
        self._gsparams = GSParams.check(gsparams)

        if rng is None:
            rng = BaseDeviate(rng)
            self._orig_rng=rng.duplicate()
        else:
            if not isinstance(rng, BaseDeviate):
                raise TypeError("rng must be an instance of galsim.BaseDeviate, got %s"%rng)
            self._orig_rng=rng.duplicate()
            # We won't use the rng yet, but make sure the original advances the right number
            # of values.
            rng.discard(2*npoints)

        if profile is None:
            if self._flux is None: self._flux = 1.0
            self._profile = Gaussian(half_light_radius=self._half_light_radius, flux=self._flux)

        else:
            self._flux=profile.flux
            try:
                # not all GSObjects have this attribute
                self._half_light_radius = profile.half_light_radius
            except Exception:
                self._half_light_radius = None


    @lazy_property
    def _sbp(self):
        fluxper = self._flux/self._npoints
        deltas = []
        for p in self.points:
            d = _galsim.SBDeltaFunction(fluxper, self.gsparams._gsp)
            d = _galsim.SBTransform(d, 0, p[0], p[1], 1.0, self.gsparams._gsp)
            deltas.append(d)
        return _galsim.SBAdd(deltas, self.gsparams._gsp)

    @property
    def input_half_light_radius(self):
        """
        Get the input half light radius (HLR).

        Note the input HLR is not necessarily the realized HLR,
        due to the finite number of points used in the profile.

        If a profile is sent, and that profile is a Transformation object (e.g.
        it has been sheared, its flux set, etc), then this value will be None.

        You can get the *calculated* half light radius using the calculateHLR
        method.  That value will be valid in all cases.
        """
        return self._half_light_radius

    @property
    def npoints(self):
        """The number of point sources.
        """
        return self._npoints

    @lazy_property
    def points(self):
        """A list of the locations (x,y) of the point sources.

        Technically, this is a numpy array of shape (npoints, 2).
        """
        rng = self._orig_rng.duplicate()
        photons = self._profile.shoot(self._npoints, rng)
        return np.column_stack([ photons.x, photons.y ])

    def calculateHLR(self):
        """
        calculate the half-light radius of the generated points
        """
        pts = self.points
        my,mx=pts.mean(axis=0)

        r=np.sqrt( (pts[:,0]-my)**2 + (pts[:,1]-mx)**2)

        hlr=np.median(r)

        return hlr

    def _verify(self):
        """
        type and range checking on the inputs
        """
        from .random import BaseDeviate

        try:
            self._npoints = int(self._npoints)
        except ValueError as err:
            raise GalSimValueError("npoints should be a number: %s", str(err))

        if self._npoints <= 0:
            raise GalSimRangeError("npoints must be > 0", self._npoints, 1)

        if self._profile is None:
            if self._half_light_radius is None:
                raise GalSimIncompatibleValuesError(
                    "half_light_radius required when not providing a profile")

            if self._half_light_radius <= 0.:
                raise GalSimRangeError(
                    "half_light_radius must be positive", self._half_light_radius, 0.)

        else:
            if self._flux is not None:
                raise GalSimIncompatibleValuesError("flux is invalid when providing a profile")

            if self._half_light_radius is not None:
                raise GalSimIncompatibleValuesError(
                    "half_light_radius is invalid when providing a profile")

            if not isinstance(self._profile, GSObject):
                raise GalSimIncompatibleValuesError("profile must be a GSObject")

    def __str__(self):
        rep = 'galsim.RandomKnots(%(npoints)d, profile=%(profile)s)'
        rep = rep % dict(
            npoints=self._npoints,
            profile=str(self._profile),
        )

        return rep

    def __repr__(self):
        rep = 'galsim.RandomKnots(%r, profile=%r, rng=%r, gsparams=%r)'%(
                self._npoints, self._profile, self._orig_rng, self._gsparams)
        return rep

    def __eq__(self, other):
        return (self is other or
                (isinstance(other, RandomKnots) and
                 self._npoints == other._npoints and
                 self._profile == other._profile and
                 self._orig_rng == other._orig_rng and
                 self._gsparams == other._gsparams))

    def __hash__(self):
        return hash(("galsim.RandomKnots", self._npoints, self._half_light_radius, self._flux,
                     self.gsparams))

    def __getstate__(self):
        d = self.__dict__.copy()
        d.pop('_sbp',None)
        return d

    def __setstate__(self, d):
        self.__dict__ = d

    @property
    def _maxk(self):
        return self._sbp.maxK()

    @property
    def _stepk(self):
        return self._sbp.stepK()

    @property
    def _centroid(self):
        return PositionD(self._sbp.centroid())

    @property
    def _positive_flux(self):
        return self._sbp.getPositiveFlux()

    @property
    def _negative_flux(self):
        return self._sbp.getNegativeFlux()

    @property
    def _max_sb(self):
        return self._sbp.maxSB()

    def _kValue(self, kpos):
        return self._sbp.kValue(kpos._p)

    def _shoot(self, photons, rng):
        self._sbp.shoot(photons._pa, rng._rng)

    def _drawKImage(self, image, jac=None):
        _jac = 0 if jac is None else jac.__array_interface__['data'][0]
        self._sbp.drawK(image._image, image.scale, _jac)

    # For all the transformations methods, apply to the internal profile, and remake points
    # in the correct locations.  This makes fft drawing much faster than the normal way
    # of applying the transformation to the k-space image.
    @doc_inherit
    def withFlux(self, flux):
        return RandomKnots(self.npoints, profile=self._profile.withFlux(flux),
                           rng=self._orig_rng.duplicate(), gsparams=self.gsparams)

    @doc_inherit
    def withScaledFlux(self, flux_ratio):
        if hasattr(flux_ratio, '__call__'):
            return GSObject.withScaledFlux(self, flux_ratio)
        else:
            return RandomKnots(self._npoints, profile=self._profile.withScaledFlux(flux_ratio),
                               rng=self._orig_rng.duplicate(), gsparams=self._gsparams)

    @doc_inherit
    def expand(self, scale):
        return RandomKnots(self._npoints, profile=self._profile.expand(scale),
                           rng=self._orig_rng.duplicate(), gsparams=self._gsparams)

    @doc_inherit
    def dilate(self, scale):
        return RandomKnots(self._npoints, profile=self._profile.dilate(scale),
                           rng=self._orig_rng.duplicate(), gsparams=self._gsparams)

    @doc_inherit
    def shear(self, *args, **kwargs):
        return RandomKnots(self._npoints, profile=self._profile.shear(*args, **kwargs),
                           rng=self._orig_rng.duplicate(), gsparams=self._gsparams)

    def _shear(self, shear):
        return RandomKnots(self._npoints, profile=self._profile._shear(shear),
                           rng=self._orig_rng.duplicate(), gsparams=self._gsparams)

    @doc_inherit
    def rotate(self, theta):
        return RandomKnots(self._npoints, profile=self._profile.rotate(theta),
                           rng=self._orig_rng.duplicate(), gsparams=self._gsparams)

    @doc_inherit
    def transform(self, dudx, dudy, dvdx, dvdy):
        return RandomKnots(self._npoints, profile=self._profile.transform(dudx,dudy,dvdx,dvdy),
                           rng=self._orig_rng.duplicate(), gsparams=self._gsparams)

    @doc_inherit
    def shift(self, *args, **kwargs):
        return RandomKnots(self._npoints, profile=self._profile.shift(*args, **kwargs),
                           rng=self._orig_rng.duplicate(), gsparams=self._gsparams)

    def _shift(self, dx, dy):
        return RandomKnots(self._npoints, profile=self._profile._shift(dx,dy),
                           rng=self._orig_rng.duplicate(), gsparams=self._gsparams)
