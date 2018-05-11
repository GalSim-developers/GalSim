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
from .position import PositionD
from .utilities import lazy_property, doc_inherit
from .errors import GalSimRangeError, convert_cpp_errors

class RandomWalk(GSObject):
    """
    A class for generating a set of point sources distributed using a random
    walk.  Uses of this profile include representing an "irregular" galaxy, or
    adding this profile to an Exponential to represent knots of star formation.

    Random walk profiles have "shape noise" that depends on the number of point
    sources used.  For example, with 100 points the shape noise is g~0.05, and
    this will decrease as more points are added.  The profile can be sheared to
    give additional ellipticity, for example to follow that of an associated
    disk.

    We use the analytic approximation of an infinite number of steps, which is
    a good approximation even if the desired number of steps were less than 10.

    The requested half light radius (hlr) should be thought of as a rough
    value.  With a finite number point sources the actual realized hlr will be
    noisy.

    Initialization
    --------------
    @param  npoints                 Number of point sources to generate.
    @param  half_light_radius       Half light radius of the distribution of
                                    points.  This is the mean half light
                                    radius produced by an infinite number of
                                    points.  A single instance will be noisy.
    @param  flux                    Optional total flux in all point sources.
                                    [default: 1]
    @param  rng                     Optional random number generator. Can be
                                    any galsim.BaseDeviate.  If None, the rng
                                    is created internally.
                                    [default: None]
    @param  gsparams                Optional GSParams for the objects
                                    representing each point source.
                                    [default: None]

    Methods
    -------

    This class inherits from galsim.Sum. Additional methods are

        calculateHLR:
            Calculate the actual half light radius of the generated points

    There are also "getters",  implemented as read-only properties

        .npoints
        .input_half_light_radius
        .flux
        .points
            The array of x,y offsets used to create the point sources

    Notes
    -----

    - The algorithm is a modified version of that presented in

          https://arxiv.org/abs/1312.5514v3

      Modifications are
        1) there is no outer cutoff to how far a point can wander
        2) We use the approximation of an infinite number of steps.
    """
    # these allow use in a galsim configuration context

    _req_params = { "npoints" : int, "half_light_radius" : float }
    _opt_params = { "flux" : float }
    _single_params = []
    _takes_rng = True

    _has_hard_edges = False
    _is_axisymmetric = False
    _is_analytic_x = False
    _is_analytic_k = True

    def __init__(self, npoints, half_light_radius, flux=1.0, rng=None, gsparams=None):
        from .random import BaseDeviate

        self._npoints = int(npoints)
        self._half_light_radius = float(half_light_radius)
        self._flux = float(flux)
        self._gsparams = GSParams.check(gsparams)

        # we will verify this in the _verify() method
        if rng is None:
            rng = BaseDeviate()
        self._rng=rng
        self._verify()

        self._set_gaussian_rng()
        self._points = self._get_points()

    @lazy_property
    def _sbp(self):
        fluxper=self._flux/self._npoints
        deltas = []
        with convert_cpp_errors():
            for p in self._points:
                d = _galsim.SBDeltaFunction(fluxper, self.gsparams._gsp)
                d = _galsim.SBTransform(d, 1.0, 0.0, 0.0, 1.0, _galsim.PositionD(p[0],p[1]), 1.0,
                                        self.gsparams._gsp)
                deltas.append(d)
            return _galsim.SBAdd(deltas, self.gsparams._gsp)

    @property
    def input_half_light_radius(self):
        """
        The input half-light radius is not necessarily the realized hlr.
        """
        return self._half_light_radius

    @property
    def npoints(self):
        return self._npoints

    @property
    def points(self):
        return self._points

    def calculateHLR(self):
        """
        calculate the half-light radius of the generated points
        """
        pts = self._points
        my,mx=pts.mean(axis=0)

        r=np.sqrt( (pts[:,0]-my)**2 + (pts[:,1]-mx)**2)

        hlr=np.median(r)

        return hlr

    def _set_gaussian_rng(self):
        """
        Set the random number generator used to create the points

        We are approximating the random walk to have infinite number
        of steps, which is just a gaussian
        """
        from .random import GaussianDeviate
        # gaussian step size in each dimension for a random walk with infinite
        # number steps
        self._sigma_step = self._half_light_radius/2.3548200450309493*2
        self._gauss_rng = GaussianDeviate(self._rng, sigma=self._sigma_step)

    def _get_points(self):
        """
        We must use a galsim random number generator, in order for
        this profile to be used in the configuration file context.

        The most efficient way is to write into an image
        """
        ar = np.empty((self._npoints,2))
        self._gauss_rng.generate(ar)
        return ar

    def _verify(self):
        """
        type and range checking on the inputs
        """
        from .random import BaseDeviate
        if not isinstance(self._rng, BaseDeviate):
            raise TypeError("rng must be an instance of galsim.BaseDeviate, got %s"%self._rng)

        if self._npoints <= 0:
            raise GalSimRangeError("npoints must be > 0", self._npoints, 1)

        if self._half_light_radius <= 0.0:
            raise GalSimRangeError("half light radius must be > 0", self._half_light_radius, 0.)
        if self._flux < 0.0:
            raise GalSimRangeError("flux must be >= 0", self._flux, 0.)

    def __str__(self):
        rep='galsim.RandomWalk(%(npoints)d, %(hlr)g, flux=%(flux)g, gsparams=%(gsparams)s)'
        rep = rep % dict(
            npoints=self._npoints,
            hlr=self._half_light_radius,
            flux=self._flux,
            gsparams=str(self.gsparams),
        )
        return rep

    def __repr__(self):
        rep='galsim.RandomWalk(%(npoints)d, %(hlr).16g, flux=%(flux).16g, gsparams=%(gsparams)s)'
        rep = rep % dict(
            npoints=self._npoints,
            hlr=self._half_light_radius,
            flux=self._flux,
            gsparams=repr(self.gsparams),
        )
        return rep

    def __eq__(self, other):
        return (isinstance(other, RandomWalk) and
                self._npoints == other._npoints and
                self._half_light_radius == other._half_light_radius and
                self._flux == other._flux and
                self.gsparams == other.gsparams)

    def __hash__(self):
        return hash(("galsim.RandomWalk", self._npoints, self._half_light_radius, self._flux,
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

    @doc_inherit
    def _kValue(self, kpos):
        return self._sbp.kValue(kpos._p)

    @doc_inherit
    def _shoot(self, photons, ud):
        self._sbp.shoot(photons._pa, ud._rng)

    @doc_inherit
    def _drawKImage(self, image):
        self._sbp.drawK(image._image, image.scale)
