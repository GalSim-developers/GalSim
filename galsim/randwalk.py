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


class RandomWalk(galsim.GSObject):
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
    @param  gsparams                Optional GSParams for the gaussians
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
        .gaussians
            The list of galsim.Gaussian objects representing the points
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

    def __init__(self, npoints, half_light_radius, flux=1.0, rng=None, gsparams=None):

        self._half_light_radius = float(half_light_radius)

        self._flux    = float(flux)
        self._npoints = int(npoints)

        # size of the galsim.Gaussian objects to use as delta functions
        self._gaussian_sigma = 1.0e-8

        self._gsparams = galsim.GSParams.check(gsparams)

        # we will verify this in the _verify() method
        if rng is None:
            rng = galsim.BaseDeviate()

        self._rng=rng

        self._verify()

        self._set_gaussian_rng()

        self._points = self._get_points()
        self._make_sbp()

    def _make_sbp(self):
        self._gaussians = self._get_gaussians(self._points)
        self._sbp = galsim._galsim.SBAdd(self._gaussians, self.gsparams._gsp)

    def calculateHLR(self):
        """
        calculate the half light radius of the generated points
        """
        pts = self._points
        my,mx=pts.mean(axis=0)

        r=np.sqrt( (pts[:,0]-my)**2 + (pts[:,1]-mx)**2)

        hlr=np.median(r)

        return hlr

    @property
    def input_half_light_radius(self):
        """
        getter for the input half light radius
        """
        return self._half_light_radius

    @property
    def flux(self):
        """
        getter for the total flux
        """
        return self._flux

    @property
    def npoints(self):
        """
        getter for the number of points
        """
        return self._npoints

    @property
    def gaussians(self):
        """
        getter for the list of gaussians
        """
        return self._gaussians

    @property
    def points(self):
        """
        getter for the array of points, shape [npoints, 2]
        """
        return self._points.copy()

    def _get_gaussians(self, points):
        """
        Create galsim.Gaussian objects for each point.

        Highly optimized
        """

        gaussians = []
        sigma=self._gaussian_sigma
        fluxper=self._flux/self._npoints

        for p in points:
            g = galsim._galsim.SBGaussian(
                sigma=sigma,
                flux=fluxper,
                gsparams=self.gsparams._gsp,
            )

            g = galsim._galsim.SBTransform(
                g,
                1.0, 0.0, 0.0, 1.0,
                galsim._galsim.PositionD(p[0],p[1]),
                1.0,
                self.gsparams._gsp,
            )

            gaussians.append(g)

        return gaussians

    def _set_gaussian_rng(self):
        """
        Set the random number generator used to create the points

        We are approximating the random walk to have infinite number
        of steps, which is just a gaussian
        """

        # gaussian step size in each dimension for a random walk with infinite
        # number steps
        self._sigma_step = self._half_light_radius/2.3548200450309493*2

        self._gauss_rng = galsim.GaussianNoise(
            self._rng,
            sigma=self._sigma_step,
        )


    def _get_points(self):
        """
        We must use a galsim random number generator, in order for
        this profile to be used in the configuration file context.

        The most efficient way is to write into an image
        """
        ny=self._npoints
        nx=2
        im=galsim.ImageD(nx, ny)

        im.addNoise(self._gauss_rng)

        return im.array

    def _verify(self):
        """
        type and range checking on the inputs
        """
        if not isinstance(self._rng, galsim.BaseDeviate):
            raise TypeError("rng must be an instance of galsim.BaseDeviate, "
                            "got %s" % str(self._rng))

        if self._npoints <= 0:
            raise ValueError("npoints must be > 0, got %s" % str(self._npoints))

        if self._half_light_radius <= 0.0:
            raise ValueError("half light radius must be > 0"
                             ", got %s" % str(self._half_light_radius))
        if self._flux < 0.0:
            raise ValueError("flux must be >= 0, got %s" % str(self._flux))

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
        return (isinstance(other, galsim.RandomWalk) and
                self._npoints == other._npoints and
                self._half_light_radius == other._half_light_radius and
                self._flux == other._flux and
                self.gsparams == other.gsparams)

    def __hash__(self):
        return hash(("galsim.RandomWalk", self._npoints, self._half_light_radius, self._flux,
                     self.gsparams))

    def __getstate__(self):
        d = self.__dict__.copy()
        del d['_gaussians']
        del d['_sbp']
        return d

    def __setstate__(self, d):
        self.__dict__ = d
        self._make_sbp()
