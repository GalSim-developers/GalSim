# Copyright (c) 2012-2023 by the GalSim developers team on GitHub
# https://github.com/GalSim-developers
#
# This file is part of GalSim: The modular galaxy image simulation toolkit.
# https://github.com/GalSim-developers/GalSim
#
# GalSim is free software: redistribution and use in source and binary forms,
# with or without modification, are permitted provided that the following
# conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice,
#    this list of conditions, and the disclaimer given in the accompanying
#    LICENSE file.
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions, and the disclaimer given in the documentation
#    and/or other materials provided with the distribution.
#

__all__ = ['Spiral']

import numpy as np
from numpy import pi as PI
from . import _galsim
from .angle import Angle
from .shear import Shear
from .gsobject import GSObject, GSParams
from .random import BaseDeviate
from .utilities import lazy_property, doc_inherit, parse_pos_args
from .errors import (
    GalSimValueError,
    GalSimRangeError,
)
from .position import PositionD
from coord import degrees, radians

GOLDEN_SPIRAL_PITCH = 17.03239

PIXEL_SCALE = 0.2


class Spiral(GSObject):
    """
    A class for generating a galaxy with logorithmic spiral arms.

    The profile is represented by a set of point sources.

    The requested half light radius (hlr) should be thought of as a rough
    value.  With a finite number point sources the actual realized hlr will be
    noisy.

    Parameters:
        npoints: int
            Number of point sources to generate in profile.
        half_light_radius: float
            half light radius of the distribution of points.  This is the mean
            half light radius produced by an infinite number of points.  A
            single instance will be noisy.
        flux: float, optional
            Optional total flux, default 1.0
        narms: int, optional
            Optional number of spiral arms, default 2
        pitch: galsim.Angle, optional
            Pitch angle for spiral, default 17.03239 degres, approximately that
            of the golden spiral.
        angle_fuzz: galsim.Angle, optional
            Sigma for the Gaussian fuzz for the angle of each point.
            This helps spread out the spiral.  Default 0.03 radians.
        xy_fuzz: float, optional
            Sigma for Gaussian fuzz on the final x,y positions, in units of
            the half light radius.  This helps to spread out the spiral.
            Default is 0.1
        rel_height: float, optional
            The disk will have a z component with scale radius
            rel_height times the scale of the radial profile.  Default 0.1
        inclination: galsim.Angle, optional
            The inclination of the disk along the line of sight, 0 degrees is
            face on and 90 is edge on.  The input must be a galsim.Angle
            object.  Default (None) is to give it a random inclination.
        rotation: galsim.Angle, optional
            Rotation of the inclined galaxy in the plane of the sky.  Default
            (None) is to give it a random rotation.
        rng: galsim random number generator, optional
            Optional random number generator. Can be any `galsim.BaseDeviate`.
            If None, the rng (the default), it is created internally.
        gsparams: GSParams, optional
            Optional `GSParams` for the objects representing each point source.
    """
    # these allow use in a galsim configuration context

    _opt_params = {
        "npoints": int,
        "half_light_radius": float,
        "flux": float,
        "narms": int,
        "pitch": Angle,
        "angle_fuzz": float,
        "xy_fuzz": float,
        "rel_height": float,
        "inclination": Angle,
        "rotation": Angle,

    }
    _takes_rng = True

    _has_hard_edges = False
    _is_axisymmetric = False
    _is_analytic_x = False
    _is_analytic_k = True

    def __init__(
        self,
        npoints=None,
        half_light_radius=None,
        flux=1,

        narms=2,
        pitch=None,
        angle_fuzz=0.05,
        xy_fuzz=0.15,

        rel_height=0.1,
        inclination=None,
        rotation=None,

        # we can also just send in the points and other inputs are ignored,
        # other than flux.
        #
        # We don't advertise this version. It is used when doing transforms,
        # where we make sure metadata keeps in sync as best as possible.  If
        # the user does it wrong the metadata can get out of sync
        points=None,

        rng=None,
        gsparams=None,
    ):

        if points is None and npoints is None:
            raise RuntimeError(
                'Spiral must be constructed with either points or npoints',
            )

        self._set_attr_check('_flux', flux, float)
        self._set_rng(rng)
        self._gsparams = GSParams.check(gsparams)

        self._set_attr_check('_narms', narms, int, gtzero=True)
        self._set_attr_check('_angle_fuzz', angle_fuzz, float, gezero=True)
        self._set_attr_check('_xy_fuzz', xy_fuzz, float, gezero=True)
        self._set_attr_check('_rel_height', rel_height, float, gezero=True)

        self._points = None

        if points is not None:
            self._set_points(points)
            self._npoints = self._points.shape[0]

            self._pitch = pitch
            self._inclination = inclination
            self._rotation = rotation
            self._half_light_radius = half_light_radius
        else:
            self._set_attr_check('_npoints', npoints, int, gtzero=True)
            self._set_attr_check(
                '_half_light_radius', half_light_radius, float, gezero=True,
            )
            self._part = SpiralParticles(
                rng=self._np_rng,
                hlr=half_light_radius,

                narms=self.narms,
                pitch=pitch,
                angle_fuzz=self.angle_fuzz,
                xy_fuzz=self.xy_fuzz,

                rel_height=self.rel_height,
                inclination=inclination,
                rotation=rotation,
            )
            # these could have been converted from None, so need
            # to get from particles
            self._pitch = self._part.pitch
            self._inclination = self._part.inclination
            self._rotation = self._part.rotation

    def _set_attr_check(self, name, val, vtype, gtzero=False, gezero=False):
        try:
            conv_val = vtype(val)
            setattr(self, name, conv_val)
        except ValueError:
            raise GalSimValueError(
                f'Could not convert {name}={val} to {vtype}', val
            ) from None

        if gtzero or gezero:
            gval = getattr(self, name)
            if gtzero and gval <= 0:
                raise GalSimRangeError(
                    f'{name} must be > 0', gval, 1,
                )
            elif gval < 0:
                raise GalSimRangeError(
                    f'{name} must be >= 0', gval, 1,
                )

    def _set_points(self, points):
        try:
            shape = points.shape
            if len(shape) != 2 or shape[1] != 3:
                raise GalSimValueError(
                    f'points must be an [n, 3] array, got {shape}', points
                )
        except AttributeError:
            tp = type(points)
            raise GalSimValueError(
                f'points must be an array, got {tp}',
                points
            )

        self._points = points

    def _set_rng(self, rng):
        if rng is None:
            rng = BaseDeviate(rng)
        else:
            if not isinstance(rng, BaseDeviate):
                raise TypeError(
                    "rng must be an instance of galsim.BaseDeviate, "
                    "got %s" % rng
                )

        self._orig_rng = rng.duplicate()
        self._rng = rng
        self._np_rng = np.random.default_rng(self._rng.raw())

    @lazy_property
    def _sbp(self):
        fluxper = self._flux/self._npoints
        deltas = []
        for p in self.points:
            dx, dy, dz = p
            d = _galsim.SBDeltaFunction(fluxper, self.gsparams._gsp)
            d = _galsim.SBTransform(d, 0, dx, dy, 1.0, self.gsparams._gsp)
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

    def _get_particles(self):
        if not hasattr(self, '_part'):
            raise RuntimeError(
                'Cannot get particles, Spiral was constructed from '
                'points'
            )
        return self._part

    @property
    def narms(self):
        """
        number of spiral arms
        """
        return self._narms

    @property
    def pitch(self):
        """
        pitch angle of arms
        """
        return self._pitch

    @property
    def angle_fuzz(self):
        """
        Gaussian fuzz for angles
        """
        return self._angle_fuzz

    @property
    def xy_fuzz(self):
        """
        Gaussian fuzz for x, y in plane of disk
        """
        return self._xy_fuzz

    @property
    def rel_height(self):
        """
        Relative height of disk as a ratio to the radial profile
        """
        return self._rel_height

    @property
    def inclination(self):
        """
        Inclination of disk along the line of sight
        """
        return self._inclination

    @property
    def rotation(self):
        """
        Rotation in the plane of the sky. Can be out of date after
        transormations
        """
        return self._rotation

    @property
    def npoints(self):
        """
        The number of point sources.
        """
        return self.points.shape[0]

    # @lazy_property
    @property
    def points(self):
        """
        Get a read only view of the points (x, y, z) in the profile.

        Returns numpy array of shape (npoints, 3).
        """
        # this is an [npoints, 3] array
        if self._points is None:
            # points need to be constructed if we didn't input then directlyh
            self._points = self._part.sample(self._npoints)
            self._points.flags['WRITEABLE'] = False

        return self._points

    def calculateHLR(self):
        """
        calculate the half-light radius of the generated points
        """
        pts = self.points
        mx, my, mz = pts.mean(axis=0)

        r = np.sqrt((pts[:, 0] - mx)**2 + (pts[:, 1] - my)**2)

        hlr = np.median(r)

        return hlr

    def calculate_e1e2(self):
        """
        calculate e1, e2 from the points

        Returns
        -------
        e1: float
            e1 calculated from the points
        e2: float
            e2 calculated from the points
        """
        points = self.points
        n = self.npoints

        xdiff = points[:, 0] - points[:, 0].mean()
        ydiff = points[:, 1] - points[:, 1].mean()

        Ixx = (xdiff**2).sum() / n
        Ixy = (xdiff * ydiff).sum() / n
        Iyy = (ydiff**2).sum() / n

        e1 = (Ixx - Iyy) / (Ixx + Iyy)
        e2 = 2 * Ixy / (Ixx + Iyy)
        return e1, e2

    def __str__(self):
        # don't try to give a full repr, since if the data have been
        # transformed a simple construction will not work
        return 'galsim.Spiral'

    def __repr__(self):
        # don't try to give a full repr, since if the data have been
        # transformed a simple construction will not work
        return 'galsim.Spiral'

    def __eq__(self, other):
        return (self is other or
                (isinstance(other, Spiral) and
                 self.npoints == other.npoints and
                 self._orig_rng == other._orig_rng and
                 self._gsparams == other._gsparams))

    def __hash__(self):
        return hash(
            ("galsim.Spiral",
             self._npoints,
             self._half_light_radius,
             self._flux,
             self._narms,
             self._pitch,
             self._angle_fuzz,
             self._xy_fuzz,
             self._rel_height,
             self._inclination,
             self._rotation,
             self.gsparams)
        )

    def __getstate__(self):
        d = self.__dict__.copy()
        d.pop('_sbp', None)
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

    def _get_constructor_kw(self):
        return dict(
            points=self.points,
            flux=self.flux,
            half_light_radius=self.input_half_light_radius,

            narms=self.narms,
            pitch=self.pitch,
            angle_fuzz=self.angle_fuzz,
            xy_fuzz=self.xy_fuzz,

            rel_height=self.rel_height,
            inclination=self.inclination,
            # this can be out of date if we transformed points
            rotation=self._rotation,

            rng=self._orig_rng.duplicate(),
            gsparams=self.gsparams,
        )

    # For some transformation methods we can speed things up by transforming
    # the points rather than making DeltaFunctions and transforming them.  This
    # makes fft drawing much faster than the normal way of applying the
    # transformation to the k-space image.

    @doc_inherit
    def withFlux(self, flux):
        kw = self._get_constructor_kw()
        kw['flux'] = flux
        return Spiral(**kw)

    @doc_inherit
    def withScaledFlux(self, flux_ratio):
        if hasattr(flux_ratio, '__call__'):
            return GSObject.withScaledFlux(self, flux_ratio)
        else:
            return self.withFlux(flux_ratio * self.flux)

    @doc_inherit
    def dilate(self, scale):
        points = self.points.copy()
        points[:, 0] *= scale
        points[:, 1] *= scale

        hlr = self.input_half_light_radius
        if hlr is not None:
            hlr = hlr * scale

        kw = self._get_constructor_kw()
        kw['points'] = points
        kw['half_light_radius'] = hlr

        return Spiral(**kw)

    @doc_inherit
    def expand(self, scale):
        points = self.points.copy()
        points[:, 0] *= scale
        points[:, 1] *= scale

        hlr = self.input_half_light_radius
        if hlr is not None:
            hlr = hlr * scale

        kw = self._get_constructor_kw()
        kw['points'] = points
        kw['half_light_radius'] = hlr
        kw['flux'] = self.flux / scale**2

        return Spiral(**kw)

    @doc_inherit
    def magnify(self, scale):
        return self.expand(scale)

    @doc_inherit
    def shear(self, *args, **kwargs):
        if len(args) == 1:
            if kwargs:
                raise TypeError(
                    'Error, gave both unnamed and named arguments to '
                    'shear'
                )

            if not isinstance(args[0], Shear):
                raise TypeError(
                    'Error, unnamed argument to GSObject.shear is '
                    'not a Shear!'
                )
            shear = args[0]
        elif len(args) > 1:
            raise TypeError("Error, too many unnamed arguments to shear")
        elif len(kwargs) == 0:
            raise TypeError("Error, shear argument is required")
        else:
            shear = Shear(**kwargs)
        return self._shear(shear)

    def _shear(self, shear):
        mat = shear.getMatrix()
        return self.transform(
            dudx=mat[0, 0],
            dudy=mat[0, 1],
            dvdx=mat[1, 0],
            dvdy=mat[1, 1],
        )

    @doc_inherit
    def transform(self, dudx, dudy, dvdx, dvdy):
        points = self.points.copy()
        points[:, 0] = (
            dudx * self.points[:, 0] + dudy * self.points[:, 1]
        )
        points[:, 1] = (
            dvdx * self.points[:, 0] + dvdy * self.points[:, 1]
        )

        scale = np.sqrt(dudx * dvdy - dudy * dvdx)
        hlr = self.input_half_light_radius
        if hlr is not None:
            hlr = hlr * scale

        kw = self._get_constructor_kw()
        kw['points'] = points
        kw['half_light_radius'] = hlr

        return Spiral(**kw)

    @doc_inherit
    def shift(self, *args, **kwargs):
        shiftobj = parse_pos_args(args, kwargs, 'dx', 'dy')
        return self._shift(shiftobj.x, shiftobj.y)

    def _shift(self, dx, dy):
        points = self.points.copy()
        points[:, 0] += dx
        points[:, 1] += dy

        kw = self._get_constructor_kw()
        kw['points'] = points

        return Spiral(**kw)

    @doc_inherit
    def rotate(self, theta):

        if not isinstance(theta, Angle):
            raise TypeError("Input theta should be an Angle")

        points = self.points.copy()
        points[:, 0], points[:, 1] = _rotate_coords(
            self.points[:, 0], self.points[:, 1], theta,
        )

        kw = self._get_constructor_kw()
        kw['points'] = points

        return Spiral(**kw)


class SpiralParticles(object):
    """
    Make a spiral galaxy with a logarithmic spiral

    Parameters
    ----------
    rng: np.random.RandomState
        The random number generator
    hlr: float
        Half light radius of the radial profile.  Note the scale r0 is hlr /
        1.67835
    narms: int
        Number of arms for spiral, default 2
    pitch: galsim.Angle
        Pitch angle for spiral, default 17.03239, approximately that of
        the golden spiral
    angle_fuzz: float
        Sigma for the Gaussian fuzz for the angle of each point.
    xy_fuzz: float
        Sigma for Gaussian fuzz for the final x,y positions, in units of the
        half light radius
    rel_height: float
        The disk will have a z component with scale radius rel_height times
        the scale of the radial profile.  Default is 0.1
    inclination: galsim.Angle
        The inclination of the disk along the line of sight, 0 is face on
        and 90 is edge on.  Default is to give it a random inclination.
    rotation: galsim.Angle
        Rotation of the inclined galaxy in the plane of the sky.  Default
        is to give it a random rotation.
    """
    def __init__(
        self,
        rng,
        hlr,
        narms=2,
        pitch=None,
        angle_fuzz=0.03,
        xy_fuzz=0.1,
        rel_height=0.1,
        inclination=None,
        rotation=None,
        shift=None,
    ):

        self.rng = rng
        self.narms = int(narms)
        if self.narms <= 0:
            raise ValueError(f'number of arms should be > 0, got {self.narms}')
        self.hlr = float(hlr)
        if self.hlr <= 0:
            raise ValueError(f'hlr must be > 0, got {self.hlr}')

        self.r0 = self.hlr / 1.67835

        self.angle_fuzz = angle_fuzz
        self.xy_fuzz = xy_fuzz
        self.rel_height = rel_height

        if pitch is None:
            pitch = GOLDEN_SPIRAL_PITCH * degrees

        if inclination is None:
            inclination = self.rng.uniform(low=0, high=2 * PI) * radians

        if rotation is None:
            rotation = self.rng.uniform(low=0, high=2 * PI) * radians

        self.pitch = pitch
        self.inclination = inclination
        self.rotation = rotation
        self.shift = shift

        _check_angle(self.pitch, 'pitch')
        _check_angle(self.inclination, 'inclination')
        _check_angle(self.rotation, 'rotation')
        _check_sequence(self.shift, 'shift', 2)

    def sample(self, n):
        """
        theta = (1/B) ln( r/r0 )

        b = 0.3063489 for theta in radians
        """

        r = self._sample_r(n)

        theta = np.log(r / self.r0) / self.pitch.tan()

        if self.angle_fuzz is not None:
            theta += theta * self.rng.normal(
                scale=self.angle_fuzz,
                size=theta.size,
            )

        # split points into arms
        if self.narms > 1:
            angle = 2 * np.pi / self.narms
            frac = int(r.size / self.narms)
            start = frac
            for i in range(self.narms-1):
                start = (i+1) * frac
                end = (i+2) * frac
                theta[start:end] += (i+1) * angle

        xyz = np.zeros((n, 3))
        xyz[:, 0] = r * np.cos(theta)
        xyz[:, 1] = r * np.sin(theta)

        if self.xy_fuzz is not None:
            xyz[:, 0] += self.rng.normal(
                scale=self.xy_fuzz * self.hlr, size=r.size,
            )
            xyz[:, 1] += self.rng.normal(
                scale=self.xy_fuzz * self.hlr, size=r.size,
            )

        xyz[:, 2] = self._get_z(r.size)

        # first give it an inclination, rotating around the x axis
        # y, z = _rotate_coords(y, z, self.inclination)
        xyz[:, 1], xyz[:, 2] = _rotate_coords(
            xyz[:, 1], xyz[:, 2], self.inclination,
        )

        # now a random rotation in the plane of the sky
        # x, y = _rotate_coords(x, y, self.rotation)
        xyz[:, 0], xyz[:, 1] = _rotate_coords(
            xyz[:, 0], xyz[:, 1], self.rotation,
        )

        if self.shift is not None:
            xyz[:, 0] += self.shift[0]
            xyz[:, 1] += self.shift[1]

        return xyz

    def _get_z(self, n):
        z = _sample_exponential(
            rng=self.rng,
            r0=self.r0 * self.rel_height,  # 0.1 is ~typical
            n=n,
        )

        # ~half go downward
        mask = self.rng.uniform(size=n) > 0.5
        np.negative(z, where=mask, out=z)
        return z

    def _sample_r(self, n):
        return _sample_exponential(
            rng=self.rng,
            r0=self.r0,
            n=n,
        )


def _sample_exponential(rng, r0, n):
    return rng.exponential(
        scale=r0 * 2.375,
        size=n,
    )


def _rotate_coords(x, y, theta):
    rot = theta / radians
    sinrot = np.sin(rot)
    cosrot = np.cos(rot)
    xp = x * cosrot - y * sinrot
    yp = x * sinrot + y * cosrot
    return xp, yp


def _check_angle(angle, name):
    if not isinstance(angle, Angle):
        tangle = str(type(angle))
        raise GalSimValueError(
            f'{name} should be a galsim.Angle, got {tangle}', angle
        )


def _check_sequence(data, name, nel):
    if data is None:
        return

    try:
        ndata = len(data)
        if ndata != nel:
            raise GalSimValueError(
                f'expected {nel} element sequence for {name}, got {ndata}',
                data
            )
    except TypeError:
        raise GalSimValueError(
            f'expected sequence for {name}, got {type(data)}',
            data
        )
