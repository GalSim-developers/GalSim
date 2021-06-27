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
import math
import cmath

from . import _galsim
from .gsobject import GSObject
from .gsparams import GSParams
from .utilities import lazy_property, WeakMethod
from .position import PositionD, _PositionD
from .errors import GalSimError

def Transform(obj, jac=None, offset=(0.,0.), flux_ratio=1., gsparams=None,
              propagate_gsparams=True):
    """A function for transforming either a `GSObject` or `ChromaticObject`.

    This function will inspect its input argument to decide if a `Transformation` object or a
    `ChromaticTransformation` object is required to represent the resulting transformed object.

    Note: the name of the flux_ratio parameter is technically wrong here if the jacobian has a
    non-unit determinant, since that would also scale the flux.  The flux_ratio parameter actually
    only refers to an overall amplitude ratio for the surface brightness profile.  The total
    flux scaling is actually ``|det(jac)| * flux_ratio``.

    Parameters:
        obj:                The object to be transformed.
        jac:                A list or tuple ( dudx, dudy, dvdx, dvdy ) describing the Jacobian
                            of the transformation.  Use None to indicate that the Jacobian is the
                            2x2 unit matrix.  [default: None]
        offset:             A galsim.PositionD or tuple giving the offset by which to shift the
                            profile. [default: (0.,0.)]
        flux_ratio:         A factor by which to multiply the surface brightness of the object.
                            (Technically, not necessarily the flux.  See above.) [default: 1]
        gsparams:           An optional `GSParams` argument. [default: None]
        propagate_gsparams: Whether to propagate gsparams to the transformed object.  This
                            is normally a good idea, but there may be use cases where one
                            would not want to do this. [default: True]

    Returns:
        a `Transformation` or `ChromaticTransformation` instance as appropriate.
    """
    from .sum import Sum
    from .convolve import Convolution
    from .chromatic import ChromaticObject
    from .chromatic import ChromaticSum, ChromaticConvolution, ChromaticTransformation
    if not (isinstance(obj, GSObject) or isinstance(obj, ChromaticObject)):
        raise TypeError("Argument to Transform must be either a GSObject or a ChromaticObject.")

    elif (hasattr(jac,'__call__') or hasattr(offset,'__call__') or
          hasattr(flux_ratio,'__call__') or isinstance(obj, ChromaticObject)):

        # Sometimes for Chromatic compound types, it is more efficient to apply the
        # transformation to the components rather than the whole.  In particular, this can
        # help preserve separability in many cases.

        # Don't transform ChromaticSum object, better to just transform the arguments.
        if isinstance(obj, ChromaticSum) or isinstance(obj, Sum):
            new_obj = ChromaticSum(
                [ Transform(o,jac,offset,flux_ratio,gsparams,propagate_gsparams)
                  for o in obj.obj_list ])
            if hasattr(obj, 'covspec'):
                if jac is None:
                    new_obj.covspec = obj.covspec * flux_ratio**2
                else:
                    dudx, dudy, dvdx, dvdy = np.asarray(jac, dtype=float).flatten()
                    new_obj.covspec = obj.covspec.transform(dudx, dudy, dvdx, dvdy) * flux_ratio**2
            return new_obj

        # If we are just flux scaling, then a Convolution can do that to the first element.
        # NB. Even better, if the flux scaling is chromatic, would be to find a component
        # that is already non-separable.  But we don't bother trying to do that currently.
        elif (isinstance(obj, ChromaticConvolution or isinstance(obj, Convolution))
              and jac is None and offset == (0.,0.)):
            first = Transform(obj.obj_list[0], flux_ratio=flux_ratio, gsparams=gsparams,
                              propagate_gsparams=propagate_gsparams)
            return ChromaticConvolution( [first] + [o for o in obj.obj_list[1:]] )

        else:
            return ChromaticTransformation(obj, jac, offset, flux_ratio, gsparams=gsparams,
                                           propagate_gsparams=propagate_gsparams)
    else:
        return Transformation(obj, jac, offset, flux_ratio, gsparams, propagate_gsparams)


class Transformation(GSObject):
    """A class for modeling an affine transformation of a `GSObject` instance.

    Typically, you do not need to construct a Transformation object explicitly.  This is the type
    returned by the various transformation methods of `GSObject` such as `GSObject.shear`,
    `GSObject.rotate`, `GSObject.shift`, `GSObject.transform`, etc.

    All the various transformations can be described as a combination of a jacobian matrix
    (i.e. `GSObject.transform`) and a translation (`GSObject.shift`), which are described by
    (dudx,dudy,dvdx,dvdy) and (dx,dy) respectively.

    .. note::
        The name of the flux_ratio parameter is technically wrong here if the jacobian has a
        non-unit determinant, since that would also scale the flux.  The flux_ratio parameter
        actually only refers to an overall amplitude ratio for the surface brightness profile.
        The total flux scaling is actually ``|det(jac)| * flux_ratio``.

    Parameters:
        obj:                The object to be transformed.
        jac:                A list, tuple or numpy array ( dudx, dudy, dvdx, dvdy ) describing
                            the Jacobian of the transformation.  Use None to indicate that the
                            Jacobian is the 2x2 unit matrix.  [default: None]
        offset:             A galsim.PositionD or tuple giving the offset by which to shift the
                            profile. [default: (0.,0.)]
        flux_ratio:         A factor by which to multiply the surface brightness of the object.
                            (Technically, not necessarily the flux.  See above.) [default: 1]
        gsparams:           An optional `GSParams` argument. [default: None]
        propagate_gsparams: Whether to propagate gsparams to the transformed object.  This
                            is normally a good idea, but there may be use cases where one
                            would not want to do this. [default: True]

    Attributes:
        original:       The original object that is being transformed.
        jac:            The jacobian of the transformation matrix.
        offset:         The offset being applied.
        flux_ratio:     The amount by which the overall surface brightness amplitude is multiplied.
        gsparams:       The usual gsparams attribute that all `GSObject` classes have.

    Note: if ``gsparams`` is unspecified (or None), then the Transformation instance inherits the
    `GSParams` from obj.  Also, note that parameters related to the Fourier-space calculations must
    be set when initializing obj, NOT when creating the Transform (at which point the accuracy and
    threshold parameters will simply be ignored).
    """
    unit_jac = np.array([[1,0],[0,1]], dtype=float)

    def __init__(self, obj, jac=None, offset=(0.,0.), flux_ratio=1.,
                 gsparams=None, propagate_gsparams=True):
        if jac is None:
            self._jac = None
        else:
            self._jac = np.asarray(jac, dtype=float).reshape(2,2)
        offset = PositionD(offset)
        self._dx = offset.x
        self._dy = offset.y
        self._flux_ratio = float(flux_ratio)
        self._gsparams = GSParams.check(gsparams, obj.gsparams)
        self._propagate_gsparams = propagate_gsparams
        if self._propagate_gsparams:
            obj = obj.withGSParams(self._gsparams)

        if isinstance(obj, Transformation):
            # Combine the two affine transformations into one.
            if obj._has_offset:
                if jac is None:
                    dx1, dy1 = obj._dx, obj._dy
                else:
                    dx1, dy1 = self._fwd_normal(obj._dx, obj._dy)
                self._dx += dx1
                self._dy += dy1
            if jac is None:
                self._jac = obj._jac
            else:
                self._jac = self._jac if obj._jac is None else self._jac.dot(obj.jac)
            self._flux_ratio *= obj._flux_ratio
            self._original = obj._original
        else:
            self._original = obj
        self._has_offset = (self._dx != 0. or self._dy != 0.)
        if self._det==0.:
            raise GalSimError("Attempt to Transform with degenerate matrix");

    @property
    def original(self):
        """The original object being transformed.
        """
        return self._original

    @property
    def jac(self):
        """The Jacobian of the transforamtion.
        """
        return self.unit_jac if self._jac is None else self._jac

    @property
    def offset(self):
        """The offset of the transformation.
        """
        return _PositionD(self._dx, self._dy)

    @property
    def flux_ratio(self):
        """The flux ratio of the transformation.
        """
        return self._flux_ratio

    @lazy_property
    def _flux(self):
        return self._flux_scaling * self._original.flux

    @lazy_property
    def _sbp(self):
        _jac = 0 if self._jac is None else self._jac.__array_interface__['data'][0]
        return _galsim.SBTransform(self._original._sbp, _jac,
                                   self._dx, self._dy, self._flux_ratio, self.gsparams._gsp)

    @lazy_property
    def _noise(self):
        from .correlatednoise import BaseCorrelatedNoise
        if self.original.noise is None:
            return None
        else:
            return BaseCorrelatedNoise(
                    self.original.noise.rng,
                    _Transform(self.original.noise._profile, self._jac,
                               flux_ratio=self.flux_ratio**2),
                    self.original.noise.wcs)

    def withGSParams(self, gsparams=None, **kwargs):
        """Create a version of the current object with the given gsparams

        .. note::

            Unless you set ``propagate_gsparams=False``, this method will also update the gsparams
            of the wrapped component object.
        """
        if gsparams == self.gsparams: return self
        from copy import copy
        ret = copy(self)
        ret._gsparams = GSParams.check(gsparams, self.gsparams, **kwargs)
        if self._propagate_gsparams:
            ret._original = self._original.withGSParams(ret._gsparams)
        return ret

    def __eq__(self, other):
        return (self is other or
                (isinstance(other, Transformation) and
                 self.original == other.original and
                 np.array_equal(self.jac, other.jac) and
                 np.array_equal(self.offset, other.offset) and
                 self.flux_ratio == other.flux_ratio and
                 self.gsparams == other.gsparams and
                 self._propagate_gsparams == other._propagate_gsparams))

    def __hash__(self):
        return hash(("galsim.Transformation", self.original, tuple(self.jac.ravel()),
                     self._dx, self._dy, self.flux_ratio, self.gsparams,
                     self._propagate_gsparams))

    def __repr__(self):
        return ('galsim.Transformation(%r, jac=%r, offset=%r, flux_ratio=%r, gsparams=%r, '
                'propagate_gsparams=%r)')%(
            self.original, None if self._jac is None else self._jac.tolist(),
            self.offset, self.flux_ratio, self.gsparams, self._propagate_gsparams)

    @classmethod
    def _str_from_jac(cls, jac):
        from .wcs import JacobianWCS
        dudx, dudy, dvdx, dvdy = jac.ravel()
        # Figure out the shear/rotate/dilate calls that are equivalent.
        jac = JacobianWCS(dudx,dudy,dvdx,dvdy)
        scale, shear, theta, flip = jac.getDecomposition()
        s = None
        if flip:
            s = 0  # Special value indicating to just use transform.
        if abs(theta.rad) > 1.e-12:
            if s is None:
                s = '.rotate(%s)'%theta
            else:
                s = 0
        if shear.g > 1.e-12:
            if s is None:
                s = '.shear(%s)'%shear
            else:
                s = 0
        if abs(scale-1.0) > 1.e-12:
            if s is None:
                s = '.expand(%s)'%scale
            else:
                s = 0
        if s == 0:
            # If flip or there are two components, then revert to transform as simpler.
            s = '.transform(%s,%s,%s,%s)'%(dudx,dudy,dvdx,dvdy)
        if s is None:
            # If nothing is large enough to show up above, give full detail of transform
            s = '.transform(%r,%r,%r,%r)'%(dudx,dudy,dvdx,dvdy)
        return s

    def __str__(self):
        s = str(self.original)
        if self._jac is not None:
            s += self._str_from_jac(self._jac)
        if self._dx != 0 or self._dy != 0:
            s += '.shift(%s,%s)'%(self._dx,self._dy)
        if self.flux_ratio != 1.:
            s += ' * %s'%self.flux_ratio
        return s

    def _prepareDraw(self):
        self._original._prepareDraw()

    # Some lazy properties to calculate things as needed.
    @lazy_property
    def _det(self):
        if self._jac is None:
            return 1
        if self._jac[0,1] == 0. and self._jac[1,0] == 0.:
            if self._jac[0,0] == 1. and self._jac[1,1] == 1.:     # jac is (1,0,0,1)
                return 1.
            else:                                               # jac is (a,0,0,b)
                return self._jac[0,0] * self._jac[1,1]
        else:                                                   # Fully general case
            return self._jac[0,0] * self._jac[1,1] - self._jac[0,1] * self._jac[1,0]

    @lazy_property
    def _invdet(self):
        return 1. if self._jac is None else 1./self._det

    @lazy_property
    def _invjac(self):
        invjac = np.array([[self._jac[1,1], -self._jac[0,1]], [-self._jac[1,0], self._jac[0,0]]])
        invjac *= self._invdet
        return invjac

    # To avoid confusion with the flux vs amplitude scaling, we use these names below, rather
    # than flux_ratio, which is really an amplitude scaling.
    @property
    def _amp_scaling(self):
        return self._flux_ratio

    @lazy_property
    def _flux_scaling(self):
        return abs(self._det) * self._flux_ratio

    # Some helper attributes to make fwd and inv transformation quicker
    @lazy_property
    def _fwd(self):
        if self._jac is None:
            return WeakMethod(self._ident)
        elif self._jac[0,1] == 0. and self._jac[1,0] == 0.:
            return WeakMethod(self._fwd_diag)
        else:
            return WeakMethod(self._fwd_normal)

    @lazy_property
    def _fwdT(self):
        if self._jac is None:
            return WeakMethod(self._ident)
        elif self._jac[0,1] == 0. and self._jac[1,0] == 0.:
            return WeakMethod(self._fwd_diag)
        else:
            return WeakMethod(self._fwdT_normal)

    @lazy_property
    def _inv(self):
        if self._jac is None:
            return WeakMethod(self._ident)
        elif self._jac[0,1] == 0. and self._jac[1,0] == 0.:
            return WeakMethod(self._inv_diag)
        else:
            return WeakMethod(self._inv_normal)

    @lazy_property
    def _kfactor(self):
        if self._dx == self._dy == 0.:
            return WeakMethod(self._kf_nophase)
        else:
            return WeakMethod(self._kf_phase)

    def _ident(self, x, y):
        return x, y

    def _fwd_diag(self, x, y):
        x *= self._jac[0,0]
        y *= self._jac[1,1]
        return x, y

    def _fwd_normal(self, x, y):
        #return self._jac[0,0] * x + self._jac[0,1] * y, self._jac[1,0] * x + self._jac[1,1] * y
        # Do this as much in place as possible
        temp = self._jac[0,1] * y
        y *= self._jac[1,1]
        y += self._jac[1,0] * x
        x *= self._jac[0,0]
        x += temp
        return x, y

    def _fwdT_normal(self, x, y):
        #return self._jac[0,0] * x + self._jac[1,0] * y, self._jac[0,1] * x + self._jac[1,1] * y
        temp = self._jac[1,0] * y
        y *= self._jac[1,1]
        y += self._jac[0,1] * x
        x *= self._jac[0,0]
        x += temp
        return x, y

    def _inv_diag(self, x, y):
        x /= self._jac[0,0]
        y /= self._jac[1,1]
        return x, y

    def _inv_normal(self, x, y):
        #return (self._invdet * (self._jac[1,1] * x - self._jac[0,1] * y),
        #        self._invdet * (-self._jac[1,0] * x + self._jac[0,0] * y))
        temp = self._invjac[0,1] * y
        y *= self._invjac[1,1]
        y += self._invjac[1,0] * x
        x *= self._invjac[0,0]
        x += temp
        return x, y

    def _kf_nophase(self, kx, ky):
        return self._flux_scaling

    def _kf_phase(self, kx, ky):
        kx *= -1j * self._dx
        ky *= -1j * self._dy
        kx += ky
        return self._flux_scaling * np.exp(kx)

    def __getstate__(self):
        d = self.__dict__.copy()
        d.pop('_sbp',None)
        d.pop('_fwd',None)
        d.pop('_fwdT',None)
        d.pop('_inv',None)
        d.pop('_kfactor',None)
        return d

    def __setstate__(self, d):
        self.__dict__ = d

    def _major_minor(self):
        if not hasattr(self, "_major"):
            if self._jac is None:
                self._major = self._minor = 1.
            else:
                h1 = math.hypot(self._jac[0,0] + self._jac[1,1], self._jac[0,1] - self._jac[1,0])
                h2 = math.hypot(self._jac[0,0] - self._jac[1,1], self._jac[0,1] + self._jac[1,0])
                self._major = 0.5 * abs(h1+h2)
                self._minor = 0.5 * abs(h1-h2)

    @lazy_property
    def _maxk(self):
        self._major_minor()
        return self._original.maxk / self._minor

    @lazy_property
    def _stepk(self):
        self._major_minor()
        stepk = self._original.stepk / self._major
        # If we have a shift, we need to further modify stepk
        #     stepk = Pi/R
        #     R <- R + |shift|
        #     stepk <- Pi/(Pi/stepk + |shift|)
        if self._dx !=  0. or self._dy != 0.:
            dr = math.hypot(self._dx, self._dy)
            stepk = math.pi / (math.pi/stepk + dr)
        return stepk

    @property
    def _has_hard_edges(self):
        return self._original.has_hard_edges

    @property
    def _is_axisymmetric(self):
        return bool(self._original.is_axisymmetric and
                    (self._jac is None or (self._jac[0,0] == self._jac[1,1] and
                                           self._jac[0,1] == -self._jac[1,0])) and
                    self._dx == self._dy == 0.)

    @property
    def _is_analytic_x(self):
        return self._original.is_analytic_x

    @property
    def _is_analytic_k(self):
        return self._original.is_analytic_k

    @property
    def _centroid(self):
        cen = self._original.centroid
        cenx, ceny = self._fwd(cen.x, cen.y)
        cenx += self._dx
        ceny += self._dy
        return _PositionD(cenx,ceny)

    @property
    def _positive_flux(self):
        return self._flux_scaling * self._original.positive_flux

    @property
    def _negative_flux(self):
        return self._flux_scaling * self._original.negative_flux

    @lazy_property
    def _flux_per_photon(self):
        return self._calculate_flux_per_photon()

    @property
    def _max_sb(self):
        return self._amp_scaling * self._original.max_sb

    def _xValue(self, pos):
        x = pos.x - self._dx
        y = pos.y - self._dy
        inv_pos = _PositionD(*self._inv(x, y))
        return self._original._xValue(inv_pos) * self._amp_scaling

    def _kValue(self, kpos):
        fwdT_kpos = _PositionD(*self._fwdT(kpos.x, kpos.y))
        return self._original._kValue(fwdT_kpos) * self._kfactor(kpos.x, kpos.y)

    def _drawReal(self, image, jac=None, offset=(0.,0.), flux_scaling=1.):
        dx, dy = offset
        if self._has_offset:
            if jac is not None:
                x1 = jac[0,0] * self._dx + jac[0,1] * self._dy
                y1 = jac[1,0] * self._dx + jac[1,1] * self._dy
            else:
                x1, y1 = self._dx, self._dy
            dx += x1
            dy += y1
        flux_scaling *= self._flux_scaling
        jac = self._jac if jac is None else jac if self._jac is None else jac.dot(self._jac)
        self._original._drawReal(image, jac, (dx, dy), flux_scaling)

    def _shoot(self, photons, rng):
        self._original._shoot(photons, rng)
        photons.x, photons.y = self._fwd(photons.x, photons.y)
        photons.x += self._dx
        photons.y += self._dy
        photons.scaleFlux(self._flux_scaling)

    def _drawKImage(self, image, jac=None):
        jac1 = self._jac if jac is None else jac if self._jac is None else jac.dot(self._jac)
        self._original._drawKImage(image, jac1)

        if self._has_offset:
            _jac = 0 if jac is None else jac.__array_interface__['data'][0]
            _galsim.ApplyKImagePhases(image._image, image.scale, _jac,
                                      self._dx, self._dy, self._flux_scaling)
        elif abs(self._flux_scaling-1.) > self._gsparams.kvalue_accuracy:
            image *= self._flux_scaling


def _Transform(obj, jac=None, offset=(0.,0.), flux_ratio=1.):
    """Approximately equivalent to Transform, but without some of the sanity checks (such as
    checking for chromatic options) or setting a new gsparams.

    For a `ChromaticObject`, you must use the regular `Transform`.

    Parameters:
        obj:                The object to be transformed.
        jac:                A 2x2 numpy array describing the Jacobian of the transformation.
                            Use None to indicate that the Jacobian is the 2x2 unit matrix.
                            [default: None]
        offset:             The offset to apply. [default (0.,0.)]
        flux_ratio:         A factor by which to multiply the surface brightness of the object.
                            [default: 1.]
    """
    ret = Transformation.__new__(Transformation)
    ret._gsparams = obj.gsparams
    ret._propagate_gsparams = True
    ret._jac = jac
    ret._dx, ret._dy = offset
    if isinstance(obj, Transformation):
        if obj._has_offset:
            if jac is None:
                dx1, dy1 = obj._dx, obj._dy
            else:
                dx1, dy1 = ret._fwd_normal(obj._dx, obj._dy)
            ret._dx += dx1
            ret._dy += dy1
        if jac is None:
            ret._jac = obj._jac
        else:
            ret._jac = ret._jac if obj._jac is None else ret._jac.dot(obj.jac)
        ret._flux_ratio = flux_ratio * obj._flux_ratio
        ret._original = obj._original
    else:
        ret._flux_ratio = flux_ratio
        ret._original = obj
    ret._has_offset = (ret._dx != 0. or ret._dy != 0.)
    return ret
