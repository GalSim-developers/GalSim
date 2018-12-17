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
"""@file transform.py
A class that handles affine tranformations of a profile including a possible flux scaling.
"""

import numpy as np
import math
import cmath

from . import _galsim
from .gsobject import GSObject
from .gsparams import GSParams
from .utilities import lazy_property, doc_inherit, WeakMethod
from .position import PositionD
from .errors import GalSimError, convert_cpp_errors

def Transform(obj, jac=(1.,0.,0.,1.), offset=PositionD(0.,0.), flux_ratio=1., gsparams=None,
              propagate_gsparams=True):
    """A function for transforming either a GSObject or ChromaticObject.

    This function will inspect its input argument to decide if a Transformation object or a
    ChromaticTransformation object is required to represent the resulting transformed object.

    Note: the name of the flux_ratio parameter is technically wrong here if the jacobian has a
    non-unit determinant, since that would also scale the flux.  The flux_ratio parameter actually
    only refers to an overall amplitude ratio for the surface brightness profile.  The total
    flux scaling is actually |det(jac)| * flux_ratio.

    @param obj              The object to be transformed.
    @param jac              A list or tuple ( dudx, dudy, dvdx, dvdy ) describing the Jacobian
                            of the transformation. [default: (1,0,0,1)]
    @param offset           A galsim.PositionD giving the offset by which to shift the profile.
    @param flux_ratio       A factor by which to multiply the surface brightness of the object.
                            (Technically, not necessarily the flux.  See above.) [default: 1]
    @param gsparams         An optional GSParams argument.  See the docstring for GSParams for
                            details. [default: None]
    @param propagate_gsparams   Whether to propagate gsparams to the transformed object.  This
                                is normally a good idea, but there may be use cases where one
                                would not want to do this. [default: True]

    @returns a Transformation or ChromaticTransformation instance as appropriate.
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
                dudx, dudy, dvdx, dvdy = np.asarray(jac, dtype=float).flatten()
                new_obj.covspec = obj.covspec.transform(dudx, dudy, dvdx, dvdy)*flux_ratio**2
            return new_obj

        # If we are just flux scaling, then a Convolution can do that to the first element.
        # NB. Even better, if the flux scaling is chromatic, would be to find a component
        # that is already non-separable.  But we don't bother trying to do that currently.
        elif (isinstance(obj, ChromaticConvolution or isinstance(obj, Convolution))
              and np.array_equal(np.asarray(jac).ravel(),(1,0,0,1))
              and offset == PositionD(0.,0.)):
            first = Transform(obj.obj_list[0], flux_ratio=flux_ratio, gsparams=gsparams,
                              propagate_gsparams=propagate_gsparams)
            return ChromaticConvolution( [first] + [o for o in obj.obj_list[1:]] )

        else:
            return ChromaticTransformation(obj, jac, offset, flux_ratio, gsparams,
                                           propagate_gsparams)
    else:
        return Transformation(obj, jac, offset, flux_ratio, gsparams, propagate_gsparams)


class Transformation(GSObject):
    """A class for modeling an affine transformation of a GSObject instance.

    Initialization
    --------------

    Typically, you do not need to construct a Transformation object explicitly.  This is the type
    returned by the various transformation methods of GSObject such as shear(), rotate(),
    shift(), transform(), etc.  All the various transformations can be described as a combination
    of transform() and shift(), which are described by (dudx,dudy,dvdx,dvdy) and (dx,dy)
    respectively.

    Note: the name of the flux_ratio parameter is technically wrong here if the jacobian has a
    non-unit determinant, since that would also scale the flux.  The flux_ratio parameter actually
    only refers to an overall amplitude ratio for the surface brightness profile.  The total
    flux scaling is actually |det(jac)| * flux_ratio.

    @param obj              The object to be transformed.
    @param jac              A list, tuple or numpy array ( dudx, dudy, dvdx, dvdy ) describing the
                            Jacobian of the transformation. [default: (1,0,0,1)]
    @param offset           A galsim.PositionD giving the offset by which to shift the profile.
    @param flux_ratio       A factor by which to multiply the surface brightness of the object.
                            (Technically, not necessarily the flux.  See above.) [default: 1]
    @param gsparams         An optional GSParams argument.  See the docstring for GSParams for
                            details. [default: None]
    @param propagate_gsparams   Whether to propagate gsparams to the transformed object.  This
                                is normally a good idea, but there may be use cases where one
                                would not want to do this. [default: True]

    Attributes
    ----------

    original        The original object that is being transformed.
    jac             The jacobian of the transformation matrix.
    offset          The offset being applied.
    flux_ratio      The amount by which the overall surface brightness amplitude is multiplied.
    gsparams        The usual gsparams attribute that all GSObjects have.

    Note: if `gsparams` is unspecified (or None), then the Transformation instance inherits the
    GSParams from obj.  Also, note that parameters related to the Fourier-space calculations must
    be set when initializing obj, NOT when creating the Transform (at which point the accuracy and
    threshold parameters will simply be ignored).
    """
    def __init__(self, obj, jac=(1.,0.,0.,1.), offset=PositionD(0.,0.), flux_ratio=1.,
                 gsparams=None, propagate_gsparams=True):
        self._jac = np.asarray(jac, dtype=float).reshape(2,2)
        self._offset = PositionD(offset)
        self._flux_ratio = float(flux_ratio)
        self._gsparams = GSParams.check(gsparams, obj.gsparams)
        self._propagate_gsparams = propagate_gsparams
        if self._propagate_gsparams:
            obj = obj.withGSParams(self._gsparams)

        if isinstance(obj, Transformation):
            # Combine the two affine transformations into one.
            dx, dy = self._fwd_normal(obj.offset.x, obj.offset.y)
            self._offset.x += dx
            self._offset.y += dy
            self._jac = self._jac.dot(obj.jac)
            self._flux_ratio *= obj._flux_ratio
            self._original = obj.original
        else:
            self._original = obj

    @property
    def original(self): return self._original
    @property
    def jac(self): return self._jac
    @property
    def offset(self): return self._offset
    @property
    def flux_ratio(self): return self._flux_ratio

    @lazy_property
    def _flux(self):
        return self._flux_scaling * self._original.flux

    @lazy_property
    def _sbp(self):
        dudx, dudy, dvdx, dvdy = self._jac.ravel()
        with convert_cpp_errors():
            return _galsim.SBTransform(self._original._sbp, dudx, dudy, dvdx, dvdy,
                                       self._offset._p, self._flux_ratio, self.gsparams._gsp)

    @lazy_property
    def _noise(self):
        from .correlatednoise import _BaseCorrelatedNoise
        if self.original.noise is None:
            return None
        else:
            dudx, dudy, dvdx, dvdy = self._jac.ravel()
            return _BaseCorrelatedNoise(
                    self.original.noise.rng,
                    _Transform(self.original.noise._profile,
                               (dudx, dudy, dvdx, dvdy),
                               flux_ratio=self.flux_ratio**2),
                    self.original.noise.wcs)

    @doc_inherit
    def withGSParams(self, gsparams):
        if gsparams == self.gsparams: return self
        from copy import copy
        ret = copy(self)
        ret._gsparams = GSParams.check(gsparams)
        if self._propagate_gsparams:
            ret._original = self.original.withGSParams(gsparams)
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
        return hash(("galsim.Transformation", self.original, tuple(self._jac.ravel()),
                     self.offset.x, self.offset.y, self.flux_ratio, self.gsparams,
                     self._propagate_gsparams))

    def __repr__(self):
        return ('galsim.Transformation(%r, jac=%r, offset=%r, flux_ratio=%r, gsparams=%r, '
                'propagate_gsparams=%r)')%(
            self.original, self._jac.tolist(), self.offset, self.flux_ratio, self.gsparams,
            self._propagate_gsparams)

    @classmethod
    def _str_from_jac(cls, jac):
        from .wcs import JacobianWCS
        dudx, dudy, dvdx, dvdy = jac.ravel()
        if dudx != 1 or dudy != 0 or dvdx != 0 or dvdy != 1:
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
        else:
            return ''

    def __str__(self):
        s = str(self.original)
        s += self._str_from_jac(self._jac)
        if self.offset.x != 0 or self.offset.y != 0:
            s += '.shift(%s,%s)'%(self.offset.x,self.offset.y)
        if self.flux_ratio != 1.:
            s += ' * %s'%self.flux_ratio
        return s

    def _prepareDraw(self):
        self._original._prepareDraw()

    # Some lazy properties to calculate things as needed.
    @lazy_property
    def _det(self):
        if self._jac[0,1] == 0. and self._jac[1,0] == 0.:
            if self._jac[0,0] == 1. and self._jac[1,1] == 1.:     # jac is (1,0,0,1)
                return 1.
            else:                                               # jac is (a,0,0,b)
                return self._jac[0,0] * self._jac[1,1]
        else:                                                   # Fully general case
            return self._jac[0,0] * self._jac[1,1] - self._jac[0,1] * self._jac[1,0]

    @lazy_property
    def _invdet(self):
        return 1./self._det

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
        if self._jac[0,1] == 0. and self._jac[1,0] == 0.:
            if self._jac[0,0] == 1. and self._jac[1,1] == 1.:
                return WeakMethod(self._ident)
            else:
                return WeakMethod(self._fwd_diag)
        else:
            return WeakMethod(self._fwd_normal)

    @lazy_property
    def _fwdT(self):
        if self._jac[0,1] == 0. and self._jac[1,0] == 0.:
            if self._jac[0,0] == 1. and self._jac[1,1] == 1.:
                return WeakMethod(self._ident)
            else:
                return WeakMethod(self._fwd_diag)
        else:
            return WeakMethod(self._fwdT_normal)

    @lazy_property
    def _inv(self):
        if self._jac[0,1] == 0. and self._jac[1,0] == 0.:
            if self._jac[0,0] == 1. and self._jac[1,1] == 1.:
                return WeakMethod(self._ident)
            else:
                return WeakMethod(self._inv_diag)
        else:
            return WeakMethod(self._inv_normal)

    @lazy_property
    def _kfactor(self):
        if self._offset == PositionD(0,0):
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
        kx *= -1j * self._offset.x
        ky *= -1j * self._offset.y
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
        if self._offset != PositionD(0.,0.):
            dr = math.hypot(self._offset.x, self._offset.y)
            stepk = math.pi / (math.pi/stepk + dr)
        return stepk

    @property
    def _has_hard_edges(self):
        return self._original.has_hard_edges

    @property
    def _is_axisymmetric(self):
        return bool(self._original.is_axisymmetric and
                    self._jac[0,0] == self._jac[1,1] and
                    self._jac[0,1] == -self._jac[1,0] and
                    self._offset == PositionD(0.,0.))

    @property
    def _is_analytic_x(self):
        return self._original.is_analytic_x

    @property
    def _is_analytic_k(self):
        return self._original.is_analytic_k

    @property
    def _centroid(self):
        cen = self._original.centroid
        cen = PositionD(self._fwd(cen.x, cen.y))
        cen += self._offset
        return cen

    @property
    def _positive_flux(self):
        return self._flux_scaling * self._original.positive_flux

    @property
    def _negative_flux(self):
        return self._flux_scaling * self._original.negative_flux

    @property
    def _max_sb(self):
        return self._amp_scaling * self._original.max_sb

    @doc_inherit
    def _xValue(self, pos):
        pos -= self._offset
        inv_pos = PositionD(self._inv(pos.x, pos.y))
        return self._original._xValue(inv_pos) * self._amp_scaling

    @doc_inherit
    def _kValue(self, kpos):
        fwdT_kpos = PositionD(self._fwdT(kpos.x, kpos.y))
        return self._original._kValue(fwdT_kpos) * self._kfactor(kpos.x, kpos.y)

    @doc_inherit
    def _drawReal(self, image):
        if self.offset == PositionD(0.,0.) and np.array_equal(self.jac.ravel(), [1,0,0,1]):
            self._original._drawReal(image)
            if self._flux_ratio != 1.:
                image *= self._flux_ratio
        else:
            # TODO: Refactor the C++ draw function to allow this to be implemented in python
            self._sbp.draw(image._image, image.scale)

    @doc_inherit
    def _shoot(self, photons, rng):
        self._original._shoot(photons, rng)
        photons.x, photons.y = self._fwd(photons.x, photons.y)
        photons.x += self.offset.x
        photons.y += self.offset.y
        photons.scaleFlux(self._flux_scaling)

    @doc_inherit
    def _drawKImage(self, image):
        self._sbp.drawK(image._image, image.scale)


def _Transform(obj, jac=(1.,0.,0.,1.), offset=PositionD(0.,0.),
               flux_ratio=1., gsparams=None):
    """Approximately equivalent to Transform, but without some of the sanity checks (such as
    checking for Chromatic options).  For ChromaticObjects, you must use the regular Transform.
    """
    return Transformation(obj, jac, offset, flux_ratio, gsparams)
