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
"""@file transform.py
A class that handles affine tranformations of a profile including a possible flux scaling.
"""

import galsim
import numpy as np
from . import _galsim

def Transform(obj, jac=(1.,0.,0.,1.), offset=galsim.PositionD(0.,0.), flux_ratio=1.,
              gsparams=None):
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

    @returns a Transformation or ChromaticTransformation instance as appropriate.
    """
    if not (isinstance(obj, galsim.GSObject) or isinstance(obj, galsim.ChromaticObject)):
        raise TypeError("Argument to Transform must be either a GSObject or a ChromaticObject.")

    elif (hasattr(jac,'__call__') or hasattr(offset,'__call__') or
          hasattr(flux_ratio,'__call__') or isinstance(obj, galsim.ChromaticObject)):

        # Sometimes for Chromatic compound types, it is more efficient to apply the
        # transformation to the components rather than the whole.  In particular, this can
        # help preserve separability in many cases.

        # Don't transform ChromaticSum object, better to just transform the arguments.
        if isinstance(obj, galsim.ChromaticSum) or isinstance(obj, galsim.Sum):
            new_obj = galsim.ChromaticSum(
                [ Transform(o,jac,offset,flux_ratio,gsparams) for o in obj.objlist ])
            if hasattr(obj, 'covspec'):
                dudx, dudy, dvdx, dvdy = np.asarray(jac, dtype=float).flatten()
                new_obj.covspec = obj.covspec.transform(dudx, dudy, dvdx, dvdy)*flux_ratio**2
            return new_obj

        # If we are just flux scaling, then a Convolution can do that to the first element.
        # NB. Even better, if the flux scaling is chromatic, would be to find a component
        # that is already non-separable.  But we don't bother trying to do that currently.
        elif (isinstance(obj, galsim.ChromaticConvolution or isinstance(obj, galsim.Convolution))
              and np.array_equal(np.asarray(jac).ravel(),(1,0,0,1))
              and offset == galsim.PositionD(0.,0.)):
            first = Transform(obj.objlist[0],flux_ratio=flux_ratio,gsparams=gsparams)
            return galsim.ChromaticConvolution( [first] + [o for o in obj.objlist[1:]] )

        else:
            return galsim.ChromaticTransformation(obj, jac, offset, flux_ratio, gsparams)
    else:
        return Transformation(obj, jac, offset, flux_ratio, gsparams)


class Transformation(galsim.GSObject):
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
    @param jac              A list or tuple ( dudx, dudy, dvdx, dvdy ) describing the Jacobian
                            of the transformation. [default: (1,0,0,1)]
    @param offset           A galsim.PositionD giving the offset by which to shift the profile.
    @param flux_ratio       A factor by which to multiply the surface brightness of the object.
                            (Technically, not necessarily the flux.  See above.) [default: 1]
    @param gsparams         An optional GSParams argument.  See the docstring for GSParams for
                            details. [default: None]

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
    def __init__(self, obj, jac=(1.,0.,0.,1.), offset=galsim.PositionD(0.,0.), flux_ratio=1.,
                 gsparams=None):
        dudx, dudy, dvdx, dvdy = np.asarray(jac, dtype=float).ravel()
        if hasattr(obj, 'original'):
            self._original = obj.original
        else:
            self._original = obj
        sbt = _galsim.SBTransform(obj.SBProfile, dudx, dudy, dvdx, dvdy, offset, flux_ratio,
                                  gsparams)
        galsim.GSObject.__init__(self, sbt)

        self._jac = np.asarray(sbt.getJac())
        self._offset = sbt.getOffset()
        self._flux_ratio = sbt.getFluxScaling()
        self._gsparams = gsparams

    def getJac(self):
        """Return the Jacobian of the transformation.
        """
        return self._jac

    def getOffset(self):
        """Return the offset of the transformation.
        """
        return self._offset

    def getFluxRatio(self):
        """Return the flux ratio of the transformation.
        """
        return self._flux_ratio

    @galsim.utilities.lazy_property
    def noise(self):
        if self.original.noise is None:
            return None
        else:
            jac = self.SBProfile.getJac()
            flux_ratio = self.SBProfile.getFluxScaling()
            return galsim.correlatednoise._BaseCorrelatedNoise(
                    self.original.noise.rng,
                    galsim._Transform(self.original.noise._profile,
                                      jac[0], jac[1], jac[2], jac[3],
                                      flux_ratio=flux_ratio**2),
                    self.original.noise.wcs)

    @property
    def original(self): return self._original
    @property
    def jac(self): return self._jac.reshape(2,2)
    @property
    def offset(self): return self._offset
    @property
    def flux_ratio(self): return self._flux_ratio

    def __eq__(self, other):
        return (isinstance(other, galsim.Transformation) and
                self.original == other.original and
                np.array_equal(self.jac, other.jac) and
                np.array_equal(self.offset, other.offset) and
                self.flux_ratio == other.flux_ratio and
                self.gsparams == other.gsparams)

    def __hash__(self):
        return hash(("galsim.Transformation", self.original, tuple(self._jac), self.offset.x,
                     self.offset.y, self.flux_ratio, self.gsparams))

    def __repr__(self):
        return 'galsim.Transformation(%r, jac=%r, offset=%r, flux_ratio=%r, gsparams=%r)'%(
            self.original, self._jac.tolist(), self.offset, self.flux_ratio, self._gsparams)

    def __str__(self):
        s = str(self.original)
        dudx, dudy, dvdx, dvdy = self._jac
        if dudx != 1 or dudy != 0 or dvdx != 0 or dvdy != 1:
            # Figure out the shear/rotate/dilate calls that are equivalent.
            jac = galsim.JacobianWCS(dudx,dudy,dvdx,dvdy)
            scale, shear, theta, flip = jac.getDecomposition()
            single = None
            if flip:
                single = 0  # Special value indicating to just use transform.
            if abs(theta.rad()) > 1.e-12:
                if single is None:
                    single = '.rotate(%s)'%theta
                else:
                    single = 0
            if shear.getG() > 1.e-12:
                if single is None:
                    single = '.shear(%s)'%shear
                else:
                    single = 0
            if abs(scale-1.0) > 1.e-12:
                if single is None:
                    single = '.expand(%s)'%scale
                else:
                    single = 0
            if single == 0:
                # If flip or there are two components, then revert to transform as simpler.
                single = '.transform(%s,%s,%s,%s)'%(dudx,dudy,dvdx,dvdy)
            if single is None:
                # If nothing is large enough to show up above, give full detail of transform
                single = '.transform(%r,%r,%r,%r)'%(dudx,dudy,dvdx,dvdy)
            s += single
        if self.offset.x != 0 or self.offset.y != 0:
            s += '.shift(%s,%s)'%(self.offset.x,self.offset.y)
        if self.flux_ratio != 1.:
            #s += '.withScaledFlux(%s)'%self.flux_ratio
            s += ' * %s'%self.flux_ratio
        return s

    def _prepareDraw(self):
        self._original._prepareDraw()
        dudx, dudy, dvdx, dvdy = self.getJac()
        self.SBProfile = galsim._galsim.SBTransform(self._original.SBProfile,
                                                    dudx, dudy, dvdx, dvdy,
                                                    self.getOffset(), self.getFluxRatio(),
                                                    self._gsparams)

    def _fwd_ident(self, x, y):
        return x, y

    def _fwd_diag(self, x, y):
        return self._jac[0] * x, self._jac[3] * y

    def _fwd_normal(self, x, y):
        return self._jac[0] * x + self._jac[1] * y, self._jac[2] * x + self._jac[3] * y

    def shoot(self, n_photons, rng=None):
        """Shoot photons into a PhotonArray.

        @param n_photons    The number of photons to use for photon shooting.
        @param rng          If provided, a random number generator to use for photon shooting,
                            which may be any kind of BaseDeviate object.  If `rng` is None, one
                            will be automatically created, using the time as a seed.
                            [default: None]
        @returns PhotonArray.
        """
        # Depending on the jacobian, it can be significantly faster to use a specialized fwd func.
        if np.array_equal(self._jac[1:3], (0,0)):
            if np.array_equal(self._jac[::3], (1,1)):   # jac is (1,0,0,1)
                fwd = self._fwd_ident
                det = 1
            else:                                       # jac is (a,0,0,b)
                fwd = self._fwd_diag
                det = abs(self._jac[0] * self._jac[3])
        else:                                           # Fully general case
            fwd = self._fwd_normal
            det = abs(self._jac[0] * self._jac[3] - self._jac[1] * self._jac[2])

        ud = galsim.UniformDeviate(rng)
        photon_array = self.original.shoot(n_photons, ud)

        newx, newy = fwd(photon_array.x,photon_array.y)
        photon_array.x = newx + self.offset.x
        photon_array.y = newy + self.offset.y
        photon_array.scaleFlux(det*self.flux_ratio)
        return photon_array

    def __getstate__(self):
        # While the SBProfile should be picklable, it is better to reconstruct it from the
        # original object, which will pickle better.  The SBProfile is only picklable via its
        # repr, which is not the most efficient serialization.  Especially for things like
        # SBInterpolatedImage.
        d = self.__dict__.copy()
        del d['SBProfile']
        return d

    def __setstate__(self, d):
        self.__dict__ = d
        self.__init__(self._original, self._jac, self._offset, self._flux_ratio, self._gsparams)

def _Transform(obj, dudx=1, dudy=0, dvdx=0, dvdy=1, offset=galsim.PositionD(0.,0.),
               flux_ratio=1., gsparams=None):
    """Approximately equivalent to Transform (but with jac expanded out), but without all the
    sanity checks and options.

    This is only valid for GSObjects.  For ChromaticObjects, you must use the regular Transform.
    """
    ret = Transformation.__new__(Transformation)
    if hasattr(obj, 'original'):
        ret._original = obj.original
    else:
        ret._original = obj
    sbt = _galsim.SBTransform(obj.SBProfile, dudx, dudy, dvdx, dvdy, offset, flux_ratio,
                              gsparams)
    galsim.GSObject.__init__(ret, sbt)
    ret._jac = np.asarray(sbt.getJac())
    ret._offset = sbt.getOffset()
    ret._flux_ratio = sbt.getFluxScaling()
    ret._gsparams = gsparams
    return ret

def SBTransform_init(self):
    obj = self.getObj()
    dudx, dudy, dvdx, dvdy = self.getJac()
    offset = self.getOffset()
    flux_ratio = self.getFluxScaling()
    gsparams = self.getGSParams()
    return (obj, dudx, dudy, dvdx, dvdy, offset, flux_ratio, gsparams)
_galsim.SBTransform.__getinitargs__ = SBTransform_init
_galsim.SBTransform.__getstate__ = lambda self: None
_galsim.SBTransform.__repr__ = lambda self: \
        'galsim._galsim.SBTransform(%r, %r, %r, %r, %r, %r, %r, %r)'%self.__getinitargs__()
