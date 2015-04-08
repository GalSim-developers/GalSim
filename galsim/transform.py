# Copyright (c) 2012-2014 by the GalSim developers team on GitHub
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
from . import _galsim

def Transform(obj, jac=(1.,0.,0.,1.), offset=galsim.PositionD(0.,0.), flux_ratio=1.,
              gsparams=None):
    """A function for transforming either a GSObject or ChromaticObject.

    This function will inspect its input argument to decide if a Transformation object or a
    ChromaticTransformation object is required to represent the resulting transformed object.

    @param obj              The object to be transformed.
    @param jac              A list or tuple ( dudx, dudy, dvdx, dvdy ) describing the Jacobian
                            of the transformation. [default: (1,0,0,1)]
    @param offset           A galsim.PositionD giving the offset by which to shift the profile.
    @param flux_ratio       A factor by which to multiply the flux of the object. [default: 1]
    @param gsparams         An optional GSParams argument.  See the docstring for GSParams for
                            details. [default: None]

    @returns a Transformation or ChromaticTransformation instance as appropriate.
    """
    if isinstance(obj, galsim.ChromaticObject):
        return galsim.ChromaticTransformation(obj, jac, offset, flux_ratio, gsparams)
    elif isinstance(obj, galsim.GSObject):
        return Transformation(obj, jac, offset, flux_ratio, gsparams)
    else:
        raise TypeError("Argument to Transform must be either a GSObject or a ChromaticObject.")


class Transformation(galsim.GSObject):
    """A class for modeling an affine transformation of a GSObject instance.

    Initialization
    --------------

    Typically, you do not need to construct a Transformation object explicitly.  This is the type
    returned by the various transformation methods of GSObject such as shear(), rotat(), 
    shift(), transform(), etc.  All the various transformations can be described as a combination
    of transform() and shift(), which are described by (dudx,dudy,dvdx,dvdy) and (dx,dy)
    respectively.

    @param obj              The object to be transformed.
    @param jac              A list or tuple ( dudx, dudy, dvdx, dvdy ) describing the Jacobian
                            of the transformation. [default: (1,0,0,1)]
    @param offset           A galsim.PositionD giving the offset by which to shift the profile.
    @param flux_ratio       A factor by which to multiply the flux of the object. [default: 1]
    @param gsparams         An optional GSParams argument.  See the docstring for GSParams for
                            details. [default: None]

    Attributes
    ----------

    original        The original object that is being transformed.
    jac             The jacobian of the transformation matrix.
    offset          The offset being applied.
    flux_ratio      The amount by which the original flux is multiplied.
    gsparams        The usual gsparams attribute that all GSObjects have.

    Note: if `gsparams` is unspecified (or None), then the Transformation instance inherits the
    GSParams from obj.  Also, note that parameters related to the Fourier-space calculations must
    be set when initializing obj, NOT when creating the Transform (at which point the accuracy and
    threshold parameters will simply be ignored).
    """
    def __init__(self, obj, jac=(1.,0.,0.,1.), offset=galsim.PositionD(0.,0.), flux_ratio=1.,
                 gsparams=None):
        dudx, dudy, dvdx, dvdy = jac
        if hasattr(obj, 'original'):
            self._original = obj.original
        else:
            self._original = obj
        sbt = _galsim.SBTransform(obj.SBProfile, dudx, dudy, dvdx, dvdy, offset, flux_ratio,
                                  gsparams)
        galsim.GSObject.__init__(self, sbt)
        self._gsparams = gsparams

    def getJac(self):
        """Return the Jacobian of the transformation
        """
        return self.SBProfile.getJac()

    def getOffset(self):
        """Return the offset of the transformation
        """
        return self.SBProfile.getOffset()

    def getFluxRatio(self):
        """Return the flux ratio of the transformation
        """
        return self.SBProfile.getFluxScaling()

    @property
    def original(self): return self._original
    @property
    def jac(self): return self.getJac()
    @property
    def offset(self): return self.getOffset()
    @property
    def flux_ratio(self): return self.getFluxRatio()

    def __repr__(self):
        return 'galsim.Transformation(%r, jac=%r, offset=%r, flux_ratio=%r, gsparams=%r)'%(
            self.original, self.jac.tolist(), self.offset, self.flux_ratio, self._gsparams)

    def __str__(self):
        s = str(self.original)
        dudx, dudy, dvdx, dvdy = self.jac
        if dudx != 1 or dudy != 0 or dvdx != 0 or dvdy != 1:
            #s += '.transform(%s,%s,%s,%s)'%(dudx,dudy,dvdx,dvdy)
            # Figure out the shear/rotate/dilate calls that are equivalent.
            jac = galsim.JacobianWCS(dudx,dudy,dvdx,dvdy)
            scale, shear, theta, flip = jac.getDecomposition()
            if flip:
                s += '.transform(0,1,1,0)'
            if abs(theta.rad() > 1.e-12):
                s += '.rotate(%s)'%theta
            if shear.getG() > 1.e-12:
                s += '.shear(%s)'%shear
            if abs(scale-1.0 > 1.e-12):
                s += '.expand(%s)'%scale
        if self.offset.x != 0 or self.offset.y != 0:
            s += '.shift(%s,%s)'%(self.offset.x,self.offset.y)
        if self.flux_ratio != 1.:
            s += '.withScaledFlux(%s)'%self.flux_ratio
        return s

    def __getstate__(self):
        # While the SBProfile should be picklable, it is better to reconstruct it from the
        # original object, which will pickle better.  The SBProfile is only picklable via its
        # repr, which is not the most efficient serialization.  Especially for things like
        # SBInterpolatedImage.
        d = self.__dict__.copy()
        del d['SBProfile']
        d['_jac'] = self.jac
        d['_offset'] = self.offset
        d['_flux_ratio'] = self.flux_ratio
        return d

    def __setstate__(self, d):
        self.__dict__ = d
        self.__init__(self._original, self._jac, self._offset, self._flux_ratio, self._gsparams)


def SBTransform_init(self):
    obj = self.getObj()
    dudx, dudy, dvdx, dvdy = self.getJac()
    offset = self.getOffset()
    flux_ratio = self.getFluxScaling()
    gsparams = self.getGSParams()
    return (obj, dudx, dudy, dvdx, dvdy, offset, flux_ratio, gsparams)
_galsim.SBTransform.__getinitargs__ = SBTransform_init
_galsim.SBTransform.__getstate__ = lambda self: None
_galsim.SBTransform.__setstate__ = lambda self, state: 1
_galsim.SBTransform.__repr__ = lambda self: \
        'galsim._galsim.SBTransform(%r, %r, %r, %r, %r, %r, %r, %r)'%self.__getinitargs__()

