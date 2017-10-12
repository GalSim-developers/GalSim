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

import galsim
from galsim.deprecated import depr

def CN_applyWhiteningTo(self, image):
    """A deprecated synonym for whitenImage"""
    depr('applyWhiteningTo', 1.2, 'whitenImage')
    return self.whitenImage(image)

def CN_createExpanded(self, scale):
    """A deprecated synonym for expand(scale)"""
    depr('createExpanded', 1.1, 'obj.expand(scale)')
    return self.expand(scale)

def CN_applyExpansion(self, scale):
    """A deprecated method that is roughly equivalent to obj = obj.expand(scale)"""
    depr('applyExpansion', 1.1, 'obj = obj.expand(scale)')
    new_obj = self.copy().expand(scale)
    self._profile = new_obj._profile
    self._profile_for_stored = None  # Reset the stored profile as it is no longer up-to-date
    self.__class__ = new_obj.__class__

def CN_createDilated(self, scale):
    """A deprecated synonym for dilate(scale)"""
    depr('createDilated', 1.1, 'obj.dilate(scale)')
    return self.dilate(scale)

def CN_applyDilation(self, scale):
    """A deprecated method that is roughly equivalent to obj = obj.dilate(scale)"""
    depr('applyDilation', 1.1, 'obj = obj.dilate(scale)')
    new_obj = self.copy().dilate(scale)
    self._profile = new_obj._profile
    self._profile_for_stored = None  # Reset the stored profile as it is no longer up-to-date
    self.__class__ = new_obj.__class__

def CN_createMagnified(self, mu):
    """A deprecated synonym for magnify(mu)"""
    depr('createMagnified', 1.1, 'obj.magnify(mu)')
    return self.magnify(mu)

def CN_applyMagnification(self, mu):
    """A deprecated method that is roughly equivalent to obj = obj.magnify(mu)"""
    depr('applyMagnification', 1.1, 'obj = obj.magnify(mu)')
    new_obj = self.copy().magnify(mu)
    self._profile = new_obj._profile
    self._profile_for_stored = None  # Reset the stored profile as it is no longer up-to-date
    self.__class__ = new_obj.__class__

def CN_createLensed(self, g1, g2, mu):
    """A deprecated synonym for lens(g1,g2,mu)"""
    depr('createLensed', 1.1, 'obj.lens(g1,g2,mu)')
    return self.lens(g1,g2,mu)

def CN_applyLensing(self, g1, g2, mu):
    """A deprecated method that is roughly equivalent to obj = obj.lens(g1,g2,mu)"""
    depr('applyLensing', 1.1, 'obj = obj.lens(g1,g2,mu)')
    new_obj = self.copy().lens(g1,g2,mu)
    self._profile = new_obj._profile
    self._profile_for_stored = None  # Reset the stored profile as it is no longer up-to-date
    self.__class__ = new_obj.__class__

def CN_createRotated(self, theta):
    """A deprecated synonym for rotate(theta)"""
    depr('createRotated', 1.1, 'obj.rotate(theta)')
    return self.rotate(theta)

def CN_applyRotation(self, theta):
    """A deprecated method that is roughly equivalent to obj = obj.rotate(theta)"""
    depr('applyRotation', 1.1, 'obj = obj.rotate(theta)')
    new_obj = self.rotate(theta)
    self._profile = new_obj._profile
    self._profile_for_stored = None  # Reset the stored profile as it is no longer up-to-date
    self.__class__ = new_obj.__class__

def CN_createSheared(self, *args, **kwargs):
    """A deprecated synonym for shear(shear)"""
    depr('createSheared', 1.1, 'obj.shear(shear)')
    return self.shear(*args,**kwargs)

def CN_applyShear(self, *args, **kwargs):
    """A deprecated method that is roughly equivalent to obj = obj.shear(shear)"""
    depr('applyShear', 1.1, 'obj = obj.shear(shear)')
    new_obj = self.copy().shear(*args, **kwargs)
    self._profile = new_obj._profile
    self._profile_for_stored = None  # Reset the stored profile as it is no longer up-to-date
    self.__class__ = new_obj.__class__

def CN_createTransformed(self, dudx, dudy, dvdx, dvdy):
    """A deprecated synonym for transform(dudx,dudy,dvdx,dvdy)"""
    depr('createTransformed', 1.1, 'obj.transform(dudx,dudy,dvdx,dvdy)')
    return self.transform(dudx,dudy,dvdx,dvdy)

def CN_applyTransformation(self, dudx, dudy, dvdx, dvdy):
    """A deprecated method that is roughly equivalent to obj = obj.transform(...)"""
    depr('applyTransformation', 1.1, 'obj = obj.transform(dudx,dudy,dvdx,dvdy)')
    new_obj = self.copy().transform(dudx,dudy,dvdx,dvdy)
    self._profile = new_obj._profile
    self._profile_for_stored = None  # Reset the stored profile as it is no longer up-to-date
    self.__class__ = new_obj.__class__

def CN_setVariance(self, variance):
    """A deprecated method that is roughly equivalent to
    corr = corr.withVariance(variance)
    """
    depr('setVariance', 1.1, 'obj = obj.withVariance(variance)')
    new_obj = self.copy().withVariance(variance)
    self._profile = new_obj._profile
    self._profile_for_stored = None  # Reset the stored profile as it is no longer up-to-date
    self.__class__ = new_obj.__class__

def CN_scaleVariance(self, variance_ratio):
    """A deprecated method that is roughly equivalent to corr = corr * variance_ratio"""
    depr('scaleVariance', 1.1, 'obj = obj * variance_ratio')
    new_obj = self.copy().withScaledVariance(variance_ratio)
    self._profile = new_obj._profile
    self._profile_for_stored = None  # Reset the stored profile as it is no longer up-to-date
    self.__class__ = new_obj.__class__

def CN_convolveWith(self, gsobject, gsparams=None):
    """A deprecated method that is roughly equivalent to
    cn = cn.convolvedWith(gsobject,gsparams)
    """
    depr('convolveWith', 1.1, 'obj = obj.convolvedWith(gsobject, gsparams)')
    new_obj = self.copy().convolvedWith(gsobject,gsparams)
    self._profile = new_obj._profile
    self._profile_for_stored = None  # Reset the stored profile as it is no longer up-to-date
    self.__class__ = new_obj.__class__

def CN_draw(self, *args, **kwargs):
    """A deprecated synonym of drawImage"""
    depr('draw', 1.1, "drawImage")
    return self.drawImage(*args, **kwargs)

def CN_calculateCovarianceMatrix(self, bounds, scale):
    """This function is deprecated and will be removed in a future version.  If you have a
    use for this function and would like to keep it, please open an issue at:

    https://github.com/GalSim-developers/GalSim/issues

    Old documentation:
    ------------------
    Calculate the covariance matrix for an image with specified properties.

    A correlation function also specifies a covariance matrix for noise in an image of known
    dimensions and pixel scale.  The user specifies these bounds and pixel scale, and this
    method returns a covariance matrix as a square Image object, with the upper triangle
    containing the covariance values.

    @param  bounds Bounds corresponding to the dimensions of the image for which a covariance
                    matrix is required.
    @param  scale  Pixel scale of the image for which a covariance matrix is required.

    @returns the covariance matrix (as an Image).
    """
    depr('calculateCovarianceMatrix',1.3,'',
         'This functionality has been removed. If you have a need for it, please open '+
         'an issue requesting the functionality.')
    return galsim._galsim._calculateCovarianceMatrix(self._profile._sbp, bounds, scale)

galsim.correlatednoise._BaseCorrelatedNoise.applyWhiteningTo = CN_applyWhiteningTo
galsim.correlatednoise._BaseCorrelatedNoise.createExpanded = CN_createExpanded
galsim.correlatednoise._BaseCorrelatedNoise.applyExpansion = CN_applyExpansion
galsim.correlatednoise._BaseCorrelatedNoise.createDilated = CN_createDilated
galsim.correlatednoise._BaseCorrelatedNoise.applyDilation = CN_applyDilation
galsim.correlatednoise._BaseCorrelatedNoise.createMagnified = CN_createMagnified
galsim.correlatednoise._BaseCorrelatedNoise.applyMagnification = CN_applyMagnification
galsim.correlatednoise._BaseCorrelatedNoise.createLensed = CN_createLensed
galsim.correlatednoise._BaseCorrelatedNoise.applyLensing = CN_applyLensing
galsim.correlatednoise._BaseCorrelatedNoise.createRotated = CN_createRotated
galsim.correlatednoise._BaseCorrelatedNoise.applyRotation = CN_applyRotation
galsim.correlatednoise._BaseCorrelatedNoise.createSheared = CN_createSheared
galsim.correlatednoise._BaseCorrelatedNoise.applyShear = CN_applyShear
galsim.correlatednoise._BaseCorrelatedNoise.createTransformed = CN_createTransformed
galsim.correlatednoise._BaseCorrelatedNoise.applyTransformation = CN_applyTransformation
galsim.correlatednoise._BaseCorrelatedNoise.setVariance = CN_setVariance
galsim.correlatednoise._BaseCorrelatedNoise.scaleVariance = CN_scaleVariance
galsim.correlatednoise._BaseCorrelatedNoise.convolveWith = CN_convolveWith
galsim.correlatednoise._BaseCorrelatedNoise.draw = CN_draw
galsim.correlatednoise._BaseCorrelatedNoise.calculateCovarianceMatrix = CN_calculateCovarianceMatrix

def _Image_getCorrelatedNoise(image):
    """Deprecated method to get a CorrelatedNoise instance by calculating the correlation function
    of image pixels.  It is equivalent to `noise = galsim.CorrelatedNoise(image)`
    """
    depr('getCorrelatedNoise',1.1,'noise = galsim.CorrelatedNoise(image)')
    return CorrelatedNoise(image)

# Then add this Image method to the Image class
galsim.Image.getCorrelatedNoise = _Image_getCorrelatedNoise

