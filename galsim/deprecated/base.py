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

# NB. drawImage and drawKImage allow a deprecated dx parameter.
# Remove those parameters when we remove this function.

def GSObject_nyquistDx(self):
    """A deprecated synonym for nyquistScale()"""
    depr('nyquistDx', 1.1, 'nyquistScale()')
    return self.nyquistScale()

def GSObject_setFlux(self, flux):
    """A deprecated method that is roughly equivalent to obj = obj.withFlux(flux)"""
    depr('setFlux', 1.1, 'obj = obj.withFlux(flux)')
    new_obj = self.copy().withFlux(flux)
    self.__class__ = new_obj.__class__
    self.__setstate__(new_obj.__getstate__())

def GSObject_scaleFlux(self, flux_ratio):
    """A deprecated method that is roughly equivalent to obj = obj * flux_ratio"""
    depr('scaleFlux', 1.1, 'obj = obj * flux_ratio')
    new_obj = self.copy() * flux_ratio
    self.__class__ = new_obj.__class__
    self.__setstate__(new_obj.__getstate__())

def GSObject_createExpanded(self, scale):
    """A deprecated synonym for expand(scale)"""
    depr('createExpanded', 1.1, 'obj.expand(scale)')
    return self.expand(scale)

def GSObject_applyExpansion(self, scale):
    """A deprecated method that is roughly equivalent to obj = obj.expand(scale)."""
    depr('applyExpansion', 1.1, 'obj = obj.expand(scale)')
    new_obj = self.copy().expand(scale)
    self.__class__ = new_obj.__class__
    self.__setstate__(new_obj.__getstate__())

def GSObject_createDilated(self, scale):
    """A deprecated synonym for dilate(scale)"""
    depr('createDilated', 1.1, 'obj.dilate(scale)')
    return self.dilate(scale)

def GSObject_applyDilation(self, scale):
    """A deprecated method that is roughly equivalent to obj = obj.dilate(scale)."""
    depr('applyDilation', 1.1, 'obj = obj.dilate(scale)')
    new_obj = self.copy().dilate(scale)
    self.__class__ = new_obj.__class__
    self.__setstate__(new_obj.__getstate__())

def GSObject_createMagnified(self, mu):
    """A deprecated synonym for magnify(mu)"""
    depr('createMagnified', 1.1, 'obj.magnify(mu)')
    return self.magnify(mu)

def GSObject_applyMagnification(self, mu):
    """A deprecated method that is roughly equivalent to obj = obj.magnify(mu)"""
    depr('applyMagnification', 1.1, 'obj = obj.magnify(mu)')
    new_obj = self.copy().magnify(mu)
    self.__class__ = new_obj.__class__
    self.__setstate__(new_obj.__getstate__())

def GSObject_createSheared(self, *args, **kwargs):
    """A deprecated synonym for shear(shear)"""
    depr('createSheared', 1.1, 'obj.shear(shear)')
    return self.shear(*args, **kwargs)

def GSObject_applyShear(self, *args, **kwargs):
    """A deprecated method that is roughly equivalent to obj = obj.shear(shear)"""
    depr('applyShear', 1.1, 'obj = obj.shear(shear)')
    new_obj = self.copy().shear(*args, **kwargs)
    self.__class__ = new_obj.__class__
    self.__setstate__(new_obj.__getstate__())

def GSObject_createLensed(self, g1, g2, mu):
    """A deprecated synonym for lens(g1,g2,mu)"""
    depr('createLensed', 1.1, 'obj.lens(g1,g2,mu)')
    return self.lens(g1,g2,mu)

def GSObject_applyLensing(self, g1, g2, mu):
    """A deprecated method that is roughly equivalent to obj = obj.lens(g1,g2,mu)"""
    depr('applyLensing', 1.1, 'obj = obj.lens(g1,g2,mu)')
    new_obj = self.copy().lens(g1,g2,mu)
    self.__class__ = new_obj.__class__
    self.__setstate__(new_obj.__getstate__())

def GSObject_createRotated(self, theta):
    """A deprecated synonym for rotate(theta)"""
    depr('createRotated', 1.1, 'obj.rotate(theta)')
    return self.rotate(theta)

def GSObject_applyRotation(self, theta):
    """A deprecated method that is roughly equivalent to obj = obj.rotate(theta)"""
    depr('applyRotation', 1.1, 'obj = obj.rotate(theta)')
    new_obj = self.copy().rotate(theta)
    self.__class__ = new_obj.__class__
    self.__setstate__(new_obj.__getstate__())

def GSObject_createTransformed(self, dudx, dudy, dvdx, dvdy):
    """A deprecated sysnonym for transform()"""
    depr('createTransformed', 1.1, 'obj.transform(dudx,dudy,dvdx,dvdy)')
    return self.transform(dudx,dudy,dvdx,dvdy)

def GSObject_applyTransformation(self, dudx, dudy, dvdx, dvdy):
    """A deprecated method that is roughly equivalent to obj = obj.transform(...)"""
    depr('applyTransformation', 1.1, 'obj = obj.transform(dudx,dudy,dvdx,dvdy)')
    new_obj = self.copy().transform(dudx,dudy,dvdx,dvdy)
    self.__class__ = new_obj.__class__
    self.__setstate__(new_obj.__getstate__())

def GSObject_createShifted(self, *args, **kwargs):
    """A deprecated synonym for shift(dx,dy)"""
    depr('createShifted', 1.1, 'obj.shift(dx,dy)')
    return self.shift(*args,**kwargs)

def GSObject_applyShift(self, *args, **kwargs):
    """A deprecated method that is roughly equivalent to obj = obj.shift(dx,dy)"""
    depr('applyShift', 1.1, 'obj = obj.shift(dx,dy)')
    new_obj = self.copy().shift(*args,**kwargs)
    self.__class__ = new_obj.__class__
    self.__setstate__(new_obj.__getstate__())

def GSObject_draw(self, *args, **kwargs):
    """A deprecated synonym for obj.drawImage(method='no_pixel')
    """
    depr('draw', 1.1, "drawImage(..., method='no_pixel')",
         'Note: drawImage has different args than draw did.  '+
         'Read the docs for the method keywords carefully.')
    normalization = kwargs.pop('normalization','f')
    if normalization in ['flux','f']:
        return self.drawImage(*args, method='no_pixel', **kwargs)
    else:
        return self.drawImage(*args, method='sb', **kwargs)

def GSObject_drawShoot(self, *args, **kwargs):
    """A deprecated synonym for obj.drawImage(method='phot')
    """
    depr('drawShoot', 1.1, "drawImage(..., method='phot')",
         'Note: drawImage has different args than draw did.  '+
         'Read the docs for the method keywords carefully.')
    normalization = kwargs.pop('normalization','f')
    if normalization in ['flux','f']:
        return self.drawImage(*args, method='phot', **kwargs)
    else:
        # We don't have a method for this, but I think it must be rare.  Photon shooting
        # with surface brightness normalization seems pretty odd.  We do use it in the test
        # suite a few times though.  So, need to reproduce a bit of code to get the
        # pixel area to switch to sb normalization (via the gain).
        if len(args) > 0:
            image = args[0]
        else:
            image = kwargs.get('image', None)
        scale = kwargs.get('scale', None)
        wcs = kwargs.get('wcs', None)
        offset = kwargs.get('offset', None)
        use_true_center = kwargs.get('use_true_center', None)
        wcs = self._determine_wcs(scale, wcs, image)
        offset = self._parse_offset(offset)
        local_wcs = self._local_wcs(wcs, image, offset, use_true_center)
        gain = kwargs.pop('gain',1.)
        gain *= local_wcs.pixelArea()
        return self.drawImage(*args, method='phot', gain=gain, **kwargs)

def GSObject_drawK(self, *args, **kwargs):
    """A deprecated synonym for drawKImage()
    """
    depr('drawK', 1.1, "drawKImage")
    return self.drawKImage(*args, **kwargs)

def GSObject_copy(self):
    """Returns a copy of an object.

    NB. This is a shallow copy, which is normally fine.  However, if the object has a noise
    attribute, then the copy will use the same rng, so calls to things like noise.whitenImage
    from the two copies would produce different realizations of the noise.  If you want
    these to be precisely identical, then copy.deepcopy will make an exact duplicate, which
    will have identical noise realizations for that kind of application.
    """
    depr('copy', 1.5, "", "GSObjects are immutable, so there's no need for copy.")
    import copy
    return copy.copy(self)


galsim.GSObject.nyquistDx = GSObject_nyquistDx
galsim.GSObject.setFlux = GSObject_setFlux
galsim.GSObject.scaleFlux = GSObject_scaleFlux
galsim.GSObject.createExpanded = GSObject_createExpanded
galsim.GSObject.applyExpansion = GSObject_applyExpansion
galsim.GSObject.createDilated = GSObject_createDilated
galsim.GSObject.applyDilation = GSObject_applyDilation
galsim.GSObject.createMagnified = GSObject_createMagnified
galsim.GSObject.applyMagnification = GSObject_applyMagnification
galsim.GSObject.createSheared = GSObject_createSheared
galsim.GSObject.applyShear = GSObject_applyShear
galsim.GSObject.createLensed = GSObject_createLensed
galsim.GSObject.applyLensing = GSObject_applyLensing
galsim.GSObject.createRotated = GSObject_createRotated
galsim.GSObject.applyRotation = GSObject_applyRotation
galsim.GSObject.createTransformed = GSObject_createTransformed
galsim.GSObject.applyTransformation = GSObject_applyTransformation
galsim.GSObject.createShifted = GSObject_createShifted
galsim.GSObject.applyShift = GSObject_applyShift
galsim.GSObject.draw = GSObject_draw
galsim.GSObject.drawShoot = GSObject_drawShoot
galsim.GSObject.drawK = GSObject_drawK
galsim.GSObject.copy = GSObject_copy


# GSParams is defined in C++ and wrapped.  But we want to modify it here slightly to add
# the obsolete name alias_threshold as a valid synonym for folding_threshold
def _get_alias_threshold(self):
    depr('alias_threshold',1.1,'folding_threshold')
    return self.folding_threshold

# Also update the constructor to allow this name.
_orig_GSP_init = galsim.GSParams.__init__
def _new_GSP_init(self, *args, **kwargs):
    if 'alias_threshold' in kwargs:
        if 'folding_threshold' in kwargs:
            raise TypeError('Cannot specify both alias_threshold and folding_threshold')
        depr('alias_threshold',1.1,'folding_threshold')
        kwargs['folding_threshold'] = kwargs.pop('alias_threshold')
    _orig_GSP_init(self, *args, **kwargs)

galsim.GSParams.alias_threshold = property(_get_alias_threshold)
galsim.GSParams.__init__ = _new_GSP_init
