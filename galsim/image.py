# Copyright 2012, 2013 The GalSim developers:
# https://github.com/GalSim-developers
#
# This file is part of GalSim: The modular galaxy image simulation toolkit.
#
# GalSim is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# GalSim is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with GalSim.  If not, see <http://www.gnu.org/licenses/>
#
"""@file image.py
A few adjustments to the galsim.Image classes at the Python layer, including the addition of 
docstrings.
"""


from . import _galsim
import numpy

# Sometimes (on 32-bit systems) there are two numpy.int32 types.  This can lead to some confusion 
# when doing arithmetic with images.  So just make sure both of them point to ImageI in the Image 
# dict.  One of them is what you get when you just write numpy.int32.  The other is what numpy 
# decides an int16 + int32 is.  The first one is usually the one that's already in the Image dict,
# but we assign both versions just to be sure.

_galsim.Image[numpy.int32] = _galsim.ImageI
_galsim.ImageView[numpy.int32] = _galsim.ImageViewI
_galsim.ConstImageView[numpy.int32] = _galsim.ConstImageViewI

alt_int32 = ( numpy.array([0]).astype(numpy.int32) + 1).dtype.type
_galsim.Image[alt_int32] = _galsim.ImageI
_galsim.ImageView[alt_int32] = _galsim.ImageViewI
_galsim.ConstImageView[alt_int32] = _galsim.ConstImageViewI

# On some systems, the above doesn't work, but this next one does.  I'll leave both active,
# just in case there are systems where this doesn't work but the above does.
alt_int32 = ( numpy.array([0]).astype(numpy.int16) + 
              numpy.array([0]).astype(numpy.int32) ).dtype.type
_galsim.Image[alt_int32] = _galsim.ImageI
_galsim.ImageView[alt_int32] = _galsim.ImageViewI
_galsim.ConstImageView[alt_int32] = _galsim.ConstImageViewI

# For more information regarding this rather unexpected behaviour for numpy.int32 types, see
# the following (closed, marked "wontfix") ticket on the numpy issue tracker:
# http://projects.scipy.org/numpy/ticket/1246


# First of all we add docstrings to the Image, ImageView and ConstImageView classes for each of the
# S, F, I & D datatypes
#
for Class in _galsim.Image.itervalues():
    Class.__doc__ = """
    The ImageS, ImageI, ImageF and ImageD classes.
    
    Image[SIFD], ImageView[SIFD] and ConstImage[SIFD] are the classes that represent the primary way
    to pass image data between Python and the GalSim C++ library.

    There is a separate Python class for each C++ template instantiation, and these can be accessed
    using numpy types as keys in the Image dict:

        ImageS == Image[numpy.int16]
        ImageI == Image[numpy.int32]
        ImageF == Image[numpy.float32]
        ImageD == Image[numpy.float64]
    
    An Image can be thought of as containing a 2-d, row-contiguous numpy array (which it may share
    with other ImageView objects), an origin point, and a pixel scale (the origin and pixel scale
    are not shared).

    There are several ways to construct an Image:

        Image(ncol, nrow, init_value=0)        # size and initial value - origin @ (1,1)
        Image(bounds=BoundsI(), init_value=0)  # bounding box and initial value

    An Image also has a '.array' attribute that provides a numpy view into the Image's pixels.

    The individual elements in the array attribute are accessed as im.array[y,x], matching the
    standard numpy convention, while the Image class's own accessors are all (x,y).
    """
    Class.getScale.__func__.__doc__ = "Get the sample scale dx for the image."
    Class.setScale.__func__.__doc__ = "Set the sample scale dx for the image."


for Class in _galsim.ImageView.itervalues():
    Class.__doc__ = """
    The ImageViewS, ImageViewI, ImageViewF and ImageViewD classes.

    ImageView[SIFD] represents a mutable view of an Image.

    There is a separate Python class for each C++ template instantiation, and these can be accessed
    using numpy types as keys in the ImageView dict:
    
        ImageViewS == ImageView[numpy.int16]
        ImageViewI == ImageView[numpy.int32]
        ImageViewF == ImageView[numpy.float32]
        ImageViewD == ImageView[numpy.float64]

    From Python, the only way to explicitly construct an ImageView is

        >>> imv = ImageView(array, xmin=1, ymin=1)       # numpy array and origin

    However, ImageView instances are also the return type of several functions such as

        >>> im.view()
        >>> im.subImage(bounds)
        >>> im[bounds]                                   # (equivalent to the subImage call above)
        >>> galsim.fits.read(...)
    
    The array argument to the constructor must have contiguous values along rows, which should be
    the case for newly-constructed arrays, but may not be true for some views and generally will not
    be true for array transposes.
    
    An ImageView also has a '.array' attribute that provides a numpy array view into the ImageView
    instance's pixels.  Regardless of how the ImageView was constructed, this array and the
    ImageView will point to the same underlying data, and modifying one view will affect any other
    views into the same data.
    
    The individual elements in the array attribute are accessed as im.array[y,x], matching the
    standard numpy convention as used in the array input to the constructor, while the ImageView 
    class's own accessors are all (x,y).
    """

for Class in _galsim.ConstImageView.itervalues():
    Class.__doc__ = """
    The ConstImageViewS, ConstImageViewI, ConstImageViewF and ConstImageViewD classes.

    ConstImageView[SIFD] represents a non-mutable view of an Image.

    There is a separate Python class for each C++ template instantiation, and these can be accessed
    using NumPy types as keys in the ConstImageView dict:

        ConstImageViewS == ConstImageView[numpy.int16]
        ConstImageViewI == ConstImageView[numpy.int32]
        ConstImageViewF == ConstImageView[numpy.float32]
        ConstImageViewD == ConstImageView[numpy.float64]

    From Python, the only way to explicitly construct an ConstImageView is

        >>> cimv = ConstImageView(array, xmin=1, ymin=1)       # NumPy array and origin

    which works just like the version for ImageView except that the resulting object cannot be used
    to modify the array.
    """

def Image_setitem(self, key, value):
    self.subImage(key).copyFrom(value)

def Image_getitem(self, key):
    return self.subImage(key)

def Image_add(self, other):
    try:
        result = self.array[:,:] + other.array
    except AttributeError:
        result = self.array[:,:] + other
    res_type = result.dtype.type
    return _galsim.ImageView[res_type](result)

def Image_iadd(self, other):
    try:
        self.array[:,:] += other.array
    except AttributeError:
        self.array[:,:] += other
    return self

def Image_sub(self, other):
    try:
        result = self.array[:,:] - other.array
    except AttributeError:
        result = self.array[:,:] - other
    res_type = result.dtype.type
    return _galsim.ImageView[res_type](result)

def Image_isub(self, other):
    try:
        self.array[:,:] -= other.array
    except AttributeError:
        self.array[:,:] -= other
    return self

def Image_mul(self, other):
    try:
        result = self.array[:,:] * other.array
    except AttributeError:
        result = self.array[:,:] * other
    res_type = result.dtype.type
    return _galsim.ImageView[res_type](result)

def Image_imul(self, other):
    try:
        self.array[:,:] *= other.array
    except AttributeError:
        self.array[:,:] *= other
    return self

def Image_div(self, other):
    try:
        result = self.array[:,:] / other.array
    except AttributeError:
        result = self.array[:,:] / other
    res_type = result.dtype.type
    return _galsim.ImageView[res_type](result)

def Image_idiv(self, other):
    try:
        self.array[:,:] /= other.array
    except AttributeError:
        self.array[:,:] /= other
    return self

def Image_NoiseSNR(self, sn, rng=_galsim.UniformDeviate()):
	sky_level=1.E6
	sn_meas=numpy.sqrt( numpy.sum(self.array**2)/sky+level )
	flux=sn/sn_meas
	self*=flux
	self+=sky_level
	self.addNoise(_galsim.CCDNoise(rng))
	self-=sky_level

# Define &, ^ and | only for integer-type images
def Image_and(self, other):
    try:
        result = self.array[:,:] & other.array
    except AttributeError:
        result = self.array[:,:] & other
    res_type = result.dtype.type
    return _galsim.ImageView[res_type](result)

def Image_iand(self, other):
    try:
        self.array[:,:] &= other.array
    except AttributeError:
        self.array[:,:] &= other
    return self

def Image_xor(self, other):
    try:
        result = self.array[:,:] ^ other.array
    except AttributeError:
        result = self.array[:,:] ^ other
    res_type = result.dtype.type
    return _galsim.ImageView[res_type](result)

def Image_ixor(self, other):
    try:
        self.array[:,:] ^= other.array
    except AttributeError:
        self.array[:,:] ^= other
    return self

def Image_or(self, other):
    try:
        result = self.array[:,:] | other.array
    except AttributeError:
        result = self.array[:,:] | other
    res_type = result.dtype.type
    return _galsim.ImageView[res_type](result)

def Image_ior(self, other):
    try:
        self.array[:,:] |= other.array
    except AttributeError:
        self.array[:,:] |= other
    return self


def Image_copy(self):
    # self can be an Image or an ImageView, but the return type needs to be an Image.
    # So use the array.dtype.type attribute to get the type of the underlying data,
    # which in turn can be used to index our Image dictionary:
    return _galsim.Image[self.array.dtype.type](self)

# Some functions to enable pickling of images
def ImageView_getinitargs(self):
    return self.array, self.xmin, self.ymin, self.scale

# An image is really pickled as an ImageView
def Image_getstate(self):
    return self.array, self.xmin, self.ymin, self.scale

def Image_setstate(self, args):
    self_type = args[0].dtype.type
    self.__class__ = _galsim.ImageView[self_type]
    self.__init__(*args)

# inject these as methods of Image classes
for Class in _galsim.Image.itervalues():
    Class.__setitem__ = Image_setitem
    Class.__getitem__ = Image_getitem
    Class.__add__ = Image_add
    Class.__radd__ = Image_add
    Class.__iadd__ = Image_iadd
    Class.__sub__ = Image_sub
    Class.__isub__ = Image_isub
    Class.__mul__ = Image_mul
    Class.__rmul__ = Image_mul
    Class.__imul__ = Image_imul
    Class.__div__ = Image_div
    Class.__truediv__ = Image_div
    Class.__idiv__ = Image_idiv
    Class.__itruediv__ = Image_idiv
    Class.copy = Image_copy
    Class.addNoiseSNR = Image_NoiseSNR
    Class.__getstate_manages_dict__ = 1
    Class.__getstate__ = Image_getstate
    Class.__setstate__ = Image_setstate

for Class in _galsim.ImageView.itervalues():
    Class.__setitem__ = Image_setitem
    Class.__getitem__ = Image_getitem
    Class.__add__ = Image_add
    Class.__radd__ = Image_add
    Class.__iadd__ = Image_iadd
    Class.__sub__ = Image_sub
    Class.__isub__ = Image_isub
    Class.__mul__ = Image_mul
    Class.__rmul__ = Image_mul
    Class.__imul__ = Image_imul
    Class.__div__ = Image_div
    Class.__truediv__ = Image_div
    Class.__idiv__ = Image_idiv
    Class.__itruediv__ = Image_idiv
    Class.copy = Image_copy
    Class.__getinitargs__ = ImageView_getinitargs

for Class in _galsim.ConstImageView.itervalues():
    Class.__getitem__ = Image_getitem
    Class.__add__ = Image_add
    Class.__radd__ = Image_add
    Class.__sub__ = Image_sub
    Class.__mul__ = Image_mul
    Class.__rmul__ = Image_mul
    Class.__div__ = Image_div
    Class.__truediv__ = Image_div
    Class.copy = Image_copy
    Class.__getinitargs__ = ImageView_getinitargs

import numpy as np
for int_type in [ np.int16, np.int32 ]:
    for Class in [ _galsim.Image[int_type], _galsim.ImageView[int_type],
                   _galsim.ConstImageView[int_type] ]:
        Class.__and__ = Image_and
        Class.__xor__ = Image_xor
        Class.__or__ = Image_or
    for Class in [ _galsim.Image[int_type], _galsim.ImageView[int_type] ]:
        Class.__iand__ = Image_iand
        Class.__ixor__ = Image_ixor
        Class.__ior__ = Image_ior

del Class    # cleanup public namespace
