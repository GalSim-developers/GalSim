"""
A few adjustments to the Image classes at the Python layer, including the addition of docstrings.
"""
from . import _galsim

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

        >>> imv = ImageView(array, xMin=1, yMin=1)       # numpy array and origin

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

        >>> cimv = ConstImageView(array, xMin=1, yMin=1)       # NumPy array and origin

    which works just like the version for ImageView except that the resulting object cannot be used
    to modify the array.
    """

def Image_setitem(self, key, value):
    self.subImage(key).copyFrom(value)

def Image_getitem(self, key):
    return self.subImage(key)

def Image_add(self, other):
    ret = self.copy()
    ret += other
    return ret

def Image_iadd(self, other):
    try:
        self.array[:,:] += other.array
    except AttributeError:
        self.array[:,:] += other
    return self

def Image_sub(self, other):
    ret = self.copy()
    ret -= other
    return ret

def Image_isub(self, other):
    try:
        self.array[:,:] -= other.array
    except AttributeError:
        self.array[:,:] -= other
    return self

def Image_mul(self, other):
    ret = self.copy()
    ret *= other
    return ret

def Image_imul(self, other):
    try:
        self.array[:,:] *= other.array
    except AttributeError:
        self.array[:,:] *= other
    return self

def Image_div(self, other):
    ret = self.copy()
    ret /= other
    return ret

def Image_idiv(self, other):
    try:
        self.array[:,:] /= other.array
    except AttributeError:
        self.array[:,:] /= other
    return self

def Image_copy(self):
    # self can be an Image or an ImageView, but the return type needs to be an Image.
    # So use the array.dtype.type attribute to get the type of the underlying data,
    # which in turn can be used to index our Image dictionary:
    return _galsim.Image[self.array.dtype.type](self)

# Some function to enable pickling of images
def ImageView_getinitargs(self):
    return self.array, self.xMin, self.yMin

# An image is really pickled as an ImageView
def Image_getstate(self):
    return self.array, self.xMin, self.yMin

def Image_setstate(self, args):
    type = args[0].dtype.type
    self.__class__ = _galsim.ImageView[type]
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

del Class    # cleanup public namespace
