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
"""@file image.py
The Image class and some modifications to the docs for the C++ layer ImageAlloc and ImageView
classes.
"""

from . import _galsim
import numpy
import galsim

# Sometimes (on 32-bit systems) there are two numpy.int32 types.  This can lead to some confusion
# when doing arithmetic with images.  So just make sure both of them point to ImageAllocI in the
# ImageAlloc dict.  One of them is what you get when you just write numpy.int32.  The other is
# what numpy decides an int16 + int32 is.  The first one is usually the one that's already in the
# ImageAlloc dict, but we assign both versions just to be sure.

_galsim.ImageAlloc[numpy.int32] = _galsim.ImageAllocI
_galsim.ImageView[numpy.int32] = _galsim.ImageViewI
_galsim.ConstImageView[numpy.int32] = _galsim.ConstImageViewI

alt_int32 = ( numpy.array([0]).astype(numpy.int32) + 1).dtype.type
_galsim.ImageAlloc[alt_int32] = _galsim.ImageAllocI
_galsim.ImageView[alt_int32] = _galsim.ImageViewI
_galsim.ConstImageView[alt_int32] = _galsim.ConstImageViewI

# On some systems, the above doesn't work, but this next one does.  I'll leave both active,
# just in case there are systems where this doesn't work but the above does.
alt_int32 = ( numpy.array([0]).astype(numpy.int16) +
              numpy.array([0]).astype(numpy.int32) ).dtype.type
_galsim.ImageAlloc[alt_int32] = _galsim.ImageAllocI
_galsim.ImageView[alt_int32] = _galsim.ImageViewI
_galsim.ConstImageView[alt_int32] = _galsim.ConstImageViewI

# For more information regarding this rather unexpected behaviour for numpy.int32 types, see
# the following (closed, marked "wontfix") ticket on the numpy issue tracker:
# http://projects.scipy.org/numpy/ticket/1246

# This meta class thing is to allow the obsolete syntax Image[float32](ncol,nrow).
# For that, we need to allow for the __getitem__ method to be a staticmethod.
# cf. http://stackoverflow.com/questions/6187932/how-to-write-a-static-python-getitem-method
class MetaImage(type):
    def __getitem__(cls,t):
        """An obsolete syntax that treats Image as a dict indexed by type"""
        Image_dict = {
            numpy.int16 : ImageS,
            numpy.int32 : ImageI,
            numpy.float32 : ImageF,
            numpy.float64 : ImageD
        }
        return Image_dict[t]

class Image(object):
    """A class for storing image data along with the pixel scale or wcs information

    The Image class encapsulates all the relevant information about an image including a NumPy array
    for the pixel values, a bounding box, and some kind of WCS that converts between pixel
    coordinates and world coordinates.  The NumPy array may be constructed by the Image class
    itself, or an existing array can be provided by the user.

    This class creates shallow copies unless a deep copy is explicitly requested using the `copy`
    method.  The main reason for this is that it allows users to work directly with and modify
    subimages of larger images (for example, to successively draw many galaxies into one large
    image).  For other implications of this convention, see the description of initialization
    instructions below.

    There are 4 data types that the Image can use for the data values.  These are `numpy.int16`,
    `numpy.int32`, `numpy.float32`, and `numpy.float64`.  If you are constructing a new Image from
    scratch, the default is `numpy.float32`, but you can specify one of the other data types.

    Initialization
    --------------

    There are several ways to construct an Image:

        Image(ncol, nrow, dtype=numpy.float32, init_value=0, ...)

                This constructs a new image, allocating memory for the pixel values according to
                the number of columns and rows.  You can specify the data type as `dtype` if you
                want.  The default is `numpy.float32` if you don't specify it.  You can also
                optionally provide an initial value for the pixels, which defaults to 0.

        Image(bounds, dtype=numpy.float32, init_value=0, ...)

                This constructs a new image, allocating memory for the pixel values according to a
                given bounds object.  The bounds should be a BoundsI instance.  You can specify the
                data type as `dtype` if you want.  The default is `numpy.float32` if you don't
                specify it.  You can also optionally provide an initial value for the pixels, which
                defaults to 0.

        Image(array, xmin=1, ymin=1, make_const=False, ...)

                This views an existing NumPy array as an Image, with updates to the array or Image
                being affecting the other object unless `Image(array.copy(), ...)` is used.  The
                dtype is taken from `array.dtype`, which must be one of the allowed types listed
                above.  You can also optionally set the origin `(xmin, ymin)` if you want it to be
                something other than (1,1).  You can also optionally force the Image to be read-only
                with `make_const=True`, though if the original NumPy array is modified then the
                contents of `Image.array` will change.

        Image(image, dtype=dtype)

                This creates a shallow copy of an Image, possibly changing the type.  e.g.

                    >>> image_float = galsim.Image(64, 64) # default dtype=numpy.float32
                    >>> image_double = galsim.Image(image_float, dtype=numpy.float64)

                To get a deep copy, use the `copy` method rather than the `Image` constructor.

    You can specify the `ncol`, `nrow`, `bounds`, `array`, or `image`  parameters by keyword
    argument if you want, or you can pass them as simple arg as shown aboves, and the constructor
    will figure out what they are.

    The other keyword arguments (shown as ... above) relate to the conversion between sky
    coordinates, which is how all the GalSim objects are defined, and the pixel coordinates.
    There are three options for this:

        scale       You can optionally specify a pixel scale to use.  This would normally have
                    units arcsec/pixel, but it doesn't have to be arcsec.  If you want to
                    use different units for the physical scale of your galsim objects, then
                    the same unit would be used here.
        wcs         A WCS object that provides a non-trivial mapping between sky units and
                    pixel units.  The `scale` parameter is equivalent to `wcs=PixelScale(scale)`.
                    But there are a number of more complicated options.  See the WCS class
                    for more details.
        None        If you do not provide either of the above, then the conversion is undefined.
                    When drawing onto such an image, a suitable pixel scale will be automatically
                    set according to the Nyquist scale of the object being drawn.


    Attributes
    ----------

    After construction, you can set or change the scale or wcs with

        >>> image.scale = new_scale
        >>> image.wcs = new_wcs

    Note that `image.scale` will only work if the WCS is a PixelScale.  Once you set the
    wcs to be something non-trivial, then you must interact with it via the `wcs` attribute.
    The `image.scale` syntax will raise an exception.

    There are also two read-only attributes:

        >>> image.bounds
        >>> image.array

    The `array` attribute is a NumPy array of the Image's pixels.  The individual elements in the
    array attribute are accessed as `image.array[y,x]`, matching the standard NumPy convention,
    while the Image class's own accessor uses `(x,y)`.


    Methods
    -------

        view        Return a view of the image.
        subImage    Return a view of a portion of the full image.
        shift       Shift the origin of the image by (dx,dy).
        setCenter   Set a new position for the center of the image.
        setOrigin   Set a new position for the origin (x,y) = (0,0) of the image.
        im(x,y)     Get the value of a single pixel.
        setValue    Set the value of a single pixel.
        resize      Resize the image to have a new bounds.
        fill        Fill the image with the same value in all pixels.
        setZero     Fill the image with zeros.
        invertSelf  Convert each value x to 1/x.
        copy        Return a deep copy of the image.

    See their doc strings for more details.

    """
    __metaclass__ = MetaImage
    cpp_valid_dtypes = _galsim.ImageView.keys()
    alias_dtypes = {
        int : numpy.int32,          # So that user gets what they would expect
        float : numpy.float64,      # if using dtype=int or float
    }
    # Note: Numpy uses int64 for int on 64 bit machines.  We don't implement int64 at all,
    # so we cannot follow this pattern.  If this becomes too confusing, we might need to
    # add an ImageL class that uses int64.  Hard to imagine a use case where this would
    # be required though...
    valid_dtypes = cpp_valid_dtypes + alias_dtypes.keys()

    def __init__(self, *args, **kwargs):
        import numpy

        # Parse the args, kwargs
        ncol = None
        nrow = None
        bounds = None
        array = None
        image = None
        if len(args) > 2:
            raise TypeError("Error, too many unnamed arguments to Image constructor")
        elif len(args) == 2:
            ncol = args[0]
            nrow = args[1]
        elif len(args) == 1:
            if isinstance(args[0], numpy.ndarray):
                array = args[0]
                xmin = kwargs.pop('xmin',1)
                ymin = kwargs.pop('ymin',1)
                make_const = kwargs.pop('make_const',False)
            elif isinstance(args[0], galsim.BoundsI):
                bounds = args[0]
            else:
                image = args[0]
        else:
            if 'array' in kwargs:
                array = kwargs.pop('array')
                xmin = kwargs.pop('xmin',1)
                ymin = kwargs.pop('ymin',1)
                make_const = kwargs.pop('make_const',False)
            elif 'bounds' in kwargs:
                bounds = kwargs.pop('bounds')
            elif 'image' in kwargs:
                image = kwargs.pop('image')
            else:
                ncol = kwargs.pop('ncol',None)
                nrow = kwargs.pop('nrow',None)

        # Pop off the other valid kwargs:
        dtype = kwargs.pop('dtype', None)
        init_value = kwargs.pop('init_value', None)
        scale = kwargs.pop('scale', None)
        wcs = kwargs.pop('wcs', None)

        # Check that we got them all
        if kwargs:
            raise TypeError("Image constructor got unexpected keyword arguments: %s",kwargs)

        # Figure out what dtype we want:
        if dtype in Image.alias_dtypes: dtype = Image.alias_dtypes[dtype]
        if dtype is not None and dtype not in Image.valid_dtypes:
            raise ValueError("dtype must be one of "+str(Image.valid_dtypes)+
                             ".  Instead got "+str(dtype))
        if array is not None:
            if array.dtype.type not in Image.cpp_valid_dtypes and dtype is None:
                raise ValueError("array's dtype.type must be one of "+str(Image.cpp_valid_dtypes)+
                                 ".  Instead got "+str(array.dtype.type)+".  Or can set "+
                                 "dtype explicitly.")
            if dtype is not None and dtype != array.dtype.type:
                array = array.astype(dtype)
            # Be careful here: we have to watch out for little-endian / big-endian issues.
            # The path of least resistance is to check whether the array.dtype is equal to the
            # native one (using the dtype.isnative flag), and if not, make a new array that has a
            # type equal to the same one but with the appropriate endian-ness.
            if not array.dtype.isnative:
                array = array.astype(array.dtype.newbyteorder('='))
            self.dtype = array.dtype.type
        elif dtype is not None:
            self.dtype = dtype
        elif image is not None:
            self.dtype = image.array.dtype.type
        else:
            self.dtype = numpy.float32

        # Construct the image attribute
        if (ncol is not None or nrow is not None):
            if bounds is not None:
                raise TypeError("Cannot specify both ncol/nrow and bounds")
            if array is not None:
                raise TypeError("Cannot specify both ncol/nrow and array")
            if image is not None:
                raise TypeError("Cannot specify both ncol/nrow and image")
            if ncol is None or nrow is None:
                raise TypeError("Both nrow and ncol must be provided")
            try:
                ncol = int(ncol)
                nrow = int(nrow)
            except:
                raise TypeError("Cannot parse ncol, nrow as integers")
            self.image = _galsim.ImageAlloc[self.dtype](ncol, nrow)
            if init_value is not None:
                self.image.fill(init_value)
        elif bounds is not None:
            if array is not None:
                raise TypeError("Cannot specify both bounds and array")
            if image is not None:
                raise TypeError("Cannot specify both bounds and image")
            if not isinstance(bounds, galsim.BoundsI):
                raise TypeError("bounds must be a galsim.BoundsI instance")
            self.image = _galsim.ImageAlloc[self.dtype](bounds)
            if init_value is not None:
                self.image.fill(init_value)
        elif array is not None:
            if image is not None:
                raise TypeError("Cannot specify both array and image")
            if not isinstance(array, numpy.ndarray):
                raise TypeError("array must be a numpy.ndarray instance")
            if make_const:
                self.image = _galsim.ConstImageView[self.dtype](array, xmin, ymin)
            else:
                self.image = _galsim.ImageView[self.dtype](array, xmin, ymin)
            if init_value is not None:
                raise TypeError("Cannot specify init_value with array")
        elif image is not None:
            if isinstance(image, Image):
                image = image.image
            self.image = None
            for im_dtype in Image.cpp_valid_dtypes:
                if ( isinstance(image,_galsim.ImageAlloc[im_dtype]) or
                     isinstance(image,_galsim.ImageView[im_dtype]) or
                     isinstance(image,_galsim.ConstImageView[im_dtype]) ):
                    if dtype is not None and im_dtype != dtype:
                        # Allow dtype to force a retyping of the provided image
                        # e.g. im = ImageF(...)
                        #      im2 = ImageD(im)
                        self.image = _galsim.ImageAlloc[dtype](image)
                    else:
                        self.image = image
                    break
            if self.image is None:
                # Then never found the dtype above:
                raise TypeError("image must be an Image or BaseImage type")
            if init_value is not None:
                raise TypeError("Cannot specify init_value with image")
        else:
            self.image = _galsim.ImageAlloc[self.dtype]()
            if init_value is not None:
                raise TypeError("Cannot specify init_value without setting an initial size")

        # Construct the wcs attribute
        if scale is not None:
            if wcs is not None:
                raise TypeError("Cannot provide both scale and wcs to Image constructor")
            self.wcs = galsim.PixelScale(scale)
        else:
            if wcs is not None and not isinstance(wcs,galsim.BaseWCS):
                raise TypeError("wcs parameters must be a galsim.BaseWCS instance")
            self.wcs = wcs

    # bounds and array are really properties which pass the request to the image
    @property
    def bounds(self): return self.image.bounds
    @property
    def array(self): return self.image.array

    # Allow scale to work as a PixelScale wcs.
    @property
    def scale(self):
        if self.wcs:
            if self.wcs.isPixelScale():
                return self.wcs.scale
            else:
                raise TypeError("image.wcs is not a simple PixelScale; scale is undefined.")
        else:
            return None

    @scale.setter
    def scale(self, value):
        if self.wcs is not None and not self.wcs.isPixelScale():
            raise TypeError("image.wcs is not a simple PixelScale; scale is undefined.")
        else:
            self.wcs = galsim.PixelScale(value)

    # Convenience functions
    @property
    def xmin(self): return self.image.bounds.xmin
    @property
    def xmax(self): return self.image.bounds.xmax
    @property
    def ymin(self): return self.image.bounds.ymin
    @property
    def ymax(self): return self.image.bounds.ymax
    def getXMin(self): return self.image.getXMin()
    def getXMax(self): return self.image.getXMax()
    def getYMin(self): return self.image.getYMin()
    def getYMax(self): return self.image.getYMax()
    def getBounds(self): return self.image.getBounds()

    def copy(self):
        return Image(image=self.image.copy(), wcs=self.wcs)

    def resize(self, bounds):
        """Resize the image to have a new bounds (must be a BoundsI instance)

        Note that the resized image will have uninitialized data.  If you want to preserve
        the existing data values, you should either use `subImage` (if you want a smaller
        portion of the current Image) or make a new Image and copy over the current values
        into a portion of the new image (if you are resizing to a larger Image).
        """
        if not isinstance(bounds, galsim.BoundsI):
            raise TypeError("bounds must be a galsim.BoundsI instance")
        try:
            self.image.resize(bounds)
        except:
            # if the image wasn't an ImageAlloc, then above won't work.  So just make it one.
            self.image = _galsim.ImageAlloc[self.dtype](bounds)

    def subImage(self, bounds):
        """Return a view of a portion of the full image
        """
        if not isinstance(bounds, galsim.BoundsI):
            raise TypeError("bounds must be a galsim.BoundsI instance")
        subimage = self.image.subImage(bounds)
        # NB. The wcs is still accurate, since the sub-image uses the same (x,y) values
        # as the original image did for those pixels.  It's only once you recenter or
        # reorigin that you need to update the wcs.  So that's taken care of in im.shift.
        return Image(image=subimage, wcs=self.wcs)

    def __getitem__(self, bounds):
        """Return a view of a portion of the full image
        """
        return self.subImage(bounds)

    def __setitem__(self, bounds, rhs):
        """Set a portion of the full image to the values in another image
        """
        self.subImage(bounds).image.copyFrom(rhs.image)

    def copyFrom(self, rhs):
        """Copy the contents of another image
        """
        self.image.copyFrom(rhs.image)

    def view(self, make_const=False):
        """Make a view of this image, which lets you change the scale, wcs, origin, etc.
        but view the same underlying data as the original image.

        You can make this a const view with the `make_const` parameter.
        """
        if make_const:
            return Image(image=_galsim.ConstImageView[self.dtype](self.image.view()),
                         wcs=self.wcs)
        else:
            return Image(image=self.image.view(), wcs=self.wcs)

    def shift(self, *args, **kwargs):
        """Shift the pixel coordinates by some (integral) dx,dy.

        The arguments here may be either (dx, dy) or a PositionI instance.
        Or you can provide dx, dy as named kwargs.
        """
        delta = galsim.utilities.parse_pos_args(args, kwargs, 'dx', 'dy', integer=True)
        self._shift(delta)

    def _shift(self, delta):
        # The parse_pos_args function is a bit slow, so go directly to this point when we
        # call shift from setCenter or setOrigin.
        if delta.x != 0 or delta.y != 0:
            self.image.shift(delta)
            if self.wcs is not None:
                self.wcs = self.wcs.withOrigin(delta)

    def setCenter(self, *args, **kwargs):
        """Set the center of the image to the given (integral) (xcen, ycen)

        The arguments here may be either (xcen, ycen) or a PositionI instance.
        Or you can provide xcen, ycen as named kwargs.
        """
        cen = galsim.utilities.parse_pos_args(args, kwargs, 'xcen', 'ycen', integer=True)
        self._shift(cen - self.image.bounds.center())

    def setOrigin(self, *args, **kwargs):
        """Set the origin of the image to the given (integral) (x0, y0)

        The arguments here may be either (x0, y0) or a PositionI instance.
        Or you can provide x0, y0 as named kwargs.
        """
        origin = galsim.utilities.parse_pos_args(args, kwargs, 'x0', 'y0', integer=True)
        self._shift(origin - self.image.bounds.origin())

    def center(self):
        """Return the current nominal center of the image.  This is a PositionI instance,
        which means that for even-sized images, it won't quite be the true center, since
        the true center is between two pixels.

        e.g the nominal center of an image with bounds (1,32,1,32) will be (17, 17).
        """
        return self.bounds.center()

    def trueCenter(self):
        """Return the current true center of the image.  This is a PositionD instance,
        and it may be half-way between two pixels.

        e.g the true center of an image with bounds (1,32,1,32) will be (16.5, 16.5).
        """
        return self.bounds.trueCenter()

    def origin(self):
        """Return the origin of the image.  i.e. the position of the lower-left pixel.

        e.g the origin of an image with bounds (1,32,1,32) will be (1, 1).
        """
        return self.bounds.origin()

    def __call__(self, *args, **kwargs):
        """Get the pixel value at given position

        The arguments here may be either (x, y) or a PositionI instance.
        Or you can provide x, y as named kwargs.
        """
        pos = galsim.utilities.parse_pos_args(args, kwargs, 'x', 'y', integer=True)
        return self.image(pos.x, pos.y)

    def at(self, x, y):
        """This method is a synonym for im(x,y).  It is a bit faster than im(x,y), since GalSim
        does not have to parse the different options available for __call__.  (i.e. im(x,y) or
        im(pos) or im(x=x,y=y))
        """
        return self.image(x,y)

    def setValue(self, *args, **kwargs):
        """Set the pixel value at given position

        The arguments here may be either (x, y, value) or (pos, value) where pos is a PositionI.
        Or you can provide x, y, value as named kwargs.
        """
        pos, value = galsim.utilities.parse_pos_args(args, kwargs, 'x', 'y', integer=True,
                                                     others=['value'])
        self.image.setValue(pos.x, pos.y, value)

    def fill(self, value):
        """Set all pixel values to the given `value`
        """
        self.image.fill(value)

    def setZero(self):
        """Set all pixel values to zero.
        """
        self.image.setZero()

    def invertSelf(self):
        """Set all pixel values to their inverse: x -> 1/x.
        """
        self.image.invertSelf()


# These are essentially aliases for the regular Image with the correct dtype
def ImageS(*args, **kwargs):
    """Alias for galsim.Image(..., dtype=numpy.int16)
    """
    kwargs['dtype'] = numpy.int16
    return Image(*args, **kwargs)

def ImageI(*args, **kwargs):
    """Alias for galsim.Image(..., dtype=numpy.int32)
    """
    kwargs['dtype'] = numpy.int32
    return Image(*args, **kwargs)

def ImageF(*args, **kwargs):
    """Alias for galsim.Image(..., dtype=numpy.float32)
    """
    kwargs['dtype'] = numpy.float32
    return Image(*args, **kwargs)

def ImageD(*args, **kwargs):
    """Alias for galsim.Image(..., dtype=numpy.float64)
    """
    kwargs['dtype'] = numpy.float64
    return Image(*args, **kwargs)

def ImageViewS(*args, **kwargs):
    """Alias for galsim.Image(..., dtype=numpy.int16)
    """
    kwargs['dtype'] = numpy.int16
    return Image(*args, **kwargs)

def ImageViewI(*args, **kwargs):
    """Alias for galsim.Image(..., dtype=numpy.int32)
    """
    kwargs['dtype'] = numpy.int32
    return Image(*args, **kwargs)

def ImageViewF(*args, **kwargs):
    """Alias for galsim.Image(..., dtype=numpy.float32)
    """
    kwargs['dtype'] = numpy.float32
    return Image(*args, **kwargs)

def ImageViewD(*args, **kwargs):
    """Alias for galsim.Image(..., dtype=numpy.float64)
    """
    kwargs['dtype'] = numpy.float64
    return Image(*args, **kwargs)

def ConstImageViewS(*args, **kwargs):
    """An obsolete alias for galsim.Image(..., dtype=numpy.int16, make_const=True)
    """
    kwargs['dtype'] = numpy.int16
    kwargs['make_const'] = True
    return Image(*args, **kwargs)

def ConstImageViewI(*args, **kwargs):
    """An obsolete alias for galsim.Image(..., dtype=numpy.int32, make_const=True)
    """
    kwargs['dtype'] = numpy.int32
    kwargs['make_const'] = True
    return Image(*args, **kwargs)

def ConstImageViewF(*args, **kwargs):
    """An obsolete alias for galsim.Image(..., dtype=numpy.float32, make_const=True)
    """
    kwargs['dtype'] = numpy.float32
    kwargs['make_const'] = True
    return Image(*args, **kwargs)

def ConstImageViewD(*args, **kwargs):
    """An obsolete alias for galsim.Image(..., dtype=numpy.float64, make_const=True)
    """
    kwargs['dtype'] = numpy.float64
    kwargs['make_const'] = True
    return Image(*args, **kwargs)

ImageView = {
    numpy.int16 : ImageViewS,
    numpy.int32 : ImageViewI,
    numpy.float32 : ImageViewF,
    numpy.float64 : ImageViewD
}

ConstImageView = {
    numpy.int16 : ConstImageViewS,
    numpy.int32 : ConstImageViewI,
    numpy.float32 : ConstImageViewF,
    numpy.float64 : ConstImageViewD
}




################################################################################################
#
# Now we have to make some modifications to the C++ layer objects.  Mostly adding some
# arithemetic functions, so they work more intuitively.
#

def Image_setitem(self, key, value):
    self.subImage(key).copyFrom(value)

def Image_getitem(self, key):
    return self.subImage(key)

# Define a utility function to be used by the arithmetic functions below
def check_image_consistency(im1, im2):
    if ( isinstance(im2, Image) or
         type(im2) in _galsim.ImageAlloc.values() or
         type(im2) in _galsim.ImageView.values() or
         type(im2) in _galsim.ConstImageView.values()):
        if im1.array.shape != im2.array.shape:
            raise ValueError("Image shapes are inconsistent")

def Image_add(self, other):
    result = self.copy()
    result += other
    return result

def Image_iadd(self, other):
    check_image_consistency(self, other)
    try:
        self.array[:,:] += other.array
    except AttributeError:
        self.array[:,:] += other
    return self

def Image_sub(self, other):
    result = self.copy()
    result -= other
    return result

def Image_rsub(self, other):
    result = self.copy()
    result *= -1
    result += other
    return result

def Image_isub(self, other):
    check_image_consistency(self, other)
    try:
        self.array[:,:] -= other.array
    except AttributeError:
        self.array[:,:] -= other
    return self

def Image_mul(self, other):
    result = self.copy()
    result *= other
    return result

def Image_imul(self, other):
    check_image_consistency(self, other)
    try:
        self.array[:,:] *= other.array
    except AttributeError:
        self.array[:,:] *= other
    return self

def Image_div(self, other):
    result = self.copy()
    result /= other
    return result

def Image_rdiv(self, other):
    result = self.copy()
    result.invertSelf()
    result *= other
    return result

def Image_idiv(self, other):
    check_image_consistency(self, other)
    try:
        self.array[:,:] /= other.array
    except AttributeError:
        self.array[:,:] /= other
    return self

def Image_pow(self, other):
    result = self.copy()
    result **= other
    return result

def Image_ipow(self, other):
    if not isinstance(other, int) and not isinstance(other, float):
        raise TypeError("Can only raise an image to a float or int power!")
    self.array[:,:] **= other
    return self

def Image_neg(self, other):
    result = self.copy()
    result *= -1
    return result

# Define &, ^ and | only for integer-type images
def Image_and(self, other):
    result = self.copy()
    result &= other
    return result

def Image_iand(self, other):
    check_image_consistency(self, other)
    try:
        self.array[:,:] &= other.array
    except AttributeError:
        self.array[:,:] &= other
    return self

def Image_xor(self, other):
    result = self.copy()
    result ^= other
    return result

def Image_ixor(self, other):
    check_image_consistency(self, other)
    try:
        self.array[:,:] ^= other.array
    except AttributeError:
        self.array[:,:] ^= other
    return self

def Image_or(self, other):
    result = self.copy()
    result |= other
    return result

def Image_ior(self, other):
    check_image_consistency(self, other)
    try:
        self.array[:,:] |= other.array
    except AttributeError:
        self.array[:,:] |= other
    return self

def Image_copy(self):
    # self can be an ImageAlloc or an ImageView, but the return type needs to be an ImageAlloc.
    # So use the array.dtype.type attribute to get the type of the underlying data,
    # which in turn can be used to index our ImageAlloc dictionary:
    return _galsim.ImageAlloc[self.array.dtype.type](self)

# Some functions to enable pickling of images
def ImageView_getinitargs(self):
    return self.array, self.xmin, self.ymin

# An image is really pickled as an ImageView
def ImageAlloc_getstate(self):
    return self.array, self.xmin, self.ymin

def ImageAlloc_setstate(self, args):
    self_type = args[0].dtype.type
    self.__class__ = _galsim.ImageView[self_type]
    self.__init__(*args)

# inject the arithmetic operators as methods of the Image class:
Image.__add__ = Image_add
Image.__radd__ = Image_add
Image.__iadd__ = Image_iadd
Image.__sub__ = Image_sub
Image.__rsub__ = Image_rsub
Image.__isub__ = Image_isub
Image.__mul__ = Image_mul
Image.__rmul__ = Image_mul
Image.__imul__ = Image_imul
Image.__div__ = Image_div
Image.__rdiv__ = Image_div
Image.__truediv__ = Image_div
Image.__rtruediv__ = Image_rdiv
Image.__idiv__ = Image_idiv
Image.__itruediv__ = Image_idiv
Image.__ipow__ = Image_ipow
Image.__pow__ = Image_pow
Image.__neg__ = Image_neg
Image.__and__ = Image_and
Image.__xor__ = Image_xor
Image.__or__ = Image_or
Image.__iand__ = Image_iand
Image.__ixor__ = Image_ixor
Image.__ior__ = Image_ior

# inject these as methods of ImageAlloc classes
for Class in _galsim.ImageAlloc.itervalues():
    Class.__setitem__ = Image_setitem
    Class.__getitem__ = Image_getitem
    Class.__add__ = Image_add
    Class.__radd__ = Image_add
    Class.__iadd__ = Image_iadd
    Class.__sub__ = Image_sub
    Class.__rsub__ = Image_rsub
    Class.__isub__ = Image_isub
    Class.__mul__ = Image_mul
    Class.__rmul__ = Image_mul
    Class.__imul__ = Image_imul
    Class.__div__ = Image_div
    Class.__rdiv__ = Image_div
    Class.__truediv__ = Image_div
    Class.__rtruediv__ = Image_rdiv
    Class.__idiv__ = Image_idiv
    Class.__itruediv__ = Image_idiv
    Class.__ipow__ = Image_ipow
    Class.__neg__ = Image_neg
    Class.__pow__ = Image_pow
    Class.copy = Image_copy
    Class.__getstate_manages_dict__ = 1
    Class.__getstate__ = ImageAlloc_getstate
    Class.__setstate__ = ImageAlloc_setstate

for Class in _galsim.ImageView.itervalues():
    Class.__setitem__ = Image_setitem
    Class.__getitem__ = Image_getitem
    Class.__add__ = Image_add
    Class.__radd__ = Image_add
    Class.__iadd__ = Image_iadd
    Class.__sub__ = Image_sub
    Class.__rsub__ = Image_rsub
    Class.__isub__ = Image_isub
    Class.__mul__ = Image_mul
    Class.__rmul__ = Image_mul
    Class.__imul__ = Image_imul
    Class.__div__ = Image_div
    Class.__rdiv__ = Image_rdiv
    Class.__truediv__ = Image_div
    Class.__rtruediv__ = Image_rdiv
    Class.__idiv__ = Image_idiv
    Class.__itruediv__ = Image_idiv
    Class.__ipow__ = Image_ipow
    Class.__pow__ = Image_pow
    Class.__neg__ = Image_neg
    Class.copy = Image_copy
    Class.__getinitargs__ = ImageView_getinitargs

for Class in _galsim.ConstImageView.itervalues():
    Class.__getitem__ = Image_getitem
    Class.__add__ = Image_add
    Class.__radd__ = Image_add
    Class.__sub__ = Image_sub
    Class.__rsub__ = Image_rsub
    Class.__mul__ = Image_mul
    Class.__rmul__ = Image_mul
    Class.__div__ = Image_div
    Class.__rdiv__ = Image_rdiv
    Class.__truediv__ = Image_div
    Class.__rtruediv__ = Image_rdiv
    Class.__pow__ = Image_pow
    Class.__neg__ = Image_neg
    Class.copy = Image_copy
    Class.__getinitargs__ = ImageView_getinitargs

for int_type in [ numpy.int16, numpy.int32 ]:
    for Class in [ _galsim.ImageAlloc[int_type], _galsim.ImageView[int_type],
                   _galsim.ConstImageView[int_type] ]:
        Class.__and__ = Image_and
        Class.__xor__ = Image_xor
        Class.__or__ = Image_or
    for Class in [ _galsim.ImageAlloc[int_type], _galsim.ImageView[int_type] ]:
        Class.__iand__ = Image_iand
        Class.__ixor__ = Image_ixor
        Class.__ior__ = Image_ior

del Class    # cleanup public namespace
