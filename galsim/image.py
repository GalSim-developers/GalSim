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
"""@file image.py
The Image class
"""

from __future__ import division
import numpy as np

from . import _galsim
from .position import PositionI, PositionD
from .bounds import BoundsI, BoundsD
from .wcs import BaseWCS, PixelScale, JacobianWCS
from . import utilities
from .errors import GalSimError, GalSimBoundsError, GalSimValueError, GalSimImmutableError
from .errors import GalSimUndefinedBoundsError, GalSimIncompatibleValuesError, convert_cpp_errors

# Sometimes (on 32-bit systems) there are two numpy.int32 types.  This can lead to some confusion
# when doing arithmetic with images.  So just make sure both of them point to ImageViewI in the
# _cpp_type dict.  One of them is what you get when you just write numpy.int32.  The other is
# what numpy decides an int16 + int32 is.
# For more information regarding this rather unexpected behaviour for numpy.int32 types, see
# the following (closed, marked "wontfix") ticket on the numpy issue tracker:
# http://projects.scipy.org/numpy/ticket/1246

alt_int32 = (np.array([0], dtype=np.int16) + np.array([0], dtype=np.int32)).dtype.type


class Image(object):
    """A class for storing image data along with the pixel scale or WCS information

    The Image class encapsulates all the relevant information about an image including a NumPy array
    for the pixel values, a bounding box, and some kind of WCS that converts between pixel
    coordinates and world coordinates.  The NumPy array may be constructed by the Image class
    itself, or an existing array can be provided by the user.

    This class creates shallow copies unless a deep copy is explicitly requested using the `copy`
    method.  The main reason for this is that it allows users to work directly with and modify
    subimages of larger images (for example, to successively draw many galaxies into one large
    image).  For other implications of this convention, see the description of initialization
    instructions below.

    In most applications with images, we will use (x,y) to refer to the coordinates.  We adopt
    the same meaning for these coordinates as most astronomy applications do: ds9, SAOImage,
    SExtractor, etc. all treat x as the column number and y as the row number.  However, this
    is different from the default convention used by numpy.  In numpy, the access is by
    [row_num,col_num], which means this is really [y,x] in terms of the normal x,y values.
    Users are typically insulated from this concern by the Image API, but if you access the
    numpy array directly via the `array` attribute, you will need to be careful about this
    difference.

    There are 6 data types that the Image can use for the data values.  These are `numpy.uint16`,
    `numpy.uint16`, `numpy.int16`, `numpy.int32`, `numpy.float32`, and `numpy.float64`.
    If you are constructing a new Image from scratch, the default is `numpy.float32`, but you
    can specify one of the other data types.

    Initialization
    --------------

    There are several ways to construct an Image:
    (Optional arguments are shown with their default values after the = sign.)

        Image(ncol, nrow, dtype=numpy.float32, init_value=0, xmin=1, ymin=1, ...)

                This constructs a new image, allocating memory for the pixel values according to
                the number of columns and rows.  You can specify the data type as `dtype` if you
                want.  The default is `numpy.float32` if you don't specify it.  You can also
                optionally provide an initial value for the pixels, which defaults to 0.
                The optional `xmin,ymin` allow you to specify the location of the lower-left
                pixel, which defaults to (1,1).  Reminder, with our convention for x,y coordinates
                described above, ncol is the number of pixels in the x direction, and nrow is the
                number of pixels in the y direction.

        Image(bounds, dtype=numpy.float32, init_value=0, ...)

                This constructs a new image, allocating memory for the pixel values according to a
                given bounds object.  The bounds should be a BoundsI instance.  You can specify the
                data type as `dtype` if you want.  The default is `numpy.float32` if you don't
                specify it.  You can also optionally provide an initial value for the pixels, which
                defaults to 0.

        Image(array, xmin=1, ymin=1, make_const=False, copy=False ...)

                This views an existing NumPy array as an Image, where updates to either the image
                or the original array will affect the other one.  The dtype is taken from
                `array.dtype`, which must be one of the allowed types listed above.  You can also
                optionally set the origin `xmin, ymin` if you want it to be something other than
                (1,1).

                You can also optionally force the Image to be read-only with `make_const=True`,
                though if the original NumPy array is modified then the contents of `Image.array`
                will change.

                If you want to make a copy of the input array, rather than just view the existing
                array, you can force a copy with

                    >>> image = galsim.Image(array, copy=True)

        Image(image, dtype=image.dtype, copy=True)

                This creates a copy of an Image, possibly changing the type.  e.g.

                    >>> image_float = galsim.Image(64, 64) # default dtype=numpy.float32
                    >>> image_double = galsim.Image(image_float, dtype=numpy.float64)

                You can see a list of valid values for dtype in `galsim.Image.valid_dtypes`.
                Without the `dtype` argument, this is equivalent to `image.copy()`, which makes
                a deep copy.  If you want a copy that shares data with the original, see
                the image.view() method.

                If you only want to enforce the image to have a given type and not make a copy
                if the array is already the correct type, you can use, e.g.

                    >>> image_double = galsim.Image(image, dtype=numpy.float64, copy=False)

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
    while the Image class's own accessor uses either `(x,y)` or `[x,y]`.

    That is, the following are equivalent:

        >>> ixy = image(x,y)
        >>> ixy = image[x,y]
        >>> ixy = image.array[y,x]
        >>> ixy = image.getValue(x,y)

    Similarly, for setting individual pixel values, the following are equivalent:

        >>> image[x,y] = new_ixy
        >>> image.array[y,x] = new_ixy
        >>> image.setValue(x,y,new_ixy)

    Methods
    -------

        view        Return a view of the image, possibly giving it a new scale or wcs.
        subImage    Return a view of a portion of the full image.
        wrap        Wrap the values in a image onto a given subimage and return the subimage.
        bin         Bin the image pixels in blocks of nx x ny pixels.
        subsample   Subdivide the image pixels into nx x ny sub-pixels.
        shift       Shift the origin of the image by (dx,dy).
        setCenter   Set a new position for the center of the image.
        setOrigin   Set a new position for the origin (x,y) = (0,0) of the image.
        getValue    Get the value of a single pixel.
        setValue    Set the value of a single pixel.
        addValue    Add to the value of a single pixel.
        resize      Resize the image to have a new bounds.
        fill        Fill the image with the same value in all pixels.
        setZero     Fill the image with zeros.
        invertSelf  Convert each value x to 1/x.
        copy        Return a deep copy of the image.

    See their doc strings for more details.

    """

    _cpp_type = { np.uint16 : _galsim.ImageViewUS,
                  np.uint32 : _galsim.ImageViewUI,
                  np.int16 : _galsim.ImageViewS,
                  np.int32 : _galsim.ImageViewI,
                  np.float32 : _galsim.ImageViewF,
                  np.float64 : _galsim.ImageViewD,
                  np.complex64 : _galsim.ImageViewCF,
                  np.complex128 : _galsim.ImageViewCD,
                }
    _cpp_valid_dtypes = list(_cpp_type.keys())

    _alias_dtypes = {
        int : np.int32,          # So that user gets what they would expect
        float : np.float64,      # if using dtype=int or float or complex
        complex : np.complex128,
        np.int64 : np.int32,          # Not equivalent, but will convert
    }
    # Note: Numpy uses int64 for int on 64 bit machines.  We don't implement int64 at all,
    # so we cannot quite match up to the numpy convention for dtype=int.  e.g. via
    #     int : numpy.zeros(1,dtype=int).dtype.type
    # If this becomes too confusing, we might need to add an ImageL class that uses int64.
    # Hard to imagine a use case where this would be required though...

    # This one is in the public API.  (No leading underscore.)
    valid_dtypes = _cpp_valid_dtypes + list(_alias_dtypes.keys())

    def __init__(self, *args, **kwargs):
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
            xmin = kwargs.pop('xmin',1)
            ymin = kwargs.pop('ymin',1)
        elif len(args) == 1:
            if isinstance(args[0], np.ndarray):
                array = args[0]
                array, xmin, ymin = self._get_xmin_ymin(array, kwargs)
                make_const = kwargs.pop('make_const',False)
            elif isinstance(args[0], BoundsI):
                bounds = args[0]
            elif isinstance(args[0], (list, tuple)):
                array = np.array(args[0])
                array, xmin, ymin = self._get_xmin_ymin(array, kwargs)
                make_const = kwargs.pop('make_const',False)
            elif isinstance(args[0], Image):
                image = args[0]
            else:
                raise TypeError("Unable to parse %s as an array, bounds, or image."%args[0])
        else:
            if 'array' in kwargs:
                array = kwargs.pop('array')
                array, xmin, ymin = self._get_xmin_ymin(array, kwargs)
                make_const = kwargs.pop('make_const',False)
            elif 'bounds' in kwargs:
                bounds = kwargs.pop('bounds')
            elif 'image' in kwargs:
                image = kwargs.pop('image')
            else:
                ncol = kwargs.pop('ncol',None)
                nrow = kwargs.pop('nrow',None)
                xmin = kwargs.pop('xmin',1)
                ymin = kwargs.pop('ymin',1)

        # Pop off the other valid kwargs:
        dtype = kwargs.pop('dtype', None)
        init_value = kwargs.pop('init_value', None)
        scale = kwargs.pop('scale', None)
        wcs = kwargs.pop('wcs', None)
        copy = kwargs.pop('copy', None)

        # Check that we got them all
        if kwargs:
            raise TypeError("Image constructor got unexpected keyword arguments: %s",kwargs)

        # Figure out what dtype we want:
        dtype = Image._alias_dtypes.get(dtype,dtype)
        if dtype is not None and dtype not in Image.valid_dtypes:
            raise GalSimValueError("Invlid dtype.", dtype, Image.valid_dtypes)
        if array is not None:
            if copy is None: copy = False
            if dtype is None:
                dtype = array.dtype.type
                if dtype in Image._alias_dtypes:
                    dtype = Image._alias_dtypes[dtype]
                    array = array.astype(dtype, copy=copy)
                elif dtype not in Image._cpp_valid_dtypes:
                    raise GalSimValueError("Invalid dtype of provided array.", array.dtype,
                                           Image._cpp_valid_dtypes)
                elif copy:
                    array = np.array(array)
            else:
                array = array.astype(dtype, copy=copy)
            # Be careful here: we have to watch out for little-endian / big-endian issues.
            # The path of least resistance is to check whether the array.dtype is equal to the
            # native one (using the dtype.isnative flag), and if not, make a new array that has a
            # type equal to the same one but with the appropriate endian-ness.
            if not array.dtype.isnative:
                array = array.astype(array.dtype.newbyteorder('='))
            self._dtype = array.dtype.type
        elif dtype is not None:
            self._dtype = dtype
        else:
            self._dtype = np.float32

        # Construct the image attribute
        if (ncol is not None or nrow is not None):
            if ncol is None or nrow is None:
                raise GalSimIncompatibleValuesError(
                    "Both nrow and ncol must be provided", ncol=ncol, nrow=nrow)
            if ncol != int(ncol) or nrow != int(nrow):
                raise TypeError("nrow, ncol must be integers")
            ncol = int(ncol)
            nrow = int(nrow)
            self._array = self._make_empty(shape=(nrow,ncol), dtype=self._dtype)
            self._bounds = BoundsI(xmin, xmin+ncol-1, ymin, ymin+nrow-1)
            if init_value:
                self.fill(init_value)
        elif bounds is not None:
            if not isinstance(bounds, BoundsI):
                raise TypeError("bounds must be a galsim.BoundsI instance")
            self._array = self._make_empty(bounds.numpyShape(), dtype=self._dtype)
            self._bounds = bounds
            if init_value:
                self.fill(init_value)
        elif array is not None:
            self._array = array.view()
            nrow,ncol = array.shape
            self._bounds = BoundsI(xmin, xmin+ncol-1, ymin, ymin+nrow-1)
            if make_const or not array.flags.writeable:
                self._array.flags.writeable = False
            if init_value is not None:
                raise GalSimIncompatibleValuesError(
                    "Cannot specify init_value with array", init_value=init_value, array=array)
        elif image is not None:
            if not isinstance(image, Image):
                raise TypeError("image must be an Image")
            if init_value is not None:
                raise GalSimIncompatibleValuesError(
                    "Cannot specify init_value with image", init_value=init_value, image=image)
            if wcs is None and scale is None:
                wcs = image.wcs
            self._bounds = image.bounds
            if dtype is None:
                self._dtype = image.dtype
            else:
                # Allow dtype to force a retyping of the provided image
                # e.g. im = ImageF(...)
                #      im2 = ImageD(im)
                self._dtype = dtype
            if copy is False:
                self._array = image.array.astype(self._dtype, copy=False)
            else:
                self._array = self._make_empty(shape=image.bounds.numpyShape(), dtype=self._dtype)
                self._array[:,:] = image.array[:,:]
        else:
            self._array = np.zeros(shape=(1,1), dtype=self._dtype)
            self._bounds = BoundsI()
            if init_value is not None:
                raise GalSimIncompatibleValuesError(
                    "Cannot specify init_value without setting an initial size",
                    init_value=init_value, ncol=ncol, nrow=nrow, bounds=bounds)

        # Construct the wcs attribute
        if scale is not None:
            if wcs is not None:
                raise GalSimIncompatibleValuesError(
                    "Cannot provide both scale and wcs to Image constructor", wcs=wcs, scale=scale)
            self.wcs = PixelScale(float(scale))
        else:
            if wcs is not None and not isinstance(wcs,BaseWCS):
                raise TypeError("wcs parameters must be a galsim.BaseWCS instance")
            self.wcs = wcs

    @staticmethod
    def _get_xmin_ymin(array, kwargs):
        """A helper function for parsing xmin, ymin, bounds options with a given array
        """
        if not isinstance(array, np.ndarray):
            raise TypeError("array must be a numpy.ndarray instance")
        xmin = kwargs.pop('xmin',1)
        ymin = kwargs.pop('ymin',1)
        if 'bounds' in kwargs:
            b = kwargs.pop('bounds')
            if not isinstance(b, BoundsI):
                raise TypeError("bounds must be a galsim.BoundsI instance")
            if b.xmax-b.xmin+1 != array.shape[1]:
                raise GalSimIncompatibleValuesError(
                    "Shape of array is inconsistent with provided bounds", array=array, bounds=b)
            if b.ymax-b.ymin+1 != array.shape[0]:
                raise GalSimIncompatibleValuesError(
                    "Shape of array is inconsistent with provided bounds", array=array, bounds=b)
            if b.isDefined():
                xmin = b.xmin
                ymin = b.ymin
            else:
                # Indication that array is formally undefined, even though provided.
                if 'dtype' not in kwargs:
                    kwargs['dtype'] = array.dtype.type
                array = None
                xmin = None
                ymin = None
        elif array.shape[1] == 0:
            # Another way to indicate that we don't have a defined image.
            if 'dtype' not in kwargs:
                kwargs['dtype'] = array.dtype.type
            array = None
            xmin = None
            ymin = None
        return array, xmin, ymin

    def __repr__(self):
        s = 'galsim.Image(bounds=%r' % self.bounds
        if self.bounds.isDefined():
            s += ', array=\n%r' % self.array
        s += ', wcs=%r' % self.wcs
        if self.isconst:
            s += ', make_const=True'
        s += ')'
        return s

    def __str__(self):
        # Get the type name without the <type '...'> part.
        t = str(self.dtype).split("'")[1]
        if self.wcs is not None and self.wcs.isPixelScale():
            return 'galsim.Image(bounds=%s, scale=%s, dtype=%s)'%(self.bounds, self.scale, t)
        else:
            return 'galsim.Image(bounds=%s, wcs=%s, dtype=%s)'%(self.bounds, self.wcs, t)

    # Pickling almost works out of the box, but numpy arrays lose their non-writeable flag
    # when pickled, so make sure to set it to preserve const Images.
    def __getstate__(self):
        return self.__dict__, self.isconst

    def __setstate__(self, args):
        d, isconst = args
        self.__dict__ = d
        if isconst:
            self._array.flags.writeable = False

    # Read-only attributes:
    @property
    def dtype(self): return self._dtype
    @property
    def bounds(self): return self._bounds
    @property
    def array(self): return self._array
    @property
    def isconst(self): return self._array.flags.writeable == False
    @property
    def iscomplex(self): return self._array.dtype.kind == 'c'
    @property
    def isinteger(self): return self._array.dtype.kind in ('i','u')

    @property
    def iscontiguous(self):
        """Indicates whether each row of the image is contiguous in memory.

        Note: it is ok for the end of one row to not be contiguous with the start of the
        next row.  This just checks that each individual row has a stride of 1.
        """
        return self._array.strides[1]//self._array.itemsize == 1

    @property
    def _image(self):
        cls = self._cpp_type[self.dtype]
        return cls(self._array.ctypes.data,
                   self._array.strides[1]//self._array.itemsize,
                   self._array.strides[0]//self._array.itemsize,
                   self._bounds._b)

    # Allow scale to work as a PixelScale wcs.
    @property
    def scale(self):
        if self.wcs:
            if self.wcs.isPixelScale():
                return self.wcs.scale
            else:
                raise GalSimError("image.wcs is not a simple PixelScale; scale is undefined.")
        else:
            return None

    @scale.setter
    def scale(self, value):
        if self.wcs is not None and not self.wcs.isPixelScale():
            raise GalSimError("image.wcs is not a simple PixelScale; scale is undefined.")
        else:
            self.wcs = PixelScale(value)

    # Convenience functions
    @property
    def xmin(self): return self._bounds.xmin
    @property
    def xmax(self): return self._bounds.xmax
    @property
    def ymin(self): return self._bounds.ymin
    @property
    def ymax(self): return self._bounds.ymax

    @property
    def outer_bounds(self):
        """The bounds of the outer edge of the pixels.

        Equivalent to galsim.BoundsD(im.xmin-0.5, im.xmax+0.5, im.ymin-0.5, im.ymax+0.5)
        """
        return BoundsD(self.xmin-0.5, self.xmax+0.5, self.ymin-0.5, self.ymax+0.5)

    # real, imag for everything, even real images.
    @property
    def real(self):
        """Return the real part of an image.

        This is a property, not a function.  So im.real, not im.real().

        This works for real or complex.  For real images, it acts the same as view().
        """
        return _Image(self.array.real, self.bounds, self.wcs)

    @property
    def imag(self):
        """Return the imaginary part of an image.

        This is a property, not a function.  So im.imag, not im.imag().

        This works for real or complex.  For real images, the returned array is read-only and
        all elements are 0.
        """
        return _Image(self.array.imag, self.bounds, self.wcs)

    @property
    def conjugate(self):
        """Return the complex conjugate of an image.

        This works for real or complex.  For real images, it acts the same as view().

        Note that for complex images, this is not a conjugate view into the original image.
        So changing the original image does not change the conjugate (or vice versa).
        """
        return _Image(self.array.conjugate(), self.bounds, self.wcs)

    def copy(self):
        return _Image(self.array.copy(), self.bounds, self.wcs)

    def _make_empty(self, shape, dtype):
        """Helper function to make an empty numpy array of the given shape, making sure that
        the array is 16-btye aligned so it is usable by FFTW.
        """
        # cf. http://stackoverflow.com/questions/9895787/memory-alignment-for-fast-fft-in-python-using-shared-arrrays
        nbytes = shape[0] * shape[1] * np.dtype(dtype).itemsize
        if nbytes == 0:
            # Make degenerate images have 1 element.  Otherwise things get weird.
            return np.zeros(shape=(1,1), dtype=self._dtype)
        buf = np.zeros(nbytes + 16, dtype=np.uint8)
        start_index = -buf.ctypes.data % 16
        a = buf[start_index:start_index + nbytes].view(dtype).reshape(shape)
        #assert a.ctypes.data % 16 == 0
        return a

    def resize(self, bounds, wcs=None):
        """Resize the image to have a new bounds (must be a BoundsI instance)

        Note that the resized image will have uninitialized data.  If you want to preserve
        the existing data values, you should either use `subImage` (if you want a smaller
        portion of the current Image) or make a new Image and copy over the current values
        into a portion of the new image (if you are resizing to a larger Image).

        @param bounds   The new bounds to resize to.
        @param wcs      If provided, also update the wcs to the given value. [default: None,
                        which means keep the existing wcs]
        """
        if self.isconst:
            raise GalSimImmutableError("Cannot modify an immutable Image", self)
        if not isinstance(bounds, BoundsI):
            raise TypeError("bounds must be a galsim.BoundsI instance")
        self._array = self._make_empty(shape=bounds.numpyShape(), dtype=self.dtype)
        self._bounds = bounds
        if wcs is not None:
            self.wcs = wcs

    def subImage(self, bounds):
        """Return a view of a portion of the full image

        This is equivalent to self[bounds]
        """
        if not isinstance(bounds, BoundsI):
            raise TypeError("bounds must be a galsim.BoundsI instance")
        if not self.bounds.isDefined():
            raise GalSimUndefinedBoundsError("Attempt to access subImage of undefined image")
        if not self.bounds.includes(bounds):
            raise GalSimBoundsError("Attempt to access subImage not (fully) in image",
                                    bounds,self.bounds)
        i1 = bounds.ymin - self.ymin
        i2 = bounds.ymax - self.ymin + 1
        j1 = bounds.xmin - self.xmin
        j2 = bounds.xmax - self.xmin + 1
        subarray = self.array[i1:i2, j1:j2]
        # NB. The wcs is still accurate, since the sub-image uses the same (x,y) values
        # as the original image did for those pixels.  It's only once you recenter or
        # reorigin that you need to update the wcs.  So that's taken care of in im.shift.
        return _Image(subarray, bounds, self.wcs)

    def setSubImage(self, bounds, rhs):
        """Set a portion of the full image to the values in another image

        This is equivalent to self[bounds] = rhs
        """
        if self.isconst:
            raise GalSimImmutableError("Cannot modify the values of an immutable Image", self)
        self.subImage(bounds).copyFrom(rhs)

    def __getitem__(self, *args):
        """Return either a subimage or a single pixel value.

        For example,
            >>> subimage = im[galsim.BoundsI(3,7,3,7)]
            >>> value = im[galsim.PositionI(5,5)]
            >>> value = im[5,5]
        """
        if len(args) == 1:
            if isinstance(args[0], BoundsI):
                return self.subImage(*args)
            elif isinstance(args[0], PositionI):
                return self(*args)
            elif isinstance(args[0], tuple):
                return self.getValue(*args[0])
            else:
                raise TypeError("image[index] only accepts BoundsI or PositionI for the index")
        elif len(args) == 2:
            return self(*args)
        else:
            raise TypeError("image[..] requires either 1 or 2 args")

    def __setitem__(self, *args):
        """Set either a subimage or a single pixel to new values.

        For example,

            >>> im[galsim.BoundsI(3,7,3,7)] = im2
            >>> im[galsim.PositionI(5,5)] = 17.
            >>> im[5,5] = 17.
        """
        if len(args) == 2:
            if isinstance(args[0], BoundsI):
                self.setSubImage(*args)
            elif isinstance(args[0], PositionI):
                self.setValue(*args)
            elif isinstance(args[0], tuple):
                self.setValue(*args)
            else:
                raise TypeError("image[index] only accepts BoundsI or PositionI for the index")
        elif len(args) == 3:
            return self.setValue(*args)
        else:
            raise TypeError("image[..] requires either 1 or 2 args")

    def wrap(self, bounds, hermitian=False):
        """Wrap the values in a image onto a given subimage and return the subimage.

        This would typically be used on a k-space image where you initially draw a larger image
        than you want for the FFT and then wrap it onto a smaller subset.  This will cause
        aliasing of course, but this is often preferable to just using the smaller image
        without wrapping.

        For complex images of FFTs, one often only stores half the image plane with the
        implicit understanding that the function is Hermitian, so im(-x,-y) == im(x,y).conjugate().
        In this case, the wrapping needs to work slightly differently, so you can specify
        that your image is implicitly Hermitian with the `hermitian` argument.  Options are:

            hermitian=False  (default) Normal non-Hermitian image.
            hermitian='x'    Only x>=0 values are stored with x<0 values being implicitly Hermitian.
                             In this case im.bounds.xmin and bounds.xmin must be 0.
            hermitian='y'    Only y>=0 values are stored with y<0 values being implicitly Hermitian.
                             In this case im.bounds.ymin and bounds.ymin must be 0.

        Also, in the two Hermitian cases, the direction that is not implicitly Hermitian must be
        symmetric in the image's bounds.  The wrap bounds must be almost symmetric, but missing
        the most negative value.  For example,

            >>> N = 100
            >>> im_full = galsim.ImageCD(bounds=galsim.BoundsI(0,N/2,-N/2,N/2), scale=dk)
            >>> # ... fill with im[i,j] = FT(kx=i*dk, ky=j*dk)
            >>> N2 = 64
            >>> im_wrap = im_full.wrap(galsim.BoundsI(0,N/2,-N2/2,N2/2-1, hermitian='x')

        This sets up im_wrap to be the properly Hermitian version of the data appropriate for
        passing to an FFT.

        Note that this routine modifies the original image (and not just the subimage onto which
        it is wrapped), so if you want to keep the original pristine, you should call
        `wrapped_image = image.copy().wrap(bounds)`.

        @param bounds       The bounds of the subimage onto which to wrap the full image.
        @param hermitian    Whether the image is implicitly Hermitian and if so, whether it is the
                            x or y values that are not stored.  [default: False]

        @returns the subimage, image[bounds], after doing the wrapping.
        """
        if not isinstance(bounds, BoundsI):
            raise TypeError("bounds must be a galsim.BoundsI instance")
        # Get this at the start to check for invalid bounds and raise the exception before
        # possibly writing data past the edge of the image.
        ret = self.subImage(bounds)
        if not hermitian:
            with convert_cpp_errors():
                _galsim.wrapImage(self._image, bounds._b, False, False)
        elif hermitian == 'x':
            if self.bounds.xmin != 0:
                raise GalSimIncompatibleValuesError(
                    "hermitian == 'x' requires self.bounds.xmin == 0",
                    hermitian=hermitian, bounds=self.bounds)
            if bounds.xmin != 0:  # pragma: no cover (This is covered, but codecov says it's not.)
                raise GalSimIncompatibleValuesError(
                    "hermitian == 'x' requires bounds.xmin == 0",
                    hermitian=hermitian, bounds=bounds)
            with convert_cpp_errors():
                _galsim.wrapImage(self._image, bounds._b, True, False)
        elif hermitian == 'y':
            if self.bounds.ymin != 0:
                raise GalSimIncompatibleValuesError(
                    "hermitian == 'y' requires self.bounds.ymin == 0",
                    hermitian=hermitian, bounds=self.bounds)
            if bounds.ymin != 0:  # pragma: no cover (This is covered, but codecov says it's not.)
                raise GalSimIncompatibleValuesError(
                    "hermitian == 'y' requires bounds.ymin == 0",
                    hermitian=hermitian, bounds=bounds)
            with convert_cpp_errors():
                _galsim.wrapImage(self._image, bounds._b, False, True)
        else:
            raise GalSimValueError("Invalid value for hermitian", hermitian, (False, 'x', 'y'))
        return ret;

    def _wrap(self, bounds, hermx, hermy):
        """Essentially equivalent to Image.wrap(bounds, hermitian=='x', hermitian=='y'), but
        without some of the sanity checks that the regular function does.
        """
        ret = self.subImage(bounds)
        with convert_cpp_errors():
            _galsim.wrapImage(self._image, bounds._b, hermx, hermy)
        return ret

    def bin(self, nx, ny):
        """Bin the image pixels in blocks of nx x ny pixels.

        This returns a new image that is a binned version of the current image.
        Adjacent pixel values in nx x ny blocks are added together to produce the flux in each
        output pixel.

        If the current number of pixels in each direction is not a multiple of nx, ny, then the
        last pixel in each direction will be the sum of fewer than nx or ny pixels as needed.

        See also subsample, which is the opposite of this.

        If the wcs is a Jacobian (or simpler), the output image will have its wcs set properly.
        But if the wcs is more complicated, the output wcs would be fairly complicated to figure
        out properly, so we leave it as None.  The user should set it themselves if required.

        @param nx       The number of adjacent pixels in the x direction to add together into each
                        output pixel.
        @param ny       The number of adjacent pixels in the y direction to add together into each
                        output pixel.

        @returns a new Image
        """
        from .wcs import JacobianWCS
        ncol = self.xmax - self.xmin + 1
        nrow = self.ymax - self.ymin + 1
        nbins_x = (ncol-1) // nx + 1
        nbins_y = (nrow-1) // ny + 1
        nbins = nbins_x * nbins_y

        # target_bins just provides a number from 0..nbins for each target pixel
        target_bins = np.arange(nbins).reshape(nbins_y, nbins_x)
        # current_bins is the same number for each pixel in the current image.
        current_bins = np.repeat(np.repeat(target_bins, ny, axis=0), nx, axis=1)
        current_bins = current_bins[0:nrow, 0:ncol]

        # bincount with weights is a tricky way to do the sum over the bins
        target_ar = np.bincount(current_bins.ravel(), weights=self.array.ravel())
        target_ar = target_ar.reshape(target_bins.shape)

        if self.wcs is None or not self.wcs.isUniform():
            target_wcs = None
        else:
            if self.wcs.isPixelScale() and nx == ny:
                target_wcs = PixelScale(self.scale * nx)
            else:
                dudx, dudy, dvdx, dvdy = self.wcs.jacobian().getMatrix().ravel()
                dudx *= nx
                dvdx *= nx
                dudy *= ny
                dvdy *= ny
                target_wcs = JacobianWCS(dudx, dudy, dvdx, dvdy)

            # Set the origin so that corresponding image positions correspond to the same world_pos
            x0 = (self.wcs.origin.x - self.xmin + 0.5) / nx + 0.5
            y0 = (self.wcs.origin.y - self.ymin + 0.5) / ny + 0.5
            target_wcs = target_wcs.withOrigin(PositionD(x0,y0), self.wcs.world_origin)

        target_bounds = BoundsI(1, nbins_x, 1, nbins_y)

        return _Image(target_ar, target_bounds, target_wcs)

    def subsample(self, nx, ny, dtype=None):
        """Subdivide the image pixels into nx x ny sub-pixels.

        This returns a new image that is a subsampled version of the current image.
        Each pixel's flux is split (uniformly) into nx x ny smaller pixels.

        See also bin, which is the opposite of this.  Note that subsample(nx,ny) followed by
        bin(nx,ny) is essentially a no op.

        If the wcs is a Jacobian (or simpler), the output image will have its wcs set properly.
        But if the wcs is more complicated, the output wcs would be fairly complicated to figure
        out properly, so we leave it as None.  The user should set it themselves if required.

        @param nx       The number of sub-pixels in the x direction for each original pixel.
        @param ny       The number of sub-pixels in the y direction for each original pixel.
        @param dtype    Optionally provide a dtype for the return image. [default: None, which
                        means to use the same dtype as the original image]

        @returns a new Image
        """
        ncol = self.xmax - self.xmin + 1
        nrow = self.ymax - self.ymin + 1
        npix_x = ncol * nx
        npix_y = nrow * ny
        flux_factor = nx * ny

        target_ar = np.repeat(np.repeat(self.array, ny, axis=0), nx, axis=1)
        target_ar = target_ar.astype(dtype, copy=False)  # Cute. This is a no op if dtype=None
        target_ar /= flux_factor

        if self.wcs is None or not self.wcs.isUniform():
            target_wcs = None
        else:
            if self.wcs.isPixelScale() and nx == ny:
                target_wcs = PixelScale(self.scale / nx)
            else:
                dudx, dudy, dvdx, dvdy = self.wcs.jacobian().getMatrix().ravel()
                dudx /= nx
                dvdx /= nx
                dudy /= ny
                dvdy /= ny
                target_wcs = JacobianWCS(dudx, dudy, dvdx, dvdy)

            # Set the origin so that corresponding image positions correspond to the same world_pos
            x0 = (self.wcs.origin.x - self.xmin + 0.5) * nx + 0.5
            y0 = (self.wcs.origin.y - self.ymin + 0.5) * ny + 0.5
            target_wcs = target_wcs.withOrigin(PositionD(x0,y0), self.wcs.world_origin)

        target_bounds = BoundsI(1, npix_x, 1, npix_y)

        return _Image(target_ar, target_bounds, target_wcs)

    def calculate_fft(self):
        """Performs an FFT of an Image in real space to produce a k-space Image.

        Note: the image will be padded with zeros as needed to make an image with bounds that
        look like BoundsI(-N/2, N/2-1, -N/2, N/2-1).

        The input image must have a PixelScale wcs.  The output image will be complex (an ImageCF
        or ImageCD instance) and its scale will be 2pi / (N dx), where dx is the scale of the input
        image.

        @returns an Image instance with the k-space image.
        """
        if self.wcs is None:
            raise GalSimError("calculate_fft requires that the scale be set.")
        if not self.wcs.isPixelScale():
            raise GalSimError("calculate_fft requires that the image has a PixelScale wcs.")
        if not self.bounds.isDefined():
            raise GalSimUndefinedBoundsError(
                    "calculate_fft requires that the image have defined bounds.")

        No2 = max(-self.bounds.xmin, self.bounds.xmax+1, -self.bounds.ymin, self.bounds.ymax+1)

        full_bounds = BoundsI(-No2, No2-1, -No2, No2-1)
        if self.bounds == full_bounds:
            # Then the image is already in the shape we need.
            ximage = self
        else:
            # Then we pad out with zeros
            ximage = Image(full_bounds, dtype=self.dtype, init_value=0)
            ximage[self.bounds] = self[self.bounds]

        dx = self.scale
        # dk = 2pi / (N dk)
        dk = np.pi / (No2 * dx)

        out = Image(BoundsI(0,No2,-No2,No2-1), dtype=np.complex128, scale=dk)
        with convert_cpp_errors():
            _galsim.rfft(ximage._image, out._image, True, True)
        out *= dx*dx
        out.setOrigin(0,-No2)
        return out

    def calculate_inverse_fft(self):
        """Performs an inverse FFT of an Image in k-space to produce a real-space Image.

        The starting image is typically an ImageCD, although if the Fourier function is real valued,
        then you could get away with using an ImageD or ImageF.

        The image is assumed to be Hermitian.  In fact, only the portion with x >= 0 needs to
        be defined, with f(-x,-y) taken to be conj(f(x,y)).

        Note: the k-space image will be padded with zeros and/or wrapped as needed to make an
        image with bounds that look like BoundsI(0, N/2, -N/2, N/2-1).  If you are building a
        larger k-space image and then wrapping, you should wrap directly into an image of
        this shape.

        The input image must have a PixelScale wcs.  The output image will be real (an ImageD
        instance) and its scale will be 2pi / (N dk), where dk is the scale of the input image.

        @returns an ImageD instance with the real-space image.
        """
        if self.wcs is None:
            raise GalSimError("calculate_inverse_fft requires that the scale be set.")
        if not self.wcs.isPixelScale():
            raise GalSimError("calculate_inverse_fft requires that the image has a PixelScale wcs.")
        if not self.bounds.isDefined():
            raise GalSimUndefinedBoundsError("calculate_inverse_fft requires that the image have "
                                             "defined bounds.")
        if not self.bounds.includes(0,0):
            raise GalSimBoundsError("calculate_inverse_fft requires that the image includes (0,0)",
                                    PositionI(0,0), self.bounds)

        No2 = max(self.bounds.xmax, -self.bounds.ymin, self.bounds.ymax)

        target_bounds = BoundsI(0, No2, -No2, No2-1)
        if self.bounds == target_bounds:
            # Then the image is already in the shape we need.
            kimage = self
        else:
            # Then we can pad out with zeros and wrap to get this in the form we need.
            full_bounds = BoundsI(0, No2, -No2, No2)
            kimage = Image(full_bounds, dtype=self.dtype, init_value=0)
            posx_bounds = BoundsI(0, self.bounds.xmax, self.bounds.ymin, self.bounds.ymax)
            kimage[posx_bounds] = self[posx_bounds]
            kimage = kimage.wrap(target_bounds, hermitian = 'x')

        dk = self.scale
        # dx = 2pi / (N dk)
        dx = np.pi / (No2 * dk)

        # For the inverse, we need a bit of extra space for the fft.
        out_extra = Image(BoundsI(-No2,No2+1,-No2,No2-1), dtype=float, scale=dx)
        with convert_cpp_errors():
            _galsim.irfft(kimage._image, out_extra._image, True, True)
        # Now cut off the bit we don't need.
        out = out_extra.subImage(BoundsI(-No2,No2-1,-No2,No2-1))
        out *= (dk * No2 / np.pi)**2
        out.setCenter(0,0)
        return out

    @classmethod
    def good_fft_size(cls, input_size):
        """Round the given input size up to the next higher power of 2 or 3 times a power of 2.

        This rounds up to the next higher value that is either 2^k or 3*2^k.  If you are
        going to be performing FFTs on an image, these will tend to be faster at performing
        the FFT.
        """
        with convert_cpp_errors():
            return _galsim.goodFFTSize(int(input_size))

    def copyFrom(self, rhs):
        """Copy the contents of another image
        """
        if self.isconst:
            raise GalSimImmutableError("Cannot modify the values of an immutable Image", self)
        if not isinstance(rhs, Image):
            raise TypeError("Trying to copyFrom a non-image")
        if self.bounds.numpyShape() != rhs.bounds.numpyShape():
            raise GalSimIncompatibleValuesError(
                "Trying to copy images that are not the same shape", self_image=self, rhs=rhs)
        self._array[:,:] = rhs.array[:,:]

    def view(self, scale=None, wcs=None, origin=None, center=None, make_const=False):
        """Make a view of this image, which lets you change the scale, wcs, origin, etc.
        but view the same underlying data as the original image.

        If you do not provide either `scale` or `wcs`, the view will keep the same wcs
        as the current Image object.

        @param scale        If provided, use this as the pixel scale for the image. [default: None]
        @param wcs          If provided, use this as the wcs for the image. [default: None]
        @param origin       If profided, use this as the origin position of the view.
                            [default: None]
        @param center       If profided, use this as the center position of the view.
                            [default: None]
        @param make_const   Make the view's data array immutable. [default: False]
        """
        if origin is not None and center is not None:
            raise GalSimIncompatibleValuesError(
                "Cannot provide both center and origin", center=center, origin=origin)

        if scale is not None:
            if wcs is not None:
                raise GalSimIncompatibleValuesError(
                    "Cannot provide both scale and wcs", scale=scale, wcs=wcs)
            wcs = PixelScale(scale)
        elif wcs is not None:
            if not isinstance(wcs,BaseWCS):
                raise TypeError("wcs parameters must be a galsim.BaseWCS instance")
        else:
            wcs = self.wcs

        if not self.bounds.isDefined():
            return Image(wcs=wcs, dtype=self.dtype)

        if make_const:
            array = self.array.view()
            array.flags.writeable = False
            ret = _Image(array, self.bounds, wcs)
        else:
            ret = _Image(self.array, self.bounds, wcs)

        if origin is not None:
            ret.setOrigin(origin)
        elif center is not None:
            ret.setCenter(center)

        return ret

    def _view(self):
        """Equivalent to im.view(), but without some of the sanity checks and extra options.
        """
        return _Image(self.array.view(), self.bounds, self.wcs)

    def shift(self, *args, **kwargs):
        """Shift the pixel coordinates by some (integral) dx,dy.

        The arguments here may be either (dx, dy) or a PositionI instance.
        Or you can provide dx, dy as named kwargs.

        In terms of columns and rows, dx means a shift in the x value of each column in the
        array, and dy means a shift in the y value of each row.  In other words, the following
        will return the same value for ixy.  The shift function just changes the coordinates (x,y)
        used for that pixel:

            >>> ixy = im(x,y)
            >>> im.shift(3,9)
            >>> ixy = im(x+3, y+9)
        """
        delta = utilities.parse_pos_args(args, kwargs, 'dx', 'dy', integer=True)
        self._shift(delta)

    def _shift(self, delta):
        """Equivalent to im.shift(delta), but without some of the sanity checks and extra options.

        @param delta    The amount to shift.  Must be a galsim.PositionI instance.
        """
        # The parse_pos_args function is a bit slow, so go directly to this point when we
        # call shift from setCenter or setOrigin.
        if delta.x != 0 or delta.y != 0:
            self._bounds = self._bounds.shift(delta)
            if self.wcs is not None:
                self.wcs = self.wcs.withOrigin(delta)

    def setCenter(self, *args, **kwargs):
        """Set the center of the image to the given (integral) (xcen, ycen)

        The arguments here may be either (xcen, ycen) or a PositionI instance.
        Or you can provide xcen, ycen as named kwargs.

        In terms of the rows and columns, xcen is the new x value for the central column, and ycen
        is the new y value of the central row.  For even-sized arrays, there is no central column
        or row, so the convention we adopt in this case is to round up.  For example:

            >>> im = galsim.Image(numpy.array(range(16),dtype=float).reshape((4,4)))
            >>> im(1,1)
            0.0
            >>> im(4,1)
            3.0
            >>> im(4,4)
            15.0
            >>> im(3,3)
            10.0
            >>> im.setCenter(0,0)
            >>> im(0,0)
            10.0
            >>> im(-2,-2)
            0.0
            >>> im(1,-2)
            3.0
            >>> im(1,1)
            15.0
            >>> im.setCenter(234,456)
            >>> im(234,456)
            10.0
            >>> im.bounds
            galsim.BoundsI(xmin=232, xmax=235, ymin=454, ymax=457)
        """
        cen = utilities.parse_pos_args(args, kwargs, 'xcen', 'ycen', integer=True)
        self._shift(cen - self.center)

    def setOrigin(self, *args, **kwargs):
        """Set the origin of the image to the given (integral) (x0, y0)

        The arguments here may be either (x0, y0) or a PositionI instance.
        Or you can provide x0, y0 as named kwargs.

        In terms of the rows and columns, x0 is the new x value for the first column,
        and y0 is the new y value of the first row.  For example:

            >>> im = galsim.Image(numpy.array(range(16),dtype=float).reshape((4,4)))
            >>> im(1,1)
            0.0
            >>> im(4,1)
            3.0
            >>> im(1,4)
            12.0
            >>> im(4,4)
            15.0
            >>> im.setOrigin(0,0)
            >>> im(0,0)
            0.0
            >>> im(3,0)
            3.0
            >>> im(0,3)
            12.0
            >>> im(3,3)
            15.0
            >>> im.setOrigin(234,456)
            >>> im(234,456)
            0.0
            >>> im.bounds
            galsim.BoundsI(xmin=234, xmax=237, ymin=456, ymax=459)
         """
        origin = utilities.parse_pos_args(args, kwargs, 'x0', 'y0', integer=True)
        self._shift(origin - self.origin)

    @property
    def center(self):
        """The current nominal center (xcen,ycen) of the image as a PositionI instance.

        In terms of the rows and columns, xcen is the x value for the central column, and ycen
        is the y value of the central row.  For even-sized arrays, there is no central column
        or row, so the convention we adopt in this case is to round up.  For example:

            >>> im = galsim.Image(numpy.array(range(16),dtype=float).reshape((4,4)))
            >>> im.center
            galsim.PositionI(x=3, y=3)
            >>> im(im.center)
            10.0
            >>> im.setCenter(56,72)
            >>> im.center
            galsim.PositionI(x=56, y=72)
            >>> im(im.center)
            10.0
        """
        return self.bounds.center

    @property
    def true_center(self):
        """The current true center of the image as a PositionD instance.

        Unline the nominal center returned by im.center, this value may be half-way between
        two pixels if the image has an even number of rows or columns.  It gives the position
        (x,y) at the exact center of the image, regardless of whether this is at the center of
        a pixel (integer value) or halfway between two (half-integer).  For example:

            >>> im = galsim.Image(numpy.array(range(16),dtype=float).reshape((4,4)))
            >>> im.center
            galsim.PositionI(x=3, y=3)
            >>> im.true_center
            galsim.PositionI(x=2.5, y=2.5)
            >>> im.setCenter(56,72)
            >>> im.center
            galsim.PositionI(x=56, y=72)
            >>> im.true_center
            galsim.PositionD(x=55.5, y=71.5)
            >>> im.setOrigin(0,0)
            >>> im.true_center
            galsim.PositionD(x=1.5, y=1.5)
        """
        return self.bounds.true_center
    
    @property
    def origin(self):
        """Return the origin of the image.  i.e. the (x,y) position of the lower-left pixel.

        In terms of the rows and columns, this is the (x,y) coordinate of the first column, and
        first row of the array.  For example:

            >>> im = galsim.Image(numpy.array(range(16),dtype=float).reshape((4,4)))
            >>> im.origin
            galsim.PositionI(x=1, y=1)
            >>> im(im.origin)
            0.0
            >>> im.setOrigin(23,45)
            >>> im.origin
            galsim.PositionI(x=23, y=45)
            >>> im(im.origin)
            0.0
            >>> im(23,45)
            0.0
            >>> im.bounds
            galsim.BoundsI(xmin=23, xmax=26, ymin=45, ymax=48)
        """
        return self.bounds.origin

    def __call__(self, *args, **kwargs):
        """Get the pixel value at given position

        The arguments here may be either (x, y) or a PositionI instance.
        Or you can provide x, y as named kwargs.
        """
        pos = utilities.parse_pos_args(args, kwargs, 'x', 'y', integer=True)
        return self.getValue(pos.x,pos.y)

    def getValue(self, x, y):
        """This method is a synonym for im(x,y).  It is a bit faster than im(x,y), since GalSim
        does not have to parse the different options available for __call__.  (i.e. im(x,y) or
        im(pos) or im(x=x,y=y))
        """
        if not self.bounds.isDefined():
            raise GalSimUndefinedBoundsError("Attempt to access values of an undefined image")
        if not self.bounds.includes(x,y):
            raise GalSimBoundsError("Attempt to access position not in bounds of image.",
                                    PositionI(x,y), self.bounds)
        return self._getValue(x,y)

    def _getValue(self, x, y):
        """Equivalent to self.getValue(x,y), except there are no checks that the values fall
        within the bounds of the image.
        """
        return self._array[y-self.ymin, x-self.xmin]

    def setValue(self, *args, **kwargs):
        """Set the pixel value at given (x,y) position

        The arguments here may be either (x, y, value) or (pos, value) where pos is a PositionI.
        Or you can provide x, y, value as named kwargs.

        This is equivalent to self[x,y] = rhs
        """
        if self.isconst:
            raise GalSimImmutableError("Cannot modify the values of an immutable Image", self)
        if not self.bounds.isDefined():
            raise GalSimUndefinedBoundsError("Attempt to set value of an undefined image")
        pos, value = utilities.parse_pos_args(args, kwargs, 'x', 'y', integer=True,
                                                     others=['value'])
        if not self.bounds.includes(pos):
            raise GalSimBoundsError("Attempt to set position not in bounds of image",
                                    pos, self.bounds)
        self._setValue(pos.x,pos.y,value)

    def _setValue(self, x, y, value):
        """Equivalent to self.setValue(x,y,value) except that there are no checks that the values
        fall within the bounds of the image.
        """
        self._array[y-self.ymin, x-self.xmin] = value

    def addValue(self, *args, **kwargs):
        """Add some amount to the pixel value at given (x,y) position

        The arguments here may be either (x, y, value) or (pos, value) where pos is a PositionI.
        Or you can provide x, y, value as named kwargs.

        This is equivalent to self[x,y] += rhs
        """
        if self.isconst:
            raise GalSimImmutableError("Cannot modify the values of an immutable Image", self)
        if not self.bounds.isDefined():
            raise GalSimUndefinedBoundsError("Attempt to set value of an undefined image")
        pos, value = utilities.parse_pos_args(args, kwargs, 'x', 'y', integer=True,
                                                     others=['value'])
        if not self.bounds.includes(pos):
            raise GalSimBoundsError("Attempt to set position not in bounds of image",
                                    pos,self.bounds)
        self._addValue(pos.x,pos.y,value)

    def _addValue(self, x, y, value):
        """Equivalent to self.addValue(x,y,value) except that there are no checks that the values
        fall within the bounds of the image.
        """
        self._array[y-self.ymin, x-self.xmin] += value

    def fill(self, value):
        """Set all pixel values to the given `value`
        """
        if self.isconst:
            raise GalSimImmutableError("Cannot modify the values of an immutable Image", self)
        if not self.bounds.isDefined():
            raise GalSimUndefinedBoundsError("Attempt to set values of an undefined image")
        self._fill(value)

    def _fill(self, value):
        """Equivalent to self.fill(value), except that there are no checks that the bounds
        are defined.
        """
        self._array[:,:] = value

    def setZero(self):
        """Set all pixel values to zero.
        """
        if self.isconst:
            raise GalSimImmutableError("Cannot modify the values of an immutable Image", self)
        self._fill(0)  # This might be made faster with a C++ call to use memset

    def invertSelf(self):
        """Set all pixel values to their inverse: x -> 1/x.

        Note: any pixels whose value is 0 originally are ignored.  They remain equal to 0
        on the output, rather than turning into inf.
        """
        if self.isconst:
            raise GalSimImmutableError("Cannot modify the values of an immutable Image", self)
        if not self.bounds.isDefined():
            raise GalSimUndefinedBoundsError("Attempt to set values of an undefined image")
        self._invertSelf()

    def _invertSelf(self):
        """Equivalent to self.invertSelf(), except that there are no checks that the bounds
        are defined.
        """
        # C++ version skips 0's to 1/0 -> 0 instead of inf.
        with convert_cpp_errors():
            _galsim.invertImage(self._image)

    def replaceNegative(self, replace_value=0):
        """Replace any negative values currently in the image with 0 (or some other value).

        Sometimes FFT drawing can result in tiny negative values, which may be undesirable for
        some purposes.  This method replaces those values with 0 or some other value if desired.

        @param replace_value    The value with which to replace any negative pixels. [default: 0]
        """
        self.array[self.array<0] = replace_value

    def calculateHLR(self, center=None, flux=None, flux_frac=0.5):
        """Returns the half-light radius of a drawn object.

        This method is equivalent to GSObject.calculateHLR when the object has already been
        been drawn onto an image.  Note that the profile should be drawn using a method that
        integrates over pixels and does not add noise. (The default method='auto' is acceptable.)

        If the image has a wcs other than a PixelScale, an AttributeError will be raised.

        @param center       The position in pixels to use for the center, r=0.
                            [default: self.true_center]
        @param flux         The total flux.  [default: sum(self.array)]
        @param flux_frac    The fraction of light to be enclosed by the returned radius.
                            [default: 0.5]

        @returns an estimate of the half-light radius in physical units defined by the pixel scale.
        """
        if center is None:
            center = self.true_center

        if flux is None:
            flux = np.sum(self.array, dtype=float)

        # Use radii at centers of pixels as approximation to the radial integral
        x,y = np.meshgrid(range(self.array.shape[1]), range(self.array.shape[0]))
        x = x - center.x + self.bounds.xmin
        y = y - center.y + self.bounds.ymin
        rsq = x*x + y*y

        # Sort by radius
        indx = np.argsort(rsq.ravel())
        rsqf = rsq.ravel()[indx]
        data = self.array.ravel()[indx]
        cumflux = np.cumsum(data, dtype=float)

        # Find the first value with cumflux > 0.5 * flux
        k = np.argmax(cumflux > flux_frac * flux)
        flux_k = cumflux[k] / flux  # normalize to unit total flux

        # Interpolate (linearly) between this and the previous value.
        if k == 0:
            hlrsq = rsqf[0] * (flux_frac / flux_k)
        else:
            fkm1 = cumflux[k-1] / flux
            # For brevity in the next formula:
            fk = flux_k
            f = flux_frac
            hlrsq = (rsqf[k-1] * (fk-f) + rsqf[k] * (f-fkm1)) / (fk-fkm1)

        # This has all been done in pixels.  So normalize according to the pixel scale.
        hlr = np.sqrt(hlrsq) * self.scale

        return hlr


    def calculateMomentRadius(self, center=None, flux=None, rtype='det'):
        """Returns an estimate of the radius based on unweighted second moments of a drawn object.

        This method is equivalent to GSObject.calculateMomentRadius when the object has already
        been drawn onto an image.  Note that the profile should be drawn using a method that
        integrates over pixels and does not add noise. (The default method='auto' is acceptable.)

        If the image has a wcs other than a PixelScale, an AttributeError will be raised.

        @param center       The position in pixels to use for the center, r=0.
                            [default: self.true_center]
        @param flux         The total flux.  [default: sum(self.array)]
        @param rtype        There are three options for this parameter:
                            - 'trace' means return sqrt(T/2)
                            - 'det' means return det(Q)^1/4
                            - 'both' means return both: (sqrt(T/2), det(Q)^1/4)
                            [default: 'det']

        @returns an estimate of the radius in physical units defined by the pixel scale
                 (or both estimates if rtype == 'both').
        """
        if rtype not in ('trace', 'det', 'both'):
            raise GalSimValueError("Invalid rtype.", rtype, ('trace', 'det', 'both'))

        if center is None:
            center = self.true_center

        if flux is None:
            flux = np.sum(self.array, dtype=float)

        # Use radii at centers of pixels as approximation to the radial integral
        x,y = np.meshgrid(range(self.array.shape[1]), range(self.array.shape[0]))
        x = x - center.x + self.bounds.xmin
        y = y - center.y + self.bounds.ymin

        if rtype in ('trace', 'both'):
            # Calculate trace measure:
            rsq = x*x + y*y
            Irr = np.sum(rsq * self.array, dtype=float) / flux

            # This has all been done in pixels.  So normalize according to the pixel scale.
            sigma_trace = (Irr/2.)**0.5 * self.scale

        if rtype in ('det', 'both'):
            # Calculate det measure:
            Ixx = np.sum(x*x * self.array, dtype=float) / flux
            Iyy = np.sum(y*y * self.array, dtype=float) / flux
            Ixy = np.sum(x*y * self.array, dtype=float) / flux

            # This has all been done in pixels.  So normalize according to the pixel scale.
            sigma_det = (Ixx*Iyy-Ixy**2)**0.25 * self.scale

        if rtype == 'trace':
            return sigma_trace
        elif rtype == 'det':
            return sigma_det
        else:
            return sigma_trace, sigma_det


    def calculateFWHM(self, center=None, Imax=0.):
        """Returns the full-width half-maximum (FWHM) of a drawn object.

        This method is equivalent to GSObject.calculateFWHM when the object has already
        been drawn onto an image.  Note that the profile should be drawn using a method that
        does not integrate over pixels, so either 'sb' or 'no_pixel'.  Also, if there is a
        significant amount of noise in the image, this method may not work well.

        If the image has a wcs other than a PixelScale, an AttributeError will be raised.

        @param center       The position in pixels to use for the center, r=0.
                            [default: self.true_center]
        @param Imax         The maximum surface brightness.  [default: max(self.array)]
                            Note: If Imax is provided, and the maximum pixel value is larger than
                            this value, Imax will be updated to use the larger value.

        @returns an estimate of the full-width half-maximum in physical units defined by the
                 pixel scale.
        """
        if center is None:
            center = self.true_center

        # If the full image has a larger maximum, use that.
        Imax2 = np.max(self.array)
        if Imax2 > Imax: Imax = Imax2

        # Use radii at centers of pixels.
        x,y = np.meshgrid(range(self.array.shape[1]), range(self.array.shape[0]))
        x = x - center.x + self.bounds.xmin
        y = y - center.y + self.bounds.ymin
        rsq = x*x + y*y

        # Sort by radius
        indx = np.argsort(rsq.ravel())
        rsqf = rsq.ravel()[indx]
        data = self.array.ravel()[indx]

        # Find the first value with I < 0.5 * Imax
        k = np.argmax(data < 0.5 * Imax)
        Ik = data[k] / Imax

        # Interpolate (linearly) between this and the previous value.
        if k == 0:
            rsqhm = rsqf[0] * (0.5 / Ik)
        else:
            Ikm1 = data[k-1] / Imax
            rsqhm = (rsqf[k-1] * (Ik-0.5) + rsqf[k] * (0.5-Ikm1)) / (Ik-Ikm1)

        # This has all been done in pixels.  So normalize according to the pixel scale.
        fwhm = 2. * np.sqrt(rsqhm) * self.scale

        return fwhm

    def __eq__(self, other):
        # Note that numpy.array_equal can return True if the dtypes of the two arrays involved are
        # different, as long as the contents of the two arrays are logically the same.  For example:
        #
        # >>> double_array = np.arange(1024).reshape(32, 32)*np.pi
        # >>> int_array = np.arange(1024).reshape(32, 32)
        # >>> assert galsim.ImageD(int_array) == galsim.ImageF(int_array) # passes
        # >>> assert galsim.ImageD(double_array) == galsim.ImageF(double_array) # fails

        return (self is other or
                (isinstance(other, Image) and
                 self.bounds == other.bounds and
                 self.wcs == other.wcs and
                 (not self.bounds.isDefined() or np.array_equal(self.array,other.array)) and
                 self.isconst == other.isconst))

    def __ne__(self, other): return not self.__eq__(other)

    # Not immutable object.  So shouldn't be used as a hash.
    __hash__ = None

def _Image(array, bounds, wcs):
    """Equivalent to Image(array, bounds, wcs), but without the overhead of sanity checks,
    and the other options for how to provide the arguments.
    """
    ret = Image.__new__(Image)
    ret.wcs = wcs
    ret._dtype = array.dtype.type
    if ret._dtype in Image._alias_dtypes:
        ret._dtype = Image._alias_dtypes[ret._dtype]
        array = array.astype(ret._dtype)
    ret._array = array
    ret._bounds = bounds
    return ret

# These are essentially aliases for the regular Image with the correct dtype
def ImageUS(*args, **kwargs):
    """Alias for galsim.Image(..., dtype=numpy.uint16)
    """
    kwargs['dtype'] = np.uint16
    return Image(*args, **kwargs)

def ImageUI(*args, **kwargs):
    """Alias for galsim.Image(..., dtype=numpy.uint32)
    """
    kwargs['dtype'] = np.uint32
    return Image(*args, **kwargs)

def ImageS(*args, **kwargs):
    """Alias for galsim.Image(..., dtype=numpy.int16)
    """
    kwargs['dtype'] = np.int16
    return Image(*args, **kwargs)

def ImageI(*args, **kwargs):
    """Alias for galsim.Image(..., dtype=numpy.int32)
    """
    kwargs['dtype'] = np.int32
    return Image(*args, **kwargs)

def ImageF(*args, **kwargs):
    """Alias for galsim.Image(..., dtype=numpy.float32)
    """
    kwargs['dtype'] = np.float32
    return Image(*args, **kwargs)

def ImageD(*args, **kwargs):
    """Alias for galsim.Image(..., dtype=numpy.float64)
    """
    kwargs['dtype'] = np.float64
    return Image(*args, **kwargs)

def ImageCF(*args, **kwargs):
    """Alias for galsim.Image(..., dtype=numpy.complex64)
    """
    kwargs['dtype'] = np.complex64
    return Image(*args, **kwargs)

def ImageCD(*args, **kwargs):
    """Alias for galsim.Image(..., dtype=numpy.complex128)
    """
    kwargs['dtype'] = np.complex128
    return Image(*args, **kwargs)


################################################################################################
#
# Now we have to make some modifications to the C++ layer objects.  Mostly adding some
# arithmetic functions, so they work more intuitively.
#

# Define a utility function to be used by the arithmetic functions below
def check_image_consistency(im1, im2, integer=False):
    if integer and not im1.isinteger:
        raise GalSimValueError("Image must have integer values.",im1)
    if isinstance(im2, Image):
        if im1.array.shape != im2.array.shape: # pragma: no cover  (covered, but codecov says it's not.)
            raise GalSimIncompatibleValuesError( "Image shapes are inconsistent", im1=im1, im2=im2)
        if integer and not im2.isinteger:
            raise GalSimValueError("Image must have integer values.",im2)

def Image_add(self, other):
    check_image_consistency(self, other)
    try:
        a = other.array
    except AttributeError:
        a = other
    return _Image(self.array + a, self.bounds, self.wcs)

def Image_iadd(self, other):
    check_image_consistency(self, other)
    try:
        a = other.array
        dt = a.dtype
    except AttributeError:
        a = other
        dt = type(a)
    if dt == self.array.dtype:
        self.array[:,:] += a
    else:
        self.array[:,:] = (self.array + a).astype(self.array.dtype, copy=False)
    return self

def Image_sub(self, other):
    check_image_consistency(self, other)
    try:
        a = other.array
    except AttributeError:
        a = other
    return _Image(self.array - a, self.bounds, self.wcs)

def Image_rsub(self, other):
    return _Image(other-self.array, self.bounds, self.wcs)

def Image_isub(self, other):
    check_image_consistency(self, other)
    try:
        a = other.array
        dt = a.dtype
    except AttributeError:
        a = other
        dt = type(a)
    if dt == self.array.dtype:
        self.array[:,:] -= a
    else:
        self.array[:,:] = (self.array - a).astype(self.array.dtype, copy=False)
    return self

def Image_mul(self, other):
    check_image_consistency(self, other)
    try:
        a = other.array
    except AttributeError:
        a = other
    return _Image(self.array * a, self.bounds, self.wcs)

def Image_imul(self, other):
    check_image_consistency(self, other)
    try:
        a = other.array
        dt = a.dtype
    except AttributeError:
        a = other
        dt = type(a)
    if dt == self.array.dtype:
        self.array[:,:] *= a
    else:
        self.array[:,:] = (self.array * a).astype(self.array.dtype, copy=False)
    return self

def Image_div(self, other):
    check_image_consistency(self, other)
    try:
        a = other.array
    except AttributeError:
        a = other
    return _Image(self.array / a, self.bounds, self.wcs)

def Image_rdiv(self, other):
    return _Image(other / self.array, self.bounds, self.wcs)

def Image_idiv(self, other):
    check_image_consistency(self, other)
    try:
        a = other.array
        dt = a.dtype
    except AttributeError:
        a = other
        dt = type(a)
    if dt == self.array.dtype and not self.isinteger:
        # if dtype is an integer type, then numpy doesn't allow true division /= to assign
        # back to an integer array.  So for integers (or mixed types), don't use /=.
        self.array[:,:] /= a
    else:
        self.array[:,:] = (self.array / a).astype(self.array.dtype, copy=False)
    return self

def Image_floordiv(self, other):
    check_image_consistency(self, other, integer=True)
    try:
        a = other.array
    except AttributeError:
        a = other
    return _Image(self.array // a, self.bounds, self.wcs)

def Image_rfloordiv(self, other):
    check_image_consistency(self, other, integer=True)
    return _Image(other // self.array, self.bounds, self.wcs)

def Image_ifloordiv(self, other):
    check_image_consistency(self, other, integer=True)
    try:
        a = other.array
        dt = a.dtype
    except AttributeError:
        a = other
        dt = type(a)
    if dt == self.array.dtype:
        self.array[:,:] //= a
    else:
        self.array[:,:] = (self.array // a).astype(self.array.dtype, copy=False)
    return self

def Image_mod(self, other):
    check_image_consistency(self, other, integer=True)
    try:
        a = other.array
    except AttributeError:
        a = other
    return _Image(self.array % a, self.bounds, self.wcs)

def Image_rmod(self, other):
    check_image_consistency(self, other, integer=True)
    return _Image(other % self.array, self.bounds, self.wcs)

def Image_imod(self, other):
    check_image_consistency(self, other, integer=True)
    try:
        a = other.array
        dt = a.dtype
    except AttributeError:
        a = other
        dt = type(a)
    if dt == self.array.dtype:
        self.array[:,:] %= a
    else:
        self.array[:,:] = (self.array % a).astype(self.array.dtype, copy=False)
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

def Image_neg(self):
    result = self.copy()
    result *= -1
    return result

# Define &, ^ and | only for integer-type images
def Image_and(self, other):
    check_image_consistency(self, other, integer=True)
    try:
        a = other.array
    except AttributeError:
        a = other
    return _Image(self.array & a, self.bounds, self.wcs)


def Image_iand(self, other):
    check_image_consistency(self, other, integer=True)
    try:
        self.array[:,:] &= other.array
    except AttributeError:
        self.array[:,:] &= other
    return self

def Image_xor(self, other):
    check_image_consistency(self, other, integer=True)
    try:
        a = other.array
    except AttributeError:
        a = other
    return _Image(self.array ^ a, self.bounds, self.wcs)

def Image_ixor(self, other):
    check_image_consistency(self, other, integer=True)
    try:
        self.array[:,:] ^= other.array
    except AttributeError:
        self.array[:,:] ^= other
    return self

def Image_or(self, other):
    check_image_consistency(self, other, integer=True)
    try:
        a = other.array
    except AttributeError:
        a = other
    return _Image(self.array | a, self.bounds, self.wcs)

def Image_ior(self, other):
    check_image_consistency(self, other, integer=True)
    try:
        self.array[:,:] |= other.array
    except AttributeError:
        self.array[:,:] |= other
    return self

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
Image.__rdiv__ = Image_rdiv
Image.__truediv__ = Image_div
Image.__rtruediv__ = Image_rdiv
Image.__idiv__ = Image_idiv
Image.__itruediv__ = Image_idiv
Image.__mod__ = Image_mod
Image.__rmod__ = Image_rmod
Image.__imod__ = Image_imod
Image.__floordiv__ = Image_floordiv
Image.__rfloordiv__ = Image_rfloordiv
Image.__ifloordiv__ = Image_ifloordiv
Image.__ipow__ = Image_ipow
Image.__pow__ = Image_pow
Image.__neg__ = Image_neg
Image.__and__ = Image_and
Image.__xor__ = Image_xor
Image.__or__ = Image_or
Image.__rand__ = Image_and
Image.__rxor__ = Image_xor
Image.__ror__ = Image_or
Image.__iand__ = Image_iand
Image.__ixor__ = Image_ixor
Image.__ior__ = Image_ior
