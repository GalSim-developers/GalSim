# Copyright (c) 2012-2015 by the GalSim developers team on GitHub
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
"""@file utilities.py
Module containing general utilities for the GalSim software.
"""

import numpy as np
import galsim

def roll2d(image, (iroll, jroll)):
    """Perform a 2D roll (circular shift) on a supplied 2D NumPy array, conveniently.

    @param image            The NumPy array to be circular shifted.
    @param (iroll, jroll)   The roll in the i and j dimensions, respectively.

    @returns the rolled image.
    """
    return np.roll(np.roll(image, jroll, axis=1), iroll, axis=0)

def kxky(array_shape=(256, 256)):
    """Return the tuple `(kx, ky)` corresponding to the DFT of a unit integer-sampled array of input
    shape.
    
    Uses the SBProfile conventions for Fourier space, so `k` varies in approximate range (-pi, pi].
    Uses the most common DFT element ordering conventions (and those of FFTW), so that `(0, 0)`
    array element corresponds to `(kx, ky) = (0, 0)`.

    See also the docstring for np.fftfreq, which uses the same DFT convention, and is called here,
    but misses a factor of pi.
    
    Adopts NumPy array index ordering so that the trailing axis corresponds to `kx`, rather than the
    leading axis as would be expected in IDL/Fortran.  See docstring for numpy.meshgrid which also
    uses this convention.

    @param array_shape   The NumPy array shape desired for `kx, ky`.
    """
    # Note: numpy shape is y,x
    k_xaxis = np.fft.fftfreq(array_shape[1]) * 2. * np.pi
    k_yaxis = np.fft.fftfreq(array_shape[0]) * 2. * np.pi
    return np.meshgrid(k_xaxis, k_yaxis)

def g1g2_to_e1e2(g1, g2):
    """Convenience function for going from `(g1, g2)` -> `(e1, e2)`.

    Here `g1` and `g2` are reduced shears, and `e1` and `e2` are distortions - see shear.py for
    definitions of reduced shear and distortion in terms of axis ratios or other ways of specifying
    ellipses.

    @param g1   First reduced shear component
    @param g2   Second reduced shear component

    @returns the corresponding distortions, e1 and e2.
    """
    # Conversion:
    # e = (a^2-b^2) / (a^2+b^2)
    # g = (a-b) / (a+b)
    # b/a = (1-g)/(1+g)
    # e = (1-(b/a)^2) / (1+(b/a)^2)
    gsq = g1*g1 + g2*g2
    if gsq > 0.:
        g = np.sqrt(gsq)
        boa = (1-g) / (1+g)
        e = (1 - boa*boa) / (1 + boa*boa)
        e1 = g1 * (e/g)
        e2 = g2 * (e/g)
        return e1, e2
    elif gsq == 0.:
        return 0., 0.
    else:
        raise ValueError("Input |g|^2 < 0, cannot convert.")

def rotate_xy(x, y, theta):
    """Rotates points in the xy-Cartesian plane counter-clockwise through an angle `theta` about the
    origin of the Cartesian coordinate system.

    @param x        NumPy array of input `x` coordinates
    @param y        NumPy array of input `y` coordinates
    @param theta    Rotation angle (+ve counter clockwise) as an Angle instance

    @return the rotated coordinates `(x_rot,y_rot)`.
    """
    if not isinstance(theta, galsim.Angle):
        raise TypeError("Input rotation angle theta must be a galsim.Angle instance.")
    sint, cost = theta.sincos()
    x_rot = x * cost - y * sint
    y_rot = x * sint + y * cost
    return x_rot, y_rot

def parse_pos_args(args, kwargs, name1, name2, integer=False, others=[]):
    """Parse the args and kwargs of a function call to be some kind of position.

    We allow four options:

        f(x,y)
        f(galsim.PositionD(x,y)) or f(galsim.PositionI(x,y))
        f( (x,y) )  (or any indexable thing)
        f(name1=x, name2=y)

    If the inputs must be integers, set `integer=True`.
    If there are other args/kwargs to parse after these, then their names should be 
    be given as the parameter `others`, which are passed back in a tuple after the position.
    """
    def canindex(arg):
        try: arg[0], arg[1]
        except: return False
        else: return True

    other_vals = []
    if len(args) == 0:
        # Then name1,name2 need to be kwargs
        # If not, then python will raise an appropriate error.
        x = kwargs.pop(name1)
        y = kwargs.pop(name2)
    elif ( ( isinstance(args[0], galsim.PositionI) or
             (not integer and isinstance(args[0], galsim.PositionD)) ) and
           len(args) <= 1+len(others) ):
        x = args[0].x
        y = args[0].y
        for arg in args[1:]:
            other_vals.append(arg)
            others.pop(0)
    elif canindex(args[0]) and len(args) <= 1+len(others):
        x = args[0][0]
        y = args[0][1]
        for arg in args[1:]:
            other_vals.append(arg)
            others.pop(0)
    elif len(args) == 1:
        raise TypeError("Cannot parse argument "+str(args[0])+" as a position")
    elif len(args) <= 2 + len(others):
        x = args[0]
        y = args[1]
        for arg in args[2:]:
            other_vals.append(arg)
            others.pop(0)
    else:
        raise TypeError("Too many arguments supplied")
    # Read any remaining other kwargs
    if others:
        for name in others:
            val = kwargs.pop(name)
            other_vals.append(val)
    if kwargs:
        raise TypeError("Received unexpected keyword arguments: %s",kwargs)

    if integer:
        pos = galsim.PositionI(int(x),int(y))
    else:
        pos = galsim.PositionD(float(x),float(y))
    if other_vals:
        return (pos,) + tuple(other_vals)
    else:
        return pos


class SimpleGenerator:
    """A simple class that is constructed with an arbitrary object.
    Then generator() will return that object.
    """
    def __init__(self, obj): self._obj = obj
    def __call__(self): return self._obj
            
class AttributeDict(object):
    """Dictionary class that allows for easy initialization and refs to key values via attributes.

    NOTE: Modified a little from Jim's bot.git AttributeDict class so that tab completion now works
    in ipython since attributes are actually added to __dict__.
    
    HOWEVER this means the __dict__ attribute has been redefined to be a collections.defaultdict()
    so that Jim's previous default attribute behaviour is also replicated.
    """
    def __init__(self):
        import collections
        object.__setattr__(self, "__dict__", collections.defaultdict(AttributeDict))

    def __getattr__(self, name):
        return self.__dict__[name]

    def __setattr__(self, name, value):
        self.__dict__[name] = value

    def merge(self, other):
        self.__dict__.update(other.__dict__)

    def _write(self, output, prefix=""):
        for k, v in self.__dict__.iteritems():
            if isinstance(v, AttributeDict):
                v._write(output, prefix="{0}{1}.".format(prefix, k))
            else:
                output.append("{0}{1} = {2}".format(prefix, k, repr(v)))

    def __nonzero__(self):
        return not not self.__dict__

    def __repr__(self):
        output = []
        self._write(output, "")
        return "\n".join(output)

    __str__ = __repr__

    def __len__(self):
        return len(self.__dict__)

def rand_arr(shape, deviate):
    """Function to make a 2d array of random deviates (of any sort).

    @param shape        A list of length 2, indicating the desired 2d array dimensions
    @param deviate      Any GalSim deviate (see random.py) such as UniformDeviate, GaussianDeviate,
                        etc. to be used to generate random numbers

    @returns a NumPy array of the desired dimensions with random numbers generated using the
    supplied deviate.
    """
    if len(shape) is not 2:
        raise ValueError("Can only make a 2d array from this function!")
    # note reversed indices due to NumPy vs. Image array indexing conventions!
    tmp_img = galsim.ImageD(shape[1], shape[0])
    tmp_img.addNoise(galsim.DeviateNoise(deviate))
    return tmp_img.array

def convert_interpolant(interpolant):
    """Convert a given interpolant to an Interpolant if it is given as a string.
    """
    if interpolant is None:
        return None  # caller is responsible for setting a default if desired.
    elif isinstance(interpolant, galsim.Interpolant):
        return interpolant
    else:
        # Will raise an appropriate exception if this is invalid.
        return galsim.Interpolant(interpolant)

# A helper function for parsing the input position arguments for PowerSpectrum and NFWHalo:
def _convertPositions(pos, units, func):
    """Convert `pos` from the valid ways to input positions to two NumPy arrays

       This is used by the functions getShear(), getConvergence(), getMagnification(), and
       getLensing() for both PowerSpectrum and NFWHalo.
    """
    # Check for PositionD or PositionI:
    if isinstance(pos,galsim.PositionD) or isinstance(pos,galsim.PositionI):
        pos = [ np.array([pos.x], dtype='float'),
                np.array([pos.y], dtype='float') ]

    # Check for list of PositionD or PositionI:
    # The only other options allow pos[0], so if this is invalid, an exception 
    # will be raised:
    elif isinstance(pos[0],galsim.PositionD) or isinstance(pos[0],galsim.PositionI):
        pos = [ np.array([p.x for p in pos], dtype='float'),
                np.array([p.y for p in pos], dtype='float') ]

    # Now pos must be a tuple of length 2
    elif len(pos) != 2:
        raise TypeError("Unable to parse the input pos argument for %s."%func)

    else:
        # Check for (x,y):
        try:
            pos = [ np.array([float(pos[0])], dtype='float'),
                    np.array([float(pos[1])], dtype='float') ]
        except:
            # Only other valid option is ( xlist , ylist )
            pos = [ np.array(pos[0], dtype='float'),
                    np.array(pos[1], dtype='float') ]

    # Check validity of units
    if isinstance(units, basestring):
        # if the string is invalid, this raises a reasonable error message.
        units = galsim.angle.get_angle_unit(units)
    if not isinstance(units, galsim.AngleUnit):
        raise ValueError("units must be either an AngleUnit or a string")

    # Convert pos to arcsec
    if units != galsim.arcsec:
        scale = 1. * units / galsim.arcsec
        # Note that for the next two lines, pos *must* be a list, not a tuple.  Assignments to
        # elements of tuples is not allowed.
        pos[0] *= scale
        pos[1] *= scale

    return pos

def thin_tabulated_values(x, f, rel_err=1.e-4, preserve_range=False):
    """
    Remove items from a set of tabulated f(x) values so that the error in the integral is still 
    accurate to a given relative accuracy.

    The input `x,f` values can be lists, NumPy arrays, or really anything that can be converted
    to a NumPy array.  The new lists will be output as python lists.

    @param x                The `x` values in the f(x) tabulation.
    @param f                The `f` values in the f(x) tabulation.
    @param rel_err          The maximum relative error to allow in the integral from the removal.
                            [default: 1.e-4]
    @param preserve_range   Should the original range of `x` be preserved? (True) Or should the ends
                            be trimmed to include only the region where the integral is 
                            significant? (False)  [default: False]

    @returns a tuple of lists `(x_new, y_new)` with the thinned tabulation.
    """
    import numpy
    x = numpy.array(x)
    f = numpy.array(f)

    # Check for valid inputs
    if len(x) != len(f):
        raise ValueError("len(x) != len(f)")
    if rel_err <= 0 or rel_err >= 1:
        raise ValueError("rel_err must be between 0 and 1")
    if not (numpy.diff(x) >= 0).all():
        raise ValueError("input x is not sorted.")

    # Check for trivial noop.
    if len(x) <= 2:
        # Nothing to do
        return x,f

    # Start by calculating the complete integral of |f|
    total_integ = numpy.trapz(abs(f),x)
    if total_integ == 0:
        return numpy.array([ x[0], x[-1] ]), numpy.array([ f[0], f[-1] ])
    thresh = rel_err * total_integ

    if not preserve_range:
        # Remove values from the front that integrate to less than thresh.
        integ = 0.5 * (abs(f[0]) + abs(f[1])) * (x[1] - x[0])
        k0 = 0
        while k0 < len(x)-2 and integ < thresh:
            k0 = k0+1
            integ += 0.5 * (abs(f[k0]) + abs(f[k0+1])) * (x[k0+1] - x[k0])
        # Now the integral from 0 to k0+1 (inclusive) is a bit too large.
        # That means k0 is the largest value we can use that will work as the staring value.

        # Remove values from the back that integrate to less than thresh.
        k1 = len(x)-1
        integ = 0.5 * (abs(f[k1-1]) + abs(f[k1])) * (x[k1] - x[k1-1])
        while k1 > k0 and integ < thresh:
            k1 = k1-1
            integ += 0.5 * (abs(f[k1-1]) + abs(f[k1])) * (x[k1] - x[k1-1])
        # Now the integral from k1-1 to len(x)-1 (inclusive) is a bit too large.
        # That means k1 is the smallest value we can use that will work as the ending value.

        x = x[k0:k1+1]  # +1 since end of range is given as one-past-the-end.
        f = f[k0:k1+1]

    # Start a new list with just the first item so far
    newx = [ x[0] ]
    newf = [ f[0] ]

    k0 = 0  # The last item currently in the new array
    k1 = 1  # The current item we are considering to skip or include
    while k1 < len(x)-1:
        # We are considering replacing all the true values between k0 and k1+1 (non-inclusive)
        # with a linear approxmation based on the points at k0 and k1+1.
        lin_f = f[k0] + (f[k1+1]-f[k0])/(x[k1+1]-x[k0]) * (x[k0:k1+2] - x[k0])
        # Integrate | f(x) - lin_f(x) | from k0 to k1+1, inclusive.
        integ = numpy.trapz(abs(f[k0:k1+2] - lin_f), x[k0:k1+2])
        # If the integral of the difference is < thresh, we can skip this item.
        if integ < thresh:
            # OK to skip item k1
            k1 = k1 + 1
        else:
            # Have to include this one.
            newx.append(x[k1])
            newf.append(f[k1])
            k0 = k1
            k1 = k1 + 1

    # Always include the last item
    newx.append(x[-1])
    newf.append(f[-1])

    return newx, newf

def _gammafn(x):
    """
    This code is not currently used, but in case we need a gamma function at some point, it will be
    here in the utilities module.

    The gamma function is present in python2.7's math module, but not 2.6.  So try using that,
    and if it fails, use some code from RosettaCode:
    http://rosettacode.org/wiki/Gamma_function#Python
    """
    try:
        import math
        return math.gamma(x)
    except:
        y  = float(x) - 1.0;
        sm = _gammafn._a[-1];
        for an in _gammafn._a[-2::-1]:
            sm = sm * y + an;
        return 1.0 / sm;

_gammafn._a = ( 1.00000000000000000000, 0.57721566490153286061, -0.65587807152025388108,
              -0.04200263503409523553, 0.16653861138229148950, -0.04219773455554433675,
              -0.00962197152787697356, 0.00721894324666309954, -0.00116516759185906511,
              -0.00021524167411495097, 0.00012805028238811619, -0.00002013485478078824,
              -0.00000125049348214267, 0.00000113302723198170, -0.00000020563384169776,
               0.00000000611609510448, 0.00000000500200764447, -0.00000000118127457049,
               0.00000000010434267117, 0.00000000000778226344, -0.00000000000369680562,
               0.00000000000051003703, -0.00000000000002058326, -0.00000000000000534812,
               0.00000000000000122678, -0.00000000000000011813, 0.00000000000000000119,
               0.00000000000000000141, -0.00000000000000000023, 0.00000000000000000002
             )

def interleaveImages(im_list,N,offsets=None,add_flux=True,suppress_warnings=False):
    """
    Interleaves two or more images and outputs a larger image.

    The sampling length of simulated images can be set arbitrarily using the `pixel_scale' argument
    in drawImage() routine appropriately. However, pixel level detector effects can be included
    only on images drawn at the native pixel scale, which are typically undersampled. Nyquist
    sampled images that also include the effects of detector non-idealities can be obtained by
    drawing muliple undersampled images (with the detector effects included) that are offset from
    each other by a fraction of a pixel. This is equivalent to obtaining a finer sampled image from
    a dither sequence, except that this routine handles only equispaced offsets. The dither sequence
    must be a list of galsim.Images instances supplied through 'im_list'.

    @param im_list           A list containing the galsim.Image instances to be interleaved.
    @param N                 Number of images to interleave in either directions. It can be of type
                             `int' if equal number of images are interleaved in both directions or
                             a tuple of two integers, containing the number of images in x and y
                             directions respectively.
    @param offsets           A list containing the offsets as galsim.PositionD instances
                             corresponding to every image in `im_list'. The offsets must be equally
                             spaced and must span an entire pixel area. The offset values must
                             be symmetric around zero, hence taking positive and negative values.
                             The default offset ordering is to vary the offset in x from positive to
                             negative for every offset in y which should go from positive to
                             negative. Providing `offsets' is highly recommended. [default:None]
    @param add_flux          Should the routine add the fluxes of all the images (True) or average
                             them (False)?
    @param suppress_warnings Suppresses the warnings about the pixel scale of the output, if True.

    @returns the interleaved image
    """

    if isinstance(N,int):
        n1,n2 = N,N
    elif isinstance(N,tuple):
        n1,n2 = N
        if not (isinstance(n1,int) and  isinstance(n2,int)):
            raise TypeError("'N' has to be of type int or a tuple of two integers")
    else:
        raise TypeError("'N' has to be of type int or a tuple of two integers")

    if len(im_list)<2:
        raise TypeError("'im_list' needs to have at least two instances of galsim.Image")

    if (n1*n2 != len(im_list)):
        raise ValueError("'N' is incompatible with the number of images in 'im_list'")

    if offsets is not None:
        if len(im_list)!=len(offsets):
            raise ValueError("'im_list' and 'offsets' must be lists of same length")
        for offset in offsets:
            if not isinstance(offset,galsim.PositionD):
                raise TypeError("'offsets' must be a list of galsim.PositionD instances")

    if isinstance(im_list[0],galsim.Image):
        y_size, x_size = im_list[0].array.shape
        scale = im_list[0].scale
    else:
        raise TypeError("'im_list' must be a list of galsim.Image instances")

    for im in im_list[1:]:
        if not isinstance(im,galsim.Image):
            raise TypeError("'im_list' must be a list of galsim.Image instances")

        if im.array.shape != (y_size,x_size):
            raise ValueError("All galsim.Image instances in 'im_list' must be of the same size")
 
        if im.scale != scale:
            raise ValueError("All galsim.Image instance in 'im_list' must have the same pixel scale")

    img_array = np.zeros((n2*y_size,n1*x_size))
    # The tricky part - going from (x,y) Image coordinates to array indices
    if offsets is None:
        # default offset settings
        for j in xrange(n2):
            for i in xrange(n1):
                img_array[j::n2,i::n1] = im_list[n1*j+i].array[:,:]
    else:
        # DX[i'] = -(i+0.5)/n+0.5 = -i/n + 0.5*(n-1)/n
        #    i  = -n DX[i'] + 0.5*(n-1)
        for k in xrange(len(offsets)):
            dx, dy = offsets[k].x, offsets[k].y
            i = int(round((n1-1)*0.5-n1*dx))
            j = int(round((n2-1)*0.5-n2*dy))
            img_array[j::n2,i::n1] = im_list[k].array[:,:]

    if add_flux is True:
        img = galsim.Image(img_array)
    else:
        img = galsim.Image((1.0/(len(im_list)))*img_array)

    if (n1==n2):
        if scale is not None:
            img.scale = im_list[0].scale*(1./n1)
    elif suppress_warnings is False:
        import warnings
        warnings.warn("Interleaved image could not be assigned a pixel scale automatically")
    return img
