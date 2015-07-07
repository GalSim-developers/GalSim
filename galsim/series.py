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
"""@file series.py
Definitions for Galsim Series class and subclasses.
"""

import galsim
import numpy as np
import copy
import math
import operator
from itertools import product, combinations, count


class Series(object):
    def __init__(self):
        """Abstract base class for GalSim profiles represented as a series of basis profiles.
        Generally, coefficients of the basis profiles can be manipulated to set parameters of the
        series profile, such as size and ellipticity for SpergelSeries and MoffatSeries, or
        wavefront expansion coefficients for LinearOpticalSeries.  This allows one to rapidly
        create images with arbitrary values of these parameters by forming linear combinations of
        images of the basis profiles, which can be precomputed and cached in memory.  The
        convolution of two series objects can also be rapidly computed once the convolution of each
        term in the Cartesian product of each series' basis profiles is computed, drawn to an image,
        and cached.

        One drawback for Series objects is that a cache is only good for a single combination of
        image parameters, such as the shape and scale.  If you want to draw a new image at a
        different scale or with a different shape, the precomputation of the basis profile images
        will be done again and cached for future use.

        The first Series object created can set the `maxcache` keyword, which sets how many Series
        objects will be cached in memory before new cache entries begin to over-write earlier cache
        entries.  This may be important to conserve RAM, as its not too hard to create caches that
        occupy 10s to 100s (or more!) MB of memory.  You can check the current state of the cache
        with the Series.inspectCache() method.
        """
        pass

    @staticmethod
    def _cube(key):
        # Create an image cube.
        # `key` here is a 3-tuple with indices:
        #       0: Series subclass
        #       1: args to pass to getBasisProfiles
        #       2: kwargs to be used in drawImage, as a tuple of tuples to make it hashable.
        #          The first item of each interior tuple is the kwarg name as a string, and the
        #          second item is the associated value.
        # This method is cached using a galsim.utilities.LRU_Cache. Note that the LRU_Cache seems
        # to only work when caching a staticmethod, which is why we pass the Series subclass in
        # here explicitly instead of using `self`.  Another subtlety is that we want different
        # Series instances to be able to use the same cache (for instance a SpergelSeries with
        # e1=0.01, and a SpergelSeries with e1=0.02 will almost always require the same cube).
        # Thus, we don't want the key to include the object instance, only the object class, which
        # is another reason to make this method static.
        series, gbargs, kwargs = key
        kwargs = dict(kwargs)
        objs = series._getBasisProfiles(*gbargs)

        iimult = kwargs['iimult']
        dtype = kwargs['dtype']

        img = objs[0].drawImage(setup_only=True, method='no_pixel')
        N = img.array.shape[0]
        scale = img.scale
        # print 'N: {}'.format(N)
        # print 'scale: {}'.format(scale)
        # print 'iimult: {}'.format(iimult)
        N = int(np.ceil(N * iimult))
        scale = objs[0].nyquistScale() / iimult
        # It's faster to store the stack of basis images as a series of 1D vectors (i.e. a 2D
        # numpy array instead of a cube (or rectangular prism, I guess...))
        # This makes the linear combination step a matrix multiply, which is fast like the wind.
        # I still think of this data structure as a cube though, so that's what I'm calling it.
        cube = np.empty((len(objs), N*N), dtype=dtype)
        for i, obj in enumerate(objs):
            cube[i] = obj.drawImage(nx=N, ny=N, scale=scale, method='no_pixel').array.ravel()
        # Need to store the image scale and shape here so we can eventually turn the 1D image
        # vector into a 2D InterpolatedImage when needed.
        return cube, scale, N

    # @staticmethod
    # def _kcube(key):
    #     # See comments for _basisCube
    #     series, gbargs, kwargs = key
    #     kwargs = dict(kwargs)
    #     objs = self._getBasisProfiles(*gbargs)

    #     re0, im0 = objs[0].drawKImage(*args, **kwargs)
    #     shape = im0.array.shape
    #     recube = np.empty((len(objs), shape[0]*shape[1]), dtype=im0.array.dtype)
    #     imcube = np.empty_like(recube)
    #     for i, obj in enumerate(objs):
    #         tmp = obj.drawKImage(*args, **kwargs)
    #         recube[i] = tmp[0].array.ravel()
    #         imcube[i] = tmp[1].array.ravel()
    #     return recube, imcube, shape

    def drawImage(self, **kwargs):
        """Draw a Series object by forming the appropriate linear combination of basis profile
        images.  This method will search the Series cache to see if the basis images for this
        object already exist, and if not then create and cache them.  Note that a separate cache is
        created for each combination of image parameters (such as shape, wcs/scale, etc.) and also
        certain profile parameters (such as the SpergelSeries `nu` parameter or the MoffatSeries
        `beta` parameter.)  Additional cubes may also be created if a profile parameter falls out of
        range for an existing cube (this applies chiefly to size and ellipticity parameters of
        SpergelSeries or MoffatSeries).

        See GSObject.drawImage() for a description of available arguments for this method.
        """
        # cube gets drawn at its nyquistScale and with a goodImageSize, (modified by iimult), and
        # is then resampled with an InterpolatedImage below.  So strip out image setup keywords
        # here.  For now, we require these to be explicitly set, though in the future sensible
        # defaults should be used.
        nx = kwargs.pop('nx')
        ny = kwargs.pop('ny')
        scale = kwargs.pop('scale')
        if 'iimult' not in kwargs:
            kwargs['iimult'] = 1.0
        if 'dtype' not in kwargs:
            kwargs['dtype'] = np.float64

        # Convolve by pixel
        prof = galsim.SeriesConvolution(self, galsim.Pixel(scale=scale))

        key = prof.__class__, prof._getBasisProfileArgs(), tuple(sorted(kwargs.items()))
        cube, iiscale, N = Series._cube_cache(key)
        coeffs = np.array(prof._getCoeffs(), dtype=cube.dtype)
        cubeim = galsim.Image(np.dot(coeffs, cube).reshape((N, N)), scale=iiscale)

        centroid = self.centroid()
        ii = (galsim.InterpolatedImage(cubeim, calculate_stepk=False, calculate_maxk=False)
              .shift(centroid))
        img = ii.drawImage(nx=nx, ny=ny, scale=scale, method='no_pixel')
        return img

    # def drawKImage(self, *args, **kwargs):
    #     """Draw the Fourier-space image of a Series object by forming the appropriate linear
    #     combination of basis profile Fourier-space images.  This method will search the Series cache
    #     to see if the basis images for this object already exist, and if not then create and cache
    #     them.  See Series.drawImage() docstring for additional caveats.

    #     See GSObject.drawKImage() for a description of available arguments for this method.
    #     """
    #     key = self.__class__, self._getBasisProfileArgs(), args, tuple(sorted(kwargs.items()))
    #     recube, imcube, shape = Series._kcube_cache(key)
    #     coeffs = np.array(self._getCoeffs(), dtype=recube.dtype)
    #     reim = np.dot(coeffs, recube).reshape(shape)
    #     imim = np.dot(coeffs, imcube).reshape(shape)
    #     # TODO: incorporate centroid into kimages
    #     return galsim.Image(reim), galsim.Image(imim)

    @staticmethod
    def drawImages(objlist, **kwargs):
        """ Usage: Series.drawImages(Series instances, draw_arguments, draw_keywords=...)
        """
        for obj in objlist[1:]:
            assert obj.__class__ == objlist[0].__class__
        scale = kwargs['scale']
        out = [None]*len(objlist)
        #group like cube indices together
        gbpas = [o._getBasisProfileArgs() for o in objlist]
        keys = []
        groups = []
        notgrouped = np.ones(len(objlist), dtype=bool)
        while any(notgrouped):
            i = np.nonzero(notgrouped)[0][0]
            key = gbpas[i]
            keys.append(key)
            group = np.nonzero([o==key for o in gbpas])[0]
            groups.append(group)
            notgrouped[group] = False
        for key, group in zip(keys, groups):
            cube, iiscale, N = Series._cube_cache((objlist[0].__class__,
                                                   key,
                                                   tuple(sorted(kwargs.items()))))
            coeffs = np.empty((len(group), cube.shape[0]), dtype=cube.dtype)
            ims = np.dot(coeffs, cube).reshape((len(group), shape[0], shape[1]))
            for i, j in enumerate(group):
                centroid = objlist[i].centroid()
                im = galsim.Image(ims[i].reshape((N, N)), scale=iiscale)
                ii = (galsim.InterpolatedImage(im, calculate_stepk=False, calculate_maxk=False)
                      .shift(centroid))
                out[j] = ii.drawImage(nx=nx, ny=ny, scale=scale, method='no_pixel')
        return out

    def kValue(self, *args, **kwargs):
        """Calculate the value of the Fourier-space image of this Series object at a particular
        coordinate by forming the appropriate linear combination of Fourier-space values of basis
        profiles.

        This method does not use the Series cache.

        See GSObject.kValue() for a description of available arguments for this method.
        """
        kvals = [obj.kValue(*args, **kwargs)
                 for obj in self._getBasisProfiles(*self._getBasisProfileArgs())]
        coeffs = self._getCoeffs()
        return np.dot(kvals, coeffs)

    def xValue(self, *args, **kwargs):
        """Calculate the value of the image of this Series object at a particular coordinate by
        forming the appropriate linear combination of values of basis profiles.

        This method does not use the Series cache.

        See GSObject.xValue() for a description of available arguments for this method.
        """
        xvals = [obj.xValue(*args, **kwargs)
                 for obj in self._getBasisProfiles(*self._getBasisProfileArgs())]
        coeffs = self._getCoeffs()
        return np.dot(xvals, coeffs)

    @staticmethod
    def inspectCache():
        """ Report details of the Series cache, including the number of cached profiles and an
        estimate of the memory footprint of the cache.
        """
        i = ik = mem = 0
        for k, v in Series._cube_cache.cache.iteritems():
            if not isinstance(k, tuple):
                continue
            i += 1
            print
            print "Cached image object: "
            print v[2]
            print "# of basis images: {0}".format(v[3][0].shape[0])
            print "images are {0} x {0} arrays".format(v[3][2])
            print "image scale is {0}".format(v[3][1])
            mem += v[3][0].nbytes

        # for k, v in Series._kcube_cache.cache.iteritems():
        #     if not isinstance(k, tuple):
        #         continue
        #     ik += 1
        #     print
        #     print "Cached kimage object: "
        #     print v[2]
        #     print "# of basis kimages: {0}".format(v[3][0].shape[0])
        #     print "kimages are {0} x {0} arrays".format(v[3][2])
        #     mem += v[3][0].nbytes
        #     mem += v[3][1].nbytes
        print
        print "Found {0} image caches".format(i)
        print "Found {0} kimage caches".format(ik)
        print "Cache occupies ~{0} bytes".format(mem)

    def _getCoeffs(self):
        raise NotImplementedError("subclasses of Series must define _getCoeffs() method")

    def _getBasisProfiles(self):
        raise NotImplementedError("subclasses of Series must define _getBasisProfiles() method")

    def _getBasisProfileArgs(self):
        raise NotImplementedError("subclasses of Series must define _getBasisProfileArgs() method")

    def __eq__(self, other): return repr(self) == repr(other)
    def __ne__(self, other): return not self.__eq__(other)
    def __hash__(self): return hash(repr(self))

Series._cube_cache = galsim.utilities.LRU_Cache(Series._cube, maxsize=100)
# Series._kcube_cache = galsim.utilities.LRU_Cache(Series._kcube, maxsize=100)


class SeriesConvolution(Series):
    def __init__(self, *args, **kwargs):
        """A Series profile representing the convolution of multiple Series objects and/or
        GSObjects.
        """
        # First check for number of arguments != 0
        if len(args) == 0:
            # No arguments. Could initialize with an empty list but draw then segfaults. Raise an
            # exception instead.
            raise ValueError("Must provide at least one Series or GSObject")
        elif len(args) == 1:
            if isinstance(args[0], (Series, galsim.GSObject)):
                args = [args[0]]
            elif isinstance(args[0], list):
                args = args[0]
            else:
                raise TypeError(
                    "Single input argument must be a Series or GSObject or a list of them.")
        # Check kwargs
        self._gsparams = kwargs.pop("gsparams", None)
        # Make sure there is nothing left in the dict.
        if kwargs:
            raise TypeError("Got unexpected keyword argument(s): %s"%kwargs.keys())
        # Unpack any Convolution of Convolutions.
        self.objlist = []
        for obj in args:
            if isinstance(obj, SeriesConvolution):
                self.objlist.extend([o for o in obj.objlist])
            else:
                self.objlist.append(obj)

        super(SeriesConvolution, self).__init__()

    def _getCoeffs(self):
        # This is a faster, but somewhat more opaque version of
        # return np.multiply.reduce([c for c in product(*[obj._getCoeffs()
        #                                                 for obj in self.objlist])], axis=1)
        # The above use to be the limiting step in some convolutions.  The new version is
        # ~100 times faster, which doesn't mean the convolution is 100 times faster, but now
        # another piece of code is the rate-limiting-step.
        return np.multiply.reduce( np.ix_(*[np.array(obj._getCoeffs())
                                            for obj in self.objlist
                                            if not isinstance(obj, galsim.GSObject)]) ).ravel()

    @staticmethod
    def _getBasisProfiles(objlist, objargs):
        return tuple([galsim.Convolve(*o)
                      for o in product(*[[obj]
                                         if isinstance(obj, galsim.GSObject)
                                         else obj._getBasisProfiles(*objarg)
                                         for obj, objarg in zip(objlist, objargs)])])

    def _getBasisProfileArgs(self):
        if not hasattr(self, '_bpargs'):
            self._objlist = tuple([o if isinstance(o, galsim.GSObject)
                                   else o.__class__
                                   for o in self.objlist])
            self._bpargs = self._objlist, tuple([None if isinstance(o, galsim.GSObject)
                                                 else o._getBasisProfileArgs()
                                                 for o in self.objlist])
        return self._bpargs

    def centroid(self):
        return np.add.reduce([obj.centroid() for obj in self.objlist])


class SpergelSeries(Series):
    def __init__(self, nu, jmax, dlnr=None, half_light_radius=None, scale_radius=None,
                 flux=1.0, gsparams=None, _A=None):
        if half_light_radius is None and scale_radius is None and _A is None:
            raise ValueError("Missing radius parameter")
        self.nu = nu
        self.jmax = jmax
        if dlnr is None:
            dlnr = np.log(1.3)
            # dlnr = np.log(1.3) means that the next precomputed r_i after 1.0 is 1.3.
            # Delta is then restricted to [1-exp(0.5 log(1.3))**2, 1-exp(-0.5 log(1.3))**2]
            #                           = [1-1.3, 1-1./1.3]
            #                           = [-0.3, 0.23] or so...
        self.dlnr = dlnr
        if half_light_radius is not None:
            prof = galsim.Spergel(nu=nu, half_light_radius=half_light_radius)
            scale_radius = prof.getScaleRadius()
        self.flux = flux
        self._gsparams = gsparams

        # Use augmented affine transformation matrix.
        # Store transformation relative to scale_radius=1.
        # I.e., It's possible that Delta is already non-zero.
        if _A is None:
            _A = np.diag(np.array([scale_radius, scale_radius, 1], dtype=float))
        self._A = _A
        super(SpergelSeries, self).__init__()

    def _getCoeffs(self):
        epsilon, phi0, ri, Delta = self._decomposeA()
        coeffs = []
        for j in xrange(self.jmax+1):
            for q in xrange(-j, j+1):
                coeff = 0.0
                for m in range(abs(q), j+1):
                    if (m+q)%2 == 1:
                        continue
                    n = (q+m)/2
                    num = (Delta-1.0)**m
                    # Have to catch 0^0=1 situations...
                    if not (Delta == 0.0 and j==m):
                        num *= Delta**(j-m)
                    if not (epsilon == 0.0 and m==0):
                        num *= epsilon**m
                    den = 2**(m-1) * math.factorial(j-m) * math.factorial(m-n) * math.factorial(n)
                    coeff += num/den
                if q > 0:
                    coeff *= self.flux * (2*q*phi0).cos()
                elif q < 0:
                    coeff *= self.flux * (2*q*phi0).sin()
                else:
                    coeff *= self.flux * 0.5
                coeffs.append(coeff)
        return coeffs

    @staticmethod
    def _getBasisProfiles(nu, jmax, ri, gsp):
        objs = []
        for j in xrange(jmax+1):
            for q in xrange(-j, j+1):
                objs.append(Spergelet(nu=nu, scale_radius=ri, j=j, q=q, gsparams=gsp))
        return objs

    def _getBasisProfileArgs(self):
        if not hasattr(self, '_bpargs'):
            _, _, ri, _ = self._decomposeA()
            self._bpargs = self.nu, self.jmax, ri, self._gsparams
        return self._bpargs

    def centroid(self):
        return galsim.PositionD(self._A[0,2], self._A[1,2])

    def copy(self):
        """Returns a copy of an object.  This preserves the original type of the object."""
        cls = self.__class__
        ret = cls.__new__(cls)
        for k, v in self.__dict__.iteritems():
            ret.__dict__[k] = copy.copy(v)
        return ret

    def _applyMatrix(self, J):
        ret = self.copy()
        ret._A = np.dot(J, self._A)
        if hasattr(ret, 'epsilon'): # reset lazy affine transformation evaluation
            del ret.epsilon
        return ret

    def dilate(self, scale):
        E = np.diag(np.array([scale, scale, 1], dtype=float))
        return self._applyMatrix(E)

    def shear(self, *args, **kwargs):
        if len(args) == 1:
            if kwargs:
                raise TypeError("Gave both unnamed and named arguments!")
            if not isinstance(args[0], galsim.Shear):
                raise TypeError("Unnamed argument is not a Shear!")
            shear = args[0]
        elif len(args) > 1:
            raise TypeError("Too many unnamed arguments!")
        elif 'shear' in kwargs:
            shear = kwargs.pop('shear')
            if kwargs:
                raise TypeError("Too many kwargs provided!")
        else:
            shear = galsim.Shear(**kwargs)
        J = np.identity(3, dtype=float)
        J[0:2, 0:2] = shear.getMatrix()
        return self._applyMatrix(J)

    def rotate(self, theta):
        sth, cth = theta.sincos()
        R = np.matrix([[cth, -sth, 0],
                       [sth,  cth, 0],
                       [  0,    0, 1]],
                      dtype=float)
        return self._applyMatrix(R)

    def shift(self, *args, **kwargs):
        offset = galsim.utilities.parse_pos_args(args, kwargs, 'dx', 'dy')
        ret = self.copy()
        ret._A[0, 2] += offset.x
        ret._A[1, 2] += offset.y
        return ret

    def _decomposeA(self):
        # Use cached result if possible.
        if not hasattr(self, 'epsilon'):
            # dilate corresponds to:
            # D(mu) = [ mu 0 ]
            #         [ 0 mu ]
            #
            # rotate corresponds to:
            # R(phi0) = [ cph  -sph ]
            #           [ sph   cph ]
            # where cph = cos(phi0), sph = sin(phi0)
            #
            # shear along x corresponds to:
            # S(eta) = [ exp(eta/2) 0           ]
            #          [ 0          exp(-eta/2) ]
            # where eta is the conformal shear; g = tanh(eta/2) and b/a = exp(-eta)
            #
            # shear with arbitrary phase 2 beta (i.e. along angle beta wrt the +x axis)
            # can be generated from a composition of rotates and x-axis shears.
            # The composition dilate(mu) x rotate(phi0) x shear(eta, beta) yields:
            #      [ a  b ]
            # mu x [ c  d ]
            # a =  cos(phi0) cosh(eta/2) + cos(2 beta + phi0) sinh(eta/2)
            # b = -sin(phi0) cosh(eta/2) + sin(2 beta + phi0) sinh(eta/2)
            # c =  sin(phi0) cosh(eta/2) + sin(2 beta + phi0) sinh(eta/2)
            # d =  cos(phi0) cosh(eta/2) - cos(2 beta + phi0) sinh(eta/2)
            #
            # To decompose A then, we work backwards.  The determinant of the above yields
            # det(A) = mu^2
            # The following can then be used to get phi0, 2 beta, and eta.
            # a + d =  2 mu cos(phi0) cosh(eta/2)
            # b + c =  2 mu sin(2 beta + phi0) sinh(eta/2)
            # a - d =  2 mu cos(2 beta + phi0) sinh(eta/2)
            # b - c = -2 mu sin(phi0) cosh(eta/2)
            # (a-d)**2 + (b+c)**2 = 2 mu**2 (cosh(eta/2) - 1)

            A = self._A
            a = A[0, 0]
            b = A[0, 1]
            c = A[1, 0]
            d = A[1, 1]

            musqr = a*d-b*c
            mu = math.sqrt(musqr)
            phi0 = math.atan2(c-b, a+d) * galsim.radians
            twobetaphi0 = math.atan2(b+c, a-d) * galsim.radians
            beta = 0.5 * (twobetaphi0 - phi0)
            eta = math.acosh(0.5*((a-d)**2 + (b+c)**2)/musqr+1.0)

            # print "mu: {}".format(mu)
            # print "phi0: {}".format(phi0.wrap())
            # print "2 beta: {}".format((2*beta).wrap())
            # print "eta: {}".format(eta)

            # Note that the galsim.shear() operation leaves a*b constant where a and b are the
            # semi-major semi-minor axes of a sheared elliptical isophote.
            # In contrast, holding Delta constant and setting a non-zero epsilon in the Spergel
            # expansion leaves a**2 + b**2 constant, and ab -> ab (1-epsilon**2).
            # So our task is to figure out what a**2 + b**2 is given the params above.
            # Some algebra:
            #    r0**2 = a**2 + b**2                   re**2 = a b
            #    q = b/a        e = (1-q**2) / (1 + q**2)  =>  q**2 = (1-e)/(1+e)
            #    re/sqrt(q) = sqrt(a b) * sqrt(a/b) = sqrt(a*a) = a
            #    re*sqrt(q) = sqrt(a b) * sqrt(b/a) = sqrt(b*b) = b
            #    r0**2 = re**2/q + re**2 * q = re**2 * (1/q + q)
            #    more algebra reveals:
            #    1+q + q = sqrt((1-e)/(1+e)) + sqrt((1+e)/(1-e)) = 2/sqrt(1-e**2)
            #    Since r0**2 = 2.0 for scale_radius=1 (i.e., a = b = 1),
            #    this is a linear factor of 1/sqrt(sqrt(1-e**2)).

            self.epsilon = galsim.Shear(eta1=eta).e1
            r0 = mu / math.sqrt(math.sqrt(1.0 - self.epsilon**2))
            # print "r0: {}".format(r0)

            # find the nearest r_i, assume that one of the r_i is 1.0 (log r_i = 0):
            f, i = np.modf(np.log(r0)/self.dlnr)
            # round up to nearest
            if f < -0.5:
                f += 1.0
                i -= 1
            # round down to nearest
            if f > 0.5:
                f -= 1.0
                i += 1
            self.ri = np.exp(self.dlnr*i)
            # print "i: {}".format(i)
            # print "ri: {}".format(self.ri)
            self.Delta = 1.0 - (r0/self.ri)**2
            # print "Delta: {}".format(self.Delta)
            self.phi0 = phi0+beta # This works because initial profile is azimuthally symmetric
        return self.epsilon, self.phi0, self.ri, self.Delta

    def __repr__(self):
        s = 'galsim.SpergelSeries(nu=%r, jmax=%r, flux=%r, dlnr=%r, _A=%r'%(
            self.nu, self.jmax, self.flux, self.dlnr, self._A)
        s += ', gsparams=%r)'%(self._gsparams)
        return s

    def __str__(self):
        s = 'galsim.SpergelSeries(nu=%s, jmax=%s'%(self.nu, self.jmax)
        if self.flux != 1.0:
            s += ', flux=%s'%self.flux
        s += ')'
        return s

class Spergelet(galsim.GSObject):
    """A basis function in the Taylor series expansion of the Spergel profile.

    @param nu               The Spergel index, nu.
    @param scale_radius     The scale radius of the profile.  Typically given in arcsec.
    @param j                Radial index.
    @param q                Azimuthal index.
    @param gsparams         An optional GSParams argument.  See the docstring for GSParams for
                            details. [default: None]
    """
    def __init__(self, nu, scale_radius, j, q, gsparams=None):
        galsim.GSObject.__init__(
            self, galsim._galsim.SBSpergelet(nu, scale_radius, j, q, gsparams=gsparams))

    def getNu(self):
        """Return the Spergel index `nu` for this profile.
        """
        return self.SBProfile.getNu()

    def getScaleRadius(self):
        """Return the scale radius for this Spergel profile.
        """
        return self.SBProfile.getScaleRadius()

    def getJQ(self):
        """Return the jq indices for this Spergelet.
        """
        return self.SBProfile.getJ(), self.SBProfile.getQ()


class MoffatSeries(Series):
    def __init__(self, beta, jmax, dlnr=None,
                 half_light_radius=None, scale_radius=None, fwhm=None,
                 flux=1.0, gsparams=None, _A=None):
        if half_light_radius is None and scale_radius is None and fwhm is None and _A is None:
            raise ValueError("Missing radius parameter")
        self.beta = beta
        self.jmax = jmax
        if dlnr is None:
            dlnr = np.log(1.3)
        self.dlnr = dlnr
        if half_light_radius is not None:
            prof = galsim.Moffat(beta=beta, half_light_radius=half_light_radius)
            scale_radius = prof.getScaleRadius()
        elif fwhm is not None:
            prof = galsim.Moffat(beta=beta, fwhm=fwhm)
            scale_radius = prof.getScaleRadius()
        self.flux = flux
        self._gsparams = gsparams

        # Store transformation relative to scale_radius=1.
        if _A is None:
            _A = np.diag(np.array([scale_radius, scale_radius, 1.0], dtype=float))
        self._A = _A

        super(MoffatSeries, self).__init__()

    def _getCoeffs(self):
        ellip, phi0, scale_radius, Delta = self._decomposeA()
        coeffs = []
        for j in xrange(self.jmax+1):
            for q in xrange(-j, j+1):
                coeff = 0.0
                for m in range(abs(q), j+1):
                    if (m+q)%2 == 1:
                        continue
                    n = (q+m)/2
                    num = (1-Delta)**(m+1)
                    # Have to catch 0^0=1 situations...
                    if not (Delta == 0.0 and j==m):
                        num *= Delta**(j-m)
                    if not (ellip == 0.0 and m==0):
                        num *= ellip**m
                    den = 2**(m-1) * math.factorial(j-m) * math.factorial(m-n) * math.factorial(n)
                    coeff += num/den
                if q > 0:
                    coeff *= self.flux * math.sqrt(1.0 - ellip**2) * math.cos(2*q*phi0)
                elif q < 0:
                    coeff *= self.flux * math.sqrt(1.0 - ellip**2) * math.sin(2*q*phi0)
                else:
                    coeff *= self.flux * math.sqrt(1.0 - ellip**2) * 0.5
                coeffs.append(coeff)
        return coeffs

    @staticmethod
    def _getBasisProfiles(beta, jmax, ri, gsp):
        objs = []
        for j in xrange(jmax+1):
            for q in xrange(-j, j+1):
                objs.append(Moffatlet(beta=beta, scale_radius=ri,
                                      j=j, q=q, gsparams=gsp))
        return objs

    def _getbasisProfileArgs(self):
        if not hasattr(self, '_bpargs'):
            _, _, ri, _ = self._decomposeA()
            self._bpargs = self.nu, self.jmax, ri, self._gsparams
        return self._bpargs

    def centroid(self):
        return galsim.PositionD(self._A[0,2], self._A[1,2])

    def copy(self):
        """Returns a copy of an object.  This preserves the original type of the object."""
        cls = self.__class__
        ret = cls.__new__(cls)
        for k, v in self.__dict__.iteritems():
            ret.__dict__[k] = copy.copy(v)
        return ret

    def _applyMatrix(self, J):
        ret = self.copy()
        ret._A = np.dot(J, self._A)
        if hasattr(ret, 'ellip'): # reset lazy affine transformation evaluation
            del ret.ellip
        return ret

    def dilate(self, scale):
        E = np.diag(np.array([scale, scale, 1.0], dtype=float))
        return self._applyMatrix(E)

    def shear(self, *args, **kwargs):
        if len(args) == 1:
            if kwargs:
                raise TypeError("Gave both unnamed and named arguments!")
            if not isinstance(args[0], galsim.Shear):
                raise TypeError("Unnamed argument is not a Shear!")
            shear = args[0]
        elif len(args) > 1:
            raise TypeError("Too many unnamed arguments!")
        elif 'shear' in kwargs:
            shear = kwargs.pop('shear')
            if kwargs:
                raise TypeError("Too many kwargs provided!")
        else:
            shear = galsim.Shear(**kwargs)
        J = np.identity(3, dtype=float)
        J[0:2, 0:2] = shear.getMatrix()
        return self._applyMatrix(J)

    def rotate(self, theta):
        sth, cth = theta.sincos()
        R = np.matrix([[cth, -sth, 0],
                       [sth,  cth, 0],
                       [  0,    0, 1]],
                      dtype=float)
        return self._applyMatrix(R)

    def shift(self, *args, **kwargs):
        offset = galsim.utilities.parse_pos_args(args, kwargs, 'dx', 'dy')
        ret = self.copy()
        ret._A[0, 2] += offset.x
        ret._A[1, 2] += offset.y
        return ret

    def _decomposeA(self):
        if not hasattr(self, 'ellip'):
            A = self._A
            a = A[0,0]
            b = A[0,1]
            c = A[1,0]
            d = A[1,1]
            mu = math.sqrt(a*d-b*c)
            phi0 = math.atan2(c-b, a+d)
            beta = math.atan2(b+c, a-d)+phi0
            eta = math.acosh(0.5*((a-d)**2 + (b+c)**2)/mu**2 + 1.0)

            # print "mu: {}".format(mu)
            # print "phi0: {}".format(phi0)
            # print "beta: {}".format(beta)
            # print "eta: {}".format(eta)

            ellip = galsim.Shear(eta1=eta).e1
            r0 = mu/np.sqrt(np.sqrt(1.0 - ellip**2))
            # find the nearest r_i:
            f, i = np.modf(np.log(r0)/self.dlnr)
            # deal with negative logs
            if f < 0.0:
                f += 1.0
                i -= 1
            # if f > 0.5, then round down from above
            if f > 0.5:
                f -= 1.0
                i += 1
            scale_radius = np.exp(self.dlnr*i)
            Delta = 1.0 - (scale_radius/r0)**2
            self.ellip = ellip
            self.scale_radius = scale_radius
            self.phi0 = phi0+beta/2
            self.Delta = Delta
        return self.ellip, self.phi0, self.scale_radius, self.Delta

    def __repr__(self):
        s = 'galsim.MoffatSeries(beta=%r, jmax=%r, flux=%r, dlnr=%r, _A=%r'%(
            self.beta, self.jmax, self.flux, self.dlnr, self._A)
        s += ', gsparams=%r)'%(self._gsparams)
        return s

    def __str__(self):
        s = 'galsim.MoffatSeries(beta=%s, jmax=%s'%(self.beta, self.jmax)
        if self.flux != 1.0:
            s += ', flux=%s'%self.flux
        s += ')'
        return s


class Moffatlet(galsim.GSObject):
    """A basis function in the Taylor series expansion of the Moffat profile.

    @param beta             The Moffat index, beta.
    @param scale_radius     The scale radius of the profile.  Typically given in arcsec.
    @param j                Radial index.
    @param q                Azimuthal index.
    @param gsparams         An optional GSParams argument.  See the docstring for GSParams for
                            details. [default: None]
    """
    def __init__(self, beta, scale_radius, j, q, gsparams=None):
        galsim.GSObject.__init__(
            self, galsim._galsim.SBMoffatlet(beta, scale_radius, j, q, gsparams=gsparams))

    def getBeta(self):
        """Return the Moffat index `beta` for this profile.
        """
        return self.SBProfile.getBeta()

    def getScaleRadius(self):
        """Return the scale radius for this Moffat profile.
        """
        return self.SBProfile.getScaleRadius()

    def getJQ(self):
        """Return the jq indices for this Moffatlet.
        """
        return self.SBProfile.getJ(), self.SBProfile.getQ()


def nollZernikeDict(jmax):
    # Return dictionary for Noll Zernike convention taking index j -> n,m.
    ret = {}
    j=1
    for n in count():
        for m in xrange(n+1):
            if (m+n) % 2 == 1:
                continue
            if m == 0:
                ret[j] = (n, m)
                j += 1
            elif j % 2 == 0:
                ret[j] = (n, m)
                ret[j+1] = (n, -m)
                j += 2
            else:
                ret[j] = (n, -m)
                ret[j+1] = (n, m)
                j += 2
            if j > jmax:
                return ret
    return ret

nollZernike = nollZernikeDict(100)


class LinearOpticalSeries(Series):
    def __init__(self, lam_over_diam=None, lam=None, diam=None, flux=1.0,
                 tip=0., tilt=0., defocus=0., astig1=0., astig2=0.,
                 coma1=0., coma2=0., trefoil1=0., trefoil2=0., spher=0.,
                 aberrations=None, scale_unit=galsim.arcsec, gsparams=None):
        """ A series representation of the PSF generated by a series expansion of the entrance
        pupil wavefront to linear order.  Works by approximating exp(i phi) ~ 1 + i phi, where phi is
        the wavefront.  Should probably only use if the amplitude of the Zernike coefficients is less
        than 0.3 wavelengths or so, depending on the desired accuracy.
        """
        if lam_over_diam is not None:
            if lam is not None or diam is not None:
                raise TypeError("If specifying lam_over_diam, then do not specify lam or diam")
        else:
            if lam is None or diam is None:
                raise TypeError("If not specifying lam_over_diam, then specify lam AND diam")
                if isinstance(scale_unit, basestring):
                    scale_unit = galsim.angle.get_angle_unit(scale_unit)
                lam_over_diam = (1.e-9*lam/diam)*(galsim.radians/scale_unit)
        self.lam_over_diam = lam_over_diam
        self.flux = flux
        self._gsparams = gsparams

        if aberrations is None:
            # Repackage the aberrations into a single array.
            aberrations = np.zeros(12)
            aberrations[2] = tip
            aberrations[3] = tilt
            aberrations[4] = defocus
            aberrations[5] = astig1
            aberrations[6] = astig2
            aberrations[7] = coma1
            aberrations[8] = coma2
            aberrations[9] = trefoil1
            aberrations[10] = trefoil2
            aberrations[11] = spher
            # Clip at largest non-zero aberration (or 2)
            nz = aberrations.nonzero()[0]
            if len(nz) == 0:
                i = 2
            else:
                i = max([nz[-1]+1, 2])
            aberrations = aberrations[:i]
        else:
            # Aberrations were passed in, so check that there are the right number of entries.
            # Note that while we do have a lower limit here, there is no upper limit (in contrast
            # to OpticalPSF).
            if len(aberrations) <= 2:
                raise ValueError("Aberrations keyword must have length > 2")
            # Make sure no individual ones were passed in, since they will be ignored.
            if np.any(
                np.array([tip, tilt, defocus, astig1, astig2,
                          coma1, coma2, trefoil1, trefoil2, spher]) != 0):
                raise TypeError("Cannot pass in individual aberrations and array!")
        # Finally, just in case it was a tuple/list, make sure we end up with NumPy array:
        self.aberrations = np.array(aberrations) * 2.0 * np.pi

        def norm(n, m):
            if m==0:
                return np.sqrt(n+1)
            else:
                return np.sqrt(2*n+2)

        # Now separate into terms that are pure real and pure imag in the complex PSF
        self.realcoeffs = [1.0]
        self.imagcoeffs = []
        self.realindices = [(0,0)]
        self.imagindices = []
        for j, ab in enumerate(self.aberrations):
            if j < 2:
                continue
            n,m = nollZernike[j]
            ipower = (m+1)%4
            if ipower == 0:
                self.realcoeffs.append(norm(n,m)*ab)
                self.realindices.append((n,m))
            elif ipower == 1:
                self.imagcoeffs.append(norm(n,m)*ab)
                self.imagindices.append((n, m))
            elif ipower == 2:
                self.realcoeffs.append(-norm(n,m)*ab)
                self.realindices.append((n,m))
            elif ipower == 3:
                self.imagcoeffs.append(-norm(n,m)*ab)
                self.imagindices.append((n, m))
            else:
                raise RuntimeError("What!?  How'd that happen?")
        # First do the square coefficients
        self.coeffs = [c**2 for c in self.realcoeffs]
        self.coeffs.extend([c**2 for c in self.imagcoeffs])
        # Now handle the cross-coefficients
        self.coeffs.extend([2*np.multiply.reduce(c) for c in combinations(self.realcoeffs, 2)])
        self.coeffs.extend([2*np.multiply.reduce(c) for c in combinations(self.imagcoeffs, 2)])

        # Do the same with the indices
        self.indices = [(i, i) for i in self.realindices]
        self.indices.extend([(i, i) for i in self.imagindices])
        self.indices.extend([i for i in combinations(self.realindices, 2)])
        self.indices.extend([i for i in combinations(self.imagindices, 2)])

        super(LinearOpticalSeries, self).__init__()

    def _getCoeffs(self):
        return self.coeffs

    @staticmethod
    def _getBasisProfiles(lam_over_diam, indices, gsp):
        return [LinearOpticalet(lam_over_diam, o[0][0], o[0][1], o[1][0], o[1][1],
                                gsparams=gsp)
                for o in self.indices]

    def _getBasisProfileArgs(self):
        if not hasattr(self, '_bpargs'):
            self._bpargs = self.lam_over_diam, self.indices, self._gsparams
        return self._bpargs

    def __repr__(self):
        s = 'galsim.LinearOpticalSeries(lam_over_diam=%r, flux=%r, aberrations=%r, gsparams=%r)'%(
            self.lam_over_diam, self.flux, self.aberrations, self._gsparams)
        return s

    def __str__(self):
        s = 'galsim.LinearOpticalSeries(lam_over_diam=%s'%self.lam_over_diam
        if self.flux != 1.0:
            s += ', flux=%s'%self.flux
        if sum(self.aberrations.nonzero()[0]) > 0:
            s += ', aberrations=%s'%self.aberrations
        s += ')'
        return s

    def centroid(self):
        return galsim.PositionD(0.0, 0.0)

class LinearOpticalet(galsim.GSObject):
    """A basis function in the Taylor series expansion of the optical wavefront.

    @param scale_radius   Set the size.
    @param n1             First radial index.
    @param m1             First azimuthal index.
    @param n2             Second radial index.
    @param m2             Second azimuthal index.
    """
    def __init__(self, scale_radius, n1, m1, n2, m2, gsparams=None):
        galsim.GSObject.__init__(
            self, galsim._galsim.SBLinearOpticalet(scale_radius,
                                                   n1, m1, n2, m2, gsparams=gsparams))

    def getScaleRadius(self):
        return self.SBProfile.getScaleRadius()

    def getIndices(self):
        return (self.SBProfile.getN1(), self.SBProfile.getM1(),
                self.SBProfile.getN2(), self.SBProfile.getM2())
