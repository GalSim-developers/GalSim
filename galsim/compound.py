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
"""@file compound.py
Some compound GSObject classes that contain other GSObject instances:

Sum = sum of multiple profiles
Convolution = convolution of multiple profiles
Deconvolution = deconvolution by a given profile
AutoConvolution = convolution of a profile by itself
AutoCorrelation = convolution of a profile by its reflection
FourierSqrt = Fourier-space square root of a profile
"""

import galsim
from . import _galsim

def Add(*args, **kwargs):
    """A function for adding 2 or more GSObject or ChromaticObject instances.

    This function will inspect its input arguments to decide if a Sum object or a
    ChromaticSum object is required to represent the sum of surface brightness profiles.

    Typically, you do not need to call Add() explicitly.  Normally, you would just use the + 
    operator, which returns a Sum:

        >>> bulge = galsim.Sersic(n=3, half_light_radius=0.8)
        >>> disk = galsim.Exponential(half_light_radius=1.4)
        >>> gal = bulge + disk
        >>> psf = galsim.Gaussian(sigma=0.3, flux=0.3) + galsim.Gaussian(sigma=0.8, flux=0.7)

    If one of the items is chromatic, it will return a ChromaticSum

        >>> disk = galsim.Exponential(half_light_radius=1.4) * galsim.SED(sed_file)
        >>> gal = bulge + disk

    @param args             Unnamed args should be a list of objects to add.
    @param gsparams         An optional GSParams argument.  See the docstring for GSParams for
                            details. [default: None]

    @returns a Sum or ChromaticSum instance as appropriate.
    """
    if len(args) == 0:
        raise ValueError("At least one ChromaticObject or GSObject must be provided.")
    elif len(args) == 1:
        # 1 argument.  Should be either a GSObject or a list of GSObjects
        if isinstance(args[0], (galsim.GSObject, galsim.ChromaticObject)):
            args = [args[0]]
        elif isinstance(args[0], list) or isinstance(args[0], tuple):
            args = args[0]
        else:
            raise TypeError("Single input argument must be a GSObject, ChromaticObject or "
                            + "a (possibly mixed) list of them.")
    # else args is already the list of objects

    if any([isinstance(a, galsim.ChromaticObject) for a in args]):
        return galsim.ChromaticSum(*args, **kwargs)
    else:
        return Sum(*args, **kwargs)


class Sum(galsim.GSObject):
    """A class for adding 2 or more GSObject instances.

    The Sum class is used to represent the sum of multiple GSObject instances.  For example, it
    might be used to represent a multiple-component galaxy as the sum of an Exponential and a
    DeVaucouleurs, or to represent a PSF as the sum of multiple Gaussian objects.

    Initialization
    --------------

    Typically, you do not need to construct a Sum object explicitly.  Normally, you would just
    use the + operator, which returns a Sum:

        >>> bulge = galsim.Sersic(n=3, half_light_radius=0.8)
        >>> disk = galsim.Exponential(half_light_radius=1.4)
        >>> gal = bulge + disk
        >>> psf = galsim.Gaussian(sigma=0.3, flux=0.3) + galsim.Gaussian(sigma=0.8, flux=0.7)

    You can also use the Add() factory function, which returns a Sum object if none of the
    individual objects are chromatic:

        >>> gal = galsim.Add([bulge,disk])

    @param args             Unnamed args should be a list of objects to add.
    @param gsparams         An optional GSParams argument.  See the docstring for GSParams for
                            details. [default: None]

    Note: if `gsparams` is unspecified (or None), then the Sum instance inherits the same GSParams
    as the first item in the list.  Also, note that parameters related to the Fourier-space
    calculations must be set when initializing the individual GSObject instances that go into the
    Sum, NOT when creating the Sum (at which point the accuracy and threshold parameters will simply
    be ignored).

    Methods
    -------

    There are no additional methods for Sum beyond the usual GSObject methods.
    """
    def __init__(self, *args, **kwargs):

        # Check kwargs first:
        gsparams = kwargs.pop("gsparams", None)
        self._gsparams = gsparams

        # Make sure there is nothing left in the dict.
        if kwargs:
            raise TypeError(
                "Sum constructor got unexpected keyword argument(s): %s"%kwargs.keys())

        if len(args) == 0:
            raise ValueError("At least one ChromaticObject or GSObject must be provided.")
        elif len(args) == 1:
            # 1 argument.  Should be either a GSObject or a list of GSObjects
            if isinstance(args[0], galsim.GSObject):
                args = [args[0]]
            elif isinstance(args[0], list) or isinstance(args[0], tuple):
                args = args[0]
            else:
                raise TypeError("Single input argument must be a GSObject or list of them.")
        # else args is already the list of objects

        # Save the list as an attribute, so it can be inspected later if necessary.
        self._obj_list = args

        if len(args) == 1 and gsparams is None:
            # No need to make an SBAdd in this case.
            galsim.GSObject.__init__(self, args[0])
            if hasattr(args[0],'noise'):
                self.noise = args[0].noise
        else:
            # If any of the objects have a noise attribute, then we propagate the sum of the
            # noises (they add like variances) to the final sum.
            noise = None
            for obj in args:
                if hasattr(obj,'noise'):
                    if noise is None:
                        noise = obj.noise
                    else:
                        noise += obj.noise
            SBList = [obj.SBProfile for obj in args]
            galsim.GSObject.__init__(self, galsim._galsim.SBAdd(SBList, gsparams))
            if noise is not None:
                self.noise = noise

    @property
    def obj_list(self): return self._obj_list

    def __repr__(self):
        return 'galsim.Sum(%r, gsparams=%r)'%(self.obj_list, self._gsparams)

    def __str__(self):
        str_list = [ str(obj) for obj in self.obj_list ]
        return '(' + ' + '.join(str_list) + ')'
        #return 'galsim.Sum([%s])'%', '.join(str_list)

    def __getstate__(self):
        d = self.__dict__.copy()
        del d['SBProfile']
        return d

    def __setstate__(self, d):
        self.__dict__ = d
        self.__init__(self._obj_list, gsparams=self._gsparams)


_galsim.SBAdd.__getinitargs__ = lambda self: (self.getObjs(), self.getGSParams())
_galsim.SBAdd.__getstate__ = lambda self: None
_galsim.SBAdd.__setstate__ = lambda self, state: 1
_galsim.SBAdd.__repr__ = lambda self: \
        'galsim._galsim.SBAdd(%r, %r)'%self.__getinitargs__()


def Convolve(*args, **kwargs):
    """A function for convolving 2 or more GSObject or ChromaticObject instances.

    This function will inspect its input arguments to decide if a Convolution object or a
    ChromaticConvolution object is required to represent the convolution of surface
    brightness profiles.

    @param args             Unnamed args should be a list of objects to convolve.
    @param real_space       Whether to use real space convolution.  [default: None, which means
                            to automatically decide this according to whether the objects have hard
                            edges.]
    @param gsparams         An optional GSParams argument.  See the docstring for GSParams for
                            details. [default: None]

    @returns a Convolution or ChromaticConvolution instance as appropriate.
    """
    # First check for number of arguments != 0
    if len(args) == 0:
        raise ValueError("At least one ChromaticObject or GSObject must be provided.")
    elif len(args) == 1:
        if isinstance(args[0], (galsim.GSObject, galsim.ChromaticObject)):
            args = [args[0]]
        elif isinstance(args[0], list) or isinstance(args[0], tuple):
            args = args[0]
        else:
            raise TypeError("Single input argument must be a GSObject, ChromaticObject, "
                            + "or a (possibly mixed) list of them.")
    # else args is already the list of objects

    if any([isinstance(a, galsim.ChromaticObject) for a in args]):
        return galsim.ChromaticConvolution(*args, **kwargs)
    else:
        return Convolution(*args, **kwargs)


class Convolution(galsim.GSObject):
    """A class for convolving 2 or more GSObject instances.

    The convolution will normally be done using discrete Fourier transforms of each of the component
    profiles, multiplying them together, and then transforming back to real space.

    There is also an option to do the convolution as integrals in real space.  To do this, use the
    optional keyword argument `real_space = True`.  Currently, the real-space integration is only
    enabled for convolving 2 profiles.  (Aside from the trivial implementaion for 1 profile.)  If
    you try to use it for more than 2 profiles, an exception will be raised.

    The real-space convolution is normally slower than the DFT convolution.  The exception is if
    both component profiles have hard edges, e.g. a truncated Moffat or Sersic with a Pixel.  In
    that case, the highest frequency `maxK` for each component is quite large since the ringing dies
    off fairly slowly.  So it can be quicker to use real-space convolution instead.  Also,
    real-space convolution tends to be more accurate in this case as well.

    If you do not specify either `real_space = True` or `False` explicitly, then we check if there
    are 2 profiles, both of which have hard edges.  In this case, we automatically use real-space
    convolution.  In all other cases, the default is not to use real-space convolution.

    Initialization
    --------------

    The normal way to use this class is to use the Convolve() factory function:

        >>> gal = galsim.Sersic(n, half_light_radius)
        >>> psf = galsim.Gaussian(sigma)
        >>> final = galsim.Convolve([gal, psf])

    The objects to be convolved may be provided either as multiple unnamed arguments (e.g.
    `Convolve(psf, gal)`) or as a list (e.g. `Convolve([psf, gal])`).  Any number of objects may
    be provided using either syntax.  (Well, the list has to include at least 1 item.)

    @param args             Unnamed args should be a list of objects to convolve.
    @param real_space       Whether to use real space convolution.  [default: None, which means
                            to automatically decide this according to whether the objects have hard
                            edges.]
    @param gsparams         An optional GSParams argument.  See the docstring for GSParams for
                            details. [default: None]

    Note: if `gsparams` is unspecified (or None), then the Convolution instance inherits the same 
    GSParams as the first item in the list.  Also, note that parameters related to the Fourier-
    space calculations must be set when initializing the individual GSObjects that go into the Sum,
    NOT when creating the Sum (at which point the accuracy and threshold parameters will simply be
    ignored).

    Methods
    -------

    There are no additional methods for Convolution beyond the usual GSObject methods.
    """
    def __init__(self, *args, **kwargs):
        # First check for number of arguments != 0
        if len(args) == 0:
            raise ValueError("At least one ChromaticObject or GSObject must be provided.")
        elif len(args) == 1:
            if isinstance(args[0], galsim.GSObject):
                args = [args[0]]
            elif isinstance(args[0], list) or isinstance(args[0], tuple):
                args = args[0]
            else:
                raise TypeError("Single input argument must be a GSObject or list of them.")
        # else args is already the list of objects

        # Check kwargs
        # real_space can be True or False (default if omitted is None), which specifies whether to
        # do the convolution as an integral in real space rather than as a product in fourier
        # space.  If the parameter is omitted (or explicitly given as None I guess), then
        # we will usually do the fourier method.  However, if there are 2 components _and_ both of
        # them have hard edges, then we use real-space convolution.
        real_space = kwargs.pop("real_space", None)
        gsparams = kwargs.pop("gsparams", None)
        self._gsparams = gsparams

        # Make sure there is nothing left in the dict.
        if kwargs:
            raise TypeError(
                "Convolution constructor got unexpected keyword argument(s): %s"%kwargs.keys())

        if len(args) == 1 and gsparams is None:
            # No need to make an SBConvolve in this case.  Can early exit.
            galsim.GSObject.__init__(self, args[0])
            if hasattr(args[0],'noise'):
                self.noise = args[0].noise
            self._real_space = real_space
            self._obj_list = args
            return

        # Check whether to perform real space convolution...
        # Start by checking if all objects have a hard edge.
        hard_edge = True
        for obj in args:
            if not obj.hasHardEdges():
                hard_edge = False

        if real_space is None:
            # The automatic determination is to use real_space if 2 items, both with hard edges.
            if len(args) == 2:
                real_space = hard_edge
            else:
                real_space = False

        # Warn if doing DFT convolution for objects with hard edges.
        if not real_space and hard_edge:
            import warnings
            if len(args) == 2:
                msg = """
                Doing convolution of 2 objects, both with hard edges.
                This might be more accurate and/or faster using real_space=True"""
            else:
                msg = """
                Doing convolution where all objects have hard edges.
                There might be some inaccuracies due to ringing in k-space."""
            warnings.warn(msg)

        if real_space:
            # Can't do real space if nobj > 2
            if len(args) > 2:
                import warnings
                msg = """
                Real-space convolution of more than 2 objects is not implemented.
                Switching to DFT method."""
                warnings.warn(msg)
                real_space = False

            # Also can't do real space if any object is not analytic, so check for that.
            else:
                for obj in args:
                    if not obj.isAnalyticX():
                        import warnings
                        msg = """
                        A component to be convolved is not analytic in real space.
                        Cannot use real space convolution.
                        Switching to DFT method."""
                        warnings.warn(msg)
                        real_space = False
                        break

        # If one of the objects has a noise attribute, then we convolve it by the others.
        # More than one is not allowed.
        noise = None
        noise_convolve = []
        for obj in args:
            if hasattr(obj,'noise'):
                if noise is not None:
                    import warnings
                    warnings.warn("Unable to propagate noise in galsim.Convolution when multiple "+
                                  "objects have noise attribute")
                    noise = None
                    break
                noise = obj.noise
                others = [ obj2 for obj2 in args if obj2 is not obj ]
                assert len(others) > 0
                if len(others) == 1: 
                    noise = noise.convolvedWith(others[0])
                else: 
                    noise = noise.convolvedWith(galsim.Convolve(others))

        # Save the construction parameters (as they are at this point) as attributes so they
        # can be inspected later if necessary.
        self._real_space = real_space
        self._obj_list = args

        # Then finally initialize the SBProfile using the objects' SBProfiles.
        SBList = [ obj.SBProfile for obj in args ]
        sbp = galsim._galsim.SBConvolve(SBList, real_space, gsparams)
        galsim.GSObject.__init__(self, sbp)
        if noise is not None:
            self.noise = noise

    @property
    def obj_list(self): return self._obj_list
    @property
    def real_space(self): return self._real_space

    def __repr__(self):
        return 'galsim.Convolution(%r, real_space=%r, gsparams=%r)'%(
                self.obj_list, self.real_space, self._gsparams)

    def __str__(self):
        str_list = [ str(obj) for obj in self.obj_list ]
        s = 'galsim.Convolve(%s'%(', '.join(str_list))
        if self.real_space:
            s += ', real_space=True'
        s += ')'
        return s

    def __getstate__(self):
        d = self.__dict__.copy()
        del d['SBProfile']
        return d

    def __setstate__(self, d):
        self.__dict__ = d
        self.__init__(self._obj_list, real_space=self._real_space, gsparams=self._gsparams)


_galsim.SBConvolve.__getinitargs__ = lambda self: (
        self.getObjs(), self.isRealSpace(), self.getGSParams())
_galsim.SBConvolve.__getstate__ = lambda self: None
_galsim.SBConvolve.__setstate__ = lambda self, state: 1
_galsim.SBConvolve.__repr__ = lambda self: \
        'galsim._galsim.SBConvolve(%r, %r, %r)'%self.__getinitargs__()


def Deconvolve(obj, gsparams=None):
    """A function for deconvolving by either a GSObject or ChromaticObject.

    This function will inspect its input argument to decide if a Deconvolution object or a
    ChromaticDeconvolution object is required to represent the deconvolution by a surface
    brightness profile.

    @param obj              The object to deconvolve.
    @param gsparams         An optional GSParams argument.  See the docstring for GSParams for
                            details. [default: None]

    @returns a Deconvolution or ChromaticDeconvolution instance as appropriate.
    """
    if isinstance(obj, galsim.ChromaticObject):
        return galsim.ChromaticDeconvolution(obj, gsparams=gsparams)
    elif isinstance(obj, galsim.GSObject):
        return Deconvolution(obj, gsparams=gsparams)
    else:
        raise TypeError("Argument to Deconvolve must be either a GSObject or a ChromaticObject.")


class Deconvolution(galsim.GSObject):
    """A class for deconvolving a GSObject.

    The Deconvolution class represents a deconvolution kernel.  Note that the Deconvolution class,
    or compound objects (Sum, Convolution) that include a Deconvolution as one of the components,
    cannot be photon-shot using the 'phot' method of drawImage() method.

    You may also specify a `gsparams` argument.  See the docstring for GSParams using
    `help(galsim.GSParams)` for more information about this option.  Note: if `gsparams` is
    unspecified (or None), then the Deconvolution instance inherits the same GSParams as the object
    being deconvolved.

    Initialization
    --------------

    The normal way to use this class is to use the Deconvolve() factory function:

        >>> inv_psf = galsim.Deconvolve(psf)
        >>> deconv_gal = galsim.Convolve(inv_psf, gal)

    @param obj              The object to deconvolve.
    @param gsparams         An optional GSParams argument.  See the docstring for GSParams for
                            details. [default: None]

    Methods
    -------

    There are no additional methods for Deconvolution beyond the usual GSObject methods.
    """
    def __init__(self, obj, gsparams=None):
        if not isinstance(obj, galsim.GSObject):
            raise TypeError("Argument to Deconvolution must be a GSObject.")

        # Save the original object as an attribute, so it can be inspected later if necessary.
        self._orig_obj = obj
        self._gsparams = gsparams

        sbp = galsim._galsim.SBDeconvolve(obj.SBProfile, gsparams)
        galsim.GSObject.__init__(self, sbp)
        if hasattr(obj,'noise'):
            import warnings
            warnings.warn("Unable to propagate noise in galsim.Deconvolution")

    @property
    def orig_obj(self): return self._orig_obj

    def __repr__(self):
        return 'galsim.Deconvolution(%r, gsparams=%r)'%(self.orig_obj, self._gsparams)

    def __str__(self):
        return 'galsim.Deconvolve(%s)'%self.orig_obj

    def __getstate__(self):
        d = self.__dict__.copy()
        del d['SBProfile']
        return d

    def __setstate__(self, d):
        self.__dict__ = d
        self.__init__(self._orig_obj, self._gsparams)



_galsim.SBDeconvolve.__getinitargs__ = lambda self: (self.getObj(), self.getGSParams())
_galsim.SBDeconvolve.__getstate__ = lambda self: None
_galsim.SBDeconvolve.__setstate__ = lambda self, state: 1
_galsim.SBDeconvolve.__repr__ = lambda self: \
        'galsim._galsim.SBDeConvolve(%r, %r)'%self.__getinitargs__()


def AutoConvolve(obj, real_space=None, gsparams=None):
    """A function for autoconvolving either a GSObject or ChromaticObject.

    This function will inspect its input argument to decide if a AutoConvolution object or a
    ChromaticAutoConvolution object is required to represent the convolution of a surface
    brightness profile with itself.

    @param obj              The object to be convolved with itself.
    @param real_space       Whether to use real space convolution.  [default: None, which means
                            to automatically decide this according to whether the object has hard
                            edges.]
    @param gsparams         An optional GSParams argument.  See the docstring for GSParams for
                            details. [default: None]

    @returns a AutoConvolution or ChromaticAutoConvolution instance as appropriate.
    """
    if isinstance(obj, galsim.ChromaticObject):
        return galsim.ChromaticAutoConvolution(obj, real_space=real_space, gsparams=gsparams)
    elif isinstance(obj, galsim.GSObject):
        return AutoConvolution(obj, real_space=real_space, gsparams=gsparams)
    else:
        raise TypeError("Argument to AutoConvolve must be either a GSObject or a ChromaticObject.")


class AutoConvolution(galsim.GSObject):
    """A special class for convolving a GSObject with itself.

    It is equivalent in functionality to `Convolve([obj,obj])`, but takes advantage of
    the fact that the two profiles are the same for some efficiency gains.

    Initialization
    --------------

    The normal way to use this class is to use the AutoConvolve() factory function:

        >>> psf_sq = galsim.AutoConvolve(psf)

    @param obj              The object to be convolved with itself.
    @param real_space       Whether to use real space convolution.  [default: None, which means
                            to automatically decide this according to whether the object has hard
                            edges.]
    @param gsparams         An optional GSParams argument.  See the docstring for GSParams for
                            details. [default: None]

    Methods
    -------

    There are no additional methods for AutoConvolution beyond the usual GSObject methods.
    """
    def __init__(self, obj, real_space=None, gsparams=None):
        if not isinstance(obj, galsim.GSObject):
            raise TypeError("Argument to AutoConvolution must be a GSObject.")

        # Check whether to perform real space convolution...
        # Start by checking if obj has a hard edge.
        hard_edge = obj.hasHardEdges()

        if real_space is None:
            # The automatic determination is to use real_space if obj has hard edges.
            real_space = hard_edge

        # Warn if doing DFT convolution for objects with hard edges.
        if not real_space and hard_edge:
            import warnings
            msg = """
            Doing auto-convolution of object with hard edges.
            This might be more accurate and/or faster using real_space=True"""
            warnings.warn(msg)

        # Can't do real space if object is not analytic, so check for that.
        if real_space and not obj.isAnalyticX():
            import warnings
            msg = """
            Object to be auto-convolved is not analytic in real space.
            Cannot use real space convolution.
            Switching to DFT method."""
            warnings.warn(msg)
            real_space = False

        # Save the construction parameters (as they are at this point) as attributes so they
        # can be inspected later if necessary.
        self._real_space = real_space
        self._orig_obj = obj
        self._gsparams = gsparams

        sbp = galsim._galsim.SBAutoConvolve(obj.SBProfile, real_space, gsparams)
        galsim.GSObject.__init__(self, sbp)
        if hasattr(obj,'noise'):
            import warnings
            warnings.warn("Unable to propagate noise in galsim.AutoConvolution")

    @property
    def orig_obj(self): return self._orig_obj
    @property
    def real_space(self): return self._real_space

    def __repr__(self):
        return 'galsim.AutoConvolution(%r, real_space=%r, gsparams=%r)'%(
                self.orig_obj, self.real_space, self._gsparams)

    def __str__(self):
        s = 'galsim.AutoConvolve(%s'%self.orig_obj
        if self.real_space:
            s += ', real_space=True'
        s += ')'
        return s

    def __getstate__(self):
        d = self.__dict__.copy()
        del d['SBProfile']
        return d

    def __setstate__(self, d):
        self.__dict__ = d
        self.__init__(self._orig_obj, self._real_space, self._gsparams)



_galsim.SBAutoConvolve.__getinitargs__ = lambda self: (
        self.getObj(), self.isRealSpace(), self.getGSParams())
_galsim.SBAutoConvolve.__getstate__ = lambda self: None
_galsim.SBAutoConvolve.__setstate__ = lambda self, state: 1
_galsim.SBAutoConvolve.__repr__ = lambda self: \
        'galsim._galsim.SBAutoConvolve(%r, %r, %r)'%self.__getinitargs__()


def AutoCorrelate(obj, real_space=None, gsparams=None):
    """A function for autocorrelating either a GSObject or ChromaticObject.

    This function will inspect its input argument to decide if a AutoCorrelation object or a
    ChromaticAutoCorrelation object is required to represent the correlation of a surface
    brightness profile with itself.

    @param obj              The object to be convolved with itself.
    @param real_space       Whether to use real space convolution.  [default: None, which means
                            to automatically decide this according to whether the object has hard
                            edges.]
    @param gsparams         An optional GSParams argument.  See the docstring for GSParams for
                            details. [default: None]

    @returns an AutoCorrelation or ChromaticAutoCorrelation instance as appropriate.
    """
    if isinstance(obj, galsim.ChromaticObject):
        return galsim.ChromaticAutoCorrelation(obj, real_space=real_space, gsparams=gsparams)
    elif isinstance(obj, galsim.GSObject):
        return AutoCorrelation(obj, real_space=real_space, gsparams=gsparams)
    else:
        raise TypeError("Argument to AutoCorrelate must be either a GSObject or a ChromaticObject.")


class AutoCorrelation(galsim.GSObject):
    """A special class for correlating a GSObject with itself.

    It is equivalent in functionality to
        galsim.Convolve([obj,obj.createRotated(180.*galsim.degrees)])
    but takes advantage of the fact that the two profiles are the same for some efficiency gains.

    This class is primarily targeted for use by the CorrelatedNoise models when convolving
    with a GSObject.

    Initialization
    --------------

    The normal way to use this class is to use the AutoCorrelate() factory function:

        >>> psf_sq = galsim.AutoCorrelate(psf)

    @param obj              The object to be convolved with itself.
    @param real_space       Whether to use real space convolution.  [default: None, which means
                            to automatically decide this according to whether the object has hard
                            edges.]
    @param gsparams         An optional GSParams argument.  See the docstring for GSParams for
                            details. [default: None]

    Methods
    -------

    There are no additional methods for AutoCorrelation beyond the usual GSObject methods.
    """
    def __init__(self, obj, real_space=None, gsparams=None):
        if not isinstance(obj, galsim.GSObject):
            raise TypeError("Argument to AutoCorrelation must be a GSObject.")

        # Check whether to perform real space convolution...
        # Start by checking if obj has a hard edge.
        hard_edge = obj.hasHardEdges()

        if real_space is None:
            # The automatic determination is to use real_space if obj has hard edges.
            real_space = hard_edge

        # Warn if doing DFT convolution for objects with hard edges.
        if not real_space and hard_edge:
            import warnings
            msg = """
            Doing auto-convolution of object with hard edges.
            This might be more accurate and/or faster using real_space=True"""
            warnings.warn(msg)

        # Can't do real space if object is not analytic, so check for that.
        if real_space and not obj.isAnalyticX():
            import warnings
            msg = """
            Object to be auto-convolved is not analytic in real space.
            Cannot use real space convolution.
            Switching to DFT method."""
            warnings.warn(msg)
            real_space = False

        # Save the construction parameters (as they are at this point) as attributes so they
        # can be inspected later if necessary.
        self._real_space = real_space
        self._orig_obj = obj
        self._gsparams = gsparams

        sbp = galsim._galsim.SBAutoCorrelate(obj.SBProfile, real_space, gsparams)
        galsim.GSObject.__init__(self, sbp)

        if hasattr(obj,'noise'):
            import warnings
            warnings.warn("Unable to propagate noise in galsim.AutoCorrelation")

    @property
    def orig_obj(self): return self._orig_obj
    @property
    def real_space(self): return self._real_space

    def __repr__(self):
        return 'galsim.AutoCorrelation(%r, real_space=%r, gsparams=%r)'%(
                self.orig_obj, self.real_space, self._gsparams)

    def __str__(self):
        s = 'galsim.AutoCorrelate(%s'%self.orig_obj
        if self.real_space:
            s += ', real_space=True'
        s += ')'
        return s

    def __getstate__(self):
        d = self.__dict__.copy()
        del d['SBProfile']
        return d

    def __setstate__(self, d):
        self.__dict__ = d
        self.__init__(self._orig_obj, self._real_space, self._gsparams)



_galsim.SBAutoCorrelate.__getinitargs__ = lambda self: (
        self.getObj(), self.isRealSpace(), self.getGSParams())
_galsim.SBAutoCorrelate.__getstate__ = lambda self: None
_galsim.SBAutoCorrelate.__setstate__ = lambda self, state: 1
_galsim.SBAutoCorrelate.__repr__ = lambda self: \
        'galsim._galsim.SBAutoCorrelate(%r, %r, %r)'%self.__getinitargs__()



def FourierSqrt(obj, gsparams=None):
    """A function for computing the Fourier-space square root of either a GSObject or ChromaticObject.

    This function will inspect its input argument to decide if a FourierSqrtProfile object or a
    ChromaticFourierSqrtProfile object is required to represent the operation applied to a surface
    brightness profile.

    @param obj              The object to compute the Fourier-space square root of.
    @param gsparams         An optional GSParams argument.  See the docstring for GSParams for
                            details. [default: None]

    @returns a FourierSqrtProfile or ChromaticFourierSqrtProfile instance as appropriate.
    """
    if isinstance(obj, galsim.ChromaticObject):
        return galsim.ChromaticFourierSqrt(obj, gsparams=gsparams)
    elif isinstance(obj, galsim.GSObject):
        return FourierSqrtProfile(obj, gsparams=gsparams)
    else:
        raise TypeError("Argument to Deconvolve must be either a GSObject or a ChromaticObject.")


class FourierSqrtProfile(galsim.GSObject):
    """A class for deconvolving a GSObject.

    The FourierSqrtProfile class represents the Fourier-space square root of another profile.  Note that the
    FourierSqrtProfile class, or compound objects (Sum, Convolution) that include a FourierSqrtProfile as
    one of the components cannot be photon-shot using the 'phot' method of drawImage() method.

    You may also specify a `gsparams` argument.  See the docstring for GSParams using
    `help(galsim.GSParams)` for more information about this option.  Note: if `gsparams` is
    unspecified (or None), then the FourierSqrtProfile instance inherits the same GSParams as the object
    being operated on.

    Initialization
    --------------

    The normal way to use this class is to use the FourierSqrt() factory function:

        >>> b = galsim.FourierSqrt(a)

    @param obj              The object to deconvolve.
    @param gsparams         An optional GSParams argument.  See the docstring for GSParams for
                            details. [default: None]

    Methods
    -------

    There are no additional methods for FourierSqrtProfile beyond the usual GSObject methods.
    """
    def __init__(self, obj, gsparams=None):
        if not isinstance(obj, galsim.GSObject):
            raise TypeError("Argument to FourierSqrtProfile must be a GSObject.")

        # Save the original object as an attribute, so it can be inspected later if necessary.
        self._orig_obj = obj
        self._gsparams = gsparams

        sbp = galsim._galsim.SBFourierSqrt(obj.SBProfile, gsparams)
        galsim.GSObject.__init__(self, sbp)
        if hasattr(obj,'noise'):
            import warnings
            warnings.warn("Unable to propagate noise in galsim.FourierSqrtProfile")

    @property
    def orig_obj(self): return self._orig_obj

    def __repr__(self):
        return 'galsim.FourierSqrtProfile(%r, gsparams=%r)'%(self.orig_obj, self._gsparams)

    def __str__(self):
        return 'galsim.FourierSqrt(%s)'%self.orig_obj

    def __getstate__(self):
        d = self.__dict__.copy()
        del d['SBProfile']
        return d

    def __setstate__(self, d):
        self.__dict__ = d
        self.__init__(self._orig_obj, self._gsparams)

_galsim.SBFourierSqrt.__getinitargs__ = lambda self: (self.getObj(), self.getGSParams())
_galsim.SBFourierSqrt.__getstate__ = lambda self: None
_galsim.SBFourierSqrt.__setstate__ = lambda self, state: 1
_galsim.SBFourierSqrt.__repr__ = lambda self: \
        'galsim._galsim.SBFourierSqrt(%r, %r)'%self.__getinitargs__()
