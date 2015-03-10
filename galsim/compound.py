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
"""@file compound.py
Some compound GSObject classes that contain other GSObject instances:

Sum = sum of multiple profiles
Convolution = convolution of multiple profiles
Deconvolution = deconvolution by a given profile
AutoConvolution = convolution of a profile by itself
AutoCorrelation = convolution of a profile by its reflection
"""

import galsim
from . import _galsim

#
# --- Compound GSObject classes: Sum, Convolution, AutoConvolve, and AutoCorrelate ---

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
        elif isinstance(args[0], list):
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
            elif isinstance(args[0], list):
                args = args[0]
            else:
                raise TypeError("Single input argument must be a GSObject or list of them.")
        # else args is already the list of objects

        # Save the list as an attribute, so it can be inspected later if necessary.
        self.obj_list = args

        if len(args) == 1:
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
            galsim.GSObject.__init__(self, galsim._galsim.SBAdd(SBList, gsparams=gsparams))
            if noise is not None:
                self.noise = noise


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
        elif isinstance(args[0], list):
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
            elif isinstance(args[0], list):
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

        # Make sure there is nothing left in the dict.
        if kwargs:
            raise TypeError(
                "Convolution constructor got unexpected keyword argument(s): %s"%kwargs.keys())

        if len(args) == 1:
            # No need to make an SBConvolve in this case.  Can early exit.
            galsim.GSObject.__init__(self, args[0])
            if hasattr(args[0],'noise'):
                self.noise = args[0].noise
            self.real_space = real_space
            self.obj_list = args
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
        self.real_space = real_space
        self.obj_list = args

        # Then finally initialize the SBProfile using the objects' SBProfiles.
        SBList = [ obj.SBProfile for obj in args ]
        sbp = galsim._galsim.SBConvolve(SBList, real_space=real_space, gsparams=gsparams)
        galsim.GSObject.__init__(self, sbp)
        if noise is not None:
            self.noise = noise


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
        self.orig_obj = obj

        galsim.GSObject.__init__(
                self, galsim._galsim.SBDeconvolve(obj.SBProfile, gsparams=gsparams))
        if hasattr(obj,'noise'):
            import warnings
            warnings.warn("Unable to propagate noise in galsim.Deconvolution")


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
        self.real_space = real_space
        self.orig_obj = obj

        sbp = galsim._galsim.SBAutoConvolve(
                obj.SBProfile, real_space=real_space, gsparams=gsparams)
        galsim.GSObject.__init__(self, sbp)
        if hasattr(obj,'noise'):
            import warnings
            warnings.warn("Unable to propagate noise in galsim.AutoConvolution")


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
        self.real_space = real_space
        self.orig_obj = obj

        sbp = galsim._galsim.SBAutoCorrelate(
                obj.SBProfile, real_space=real_space, gsparams=gsparams)
        galsim.GSObject.__init__(self, sbp)

        if hasattr(obj,'noise'):
            import warnings
            warnings.warn("Unable to propagate noise in galsim.AutoCorrelation")

class Transform(galsim.GSObject):
    """A class for modeling an affine transformation of a GSObject instance.

    Initialization
    --------------

    Typically, you do not need to construct a Transform object explicitly.  This is the type
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

    Note: if `gsparams` is unspecified (or None), then the Transform instance inherits the GSParams
    from obj.  Also, note that parameters related to the Fourier-space calculations must be set
    when initializing obj, NOT when creating the Transform (at which point the accuracy and
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
        return 'galsim.Transform(%r, jac=%r, offset=%r, flux_ratio=%r, gsparams=%r)'%(
            self.original, self.jac.tolist(), self.offset, self.flux_ratio, self.gsparams)

    def __str__(self):
        s = str(self.original)
        dudx, dudy, dvdx, dvdy = self.jac
        if dudx != 1 or dudy != 0 or dvdx != 0 or dvdy != 1:
            # MJ: If we want to get fancy, we could try to determine the minimal call to make.
            #     e.g. is it just a shear?  just a rotation? etc.  Probably not worth it though.
            s += '.transform(%f,%f,%f,%f)'%(dudx,dudy,dvdx,dvdy)
        offset = self.offset
        if offset.x != 0 or offset.y != 0:
            s += '.shift(%f,%f)'%(offset.x,offset.y)
        if self.flux_ratio != 1.:
            s += '.withScaledFlux(%f)'%self.flux_ratio
        return s

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

#def SBTransform_repr(self):
    #obj = self.getObj()
    #dudx, dudy, dvdx, dvdy = self.getJac()
    #offset = self.getOffset()
    #flux_ratio = self.getFluxScaling()
    #gsparams = self.getGSParams()
    #return 'galsim._galsim.SBTransform(%r, %r, %r, %r, %r, %r, %r, %r)'%(*self.__getinitargs__())
            #obj, dudx, dudy, dvdx, dvdy, offset, flux_ratio, gsparams)
#_galsim.SBTransform.__repr__ = SBTransform_repr
_galsim.SBTransform.__repr__ = lambda self: \
        'galsim._galsim.SBTransform(%r, %r, %r, %r, %r, %r, %r, %r)'%self.__getinitargs__()

