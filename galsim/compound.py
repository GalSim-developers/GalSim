# Copyright 2012-2014 The GalSim developers:
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
"""@file compound.py
Some compound GSObject classes that contain other GSObjects:

Sum = sum of multiple profiles
Convolution = convolution of multiple profiles
Deconvolution = deconvolution by a given profile
AutoConvolve = convolution of a profile by itself
AutoCorrelate = convolution of a profile by its reflection
"""

import galsim
from . import _galsim

#
# --- Compound GSObject classes: Sum, Convolution, AutoConvolve, and AutoCorrelate ---

def Add(*args, **kwargs):
    """A function for adding 2 or more GSObjects or ChromaticObjects.

    This function will inspect its input arguments to decide if a galsim.Sum object or a
    galsim.ChromaticSum object is required to represent the sum of surface brightness profiles.

    Typically, you do not need to call a `Add` explicitly.  Normally, you would just use the + 
    operator, which returns a Sum:

        >>> bulge = galsim.Sersic(n=3, half_light_radius=0.8)
        >>> disk = galsim.Exponential(half_light_radius=1.4)
        >>> gal = bulge + disk
        >>> psf = galsim.Gaussian(sigma=0.3, flux=0.3) + galsim.Gaussian(sigma=0.8, flux=0.7)

    If one of the items is Chromatic, it will return a ChromaticSum

        >>> disk = galsim.Exponential(half_light_radius=1.4) * galsim.SED(sed_file)
        >>> gal = bulge + disk


    @param args             Unnamed args should be a list of objects to add.
    @param gsparams         An optional GSParams argument.  See the docstring for galsim.GSParams
                            for details. [default: None]

    @returns a galsim.Sum or galsim.ChromaticSum instance as appropriate.
    """
    if len(args) == 0:
        # No arguments. Could initialize with an empty list but draw then segfaults. Raise an
        # exception instead.
        raise ValueError("Sum must be initialized with at least one ChromaticObject or GSObject.")
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
    """A class for adding 2 or more GSObjects.

    The Sum class is used to represent the sum of multiple GSObjects.  For example, it might be used
    to represent a multiple-component galaxy as the sum of an Exponential and a DeVaucouleurs, or to
    represent a PSF as the sum of multiple Gaussians.

    Typically, you do not need to construct a `Sum` object explicitly.  Normally, you would just
    use the + operator, which returns a Sum:

        >>> bulge = galsim.Sersic(n=3, half_light_radius=0.8)
        >>> disk = galsim.Exponential(half_light_radius=1.4)
        >>> gal = bulge + disk
        >>> psf = galsim.Gaussian(sigma=0.3, flux=0.3) + galsim.Gaussian(sigma=0.8, flux=0.7)

    Initialization
    --------------

    @param args             Unnamed args should be a list of objects to add.
    @param gsparams         An optional GSParams argument.  See the docstring for galsim.GSParams
                            for details. [default: None]

    Note: if gsparams is unspecified (or None), then the Sum instance inherits the same GSParams
    as the first item in the list.  Also, note that parameters related to the Fourier-space
    calculations must be set when initializing the individual GSObjects that go into the Sum, NOT
    when creating the Sum (at which point the accuracy and threshold parameters will simply be
    ignored).


    Methods
    -------

    There are no additional methods for Sum beyond the usual GSObject methods.
    """

    # --- Public Class methods ---
    def __init__(self, *args, **kwargs):

        # Check kwargs first:
        gsparams = kwargs.pop("gsparams", None)

        # Make sure there is nothing left in the dict.
        if kwargs:
            raise TypeError(
                "Sum constructor got unexpected keyword argument(s): %s"%kwargs.keys())

        if len(args) == 0:
            # No arguments. Could initialize with an empty list but draw then segfaults. Raise an
            # exception instead.
            raise ValueError("Sum must be initialized with at least one GSObject.")
        elif len(args) == 1:
            # 1 argument.  Should be either a GSObject or a list of GSObjects
            if isinstance(args[0], galsim.GSObject):
                args = [args[0]]
            elif isinstance(args[0], list):
                args = args[0]
            else:
                raise TypeError("Single input argument must be a GSObject or list of them.")
        # else args is already the list of objects

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
    """A function for convolving 2 or more GSObjects or ChromaticObjects.

    This function will inspect its input arguments to decide if a galsim.Convolution object or a
    galsim.ChromaticConvolution object is required to represent the convolution of surface
    brightness profiles.

    @param args             Unnamed args should be a list of objects to convolve.
    @param real_space       Whether to use real space convolution.  [default: None, which means
                            to automatically decide this according to whether the objects have hard
                            edges.]
    @param gsparams         An optional GSParams argument.  See the docstring for galsim.GSParams
                            for details. [default: None]

    @returns a galsim.Convolution or galsim.ChromaticConvolution instance as appropriate.
    """
    # First check for number of arguments != 0
    if len(args) == 0:
        # No arguments. Could initialize with an empty list but draw then segfaults. Raise an
        # exception instead.
        raise ValueError("Convolution must be initialized with at least one GSObject "
                         + "or ChromaticObject.")
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
    """A class for convolving 2 or more GSObjects.

    Initialization
    --------------

    The objects to be convolved may be provided either as multiple unnamed arguments (e.g.
    `Convolve(psf, gal, pix)`) or as a list (e.g. `Convolve([psf, gal, pix])`).  Any number of
    objects may be provided using either syntax.  (Even 0 or 1, although that doesn't really make
    much sense.)

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

    @param args             Unnamed args should be a list of objects to convolve.
    @param real_space       Whether to use real space convolution.  [default: None, which means
                            to automatically decide this according to whether the objects have hard
                            edges.]
    @param gsparams         An optional GSParams argument.  See the docstring for galsim.GSParams
                            for details. [default: None]

    Note: if gsparams is unspecified (or None), then the Convolution instance inherits the same 
    GSParams as the first item in the list.  Also, note that parameters related to the Fourier-
    space calculations must be set when initializing the individual GSObjects that go into the Sum,
    NOT when creating the Sum (at which point the accuracy and threshold parameters will simply be
    ignored).

    Methods
    -------

    There are no additional methods for Convolution beyond the usual GSObject methods.
    """

    # --- Public Class methods ---
    def __init__(self, *args, **kwargs):

        # First check for number of arguments != 0
        if len(args) == 0:
            # No arguments. Could initialize with an empty list but draw then segfaults. Raise an
            # exception instead.
            raise ValueError("Convolution must be initialized with at least one GSObject.")
        elif len(args) == 1:
            if isinstance(args[0], galsim.GSObject):
                args = [args[0]]
            elif isinstance(args[0], list):
                args = args[0]
            else:
                raise TypeError("Single input argument must be a GSObject or list of them.")
        # else args is already the list of objects

        if len(args) == 1:
            # No need to make an SBConvolve in this case.  Can early exit.
            galsim.GSObject.__init__(self, args[0])
            if hasattr(args[0],'noise'):
                self.noise = args[0].noise
            return

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

        # Then finally initialize the SBProfile using the objects' SBProfiles.
        SBList = [ obj.SBProfile for obj in args ]
        galsim.GSObject.__init__(self, galsim._galsim.SBConvolve(SBList, real_space=real_space,
                                                          gsparams=gsparams))
        if noise is not None:
            self.noise = noise


def Deconvolve(obj, gsparams=None):
    """A function for deconvolving by either a GSObject or ChromaticObject.

    This function will inspect its input argument to decide if a galsim.Deconvolution object or a
    galsim.ChromaticDeconvolution object is required to represent the deconvolution by a surface
    brightness profile.

    @param obj              The object to deconvolve.
    @param gsparams         An optional GSParams argument.  See the docstring for galsim.GSParams
                            for details. [default: None]

    @returns a galsim.Deconvolution or galsim.ChromaticDeconvolution instance as appropriate.
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
    cannot be photon-shot using the drawShoot method.

    You may also specify a gsparams argument.  See the docstring for galsim.GSParams using
    help(galsim.GSParams) for more information about this option.  Note: if gsparams is unspecified
    (or None), then the Deconvolution instance inherits the same GSParams as the object being
    deconvolved.

    @param obj              The object to deconvolve.
    @param gsparams         An optional GSParams argument.  See the docstring for galsim.GSParams
                            for details. [default: None]


    Methods
    -------

    There are no additional methods for Deconvolution beyond the usual GSObject methods.
    """
    # --- Public Class methods ---
    def __init__(self, obj, gsparams=None):
        if not isinstance(obj, galsim.GSObject):
            raise TypeError("Argument to Deconvolution must be a GSObject.")
        galsim.GSObject.__init__(
                self, galsim._galsim.SBDeconvolve(obj.SBProfile, gsparams=gsparams))
        if hasattr(obj,'noise'):
            import warnings
            warnings.warn("Unable to propagate noise in galsim.Deconvolution")


def AutoConvolve(obj, real_space=None, gsparams=None):
    """A function for autoconvolving either a GSObject or ChromaticObject.

    This function will inspect its input argument to decide if a galsim.AutoConvolution object or a
    galsim.ChromaticAutoConvolution object is required to represent the convolution of a surface
    brightness profile with itself.

    @param obj              The object to be convolved with itself.
    @param real_space       Whether to use real space convolution.  [default: None, which means
                            to automatically decide this according to whether the objects have hard
                            edges.]
    @param gsparams         An optional GSParams argument.  See the docstring for galsim.GSParams
                            for details. [default: None]

    @returns a galsim.AutoConvolution or galsim.ChromaticAutoConvolution instance as appropriate.
    """
    if isinstance(obj, galsim.ChromaticObject):
        return galsim.ChromaticAutoConvolution(obj, real_space=real_space, gsparams=gsparams)
    elif isinstance(obj, galsim.GSObject):
        return AutoConvolution(obj, real_space=real_space, gsparams=gsparams)
    else:
        raise TypeError("Argument to AutoConvolve must be either a GSObject or a ChromaticObject.")


class AutoConvolution(galsim.GSObject):
    """A special class for convolving a GSObject with itself.

    It is equivalent in functionality to galsim.Convolve([obj,obj]), but takes advantage of
    the fact that the two profiles are the same for some efficiency gains.

    @param obj              The object to be convolved with itself.
    @param real_space       Whether to use real space convolution.  [default: None, which means
                            to automatically decide this according to whether the objects have hard
                            edges.]
    @param gsparams         An optional GSParams argument.  See the docstring for galsim.GSParams
                            for details. [default: None]


    Methods
    -------

    There are no additional methods for AutoConvolution beyond the usual GSObject methods.
    """
    # --- Public Class methods ---
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

        sbp = galsim._galsim.SBAutoConvolve(
                obj.SBProfile, real_space=real_space, gsparams=gsparams)
        galsim.GSObject.__init__(self, sbp)
        if hasattr(obj,'noise'):
            import warnings
            warnings.warn("Unable to propagate noise in galsim.AutoConvolution")


def AutoCorrelate(obj, real_space=None, gsparams=None):
    """A function for autocorrelating either a GSObject or ChromaticObject.

    This function will inspect its input argument to decide if a galsim.AutoCorrelation object or a
    galsim.ChromaticAutoCorrelation object is required to represent the correlation of a surface
    brightness profile with itself.

    @param obj              The object to be convolved with itself.
    @param real_space       Whether to use real space convolution.  [default: None, which means
                            to automatically decide this according to whether the objects have hard
                            edges.]
    @param gsparams         An optional GSParams argument.  See the docstring for galsim.GSParams
                            for details. [default: None]

    @returns a galsim.AutoCorrelation or galsim.ChromaticAutoCorrelation instance as appropriate.
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

    This class is primarily targeted for use by the galsim.CorrelatedNoise models when convolving
    with a GSObject.

    @param obj              The object to be convolved with itself.
    @param real_space       Whether to use real space convolution.  [default: None, which means
                            to automatically decide this according to whether the objects have hard
                            edges.]
    @param gsparams         An optional GSParams argument.  See the docstring for galsim.GSParams
                            for details. [default: None]


    Methods
    -------

    There are no additional methods for AutoCorrelation beyond the usual GSObject methods.
    """
    # --- Public Class methods ---
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

        sbp = galsim._galsim.SBAutoCorrelate(
                obj.SBProfile, real_space=real_space, gsparams=gsparams)
        galsim.GSObject.__init__(self, sbp)

        if hasattr(obj,'noise'):
            import warnings
            warnings.warn("Unable to propagate noise in galsim.AutoCorrelation")
