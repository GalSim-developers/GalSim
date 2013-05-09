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
"""@file compound.py 
Some compount GSObject classes that contain other GSObjects:

Add = sum of multiple profiles
Convolve = convolution of multiple profiles
Deconvolve = deconvolution by a given profile
AutoConvolve = convolution of a profile by itself
AutoCorrelate = convolution of a profile by its reflection
"""

import galsim
from galsim import GSObject

#
# --- Compound GSObject classes: Add, Convolve, AutoConvolve, and AutoCorrelate ---

class Add(GSObject):
    """A class for adding 2 or more GSObjects.  Has an SBAdd in the SBProfile attribute.

    The Add class is used to represent the sum of multiple GSObjects.  For example, it might be used
    to represent a multiple-component galaxy as the sum of an Exponential and a DeVaucouleurs, or to
    represent a PSF as the sum of multiple Gaussians.

    You may also specify a gsparams argument.  See the docstring for galsim.GSParams using
    help(galsim.GSParams) for more information about this option.  Note: if gsparams is unspecified
    (or None), then the Add instance inherits the same GSParams as the first item in the list.
    Also, note that parameters related to the Fourier-space calculations must be set when
    initializing the individual GSObjects that go into the Add, NOT when creating the Add (at which
    point the accuracy and threshold parameters will simply be ignored).

    Methods
    -------
    The Add is a GSObject, and inherits all of the GSObject methods (draw(), drawShoot(),
    applyShear() etc.) and operator bindings.
    """
    
    # --- Public Class methods ---
    def __init__(self, *args, **kwargs):

        # Check kwargs first:
        gsparams = kwargs.pop("gsparams", None)

        # Make sure there is nothing left in the dict.
        if kwargs:
            raise TypeError(
                "Add constructor got unexpected keyword argument(s): %s"%kwargs.keys())

        if len(args) == 0:
            # No arguments. Could initialize with an empty list but draw then segfaults. Raise an
            # exception instead.
            raise ValueError("Add must be initialized with at least one GSObject.")
        elif len(args) == 1:
            # 1 argument.  Should be either a GSObject or a list of GSObjects
            if isinstance(args[0], GSObject):
                SBList = [args[0].SBProfile]
            elif isinstance(args[0], list):
                SBList = []
                for obj in args[0]:
                    if isinstance(obj, GSObject):
                        SBList.append(obj.SBProfile)
                    else:
                        raise TypeError("Input list must contain only GSObjects.")
            else:
                raise TypeError("Single input argument must be a GSObject or list of them.")
            GSObject.__init__(self, galsim.SBAdd(SBList, gsparams=gsparams))
        elif len(args) >= 2:
            # >= 2 arguments.  Convert to a list of SBProfiles
            SBList = [obj.SBProfile for obj in args]
            GSObject.__init__(self, galsim.SBAdd(SBList, gsparams=gsparams))

class Convolve(GSObject):
    """A class for convolving 2 or more GSObjects.  Has an SBConvolve in the SBProfile attribute.

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

    You may also specify a gsparams argument.  See the docstring for galsim.GSParams using
    help(galsim.GSParams) for more information about this option.  Note: if gsparams is unspecified
    (or None), then the Convolve instance inherits the same GSParams as the first item in the list.
    Also, note that parameters related to the Fourier-space calculations must be set when
    initializing the individual GSObjects that go into the Convolve, NOT when creating the Convolve
    (at which point the accuracy and threshold parameters will simply be ignored).
    """
                    
    # --- Public Class methods ---
    def __init__(self, *args, **kwargs):

        # First check for number of arguments != 0
        if len(args) == 0:
            # No arguments. Could initialize with an empty list but draw then segfaults. Raise an
            # exception instead.
            raise ValueError("Convolve must be initialized with at least one GSObject.")
        elif len(args) == 1:
            if isinstance(args[0], GSObject):
                SBList = [args[0].SBProfile]
            elif isinstance(args[0], list):
                SBList=[]
                for obj in args[0]:
                    if isinstance(obj, GSObject):
                        SBList.append(obj.SBProfile)
                    else:
                        raise TypeError("Input list must contain only GSObjects.")
            else:
                raise TypeError("Single input argument must be a GSObject or list of them.")
        elif len(args) >= 2:
            # >= 2 arguments.  Convert to a list of SBProfiles
            SBList = []
            for obj in args:
                if isinstance(obj, GSObject):
                    SBList.append(obj.SBProfile)
                else:
                    raise TypeError("Input args must contain only GSObjects.")

        # Having built the list of SBProfiles or thrown exceptions if necessary, see now whether
        # to perform real space convolution...

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
                "Convolve constructor got unexpected keyword argument(s): %s"%kwargs.keys())


        # If 1 argument, check if it is a list:
        if len(args) == 1 and isinstance(args[0], list):
            args = args[0]

        hard_edge = True
        for obj in args:
            if not obj.hasHardEdges():
                hard_edge = False

        if real_space is None:
            # Figure out if it makes more sense to use real-space convolution.
            if len(args) == 2:
                real_space = hard_edge
            elif len(args) == 1:
                real_space = obj.isAnalyticX()
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

        # Then finally initialize the SBProfile using the objects' SBProfiles in SBList
        GSObject.__init__(self, galsim.SBConvolve(SBList, real_space=real_space,
                                                  gsparams=gsparams))


class Deconvolve(GSObject):
    """Base class for defining the python interface to the SBDeconvolve C++ class.

    The Deconvolve class represents a deconvolution kernel.  Note that the Deconvolve class, or
    compound objects (Add, Convolve) that include a Deconvolve as one of the components, cannot be
    photon-shot using the drawShoot method.

    You may also specify a gsparams argument.  See the docstring for galsim.GSParams using
    help(galsim.GSParams) for more information about this option.  Note: if gsparams is unspecified
    (or None), then the Deconvolve instance inherits the same GSParams as the object being
    deconvolved.
    """
    # --- Public Class methods ---
    def __init__(self, farg, gsparams=None):
        if isinstance(farg, GSObject):
            self.farg = farg
            GSObject.__init__(self, galsim.SBDeconvolve(self.farg.SBProfile, gsparams=gsparams))
        else:
            raise TypeError("Argument to Deconvolve must be a GSObject.")


class AutoConvolve(GSObject):
    """A special class for convolving a GSObject with itself.

    It is equivalent in functionality to galsim.Convolve([obj,obj]), but takes advantage of
    the fact that the two profiles are the same for some efficiency gains.

    @param obj       The object to be convolved with itself.
    @param gsparams  You may also specify a gsparams argument.  See the docstring for
                     galsim.GSParams using help(galsim.GSParams) for more information about this
                     option.  Note that parameters related to the Fourier-space calculations must be
                     set when initializing the GSObject that goes into the AutoConvolve, NOT when
                     creating the AutoConvolve (at which point the accuracy and threshold parameters
                     will simply be ignored).
    """
    # --- Public Class methods ---
    def __init__(self, obj, gsparams=None):
        if not isinstance(obj, GSObject):
            raise TypeError("Argument to AutoConvolve must be a GSObject.")
        GSObject.__init__(self, galsim.SBAutoConvolve(obj.SBProfile, gsparams=gsparams))


class AutoCorrelate(GSObject):
    """A special class for correlating a GSObject with itself.

    It is equivalent in functionality to 
        galsim.Convolve([obj,obj.createRotated(180.*galsim.degrees)])
    but takes advantage of the fact that the two profiles are the same for some efficiency gains.

    This class is primarily targeted for use by the galsim.CorrelatedNoise models when convolving 
    with a GSObject.

    @param obj       The object to be correlated with itself.

    @param gsparams  You may also specify a gsparams argument.  See the docstring for
                     galsim.GSParams using help(galsim.GSParams) for more information about this
                     option.  Note that parameters related to the Fourier-space calculations must be
                     set when initializing the GSObject that goes into the AutoCorrelate, NOT when
                     creating the AutoCorrelate (at which point the accuracy and threshold
                     parameters will simply be ignored).
    """
    # --- Public Class methods ---
    def __init__(self, obj, gsparams=None):
        if not isinstance(obj, GSObject):
            raise TypeError("Argument to AutoCorrelate must be a GSObject.")
        GSObject.__init__(self, galsim.SBAutoCorrelate(obj.SBProfile, gsparams=gsparams))



