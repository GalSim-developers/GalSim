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
    """A class for adding 2 or more GSObjects.

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
                args = [args[0]]
            elif isinstance(args[0], list):
                args = args[0]
            else:
                raise TypeError("Single input argument must be a GSObject or list of them.")
        # else args is already the list of objects

        if len(args) == 1:
            # No need to make an SBAdd in this case.
            GSObject.__init__(self, args[0])
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
            GSObject.__init__(self, galsim.SBAdd(SBList, gsparams=gsparams))
            if noise is not None: 
                self.noise = noise

class Convolve(GSObject):
    """A class for convolving 2 or more GSObjects.

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

    @param args       Unnamed args should be a list of objects to convolve.
    @param real_space Whether to use real space convolution.  Default is to automatically select
                      this according to whether the object has hard edges.
    @param gsparams   Optional gsparams argument.
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
                args = [args[0]]
            elif isinstance(args[0], list):
                args = args[0]
            else:
                raise TypeError("Single input argument must be a GSObject or list of them.")
        # else args is already the list of objects

        if len(args) == 1:
            # No need to make an SBConvolve in this case.  Can early exit.
            GSObject.__init__(self, args[0])
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
                "Convolve constructor got unexpected keyword argument(s): %s"%kwargs.keys())

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
                    warnings.warn("Unable to propagate noise in galsim.Convolve when multiple "+
                                  "objects have noise attribute")
                    noise = None
                    break
                noise = obj.noise
                others = [ obj2 for obj2 in args if obj2 is not obj ]
                assert len(others) > 0
                if len(others) == 1: 
                    noise.convolveWith(others[0])
                else: 
                    noise.convolveWith(galsim.Convolve(others))

        # Then finally initialize the SBProfile using the objects' SBProfiles.
        SBList = [ obj.SBProfile for obj in args ]
        GSObject.__init__(self, galsim.SBConvolve(SBList, real_space=real_space,
                                                  gsparams=gsparams))
        if noise is not None: 
            self.noise = noise


class Deconvolve(GSObject):
    """A class for deconvolving a GSObject.

    The Deconvolve class represents a deconvolution kernel.  Note that the Deconvolve class, or
    compound objects (Add, Convolve) that include a Deconvolve as one of the components, cannot be
    photon-shot using the drawShoot method.

    You may also specify a gsparams argument.  See the docstring for galsim.GSParams using
    help(galsim.GSParams) for more information about this option.  Note: if gsparams is unspecified
    (or None), then the Deconvolve instance inherits the same GSParams as the object being
    deconvolved.

    @param obj        The object to deconvolve.
    @param gsparams   Optional gsparams argument.
    """
    # --- Public Class methods ---
    def __init__(self, obj, gsparams=None):
        if not isinstance(obj, GSObject):
            raise TypeError("Argument to Deconvolve must be a GSObject.")
        GSObject.__init__(self, galsim.SBDeconvolve(obj.SBProfile, gsparams=gsparams))
        if hasattr(obj,'noise'):
            import warnings
            warnings.warn("Unable to propagate noise in galsim.Deconvolve")


class AutoConvolve(GSObject):
    """A special class for convolving a GSObject with itself.

    It is equivalent in functionality to galsim.Convolve([obj,obj]), but takes advantage of
    the fact that the two profiles are the same for some efficiency gains.

    @param obj        The object to be convolved with itself.
    @param real_space Whether to use real space convolution.  Default is to automatically select
                      this according to whether the object has hard edges.
    @param gsparams   Optional gsparams argument.
    """
    # --- Public Class methods ---
    def __init__(self, obj, real_space=None, gsparams=None):
        if not isinstance(obj, GSObject):
            raise TypeError("Argument to AutoConvolve must be a GSObject.")

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

        GSObject.__init__(self, galsim.SBAutoConvolve(obj.SBProfile, real_space=real_space,
                                                      gsparams=gsparams))
        if hasattr(obj,'noise'):
            import warnings
            warnings.warn("Unable to propagate noise in galsim.AutoConvolve")


class AutoCorrelate(GSObject):
    """A special class for correlating a GSObject with itself.

    It is equivalent in functionality to 
        galsim.Convolve([obj,obj.createRotated(180.*galsim.degrees)])
    but takes advantage of the fact that the two profiles are the same for some efficiency gains.

    This class is primarily targeted for use by the galsim.CorrelatedNoise models when convolving 
    with a GSObject.

    @param obj        The object to be convolved with itself.
    @param real_space Whether to use real space convolution.  Default is to automatically select
                      this according to whether the object has hard edges.
    @param gsparams   Optional gsparams argument.
    """
    # --- Public Class methods ---
    def __init__(self, obj, real_space=None, gsparams=None):
        if not isinstance(obj, GSObject):
            raise TypeError("Argument to AutoCorrelate must be a GSObject.")

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

        GSObject.__init__(self, galsim.SBAutoCorrelate(obj.SBProfile, real_space=real_space,
                                                       gsparams=gsparams))

        if hasattr(obj,'noise'):
            import warnings
            warnings.warn("Unable to propagate noise in galsim.AutoCorrelate")



