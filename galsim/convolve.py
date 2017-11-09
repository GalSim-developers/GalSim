# Copyright (c) 2012-2017 by the GalSim developers team on GitHub
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

import numpy as np

import galsim
from . import _galsim
from .utilities import lazy_property


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
        raise TypeError("At least one ChromaticObject or GSObject must be provided.")
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
    that case, the highest frequency `maxk` for each component is quite large since the ringing dies
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
            raise TypeError("At least one ChromaticObject or GSObject must be provided.")
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

        # Make sure there is nothing left in the dict.
        if kwargs:
            raise TypeError(
                "Convolution constructor got unexpected keyword argument(s): %s"%kwargs.keys())

        # Check whether to perform real space convolution...
        # Start by checking if all objects have a hard edge.
        hard_edge = True
        for obj in args:
            if not isinstance(obj, galsim.GSObject):
                raise TypeError("Arguments to Convolution must be GSObjects, not %s"%obj)
            if not obj.has_hard_edges:
                hard_edge = False

        if real_space is None:
            # The automatic determination is to use real_space if 2 items, both with hard edges.
            if len(args) <= 2:
                real_space = hard_edge
            else:
                real_space = False

        # Warn if doing DFT convolution for objects with hard edges
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
                    if not obj.is_analytic_x:
                        import warnings
                        msg = """
                        A component to be convolved is not analytic in real space.
                        Cannot use real space convolution.
                        Switching to DFT method."""
                        warnings.warn(msg)
                        real_space = False
                        break

        # Save the construction parameters (as they are at this point) as attributes so they
        # can be inspected later if necessary.
        self._real_space = real_space
        self._obj_list = args
        self._gsparams = galsim.GSParams.check(gsparams, self._obj_list[0].gsparams)

        # Then finally initialize the SBProfile using the objects' SBProfiles.
        SBList = [ obj._sbp for obj in args ]
        self._sbp = galsim._galsim.SBConvolve(SBList, real_space, self.gsparams._gsp)

    @lazy_property
    def noise(self):
        # If one of the objects has a noise attribute, then we convolve it by the others.
        # More than one is not allowed.
        _noise = None
        for i, obj in enumerate(self._obj_list):
            if obj.noise is not None:
                if _noise is not None:
                    import warnings
                    warnings.warn("Unable to propagate noise in galsim.Convolution when "
                                  "multiple objects have noise attribute")
                    break
                _noise = obj.noise
                others = [ obj2 for k, obj2 in enumerate(self._obj_list) if k != i ]
                assert len(others) > 0
                if len(others) == 1:
                    _noise = _noise.convolvedWith(others[0])
                else:
                    _noise = _noise.convolvedWith(galsim.Convolve(others))
        return _noise

    @property
    def obj_list(self): return self._obj_list
    @property
    def real_space(self): return self._real_space

    def __eq__(self, other):
        return (isinstance(other, galsim.Convolution) and
                self.obj_list == other.obj_list and
                self.real_space == other.real_space and
                self.gsparams == other.gsparams)

    def __hash__(self):
        return hash(("galsim.Convolution", tuple(self.obj_list), self.real_space, self.gsparams))

    def __repr__(self):
        return 'galsim.Convolution(%r, real_space=%r, gsparams=%r)'%(
                self.obj_list, self.real_space, self.gsparams)

    def __str__(self):
        str_list = [ str(obj) for obj in self.obj_list ]
        s = 'galsim.Convolve(%s'%(', '.join(str_list))
        if self.real_space:
            s += ', real_space=True'
        s += ')'
        return s

    def _prepareDraw(self):
        for obj in self._obj_list:
            obj._prepareDraw()
        SBList = [obj._sbp for obj in self._obj_list]
        self._sbp = galsim._galsim.SBConvolve(SBList, self._real_space, self.gsparams._gsp)

    def shoot(self, n_photons, rng=None):
        """Shoot photons into a PhotonArray.

        @param n_photons    The number of photons to use for photon shooting.
        @param rng          If provided, a random number generator to use for photon shooting,
                            which may be any kind of BaseDeviate object.  If `rng` is None, one
                            will be automatically created, using the time as a seed.
                            [default: None]
        @returns PhotonArray.
        """
        ud = galsim.UniformDeviate(rng)

        photon_array = self._obj_list[0].shoot(n_photons, ud)
        # It may be necessary to shuffle when convolving because we do not have a
        # gaurantee that the convolvee's photons are uncorrelated, e.g., they might
        # both have their negative ones at the end.
        # However, this decision is now made by the convolve method.
        for obj in self._obj_list[1:]:
            photon_array.convolve(obj.shoot(n_photons, ud), ud)
        return photon_array

    def __getstate__(self):
        d = self.__dict__.copy()
        del d['_sbp']
        return d

    def __setstate__(self, d):
        self.__dict__ = d
        self.__init__(self._obj_list, real_space=self._real_space, gsparams=self.gsparams)


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
        self._gsparams = galsim.GSParams.check(gsparams, self._orig_obj.gsparams)

        self._sbp = galsim._galsim.SBDeconvolve(obj._sbp, self.gsparams._gsp)
        if obj.noise is not None:
            import warnings
            warnings.warn("Unable to propagate noise in galsim.Deconvolution")

    @property
    def orig_obj(self): return self._orig_obj

    def __eq__(self, other):
        return (isinstance(other, galsim.Deconvolution) and
                self.orig_obj == other.orig_obj and
                self.gsparams == other.gsparams)

    def __hash__(self):
        return hash(("galsim.Deconvolution", self.orig_obj, self.gsparams))

    def __repr__(self):
        return 'galsim.Deconvolution(%r, gsparams=%r)'%(self.orig_obj, self.gsparams)

    def __str__(self):
        return 'galsim.Deconvolve(%s)'%self.orig_obj

    def _prepareDraw(self):
        self._orig_obj._prepareDraw()
        self._sbp = galsim._galsim.SBDeconvolve(self._orig_obj._sbp, self.gsparams._gsp)

    def __getstate__(self):
        d = self.__dict__.copy()
        del d['_sbp']
        return d

    def __setstate__(self, d):
        self.__dict__ = d
        self.__init__(self._orig_obj, self._gsparams)


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
        hard_edge = obj.has_hard_edges

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
        if real_space and not obj.is_analytic_x:
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
        self._gsparams = galsim.GSParams.check(gsparams, self._orig_obj.gsparams)

        self._sbp = galsim._galsim.SBAutoConvolve(obj._sbp, real_space, self.gsparams._gsp)
        if obj.noise is not None:
            import warnings
            warnings.warn("Unable to propagate noise in galsim.AutoConvolution")

    @property
    def orig_obj(self): return self._orig_obj
    @property
    def real_space(self): return self._real_space

    def __eq__(self, other):
        return (isinstance(other, galsim.AutoConvolution) and
                self.orig_obj == other.orig_obj and
                self.real_space == other.real_space and
                self.gsparams == other.gsparams)

    def __hash__(self):
        return hash(("galsim.AutoConvolution", self.orig_obj, self.real_space, self.gsparams))

    def __repr__(self):
        return 'galsim.AutoConvolution(%r, real_space=%r, gsparams=%r)'%(
                self.orig_obj, self.real_space, self.gsparams)

    def __str__(self):
        s = 'galsim.AutoConvolve(%s'%self.orig_obj
        if self.real_space:
            s += ', real_space=True'
        s += ')'
        return s

    def _prepareDraw(self):
        self._orig_obj._prepareDraw()
        self._sbp = galsim._galsim.SBAutoConvolve(self._orig_obj._sbp, self._real_space,
                                                  self.gsparams._gsp)

    def shoot(self, n_photons, rng=None):
        """Shoot photons into a PhotonArray.

        @param n_photons    The number of photons to use for photon shooting.
        @param rng          If provided, a random number generator to use for photon shooting,
                            which may be any kind of BaseDeviate object.  If `rng` is None, one
                            will be automatically created, using the time as a seed.
                            [default: None]
        @returns PhotonArray.
        """
        ud = galsim.UniformDeviate(rng)

        photon_array = self._orig_obj.shoot(n_photons, ud)
        photon_array.convolve(self._orig_obj.shoot(n_photons, ud), ud)
        return photon_array

    def __getstate__(self):
        d = self.__dict__.copy()
        del d['_sbp']
        return d

    def __setstate__(self, d):
        self.__dict__ = d
        self.__init__(self._orig_obj, self._real_space, self._gsparams)


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
        hard_edge = obj.has_hard_edges

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
        if real_space and not obj.is_analytic_x:
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
        self._gsparams = galsim.GSParams.check(gsparams, self._orig_obj.gsparams)

        self._sbp = galsim._galsim.SBAutoCorrelate(obj._sbp, real_space, self.gsparams._gsp)
        if obj.noise is not None:
            import warnings
            warnings.warn("Unable to propagate noise in galsim.AutoCorrelation")

    @property
    def orig_obj(self): return self._orig_obj
    @property
    def real_space(self): return self._real_space

    def __eq__(self, other):
        return (isinstance(other, galsim.AutoCorrelation) and
                self.orig_obj == other.orig_obj and
                self.real_space == other.real_space and
                self.gsparams == other.gsparams)

    def __hash__(self):
        return hash(("galsim.AutoCorrelation", self.orig_obj, self.real_space, self.gsparams))

    def __repr__(self):
        return 'galsim.AutoCorrelation(%r, real_space=%r, gsparams=%r)'%(
                self.orig_obj, self.real_space, self.gsparams)

    def __str__(self):
        s = 'galsim.AutoCorrelate(%s'%self.orig_obj
        if self.real_space:
            s += ', real_space=True'
        s += ')'
        return s

    def _prepareDraw(self):
        self._orig_obj._prepareDraw()
        self._sbp = galsim._galsim.SBAutoCorrelate(self._orig_obj._sbp,
                                                   self._real_space, self.gsparams._gsp)

    def shoot(self, n_photons, rng=None):
        """Shoot photons into a PhotonArray.

        @param n_photons    The number of photons to use for photon shooting.
        @param rng          If provided, a random number generator to use for photon shooting,
                            which may be any kind of BaseDeviate object.  If `rng` is None, one
                            will be automatically created, using the time as a seed.
                            [default: None]
        @returns PhotonArray.
        """
        ud = galsim.UniformDeviate(rng)

        photons = self._orig_obj.shoot(n_photons, ud)
        photons2 = self._orig_obj.shoot(n_photons, ud)

        # Flip sign of (x, y) in one of the results
        photons2.scaleXY(-1)

        photons.convolve(photons2, ud)
        return photons

    def __getstate__(self):
        d = self.__dict__.copy()
        del d['_sbp']
        return d

    def __setstate__(self, d):
        self.__dict__ = d
        self.__init__(self._orig_obj, self._real_space, self._gsparams)
