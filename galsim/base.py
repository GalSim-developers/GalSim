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
"""@file base.py
Definitions for the GalSim base classes and associated methods

This file includes the key parts of the user interface to GalSim: base classes representing surface
brightness profiles for astronomical objects (galaxies, PSFs, pixel response).  These base classes
are collectively known as GSObjects.  They include simple objects like the Gaussian class, a 2d
Gaussian intensity profile, whereas compound GSObjects can be found in other appropriately-named
files.  For example, the Sum and Convolution, which represent the sum and convolution of multiple
GSObjects, respectively, can be found in compound.py; RealGalaxy objects are defined in real.py;
and so on.

These classes also have associated methods to (a) retrieve information (like the flux, half-light
radius, or intensity at a particular point); (b) carry out common operations, like shearing,
rescaling of flux or size, rotating, and shifting; and (c) actually make images of the surface
brightness profiles.

For a description of units conventions for scale radii for our base classes see
`doc/GalSim_Quick_Reference.pdf`, section 2.2.  In short, any system that will ensure consistency
between the scale radii used to specify the size of the GSObject and between the pixel scale of the
Image is acceptable.
"""

import os

import numpy as np

import galsim
import utilities

from . import _galsim
from ._galsim import GSParams


class GSObject(object):
    """Base class for all GalSim classes that represent some kind of surface brightness profile.

    A GSObject is not intended to be constructed directly.  Normally, you would use whatever
    derived class is appropriate for the surface brightness profile you want:

        >>> gal = galsim.Sersic(n=4, half_light_radius=4.3)
        >>> psf = galsim.Moffat(beta=3, fwhm=2.85)
        >>> conv = galsim.Convolve([gal,psf])

    All of these classes are subclasses of GSObject, so you should see those docstrings for
    more details about how to construct the various profiles.

    Note that most GSObjects have some kind of size specification.  Typically, these would be
    given in terms of arcsec, with the connection to the pixel size being given in the Pixel
    class (0.2 arcsec/pixel in the above example).  However, you can have a more complicated
    relationship between pixel and sky coordinates.  See BaseWCS for more details about
    how to specify various kinds of world coordinate systems.

    Transforming Methods
    --------------------

    The GSObject class uses an "immutable" design[1], so all methods that would potentially modify
    the object actually return a new object instead.  This uses pointers and such behind the
    scenes, so it all happens efficiently, but it makes using the objects a bit simpler, since
    you don't need to worry about some function changing your object behind your back.

    In all cases below, we just give an example usage.  See the docstrings for the methods for
    more details about how to use them.

        >>> obj = obj.shear(shear)      # Apply a shear to the object.
        >>> obj = obj.dilate(scale)     # Apply a flux-preserving dilation.
        >>> obj = obj.magnify(mu)       # Apply a surface-brightness-preserving magnification.
        >>> obj = obj.rotate(theta)     # Apply a rotation.
        >>> obj = obj.shift(dx,dy)      # Shft the object in real space.
        >>> obj = obj.transform(dudx,dudy,dvdx,dvdy)    # Apply a general jacobian transformation.
        >>> obj = obj.lens(g1,g2,mu)    # Apply both a lensing shear and magnification.
        >>> obj = obj.withFlux(flux)    # Set a new flux value.
        >>> obj = obj * ratio           # Scale the surface brightness profile by some factor.

    [1]: Technically, there are some methods that do modify the object directly.  However, these
    methods are only present for backwards compatibitility with previous versions of GalSim,
    and they are currently discouraged.  They will be deprecated in a future version and
    removed entirely in version 2.0.

    Access Methods
    --------------

    There are some access methods that are available for all GSObjects.  Again, see the docstrings
    for each method for more details.

        >>> flux = obj.getFlux()
        >>> centroid = obj.centroid()
        >>> f_xy = obj.xValue(x,y)
        >>> fk_xy = obj.kValue(kx,ky)
        >>> nyq = obj.nyquistScale()
        >>> stepk = obj.stepK()
        >>> maxk = obj.maxK()
        >>> hard = obj.hasHardEdges()
        >>> axisym = obj.isAxisymmetric()
        >>> analytic = obj.isAnalyticX()

    Most subclasses have additional methods that are available for values that are particular to
    that specific surface brightness profile.  e.g. `sigma = gauss.getSigma()`.  However, note
    that class-specific methods are not available after performing one of the above transforming
    operations.

        >>> gal = galsim.Gaussian(sigma=5)
        >>> gal = gal.shear(g1=0.2, g2=0.05)
        >>> sigma = gal.getSigma()              # This will raise an exception.

    It is however possible to access the original object that was transformed via the
    `original` attribute.

        >>> sigma = gal.original.getSigma()     # This works.

    No matter how many transformations are performed, the `original` attribute will contain the
    _original_ object (not necessarily the most recent ancestor).

    Drawing Methods
    ---------------

    The main thing to do with a GSObject once you have built it is to draw it onto an image.
    There are two methods that do this.  In both cases, there are lots of optional parameters.
    See the docstrings for these methods for more details.

        >>> image = obj.drawImage(...)
        >>> kimage_r, kimage_i = obj.drawKImage(...)

    Attributes
    ----------

    There two attributes that may be available for a GSObject.

        original    This was mentioned above as a way to access the original object that has
                    been transformed by one of the transforming methods.

        noise       Some types, like RealGalaxy, set this attribute to be the intrinsic noise that
                    is already inherent in the profile and will thus be present when you draw the
                    object.  The noise is propagated correctly through the various transforming
                    methods, as well as convolutions and flux rescalings.  Note that the `noise`
                    attribute can be set directly by users even for GSObjects that do not naturally
                    have one. The typical use for this attribute is to use it to whiten the noise in
                    the image after drawing.  See CorrelatedNoise for more details.

    GSParams
    --------

    All GSObject classes take an optional `gsparams` argument, so we document that feature here.
    For all documentation about the specific derived classes, please see the docstring for each
    one individually.

    The `gsparams` argument can be used to specify various numbers that govern the tradeoff between
    accuracy and speed for the calculations made in drawing a GSObject.  The numbers are
    encapsulated in a class called GSParams, and the user should make careful choices whenever they
    opt to deviate from the defaults.  For more details about the parameters and their default
    values, please see the docstring of the GSParams class (e.g. type `help(galsim.GSParams)`).

    For example, let's say you want to do something that requires an FFT larger than 4096 x 4096
    (and you have enough memory to handle it!).  Then you can create a new GSParams object with a
    larger `maximum_fft_size` and pass that to your GSObject on construction:

        >>> gal = galsim.Sersic(n=4, half_light_radius=4.3)
        >>> psf = galsim.Moffat(beta=3, fwhm=2.85)
        >>> conv = galsim.Convolve([gal,psf])
        >>> im = galsim.Image(1000,1000, scale=0.05)        # Note the very small pixel scale!
        >>> im = conv.drawImage(image=im)                   # This uses the default GSParams.
        Traceback (most recent call last):
          File "<stdin>", line 1, in <module>
          File "galsim/base.py", line 1236, in drawImage
            image.added_flux = prof.SBProfile.draw(imview.image, gain, wmult)
        RuntimeError: SB Error: fourierDraw() requires an FFT that is too large, 6144
        If you can handle the large FFT, you may update gsparams.maximum_fft_size.
        >>> big_fft_params = galsim.GSParams(maximum_fft_size=10240)
        >>> conv = galsim.Convolve([gal,psf],gsparams=big_fft_params)
        >>> im = conv.drawImage(image=im)                   # Now it works (but is slow!)
        >>> im.write('high_res_sersic.fits')

    Note that for compound objects in compound.py, like Convolution or Sum, not all GSParams can be
    changed when the compound object is created.  In the example given here, it is possible to
    change parameters related to the drawing, but not the Fourier space parameters for the
    components that go into the Convolution.  To get better sampling in Fourier space, for example,
    the `gal` and/or `psf` should be created with `gsparams` that have a non-default value of
    `folding_threshold`.  This statement applies to the threshold and accuracy parameters.
    """
    _gsparams = { 'minimum_fft_size' : int,
                  'maximum_fft_size' : int,
                  'folding_threshold' : float,
                  'stepk_minimum_hlr' : float,
                  'maxk_threshold' : float,
                  'kvalue_accuracy' : float,
                  'xvalue_accuracy' : float,
                  'realspace_relerr' : float,
                  'realspace_abserr' : float,
                  'integration_relerr' : float,
                  'integration_abserr' : float,
                  'shoot_accuracy' : float,
                  'allowed_flux_variation' : float,
                  'range_division_for_extrema' : int,
                  'small_fraction_of_flux' : float
                }
    def __init__(self, obj):
        # This guarantees that all GSObjects have an SBProfile
        if isinstance(obj, GSObject):
            self.SBProfile = obj.SBProfile
            if hasattr(obj,'noise'):
                self.noise = obj.noise.copy()
        elif isinstance(obj, galsim._galsim.SBProfile):
            self.SBProfile = obj
        else:
            raise TypeError("GSObject must be initialized with an SBProfile or another GSObject!")
        # a couple of definitions for using GSObjects as duck-typed ChromaticObjects
        self.separable = True
        self.SED = lambda w: 1.0 # flat spectrum in photons/nanometer
        self.wave_list = np.array([], dtype=float)

    # Also need this method to duck-type as a ChromaticObject
    def evaluateAtWavelength(self, wave):
        """Return profile at a given wavelength.  For GSObject instances, this is just `self`.
        This allows GSObject instances to be duck-typed as ChromaticObject instances."""
        return self

    # Make op+ of two GSObjects work to return an Add object
    # Note: we don't define __iadd__ and similar.  Let python handle this automatically
    # to make obj += obj2 be equivalent to obj = obj + obj2.
    def __add__(self, other):
        return galsim.Add([self, other])

    # op- is unusual, but allowed.  It subtracts off one profile from another.
    def __sub__(self, other):
        return galsim.Add([self, (-1. * other)])

    # Make op* work to adjust the flux of an object
    def __mul__(self, other):
        """Scale the flux of the object by the given factor.

        obj * flux_ratio is equivalent to obj.withScaledFlux(flux_ratio)

        It creates a new object that has the same profile as the original, but with the
        surface brightness at every location scaled by the given amount.

        You can also multiply by an SED, which will create a ChromaticObject where the SED
        acts like a wavelength-dependent `flux_ratio`.

        obj * sed is equivalent to galsim.Chromatic(obj, sed)
        """
        if isinstance(other, galsim.SED):
            return galsim.Chromatic(self, other)
        else:
            return self.withScaledFlux(other)

    def __rmul__(self, other):
        """Equivalent to obj * other"""
        return self.__mul__(other)

    # Likewise for op/
    def __div__(self, other):
        """Equivalent to obj * (1/other)"""
        return self * (1. / other)

    def __truediv__(self, other):
        """Equivalent to obj * (1/other)"""
        return __div__(self, other)

    # Make a copy of an object
    def copy(self):
        """Returns a copy of an object.

        This preserves the original type of the object, so if the caller is a Gaussian (for
        example), the copy will also be a Gaussian, and can thus call the methods that are not in
        GSObject, but are in Gaussian (e.g. getSigma()).  However, not necessarily all instance
        attributes will be copied across (e.g. the interpolant stored by an OpticalPSF object).
        """
        # Re-initialize a return GSObject with self's SBProfile
        sbp = self.SBProfile.__class__(self.SBProfile)
        ret = GSObject(sbp)
        ret.__class__ = self.__class__
        if hasattr(self,'noise'): ret.noise = self.noise.copy()
        return ret

    # Now define direct access to all SBProfile methods via calls to self.SBProfile.method_name()
    #
    def maxK(self):
        """Returns value of k beyond which aliasing can be neglected.
        """
        return self.SBProfile.maxK()

    def nyquistScale(self):
        """Returns Image pixel spacing that does not alias maxK.
        """
        return self.SBProfile.nyquistDx()

    def nyquistDx(self):
        """An obsolete synonym for nyquistScale()"""
        return self.nyquistScale()

    def stepK(self):
        """Returns sampling in k space necessary to avoid folding of image in x space.
        """
        return self.SBProfile.stepK()

    def hasHardEdges(self):
        """Returns True if there are any hard edges in the profile, which would require very small k
        spacing when working in the Fourier domain.
        """
        return self.SBProfile.hasHardEdges()

    def isAxisymmetric(self):
        """Returns True if axially symmetric: affects efficiency of evaluation.
        """
        return self.SBProfile.isAxisymmetric()

    def isAnalyticX(self):
        """Returns True if real-space values can be determined immediately at any position without
        requiring a Discrete Fourier Transform.
        """
        return self.SBProfile.isAnalyticX()

    def isAnalyticK(self):
        """Returns True if k-space values can be determined immediately at any position without
        requiring a Discrete Fourier Transform.
        """
        return self.SBProfile.isAnalyticK()

    def centroid(self):
        """Returns the (x, y) centroid of an object as a Position.
        """
        return self.SBProfile.centroid()

    def getFlux(self):
        """Returns the flux of the object.
        """
        return self.SBProfile.getFlux()

    def xValue(self, *args, **kwargs):
        """Returns the value of the object at a chosen 2D position in real space.

        This function returns the surface brightness of the object at a particular position
        in real space.  The position argument may be provided as a PositionD or PositionI
        argument, or it may be given as x,y (either as a tuple or as two arguments).

        The object surface brightness profiles are typically defined in world coordinates, so
        the position here should be in world coordinates as well.

        Not all GSObject classes can use this method.  Classes like Convolution that require a
        Discrete Fourier Transform to determine the real space values will not do so for a single
        position.  Instead a RuntimeError will be raised.  The xValue() method is available if and
        only if `obj.isAnalyticX() == True`.

        Users who wish to use the xValue() method for an object that is the convolution of other
        profiles can do so by drawing the convolved profile into an image, using the image to
        initialize a new InterpolatedImage, and then using the xValue() method for that new object.

        @param position  The position at which you want the surface brightness of the object.

        @returns the surface brightness at that position.
        """
        pos = galsim.utilities.parse_pos_args(args,kwargs,'x','y')
        return self.SBProfile.xValue(pos)

    def kValue(self, *args, **kwargs):
        """Returns the value of the object at a chosen 2D position in k space.

        This function returns the amplitude of the fourier transform of the surface brightness
        profile at a given position in k space.  The position argument may be provided as a
        PositionD or PositionI argument, or it may be given as kx,ky (either as a tuple or as two
        arguments).

        Techinically, kValue() is available if and only if the given obj has `obj.isAnalyticK()
        == True`, but this is the case for all GSObjects currently, so that should never be an
        issue (unlike for xValue()).

        @param position  The position in k space at which you want the fourier amplitude.

        @returns the amplitude of the fourier transform at that position.
        """
        kpos = galsim.utilities.parse_pos_args(args,kwargs,'kx','ky')
        return self.SBProfile.kValue(kpos)

    def withFlux(self, flux):
        """Create a version of the current object with a different flux.

        This function is equivalent to `obj.withScaledFlux(flux / obj.getFlux())`.

        It creates a new object that has the same profile as the original, but with the
        surface brightness at every location rescaled such that the total flux will be
        the given value.

        @param flux     The new flux for the object.

        @returns the object with the new flux
        """
        return self.withScaledFlux(flux / self.getFlux())

    def withScaledFlux(self, flux_ratio):
        """Create a version of the current object with the flux scaled by the given `flux_ratio`.

        This function is equivalent to `obj.withFlux(flux_ratio * obj.getFlux())`.  However, this
        function is the more efficient one, since it doesn't actually require the call to
        getFlux().  Indeed, withFlux() is implemented in terms of this one and getFlux().

        It creates a new object that has the same profile as the original, but with the
        surface brightness at every location scaled by the given amount.

        An equivalent, and usually simpler, way to effect this scaling is

            obj = obj * flux_ratio

        @param flux_ratio   The ratio by which to rescale the flux of the object when creating a new
                            one.

        @returns the object with the new flux.
        """
        new_obj = GSObject(self.SBProfile.scaleFlux(flux_ratio))
        if hasattr(self,'original'):
            new_obj.original = self.original
        else:
            new_obj.original = self

        if hasattr(self,'noise'):
            new_obj.noise = self.noise * flux_ratio**2
        return new_obj

    def setFlux(self, flux):
        """This is an obsolete method that is roughly equivalent to obj = obj.withFlux(flux)"""
        new_obj = self.withFlux(flux)
        self.SBProfile = new_obj.SBProfile
        if hasattr(self,'noise'): self.noise = new_obj.noise
        self.__class__ = new_obj.__class__

    def scaleFlux(self, flux_ratio):
        """This is an obsolete method that is roughly equivalent to obj = obj * flux_ratio"""
        new_obj = self * flux_ratio
        self.SBProfile = new_obj.SBProfile
        if hasattr(self,'noise'): self.noise = new_obj.noise
        self.__class__ = new_obj.__class__

    def expand(self, scale):
        """Expand the linear size of the profile by the given `scale` factor, while preserving
        surface brightness.

        e.g. `half_light_radius` <-- `half_light_radius * scale`

        This doesn't correspond to either of the normal operations one would typically want to do to
        a galaxy.  The functions dilate() and magnify() are the more typical usage.  But this
        function is conceptually simple.  It rescales the linear dimension of the profile, while
        preserving surface brightness.  As a result, the flux will necessarily change as well.

        See dilate() for a version that applies a linear scale factor while preserving flux.

        See magnify() for a version that applies a scale factor to the area while preserving surface
        brightness.

        @param scale    The factor by which to scale the linear dimension of the object.

        @returns the expanded object.
        """
        new_obj = GSObject(self.SBProfile.expand(scale))
        if hasattr(self,'original'):
            new_obj.original = self.original
        else:
            new_obj.original = self

        if hasattr(self,'noise'):
            new_obj.noise = self.noise.expand(scale)
        return new_obj

    def createExpanded(self, scale):
        """This is an obsolete synonym for expand(scale)"""
        return self.expand(scale)

    def applyExpansion(self, scale):
        """This is an obsolete method that is roughly equivalent to obj = obj.expand(scale)."""
        new_obj = self.expand(scale)
        self.SBProfile = new_obj.SBProfile
        if hasattr(self,'noise'): self.noise = new_obj.noise
        self.__class__ = new_obj.__class__

    def dilate(self, scale):
        """Dilate the linear size of the profile by the given `scale` factor, while preserving
        flux.

        e.g. `half_light_radius` <-- `half_light_radius * scale`

        See expand() and magnify() for versions that preserve surface brightness, and thus
        changes the flux.

        @param scale    The linear rescaling factor to apply.

        @returns the dilated object.
        """
        return self.expand(scale) * (1./scale**2)  # conserve flux

    def createDilated(self, scale):
        """This is an obsolete synonym for dilate(scale)"""
        return self.dilate(scale)

    def applyDilation(self, scale):
        """This is an obsolete method that is roughly equivalent to obj = obj.dilate(scale)."""
        new_obj = self.dilate(scale)
        self.SBProfile = new_obj.SBProfile
        if hasattr(self,'noise'): self.noise = new_obj.noise
        self.__class__ = new_obj.__class__

    def magnify(self, mu):
        """Create a version of the current object with a lensing magnification applied to it,
        scaling the area and flux by `mu` at fixed surface brightness.

        This process applies a lensing magnification mu, which scales the linear dimensions of the
        image by the factor sqrt(mu), i.e., `half_light_radius` <-- `half_light_radius * sqrt(mu)`
        while increasing the flux by a factor of mu.  Thus, magnify() preserves surface brightness.

        See dilate() for a version that applies a linear scale factor while preserving flux.

        See expand() for a version that applies a linear scale factor while preserving surface
        brightness.

        @param mu   The lensing magnification to apply.

        @returns the magnified object.
        """
        import math
        return self.expand(math.sqrt(mu))

    def createMagnified(self, mu):
        """This is an obsolete synonym for magnify(mu)"""
        return self.magnify(mu)

    def applyMagnification(self, mu):
        """This is an obsolete method that is roughly equivalent to obj = obj.magnify(mu)"""
        new_obj = self.magnify(mu)
        self.SBProfile = new_obj.SBProfile
        if hasattr(self,'noise'): self.noise = new_obj.noise
        self.__class__ = new_obj.__class__

    def shear(self, *args, **kwargs):
        """Create a version of the current object with an area-preserving shear applied to it.

        The arguments may be either a Shear instance or arguments to be used to initialize one.

        For more details about the allowed keyword arguments, see the documentation for Shear
        (for doxygen documentation, see galsim.shear.Shear).

        The shear() method precisely preserves the area.  To include a lensing distortion with
        the appropriate change in area, either use shear() with magnify(), or use lens(), which
        combines both operations.

        @param shear    The Shear to be applied. Or, as described above, you may instead supply
                        parameters do construct a shear directly.  eg. `obj.shear(g1=g1,g2=g2)`.

        @returns the sheared object.
        """
        if len(args) == 1:
            if kwargs:
                raise TypeError("Error, gave both unnamed and named arguments to GSObject.shear!")
            if not isinstance(args[0], galsim.Shear):
                raise TypeError("Error, unnamed argument to GSObject.shear is not a Shear!")
            shear = args[0]
        elif len(args) > 1:
            raise TypeError("Error, too many unnamed arguments to GSObject.shear!")
        else:
            shear = galsim.Shear(**kwargs)

        new_obj = GSObject(self.SBProfile.shear(shear._shear))
        if hasattr(self,'original'):
            new_obj.original = self.original
        else:
            new_obj.original = self

        if hasattr(self,'noise'):
            new_obj.noise = self.noise.shear(shear)
        return new_obj

    def createSheared(self, *args, **kwargs):
        """This is an obsolete synonym for shear(shear)"""
        return self.shear(*args, **kwargs)

    def applyShear(self, *args, **kwargs):
        """This is an obsolete method that is roughly equivalent to obj = obj.shear(shear)"""
        new_obj = self.shear(*args, **kwargs)
        self.SBProfile = new_obj.SBProfile
        if hasattr(self,'noise'): self.noise = new_obj.noise
        self.__class__ = new_obj.__class__

    def lens(self, g1, g2, mu):
        """Create a version of the current object with both a lensing shear and magnification
        applied to it.

        This GSObject method applies a lensing (reduced) shear and magnification.  The shear must be
        specified using the g1, g2 definition of shear (see Shear documentation for more details).
        This is the same definition as the outputs of the PowerSpectrum and NFWHalo classes, which
        compute shears according to some lensing power spectrum or lensing by an NFW dark matter
        halo.  The magnification determines the rescaling factor for the object area and flux,
        preserving surface brightness.

        @param g1       First component of lensing (reduced) shear to apply to the object.
        @param g2       Second component of lensing (reduced) shear to apply to the object.
        @param mu       Lensing magnification to apply to the object.  This is the factor by which
                        the solid angle subtended by the object is magnified, preserving surface
                        brightness.

        @returns the lensed object.
        """
        return self.shear(g1=g1,g2=g2).magnify(mu)

    def createLensed(self, g1, g2, mu):
        """This is an obsolete synonym for lens(g1,g2,mu)"""
        return self.lens(g1,g2,mu)

    def applyLensing(self, g1, g2, mu):
        """This is an obsolete method that is roughly equivalent to obj = obj.lens(g1,g2,mu)"""
        new_obj = self.lens(g1,g2,mu)
        self.SBProfile = new_obj.SBProfile
        if hasattr(self,'noise'): self.noise = new_obj.noise
        self.__class__ = new_obj.__class__

    def rotate(self, theta):
        """Rotate this object by an Angle `theta`.

        @param theta    Rotation angle (Angle object, +ve anticlockwise).

        @returns the rotated object.
        """
        if not isinstance(theta, galsim.Angle):
            raise TypeError("Input theta should be an Angle")
        new_obj = GSObject(self.SBProfile.rotate(theta))
        if hasattr(self,'original'):
            new_obj.original = self.original
        else:
            new_obj.original = self

        if hasattr(self,'noise'):
            new_obj.noise = self.noise.rotate(theta)
        return new_obj

    def createRotated(self, theta):
        """This is an obsolete synonym for rotate(theta)"""
        return self.rotate(theta)

    def applyRotation(self, theta):
        """This is an obsolete method that is roughly equivalent to obj = obj.rotate(theta)"""
        new_obj = self.rotate(theta)
        self.SBProfile = new_obj.SBProfile
        if hasattr(self,'noise'): self.noise = new_obj.noise
        self.__class__ = new_obj.__class__

    def transform(self, dudx, dudy, dvdx, dvdy):
        """Create a version of the current object with an arbitrary Jacobian matrix transformation
        applied to it.

        This applies a Jacobian matrix to the coordinate system in which this object
        is defined.  It changes a profile defined in terms of (x,y) to one defined in
        terms of (u,v) where:

            u = dudx x + dudy y
            v = dvdx x + dvdy y

        That is, an arbitrary affine transform, but without the translation (which is
        easily effected via the shift() method).

        Note that this function is similar to expand in that it preserves surface brightness,
        not flux.  If you want to preserve flux, you should also do

            >>> prof *= 1./abs(dudx*dvdy - dudy*dvdx)

        @param dudx     du/dx, where (x,y) are the current coords, and (u,v) are the new coords.
        @param dudy     du/dy, where (x,y) are the current coords, and (u,v) are the new coords.
        @param dvdx     dv/dx, where (x,y) are the current coords, and (u,v) are the new coords.
        @param dvdy     dv/dy, where (x,y) are the current coords, and (u,v) are the new coords.

        @returns the transformed object
        """
        new_obj = GSObject(self.SBProfile.transform(dudx,dudy,dvdx,dvdy))
        if hasattr(self,'original'):
            new_obj.original = self.original
        else:
            new_obj.original = self

        if hasattr(self,'noise'):
            new_obj.noise = self.noise.transform(dudx,dudy,dvdx,dvdy)
        return new_obj

    def createTransformed(self, dudx, dudy, dvdx, dvdy):
        """This is an obsolete sysnonym for transform()"""
        return self.transform(dudx,dudy,dvdx,dvdy)

    def applyTransformation(self, dudx, dudy, dvdx, dvdy):
        """This is an obsolete method that is roughly equivalent to obj = obj.transform(...)"""
        new_obj = self.transform(dudx,dudy,dvdx,dvdy)
        self.SBProfile = new_obj.SBProfile
        if hasattr(self,'noise'): self.noise = new_obj.noise
        self.__class__ = new_obj.__class__

    def shift(self, *args, **kwargs):
        """Create a version of the current object shifted by some amount in real space.

        After this call, the caller's type will be a GSObject.
        This means that if the caller was a derived type that had extra methods beyond
        those defined in GSObject (e.g. getSigma() for a Gaussian), then these methods
        are no longer available.

        Note: in addition to the dx,dy parameter names, you may also supply dx,dy as a tuple,
        or as a PositionD or PositionI object.

        @param dx       Horizontal shift to apply.
        @param dy       Vertical shift to apply.

        @returns the shifted object.
        """
        delta = galsim.utilities.parse_pos_args(args, kwargs, 'dx', 'dy')
        new_obj = GSObject(self.SBProfile.shift(delta))
        if hasattr(self,'original'):
            new_obj.original = self.original
        else:
            new_obj.original = self

        if hasattr(self,'noise'):
            new_obj.noise = self.noise.copy()
        return new_obj

    def createShifted(self, *args, **kwargs):
        """This is an obsolete synonym for shift(dx,dy)"""
        return self.shift(*args,**kwargs)

    def applyShift(self, *args, **kwargs):
        """This is an obsolete method that is roughly equivalent to obj = obj.shift(dx,dy)"""
        new_obj = self.shift(*args,**kwargs)
        self.SBProfile = new_obj.SBProfile
        if hasattr(self,'noise'): self.noise = new_obj.noise
        self.__class__ = new_obj.__class__


    # Make sure the image is defined with the right size and wcs for drawImage()
    def _setup_image(self, image, nx, ny, bounds, wmult, add_to_image, dtype):
        # Check validity of nx,ny,bounds:
        if image is not None:
            if bounds is not None:
                raise ValueError("Cannot provide bounds if image is provided")
            if nx is not None or ny is not None:
                raise ValueError("Cannot provide nx,ny if image is provided")
            if dtype is not None:
                raise ValueError("Cannot specify dtype if image is provided")

        # Make image if necessary
        if image is None:
            # Can't add to image if none is provided.
            if add_to_image:
                raise ValueError("Cannot add_to_image if image is None")
            # Use bounds or nx,ny if provided
            if bounds is not None:
                if nx is not None or ny is not None:
                    raise ValueError("Cannot set both bounds and (nx, ny)")
                image = galsim.Image(bounds=bounds, dtype=dtype)
            elif nx is not None or ny is not None:
                if nx is None or ny is None:
                    raise ValueError("Must set either both or neither of nx, ny")
                image = galsim.Image(nx, ny, dtype=dtype)
            else:
                N = self.SBProfile.getGoodImageSize(1.0, wmult)
                image = galsim.Image(N, N, dtype=dtype)

        # Resize the given image if necessary
        elif not image.bounds.isDefined():
            # Can't add to image if need to resize
            if add_to_image:
                raise ValueError("Cannot add_to_image if image bounds are not defined")
            N = self.SBProfile.getGoodImageSize(1.0, wmult)
            bounds = galsim.BoundsI(1,N,1,N)
            image.resize(bounds)
            image.setZero()

        # Else use the given image as is
        else:
            # Clear the image if we are not adding to it.
            if not add_to_image:
                image.setZero()

        return image

    def _local_wcs(self, wcs, image, offset, use_true_center):
        # Get the local WCS at the location of the object.

        if wcs.isUniform():
            return wcs.local()
        elif image is None:
            # Should have already checked for this, but just to be safe, repeat the check here.
            raise ValueError("Cannot provide non-local wcs when image is None")
        elif not image.bounds.isDefined():
            raise ValueError("Cannot provide non-local wcs when image has undefined bounds")
        elif use_true_center:
            obj_cen = image.bounds.trueCenter()
        else:
            obj_cen = image.bounds.center()
            # Convert from PositionI to PositionD
            obj_cen = galsim.PositionD(obj_cen.x, obj_cen.y)
        if offset:
            obj_cen += offset
        return wcs.local(image_pos=obj_cen)

    def _parse_offset(self, offset):
        if offset is None:
            return galsim.PositionD(0,0)
        else:
            if isinstance(offset, galsim.PositionD) or isinstance(offset, galsim.PositionI):
                return galsim.PositionD(offset.x, offset.y)
            else:
                # Let python raise the appropriate exception if this isn't valid.
                return galsim.PositionD(offset[0], offset[1])


    def _fix_center(self, image, offset, use_true_center, reverse):
        # Note: this assumes self is in terms of image coordinates.
        if image is None or not image.bounds.isDefined():
            shape = (0,0)
        else:
            shape = image.array.shape

        if use_true_center:
            # For even-sized images, the SBProfile draw function centers the result in the
            # pixel just up and right of the real center.  So shift it back to make sure it really
            # draws in the center.
            # Also, remember that numpy's shape is ordered as [y,x]
            dx = offset.x
            dy = offset.y
            if shape[1] % 2 == 0: dx -= 0.5
            if shape[0] % 2 == 0: dy -= 0.5
            offset = galsim.PositionD(dx,dy)

        # For InterpolatedImage offsets, we apply the offset in the opposite direction.
        if reverse:
            offset = -offset

        if offset == galsim.PositionD(0,0):
            return self.copy()
        else:
            return self.shift(offset)

    def _determine_wcs(self, scale, wcs, image):
        # Determine the correct wcs given the input scale, wcs and image.
        if wcs is not None:
            if scale is not None:
                raise ValueError("Cannot provide both wcs and scale")
            if not wcs.isUniform():
                if image is None:
                    raise ValueError("Cannot provide non-local wcs when image is None")
                if not image.bounds.isDefined():
                    raise ValueError("Cannot provide non-local wcs when image has undefined bounds")
            if not isinstance(wcs, galsim.BaseWCS):
                raise TypeError("wcs must be a BaseWCS instance")
            if image is not None: image.wcs = None
        elif scale is not None:
            wcs = galsim.PixelScale(scale)
            if image is not None: image.wcs = None
        elif image is not None and image.wcs is not None:
            wcs = image.wcs

        # If the input scale <= 0, or wcs is still None at this point, then use the Nyquist scale:
        if wcs is None or (wcs.isPixelScale() and wcs.scale <= 0):
            wcs = galsim.PixelScale(self.nyquistScale())

        return wcs

    def drawImage(self, image=None, nx=None, ny=None, bounds=None, scale=None, wcs=None, dtype=None,
                  method='auto', gain=1., wmult=1., add_to_image=False, use_true_center=True,
                  offset=None, n_photons=0., rng=None, max_extra_noise=0., poisson_flux=None,
                  setup_only=False, dx=None):
        """Draws an Image of the object.

        The drawImage() method is used to draw an Image of the current object using one of several
        possible rendering methods (see below).  It can create a new Image or can draw onto an
        existing one if provided by the `image` parameter.  If the `image` is given, you can also
        optionally add to the given Image if `add_to_image = True`, but the default is to replace
        the current contents with new values.

        Note that if you provide an `image` parameter, it is the image onto which the profile
        will be drawn.  The provided image *will be modified*.  A reference to the same image
        is also returned to provide a parallel return behavior to when `image` is `None`
        (described above).

        This option is useful in practice because you may want to construct the image first and
        then draw onto it, perhaps multiple times. For example, you might be drawing onto a
        subimage of a larger image. Or you may want to draw different components of a complex
        profile separately.  In this case, the returned value is typically ignored.  For example:

                >>> im1 = bulge.drawImage()
                >>> im2 = disk.drawImage(image=im1, add_to_image=True)
                >>> assert im1 is im2

                >>> full_image = galsim.Image(2048, 2048, scale=pixel_scale)
                >>> b = galsim.BoundsI(x-32, x+32, y-32, y+32)
                >>> stamp = obj.drawImage(image = full_image[b])
                >>> assert (stamp.array == full_image[b].array).all()

        If drawImage() will be creating the image from scratch for you, then there are several ways
        to control the size of the new image.  If the `nx` and `ny` keywords are present, then an
        image with these numbers of pixels on a side will be created.  Similarly, if the `bounds`
        keyword is present, then an image with the specified bounds will be created.  Note that it
        is an error to provide an existing Image when also specifying `nx`, `ny`, or `bounds`. In
        the absence of `nx`, `ny`, and `bounds`, drawImage will decide a good size to use based on
        the size of the object being drawn.  Basically, it will try to use an area large enough to
        include at least 99.5% of the flux.  (Note: the value 0.995 is really `1 -
        folding_threshold`.  You can change the value of `folding_threshold` for any object via
        GSParams.  See `help(GSParams)` for more details.)  You can set the pixel scale of the
        constructed image with the `scale` parameter, or set a WCS function with `wcs`.  If you do
        not provide either `scale` or `wcs`, then drawImage() will default to using the Nyquist
        scale for the current object.  You can also set the data type used in the new Image with the
        `dtype` parameter that has the same options as for the Image constructor.

        There are several different possible methods drawImage() can use for rendering the image.
        This is set by the `method` parameter.  The options are:

            'auto'      This is the default, which will normally be equivalent to 'fft'.  However,
                        if the object being rendered is simple (no convolution) and has hard edges
                        (e.g. a Box or a truncated Moffat or Sersic), then it will switch to
                        'real_space', since that is often both faster and more accurate in these
                        cases (due to ringing in Fourier space).

            'fft'       The integration of the light within each pixel is mathematically equivalent
                        to convolving by the pixel profile (a Pixel object) and sampling the result
                        at the centers of the pixels.  This method will do that convolution using
                        a discrete Fourier transform.  Furthermore, if the object (or any component
                        of it) has been transformed via shear(), dilate(), etc., then these
                        transformations are done in Fourier space as well.

            'real_space'  This uses direct integrals (using the Gauss-Kronrod-Patterson algorithm)
                        in real space for the integration over the pixel response.  It is usually
                        slower than the 'fft' method, but if the profile has hard edges that cause
                        ringing in Fourier space, it can be faster and/or more accurate.  If you
                        use 'real_space' with something that is already a Convolution, then this
                        will revert to 'fft', since the double convolution that is required to also
                        handle the pixel response is far too slow to be practical using real-space
                        integrals.

            'phot'      This uses a technique called photon shooting to render the image.
                        Essentially, the object profile is taken as a probability distribution
                        from which a finite number of photons are "shot" onto the image.  Each
                        photon's flux gets added to whichever pixel the photon hits.  This process
                        automatically accounts for the integration of the light over the pixel
                        area, since all photons that hit any part of the pixel are counted.
                        Convolutions and transformations are simple geometric processes in this
                        framework.  However, there are two caveats with this method: (1) the
                        resulting image will have Poisson noise from the finite number of photons,
                        and (2) it is not available for all object types (notably anything that
                        includes a Deconvolution).

            'no_pixel'  Instead of integrating over the pixels, this method will sample the profile
                        at the centers of the pixels and multiply by the pixel area.  If there is
                        a convolution involved, the choice of whether this will use an FFT or
                        real-space calculation is governed by the `real_space` parameter of the
                        Convolution class.  This method is the appropriate choice if you are using
                        a PSF that already includes a convolution by the pixel response.  For
                        example, if you are using a PSF from an observed image of a star, then it
                        has already been convolved by the pixel, so you would not want to do so
                        again.  Note: The multiplication by the pixel area gets the flux
                        normalization right for the above use case.  cf. `method = 'sb'`.

            'sb'        This is a lot like 'no_pixel', except that the image values will simply be
                        the sampled object profile's surface brightness, not multiplied by the
                        pixel area.  This does not correspond to any real observing scenario, but
                        it could be useful if you want to view the surface brightness profile of an
                        object directly, without including the pixel integration.

        Normally, the flux of the object should be equal to the sum of all the pixel values in the
        image, less some small amount of flux that may fall off the edge of the image (assuming you
        don't use `method='sb'`).  However, you may optionally set a `gain` value, which converts
        between photons and ADU (so-called analog-to-digital units), the units of the pixel values
        in real images.  Normally, the gain of a CCD is in electrons/ADU, but in GalSim, we fold
        the quantum efficiency into the gain as well, so the units are photons/ADU.  The flux of
        the object is then taken to be in photons, and the `gain` converts this to ADU.

        The 'phot' method has a few extra parameters that adjust how it functions.  The total
        number of photons to shoot is normally calculated from the object's flux.  This flux is
        taken to be given in photons, so for most simple profiles, this will equal the number of
        photons shot.  (See the discussion in Rowe et al, 2014, for why this might be modified for
        InterpolatedImage and related profiles.)  However, you can manually set a different number
        of photons with `n_photons`.  You can also set `max_extra_noise` to tell drawImage() to use
        fewer photons than normal (and so is faster) such that no more than that much extra noise
        is added to any pixel.  This is particularly useful if you will be subsequently adding sky
        noise, and you can thus tolerate more noise than the normal number of photons would give
        you, since using fewer photons is of course faster.  Finally, the default behavior is to
        have the total flux vary as a Poisson random variate, which is normally appropriate with
        photon shooting.  But you can turn this off with `poisson_flux=False`.  It also defaults to
        False if you set an explicit value for `n_photons`.

        The object will by default be drawn with its nominal center at the center location of the
        image.  There is thus a qualitative difference in the appearance of the rendered profile
        when drawn on even- and odd-sized images.  For a profile with a maximum at (0,0), this
        maximum will fall in the central pixel of an odd-sized image, but in the corner of the four
        central pixels of an even-sized image.  There are two parameters that can affect this
        behavior.  If you want the nominal center to always fall at the center of a pixel, you can
        use `use_true_center=False`.  This will put the object's center at the position
        `image.center()` which is an integer pixel value, and is not the true center of an
        even-sized image.  You can also arbitrarily offset the profile from the image center with
        the `offset` parameter to handle any sub-pixel dithering you want.

        On return, the image will have an attribute `added_flux`, which will be set to be the total
        flux added to the image.  This may be useful as a sanity check that you have provided a
        large enough image to catch most of the flux.  For example:

            >>> obj.drawImage(image)
            >>> assert image.added_flux > 0.99 * obj.getFlux()

        The appropriate threshold will depend on your particular application, including what kind
        of profile the object has, how big your image is relative to the size of your object,
        whether you are keeping `poisson_flux=True`, etc.

        Given the periodicity implicit in the use of FFTs, there can occasionally be artifacts due
        to wrapping at the edges, particularly for objects that are quite extended (e.g., due to
        the nature of the radial profile).  Use of the keyword parameter `wmult > 1` can be used to
        reduce the size of these artifacts (by making larger FFT images), at the expense of the
        calculations taking longer and using more memory.  Alternatively, the objects that go into
        the image can be created with a `gsparams` keyword that has a lower-than-default value for
        `folding_threshold`; see `help(galsim.GSParams)` for more information.

        @param image        If provided, this will be the image on which to draw the profile.
                            If `image` is None, then an automatically-sized Image will be created.
                            If `image` is given, but its bounds are undefined (e.g. if it was
                            constructed with `image = galsim.Image()`), then it will be resized
                            appropriately based on the profile's size [default: None].
        @param nx           If provided and `image` is None, use to set the x-direction size of the
                            image.  Must be accompanied by `ny`.
        @param ny           If provided and `image` is None, use to set the y-direction size of the
                            image.  Must be accompanied by `nx`.
        @param bounds       If provided and `image` is None, use to set the bounds of the image.
        @param scale        If provided, use this as the pixel scale for the image.
                            If `scale` is None and `image` is given, then take the provided
                            image's pixel scale.
                            If `scale` is None and `image` is None, then use the Nyquist scale.
                            If `scale <= 0` (regardless of `image`), then use the Nyquist scale.
                            If `scale > 0` and `image` is given, then override `image.scale` with
                            the value given as a keyword.
                            [default: None]
        @param wcs          If provided, use this as the wcs for the image (possibly overriding any
                            existing `image.wcs`).  At most one of `scale` or `wcs` may be provided.
                            [default: None]
        @param dtype        The data type to use for an automatically constructed image.  Only
                            valid if `image` is None. [default: None, which means to use
                            numpy.float32]
        @param method       Which method to use for rendering the image.  See discussion above
                            for the various options and what they do. [default: 'auto']
        @param gain         The number of photons per ADU ("analog to digital units", the units of
                            the numbers output from a CCD).  [default: 1]
        @param wmult        A multiplicative factor by which to enlarge (in each direction) the
                            default automatically calculated FFT grid size used for any
                            intermediate calculations in Fourier space.  The size of the
                            intermediate images is normally automatically chosen to reach some
                            preset accuracy targets [cf. GSParams]; however, if you see strange
                            artifacts in the image, you might try using `wmult > 1`.  This will
                            take longer of course, but it will produce more accurate images, since
                            they will have less "folding" in Fourier space. If the image size is
                            not specified, then the output real-space image will be enlarged by
                            a factor of `wmult`.  If the image size is specified by the user,
                            rather than automatically-sized, use of `wmult>1` will still affect the
                            size of the images used for the Fourier-space calculations and hence
                            can reduce image artifacts, even though the image that is returned will
                            be the requested size. [default: 1]
        @param add_to_image Whether to add flux to the existing image rather than clear out
                            anything in the image before drawing.
                            Note: This requires that `image` be provided and that it have defined
                            bounds. [default: False]
        @param use_true_center  Normally, the profile is drawn to be centered at the true center
                            of the image (using the function image.bounds.trueCenter()).
                            If you would rather use the integer center (given by
                            image.bounds.center()), set this to `False`.  [default: True]
        @param offset       The location at which to center the profile being drawn relative to the
                            center of the image (either the true center if `use_true_center=True`,
                            or the nominal center if `use_true_center=False`). [default: None]
        @param n_photons    If provided, the number of photons to use for photon shooting.
                            If not provided (i.e. `n_photons = 0`), use as many photons as
                            necessary to result in an image with the correct Poisson shot
                            noise for the object's flux.  For positive definite profiles, this
                            is equivalent to `n_photons = flux`.  However, some profiles need
                            more than this because some of the shot photons are negative
                            (usually due to interpolants).
                            [default: 0]
        @param rng          If provided, a random number generator to use for photon shooting,
                            which may be any kind of BaseDeviate object.  If `rng` is None, one
                            will be automatically created, using the time as a seed.
                            [default: None]
        @param max_extra_noise  If provided, the allowed extra noise in each pixel when photon
                            shooting.  This is only relevant if `n_photons=0`, so the number of
                            photons is being automatically calculated.  In that case, if the image
                            noise is dominated by the sky background, then you can get away with
                            using fewer shot photons than the full `n_photons = flux`.  Essentially
                            each shot photon can have a `flux > 1`, which increases the noise in
                            each pixel.  The `max_extra_noise` parameter specifies how much extra
                            noise per pixel is allowed because of this approximation.  A typical
                            value for this might be `max_extra_noise = sky_level / 100` where
                            `sky_level` is the flux per pixel due to the sky.  Note that this uses
                            a "variance" definition of noise, not a "sigma" definition.
                            [default: 0.]
        @param poisson_flux Whether to allow total object flux scaling to vary according to
                            Poisson statistics for `n_photons` samples when photon shooting.
                            [default: True, unless `n_photons` is given, in which case the default
                            is False]
        @param setup_only   Don't actually draw anything on the image.  Just make sure the image
                            is set up correctly.  This is used internally by GalSim, but there
                            may be cases where the user will want the same functionality.
                            [default: False]

        @returns the drawn Image.
        """
        # Check for obsolete dx parameter
        if dx is not None and scale is None: scale = dx

        # Check that image is sane
        if image is not None and not isinstance(image, galsim.Image):
            raise ValueError("image is not an Image instance")

        # Make sure the type of gain is correct and has a valid value:
        if type(gain) != float:
            gain = float(gain)
        if gain <= 0.:
            raise ValueError("Invalid gain <= 0.")

        # Make sure the type of wmult is correct and has a valid value
        if type(wmult) != float:
            wmult = float(wmult)
        if wmult <= 0:
            raise ValueError("Invalid wmult <= 0.")

        if method not in ['auto', 'fft', 'real_space', 'phot', 'no_pixel', 'sb']:
            raise ValueError("Invalid method name = %s"%method)

        # Some checks that are only relevant for method == 'phot'
        if method == 'phot':
            # Make sure the type of n_photons is correct and has a valid value:
            if type(n_photons) != float:
                n_photons = float(n_photons)
            if n_photons < 0.:
                raise ValueError("Invalid n_photons < 0.")

            if poisson_flux is None:
                if n_photons == 0.: poisson_flux = True
                else: poisson_flux = False

            # Make sure the type of max_extra_noise is correct and has a valid value:
            if type(max_extra_noise) != float:
                max_extra_noise = float(max_extra_noise)

            # Setup the uniform_deviate if not provided one.
            if rng is None:
                uniform_deviate = galsim.UniformDeviate()
            elif isinstance(rng,galsim.BaseDeviate):
                # If it's a BaseDeviate, we can convert to UniformDeviate
                uniform_deviate = galsim.UniformDeviate(rng)
            else:
                raise TypeError("The rng provided is not a BaseDeviate")

            # Check that either n_photons is set to something or flux is set to something
            if n_photons == 0. and self.getFlux() == 1.:
                import warnings
                msg = "Warning: drawImage for object with flux == 1, but n_photons == 0.\n"
                msg += "This will only shoot a single photon (since flux = 1)."
                warnings.warn(msg)
        else:
            if n_photons != 0.:
                raise ValueError("n_photons is only relevant for method='phot'")
            if rng is not None:
                raise ValueError("rng is only relevant for method='phot'")
            if max_extra_noise != 0.:
                raise ValueError("max_extra_noise is only relevant for method='phot'")
            if poisson_flux is not None:
                raise ValueError("poisson_flux is only relevant for method='phot'")

        # Check that the user isn't convolving by a Pixel already.  This is almost always an error.
        if method == 'auto' and isinstance(self, galsim.Convolution):
            if any([ isinstance(obj, galsim.Pixel) for obj in self.obj_list ]):
                import warnings
                warnings.warn(
                    "You called drawImage with no `method` parameter "
                    "for an object that includes convolution by a Pixel.  "
                    "This is probably an error.  Normally, you should let GalSim "
                    "handle the Pixel convolution for you.  If you want to handle the Pixel "
                    "convolution yourself, you can use method=no_pixel.  Or if you really meant "
                    "for your profile to include the Pixel and also have GalSim convolve by"
                    "an _additional_ Pixel, you can suppress this warning by using method=fft.")

        # Check for scale if using nx, ny, or bounds
        if (scale is None and wcs is None and
            (nx is not None or ny is not None or bounds is not None)):
            raise ValueError("Must provide scale if providing nx,ny or bounds")

        # Figure out what wcs we are going to use.
        wcs = self._determine_wcs(scale, wcs, image)

        # Make sure offset is a PositionD
        offset = self._parse_offset(offset)

        # Get the local WCS, accounting for the offset correctly.
        local_wcs = self._local_wcs(wcs, image, offset, use_true_center)

        # Convert the profile in world coordinates to the profile in image coordinates:
        prof = local_wcs.toImage(self)

        # If necessary, convolve by the pixel
        if method in ['auto', 'fft', 'real_space']:
            if method == 'auto':
                real_space = None
            elif method == 'fft':
                real_space = False
            else:
                real_space = True
            prof = galsim.Convolve(prof, galsim.Pixel(scale = 1.0), real_space=real_space)

        # Apply the offset, and possibly fix the centering for even-sized images
        prof = prof._fix_center(image, offset, use_true_center, reverse=False)

        # Make sure image is setup correctly
        image = prof._setup_image(image, nx, ny, bounds, wmult, add_to_image, dtype)
        image.wcs = wcs

        if setup_only:
            image.added_flux = 0.
            return image

        # For surface brightness normalization, scale gain by the pixel area.
        if method == 'sb':
            gain *= local_wcs.pixelArea()

        # Making a view of the image lets us change the center without messing up the original.
        imview = image.view()
        imview.setCenter(0,0)

        if method == 'phot':
            try:
                image.added_flux = prof.SBProfile.drawShoot(
                    imview.image, n_photons, uniform_deviate, gain, max_extra_noise,
                    poisson_flux, add_to_image)
            except RuntimeError:
                # Give some extra explanation as a warning, then raise the original exception
                # so the traceback shows as much detail as possible.
                import warnings
                warnings.warn(
                    "Unable to draw this GSObject with method='phot'.  Perhaps it is a "+
                    "Deconvolve or is a compound including one or more Deconvolve objects.")
                raise
        else:
            image.added_flux = prof.SBProfile.draw(imview.image, gain, wmult)

        return image

    def draw(self, *args, **kwargs):
        """An obsolete synonym for obj.drawImage(method='no_pixel')
        """
        normalization = kwargs.pop('normalization','f')
        if normalization in ['flux','f']:
            return self.drawImage(*args, method='no_pixel', **kwargs)
        else:
            return self.drawImage(*args, method='sb', **kwargs)

    def drawShoot(self, *args, **kwargs):
        """An obsolete synonym for obj.drawImage(method='phot')
        """
        normalization = kwargs.pop('normalization','f')
        if normalization in ['flux','f']:
            return self.drawImage(*args, method='phot', **kwargs)
        else:
            # We don't have a method for this, but I think it must be rare.  Photon shooting
            # with surface brightness normalization seems pretty odd.  We do use it in the test
            # suite a few times though.  So, need to reproduce a bit of code to get the
            # pixel area to switch to sb normalization (via the gain).
            if len(args) > 0:
                image = args[0]
            else:
                image = kwargs.get('image', None)
            scale = kwargs.get('scale', None)
            wcs = kwargs.get('wcs', None)
            offset = kwargs.get('offset', None)
            use_true_center = kwargs.get('use_true_center', None)
            wcs = self._determine_wcs(scale, wcs, image)
            offset = self._parse_offset(offset)
            local_wcs = self._local_wcs(wcs, image, offset, use_true_center)
            gain = kwargs.pop('gain',1.)
            gain *= local_wcs.pixelArea()
            return self.drawImage(*args, method='phot', gain=gain, **kwargs)

    def drawKImage(self, re=None, im=None, nx=None, ny=None, bounds=None, scale=None, dtype=None,
                   gain=1., wmult=1., add_to_image=False, dk=None):
        """Draws the k-space Image (both real and imaginary parts) of the object, with bounds
        optionally set by input Image instances.

        Normalization is always such that re(0,0) = flux.  Unlike the real-space drawImage()
        function, the (0,0) point will always be one of the actual pixel values.  For even-sized
        images, it will be 1/2 pixel above and to the right of the true center of the image.

        Another difference from  drawImage() is that a wcs other than a simple pixel scale is not
        allowed.  There is no `wcs` parameter here, and if the images have a non-trivial wcs (and
        you don't override it with the `scale` parameter), a TypeError will be raised.

        Also, there is no convolution by a pixel.  This is just a direct image of the Fourier
        transform of the surface brightness profile.

        @param re           If provided, this will be the real part of the k-space image.
                            If `re` and `im` are None, then automatically-sized images will be
                            created.  If they are given, but their bounds are undefined, then they
                            will be resized appropriately based on the profile's size.
                            [default: None]
        @param im           If provided, this will be the imaginary part of the k-space image.
                            A provided `im` must match the size and scale of `re`.
                            If `im` is None, then `re` must also be None. [default: None]
        @param scale        If provided, use this as the pixel scale, dk, for the images.
                            If `scale` is None and `re` and `im` are given, then take the provided
                            images' pixel scale (which must be equal).
                            If `scale` is None and `re` and `im` are None, then use the Nyquist
                            scale.
                            If `scale <= 0` (regardless of `re`, `im`), then use the Nyquist scale.
                            [default: None]
        @param dtype        The data type to use for automatically constructed images.  Only
                            valid if `re` and `im` are None. [default: None, which means to
                            use numpy.float32]
        @param gain         The number of photons per ADU ("analog to digital units", the units of
                            the numbers output from a CCD).  [default: 1.]
        @param wmult        A multiplicative factor by which to enlarge (in each direction) the
                            size of the image, if you are having drawKImage() automatically
                            construct the images for you.  [default: 1]
        @param add_to_image Whether to add to the existing images rather than clear out
                            anything in the image before drawing.
                            Note: This requires that `re` and `im` be provided and that they have
                            defined bounds. [default: False]

        @returns the tuple of Image instances, `(re, im)` (created if necessary)
        """
        # Check for obsolete dk parameter
        if dk is not None and scale is None: scale = dk

        # Make sure the type of gain is correct and has a valid value:
        if type(gain) != float:
            gain = float(gain)
        if gain <= 0.:
            raise ValueError("Invalid gain <= 0.")

        # Make sure the type of wmult is correct and has a valid value
        if type(wmult) != float:
            wmult = float(wmult)
        if wmult <= 0:
            raise ValueError("Invalid wmult <= 0.")

        # Check for scale if using nx, ny, or bounds
        if (scale is None and
            (nx is not None or ny is not None or bounds is not None)):
            raise ValueError("Must provide scale if providing nx,ny or bounds")

        # Check that the images are consistent, and possibly get the scale from them.
        if re is None:
            if im is not None:
                raise ValueError("re is None, but im is not None")
        else:
            if im is None:
                raise ValueError("im is None, but re is not None")
            if scale is None:
                # This check will raise a TypeError if re.wcs or im.wcs is not a PixelScale
                if re.scale != im.scale:
                    raise ValueError("re and im do not have the same input scale")
                # Grab the scale to use from the image.
                scale = re.scale
            if re.bounds.isDefined() or im.bounds.isDefined():
                if re.bounds != im.bounds:
                    raise ValueError("re and im do not have the same defined bounds")

        # The input scale (via scale or re.scale) is really a dk value, so call it that for
        # clarity here, since we also need the real-space pixel scale, which we will call dx.
        if scale is None or scale <= 0:
            dk = self.stepK()
        else:
            dk = float(scale)
        if re is not None and re.bounds.isDefined():
            dx = 2.*np.pi/( np.max(re.array.shape) * dk )
        else:
            dx = self.nyquistScale()

        # If the profile needs to be constructed from scratch, the _setup_image function will
        # do that, but only if the profile is in image coordinates for the real space image.
        # So make that profile.
        real_prof = galsim.PixelScale(dx).toImage(self)
        re = real_prof._setup_image(re, nx, ny, bounds, wmult, add_to_image, dtype)
        im = real_prof._setup_image(im, nx, ny, bounds, wmult, add_to_image, dtype)

        # Set the wcs of the images to use the dk scale size
        re.scale = dk
        im.scale = dk

        # Now, for drawing the k-space image, we need the profile to be in the image coordinates
        # that correspond to having unit-sized pixels in k space. The conversion to image
        # coordinates in this case is to apply the inverse dk pixel scale.
        prof = galsim.PixelScale(1./dk).toImage(self)

        # Making views of the images lets us change the centers without messing up the originals.
        review = re.view()
        review.setCenter(0,0)
        imview = im.view()
        imview.setCenter(0,0)

        prof.SBProfile.drawK(review.image, imview.image, gain, wmult)

        return re,im

    def drawK(self, *args, **kwargs):
        """An obsolete synonym for drawKImage()
        """
        return self.drawKImage(*args, **kwargs)


# --- Now defining the derived classes ---
#
# All derived classes inherit the GSObject method interface, but therefore have a "has a"
# relationship with the C++ SBProfile class rather than an "is a" one...
#
# The __init__ method is usually simple and all the GSObject methods & attributes are inherited.
#
class Gaussian(GSObject):
    """A class describing a 2D Gaussian surface brightness profile.

    The Gaussian surface brightness profile is characterized by two properties, its `flux`
    and the characteristic size `sigma` where the radial profile of the circular Gaussian
    drops off as `exp[-r^2 / (2. * sigma^2)]`.

    Initialization
    --------------

    A Gaussian can be initialized using one (and only one) of three possible size parameters:
    `sigma`, `fwhm`, or `half_light_radius`.  Exactly one of these three is required.

    @param sigma            The value of sigma of the profile.  Typically given in arcsec.
                            [One of `sigma`, `fwhm`, or `half_light_radius` is required.]
    @param fwhm             The full-width-half-max of the profile.  Typically given in arcsec.
                            [One of `sigma`, `fwhm`, or `half_light_radius` is required.]
    @param half_light_radius  The half-light radius of the profile.  Typically given in arcsec.
                            [One of `sigma`, `fwhm`, or `half_light_radius` is required.]
    @param flux             The flux (in photons) of the profile. [default: 1]
    @param gsparams         An optional GSParams argument.  See the docstring for GSParams for
                            details. [default: None]

    Methods
    -------

    In addition to the usual GSObject methods, Gaussian has the following access methods:

        >>> sigma = gauss.getSigma()
        >>> fwhm = gauss.getFWHM()
        >>> hlr = gauss.getHalfLightRadius()
    """

    # Initialization parameters of the object, with type information, to indicate
    # which attributes are allowed / required in a config file for this object.
    # _req_params are required
    # _opt_params are optional
    # _single_params are a list of sets for which exactly one in the list is required.
    # _takes_rng indicates whether the constructor should be given the current rng.
    # _takes_logger indicates whether the constructor takes a logger object for debug logging.
    _req_params = {}
    _opt_params = { "flux" : float }
    _single_params = [ { "sigma" : float, "half_light_radius" : float, "fwhm" : float } ]
    _takes_rng = False
    _takes_logger = False

    # --- Public Class methods ---
    def __init__(self, half_light_radius=None, sigma=None, fwhm=None, flux=1., gsparams=None):
        # Initialize the SBProfile
        GSObject.__init__(
            self, galsim._galsim.SBGaussian(
                sigma=sigma, half_light_radius=half_light_radius, fwhm=fwhm, flux=flux,
                gsparams=gsparams))

    def getSigma(self):
        """Return the sigma scale length for this Gaussian profile.
        """
        return self.SBProfile.getSigma()

    def getFWHM(self):
        """Return the FWHM for this Gaussian profile.
        """
        return self.SBProfile.getSigma() * 2.3548200450309493 # factor = 2 sqrt[2ln(2)]

    def getHalfLightRadius(self):
        """Return the half light radius for this Gaussian profile.
        """
        return self.SBProfile.getSigma() * 1.1774100225154747 # factor = sqrt[2ln(2)]


class Moffat(GSObject):
    """A class describing a Moffat surface brightness profile.

    The Moffat surface brightness profile is I(R) ~ [1 + (r/scale_radius)^2]^(-beta).  The
    GalSim representation of a Moffat profile also includes an optional truncation beyond a given
    radius.

    For more information, refer to

        http://home.fnal.gov/~neilsen/notebook/astroPSF/astroPSF.html

    Initialization
    --------------

    A Moffat can be initialized using one (and only one) of three possible size parameters:
    `scale_radius`, `fwhm`, or `half_light_radius`.  Exactly one of these three is required.

    @param beta             The `beta` parameter of the profile.
    @param scale_radius     The scale radius of the profile.  Typically given in arcsec.
                            [One of `scale_radius`, `fwhm`, or `half_light_radius` is required.]
    @param half_light_radius  The half-light radius of the profile.  Typically given in arcsec.
                            [One of `scale_radius`, `fwhm`, or `half_light_radius` is required.]
    @param fwhm             The full-width-half-max of the profile.  Typically given in arcsec.
                            [One of `scale_radius`, `fwhm`, or `half_light_radius` is required.]
    @param trunc            An optional truncation radius at which the profile is made to drop to
                            zero, in the same units as the size parameter.
                            [default: 0, indicating no truncation]
    @param flux             The flux (in photons) of the profile. [default: 1]
    @param gsparams         An optional GSParams argument.  See the docstring for GSParams for
                            details. [default: None]

    Methods
    -------

    In addition to the usual GSObject methods, Moffat has the following access methods:

        >>> beta = moffat_obj.getBeta()
        >>> rD = moffat_obj.getScaleRadius()
        >>> fwhm = moffat_obj.getFWHM()
        >>> hlr = moffat_obj.getHalfLightRadius()
    """

    # Initialization parameters of the object, with type information
    _req_params = { "beta" : float }
    _opt_params = { "trunc" : float , "flux" : float }
    _single_params = [ { "scale_radius" : float, "half_light_radius" : float, "fwhm" : float } ]
    _takes_rng = False
    _takes_logger = False

    # --- Public Class methods ---
    def __init__(self, beta, scale_radius=None, half_light_radius=None, fwhm=None, trunc=0.,
                 flux=1., gsparams=None):
        GSObject.__init__(
            self, galsim._galsim.SBMoffat(
                beta, scale_radius=scale_radius, half_light_radius=half_light_radius, fwhm=fwhm,
                trunc=trunc, flux=flux, gsparams=gsparams))

    def getBeta(self):
        """Return the beta parameter for this Moffat profile.
        """
        return self.SBProfile.getBeta()

    def getScaleRadius(self):
        """Return the scale radius for this Moffat profile.
        """
        return self.SBProfile.getScaleRadius()

    def getFWHM(self):
        """Return the FWHM for this Moffat profile.
        """
        return self.SBProfile.getFWHM()

    def getHalfLightRadius(self):
        """Return the half light radius for this Moffat profile.
        """
        return self.SBProfile.getHalfLightRadius()


class Airy(GSObject):
    """A class describing the surface brightness profile for an Airy disk (perfect
    diffraction-limited PSF for a circular aperture), with an optional central obscuration.

    For more information, refer to

        http://en.wikipedia.org/wiki/Airy_disc

    Initialization
    --------------

    The Airy profile is defined in terms of the diffraction angle, which is a function of the
    ratio lambda / D, where lambda is the wavelength of the light (say in the middle of the
    bandpass you are using) and D is the diameter of the telescope.  This ratio is the input
    parameter to pass to the Airy constructor, but as it is naturally in radians, you would
    typically convert to arcsec.  e.g.

        >>> lam = 700  # nm
        >>> diam = 4.0    # meters
        >>> lam_over_diam = (lambda * 1.e-9) / diam  # radians
        >>> lam_over_diam *= 206265  # Convert to arcsec

    @param lam_over_diam    The parameter that governs the scale size of the profile.
                            See above for details about calculating it.
    @param obscuration      The linear dimension of a central obscuration as a fraction of the
                            pupil dimension.  [default: 0]
    @param flux             The flux (in photons) of the profile. [default: 1]
    @param gsparams         An optional GSParams argument.  See the docstring for GSParams for
                            details. [default: None]

    Methods
    -------

    In addition to the usual GSObject methods, Airy has the following access methods:

        >>> lam_over_diam = airy_obj.getLamOverD()
        >>> fwhm = airy_obj.getFWHM()
        >>> hlr = airy_obj.getHalfLightRadius()

    The latter two are only available if the obscuration is 0.
    """

    # Initialization parameters of the object, with type information
    _req_params = { "lam_over_diam" : float }
    _opt_params = { "flux" : float , "obscuration" : float }
    _single_params = []
    _takes_rng = False
    _takes_logger = False

    # --- Public Class methods ---
    def __init__(self, lam_over_diam, obscuration=0., flux=1., gsparams=None):
        GSObject.__init__(
            self, galsim._galsim.SBAiry(lam_over_diam=lam_over_diam, obscuration=obscuration,
                                        flux=flux, gsparams=gsparams))

    def getHalfLightRadius(self):
        """Return the half light radius of this Airy profile (only supported for
        obscuration = 0.).
        """
        if self.SBProfile.getObscuration() == 0.:
            # For an unobscured Airy, we have the following factor which can be derived using the
            # integral result given in the Wikipedia page (http://en.wikipedia.org/wiki/Airy_disk),
            # solved for half total flux using the free online tool Wolfram Alpha.
            # At www.wolframalpha.com:
            # Type "Solve[BesselJ0(x)^2+BesselJ1(x)^2=1/2]" ... and divide the result by pi
            return self.SBProfile.getLamOverD() * 0.5348321477242647
        else:
            # In principle can find the half light radius as a function of lam_over_diam and
            # obscuration too, but it will be much more involved...!
            raise NotImplementedError("Half light radius calculation not implemented for Airy "+
                                      "objects with non-zero obscuration.")

    def getFWHM(self):
        """Return the FWHM of this Airy profile (only supported for obscuration = 0.).
        """
        # As above, likewise, FWHM only easy to define for unobscured Airy
        if self.SBProfile.getObscuration() == 0.:
            return self.SBProfile.getLamOverD() * 1.028993969962188;
        else:
            # In principle can find the FWHM as a function of lam_over_diam and obscuration too,
            # but it will be much more involved...!
            raise NotImplementedError("FWHM calculation not implemented for Airy "+
                                      "objects with non-zero obscuration.")

    def getLamOverD(self):
        """Return the `lam_over_diam` parameter of this Airy profile.
        """
        return self.SBProfile.getLamOverD()


class Kolmogorov(GSObject):
    """A class describing a Kolmogorov surface brightness profile, which represents a long
    exposure atmospheric PSF.

    For more information, refer to

        http://en.wikipedia.org/wiki/Atmospheric_seeing#The_Kolmogorov_model_of_turbulence

    Initialization
    --------------

    The Kolmogorov profile is normally defined in terms of the ratio lambda / r0, where lambda is
    the wavelength of the light (say in the middle of the bandpass you are using) and r0 is the
    Fried parameter.  Typical values for the Fried parameter are on the order of 10cm for
    most observatories and up to 20cm for excellent sites. The values are usually quoted at
    lambda = 500nm and r0 depends on wavelength as [r0 ~ lambda^(-6/5)].

    This ratio is naturally in radians, so you would typically convert to arcsec.  e.g.

        >>> lam = 700  # nm
        >>> r0 = 0.15 * (lam/500)**(-1.2)  # meters
        >>> lam_over_r0 = (lam * 1.e-9) / r0  # radians
        >>> lam_over_r0 *= 206265  # Convert to arcsec

    The FWHM of the Kolmogorov PSF is ~0.976 lambda/r0 (e.g., Racine 1996, PASP 699, 108).

    A Kolmogorov can be initialized using one (and only one) of three possible size parameters:
    `lam_over_r0`, `fwhm`, or `half_light_radius`.  Exactly one of these three is required.

    @param lam_over_r0      The parameter that governs the scale size of the profile.
                            See above for details about calculating it.  [One of `lam_over_r0`,
                            `fwhm`, or `half_light_radius` is required.]
    @param fwhm             The full-width-half-max of the profile.  Typically given in arcsec.
                            [One of `lam_over_r0`, `fwhm`, or `half_light_radius` is required.]
    @param half_light_radius  The half-light radius of the profile.  Typically given in arcsec.
                            [One of `lam_over_r0`, `fwhm`, or `half_light_radius` is required.]
    @param flux             The flux (in photons) of the profile. [default: 1]
    @param gsparams         An optional GSParams argument.  See the docstring for GSParams for
                            details. [default: None]

    Methods
    -------

    In addition to the usual GSObject methods, Kolmogorov has the following access methods:

        >>> lam_over_r0 = kolm.getLamOverR0()
        >>> fwhm = kolm.getFWHM()
        >>> hlr = kolm.getHalfLightRadius()
    """

    # The FWHM of the Kolmogorov PSF is ~0.976 lambda/r0 (e.g., Racine 1996, PASP 699, 108).
    # In SBKolmogorov.cpp we refine this factor to 0.975865
    _fwhm_factor = 0.975865
    # Similarly, SBKolmogorov calculates the relation between lambda/r0 and half-light radius
    _hlr_factor = 0.554811

    # Initialization parameters of the object, with type information
    _req_params = {}
    _opt_params = { "flux" : float }
    _single_params = [ { "lam_over_r0" : float, "fwhm" : float, "half_light_radius" : float } ]
    _takes_rng = False
    _takes_logger = False

    # --- Public Class methods ---
    def __init__(self, lam_over_r0=None, fwhm=None, half_light_radius=None, flux=1.,
                 gsparams=None):

        if fwhm is not None :
            if lam_over_r0 is not None or half_light_radius is not None:
                raise TypeError(
                        "Only one of lam_over_r0, fwhm, and half_light_radius may be " +
                        "specified for Kolmogorov")
            else:
                lam_over_r0 = fwhm / Kolmogorov._fwhm_factor
        elif half_light_radius is not None:
            if lam_over_r0 is not None:
                raise TypeError(
                        "Only one of lam_over_r0, fwhm, and half_light_radius may be " +
                        "specified for Kolmogorov")
            else:
                lam_over_r0 = half_light_radius / Kolmogorov._hlr_factor
        elif lam_over_r0 is None:
                raise TypeError(
                        "One of lam_over_r0, fwhm, or half_light_radius must be " +
                        "specified for Kolmogorov")

        GSObject.__init__(self, galsim._galsim.SBKolmogorov(lam_over_r0=lam_over_r0, flux=flux,
                                                            gsparams=gsparams))

    def getLamOverR0(self):
        """Return the `lam_over_r0` parameter of this Kolmogorov profile.
        """
        return self.SBProfile.getLamOverR0()

    def getFWHM(self):
        """Return the FWHM of this Kolmogorov profile.
        """
        return self.SBProfile.getLamOverR0() * Kolmogorov._fwhm_factor

    def getHalfLightRadius(self):
        """Return the half light radius of this Kolmogorov profile.
        """
        return self.SBProfile.getLamOverR0() * Kolmogorov._hlr_factor


class Pixel(GSObject):
    """A class describing a pixel profile.  This is just a 2D square top-hat function.

    This class is typically used to represent a pixel response function.  It is used internally by
    the drawImage() function, but there may be cases where the user would want to use this profile
    directly.

    Initialization
    --------------

    @param scale            The linear scale size of the pixel.  Typically given in arcsec.
    @param flux             The flux (in photons) of the profile.  This should almost certainly
                            be left at the default value of 1. [default: 1]
    @param gsparams         An optional GSParams argument.  See the docstring for GSParams for
                            details. [default: None]

    Methods
    -------

    In addition to the usual GSObject methods, Pixel has the following access method:

        >>> scale = pixel.getScale()

    Note: We have not implemented drawing a sheared or rotated Pixel in real space.  It's a
          bit tricky to get right at the edges where fractional fluxes are required.
          Fortunately, this is almost never needed.  Pixels are almost always convolved by
          something else rather than drawn by themselves, in which case either the fourier
          space method is used, or photon shooting.  Both of these are implemented in GalSim.
          If you need to draw sheared or rotated Pixels in real space, please file an issue, and
          maybe we'll implement that function.  Until then, you will get an exception if you try.
    """

    # Initialization parameters of the object, with type information
    _req_params = { "scale" : float }
    _opt_params = { "flux" : float }
    _single_params = []
    _takes_rng = False
    _takes_logger = False

    # --- Public Class methods ---
    def __init__(self, scale, flux=1., gsparams=None):
        GSObject.__init__(self, galsim._galsim.SBBox(scale, scale, flux=flux, gsparams=gsparams))

    def getScale(self):
        """Return the pixel scale.
        """
        return self.SBProfile.getWidth()


class Box(GSObject):
    """A class describing a box profile.  This is just a 2D top-hat function, where the
    width and height are allowed to be different.

    Initialization
    --------------

    @param width            The width of the Box.
    @param height           The height of the Box.
    @param flux             The flux (in photons) of the profile. [default: 1]
    @param gsparams         An optional GSParams argument.  See the docstring for GSParams for
                            details. [default: None]

    Methods
    -------

    In addition to the usual GSObject methods, Box has the following access methods:

        >>> width = box.getWidth()
        >>> height = box.getHeight()

    Note: We have not implemented drawing a sheared or rotated Box in real space.  It's a
          bit tricky to get right at the edges where fractional fluxes are required.
          Fortunately, this is almost never needed.  Box profiles are almost always convolved
          by something else rather than drawn by themselves, in which case either the fourier
          space method is used, or photon shooting.  Both of these are implemented in GalSim.
          If you need to draw sheared or rotated Boxes in real space, please file an issue, and
          maybe we'll implement that function.  Until then, you will get an exception if you try.
    """

    # Initialization parameters of the object, with type information
    _req_params = { "width" : float, "height" : float }
    _opt_params = { "flux" : float }
    _single_params = []
    _takes_rng = False
    _takes_logger = False

    # --- Public Class methods ---
    def __init__(self, width, height, flux=1., gsparams=None):
        width = float(width)
        height = float(height)
        GSObject.__init__(self, galsim._galsim.SBBox(width, height, flux=flux, gsparams=gsparams))

    def getWidth(self):
        """Return the width of the box in the x dimension.
        """
        return self.SBProfile.getWidth()

    def getHeight(self):
        """Return the height of the box in the y dimension.
        """
        return self.SBProfile.getHeight()


class Sersic(GSObject):
    """A class describing a Sersic profile.

    The Sersic surface brightness profile is characterized by three properties: its Sersic index
    `n`, its `flux`, and either the `half_light_radius` or `scale_radius`.  Given these properties,
    the surface brightness profile scales as I(r) ~ exp[-(r/scale_radius)^{1/n}], or
    I(r) ~ exp[-b*(r/half_light_radius)^{1/n}] (where b is calculated to give the right
    half-light radius).

    For more information, refer to

        http://en.wikipedia.org/wiki/Sersic_profile

    Initialization
    --------------

    The allowed range of values for the `n` parameter is 0.3 <= n <= 6.2.  An exception will be
    thrown if you provide a value outside that range.  Below n=0.3, there are severe numerical
    problems.  Above n=6.2, we found that the code begins to be inaccurate when sheared or
    magnified (at the level of upcoming shear surveys), so we do not recommend extending beyond
    this.  See Issues #325 and #450 for more details.

    Sersic profile calculations take advantage of Hankel transform tables that are precomputed for a
    given value of n when the Sersic profile is initialized.  Making additional objects with the
    same n can therefore be many times faster than making objects with different values of n that
    have not been used before.  Moreover, these Hankel transforms are only cached for a maximum of
    100 different n values at a time.  For this reason, for large sets of simulations, it is worth
    considering the use of only discrete n values rather than allowing it to vary continuously.  For
    more details, see https://github.com/GalSim-developers/GalSim/issues/566.

    Note that if you are building many Sersic profiles using truncation, the code will be more
    efficient if the truncation is always the same multiple of `scale_radius`, since it caches
    many calculations that depend on the ratio `trunc/scale_radius`.

    A Sersic can be initialized using one (and only one) of two possible size parameters:
    `scale_radius` or `half_light_radius`.  Exactly one of these two is required.

    @param n                The Sersic index, n.
    @param half_light_radius  The half-light radius of the profile.  Typically given in arcsec.
                            [One of `scale_radius` or `half_light_radius` is required.]
    @param scale_radius     The scale radius of the profile.  Typically given in arcsec.
                            [One of `scale_radius` or `half_light_radius` is required.]
    @param flux             The flux (in photons) of the profile. [default: 1]
    @param trunc            An optional truncation radius at which the profile is made to drop to
                            zero.  [default: 0, indicating no truncation]
    @param flux_untruncated Should the provided `flux` and `half_light_radius` refer to the
                            untruncated profile? See below for more details. [default: False]
    @param gsparams         An optional GSParams argument.  See the docstring for GSParams for
                            details. [default: None]

    Flux of a truncated profile
    ---------------------------

    If you are truncating the profile, the optional parameter, `flux_untruncated`, specifies
    whether the `flux` and `half_light_radius` specifications correspond to the untruncated
    profile (`True`) or to the truncated profile (`False`, default).  The impact of this parameter
    is a little subtle, so we'll go through a few examples to show how it works.

    First, let's examine the case where we specify the size according to the half-light radius.
    If `flux_untruncated` is True (and `trunc > 0`), then the profile will be identical
    to the version without truncation up to the truncation radius, beyond which it drops to 0.
    In this case, the actual half-light radius will be different from the specified half-light
    radius.  The getHalfLightRadius() method will return the true half-light radius.  Similarly,
    the actual flux will not be the same as the specified value; the true flux is also returned
    by the getFlux() method.

    Example:

        >>> sersic_obj1 = galsim.Sersic(n=3.5, half_light_radius=2.5, flux=40.)
        >>> sersic_obj2 = galsim.Sersic(n=3.5, half_light_radius=2.5, flux=40., trunc=10.)
        >>> sersic_obj3 = galsim.Sersic(n=3.5, half_light_radius=2.5, flux=40., trunc=10., \\
                                        flux_untruncated=True)

        >>> sersic_obj1.xValue(galsim.PositionD(0.,0.))
        237.3094228615618
        >>> sersic_obj2.xValue(galsim.PositionD(0.,0.))
        142.54505376530574    # Normalization and scale radius adjusted (same half-light radius)
        >>> sersic_obj3.xValue(galsim.PositionD(0.,0.))
        237.30942286156187

        >>> sersic_obj1.xValue(galsim.PositionD(10.0001,0.))
        0.011776164687304694
        >>> sersic_obj2.xValue(galsim.PositionD(10.0001,0.))
        0.0
        >>> sersic_obj3.xValue(galsim.PositionD(10.0001,0.))
        0.0

        >>> sersic_obj1.getHalfLightRadius()
        2.5
        >>> sersic_obj2.getHalfLightRadius()
        2.5
        >>> sersic_obj3.getHalfLightRadius()
        1.9795101383056892    # The true half-light radius is smaller than the specified value

        >>> sersic_obj1.getFlux()
        40.0
        >>> sersic_obj2.getFlux()
        40.0
        >>> sersic_obj3.getFlux()
        34.56595186009519     # Flux is missing due to truncation

        >>> sersic_obj1.getScaleRadius()
        0.003262738739834598
        >>> sersic_obj2.getScaleRadius()
        0.004754602453641744  # the scale radius needed adjustment to accommodate HLR
        >>> sersic_obj3.getScaleRadius()
        0.003262738739834598  # the scale radius is still identical to the untruncated case

    When the truncated Sersic scale is specified with `scale_radius`, the behavior between the
    three cases (untruncated, `flux_untruncated=True` and `flux_untruncated=False`) will be
    somewhat different from above.  Since it is the scale radius that is being specified, and since
    truncation does not change the scale radius the way it can change the half-light radius, the
    scale radius will remain unchanged in all cases.  This also results in the half-light radius
    being the same between the two truncated cases (although different from the untruncated case).
    The flux normalization is the only difference between `flux_untruncated=True` and
    `flux_untruncated=False` in this case.

    Example:

        >>> sersic_obj1 = galsim.Sersic(n=3.5, scale_radius=0.05, flux=40.)
        >>> sersic_obj2 = galsim.Sersic(n=3.5, scale_radius=0.05, flux=40., trunc=10.)
        >>> sersic_obj3 = galsim.Sersic(n=3.5, scale_radius=0.05, flux=40., trunc=10., \\
                                        flux_untruncated=True)

        >>> sersic_obj1.xValue(galsim.PositionD(0.,0.))
        1.010507575186637
        >>> sersic_obj2.xValue(galsim.PositionD(0.,0.))
        5.786692612210923     # Normalization adjusted to accomodate the flux within trunc radius
        >>> sersic_obj3.xValue(galsim.PositionD(0.,0.))
        1.010507575186637

        >>> sersic_obj1.getHalfLightRadius()
        38.311372735390016
        >>> sersic_obj2.getHalfLightRadius()
        5.160062547614234
        >>> sersic_obj3.getHalfLightRadius()
        5.160062547614234     # For the truncated cases, the half-light radii are the same

        >>> sersic_obj1.getFlux()
        40.0
        >>> sersic_obj2.getFlux()
        40.0
        >>> sersic_obj3.getFlux()
        6.985044085834393     # Flux is missing due to truncation

        >>> sersic_obj1.getScaleRadius()
        0.05
        >>> sersic_obj2.getScaleRadius()
        0.05
        >>> sersic_obj3.getScaleRadius()
        0.05

    Methods
    -------

    In addition to the usual GSObject methods, Sersic has the following access methods:

        >>> n = sersic_obj.getN()
        >>> r0 = sersic_obj.getScaleRadius()
        >>> hlr = sersic_obj.getHalfLightRadius()
    """

    # Initialization parameters of the object, with type information
    _req_params = { "n" : float }
    _opt_params = { "flux" : float, "trunc": float, "flux_untruncated" : bool }
    _single_params = [ { "scale_radius" : float , "half_light_radius" : float } ]
    _takes_rng = False
    _takes_logger = False

    # --- Public Class methods ---
    def __init__(self, n, half_light_radius=None, scale_radius=None,
                 flux=1., trunc=0., flux_untruncated=False, gsparams=None):
        GSObject.__init__(
            self, galsim._galsim.SBSersic(n, half_light_radius=half_light_radius,
                                          scale_radius=scale_radius, flux=flux,
                                          trunc=trunc, flux_untruncated=flux_untruncated,
                                          gsparams=gsparams))

    def getN(self):
        """Return the Sersic index `n` for this profile.
        """
        return self.SBProfile.getN()

    def getHalfLightRadius(self):
        """Return the half light radius for this Sersic profile.
        """
        return self.SBProfile.getHalfLightRadius()

    def getScaleRadius(self):
        """Return the scale radius for this Sersic profile.
        """
        return self.SBProfile.getScaleRadius()


class Exponential(GSObject):
    """A class describing an exponential profile.

    Surface brightness profile with I(r) ~ exp[-r/scale_radius].  This is a special case of
    the Sersic profile, but is given a separate class since the Fourier transform has closed form
    and can be generated without lookup tables.

    Initialization
    --------------

    An Exponential can be initialized using one (and only one) of two possible size parameters:
    `scale_radius` or `half_light_radius`.  Exactly one of these two is required.

    @param half_light_radius  The half-light radius of the profile.  Typically given in arcsec.
                            [One of `scale_radius` or `half_light_radius` is required.]
    @param scale_radius     The scale radius of the profile.  Typically given in arcsec.
                            [One of `scale_radius` or `half_light_radius` is required.]
    @param flux             The flux (in photons) of the profile. [default: 1]
    @param gsparams         An optional GSParams argument.  See the docstring for GSParams for
                            details. [default: None]

    Methods
    -------

    In addition to the usual GSObject methods, Exponential has the following access methods:

        >>> r0 = exp_obj.getScaleRadius()
        >>> hlr = exp_obj.getHalfLightRadius()
    """

    # Initialization parameters of the object, with type information
    _req_params = {}
    _opt_params = { "flux" : float }
    _single_params = [ { "scale_radius" : float , "half_light_radius" : float } ]
    _takes_rng = False
    _takes_logger = False

    # --- Public Class methods ---
    def __init__(self, half_light_radius=None, scale_radius=None, flux=1., gsparams=None):
        GSObject.__init__(
            self, galsim._galsim.SBExponential(
                half_light_radius=half_light_radius, scale_radius=scale_radius, flux=flux,
                gsparams=gsparams))

    def getScaleRadius(self):
        """Return the scale radius for this Exponential profile.
        """
        return self.SBProfile.getScaleRadius()

    def getHalfLightRadius(self):
        """Return the half light radius for this Exponential profile.
        """
        # Factor not analytic, but can be calculated by iterative solution of equation:
        #  (re / r0) = ln[(re / r0) + 1] + ln(2)
        return self.SBProfile.getScaleRadius() * 1.6783469900166605


class DeVaucouleurs(GSObject):
    """A class describing DeVaucouleurs profile objects.

    Surface brightness profile with I(r) ~ exp[-(r/scale_radius)^{1/4}].  This is completely
    equivalent to a Sersic with n=4.

    For more information, refer to

        http://en.wikipedia.org/wiki/De_Vaucouleurs'_law


    Initialization
    --------------

    A DeVaucouleurs can be initialized using one (and only one) of two possible size parameters:
    `scale_radius` or `half_light_radius`.  Exactly one of these two is required.

    @param scale_radius     The value of sigma of the profile.  Typically given in arcsec.
                            [One of `scale_radius` or `half_light_radius` is required.]
    @param half_light_radius  The half-light radius of the profile.  Typically given in arcsec.
                            [One of `scale_radius` or `half_light_radius` is required.]
    @param flux             The flux (in photons) of the profile. [default: 1]
    @param trunc            An optional truncation radius at which the profile is made to drop to
                            zero.  [default: 0, indicating no truncation]
    @param flux_untruncated Should the provided `flux` and `half_light_radius` refer to the
                            untruncated profile? See the docstring for Sersic for more details.
                            [default: False]
    @param gsparams         An optional GSParams argument.  See the docstring for GSParams for
                            details. [default: None]

    Methods
    -------

    In addition to the usual GSObject methods, DeVaucouleurs has the following access methods:

        >>> r0 = devauc_obj.getScaleRadius()
        >>> hlr = devauc_obj.getHalfLightRadius()
    """

    # Initialization parameters of the object, with type information
    _req_params = {}
    _opt_params = { "flux" : float, "trunc" : float, "flux_untruncated" : bool }
    _single_params = [ { "scale_radius" : float , "half_light_radius" : float } ]
    _takes_rng = False
    _takes_logger = False

    # --- Public Class methods ---
    def __init__(self, half_light_radius=None, scale_radius=None, flux=1., trunc=0.,
                 flux_untruncated=False, gsparams=None):
        GSObject.__init__(
            self, galsim._galsim.SBDeVaucouleurs(half_light_radius=half_light_radius,
                                                 scale_radius=scale_radius, flux=flux, trunc=trunc,
                                                 flux_untruncated=flux_untruncated,
                                                 gsparams=gsparams))

    def getHalfLightRadius(self):
        """Return the half light radius for this DeVaucouleurs profile.
        """
        return self.SBProfile.getHalfLightRadius()

    def getScaleRadius(self):
        """Return the scale radius for this DeVaucouleurs profile.
        """
        return self.SBProfile.getScaleRadius()


# GSParams is defined in C++ and wrapped.  But we want to modify it here slightly to add
# the obsolete name alias_threshold as a valid synonym for folding_threshold
GSParams.alias_threshold = property(lambda self: self.folding_threshold,
                                    lambda self, val: setattr(self,'folding_threshold',val))

# Also update the constructor to allow this name.
_orig_GSP_init = GSParams.__init__
def _new_GSP_init(self, *args, **kwargs):
    if 'alias_threshold' in kwargs:
        if 'folding_threshold' in kwargs:
            raise TypeError('Cannot specify both alias_threshold and folding_threshold')
        kwargs['folding_threshold'] = kwargs.pop('alias_threshold')
    _orig_GSP_init(self, *args, **kwargs)
GSParams.__init__ = _new_GSP_init


# Set the docstring for GSParams here.  It's easier to edit in the python layer than using
# the boost python doc parameter.
GSParams.__doc__ = """
GSParams stores a set of numbers that govern how GSObjects make various speed/accuracy tradeoff
decisions.  All GSObjects can take an optional parameter named `gsparams`, which would be an
instance of this class.  e.g.

    >>> gsp = galsim.GSParams(folding_threshold=1.e-3)
    >>> gal = galsim.Sersic(n=3.4, half_light_radius=3.2, flux=200, gsparams=gsp)

Note that `gsparams` needs to be provided when the object is initialized, rather than when the
object is drawn (as would perhaps be slightly more intuitive), because most of the relevant
approximations happen during the initialization of the object, rather than during the actual
rendering.

Initialization
--------------

All parameters have reasonable default values.  You only need to specify the ones you want
to change.

@param minimum_fft_size     The minimum size of any FFT that may need to be performed.
                            [default: 128]
@param maximum_fft_size     The maximum allowed size of an image for performing an FFT.  This
                            is more about memory use than accuracy.  We have this maximum
                            value to help prevent the user from accidentally trying to perform
                            an extremely large FFT that crashes the program. Instead, GalSim
                            will raise an exception indicating that the image is too large,
                            which is often a sign of an error in the user's code. However, if
                            you have the memory to handle it, you can raise this limit to
                            allow the calculation to happen. [default: 4096]
@param folding_threshold    This sets a maximum amount of real space folding that is allowed,
                            an effect caused by the periodic nature of FFTs.  FFTs implicitly
                            use periodic boundary conditions, and a profile specified on a
                            finite grid in Fourier space corresponds to a real space image
                            that will have some overlap with the neighboring copies of the real
                            space profile.  As the step size in k increases, the spacing between
                            neighboring aliases in real space decreases, increasing the amount of
                            folded, overlapping flux.  `folding_threshold` is used to set an
                            appropriate step size in k to allow at most this fraction of the flux
                            to be folded.
                            This parameter is also relevant when you let GalSim decide how large
                            an image to use for your object.  The image is made to be large enough
                            that at most a fraction `folding_threshold` of the total flux is
                            allowed to fall off the edge of the image. [default: 5.e-3]
@param stepk_minimum_hlr    In addition to the above constraint for aliasing, also set stepk
                            such that pi/stepk is at least `stepk_minimum_hlr` times the
                            profile's half-light radius (for profiles that have a well-defined
                            half-light radius). [default: 5]
@param maxk_threshold       This sets the maximum amplitude of the high frequency modes in
                            Fourier space that are excluded by truncating the FFT at some
                            maximum k value. Lowering this parameter can help minimize the
                            effect of "ringing" if you see that in your imags. [default: 1.e-3]
@param kvalue_accuracy      This sets the accuracy of values in Fourier space. Whenever there is
                            some kind of approximation to be made in the calculation of a
                            Fourier space value, the error in the approximation is constrained
                            to be no more than this value times the total flux. [default: 1.e-5]
@param xvalue_accuracy      This sets the accuracy of values in real space. Whenever there is
                            some kind of approximation to be made in the calculation of a
                            real space value, the error in the approximation is constrained
                            to be no more than this value times the total flux. [default: 1.e-5]
@param table_spacing        Several profiles use lookup tables for either the Hankel transform
                            (Sersic, Moffat) or the real space radial function (Kolmogorov).
                            We try to estimate a good spacing between values in the lookup
                            tables based on either `xvalue_accuracy` or `kvalue_accuracy` as
                            appropriate. However, you may change the spacing with this
                            parameter. Using `table_spacing < 1` will use a spacing value that
                            is that much smaller than the default, which should produce more
                            accurate interpolations. [default: 1]
@param realspace_relerr     This sets the relative error tolerance for real-space integration.
                            [default: 1.e-4]
@param realspace_abserr     This sets the absolute error tolerance for real-space integration.
                            [default: 1.e-6]
                            The estimated integration error for the flux value in each pixel
                            when using the real-space rendering method (either explicitly with
                            `method='real_space'` or if it is triggered automatically with
                            `method='auto'`) is constrained to be no larger than either
                            `realspace_relerr` times the pixel flux or `realspace_abserr`
                            times the object's total flux.
@param integration_relerr   The relative error tolerance for integrations other than real-space
                            rendering. [default: 1.e-6]
@param integration_abserr   The absolute error tolerance for integrations other than real-space
                            rendering. [default: 1.e-8]
@param shoot_accuracy       This sets the relative accuracy on the total flux when photon
                            shooting.  The photon shooting algorithm at times needs to make
                            approximations, such as how high in radius it needs to sample the
                            radial profile. When such approximations need to be made, it makes
                            sure that the resulting fractional error in the flux will be at
                            most this much. [default: 1.e-5]
allowed_flux_variation      The maximum range of allowed (abs value of) photon fluxes within
                            an interval before the rejection sampling algorithm is invoked for
                            photon shooting. [default: 0.81]
range_division_for_extrema  The number of parts into which to split a range to bracket extrema
                            when photon shooting. [default: 32]
small_fraction_of_flux      When photon shooting, intervals with less than this fraction of
                            probability are considered ok to use with the dominant-sampling
                            algorithm. [default: 1.e-4]
"""
