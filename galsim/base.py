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
"""@file base.py
Definitions for the GalSim base classes and associated methods

This file includes the key parts of the user interface to GalSim: base classes representing surface
brightness profiles for astronomical objects (galaxies, PSFs, pixel response).  These base classes
are collectively known as GSObjects.  They include both simple objects like the galsim.Gaussian, a
2d Gaussian intensity profile, and compound objects like the galsim.Add and galsim.Convolve, which
represent the sum and convolution of multiple GSObjects, respectively.

These classes also have associated methods to (a) retrieve information (like the flux, half-light
radius, or intensity at a particular point); (b) carry out common operations, like shearing,
rescaling of flux or size, rotating, and shifting; and (c) actually make images of the surface
brightness profiles.

For a description of units conventions for scale radii for our base classes, see
doc/GalSim_Quick_Reference.pdf section 2.2.  In short, any system that will ensure consistency
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
    derived class is appropriate for the surface brightness function you want:

        gal = galsim.Sersic(n=4, half_light_radius=4.3)
        psf = galsim.Moffat(beta=3, fwhm=2.85)
        pix = galsim.Pixel(scale=0.2)
        conv = galsim.Convolve([gal,psf,pix])

    All of these classes are subclasses of GSObject, so you should see those doc strings for
    more details about how to construct the various profiles.

    Note that most GSObjects have some kind of size specification.  Typically, these would be
    given in terms of arcsec, with the connection to the pixel size being given in the `Pixel`
    class (0.2 arcsec/pixel in the above example).  However, you can have a more complicated
    relationship between pixel and sky coordinates.  See `BaseWCS` for more details about
    how to specify various kinds of world coordinate systems.


    Transforming Methods
    --------------------

    The GSObject class uses an "immutable" design[1], so all methods that would potentially modify
    the object actually return a new object instead.  This uses pointers and such behind the
    scenes, so it all happens efficiently, but it makes using the objects a bit simpler, since
    you don't need to worry about some function changing your object behind your back.

    In all cases below, we just give an example usage.  See the docstrings for the methods for
    more details about how to use them.

        obj = obj.shear(shear)      # Apply a shear to the object.
        obj = obj.dilate(scale)     # Apply a flux-preserving dilation.
        obj = obj.magnify(mu)       # Apply a surface-brightness-preserving magnification.
        obj = obj.rotate(theta)     # Apply a rotation.
        obj = obj.shift(dx,dy)      # Shft the object in real space.
        obj = obj.transform(dudx,dudy,dvdx,dvdy)    # Apply a general jacobian transformation.
        obj = obj.lens(g1,g2,mu)    # Apply both a lensing shear and magnification.
        obj = obj.withFlux(flux)    # Set a new flux value.
        obj = obj * ratio           # Scale the surface brightness profile by some factor.

    [1]: Technically, there are some methods that do modify the object directly.  However, these
    methods are only present for backwards compatibitility with previous versions of GalSim,
    and they are currently discouraged.  They will be deprecated in a future version and
    removed entirely in version 2.0.

    Access Methods
    --------------

    There are some access methods that are available for all GSObjects.  Again, see the docstrings
    for each method for more details.

        flux = obj.getFlux()
        centroid = obj.centroid()
        f_xy = obj.xValue(x,y)
        fk_xy = obj.kValue(kx,ky)
        nyq = obj.nyquistScale()
        stepk = obj.stepK()
        maxk = obj.maxK()
        hard = obj.hasHardEdges()
        axisym = obj.isAxisymmetric()
        analytic = obj.isAnalyticX()

    Most subclasses have additional methods that are available for values that are particular to
    that specific surface brightness profile.  e.g. `sigma = gauss.getSigma()`.  However, note
    that class-specific methods are not available after performing one of the above transforming
    operations.

        gal = galsim.Gaussian(sigma=5)
        gal = gal.shear(g1=0.2, g2=0.05)
        sigma = gal.getSigma()              # This will raise an exception.

    It is however possible to access the original object that was transformed via the 
    `original` attribute.  

        sigma = gal.original.getSigma()     # This works.

    Drawing Methods
    ---------------

    The main thing to do with a GSObject once you have built it is to draw it onto an image.
    There are three methods that do this.  In all cases, there are lots of optional parameters.
    See the docstrings for these methods for more details.

        image = obj.draw()
        image = obj.drawShoot()
        kimage_r, kimage_i = obj.drawK()


    Attributes
    ----------

    There two attributes that may be available for a GSObject.

        original    This was mentioned above as a way to access the original object that has
                    been transformed by one of the transforming methods.

        noise       Some types, like RealGalaxy, set this attribute to be the intrinsic noise
                    that is already inherent in the profile and will thus be present when you
                    draw the object.  The noise is propagated correctly through the various
                    transforming methods, as well as convolutions and flux rescalings.
                    The typical use for this attribute is to use it to whiten the noise in
                    the image after drawing.  See CorrelatedNoise for more details.

    GSParams
    --------

    All GSObject classes take an optional `gsparams` argument so we document that feature here.
    For all documentation about the specific derived classes, please see the docstring for each
    one individually.

    The gsparams argument can be used to specify various numbers that govern the tradeoff between
    accuracy and speed for the calculations made in drawing a GSObject.  The numbers are
    encapsulated in a class called GSParams, and the user should make careful choices whenever they
    opt to deviate from the defaults.  For more details about the parameters and their default
    values, please see the docstring of the GSParams class (e.g. type `help(galsim.GSParams)`).

    For example, let's say you want to do something that requires an FFT larger than 4096 x 4096
    (and you have enough memory to handle it!).  Then you can create a new GSParams object with a
    larger maximum_fft_size and pass that to your GSObject on construction:

        >>> gal = galsim.Sersic(n=4, half_light_radius=4.3)
        >>> psf = galsim.Moffat(beta=3, fwhm=2.85)
        >>> pix = galsim.Pixel(scale=0.05)                  # Note the very small pixel scale!
        >>> conv = galsim.Convolve([gal,psf,pix])
        >>> im = galsim.Image(1000,1000, scale=0.05)        # Use the same pixel scale on the image.
        >>> im = conv.draw(image=im)                        # This uses the default GSParams.
        Traceback (most recent call last):
          File "<stdin>", line 1, in <module>
          File "galsim/base.py", line 885, in draw
            image.added_flux = prof.SBProfile.draw(image.image, gain, wmult)
        RuntimeError: SB Error: fourierDraw() requires an FFT that is too large, 6144
        If you can handle the large FFT, you may update gsparams.maximum_fft_size.
        >>> big_fft_params = galsim.GSParams(maximum_fft_size=10240)
        >>> conv = galsim.Convolve([gal,psf,pix],gsparams=big_fft_params)
        >>> im = conv.draw(image=im)                        # Now it works (but is slow!)
        >>> im.write('high_res_sersic.fits')

    Note that for compound objects like Convolve or Add, not all gsparams can be changed when the
    compound object is created.  In the example given here, it is possible to change parameters
    related to the drawing, but not the Fourier space parameters for the components that go into the
    Convolve.  To get better sampling in Fourier space, for example, the `gal`, `psf`, and/or `pix`
    should be created with `gsparams` that have a non-default value of `alias_threshold`.  This
    statement applies to the threshold and accuracy parameters.
    """
    _gsparams = { 'minimum_fft_size' : int,
                  'maximum_fft_size' : int,
                  'alias_threshold' : float,
                  'stepk_minimum_hlr' : float,
                  'maxk_threshold' : float,
                  'kvalue_accuracy' : float,
                  'xvalue_accuracy' : float,
                  'realspace_relerr' : float,
                  'realspace_abserr' : float,
                  'integration_relerr' : float,
                  'integration_abserr' : float,
                  'shoot_accuracy' : float,
                  'shoot_relerr' : float,
                  'shoot_abserr' : float,
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
        """Return profile at a given wavelength.  For `GSObject` instances, this is just `self`.
        This allows `GSObject` instances to be duck-typed as `ChromaticObject` instances."""
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
        acts like a wavelength-dependent flux_ratio.

        obj * sed is equivalend to galsim.Chromatic(obj, sed)
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

        Not all GSObject classes can use this method.  Classes like Convolve that require a
        Discrete Fourier Transform to determine the real space values will not do so for a
        single position.  Instead a RuntimeError will be raised.  The xValue(pos) method
        is available if and only if obj.isAnalyticX() == True.

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

        Techinically, kValue() is available if and only if the given obj has obj.isAnalyticK()
        == True, but this is the case for all GSObjects currently, so that should never be an
        issue (unlike for xValue).

        @param position  The position in k space at which you want the fourier amplitude.

        @returns the amplitude of the fourier transform at that position.
        """
        kpos = galsim.utilities.parse_pos_args(args,kwargs,'kx','ky')
        return self.SBProfile.kValue(kpos)

    def withFlux(self, flux):
        """Create a version of the current object with a different flux.

        This function is equivalent to obj.withScaledFlux(flux / obj.getFlux())

        It creates a new object that has the same profile as the original, but with the
        surface brightness at every location rescaled such that the total flux will be
        the given value.

        @param flux     The new flux for the object.

        @returns the object with the new flux
        """
        return self.withScaledFlux(flux / self.getFlux())

    def withScaledFlux(self, flux_ratio):
        """Create a version of the current object with the flux scaled by the given flux ratio.

        This function is equivalent to obj.withFlux(flux_ratio * obj.getFlux()).  However, this
        function is the more efficient one, since it doesn't actually require the call to
        obj.getFlux().  Indeed, withFlux() is implemented in terms of this one and getFlux().

        It creates a new object that has the same profile as the original, but with the
        surface brightness at every location scaled by the given amount.

        An equivalent, and usually simpler, way to effect this scaling is

            obj = obj * flux_ratio

        @param flux_ratio   The new flux for the object.

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
        """Expand the linear size of the profile by the given scale factor, while preserving
        surface brightness.

        e.g. `half_light_radius` <-- `half_light_radius * scale`

        This doesn't correspond to either of the normal operations one would typically want to
        do to a galaxy.  The functions dilate and magnify are the more typical usage.  But this
        function is conceptually simple.  It rescales the linear dimension of the profile, while
        preserving surface brightness.  As a result, the flux will necessarily change as well.

        See dilate for a version that applies a linear scale factor while preserving flux.

        See magnify for a version that applies a scale factor to the area while preserving surface
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
        """Dilate the linear size of the profile by the given scale factor, while preserving
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
        scaling the area and flux by mu at fixed surface brightness.

        This process applies a lensing magnification mu, which scales the linear dimensions of the
        image by the factor sqrt(mu), i.e., `half_light_radius` <-- `half_light_radius * sqrt(mu)`
        while increasing the flux by a factor of mu.  Thus, magnify preserves surface brightness.

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

        The arguments may be either a galsim.Shear or arguments to be used to initialize one.

        For more details about the allowed keyword arguments, see the documentation for galsim.Shear
        (for doxygen documentation, see galsim.shear.Shear).

        The shear() method precisely preserves the area.  To include a lensing distortion with
        the appropriate change in area, either use shear() with magnify(), or use lens(), which
        combines both operations.

        @param shear    The shear to be applied. Or, as described above, you may instead supply
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
        specified using the g1, g2 definition of shear (see galsim.Shear documentation for more
        details).  This is the same definition as the outputs of the galsim.PowerSpectrum and
        galsim.NFWHalo classes, which compute shears according to some lensing power spectrum or
        lensing by an NFW dark matter halo.  The magnification determines the rescaling factor for
        the object area and flux, preserving surface brightness.

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
        """Rotate this object by an Angle theta.

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
        """Create a version of the current object with an aribtrary Jacobian matrix transformation
        applied to it.

        This applies a Jacobian matrix to the coordinate system in which this object
        is defined.  It changes a profile defined in terms of (x,y) to one defined in
        terms of (u,v) where:

            u = dudx x + dudy y
            v = dvdx x + dvdy y

        That is, an arbitrary affine transform, but without the translation (which is
        easily effected via shift).

        Note that this function is similar to expand in that it preserves surface brightness,
        not flux.  If you want to preserve flux, you should also do

            prof *= 1./abs(dudx*dvdy - dudy*dvdx)

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
        or as a galsim.PositionD or galsim.PositionI object.

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


    # Make sure the image is defined with the right size and wcs for the draw and
    # drawShoot commands.
    def _draw_setup_image(self, image, wcs, wmult, add_to_image, scale_is_dk=False):

        # If image already exists, and its wcs is not a PixelScale, then we're all set.
        # No need to run through the rest of this.
        if image is not None and image.wcs is not None and not image.wcs.isPixelScale():
            # Clear the image if we are not adding to it.
            if not add_to_image:
                image.setZero()
            return image

        scale = None
        if wcs is not None:
            scale = wcs.maxLinearScale()

        # Save the input value, since we'll need to make a new scale (in case image is None)
        if scale_is_dk: dk = scale

        # Check scale value and adjust if necessary
        if scale is None:
            if image is not None and image.scale > 0.:
                if scale_is_dk:
                    # scale = 2pi / (N*dk)
                    dk = image.scale
                    scale = 2.*np.pi/( np.max(image.array.shape) * image.scale )
                else:
                    scale = image.scale
            else:
                scale = self.nyquistScale()
                if scale_is_dk:
                    dk = self.stepK()
        elif scale <= 0:
            scale = self.nyquistScale()
            wcs = None # Mark that the input wcs should not be used.
            if scale_is_dk:
                dk = self.stepK()
        elif scale_is_dk:
            dk = float(scale)
            if image is not None:
                scale = 2.*np.pi/( np.max(image.array.shape) * dk )
            else:
                scale = self.nyquistScale()
        elif type(scale) != float:
            scale = float(scale)
        # At this point scale is really scale, not dk.  So we can use it to determine
        # the "GoodImageSize".

        # Make image if necessary
        if image is None:
            # Can't add to image if none is provided.
            if add_to_image:
                raise ValueError("Cannot add_to_image if image is None")
            N = self.SBProfile.getGoodImageSize(scale,wmult)
            image = galsim.Image(N,N)

        # Resize the given image if necessary
        elif not image.bounds.isDefined():
            # Can't add to image if need to resize
            if add_to_image:
                raise ValueError("Cannot add_to_image if image bounds are not defined")
            N = self.SBProfile.getGoodImageSize(scale,wmult)
            bounds = galsim.BoundsI(1,N,1,N)
            image.resize(bounds)
            image.setZero()

        # Else use the given image as is
        else:
            # Clear the image if we are not adding to it.
            if not add_to_image:
                image.setZero()

        # Set the image wcs
        if scale_is_dk:
            image.scale = dk
        elif wcs is not None:
            image.wcs = wcs
        else:
            image.scale = scale

        return image

    def _obj_center(self, image, offset, use_true_center):
        # This just encapsulates this calculation that we do in a few places.
        if use_true_center:
            obj_cen = image.bounds.trueCenter()
        else:
            obj_cen = image.bounds.center()
            # Convert from PositionI to PositionD
            obj_cen = galsim.PositionD(obj_cen.x, obj_cen.y)
        if offset:
            obj_cen += offset
        return obj_cen

    def _parse_offset(self, offset):
        if offset is None:
            return galsim.PositionD(0,0)
        else:
            if isinstance(offset, galsim.PositionD) or isinstance(offset, galsim.PositionI):
                return galsim.PositionD(offset.x, offset.y)
            else:
                # Let python raise the appropriate exception if this isn't valid.
                return galsim.PositionD(offset[0], offset[1])


    def _fix_center(self, image, wcs, offset, use_true_center, reverse):
        # This is a touch circular since we may not know the image shape or scale yet.
        # So we need to repeat a little bit of what will be done again in _draw_setup_image
        #
        # - If the image is None, then it will be built with even sizes, and all fix_center
        #   cares about is the odd/even-ness of the shape, so juse use (0,0).
        # - If wcs is None, first try to get it from the image if given.
        # - If wcs is None, and image is None (or im.scale <= 0), then use the nyquist scale.
        #   Here is the really circular one, since the nyquist scale depends on stepK, which
        #   changes when we offset, but we need scale to know how much to offset.  So just
        #   use the current nyquist scale and hope the offset isn't too large, so it won't
        #   make that large a difference.

        if image is None or not image.bounds.isDefined():
            shape = (0,0)
        else:
            shape = image.array.shape

        if wcs is None:
            if image is not None and image.wcs is not None:
                if image.wcs.isPixelScale():
                    if image.scale <= 0:
                        wcs = galsim.PixelScale(self.nyquistScale())
                    else:
                        wcs = image.wcs.local()
                else:
                    # Not a PixelScale.  Just make sure we are using a local wcs
                    obj_cen = self._obj_center(image, offset, use_true_center)
                    wcs = image.wcs.local(image_pos=obj_cen)
            else:
                wcs = galsim.PixelScale(self.nyquistScale())
        elif wcs.isPixelScale() and wcs.scale <= 0:
            wcs = galsim.PixelScale(self.nyquistScale())
        else:
            # Should have already checked that wcs is uniform, so local() without an
            # image_pos argument should be ok.
            wcs = wcs.local()

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
            return self.shift(wcs.toWorld(offset))

    def _check_wcs(self, scale, wcs, image):
        # Get the correct wcs given the input scale, wcs and image.
        if wcs is not None:
            if scale is not None:
                raise ValueError("Cannot provide both wcs and scale")
            if not wcs.isUniform():
                if image is None:
                    raise ValueError("Cannot provide non-local wcs when image == None")
                if not image.bounds.isDefined():
                    raise ValueError("Cannot provide non-local wcs when image has undefined bounds")
            if not isinstance(wcs, galsim.BaseWCS):
                raise TypeError("wcs must be a BaseWCS instance")
            return wcs
        elif scale is not None:
            return galsim.PixelScale(scale)
        else:
            # else leave wcs = None
            return None

    def draw(self, image=None, scale=None, wcs=None, gain=1., wmult=1., normalization="flux",
             add_to_image=False, use_true_center=True, offset=None, dx=None):
        """Draws an Image of the object, with bounds optionally set by an input Image.

        The draw method is used to draw an Image of the GSObject, typically using Fourier space
        convolution (or, for certain GSObjects that have hard edges, real-space convolution may be
        used), and using interpolation to carry out image transformations such as shearing.  This
        method can create a new Image or can draw into an existing one, depending on the choice of
        the `image` keyword parameter.  Other keywords of particular relevance for users are those
        that set the pixel scale or wcs for the image (`scale`, `wcs`), that choose the
        normalization convention for the flux (`normalization`), and that decide whether to clear
        the input Image before drawing into it (`add_to_image`).

        The object will always be drawn with its nominal center at the center location of the
        image.  There is thus a distinction in the behavior at the center for even- and odd-sized
        images.  For a profile with a maximum at (0,0), this maximum will fall at the central
        pixel of an odd-sized image, but in the corner of the 4 central pixels of an even-sized
        image.  If you care about how the sub-pixel offsets are drawn, you should either make
        sure you provide an image with the right kind of size, or shift the profile by half
        a pixel as desired to get the profile's (0,0) location where you want it.

        Note that when drawing a GSObject that was defined with a particular value of flux, it is
        not necessarily the case that a drawn image with 'normalization=flux' will have the sum of
        pixel values equal to flux.  That condition is guaranteed to be satisfied only if the
        profile has been convolved with a pixel response. If there was no convolution by a pixel
        response, then the draw method is effectively sampling the surface brightness profile of the
        GSObject at pixel centers without integrating over the flux within pixels, so for profiles
        that are poorly sampled and/or varying rapidly (e.g., high n Sersic profiles), the sum of
        pixel values might differ significantly from the GSObject flux.

        On return, the image will have a member `added_flux`, which will be set to be the total
        flux added to the image.  This may be useful as a sanity check that you have provided a
        large enough image to catch most of the flux.  For example:

            obj.draw(image)
            assert image.added_flux > 0.99 * obj.getFlux()

        The appropriate threshold will depend on your particular application, including what kind
        of profile the object has, how big your image is relative to the size of your object, etc.

        Given the periodicity implicitly assumed by use of FFTs, there can occasionally be artifacts
        due to wrapping at the edges, particularly for objects that are quite extended (e.g., due to
        the nature of the radial profile).  Use of the keyword parameter `wmult > 1` can be used to
        reduce the size of these artifacts, at the expense of the calculations taking longer and
        using more memory.  Alternatively, the objects that go into the image can be created with a
        `gsparams` keyword that has a lower-than-default value for `alias_threshold`; see
        help(galsim.GSParams) for more information.

        @param image        If provided, this will be the image on which to draw the profile.
                            If `image = None`, then an automatically-sized image will be created.
                            If `image != None`, but its bounds are undefined (e.g. if it was
                            constructed with `image = galsim.Image()`), then it will be resized
                            appropriately based on the profile's size [default: None].
        @param scale        If provided, use this as the pixel scale for the image.
                            If `scale` is `None` and `image != None`, then take the provided
                            image's pixel scale.
                            If `scale` is `None` and `image == None`, then use the Nyquist scale.
                            If `scale <= 0` (regardless of image), then use the Nyquist scale.
                            [default: None]
        @param wcs          If provided, use this as the wcs for the image.  At most one of scale
                            or wcs may be provided. [default: None]
        @param gain         The number of photons per ADU ("analog to digital units", the units of
                            the numbers output from a CCD).  [default: 1]
        @param wmult        A multiplicative factor by which to enlarge (in each direction) the
                            default automatically calculated FFT grid size used for any
                            intermediate calculations in Fourier space.  The size of the
                            intermediate images is normally automatically chosen to reach some
                            preset accuracy targets [cf. galsim.GSParams()]; however, if you see
                            strange artifacts in the image, you might try using `wmult > 1`.  This
                            will take longer of course, but it will produce more accurate images,
                            since they will have less "folding" in Fourier space. If the image size
                            is not specified, then the output real-space image will be enlarged by
                            a factor of `wmult`.  If the image size is specified by the user,
                            rather than automatically-sized, use of `wmult>1` will still affect the
                            size of the images used for the Fourier-space calculations and hence
                            can reduce image artifacts, even though the image that is returned will
                            be the size that was requested. [default: 1]
        @param normalization  Two options for the normalization:
                            "flux" or "f" means that the sum of the output pixels is normalized
                               to be equal to the total flux.  (Modulo any flux that falls off
                               the edge of the image of course, and note the caveat in the draw
                               method documentation regarding the need to convolve with a pixel
                               response.)
                            "surface brightness" or "sb" means that the output pixels sample
                               the surface brightness distribution at each location.
                            [default: "flux"]
        @param add_to_image Whether to add flux to the existing image rather than clear out
                            anything in the image before drawing.
                            Note: This requires that image be provided and that it have defined
                            bounds. [default: False]
        @param use_true_center  Normally, the profile is drawn to be centered at the true center
                            of the image (using the function `image.bounds.trueCenter()`).
                            If you would rather use the integer center (given by
                            `image.bounds.center()`), set this to `False`.  [default: True]
        @param offset       The location at which to center the profile being drawn relative to the
                            center of the image (either the true center if use_true_center=True,
                            or the nominal center if use_true_center=False). [default: None]

        @returns the drawn image.
        """
        # Check for obsolete dx parameter
        if dx is not None and scale is None: scale = dx

        # Raise an exception immediately if the normalization type is not recognized
        if not normalization.lower() in ("flux", "f", "surface brightness", "sb"):
            raise ValueError(("Invalid normalization requested: '%s'. Expecting one of 'flux', "+
                              "'f', 'surface brightness' or 'sb'.") % normalization)

        # Make sure the type of gain is correct and has a valid value:
        if type(gain) != float:
            gain = float(gain)
        if gain <= 0.:
            raise ValueError("Invalid gain <= 0. in draw command")

        # Make sure the type of wmult is correct and has a valid value
        if type(wmult) != float:
            wmult = float(wmult)
        if wmult <= 0:
            raise ValueError("Invalid wmult <= 0 in draw command")

        # Check for non-trivial wcs
        wcs = self._check_wcs(scale, wcs, image)

        # Make sure offset is a PositionD
        offset = self._parse_offset(offset)

        # Apply the offset, and possibly fix the centering for even-sized images
        # Note: We need to do this before we call _draw_setup_image, since the shift
        # affects stepK (especially if the offset is rather large).
        prof = self._fix_center(image, wcs, offset, use_true_center, reverse=False)

        # Make sure image is setup correctly
        image = prof._draw_setup_image(image, wcs, wmult, add_to_image)

        # Figure out the position of the center of the object
        obj_cen = self._obj_center(image, offset, use_true_center)

        # Surface brightness normalization requires scaling the flux value of each pixel
        # by the area of the pixel.  We do this by changing the gain.
        if normalization.lower() in ['surface brightness','sb']:
            gain *= image.wcs.pixelArea(image_pos=obj_cen)

        # Convert the profile in world coordinates to the profile in image coordinates:
        prof = image.wcs.toImage(prof, image_pos=obj_cen)

        imview = image.view()
        imview.setCenter(0,0)
        image.added_flux = prof.SBProfile.draw(imview.image, gain, wmult)

        return image

    def drawShoot(self, image=None, scale=None, wcs=None, gain=1., wmult=1., normalization="flux",
                  add_to_image=False, use_true_center=True, offset=None,
                  n_photons=0., rng=None, max_extra_noise=0., poisson_flux=None, dx=None):
        """Draw an image of the object by shooting individual photons drawn from the surface
        brightness profile of the object.

        The drawShoot() method is used to draw an image of an object by shooting a number of photons
        to randomly sample the profile of the object. The resulting image will thus have Poisson
        noise due to the finite number of photons shot.  drawShoot() can create a new Image or use
        an existing one, depending on the choice of the `image` keyword parameter.  Other keywords
        of particular relevance for users are those that set the pixel scale or wcs for the image
        (`scale`, `wcs`), that choose the normalization convention for the flux (`normalization`),
        and that decide whether the clear the input Image before shooting photons into it
        (`add_to_image`).

        As for the draw command, the object will always be drawn with its nominal center at the
        center location of the image.  See the documentation for draw for more discussion about
        the implications of this for even- and odd-sized images.

        It is important to remember that the image produced by drawShoot() represents the object as
        convolved with the square image pixel.  So when using drawShoot() instead of draw(), you
        should not explicitly include the pixel response by convolving with a Pixel GSObject.  Using
        drawShoot without convolving with a Pixel will produce the equivalent image (for very large
        n_photons) as draw() produces when the same object is convolved with `Pixel(scale=scale)`
        when drawing onto an image with pixel scale `scale`.

        Note that the drawShoot method is unavailable for Deconvolve objects or compound objects
        (e.g. Add, Convolve) that include a Deconvolve.

        On return, the image will have a member `added_flux`, which will be set to be the total
        flux of photons that landed inside the image bounds.  This may be useful as a sanity check
        that you have provided a large enough image to catch most of the flux.  For example:

            obj.drawShoot(image)
            assert image.added_flux > 0.99 * obj.getFlux()

        The appropriate threshold will depend on your particular application, including what kind
        of profile the object has, how big your image is relative to the size of your object,
        whether you are keeping `poisson_flux = True`, etc.

        @param image        If provided, this will be the image on which to draw the profile.
                            If `image = None`, then an automatically-sized image will be created.
                            If `image != None`, but its bounds are undefined (e.g. if it was
                            constructed with `image = galsim.Image()`), then it will be resized
                            appropriately based on the profile's size [default: None].
        @param scale        If provided, use this as the pixel scale for the image.
                            If `scale` is `None` and `image != None`, then take the provided
                            image's pixel scale.
                            If `scale` is `None` and `image == None`, then use the Nyquist scale.
                            If `scale <= 0` (regardless of image), then use the Nyquist scale.
                            [default: None]
        @param wcs          If provided, use this as the wcs for the image.  At most one of scale
                            or wcs may be provided. [default: None]
        @param gain         The number of photons per ADU ("analog to digital units", the units of
                            the numbers output from a CCD).  [default: 1]
        @param wmult        A factor by which to make an automatically-sized image larger than
                            it would normally be made. [default: 1]
        @param normalization  Two options for the normalization:
                            "flux" or "f" means that the sum of the output pixels is normalized
                               to be equal to the total flux.  (Modulo any flux that falls off
                               the edge of the image of course, and note the caveat in the draw
                               method documentation regarding the need to convolve with a pixel
                               response.)
                            "surface brightness" or "sb" means that the output pixels sample
                               the surface brightness distribution at each location.
                            [default: "flux"]
        @param add_to_image Whether to add flux to the existing image rather than clear out
                            anything in the image before drawing.
                            Note: This requires that image be provided and that it have defined
                            bounds. [default: False]
        @param use_true_center  Normally, the profile is drawn to be centered at the true center
                            of the image (using the function `image.bounds.trueCenter()`).
                            If you would rather use the integer center (given by
                            `image.bounds.center()`), set this to `False`.  [default: True]
        @param offset       The location at which to center the profile being drawn relative to the
                            center of the image (either the true center if use_true_center=True,
                            or the nominal center if use_true_center=False). [default: None]
        @param n_photons    If provided, the number of photons to use.
                            If not provided (i.e. `n_photons = 0`), use as many photons as
                            necessary to result in an image with the correct Poisson shot
                            noise for the object's flux.  For positive definite profiles, this
                            is equivalent to `n_photons = flux`.  However, some profiles need
                            more than this because some of the shot photons are negative
                            (usually due to interpolants).
                            [default: 0]
        @param rng          If provided, a random number generator to use for photon shooting.
                            (may be any kind of `galsim.BaseDeviate` object)
                            If `rng=None`, one will be automatically created, using the time
                            as a seed.  [default: None]
        @param max_extra_noise  If provided, the allowed extra noise in each pixel.
                            This is only relevant if `n_photons=0`, so the number of photons
                            is being automatically calculated.  In that case, if the image
                            noise is dominated by the sky background, you can get away with
                            using fewer shot photons than the full `n_photons = flux`.
                            Essentially each shot photon can have a `flux > 1`, which
                            increases the noise in each pixel.  The `max_extra_noise`
                            parameter specifies how much extra noise per pixel is allowed
                            because of this approximation.  A typical value for this might be
                            `max_extra_noise = sky_level / 100` where `sky_level` is the flux
                            per pixel due to the sky.  If the natural number of photons
                            produces less noise than this value for all pixels, we lower the
                            number of photons to bring the resultant noise up to this value.
                            If the natural value produces more noise than this, we accept it
                            and just use the natural value.  Note that this uses a "variance"
                            definition of noise, not a "sigma" definition.  [default: 0.]
        @param poisson_flux Whether to allow total object flux scaling to vary according to
                            Poisson statistics for `n_photons` samples. [default: True,
                            unless n_photons is given, in which case the default is False]

        @returns the drawn image.
        """
        # Check for obsolete dx parameter
        if dx is not None and scale is None: scale = dx

        # Raise an exception immediately if the normalization type is not recognized
        if not normalization.lower() in ("flux", "f", "surface brightness", "sb"):
            raise ValueError(("Invalid normalization requested: '%s'. Expecting one of 'flux', "+
                              "'f', 'surface brightness' or 'sb'.") % normalization)

        # Make sure the type of gain is correct and has a valid value:
        if type(gain) != float:
            gain = float(gain)
        if gain <= 0.:
            raise ValueError("Invalid gain <= 0. in draw command")

        # Make sure the type of wmult is correct and has a valid value
        if type(wmult) != float:
            wmult = float(wmult)
        if wmult <= 0:
            raise ValueError("Invalid wmult <= 0 in draw command")

        # Make sure the type of n_photons is correct and has a valid value:
        if type(n_photons) != float:
            n_photons = float(n_photons)
        if n_photons < 0.:
            raise ValueError("Invalid n_photons < 0. in draw command")
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
            raise TypeError("The rng provided to drawShoot is not a BaseDeviate")

        # Check that either n_photons is set to something or flux is set to something
        if n_photons == 0. and self.getFlux() == 1.:
            import warnings
            msg = "Warning: drawShoot for object with flux == 1, but n_photons == 0.\n"
            msg += "This will only shoot a single photon (since flux = 1)."
            warnings.warn(msg)

        # Check for non-trivial wcs
        wcs = self._check_wcs(scale, wcs, image)

        # Make sure offset is a PositionD
        offset = self._parse_offset(offset)

        # Apply the offset, and possibly fix the centering for even-sized images
        prof = self._fix_center(image, wcs, offset, use_true_center, reverse=False)

        # Make sure image is setup correctly
        image = prof._draw_setup_image(image, wcs, wmult, add_to_image)

        obj_cen = self._obj_center(image, offset, use_true_center)

        # Surface brightness normalization requires scaling the flux value of each pixel
        # by the area of the pixel.  We do this by changing the gain.
        if normalization.lower() in ['surface brightness','sb']:
            gain *= image.wcs.pixelArea(image_pos=obj_cen)

        # Convert the profile in world coordinates to the profile in chip coordinates:
        prof = image.wcs.toImage(prof, image_pos=obj_cen)

        imview = image.view()
        imview.setCenter(0,0)

        try:
            image.added_flux = prof.SBProfile.drawShoot(
                imview.image, n_photons, uniform_deviate, gain, max_extra_noise,
                poisson_flux, add_to_image)
        except RuntimeError:
            # Give some extra explanation as a warning, then raise the original exception
            # so the traceback shows as much detail as possible.
            import warnings
            warnings.warn(
                "Unable to drawShoot from this GSObject, perhaps it is a Deconvolve "+
                "or is a compound including one or more Deconvolve objects.")
            raise

        return image

    def drawK(self, re=None, im=None, scale=None, gain=1., add_to_image=False, dk=None):
        """Draws the k-space Images (real and imaginary parts) of the object, with bounds
        optionally set by input Images.

        Normalization is always such that re(0,0) = flux.  Unlike the real-space draw and
        drawShoot functions, the (0,0) point will always be one of the actual pixel values.
        For even-sized images, it will be 1/2 pixel above and to the right of the true
        center of the image.

        Unlike for the draw and drawShoot commands, a wcs other than a simple pixel scale
        is not allowed.  There is no wcs parameter here, and if the images have a non-trivial
        wcs (and you don't override it with the scale parameter), a TypeError will be raised.

        @param re           If provided, this will be the real part of the k-space image.
                            If `re = None`, then an automatically-sized image will be created.
                            If `re != None`, but its bounds are undefined (e.g. if it was
                            constructed with `re = galsim.Image()`), then it will be resized
                            appropriately based on the profile's size. [default: None]
        @param im           If provided, this will be the imaginary part of the k-space image.
                            A provided im must match the size and scale of re.
                            If `im = None`, then an automatically-sized image will be created.
                            If `im != None`, but its bounds are undefined (e.g. if it was
                            constructed with `im = galsim.Image()`), then it will be resized
                            appropriately based on the profile's size. [default: None]
        @param scale        If provided, use this as the pixel scale for the images.
                            If `scale` is `None` and `re, im != None`, then take the provided
                            images' pixel scale (which must be equal).
                            If `scale` is `None` and `re, im == None`, then use the Nyquist scale.
                            If `scale <= 0` (regardless of image), then use the Nyquist scale.
                            [default: None]
        @param gain         The number of photons per ADU ("analog to digital units", the units of
                            the numbers output from a CCD).  [default: 1.]
        @param add_to_image Whether to add to the existing images rather than clear out
                            anything in the image before drawing.
                            Note: This requires that images be provided (i.e. `re`, `im` are
                            not `None`) and that they have defined bounds. [default: False]

        @returns the tuple of images, (re, im) (created if necessary)
        """
        # Check for obsolete dk parameter
        if dk is not None and scale is None: scale = dk

        # Make sure the type of gain is correct and has a valid value:
        if type(gain) != float:
            gain = float(gain)
        if gain <= 0.:
            raise ValueError("Invalid gain <= 0. in draw command")
        if re is None:
            if im != None:
                raise ValueError("re is None, but im is not None")
        else:
            if im is None:
                raise ValueError("im is None, but re is not None")
            if scale is None:
                # This check will raise a TypeError if re.wcs or im.wcs is not a PixelScale
                if re.scale != im.scale:
                    raise ValueError("re and im do not have the same input scale")
            if re.bounds.isDefined() or im.bounds.isDefined():
                if re.bounds != im.bounds:
                    raise ValueError("re and im do not have the same defined bounds")

        # Make sure images are setup correctly
        wcs = galsim.PixelScale(scale)
        re = self._draw_setup_image(re,wcs,1.0,add_to_image,scale_is_dk=True)
        im = self._draw_setup_image(im,wcs,1.0,add_to_image,scale_is_dk=True)

        # Convert the profile in world coordinates to the profile in image coordinates:
        # The scale in the wcs objects is the dk scale, not dx.  So the conversion to
        # image coordinates needs to apply the inverse pixel scale.
        # The following are all equivalent ways to do this:
        #    re.wcs.toWorld(prof)
        #    galsim.PixelScale(1./re.scale).toImage(prof)
        #    prof.dilate(re.scale)
        prof = self.dilate(re.scale)

        # wmult isn't really used by drawK, but we need to provide it.
        wmult = 1.0

        review = re.view()
        review.setCenter(0,0)
        imview = im.view()
        imview.setCenter(0,0)

        prof.SBProfile.drawK(review.image, imview.image, gain, wmult)

        return re,im



# --- Now defining the derived classes ---
#
# All derived classes inherit the GSObject method interface, but therefore have a "has a"
# relationship with the C++ SBProfile class rather than an "is a" one...
#
# The __init__ method is usually simple and all the GSObject methods & attributes are inherited.
#
class Gaussian(GSObject):
    """A class describing a 2-D Gaussian surface brightness profile.

    The Gaussian surface brightness profile is characterized by two properties, its `flux`
    and the characteristic size `sigma` where the radial profile of the circular Gaussian
    drops off as `exp[-r^2 / (2. * sigma^2)]`.

    Initialization
    --------------

    A Gaussian can be initialized using one (and only one) of three possible size parameters:
    `sigma`, `fwhm`, or `half_light_radius`.  Exactly one of these three is required.

    @param sigma            The value of sigma of the profile.  Typically given in arcsec.
                            [One of sigma, fwhm, or half_light_radius is required.]
    @param fwhm             The full-width-half-max of the profile.  Typically given in arcsec.
                            [One of sigma, fwhm, or half_light_radius is required.]
    @param half_light_radius  The half-light radius of the profile.  Typically given in arcsec.
                            [One of sigma, fwhm, or half_light_radius is required.]
    @param flux             The flux (in photons) of the profile. [default: 1]
    @param gsparams         An optional GSParams argument.  See the docstring for galsim.GSParams
                            for details. [default: None]

    Methods
    -------

    In addition to the usual GSObject methods, Gaussian has the following access methods:

        sigma = gauss.getSigma()
        fwhm = gauss.getFWHM()
        hlr = gauss.getHalfLightRadius()
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

    The Moffat surface brightness profile is I(R) propto [1 + (r/scale_radius)^2]^(-beta).  The
    SBProfile representation of a Moffat profile also includes an optional truncation beyond a
    given radius.

    For more information, refer to

        http://home.fnal.gov/~neilsen/notebook/astroPSF/astroPSF.html

    Initialization
    --------------

    A Moffat can be initialized using one (and only one) of three possible size parameters:
    `scale_radius`, `fwhm`, or `half_light_radius`.  Exactly one of these three is required.

    @param beta             The beta parameter of the profile.
    @param scale_radius     The scale radius of the profile.  Typically given in arcsec.
                            [One of scale_radius, fwhm, or half_light_radius is required.]
    @param half_light_radius  The half-light radius of the profile.  Typically given in arcsec.
                            [One of scale_radius, fwhm, or half_light_radius is required.]
    @param fwhm             The full-width-half-max of the profile.  Typically given in arcsec.
                            [One of scale_radius, fwhm, or half_light_radius is required.]
    @param trunc            An optional truncation radius at which the profile is made to drop to
                            zero.  [default: 0, indicating no truncation]
    @param flux             The flux (in photons) of the profile. [default: 1]
    @param gsparams         An optional GSParams argument.  See the docstring for galsim.GSParams
                            for details. [default: None]

    Methods
    -------

    In addition to the usual GSObject methods, Moffat has the following access methods:

        beta = moffat_obj.getBeta()
        rD = moffat_obj.getScaleRadius()
        fwhm = moffat_obj.getFWHM()
        hlr = moffat_obj.getHalfLightRadius()
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
    @param gsparams         An optional GSParams argument.  See the docstring for galsim.GSParams
                            for details. [default: None]

    Methods
    -------

    In addition to the usual GSObject methods, Airy has the following access methods:

        lam_over_diam = airy_obj.getLamOverD()
        fwhm = airy_obj.getFWHM()
        hlr = airy_obj.getHalfLightRadius()

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
        """Return the lam_over_diam parameter of this Airy profile.
        """
        return self.SBProfile.getLamOverD()


class Kolmogorov(GSObject):
    """A class describing a Kolmogorov surface brightness profile, which represents a long
    exposure atmospheric PSF.

    For more information, refer to

        http://en.wikipedia.org/wiki/Atmospheric_seeing#The_Kolmogorov_model_of_turbulence

    Initialization
    --------------

    The Kolmogorov profile is normally defined in terms of the ratio lambda / D, where lambda is
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
                            See above for details about calculating it.  [One of lam_over_r0,
                            fwhm, or half_light_radius is required.]
    @param fwhm             The full-width-half-max of the profile.  Typically given in arcsec.
                            [One of lam_over_r0, fwhm, or half_light_radius is required.]
    @param half_light_radius  The half-light radius of the profile.  Typically given in arcsec.
                            [One of sigma, fwhm, or half_light_radius is required.]
    @param flux             The flux (in photons) of the profile. [default: 1]
    @param gsparams         An optional GSParams argument.  See the docstring for galsim.GSParams
                            for details. [default: None]

    Methods
    -------

    In addition to the usual GSObject methods, Kolmogorov has the following access methods:

        lam_over_r0 = kolm.getLamOverR0()
        fwhm = kolm.getFWHM()
        hlr = kolm.getHalfLightRadius()
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
    """A class describing a pixel profile.  This is just a 2-d square top-hat function.

    This class is typically used to represent a Pixel response function, and therefore is only
    needed when drawing images using Fourier transform or real-space convolution (with the draw
    method), not when using photon-shooting (with the drawShoot method).

    Initialization
    --------------

    @param scale            The linear scale size of the pixel.  Typically given in arcsec.
    @param flux             The flux (in photons) of the profile.  This should almost certainly
                            be left at the default value of 1. [default: 1]
    @param gsparams         An optional GSParams argument.  See the docstring for galsim.GSParams
                            for details. [default: None]

    Methods
    -------

    In addition to the usual GSObject methods, Pixel has the following access method:

        scale = pixel.getScale()

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
        """Return the pixel scale
        """
        return self.SBProfile.getWidth()


class Box(GSObject):
    """A class describing a box profile.  This is just a 2-d top-hat function, where the
    width and height are allowed to be different.

    Initialization
    --------------

    @param width            The width of the Box.
    @param height           The height of the Box.
    @param flux             The flux (in photons) of the profile. [default: 1]
    @param gsparams         An optional GSParams argument.  See the docstring for galsim.GSParams
                            for details. [default: None]

    Methods
    -------

    In addition to the usual GSObject methods, Box has the following access methods:

        width = box.getWidth()
        height = box.getHeight()

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
    the surface brightness profile scales as `I(r) propto exp[-(r/scale_radius)^{1/n}]`, or
    `I(r) propto exp[-b*(r/half_light_radius)^{1/n}]` (where b is calculated to give the right
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

    Note that if you are building many Sersic profiles using truncation, the code will be more
    efficient if the truncation is always the same multiple of `scale_radius`, since it caches
    many calculations that depend on the ratio `trunc/scale_radius`.

    A Sersic can be initialized using one (and only one) of two possible size parameters:
    `scale_radius` or `half_light_radius`.  Exactly one of these two is required.

    @param n                The Sersic index, n.
    @param half_light_radius  The half-light radius of the profile.  Typically given in arcsec.
                            [One of scale_radius or half_light_radius is required.]
    @param scale_radius     The scale radius of the profile.  Typically given in arcsec.
                            [One of scale_radius or half_light_radius is required.]
    @param flux             The flux (in photons) of the profile. [default: 1]
    @param trunc            An optional truncation radius at which the profile is made to drop to
                            zero.  [default: 0, indicating no truncation]
    @param flux_untruncated Should the provided flux and half_light_radius refer to the
                            untruncated profile? See below for more details. [default: False]
    @param gsparams         An optional GSParams argument.  See the docstring for galsim.GSParams
                            for details. [default: None]

    Flux of a truncated profile
    ---------------------------

    If you are truncating the profile, the optional parameter, `flux_untruncated`, specifies
    whether the `flux` and `half_light_radius` specifications correspond to the untruncated
    profile (`True`) or to the truncated profile (`False`, default).  The impact of this parameter
    is a little be subtle, so we'll go through a few examples to show how it works.

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

        n = sersic_obj.getN()
        r0 = sersic_obj.getScaleRadius()
        hlr = sersic_obj.getHalfLightRadius()
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

    Surface brightness profile with I(r) propto exp[-r/scale_radius].  This is a special case of
    the Sersic profile, but is given a separate class since the Fourier transform has closed form
    and can be generated without lookup tables.

    Initialization
    --------------

    An Exponential can be initialized using one (and only one) of two possible size parameters:
    `scale_radius` or `half_light_radius`.  Exactly one of these two is required.

    @param half_light_radius  The half-light radius of the profile.  Typically given in arcsec.
                            [One of scale_radius or half_light_radius is required.]
    @param scale_radius     The scale radius of the profile.  Typically given in arcsec.
                            [One of scale_radius or half_light_radius is required.]
    @param flux             The flux (in photons) of the profile. [default: 1]
    @param gsparams         An optional GSParams argument.  See the docstring for galsim.GSParams
                            for details. [default: None]

    Methods
    -------

    In addition to the usual GSObject methods, Exponential has the following access methods:

        r0 = exp_obj.getScaleRadius()
        hlr = exp_obj.getHalfLightRadius()
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

    Surface brightness profile with I(r) propto exp[-(r/scale_radius)^{1/4}].
    This is completely equivalent to a Sersic with n=4.

    For more information, refer to

        http://en.wikipedia.org/wiki/De_Vaucouleurs'_law


    Initialization
    --------------

    A DeVaucouleurs can be initialized using one (and only one) of two possible size parameters:
    `scale_radius` or `half_light_radius`.  Exactly one of these two is required.

    @param scale_radius     The value of sigma of the profile.  Typically given in arcsec.
                            [One of scale_radius or half_light_radius is required.]
    @param half_light_radius  The half-light radius of the profile.  Typically given in arcsec.
                            [One of scale_radius or half_light_radius is required.]
    @param flux             The flux (in photons) of the profile. [default: 1]
    @param trunc            An optional truncation radius at which the profile is made to drop to
                            zero.  [default: 0, indicating no truncation]
    @param flux_untruncated Should the provided flux and half_light_radius refer to the
                            untruncated profile? See the docstring for Sersic for more details.
                            [default: False]
    @param gsparams         An optional GSParams argument.  See the docstring for galsim.GSParams
                            for details. [default: None]

    Methods
    -------

    In addition to the usual GSObject methods, DeVaucouleurs has the following access methods:

        r0 = devauc_obj.getScaleRadius()
        hlr = devauc_obj.getHalfLightRadius()
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
