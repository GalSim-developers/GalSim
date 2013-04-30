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
import collections
import numpy as np
import galsim
import utilities

version = '0.4.1'

class GSObject(object):
    """Base class for defining the interface with which all GalSim Objects access their shared 
    methods and attributes, particularly those from the C++ SBProfile classes.

    All GSObject classes take an optional `gsparams` argument so we document that feature here.
    For all documentation about the specific derived classes, please see the docstring for each 
    one individually.  
    
    The gsparams argument can be used to specify various numbers that govern the tradeoff between
    accuracy and speed for the calculations made in drawing a GSObject.  The numbers are
    encapsulated in a class called GSParams, and the user should make careful choices whenever they
    opt to deviate from the defaults.  For more details about the parameters and their default
    values, use `help(galsim.GSParams)`.

    Example usage:
    
    Let's say you want to do something that requires an FFT larger than 4096 x 4096 (and you have 
    enough memory to handle it!).  Then you can create a new GSParams object with a larger 
    maximum_fft_size and pass that to your GSObject on construction:

        >>> gal = galsim.Sersic(n=4, half_light_radius=4.3)
        >>> psf = galsim.Moffat(beta=3, fwhm=2.85)
        >>> pix = galsim.Pixel(0.05)                       # Note the very small pixel scale!
        >>> conv = galsim.Convolve([gal,psf,pix])
        >>> im = galsim.ImageD(1000,1000)
        >>> im.scale = 0.05                                # Use the same pixel scale on the image.
        >>> conv.draw(im,normalization='sb')               # This uses the default GSParams.
        Traceback (most recent call last):
          File "<stdin>", line 1, in <module>
          File "/Library/Python/2.6/site-packages/galsim/base.py", line 579, in draw
            self.SBProfile.draw(image.view(), gain, wmult)
        RuntimeError: SB Error: fourierDraw() requires an FFT that is too large, 6144
        >>> big_fft_params = galsim.GSParams(maximum_fft_size = 10240)
        >>> conv = galsim.Convolve([gal,psf,pix],gsparams=big_fft_params)
        >>> conv.draw(im,normalization='sb')               # Now it works (but is slow!)
        <galsim._galsim.ImageD object at 0x1037823c0>
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
                  'maxk_threshold' : float,
                  'kvalue_accuracy' : float,
                  'xvalue_accuracy' : float,
                  'shoot_accuracy' : float,
                  'realspace_relerr' : float,
                  'realspace_abserr' : float,
                  'integration_relerr' : float,
                  'integration_abserr' : float
                }
    def __init__(self, rhs):
        # This guarantees that all GSObjects have an SBProfile
        if isinstance(rhs, galsim.GSObject):
            self.SBProfile = rhs.SBProfile
        elif isinstance(rhs, galsim.SBProfile):
            self.SBProfile = rhs
        else:
            raise TypeError("GSObject must be initialized with an SBProfile or another GSObject!")
    
    # Make op+ of two GSObjects work to return an Add object
    def __add__(self, other):
        return Add(self, other)

    # op+= converts this into the equivalent of an Add object
    def __iadd__(self, other):
        GSObject.__init__(self, galsim.SBAdd([self.SBProfile, other.SBProfile]))
        self.__class__ = Add
        return self

    # Make op* and op*= work to adjust the flux of an object
    def __imul__(self, other):
        self.scaleFlux(other)
        return self

    def __mul__(self, other):
        ret = self.copy()
        ret *= other
        return ret

    def __rmul__(self, other):
        ret = self.copy()
        ret *= other
        return ret

    # Likewise for op/ and op/=
    def __idiv__(self, other):
        self.scaleFlux(1. / other)
        return self

    def __div__(self, other):
        ret = self.copy()
        ret /= other
        return ret

    def __itruediv__(self, other):
        return __idiv__(self, other)

    def __truediv__(self, other):
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
        return ret

    # Now define direct access to all SBProfile methods via calls to self.SBProfile.method_name()
    #
    def maxK(self):
        """Returns value of k beyond which aliasing can be neglected.
        """
        return self.SBProfile.maxK()

    def nyquistDx(self):
        """Returns Image pixel spacing that does not alias maxK.
        """
        return self.SBProfile.nyquistDx()

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

    def xValue(self, position):
        """Returns the value of the object at a chosen 2D position in real space.
        
        xValue() is available if obj.isAnalyticX() == True.

        As in SBProfile, this function assumes all are real-valued.  xValue() may not be implemented
        for derived classes (e.g. SBConvolve) that require a Discrete Fourier Transform to 
        determine real space values.  In this case, an SBError will be thrown at the C++ layer 
        (raises a RuntimeError in Python).
        
        @param position  A 2D galsim.PositionD/galsim.PositionI instance giving the position in real
                         space.
        """
        return self.SBProfile.xValue(position)

    def kValue(self, position):
        """Returns the value of the object at a chosen 2D position in k space.

        kValue() is available if the given obj has obj.isAnalyticK() == True. 

        kValue() can be used for all of our simple base classes.  However, if a Convolve object
        representing the convolution of multiple objects uses real-space convolution rather than the
        DFT approach, i.e., real_space=True (either by argument or if it decides on its own to do
        so), then it is not analytic in k-space, so kValue() will raise an exception.  An SBError
        will be thrown at the C++ layer (raises a RuntimeError in Python).

        @param position  A 2D galsim.PositionD/galsim.PositionI instance giving the position in k 
                         space.
        """
        return self.SBProfile.kValue(position)

    def scaleFlux(self, flux_ratio):
        """Multiply the flux of the object by flux_ratio
           
        After this call, the caller's type will be a GSObject.
        This means that if the caller was a derived type that had extra methods beyond
        those defined in GSObject (e.g. getSigma() for a Gaussian), then these methods
        are no longer available.

        @param flux_ratio The factor by which to scale the flux.
        """
        self.SBProfile.scaleFlux(flux_ratio)
        self.__class__ = GSObject

    def setFlux(self, flux):
        """Set the flux of the object.
           
        After this call, the caller's type will be a GSObject.
        This means that if the caller was a derived type that had extra methods beyond
        those defined in GSObject (e.g. getSigma() for a Gaussian), then these methods
        are no longer available.

        @param flux The new flux for the object.
        """
        self.SBProfile.setFlux(flux)
        self.__class__ = GSObject

    def applyTransformation(self, ellipse):
        """Apply a galsim.Ellipse distortion to this object.
           
        galsim.Ellipse objects can be initialized in a variety of ways (see documentation of this
        class, galsim.ellipse.Ellipse in the doxygen documentation, for details).

        Note: if the ellipse includes a dilation, then this transformation will not be
        flux-conserving.  It conserves surface brightness instead.  Thus, the flux will increase by
        the increase in area = dilation^2.

        After this call, the caller's type will be a GSObject.
        This means that if the caller was a derived type that had extra methods beyond
        those defined in GSObject (e.g. getSigma() for a Gaussian), then these methods
        are no longer available.

        @param ellipse The galsim.Ellipse transformation to apply
        """
        if not isinstance(ellipse, galsim.Ellipse):
            raise TypeError("Argument to applyTransformation must be a galsim.Ellipse!")
        self.SBProfile.applyTransformation(ellipse._ellipse)
        self.__class__ = GSObject
 
    def applyDilation(self, scale):
        """Apply a dilation of the linear size by the given scale.

        Scales the linear dimensions of the image by the factor scale.
        e.g. `half_light_radius` <-- `half_light_radius * scale`

        This operation preserves flux.
        See applyMagnification() for a version that preserves surface brightness, and thus 
        changes the flux.

        After this call, the caller's type will be a GSObject.
        This means that if the caller was a derived type that had extra methods beyond
        those defined in GSObject (e.g. getSigma() for a Gaussian), then these methods
        are no longer available.

        @param scale The linear rescaling factor to apply.
        """
        old_flux = self.getFlux()
        self.applyTransformation(galsim.Ellipse(np.log(scale)))
        self.setFlux(old_flux) # conserve flux

    def applyMagnification(self, mu):
        """Apply a lensing magnification, scaling the area and flux by mu at fixed surface
        brightness.
        
        This process applies a lensing magnification mu, which scales the linear dimensions of the
        image by the factor sqrt(mu), i.e., `half_light_radius` <-- `half_light_radius * sqrt(mu)`
        while increasing the flux by a factor of mu.  Thus, applyMagnification preserves surface
        brightness.

        See applyDilation for a version that applies a linear scale factor in the size while
        preserving flux.

        After this call, the caller's type will be a GSObject.
        This means that if the caller was a derived type that had extra methods beyond
        those defined in GSObject (e.g. getSigma for a Gaussian), then these methods
        are no longer available.

        @param mu The lensing magnification to apply.
        """
        self.applyTransformation(galsim.Ellipse(np.log(np.sqrt(mu))))
       
    def applyShear(self, *args, **kwargs):
        """Apply a shear to this object, where arguments are either a galsim.Shear, or arguments
        that will be used to initialize one.

        For more details about the allowed keyword arguments, see the documentation for galsim.Shear
        (for doxygen documentation, see galsim.shear.Shear).

        After this call, the caller's type will be a GSObject.
        This means that if the caller was a derived type that had extra methods beyond
        those defined in GSObject (e.g. getSigma() for a Gaussian), then these methods
        are no longer available.
        """
        if len(args) == 1:
            if kwargs:
                raise TypeError("Error, gave both unnamed and named arguments to applyShear!")
            if not isinstance(args[0], galsim.Shear):
                raise TypeError("Error, unnamed argument to applyShear is not a Shear!")
            shear = args[0]
        elif len(args) > 1:
            raise TypeError("Error, too many unnamed arguments to applyShear!")
        else:
            shear = galsim.Shear(**kwargs)
        self.SBProfile.applyShear(shear._shear)
        self.__class__ = GSObject

    def applyLensing(self, g1, g2, mu):
        """Apply a lensing shear and magnification to this object.

        This GSObject method applies a lensing (reduced) shear and magnification.  The shear must be
        specified using the g1, g2 definition of shear (see galsim.Shear documentation for more
        details).  This is the same definition as the outputs of the galsim.PowerSpectrum and
        galsim.NFWHalo classes, which compute shears according to some lensing power spectrum or
        lensing by an NFW dark matter halo.  The magnification determines the rescaling factor for
        the object area and flux, preserving surface brightness.

        After this call, the caller's type will be a GSObject.
        This means that if the caller was a derived type that had extra methods beyond
        those defined in GSObject (e.g. getSigma() for a Gaussian), then these methods
        are no longer available.

        @param g1      First component of lensing (reduced) shear to apply to the object.
        @param g2      Second component of lensing (reduced) shear to apply to the object.
        @param mu      Lensing magnification to apply to the object.  This is the factor by which
                       the solid angle subtended by the object is magnified, preserving surface
                       brightness.
        """
        self.applyShear(g1=g1, g2=g2)
        self.applyMagnification(mu)

    def applyRotation(self, theta):
        """Apply a rotation theta to this object.
           
        After this call, the caller's type will be a GSObject.
        This means that if the caller was a derived type that had extra methods beyond
        those defined in GSObject (e.g. getSigma() for a Gaussian), then these methods
        are no longer available.

        @param theta Rotation angle (Angle object, +ve anticlockwise).
        """
        if not isinstance(theta, galsim.Angle):
            raise TypeError("Input theta should be an Angle")
        self.SBProfile.applyRotation(theta)
        self.__class__ = GSObject

    def applyShift(self, dx, dy):
        """Apply a (dx, dy) shift to this object.
           
        After this call, the caller's type will be a GSObject.
        This means that if the caller was a derived type that had extra methods beyond
        those defined in GSObject (e.g. getSigma() for a Gaussian), then these methods
        are no longer available.

        @param dx Horizontal shift to apply (float).
        @param dy Vertical shift to apply (float).
        """
        self.SBProfile.applyShift(dx,dy)
        self.__class__ = GSObject

    # Also add methods which create a new GSObject with the transformations applied...
    #
    def createTransformed(self, ellipse):
        """Returns a new GSObject by applying a galsim.Ellipse transformation 
        (shear, dilate, and/or shift).

        Note that galsim.Ellipse objects can be initialized in a variety of ways (see documentation
        of this class, galsim.ellipse.Ellipse in the doxygen documentation, for details).

        @param ellipse The galsim.Ellipse transformation to apply
        @returns The transformed GSObject.
        """
        if not isinstance(ellipse, galsim.Ellipse):
            raise TypeError("Argument to createTransformed must be a galsim.Ellipse!")
        ret = self.copy()
        ret.applyTransformation(ellipse)
        return ret

    def createDilated(self, scale):
        """Returns a new GSObject by applying a dilation of the linear size by the given scale.
        
        Scales the linear dimensions of the image by the factor scale.
        e.g. `half_light_radius` <-- `half_light_radius * scale`

        This operation preserves flux.
        See createMagnified() for a version that preserves surface brightness, and thus 
        changes the flux.

        @param scale The linear rescaling factor to apply.
        @returns The rescaled GSObject.
        """
        ret = self.copy()
        ret.applyDilation(scale)
        return ret

    def createMagnified(self, mu):
        """Returns a new GSObject by applying a lensing magnification, scaling the area and flux by
        mu at fixed surface brightness.

        This process returns a new object with a lensing magnification mu, which scales the linear
        dimensions of the image by the factor sqrt(mu), i.e., `half_light_radius` <--
        `half_light_radius * sqrt(mu)` while increasing the flux by a factor of mu.  Thus, the new
        object has the same surface brightness as the original, but different size and flux.

        See createDilated() for a version that preserves flux.

        @param mu The lensing magnification to apply.
        @returns The rescaled GSObject.
        """
        ret = self.copy()
        ret.applyMagnification(mu)
        return ret

    def createSheared(self, *args, **kwargs):
        """Returns a new GSObject by applying a shear, where arguments are either a galsim.Shear or
        keyword arguments that can be used to create one.

        For more details about the allowed keyword arguments, see the documentation of galsim.Shear
        (for doxygen documentation, see galsim.shear.Shear).

        @returns The sheared GSObject.
        """
        ret = self.copy()
        ret.applyShear(*args, **kwargs)
        return ret

    def createLensed(self, g1, g2, mu):
        """Returns a new GSObject by applying a lensing shear and magnification.

        This method returns a new GSObject to which the supplied lensing (reduced) shear and
        magnification has been applied.  The shear must be specified using the g1, g2 definition of
        shear (see galsim.Shear documentation for more details).  This is the same definition as the
        outputs of the galsim.PowerSpectrum and galsim.NFWHalo classes, which compute shears
        according to some lensing power spectrum or lensing by an NFW dark matter halo. The
        magnification determines the rescaling factor for the object area and flux, preserving
        surface brightness.

        @param g1      First component of lensing (reduced) shear to apply to the object.
        @param g2      Second component of lensing (reduced) shear to apply to the object.
        @param mu      Lensing magnification to apply to the object.  This is the factor by which
                       the solid angle subtended by the object is magnified, preserving surface
                       brightness.
        @returns       The lensed GSObject.
        """
        ret = self.copy()
        ret.applyLensing(g1, g2, mu)
        return ret

    def createRotated(self, theta):
        """Returns a new GSObject by applying a rotation.

        @param theta Rotation angle (Angle object, +ve anticlockwise).
        @returns The rotated GSObject.
        """
        if not isinstance(theta, galsim.Angle):
            raise TypeError("Input theta should be an Angle")
        ret = self.copy()
        ret.applyRotation(theta)
        return ret
        
    def createShifted(self, dx, dy):
        """Returns a new GSObject by applying a shift.

        @param dx Horizontal shift to apply (float).
        @param dy Vertical shift to apply (float).
        @returns The shifted GSObject.
        """
        ret = self.copy()
        ret.applyShift(dx, dy)
        return ret

    # Make sure the image is defined with the right size and scale for the draw and
    # drawShoot commands.
    def _draw_setup_image(self, image, dx, wmult, add_to_image):

        # Make sure the type of wmult is correct and has a valid value:
        if type(wmult) != float:
            wmult = float(wmult)
        if wmult <= 0:
            raise ValueError("Invalid wmult <= 0 in draw command")

        # Check dx value and adjust if necessary:
        if dx is None:
            if image is not None and image.getScale() > 0.:
                dx = image.getScale()
            else:
                dx = self.SBProfile.nyquistDx()
        elif dx <= 0:
            dx = self.SBProfile.nyquistDx()
        elif type(dx) != float:
            dx = float(dx)

        # Make image if necessary:
        if image is None:
            # Can't add to image if none is provided.
            if add_to_image:
                raise ValueError("Cannot add_to_image if image is None")
            N = self.SBProfile.getGoodImageSize(dx,wmult)
            image = galsim.ImageF(N,N)
            image.setScale(dx)

        # Resize the given image if necessary:
        elif not image.getBounds().isDefined():
            # Can't add to image if need to resize
            if add_to_image:
                raise ValueError("Cannot add_to_image if image bounds are not defined")
            N = self.SBProfile.getGoodImageSize(dx,wmult)
            bounds = galsim.BoundsI(1,N,1,N)
            image.resize(bounds)
            image.setScale(dx)
            image.setZero()

        # Else just make sure the scale is set correctly:
        else:
            # Clear the image if we are not adding to it.
            if not add_to_image:
                image.setZero()
            image.setScale(dx)

        return image, dx

    def _fix_center(self, image, scale):
        # For even-sized images, the SBProfile draw function centers the result in the 
        # pixel just up and right of the real center.  So shift it back to make sure it really
        # draws in the center.
        even_x = (image.xmax-image.xmin+1) % 2 == 0
        even_y = (image.ymax-image.ymin+1) % 2 == 0
        if even_x:
            if even_y: prof = self.createShifted(-0.5*scale,-0.5*scale)
            else: prof = self.createShifted(-0.5*scale,0.)
        else:
            if even_y: prof = self.createShifted(0.,-0.5*scale)
            else: prof = self
        return prof


    def draw(self, image=None, dx=None, gain=1., wmult=1., normalization="flux",
             add_to_image=False, use_true_center=True):
        """Draws an Image of the object, with bounds optionally set by an input Image.

        The draw method is used to draw an Image of the GSObject, typically using Fourier space
        convolution (or, for certain GSObjects that have hard edges, real-space convolution may be
        used), and using interpolation to carry out image transformations such as shearing.  This
        method can create a new Image or can draw into an existing one, depending on the choice of
        the `image` keyword parameter.  Other keywords of particular relevance for users are those
        that set the pixel scale for the image (`dx`), that choose the normalization convention for
        the flux (`normalization`), and that decide whether the clear the input Image before drawing
        into it (`add_to_image`).

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

        @param image  If provided, this will be the image on which to draw the profile.
                      If `image = None`, then an automatically-sized image will be created.
                      If `image != None`, but its bounds are undefined (e.g. if it was 
                        constructed with `image = galsim.ImageF()`), then it will be resized
                        appropriately based on the profile's size (default `image = None`).

        @param dx     If provided, use this as the pixel scale for the image.
                      If `dx` is `None` and `image != None`, then take the provided image's pixel 
                        scale.
                      If `dx` is `None` and `image == None`, then use the Nyquist scale 
                        `= pi/maxK()`.
                      If `dx <= 0` (regardless of image), then use the Nyquist scale `= pi/maxK()`.
                      (Default `dx = None`.)

        @param gain   The number of photons per ADU ("analog to digital units", the units of the 
                      numbers output from a CCD).  (Default `gain =  1.`)

        @param wmult  A factor by which to make an automatically-sized image larger than it would 
                      normally be made.  This factor also applies to any intermediate images during
                      Fourier calculations.  The size of the intermediate images are normally 
                      automatically chosen to reach some preset accuracy targets (see 
                      include/galsim/SBProfile.h); however, if you see strange artifacts in the 
                      image, you might try using `wmult > 1`.  This will take longer of 
                      course, but it will produce more accurate images, since they will have 
                      less "folding" in Fourier space. (Default `wmult = 1.`)

        @param normalization  Two options for the normalization:
                              "flux" or "f" means that the sum of the output pixels is normalized
                                  to be equal to the total flux.  (Modulo any flux that falls off 
                                  the edge of the image of course, and note the caveat in the draw
                                  method documentation regarding the need to convolve with a pixel
                                  response.)
                              "surface brightness" or "sb" means that the output pixels sample
                                  the surface brightness distribution at each location.
                              (Default `normalization = "flux"`)

        @param add_to_image  Whether to add flux to the existing image rather than clear out
                             anything in the image before drawing.
                             Note: This requires that image be provided (i.e. `image` is not `None`)
                             and that it have defined bounds (default `add_to_image = False`).

        @param use_true_center  Normally, the profile is drawn to be centered at the true center
                                of the image (using the function `image.bounds.trueCenter()`).
                                If you would rather use the integer center (given by
                                `image.bounds.center()`), set this to `False`.  
                                (default `use_true_center = True`)

        @returns      The drawn image.
        """
        # Raise an exception immediately if the normalization type is not recognized
        if not normalization.lower() in ("flux", "f", "surface brightness", "sb"):
            raise ValueError(("Invalid normalization requested: '%s'. Expecting one of 'flux', "+
                              "'f', 'surface brightness' or 'sb'.") % normalization)

        # Make sure the type of gain is correct and has a valid value:
        if type(gain) != float:
            gain = float(gain)
        if gain <= 0.:
            raise ValueError("Invalid gain <= 0. in draw command")

        # Make sure image is setup correctly
        image, dx = self._draw_setup_image(image,dx,wmult,add_to_image)

        # Fix the centering for even-sized images
        if use_true_center:
            prof = self._fix_center(image, image.scale)
        else:
            prof = self

        # SBProfile draw command uses surface brightness normalization.  So if we
        # want flux normalization, we need to scale the flux by dx^2
        if normalization.lower() == "flux" or normalization.lower() == "f":
            # Rather than change the flux of the GSObject though, we change the gain.
            # gain is photons / ADU.  The photons are the given flux, and we want to 
            # multiply the ADU by dx^2.  i.e. divide gain by dx^2.
            gain /= dx**2

        image.added_flux = prof.SBProfile.draw(image.view(), gain, wmult)

        return image

    def drawShoot(self, image=None, dx=None, gain=1., wmult=1., normalization="flux",
                  add_to_image=False, use_true_center=True,
                  n_photons=0., rng=None, max_extra_noise=0., poisson_flux=None):
        """Draw an image of the object by shooting individual photons drawn from the surface 
        brightness profile of the object.

        The drawShoot() method is used to draw an image of an object by shooting a number of photons
        to randomly sample the profile of the object. The resulting image will thus have Poisson
        noise due to the finite number of photons shot.  drawShoot() can create a new Image or use
        an existing one, depending on the choice of the `image` keyword parameter.  Other keywords
        of particular relevance for users are those that set the pixel scale for the image (`dx`),
        that choose the normalization convention for the flux (`normalization`), and that decide
        whether the clear the input Image before shooting photons into it (`add_to_image`).

        As for the draw command, the object will always be drawn with its nominal center at the 
        center location of the image.  See the documentation for draw for more discussion about
        the implications of this for even- and odd-sized images.

        It is important to remember that the image produced by drawShoot() represents the object as
        convolved with the square image pixel.  So when using drawShoot() instead of draw(), you
        should not explicitly include the pixel response by convolving with a Pixel GSObject.  Using
        drawShoot without convolving with a Pixel will produce the equivalent image (for very large
        n_photons) as draw() produces when the same object is convolved with `Pixel(xw=dx)` when
        drawing onto an image with pixel scale `dx`.

        Note that the drawShoot method is unavailable for objects which contain an SBDeconvolve,
        or are compound objects (e.g. Add, Convolve) that include an SBDeconvolve.

        On return, the image will have a member `added_flux`, which will be set to be the total
        flux of photons that landed inside the image bounds.  This may be useful as a sanity check 
        that you have provided a large enough image to catch most of the flux.  For example:
        
            obj.drawShoot(image)
            assert image.added_flux > 0.99 * obj.getFlux()

        The appropriate threshold will depend on your particular application, including what kind
        of profile the object has, how big your image is relative to the size of your object, 
        whether you are keeping `poisson_flux = True`, etc.

        @param image  If provided, this will be the image on which to draw the profile.
                      If `image = None`, then an automatically-sized image will be created.
                      If `image != None`, but its bounds are undefined (e.g. if it was constructed 
                        with `image = galsim.ImageF()`), then it will be resized appropriately base 
                        on the profile's size.
                      (Default `image = None`.)

        @param dx     If provided, use this as the pixel scale for the image.
                      If `dx` is `None` and `image != None`, then take the provided image's pixel 
                        scale.
                      If `dx` is `None` and `image == None`, then use the Nyquist scale 
                        `= pi/maxK()`.
                      If `dx <= 0` (regardless of image), then use the Nyquist scale `= pi/maxK()`.
                      (Default `dx = None`.)

        @param gain   The number of photons per ADU ("analog to digital units", the units of the 
                      numbers output from a CCD).  (Default `gain =  1.`)

        @param wmult  A factor by which to make an automatically-sized image larger than 
                      it would normally be made. (Default `wmult = 1.`)

        @param normalization    Two options for the normalization:
                                 "flux" or "f" means that the sum of the output pixels is normalized
                                   to be equal to the total flux.  (Modulo any flux that falls off 
                                   the edge of the image of course.)
                                 "surface brightness" or "sb" means that the output pixels sample
                                   the surface brightness distribution at each location.
                                (Default `normalization = "flux"`)

        @param add_to_image     Whether to add flux to the existing image rather than clear out
                                anything in the image before drawing.
                                Note: This requires that image be provided (i.e. `image != None`)
                                and that it have defined bounds (default `add_to_image = False`).
                              
        @param use_true_center  Normally, the profile is drawn to be centered at the true center
                                of the image (using the function `image.bounds.trueCenter()`).
                                If you would rather use the integer center (given by
                                `image.bounds.center()`), set this to `False`.  
                                (default `use_true_center = True`)

        @param n_photons        If provided, the number of photons to use.
                                If not provided (i.e. `n_photons = 0`), use as many photons as
                                  necessary to result in an image with the correct Poisson shot 
                                  noise for the object's flux.  For positive definite profiles, this
                                  is equivalent to `n_photons = flux`.  However, some profiles need
                                  more than this because some of the shot photons are negative 
                                  (usually due to interpolants).
                                (Default `n_photons = 0`).

        @param rng              If provided, a random number generator to use for photon shooting.
                                  (may be any kind of `galsim.BaseDeviate` object)
                                If `rng=None`, one will be automatically created, using the time
                                  as a seed.
                                (Default `rng = None`)

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
                                  definition of noise, not a "sigma" definition.
                                (Default `max_extra_noise = 0.`)

        @param poisson_flux     Whether to allow total object flux scaling to vary according to 
                                Poisson statistics for `n_photons` samples (default 
                                `poisson_flux = True` unless n_photons is given, in which case
                                the default is `poisson_flux = False`).

        @returns      The drawn image.
        """

        # Raise an exception immediately if the normalization type is not recognized
        if not normalization.lower() in ("flux", "f", "surface brightness", "sb"):
            raise ValueError(("Invalid normalization requested: '%s'. Expecting one of 'flux', "+
                              "'f', 'surface brightness' or 'sb'.") % normalization)

        # Make sure the type of gain is correct and has a valid value:
        if type(gain) != float:
            gain = float(gain)
        if gain <= 0.:
            raise ValueError("Invalid gain <= 0. in draw command")

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

        # Make sure image is setup correctly
        image, dx = self._draw_setup_image(image,dx,wmult,add_to_image)

        # Fix the centering for even-sized images
        if use_true_center:
            prof = self._fix_center(image, image.scale)
        else:
            prof = self

        # SBProfile drawShoot command uses surface brightness normalization.  So if we
        # want flux normalization, we need to scale the flux by dx^2
        if normalization.lower() == "flux" or normalization.lower() == "f":
            # Rather than change the flux of the GSObject though, we change the gain.
            # gain is photons / ADU.  The photons are the given flux, and we want to 
            # multiply the ADU by dx^2.  i.e. divide gain by dx^2.
            gain /= dx**2

        try:
            image.added_flux = prof.SBProfile.drawShoot(
                image.view(), n_photons, uniform_deviate, gain, max_extra_noise,
                poisson_flux, add_to_image)
        except RuntimeError:
            # Give some extra explanation as a warning, then raise the original exception
            # so the traceback shows as much detail as possible.
            import warnings
            warnings.warn(
                "Unable to drawShoot from this GSObject, perhaps it contains an SBDeconvolve "+
                "in the SBProfile attribute or is a compound including one or more Deconvolve "+
                "objects.")
            raise

        return image

    def drawK(self, re=None, im=None, dk=None, gain=1., wmult=1., add_to_image=False):
        """Draws the k-space Images (real and imaginary parts) of the object, with bounds
        optionally set by input Images.

        Normalization is always such that re(0,0) = flux.  Unlike the real-space draw and
        drawShoot functions, the (0,0) point will always be one of the actual pixel values.
        For even-sized images, it will be 1/2 pixel above and to the right of the true 
        center of the image.

        @param re     If provided, this will be the real part of the k-space image.
                      If `re = None`, then an automatically-sized image will be created.
                      If `re != None`, but its bounds are undefined (e.g. if it was 
                        constructed with `re = galsim.ImageF()`), then it will be resized
                        appropriately based on the profile's size (default `re = None`).

        @param im     If provided, this will be the imaginary part of the k-space image.
                      A provided im must match the size and scale of re.
                      If `im = None`, then an automatically-sized image will be created.
                      If `im != None`, but its bounds are undefined (e.g. if it was 
                        constructed with `im = galsim.ImageF()`), then it will be resized
                        appropriately based on the profile's size (default `im = None`).

        @param dk     If provided, use this as the pixel scale for the images.
                      If `dk` is `None` and `re, im != None`, then take the provided images' pixel 
                        scale (which must be equal).
                      If `dk` is `None` and `re, im == None`, then use the Nyquist scale 
                        `= pi/maxK()`.
                      If `dk <= 0` (regardless of image), then use the Nyquist scale `= pi/maxK()`.
                      (Default `dk = None`.)

        @param gain   The number of photons per ADU ("analog to digital units", the units of the 
                      numbers output from a CCD).  (Default `gain =  1.`)

        @param wmult  A factor by which to make an automatically-sized image larger than it would 
                      normally be made.  This factor also applies to any intermediate images during
                      Fourier calculations.  The size of the intermediate images are normally 
                      automatically chosen to reach some preset accuracy targets (see 
                      include/galsim/SBProfile.h); however, if you see strange artifacts in the 
                      image, you might try using `wmult > 1`.  This will take longer of 
                      course, but it will produce more accurate images, since they will have 
                      less "folding" in Fourier space. (Default `wmult = 1.`)

        @param add_to_image  Whether to add to the existing images rather than clear out
                             anything in the image before drawing.
                             Note: This requires that images be provided (i.e. `re`, `im` are
                             not `None`) and that they have defined bounds (default 
                             `add_to_image = False`).

        @returns      (re, im)  (created if necessary)
        """
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
            if dk is None:
                if re.getScale() != im.getScale():
                    raise ValueError("re and im do not have the same input scale")
            if re.getBounds().isDefined() or im.getBounds().isDefined():
                if re.getBounds() != im.getBounds():
                    raise ValueError("re and im do not have the same defined bounds")

        # Make sure images are setup correctly
        re, dk = self._draw_setup_image(re,dk,wmult,add_to_image)
        im, dk = self._draw_setup_image(im,dk,wmult,add_to_image)

        self.SBProfile.drawK(re.view(), im.view(), gain, wmult)

        return re,im



# --- Now defining the derived classes ---
#
# All derived classes inherit the GSObject method interface, but therefore have a "has a" 
# relationship with the C++ SBProfile class rather than an "is a" one...
#
# The __init__ method is usually simple and all the GSObject methods & attributes are inherited.
# 
class Gaussian(GSObject):
    """A class describing Gaussian profile objects.  Has an SBGaussian in the SBProfile attribute.

    For more details of the Gaussian Surface Brightness profile, please see the SBGaussian
    documentation produced by doxygen.

    Initialization
    --------------
    A Gaussian can be initialized using one (and only one) of three possible size parameters

        half_light_radius
        sigma
        fwhm

    and an optional `flux` parameter [default `flux = 1`].

    Example:
        
        >>> gauss_obj = galsim.Gaussian(flux=3., sigma=1.)
        >>> gauss_obj.getHalfLightRadius()
        1.1774100225154747
        >>> gauss_obj = galsim.Gaussian(flux=3, half_light_radius=1.)
        >>> gauss_obj.getSigma()
        0.8493218002880191

    Attempting to initialize with more than one size parameter is ambiguous, and will raise a
    TypeError exception.

    You may also specify a gsparams argument.  See the docstring for galsim.GSParams using
    help(galsim.GSParams) for more information about this option.

    Methods
    -------
    The Gaussian is a GSObject, and inherits all of the GSObject methods (draw(), drawShoot(),
    applyShear() etc.) and operator bindings.
    """
    
    # Initialization parameters of the object, with type information, to indicate
    # which attributes are allowed / required in a config file for this object.
    # _req_params are required
    # _opt_params are optional
    # _single_params are a list of sets for which exactly one in the list is required.
    # _takes_rng indicates whether the constructor should be given the current rng.
    _req_params = {}
    _opt_params = { "flux" : float }
    _single_params = [ { "sigma" : float, "half_light_radius" : float, "fwhm" : float } ]
    _takes_rng = False
    
    # --- Public Class methods ---
    def __init__(self, half_light_radius=None, sigma=None, fwhm=None, flux=1., gsparams=None):
        # Initialize the SBProfile
        GSObject.__init__(
            self, galsim.SBGaussian(
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
    """A class describing Moffat PSF profiles.  Has an SBMoffat in the SBProfile attribute.

    For more details of the Moffat Surface Brightness profile, please see the SBMoffat
    documentation produced by doxygen, or refer to 
    http://home.fnal.gov/~neilsen/notebook/astroPSF/astroPSF.html.

    Initialization
    --------------
    A Moffat is initialized with a slope parameter beta, one (and only one) of three possible size
    parameters

        scale_radius
        half_light_radius
        fwhm

    an optional truncation radius parameter `trunc` [default `trunc = 0.`, indicating no truncation]
    and a `flux` parameter [default `flux = 1`].

    Example:
    
        >>> moffat_obj = galsim.Moffat(beta=3., scale_radius=3., flux=0.5)
        >>> moffat_obj.getHalfLightRadius()
        1.9307827587167474
        >>> moffat_obj = galsim.Moffat(beta=3., half_light_radius=1., flux=0.5)
        >>> moffat_obj.getScaleRadius()
        1.5537739740300376

    Attempting to initialize with more than one size parameter is ambiguous, and will raise a
    TypeError exception.

    You may also specify a gsparams argument.  See the docstring for galsim.GSParams using
    help(galsim.GSParams) for more information about this option.

    Methods
    -------
    The Moffat is a GSObject, and inherits all of the GSObject methods (draw(), drawShoot(),
    applyShear() etc.) and operator bindings.
    """

    # Initialization parameters of the object, with type information
    _req_params = { "beta" : float }
    _opt_params = { "trunc" : float , "flux" : float }
    _single_params = [ { "scale_radius" : float, "half_light_radius" : float, "fwhm" : float } ]
    _takes_rng = False

    # --- Public Class methods ---
    def __init__(self, beta, scale_radius=None, half_light_radius=None,  fwhm=None, trunc=0.,
                 flux=1., gsparams=None):
        GSObject.__init__(
            self, galsim.SBMoffat(
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


class AtmosphericPSF(GSObject):
    """Base class for long exposure Kolmogorov PSF.  Currently deprecated: use Kolmogorov.

    Initialization
    --------------
    Example:    

        >>> atmospheric_psf = galsim.AtmosphericPSF(lam_over_r0, interpolant=None, oversampling=1.5)
    
    Initializes atmospheric_psf as a galsim.AtmosphericPSF() instance.  This class is currently
    deprecated in favour of the newer Kolmogorov class which does not require grid FFTs.  However,
    it is retained as a placeholder for a future AtmosphericPSF which will model the turbulent
    atmosphere stochastically.

    @param lam_over_r0     lambda / r0 in the physical units adopted for apparent sizes (user 
                           responsible for consistency), where r0 is the Fried parameter.  The FWHM
                           of the Kolmogorov PSF is ~0.976 lambda/r0 (e.g., Racine 1996, PASP 699, 
                           108). Typical  values for the Fried parameter are on the order of 10cm 
                           for most observatories and up to 20cm for excellent sites.  The values 
                           are usually quoted at lambda = 500nm and r0 depends on wavelength as
                           [r0 ~ lambda^(-6/5)].
    @param fwhm            FWHM of the Kolmogorov PSF.
                           Either `fwhm` or `lam_over_r0` (and only one) must be specified.
    @param interpolant     Either an Interpolant2d (or Interpolant) instance or a string indicating
                           which interpolant should be used.  Options are 'nearest', 'sinc', 
                           'linear', 'cubic', 'quintic', or 'lanczosN' where N should be the 
                           integer order to use. [default `interpolant = galsim.Quintic()`]
    @param oversampling    Optional oversampling factor for the SBInterpolatedImage table 
                           [default `oversampling = 1.5`], setting `oversampling < 1` will produce 
                           aliasing in the PSF (not good).
    @param flux            Total flux of the profile [default `flux=1.`]
    @param gsparams        You may also specify a gsparams argument.  See the docstring for
                           galsim.GSParams using help(galsim.GSParams) for more information about
                           this option.

    Methods
    -------
    The AtmosphericPSF is a GSObject, and inherits all of the GSObject methods (draw(), drawShoot(),
    applyShear() etc.) and operator bindings.

    """

    # Initialization parameters of the object, with type information
    _req_params = {}
    _opt_params = { "oversampling" : float , "interpolant" : str , "flux" : float }
    _single_params = [ { "lam_over_r0" : float , "fwhm" : float } ]
    _takes_rng = False

    # --- Public Class methods ---
    def __init__(self, lam_over_r0=None, fwhm=None, interpolant=None, oversampling=1.5, flux=1.,
                 gsparams=None):

        # The FWHM of the Kolmogorov PSF is ~0.976 lambda/r0 (e.g., Racine 1996, PASP 699, 108).
        if lam_over_r0 is None :
            if fwhm is not None :
                lam_over_r0 = fwhm / 0.976
            else:
                raise TypeError("Either lam_over_r0 or fwhm must be specified for AtmosphericPSF")
        else :
            if fwhm is None:
                fwhm = 0.976 * lam_over_r0
            else:
                raise TypeError(
                        "Only one of lam_over_r0 and fwhm may be specified for AtmosphericPSF")
        # Set the lookup table sample rate via FWHM / 2 / oversampling (BARNEY: is this enough??)
        dx_lookup = .5 * fwhm / oversampling

        # Fold at 10 times the FWHM
        stepk_kolmogorov = np.pi / (10. * fwhm)

        # Odd array to center the interpolant on the centroid. Might want to pad this later to
        # make a nice size array for FFT, but for typical seeing, arrays will be very small.
        npix = 1 + 2 * (np.ceil(np.pi / stepk_kolmogorov)).astype(int)
        atmoimage = galsim.atmosphere.kolmogorov_psf_image(
            array_shape=(npix, npix), dx=dx_lookup, lam_over_r0=lam_over_r0, flux=flux)
        
        # Run checks on the interpolant and build default if None
        if interpolant is None:
            quintic = galsim.Quintic(tol=1e-4)
            self.interpolant = galsim.InterpolantXY(quintic)
        else:
            self.interpolant = galsim.utilities.convert_interpolant_to_2d(interpolant)

        # Then initialize the SBProfile
        GSObject.__init__(
            self, galsim.SBInterpolatedImage(atmoimage, xInterp=self.interpolant, dx=dx_lookup,
                                             gsparams=gsparams))

        # The above procedure ends up with a larger image than we really need, which
        # means that the default stepK value will be smaller than we need.  
        # Thus, we call the function calculateStepK() to refine the value.
        self.SBProfile.calculateStepK()
        self.SBProfile.calculateMaxK()

    def getHalfLightRadius(self):
        # TODO: This seems like it would not be impossible to calculate
        raise NotImplementedError("Half light radius calculation not yet implemented for "+
                                  "Atmospheric PSF objects (could be though).")


class Airy(GSObject):
    """A class describing Airy PSF profiles.  Has an SBAiry in the SBProfile attribute.

    For more details of the Airy Surface Brightness profile, please see the SBAiry documentation
    produced by doxygen, or refer to http://en.wikipedia.org/wiki/Airy_disc.

    Initialization
    --------------
    An Airy can be initialized using one size parameter `lam_over_diam`, an optional `obscuration`
    parameter [default `obscuration=0.`] and an optional flux parameter [default `flux = 1.`].  The
    half light radius or FWHM can subsequently be calculated using the getHalfLightRadius() method
    or getFWHM(), respectively, if `obscuration = 0.`

    Example:
    
        >>> airy_obj = galsim.Airy(flux=3., lam_over_diam=2.)
        >>> airy_obj.getHalfLightRadius()
        1.0696642954485294

    Attempting to initialize with more than one size parameter is ambiguous, and will raise a
    TypeError exception.

    You may also specify a gsparams argument.  See the docstring for galsim.GSParams using
    help(galsim.GSParams) for more information about this option.

    Methods
    -------
    The Airy is a GSObject, and inherits all of the GSObject methods (draw(), drawShoot(), 
    applyShear() etc.) and operator bindings.
    """
    
    # Initialization parameters of the object, with type information
    _req_params = { "lam_over_diam" : float }
    _opt_params = { "flux" : float , "obscuration" : float }
    _single_params = []
    _takes_rng = False

    # --- Public Class methods ---
    def __init__(self, lam_over_diam, obscuration=0., flux=1., gsparams=None):
        GSObject.__init__(
            self, galsim.SBAiry(lam_over_diam=lam_over_diam, obscuration=obscuration, flux=flux,
                                gsparams=gsparams))

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
    """A class describing Kolmogorov PSF profiles.  Has an SBKolmogorov in the SBProfile attribute.
       
    Represents a long exposure Kolmogorov PSF.  For more information, refer to 
    http://en.wikipedia.org/wiki/Atmospheric_seeing#The_Kolmogorov_model_of_turbulence.

    Initialization
    --------------
    
    A Kolmogorov is initialized with one (and only one) of three possible size parameters

        lam_over_r0
        half_light_radius
        fwhm

    and an optional `flux` parameter [default `flux = 1`].

    Example:

        >>> psf = galsim.Kolmogorov(lam_over_r0=3., flux=1.)

    Initializes psf as a galsim.Kolmogorov() instance.

    @param lam_over_r0        lambda / r0 in the physical units adopted (user responsible for 
                              consistency), where r0 is the Fried parameter. The FWHM of the
                              Kolmogorov PSF is ~0.976 lambda/r0 (e.g., Racine 1996, PASP 699, 108).
                              Typical values for the Fried parameter are on the order of 10cm for
                              most observatories and up to 20cm for excellent sites. The values are
                              usually quoted at lambda = 500nm and r0 depends on wavelength as
                              [r0 ~ lambda^(-6/5)].
    @param fwhm               FWHM of the Kolmogorov PSF.
    @param half_light_radius  Half-light radius of the Kolmogorov PSF.
                              One of `lam_over_r0`, `fwhm` and `half_light_radius` (and only one) 
                              must be specified.
    @param flux               Optional flux value [default `flux = 1.`].
    @param gsparams           You may also specify a gsparams argument.  See the docstring for
                              galsim.GSParams using help(galsim.GSParams) for more information about
                              this option.
    
    Methods
    -------
    The Kolmogorov is a GSObject, and inherits all of the GSObject methods (draw(), drawShoot(), 
    applyShear() etc.) and operator bindings.

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

        GSObject.__init__(self, galsim.SBKolmogorov(lam_over_r0=lam_over_r0, flux=flux,
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


class OpticalPSF(GSObject):
    """A class describing aberrated PSFs due to telescope optics.  Has an SBInterpolatedImage in the
    SBProfile attribute.

    Input aberration coefficients are assumed to be supplied in units of wavelength, and correspond
    to the Zernike polynomials in the Noll convention definined in
    Noll, J. Opt. Soc. Am. 66, 207-211(1976).  For a brief summary of the polynomials, refer to
    http://en.wikipedia.org/wiki/Zernike_polynomials#Zernike_polynomials.

    Initialization
    --------------
    
        >>> optical_psf = galsim.OpticalPSF(lam_over_diam, defocus=0., astig1=0., astig2=0.,
                                            coma1=0., coma2=0., trefoil1=0., trefoil2=0., spher=0.,
                                            circular_pupil=True, obscuration=0., interpolant=None,
                                            oversampling=1.5, pad_factor=1.5)

    Initializes optical_psf as a galsim.OpticalPSF() instance.

    @param lam_over_diam   lambda / telescope diameter in the physical units adopted for dx 
                           (user responsible for consistency).
    @param defocus         defocus in units of incident light wavelength.
    @param astig1          astigmatism (like e2) in units of incident light wavelength.
    @param astig2          astigmatism (like e1) in units of incident light wavelength.
    @param coma1           coma along y in units of incident light wavelength.
    @param coma2           coma along x in units of incident light wavelength.
    @param trefoil1        trefoil (one of the arrows along y) in units of incident light
                           wavelength.
    @param trefoil2        trefoil (one of the arrows along x) in units of incident light
                           wavelength.
    @param spher           spherical aberration in units of incident light wavelength.
    @param circular_pupil  adopt a circular pupil?  [default `circular_pupil = True`]
    @param obscuration     linear dimension of central obscuration as fraction of pupil linear
                           dimension, [0., 1.)
    @param interpolant     Either an Interpolant2d (or Interpolant) instance or a string indicating
                           which interpolant should be used.  Options are 'nearest', 'sinc', 
                           'linear', 'cubic', 'quintic', or 'lanczosN' where N should be the 
                           integer order to use. [default `interpolant = galsim.Quintic()`]
    @param oversampling    Optional oversampling factor for the SBInterpolatedImage table 
                           [default `oversampling = 1.5`], setting oversampling < 1 will produce 
                           aliasing in the PSF (not good).
    @param pad_factor      Additional multiple by which to zero-pad the PSF image to avoid folding
                           compared to what would be employed for a simple galsim.Airy 
                           [default `pad_factor = 1.5`].  Note that `pad_factor` may need to be 
                           increased for stronger aberrations, i.e. those larger than order unity.
    @param flux            Total flux of the profile [default `flux=1.`].
    @param gsparams        You may also specify a gsparams argument.  See the docstring for
                           galsim.GSParams using help(galsim.GSParams) for more information about
                           this option.
     
    Methods
    -------
    The OpticalPSF is a GSObject, and inherits all of the GSObject methods (draw(), drawShoot(), 
    applyShear() etc.) and operator bindings.
    """

    # Initialization parameters of the object, with type information
    _req_params = { "lam_over_diam" : float }
    _opt_params = {
        "defocus" : float ,
        "astig1" : float ,
        "astig2" : float ,
        "coma1" : float ,
        "coma2" : float ,
        "trefoil1" : float ,
        "trefoil2" : float ,
        "spher" : float ,
        "circular_pupil" : bool ,
        "obscuration" : float ,
        "oversampling" : float ,
        "pad_factor" : float ,
        "interpolant" : str ,
        "flux" : float }
    _single_params = []
    _takes_rng = False

    # --- Public Class methods ---
    def __init__(self, lam_over_diam, defocus=0.,
                 astig1=0., astig2=0., coma1=0., coma2=0., trefoil1=0., trefoil2=0., spher=0., 
                 circular_pupil=True, obscuration=0., interpolant=None, oversampling=1.5,
                 pad_factor=1.5, flux=1., gsparams=None):

        # Currently we load optics, noise etc in galsim/__init__.py, but this might change (???)
        import galsim.optics
        
        # Choose dx for lookup table using Nyquist for optical aperture and the specified
        # oversampling factor
        dx_lookup = .5 * lam_over_diam / oversampling
        
        # We need alias_threshold here, so don't wait to make this a default GSParams instance
        # if the user didn't specify anything else.
        if not gsparams:
            gsparams = galsim.GSParams()

        # Use a similar prescription as SBAiry to set Airy stepK and thus reference unpadded image
        # size in physical units
        stepk_airy = min(
            gsparams.alias_threshold * .5 * np.pi**3 * (1. - obscuration) / lam_over_diam,
            np.pi / 5. / lam_over_diam)
        
        # Boost Airy image size by a user-specifed pad_factor to allow for larger, aberrated PSFs,
        # also make npix always *odd* so that opticalPSF lookup table array is correctly centred:
        npix = 1 + 2 * (np.ceil(pad_factor * (np.pi / stepk_airy) / dx_lookup)).astype(int)
        
        # Make the psf image using this dx and array shape
        optimage = galsim.optics.psf_image(
            lam_over_diam=lam_over_diam, dx=dx_lookup, array_shape=(npix, npix), defocus=defocus,
            astig1=astig1, astig2=astig2, coma1=coma1, coma2=coma2, trefoil1=trefoil1,
            trefoil2=trefoil2, spher=spher, circular_pupil=circular_pupil, obscuration=obscuration,
            flux=flux)
        
        # If interpolant not specified on input, use a Quintic interpolant
        if interpolant is None:
            quintic = galsim.Quintic(tol=1e-4)
            self.interpolant = galsim.InterpolantXY(quintic)
        else:
            self.interpolant = galsim.utilities.convert_interpolant_to_2d(interpolant)

        # Initialize the SBProfile
        GSObject.__init__(
            self, galsim.SBInterpolatedImage(optimage, xInterp=self.interpolant, dx=dx_lookup,
                                             gsparams=gsparams))

        # The above procedure ends up with a larger image than we really need, which
        # means that the default stepK value will be smaller than we need.  
        # Thus, we call the function calculateStepK() to refine the value.
        self.SBProfile.calculateStepK()
        self.SBProfile.calculateMaxK()

class InterpolatedImage(GSObject):
    """A class describing non-parametric objects specified using an Image, which can be interpolated
    for the purpose of carrying out transformations.

    The input Image and optional interpolants are used to create an SBInterpolatedImage.  The
    InterpolatedImage class is useful if you have a non-parametric description of an object as an
    Image, that you wish to manipulate / transform using GSObject methods such as applyShear(),
    applyMagnification(), applyShift(), etc.  The input Image can be any BaseImage (i.e., Image,
    ImageView, or ConstImageView).

    The constructor needs to know how the Image was drawn: is it an Image of flux or of surface
    brightness?  Since our default for drawing Images using draw() and drawShoot() is that
    `normalization == 'flux'` (i.e., sum of pixel values equals the object flux), the default for 
    the InterpolatedImage class is to assume the same flux normalization.  However, the user can 
    specify 'surface brightness' normalization if desired, or alternatively, can instead specify 
    the desired flux for the object.

    If the input Image has a scale associated with it, then there is no need to specify an input
    scale `dx`.

    The user may optionally specify an interpolant, `x_interpolant`, for real-space manipulations
    (e.g., shearing, resampling).  If none is specified, then by default, a 5th order Lanczos
    interpolant is used.  The user may also choose to specify two quantities that can affect the
    Fourier space convolution: the k-space interpolant (`k_interpolant`) and the amount of padding
    to include around the original images (`pad_factor`).  The default values for `x_interpolant`,
    `k_interpolant`, and `pad_factor` were chosen based on preliminary tests suggesting that they
    lead to a high degree of accuracy without being excessively slow.  Users should be particularly
    wary about changing `k_interpolant` and `pad_factor` from the defaults without careful testing.
    The user is given complete freedom to choose interpolants and pad factors, and no warnings are
    raised when the code is modified to choose some combination that is known to give significant
    error.  More details can be found in devel/modules/finterp.pdf, especially table 1, in the
    GalSim repository.

    The user can choose to have the image padding use zero (default), Gaussian random noise of some
    variance, or a Gaussian but correlated noise field that is specified either as a 
    CorrelatedNoise instance, an Image (from which a correlated noise model is derived), or a string
    (interpreted as a filename containing an image to use for deriving a CorrelatedNoise).  The user
    can also pass in a random number generator to be used for noise generation.  Finally, the user
    can pass in a `pad_image` for deterministic image padding.

    By default, the InterpolatedImage recalculates the Fourier-space step and number of points to
    use for further manipulations, rather than using the most conservative possibility.  For typical
    objects representing galaxies and PSFs this can easily make the difference between several
    seconds (conservative) and 0.04s (recalculated).  However, the user can turn off this option,
    and may especially wish to do so when using images that do not contain a high S/N object - e.g.,
    images of noise fields.

    Initialization
    --------------
    
        >>> interpolated_image = galsim.InterpolatedImage(image, x_interpolant = None,
                                                          k_interpolant = None,
                                                          normalization = 'f', dx = None,
                                                          flux = None, pad_factor = 0.,
                                                          noise_pad = 0., rng = None,
                                                          pad_image = None,
                                                          calculate_stepk = True,
                                                          calculate_maxk = True,
                                                          use_cache = True)

    Initializes interpolated_image as a galsim.InterpolatedImage() instance.

    For comparison of the case of padding with noise or zero when the image itself includes noise,
    compare `im1` and `im2` from the following code snippet (which can be executed from the
    examples/ directory):

        image = galsim.fits.read('data/147246.0_150.416558_1.998697_masknoise.fits')
        int_im1 = galsim.InterpolatedImage(image)
        int_im2 = galsim.InterpolatedImage(image, noise_pad='../tests/blankimg.fits')
        im1 = galsim.ImageF(1000,1000)
        im2 = galsim.ImageF(1000,1000)
        int_im1.draw(im1)
        int_im2.draw(im2)

    Examination of these two images clearly shows how padding with a correlated noise field that is
    similar to the one in the real data leads to a more reasonable appearance for the result when
    re-drawn at a different size.

    @param image           The Image from which to construct the object.
                           This may be either an Image (or ImageView) instance or a string
                           indicating a fits file from which to read the image.
    @param x_interpolant   Either an Interpolant2d (or Interpolant) instance or a string indicating
                           which real-space interpolant should be used.  Options are 'nearest',
                           'sinc', 'linear', 'cubic', 'quintic', or 'lanczosN' where N should be the
                           integer order to use. (Default `x_interpolant = galsim.Quintic()`)
    @param k_interpolant   Either an Interpolant2d (or Interpolant) instance or a string indicating
                           which k-space interpolant should be used.  Options are 'nearest', 'sinc',
                           'linear', 'cubic', 'quintic', or 'lanczosN' where N should be the integer
                           order to use.  We strongly recommend leaving this parameter at its
                           default value; see text above for details.  (Default `k_interpolant =
                           galsim.Quintic()`)
    @param normalization   Two options for specifying the normalization of the input Image:
                              "flux" or "f" means that the sum of the pixels is normalized
                                  to be equal to the total flux.
                              "surface brightness" or "sb" means that the pixels sample
                                  the surface brightness distribution at each location.
                              (Default `normalization = "flux"`)
    @param dx              If provided, use this as the pixel scale for the Image; this will
                           override the pixel scale stored by the provided Image, in any.  If `dx`
                           is `None`, then take the provided image's pixel scale.
                           (Default `dx = None`.)
    @param flux            Optionally specify a total flux for the object, which overrides the
                           implied flux normalization from the Image itself.
    @param pad_factor      Factor by which to pad the Image when creating the SBInterpolatedImage;
                           `pad_factor <= 0` results in the use of the default value, 4.  We
                           strongly recommend leaving this parameter at its default value; see text
                           above for details.
                           (Default `pad_factor = 0`, unless a `pad_image` is passed in, which
                           results in a default value of `pad_factor = 1`.)
    @param noise_pad       Noise properties to use when padding the original image with
                           noise.  This can be specified in several ways:
                               (a) as a float, which is interpreted as being a variance to use when
                                   padding with uncorrelated Gaussian noise; 
                               (b) as a galsim.CorrelatedNoise, which contains information about the
                                   desired noise power spectrum - any random number generator passed
                                   to the `rng` keyword will take precedence over that carried in an
                                   input galsim.CorrelatedNoise;
                               (c) as a galsim.Image of a noise field, which is used to calculate
                                   the desired noise power spectrum; or
                               (d) as a string which is interpreted as a filename containing an
                                   example noise field with the proper noise power spectrum.
                           It is important to keep in mind that the calculation of the correlation
                           function that is internally stored within a galsim.CorrelatedNoise is a 
                           non-negligible amount of overhead, so the recommended means of specifying
                           a correlated noise field for padding are (b) or (d). In the case of (d),
                           if the same file is used repeatedly, then the `use_cache` keyword (see 
                           below) can be used to prevent the need for repeated 
                           galsim.CorrelatedNoise initializations.
                           (Default `noise_pad = 0.`, i.e., pad with zeros.)
    @param rng             If padding by noise, the user can optionally supply the random noise
                           generator to use for drawing random numbers as `rng` (may be any kind of
                           `galsim.BaseDeviate` object).  Such a user-input random number generator
                           takes precedence over any stored within a user-input CorrelatedNoise 
                           instance (see the `noise_pad` param).
                           If `rng=None`, one will be automatically created, using the time as a
                           seed. (Default `rng = None`)
    @param pad_image       Image to be used for deterministically padding the original image.  This
                           can be specified in two ways:
                               (a) as a galsim.Image; or
                               (b) as a string which is interpreted as a filename containing an
                                   image to use.
                           The size of the image that is passed in is taken to specify the amount of
                           padding, and so the `pad_factor` keyword should be equal to 1, i.e., no
                           padding.  The `pad_image` scale is ignored, and taken to be equal to that
                           of the `image`. Note that `pad_image` can be used together with
                           `noise_pad`.  However, the user should be careful to ensure that the
                           image used for padding has roughly zero mean.  The purpose of this
                           keyword is to allow for a more flexible representation of some noise
                           field around an object; if the user wishes to represent the sky level
                           around an object, they should do that when they have drawn the final
                           image instead.  (Default `pad_image = None`.)
    @param calculate_stepk Specify whether to perform an internal determination of the extent of 
                           the object being represented by the InterpolatedImage; often this is 
                           useful in choosing an optimal value for the stepsize in the Fourier 
                           space lookup table. (Default `calculate_stepk = True`)
    @param calculate_maxk  Specify whether to perform an internal determination of the highest 
                           spatial frequency needed to accurately render the object being 
                           represented by the InterpolatedImage; often this is useful in choosing 
                           an optimal value for the extent of the Fourier space lookup table.
                           (Default `calculate_maxk = True`)
    @param use_cache       Specify whether to cache noise_pad read in from a file to save having
                           to build a CorrelatedNoise object repeatedly from the same image.
                           (Default `use_cache = True`)
    @param use_true_center Similar to the same parameter in the GSObject.draw function, this
                           sets whether to use the true center of the provided image as the 
                           center of the profile (if `use_true_center=True`) or the nominal
                           center returned by `image.bounds.center()` (if `use_true_center=False`)
                           [default `use_true_center = True`]
    @param gsparams        You may also specify a gsparams argument.  See the docstring for
                           galsim.GSParams using help(galsim.GSParams) for more information about
                           this option.

    Methods
    -------
    The InterpolatedImage is a GSObject, and inherits all of the GSObject methods (draw(),
    drawShoot(), applyShear() etc.) and operator bindings.
    """

    # Initialization parameters of the object, with type information
    _req_params = { 'image' : str }
    _opt_params = {
        'x_interpolant' : str ,
        'k_interpolant' : str ,
        'normalization' : str ,
        'dx' : float ,
        'flux' : float ,
        'pad_factor' : float ,
        'noise_pad' : str ,
        'pad_image' : str ,
        'calculate_stepk' : bool ,
        'calculate_maxk' : bool,
        'use_true_center' : bool
    }
    _single_params = []
    _takes_rng = True
    _cache_noise_pad = {}

    # --- Public Class methods ---
    def __init__(self, image, x_interpolant = None, k_interpolant = None, normalization = 'flux',
                 dx = None, flux = None, pad_factor = 0., noise_pad = 0., rng = None,
                 pad_image = None, calculate_stepk=True, calculate_maxk=True,
                 use_cache=True, use_true_center=True, gsparams=None):
        # first try to read the image as a file.  If it's not either a string or a valid
        # pyfits hdu or hdulist, then an exception will be raised, which we ignore and move on.
        try:
            image = galsim.fits.read(image)
        except:
            pass

        # make sure image is really an image and has a float type
        if not isinstance(image, galsim.BaseImageF) and not isinstance(image, galsim.BaseImageD):
            raise ValueError("Supplied image is not an image of floats or doubles!")

        # it must have well-defined bounds, otherwise seg fault in SBInterpolatedImage constructor
        if not image.getBounds().isDefined():
            raise ValueError("Supplied image does not have bounds defined!")

        # check what normalization was specified for the image: is it an image of surface
        # brightness, or flux?
        if not normalization.lower() in ("flux", "f", "surface brightness", "sb"):
            raise ValueError(("Invalid normalization requested: '%s'. Expecting one of 'flux', "+
                              "'f', 'surface brightness', or 'sb'.") % normalization)

        # set up the interpolants if none was provided by user, or check that the user-provided ones
        # are of a valid type
        if x_interpolant is None:
            self.x_interpolant = galsim.InterpolantXY(galsim.Quintic(tol=1e-4))
        else:
            self.x_interpolant = galsim.utilities.convert_interpolant_to_2d(x_interpolant)
        if k_interpolant is None:
            self.k_interpolant = galsim.InterpolantXY(galsim.Quintic(tol=1e-4))
        else:
            self.k_interpolant = galsim.utilities.convert_interpolant_to_2d(k_interpolant)

        # Check for input dx, and check whether Image already has one set.  At the end of this
        # code block, either an exception will have been raised, or the input image will have a
        # valid scale set.
        if dx is None:
            dx = image.getScale()
            if dx == 0:
                raise ValueError("No information given with Image or keywords about pixel scale!")
        else:
            if type(dx) != float:
                dx = float(dx)
            # Don't change the original image.  Make a new view if we need to set the scale.
            image = image.view()
            image.setScale(dx)
            if dx == 0.0:
                raise ValueError("dx may not be 0.0")

        # Set up the GaussianDeviate if not provided one, or check that the user-provided one is
        # of a valid type.
        if rng is None:
            gaussian_deviate = galsim.GaussianDeviate()
        elif isinstance(rng, galsim.BaseDeviate):
            # Even if it's already a GaussianDeviate, we still want to make a new Gaussian deviate
            # that would generate the same sequence, because later we change the sigma and we don't
            # want to change it for the original one that was passed in.  So don't distinguish
            # between GaussianDeviate and the other BaseDeviates here.
            gaussian_deviate = galsim.GaussianDeviate(rng)
        else:
            raise TypeError("rng provided to InterpolatedImage constructor is not a BaseDeviate")

        # decide about deterministic image padding
        specify_size = False
        padded_size = image.getPaddedSize(pad_factor)
        if pad_image:
            specify_size = True
            if isinstance(pad_image, str):
                pad_image = galsim.fits.read(pad_image)
            if ( not isinstance(pad_image, galsim.BaseImageF) and 
                 not isinstance(pad_image, galsim.BaseImageD) ):
                raise ValueError("Supplied pad_image is not one of the allowed types!")

            # If an image was supplied directly or from a file, check its size:
            #    Cannot use if too small.
            #    Use to define the final image size otherwise.
            deltax = (1+pad_image.getXMax()-pad_image.getXMin())-(1+image.getXMax()-image.getXMin())
            deltay = (1+pad_image.getYMax()-pad_image.getYMin())-(1+image.getYMax()-image.getYMin())
            if deltax < 0 or deltay < 0:
                raise RuntimeError("Image supplied for padding is too small!")
            if pad_factor != 1. and pad_factor != 0.:
                import warnings
                msg =  "Warning: ignoring specified pad_factor because user also specified\n"
                msg += "         an image to use directly for the padding."
                warnings.warn(msg)
        elif noise_pad:
            if isinstance(image, galsim.BaseImageF):
                pad_image = galsim.ImageF(padded_size, padded_size)
            if isinstance(image, galsim.BaseImageD):
                pad_image = galsim.ImageD(padded_size, padded_size)

        # now decide about noise padding
        # First, see if the input is consistent with a float.
        # i.e. it could be an int, or a str that converts to a number.
        try:
            noise_pad = float(noise_pad)
        except:
            pass
        if isinstance(noise_pad, float):
            if noise_pad < 0.:
                raise ValueError("Noise variance cannot be negative!")
            elif noise_pad > 0.:
                # Note: make sure the sigma is properly set to sqrt(noise_pad).
                gaussian_deviate.setSigma(np.sqrt(noise_pad))
                pad_image.addNoise(galsim.DeviateNoise(gaussian_deviate))
        else:
            if isinstance(noise_pad, galsim.correlatednoise._BaseCorrelatedNoise):
                cn = noise_pad.copy()
                if rng: # Let a user supplied RNG take precedence over that in user CN
                    cn.setRNG(gaussian_deviate)
            elif isinstance(noise_pad,galsim.BaseImageF) or isinstance(noise_pad,galsim.BaseImageD):
                cn = galsim.CorrelatedNoise(gaussian_deviate, noise_pad)
            elif use_cache and noise_pad in InterpolatedImage._cache_noise_pad:
                cn = InterpolatedImage._cache_noise_pad[noise_pad]
                if rng:
                    # Make sure that we are using a specified RNG by resetting that in this cached
                    # CorrelatedNoise instance, otherwise preserve the cached RNG
                    cn.setRNG(gaussian_deviate)
            elif isinstance(noise_pad, str):
                cn = galsim.CorrelatedNoise(gaussian_deviate, galsim.fits.read(noise_pad))
                if use_cache: 
                    InterpolatedImage._cache_noise_pad[noise_pad] = cn
            else:
                raise ValueError(
                    "Input noise_pad must be a float/int, a CorrelatedNoise, Image, or filename "+
                    "containing an image to use to make a CorrelatedNoise!")
            pad_image.addNoise(cn)

        # Now we have to check: was the padding determined using pad_factor?  Or by passing in an
        # image for padding?  Treat these cases differently:
        # (1) If the former, then we can simply have the C++ handle the padding process.
        # (2) If the latter, then we have to do the padding ourselves, and pass the resulting image
        # to the C++ with pad_factor explicitly set to 1.
        if specify_size is False:
            # Make the SBInterpolatedImage out of the image.
            sbinterpolatedimage = galsim.SBInterpolatedImage(
                    image, xInterp=self.x_interpolant, kInterp=self.k_interpolant,
                    dx=dx, pad_factor=pad_factor, pad_image=pad_image, gsparams=gsparams)
            self.x_size = padded_size
            self.y_size = padded_size
        else:
            # Leave the original image as-is.  Instead, we shift around the image to be used for
            # padding.  Find out how much x and y margin there should be on lower end:
            x_marg = int(np.round(0.5*deltax))
            y_marg = int(np.round(0.5*deltay))
            # Now reset the pad_image to contain the original image in an even way
            pad_image = pad_image.view()
            pad_image.setScale(dx)
            pad_image.setOrigin(image.getXMin()-x_marg, image.getYMin()-y_marg)
            # Set the central values of pad_image to be equal to the input image
            pad_image[image.bounds] = image
            sbinterpolatedimage = galsim.SBInterpolatedImage(
                    pad_image, xInterp=self.x_interpolant, kInterp=self.k_interpolant,
                    dx=dx, pad_factor=1., gsparams=gsparams)
            self.x_size = 1+pad_image.getXMax()-pad_image.getXMin()
            self.y_size = 1+pad_image.getYMax()-pad_image.getYMin()

        # GalSim cannot automatically know what stepK and maxK are appropriate for the 
        # input image.  So it is usually worth it to do a manual calculation here.
        if calculate_stepk:
            sbinterpolatedimage.calculateStepK()
        if calculate_maxk:
            sbinterpolatedimage.calculateMaxK()

        # If the user specified a flux, then set to that flux value.
        if flux != None:
            if type(flux) != flux:
                flux = float(flux)
            sbinterpolatedimage.setFlux(flux)
        # If the user specified a flux normalization for the input Image, then since
        # SBInterpolatedImage works in terms of surface brightness, have to rescale the values to
        # get proper normalization.
        elif flux is None and normalization.lower() in ['flux','f'] and dx != 1.:
            sbinterpolatedimage.scaleFlux(1./(dx**2))
        # If the input Image normalization is 'sb' then since that is the SBInterpolated default
        # assumption, no rescaling is needed.

        # Initialize the SBProfile
        GSObject.__init__(self, sbinterpolatedimage)

        # Fix the center to be in the right place.
        # Note the minus sign in front of image.scale, since we want to fix the center in the 
        # opposite sense of what the draw function does.
        if use_true_center:
            prof = self._fix_center(image, -image.scale)
            GSObject.__init__(self, prof.SBProfile)
            


class Pixel(GSObject):
    """A class describing pixels.  Has an SBBox in the SBProfile attribute.

    This class is typically used to represent a Pixel response function, and therefore is only
    needed when drawing images using Fourier transform or real-space convolution (with the draw
    method), not when using photon-shooting (with the drawShoot method).

    Initialization
    --------------
    A Pixel is initialized with an x dimension width `xw`, an optional y dimension width (if
    unspecifed `yw=xw` is assumed) and an optional flux parameter [default `flux = 1.`].

    You may also specify a gsparams argument.  See the docstring for galsim.GSParams using
    help(galsim.GSParams) for more information about this option.

    Methods
    -------
    The Pixel is a GSObject, and inherits all of the GSObject methods (draw(), drawShoot(), 
    applyShear() etc.) and operator bindings.

    Note: We have not implemented drawing a sheared or rotated Pixel in real space.  It's a 
          bit tricky to get right at the edges where fractional fluxes are required.  
          Fortunately, this is almost never needed.  Pixels are almost always convolved by
          something else rather than drawn by themselves, in which case either the fourier
          space method is used, or photon shooting.  Both of these are implemented in GalSim.
          If need to draw sheared or rotated Pixels in real space, please file an issue, and
          maybe we'll implement that function.  Until then, you will get an exception if you try.
    """

    # Initialization parameters of the object, with type information
    _req_params = { "xw" : float }
    _opt_params = { "yw" : float , "flux" : float }
    _single_params = []
    _takes_rng = False

    # --- Public Class methods ---
    def __init__(self, xw, yw=None, flux=1., gsparams=None):
        if yw is None:
            yw = xw
        GSObject.__init__(self, galsim.SBBox(xw=xw, yw=yw, flux=flux, gsparams=gsparams))

    def getXWidth(self):
        """Return the width of the pixel in the x dimension.
        """
        return self.SBProfile.getXWidth()

    def getYWidth(self):
        """Return the width of the pixel in the y dimension.
        """
        return self.SBProfile.getYWidth()


class Sersic(GSObject):
    """A class describing Sersic profile objects.  Has an SBSersic in the SBProfile attribute.

    For more details of the Sersic Surface Brightness profile, please see the SBSersic documentation
    produced by doxygen, or refer to http://en.wikipedia.org/wiki/Sersic_profile.

    Initialization
    --------------
    A Sersic is initialized with `n`, the Sersic index of the profile, and the half light radius 
    size parameter `half_light_radius`.  Optional parameters are truncation radius `trunc` [default
    `trunc = 0.`, indicating no truncation] and a `flux` parameter [default `flux = 1`].  If `trunc`
    is set to a non-zero value, then it is assumed to be in the same system of units as
    `half_light_radius`.

    Note that the code will be more efficient if the truncation is always the same multiple of
    `half_light_radius`, since it caches many calculations that depend on the ratio
    `trunc/half_light_radius`.

    Example:

        >>> sersic_obj = galsim.Sersic(n=3.5, half_light_radius=2.5, flux=40.)
        >>> sersic_obj.getHalfLightRadius()
        2.5
        >>> sersic_obj.getN()
        3.5

    Another optional parameter, `flux_untruncated`, specifies whether the `flux` and
    `half_light_radius` correspond to the untruncated profile or the truncated profile. If
    `flux_untruncated` is True (and `trunc > 0` of course), then the profile will be identical
    to the version without truncation up to the truncation radius, at which point it drops to 0.
    If `flux_untruncated` is False (the default), then the scale radius will be larger and the
    central peak will be higher than the untruncated profile, in order to maintain the correct
    provided `flux` and `half_light_radius`.

    When `trunc > 0.` and `flux_untruncated == True`, the actual half-light radius will be different
    from the specified half-light radius.  The getHalfLightRadius() method will return the true
    half-light radius.  Similarly, the actual flux will not be the same as the specified value; the
    true flux is also returned by the getFlux() method.

    Example:

        >>> sersic_obj = galsim.Sersic(n=3.5, half_light_radius=2.5, flux=40.)
        >>> sersic_obj2 = galsim.Sersic(n=3.5, half_light_radius=2.5, flux=40., trunc=10., \\
                                        flux_untruncated=True)
        >>> sersic_obj.xValue(galsim.PositionD(0.,0.))
        237.3094228614579
        >>> sersic_obj2.xValue(galsim.PositionD(0.,0.))
        237.3094228614579     # The xValues are the same inside the truncation radius ...
        >>> sersic_obj.xValue(galsim.PositionD(10.,0.))
        0.011776164687306839
        >>> sersic_obj2.xValue(galsim.PositionD(10.,0.))
        0.0                   # ... but different outside the truncation radius
        >>> sersic_obj.getHalfLightRadius()
        2.5
        >>> sersic_obj2.getHalfLightRadius()
        1.9795101421751533    # The true half-light radius is smaller than the specified value
        >>> sersic_obj.getFlux()
        40.0
        >>> sersic_obj2.getFlux()
        34.56595186009358     # Flux is missing due to truncation

    You may also specify a gsparams argument.  See the docstring for galsim.GSParams using
    help(galsim.GSParams) for more information about this option.

    Methods
    -------
    The Sersic is a GSObject, and inherits all of the GSObject methods (draw(), drawShoot(),
    applyShear() etc.) and operator bindings.
    """

    # Initialization parameters of the object, with type information
    _req_params = { "n" : float , "half_light_radius" : float }
    _opt_params = { "flux" : float, "trunc": float, "flux_untruncated" : bool }
    _single_params = []
    _takes_rng = False

    # --- Public Class methods ---
    def __init__(self, n, half_light_radius, flux=1., trunc=0., flux_untruncated=False,
                 gsparams=None):
        GSObject.__init__(
            self, galsim.SBSersic(n, half_light_radius=half_light_radius, flux=flux,
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


class Exponential(GSObject):
    """A class describing exponential profile objects.  Has an SBExponential in the SBProfile 
    attribute.

    For more details of the Exponential Surface Brightness profile, please see the SBExponential
    documentation produced by doxygen.

    Initialization
    --------------
    An Exponential can be initialized using one (and only one) of two possible size parameters

        half_light_radius
        scale_radius

    and an optional `flux` parameter [default `flux = 1.`].

    Example:

        >>> exp_obj = galsim.Exponential(flux=3., scale_radius=5.)
        >>> exp_obj.getHalfLightRadius()
        8.391734950083302
        >>> exp_obj = galsim.Exponential(flux=3., half_light_radius=1.)
        >>> exp_obj.getScaleRadius()
        0.5958243473776976

    Attempting to initialize with more than one size parameter is ambiguous, and will raise a
    TypeError exception.

    You may also specify a gsparams argument.  See the docstring for galsim.GSParams using
    help(galsim.GSParams) for more information about this option.

    Methods
    -------
    The Exponential is a GSObject, and inherits all of the GSObject methods (draw(), drawShoot(),
    applyShear() etc.) and operator bindings.
    """

    # Initialization parameters of the object, with type information
    _req_params = {}
    _opt_params = { "flux" : float }
    _single_params = [ { "scale_radius" : float , "half_light_radius" : float } ]
    _takes_rng = False

    # --- Public Class methods ---
    def __init__(self, half_light_radius=None, scale_radius=None, flux=1., gsparams=None):
        GSObject.__init__(
            self, galsim.SBExponential(
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
    """A class describing DeVaucouleurs profile objects.  Has an SBDeVaucouleurs in the SBProfile 
    attribute.

    For more details of the DeVaucouleurs Surface Brightness profile, please see the
    SBDeVaucouleurs documentation produced by doxygen, or refer to 
    http://en.wikipedia.org/wiki/De_Vaucouleurs'_law.

    Initialization
    --------------
    A DeVaucouleurs is initialized with the half light radius size parameter `half_light_radius`.
    Optional parameters are truncation radius `trunc` [default `trunc = 0.`, indicating no
    truncation] and a `flux` parameter [default `flux = 1.`].  If `trunc` is set to a non-zero
    value, then it is assumed to be in the same system of units as `half_light_radius`.

    Note that the code will be more efficient if the truncation is always the same multiple of
    `half_light_radius`, since it caches many calculations that depend on the ratio
    `trunc/half_light_radius`.

    Example:

        >>> dvc_obj = galsim.DeVaucouleurs(half_light_radius=2.5, flux=40.)
        >>> dvc_obj.getHalfLightRadius()
        2.5
        >>> dvc_obj.getFlux()
        40.0

    Another optional parameter, `flux_untruncated`, specifies whether the `flux` and
    `half_light_radius` correspond to the untruncated profile or the truncated profile. If
    `flux_untruncated` is True (and `trunc > 0` of course), then the profile will be identical
    to the version without truncation up to the truncation radius, at which point it drops to 0.
    If `flux_untruncated` is False (the default), then the scale radius will be larger and the
    central peak will be higher than the untruncated profile, in order to maintain the correct
    provided `flux` and `half_light_radius`.

    When `trunc > 0.` and `flux_untruncated == True`, the actual half-light radius will be different
    from the specified half-light radius.  The getHalfLightRadius() method will return the true
    half-light radius.  Similarly, the actual flux will not be the same as the specified value; the
    true flux is also returned by the getFlux() method.

    Example:

        >>> dvc_obj = galsim.DeVaucouleurs(half_light_radius=2.5, flux=40.)
        >>> dvc_obj2 = galsim.DeVaucouleurs(half_light_radius=2.5, flux=40., trunc=10., \\
                                            flux_untruncated=True)
        >>> dvc_obj.xValue(galsim.PositionD(0.,0.))
        604.6895805968326
        >>> dvc_obj2.xValue(galsim.PositionD(0.,0.))
        604.6895805968326     # The xValues are the same inside the truncation radius ...
        >>> dvc_obj.xValue(galsim.PositionD(10.0,0.))
        0.011781304853174116
        >>> dvc_obj2.xValue(galsim.PositionD(10.0,0.))
        0.0                   # ... but different outside the truncation radius
        >>> dvc_obj.getHalfLightRadius()
        2.5
        >>> dvc_obj2.getHalfLightRadius()
        1.886276579179012     # The true half-light radius is smaller than the specified value
        >>> dvc_obj.getFlux()
        40.0
        >>> dvc_obj2.getFlux()
        33.863171136492156    # The flux from the truncation is missing

    You may also specify a gsparams argument.  See the docstring for galsim.GSParams using
    help(galsim.GSParams) for more information about this option.

    Methods
    -------
    The DeVaucouleurs is a GSObject, and inherits all of the GSObject methods (draw(), drawShoot(),
    applyShear() etc.) and operator bindings.
    """

    # Initialization parameters of the object, with type information
    _req_params = { "half_light_radius" : float }
    _opt_params = { "flux" : float, "trunc" : float, "flux_untruncated" : float }
    _single_params = []
    _takes_rng = False

    # --- Public Class methods ---
    def __init__(self, half_light_radius=None, flux=1., trunc=0., flux_untruncated=False,
                 gsparams=None):
        GSObject.__init__(
            self, galsim.SBDeVaucouleurs(half_light_radius=half_light_radius, flux=flux,
                                         trunc=trunc, flux_untruncated=flux_untruncated,
                                         gsparams=gsparams))

    def getHalfLightRadius(self):
        """Return the half light radius for this DeVaucouleurs profile.
        """
        return self.SBProfile.getHalfLightRadius()

     
class RealGalaxy(GSObject):
    """A class describing real galaxies from some training dataset.  Has an SBConvolve in the
    SBProfile attribute.

    This class uses a catalog describing galaxies in some training data (for more details, see the
    RealGalaxyCatalog documentation) to read in data about realistic galaxies that can be used for
    simulations based on those galaxies.  Also included in the class is additional information that
    might be needed to make or interpret the simulations, e.g., the noise properties of the training
    data.

    The GSObject drawShoot method is unavailable for RealGalaxy instances.

    Initialization
    --------------
    
        real_galaxy = galsim.RealGalaxy(real_galaxy_catalog, index=None, id=None, random=False, 
                                        rng=None, x_interpolant=None, k_interpolant=None,
                                        flux=None, pad_factor = 0, noise_pad=False, pad_image=None,
                                        use_cache = True)

    This initializes real_galaxy with three SBInterpolatedImage objects (one for the deconvolved
    galaxy, and saved versions of the original HST image and PSF). Note that there are multiple
    keywords for choosing a galaxy; exactly one must be set.  In future we may add more such
    options, e.g., to choose at random but accounting for the non-constant weight factors
    (probabilities for objects to make it into the training sample).  Like other GSObjects, the
    RealGalaxy contains an SBProfile attribute which is an SBConvolve representing the deconvolved
    HST galaxy.

    Note that preliminary tests suggest that for optimal balance between accuracy and speed,
    `k_interpolant` and `pad_factor` should be kept at their default values.  The user should be
    aware that significant inaccuracy can result from using other combinations of these parameters;
    see devel/modules/finterp.pdf, especially table 1, in the GalSim repository.

    @param real_galaxy_catalog  A RealGalaxyCatalog object with basic information about where to
                                find the data, etc.
    @param index                Index of the desired galaxy in the catalog.
    @param id                   Object ID for the desired galaxy in the catalog.
    @param random               If true, then just select a completely random galaxy from the
                                catalog.
    @param rng                  A random number generator to use for selecting a random galaxy 
                                (may be any kind of BaseDeviate or None) and to use in generating
                                any noise field when padding.  This user-input random number
                                generator takes precedence over any stored within a user-input
                                CorrelatedNoise instance (see `noise_pad` param below).
    @param x_interpolant        Either an Interpolant2d (or Interpolant) instance or a string 
                                indicating which real-space interpolant should be used.  Options are 
                                'nearest', 'sinc', 'linear', 'cubic', 'quintic', or 'lanczosN' 
                                where N should be the integer order to use. [default 
                                `x_interpolant = galsim.Lanczos(5,...)'].
    @param k_interpolant        Either an Interpolant2d (or Interpolant) instance or a string 
                                indicating which k-space interpolant should be used.  Options are 
                                'nearest', 'sinc', 'linear', 'cubic', 'quintic', or 'lanczosN' 
                                where N should be the integer order to use.  We strongly recommend
                                leaving this parameter at its default value; see text above for
                                details.  [default `k_interpolant = galsim.Quintic()'].
    @param flux                 Total flux, if None then original flux in galaxy is adopted without
                                change [default `flux = None`].
    @param pad_factor           Factor by which to pad the Image when creating the
                                SBInterpolatedImage; `pad_factor <= 0` results in the use of the
                                default value, 4.  We strongly recommend leaving this parameter at
                                its default value; see text above for details.
                                [Default `pad_factor = 0`.]
    @param noise_pad            When creating the SBProfile attribute for this GSObject, pad the
                                SBInterpolated image with zeros, or with noise of a level specified
                                in the training dataset?  There are several options here: 
                                    Use `noise_pad = False` if you wish to pad with zeros.
                                    Use `noise_pad = True` if you wish to pad with uncorrelated
                                        noise of the proper variance.
                                    Set `noise_pad` equal to a galsim.CorrelatedNoise, an Image, or
                                        a filename containing an Image of an example noise field
                                        that will be used to calculate the noise power spectrum and
                                        generate noise in the padding region.  Any random number
                                        generator passed to the `rng` keyword will take precedence
                                        over that carried in an input galsim.CorrelatedNoise.
                                In the last case, if the same file is used repeatedly, then use of
                                the `use_cache` keyword (see below) can be used to prevent the need
                                for repeated galsim.CorrelatedNoise initializations.
                                [default `noise_pad = False`]
    @param pad_image            Image to be used for deterministically padding the original image.
                                This can be specified in two ways:
                                   (a) as a galsim.Image; or
                                   (b) as a string which is interpreted as a filename containing an
                                       image to use.
                               The size of the image that is passed in is taken to specify the
                                amount of padding, and so the `pad_factor` keyword should be equal
                                to 1, i.e., no padding.  The `pad_image` scale is ignored, and taken
                                to be equal to that of the `image`. Note that `pad_image` can be
                                used together with `noise_pad`.  However, the user should be careful
                                to ensure that the image used for padding has roughly zero mean.
                                The purpose of this keyword is to allow for a more flexible
                                representation of some noise field around an object; if the user
                                wishes to represent the sky level around an object, they should do
                                that when they have drawn the final image instead.  (Default
                                `pad_image = None`.)
    @param use_cache            Specify whether to cache noise_pad read in from a file to save
                                having to build an CorrelatedNoise repeatedly from the same image.
                                (Default `use_cache = True`)
    @param gsparams             You may also specify a gsparams argument.  See the docstring for
                                galsim.GSParams using help(galsim.GSParams) for more information
                                about this option.

    Methods
    -------
    The RealGalaxy is a GSObject, and inherits all of the GSObject methods (draw(), applyShear(), 
    etc. except drawShoot() which is unavailable), and operator bindings.
    """

    # Initialization parameters of the object, with type information
    _req_params = {}
    _opt_params = { "x_interpolant" : str ,
                    "k_interpolant" : str,
                    "flux" : float ,
                    "pad_factor" : float,
                    "noise_pad" : str,
                    "pad_image" : str}
    _single_params = [ { "index" : int , "id" : str } ]
    _takes_rng = True
    _cache_noise_pad = {}
    _cache_variance = {}

    # --- Public Class methods ---
    def __init__(self, real_galaxy_catalog, index=None, id=None, random=False,
                 rng=None, x_interpolant=None, k_interpolant=None, flux=None, pad_factor = 0,
                 noise_pad=False, pad_image=None, use_cache=True, gsparams=None):

        import pyfits

        # Code block below will be for galaxy selection; not all are currently implemented.  Each
        # option must return an index within the real_galaxy_catalog.        
        if index is not None:
            if id is not None or random is True:
                raise AttributeError('Too many methods for selecting a galaxy!')
            use_index = index
        elif id is not None:
            if random is True:
                raise AttributeError('Too many methods for selecting a galaxy!')
            use_index = real_galaxy_catalog._get_index_for_id(id)
        elif random is True:
            if rng is None:
                uniform_deviate = galsim.UniformDeviate()
            elif isinstance(rng, galsim.BaseDeviate):
                uniform_deviate = galsim.UniformDeviate(rng)
            else:
                raise TypeError("The rng provided to RealGalaxy constructor is not a BaseDeviate")
            use_index = int(real_galaxy_catalog.nobjects * uniform_deviate()) 
        else:
            raise AttributeError('No method specified for selecting a galaxy!')

        # read in the galaxy, PSF images; for now, rely on pyfits to make I/O errors. Should
        # consider exporting this code into fits.py in some function that takes a filename and HDU,
        # and returns an ImageView

        gal_image = real_galaxy_catalog.getGal(use_index)
        PSF_image = real_galaxy_catalog.getPSF(use_index)

        # choose proper interpolant
        if x_interpolant is None:
            lan5 = galsim.Lanczos(5, conserve_flux=True, tol=1.e-4)
            self.x_interpolant = galsim.InterpolantXY(lan5)
        else:
            self.x_interpolant = galsim.utilities.convert_interpolant_to_2d(x_interpolant)
        if k_interpolant is None:
            self.k_interpolant = galsim.InterpolantXY(galsim.Quintic(tol=1.e-4))
        else:
            self.k_interpolant = galsim.utilities.convert_interpolant_to_2d(k_interpolant)

        # read in data about galaxy from FITS binary table; store as normal attributes of RealGalaxy

        # save any other relevant information as instance attributes
        self.catalog_file = real_galaxy_catalog.file_name
        self.index = use_index
        self.pixel_scale = float(real_galaxy_catalog.pixel_scale[use_index])

        # handle padding by an image
        specify_size = False
        padded_size = gal_image.getPaddedSize(pad_factor)
        if pad_image is not None:
            specify_size = True
            if isinstance(pad_image,str):
                pad_image = galsim.fits.read(pad_image)
            if ( not isinstance(pad_image, galsim.BaseImageF) and 
                 not isinstance(pad_image, galsim.BaseImageD) ):
                raise ValueError("Supplied pad_image is not one of the allowed types!")
            # If an image was supplied directly or from a file, check its size:
            #    Cannot use if too small.
            #    Use to define the final image size otherwise.
            deltax = ((1 + pad_image.getXMax() - pad_image.getXMin()) - 
                      (1 + gal_image.getXMax() - gal_image.getXMin()))
            deltay = ((1 + pad_image.getYMax() - pad_image.getYMin()) - 
                      (1 + gal_image.getYMax() - gal_image.getYMin()))
            if deltax < 0 or deltay < 0:
                raise RuntimeError("Image supplied for padding is too small!")
            if pad_factor != 1. and pad_factor != 0.:
                import warnings
                msg =  "Warning: ignoring specified pad_factor because user also specified\n"
                msg += "         an image to use directly for the padding."
                warnings.warn(msg)
        else:
            if isinstance(gal_image, galsim.BaseImageF):
                pad_image = galsim.ImageF(padded_size, padded_size)
            if isinstance(gal_image, galsim.BaseImageD):
                pad_image = galsim.ImageD(padded_size, padded_size)

        # Set up the GaussianDeviate if not provided one, or check that the user-provided one
        # is of a valid type.  Note if random was selected, we can use that uniform_deviate safely.
        if random is True:
            gaussian_deviate = galsim.GaussianDeviate(uniform_deviate)
        else:
            if rng is None:
                gaussian_deviate = galsim.GaussianDeviate()
            elif isinstance(rng,galsim.BaseDeviate):
                # Even if it's already a GaussianDeviate, we still want to make a new Gaussian
                # deviate that would generate the same sequence, because later we change the sigma
                # and we don't want to change it for the original one that was passed in.  So don't
                # distinguish between GaussianDeviate and the other BaseDeviates here.
                gaussian_deviate = galsim.GaussianDeviate(rng)
            else:
                raise TypeError("rng provided to RealGalaxy constructor is not a BaseDeviate")

        # handle noise-padding options
        try:
            noise_pad = galsim.config.value._GetBoolValue(noise_pad,'')
        except:
            pass
        if noise_pad:
            self.pad_variance = float(real_galaxy_catalog.variance[use_index])

            # Check, is it "True" or something else?  If True, we use Gaussian uncorrelated noise
            # using the stored variance in the catalog.  Otherwise, if it's a CorrelatedNoise we use
            # it directly; if it's an Image of some sort we use it to make a CorrelatedNoise; if
            # it's a string, we read in the image from file and make a CorrelatedNoise.
            if type(noise_pad) is not bool:
                if isinstance(noise_pad, galsim.correlatednoise._BaseCorrelatedNoise):
                    cn = noise_pad.copy()
                    if rng: # Let user supplied RNG take precedence over that in user CN
                        cn.setRNG(gaussian_deviate)
                    # This small patch may have different overall variance, so rescale while
                    # preserving the correlation structure by default                  
                    cn.setVariance(self.pad_variance)
                elif (isinstance(noise_pad,galsim.BaseImageF) or 
                      isinstance(noise_pad,galsim.BaseImageD)):
                    cn = galsim.CorrelatedNoise(gaussian_deviate, noise_pad)
                elif use_cache and noise_pad in RealGalaxy._cache_noise_pad:
                    cn = RealGalaxy._cache_noise_pad[noise_pad]
                    # Make sure that we are using the desired RNG by resetting that in this cached
                    # CorrelatedNoise instance
                    if rng:
                        cn.setRNG(gaussian_deviate)
                    # This small patch may have different overall variance, so rescale while
                    # preserving the correlation structure
                    cn.setVariance(self.pad_variance)
                elif isinstance(noise_pad, str):
                    tmp_img = galsim.fits.read(noise_pad)
                    cn = galsim.CorrelatedNoise(gaussian_deviate, tmp_img)
                    if use_cache:
                        RealGalaxy._cache_noise_pad[noise_pad] = cn
                    # This small patch may have different overall variance, so rescale while
                    # preserving the correlation structure
                    cn.setVariance(self.pad_variance)
                else:
                    raise RuntimeError("noise_pad must be either a bool, CorrelatedNoise, Image, "+
                                       "or a filename for reading in an Image")

            # Set the GaussianDeviate sigma          
            gaussian_deviate.setSigma(np.sqrt(self.pad_variance))

            # populate padding image with noise field
            if type(noise_pad) is bool:
                pad_image.addNoise(galsim.DeviateNoise(gaussian_deviate))
            else:
                pad_image.addNoise(cn)
        else:
            self.pad_variance=0.

        # Now we have to check: was the padding determined using pad_factor?  Or by passing in an
        # image for padding?  Treat these cases differently:
        # (1) If the former, then we can simply have the C++ handle the padding process.
        # (2) If the latter, then we have to do the padding ourselves, and pass the resulting image
        # to the C++ with pad_factor explicitly set to 1.
        if specify_size is False:
            # Make the SBInterpolatedImage out of the image.
            self.original_image = galsim.SBInterpolatedImage(
                gal_image, xInterp=self.x_interpolant, kInterp=self.k_interpolant,
                dx=self.pixel_scale, pad_factor=pad_factor, pad_image=pad_image, gsparams=gsparams)
        else:
            # Leave the original image as-is.  Instead, we shift around the image to be used for
            # padding.  Find out how much x and y margin there should be on lower end:
            x_marg = int(np.round(0.5*deltax))
            y_marg = int(np.round(0.5*deltay))
            # Now reset the pad_image to contain the original image in an even way
            pad_image = pad_image.view()
            pad_image.setScale(self.pixel_scale)
            pad_image.setOrigin(gal_image.getXMin()-x_marg, gal_image.getYMin()-y_marg)
            # Set the central values of pad_image to be equal to the input image
            pad_image[gal_image.bounds] = gal_image
            self.original_image = galsim.SBInterpolatedImage(
                pad_image, xInterp=self.x_interpolant, kInterp=self.k_interpolant,
                dx=self.pixel_scale, pad_factor=1., gsparams=gsparams)

        # also make the original PSF image, with far less fanfare: we don't need to pad with
        # anything interesting.
        self.original_PSF = galsim.SBInterpolatedImage(
            PSF_image, xInterp=self.x_interpolant, kInterp=self.k_interpolant, dx=self.pixel_scale,
            gsparams=gsparams)

        # recalculate Fourier-space attributes rather than using overly-conservative defaults
        self.original_image.calculateStepK()
        self.original_image.calculateMaxK()
        self.original_PSF.calculateStepK()
        self.original_PSF.calculateMaxK()
        
        if flux != None:
            self.original_image.setFlux(flux)
            self.original_image.__class__ = galsim.SBTransform # correctly reflect SBProfile change
        self.original_PSF.setFlux(1.0)
        self.original_PSF.__class__ = galsim.SBTransform # correctly reflect SBProfile change

        # Calculate the PSF "deconvolution" kernel
        psf_inv = galsim.SBDeconvolve(self.original_PSF, gsparams=gsparams)
        # Initialize the SBProfile attribute
        GSObject.__init__(
            self, galsim.SBConvolve([self.original_image, psf_inv], gsparams=gsparams))

    def getHalfLightRadius(self):
        raise NotImplementedError("Half light radius calculation not implemented for RealGalaxy "
                                   +"objects.")

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

class Shapelet(GSObject):
    """A class describing polar shapelet surface brightness profiles.

    This class describes an arbitrary profile in terms of a shapelet decomposition.  A shapelet
    decomposition is an eigenfunction decomposition of a 2-d function using the eigenfunctions
    of the 2-d quantum harmonic oscillator.  The functions are Laguerre polynomials multiplied
    by a Gaussian.  See Bernstein & Jarvis, 2002 or Massey & Refregier, 2005 for more detailed 
    information about this kind of decomposition.  For this class, we follow the notation of 
    Bernstein & Jarvis.

    The decomposition is described by an overall scale length, sigma, and a vector of 
    coefficients, b.  The b vector is indexed by two values, which can be either (p,q) or (N,m).
    In terms of the quantum solution of the 2-d harmonic oscillator, p and q are the number of 
    quanta with positive and negative angular momentum (respectively).  Then, N=p+q, m=p-q.

    The 2D image is given by (in polar coordinates):

        I(r,theta) = 1/sigma^2 Sum_pq b_pq psi_pq(r/sigma, theta)

    where psi_pq are the shapelet eigenfunctions, given by:

        psi_pq(r,theta) = (-)^q/sqrt(pi) sqrt(q!/p!) r^m exp(i m theta) exp(-r^2/2) L_q^(m)(r^2)

    and L_q^(m)(x) are generalized Laguerre polynomials.
    
    The coeffients b_pq are in general complex.  However, we require that the resulting 
    I(r,theta) be purely real, which implies that b_pq = b_qp* (where * means complex conjugate).
    This further implies that b_pp (i.e. b_pq with p==q) is real. 


    Initialization
    --------------
    
    1. Make a blank Shapelet instance with all b_pq = 0.

        shapelet = galsim.Shapelet(sigma=sigma, order=order)

    2. Make a Shapelet instance using a given vector for the b_pq values.

        order = 2
        bvec = [ 1, 0, 0, 0.2, 0.3, -0.1 ]
        shapelet = galsim.Shapelet(sigma=sigma, order=order, bvec=bvec)

    We use the following order for the coeffiecients, where the subscripts are in terms of p,q.

    [ b00  Re(b10)  Im(b10)  Re(b20)  Im(b20)  b11  Re(b30)  Im(b30)  Re(b21)  Im(b21) ... ]

    i.e. we progressively increase N, and for each value of N, we start with m=N and go down to 
    m=0 or 1 as appropriate.  And since m=0 is intrinsically real, it only requires one spot
    in the list.

    @param sigma          The scale size in the standard units (usually arcsec).
    @param order          Specify the order of the shapelet decomposition.  This is the maximum
                          N=p+q included in the decomposition.
    @param bvec           The initial vector of coefficients.  (Default: all zeros)


    Methods
    -------

    The Shapelet is a GSObject, and inherits most of the GSObject methods (draw(), applyShear(),
    etc.) and operator bindings.  The exception is drawShoot, which is not yet implemented for 
    Shapelet instances.
    
    In addition, Shapelet has the following methods:

    getSigma()         Get the sigma value.
    getOrder()         Get the order, the maximum N=p+q used by the decomposition.
    getBVec()          Get the vector of coefficients, returned as a numpy array.
    getPQ(p,q)         Get b_pq.  Returned as tuple (re, im) (even if p==q).
    getNM(N,m)         Get b_Nm.  Returned as tuple (re, im) (even if m=0).

    setSigma(sigma)    Set the sigma value.
    setOrder(order)    Set the order.
    setBVec(bvec)      Set the vector of coefficients.
    setPQ(p,q,re,im=0) Set b_pq.
    setNM(N,m,re,im=0) Set b_Nm.

    fitImage(image)    Fit for a shapelet decomposition of the given image.
    """

    # Initialization parameters of the object, with type information
    _req_params = { "sigma" : float, "order" : int }
    _opt_params = {}
    _single_params = []
    _takes_rng = False

    # --- Public Class methods ---
    def __init__(self, sigma, order, bvec=None):
        # Make sure order and sigma are the right type:
        order = int(order)
        sigma = float(sigma)

        # Make bvec if necessary
        if bvec is None:
            bvec = galsim.LVector(order)
        else:
            bvec_size = galsim.LVectorSize(order)
            if len(bvec) != bvec_size:
                raise ValueError("bvec is the wrong size for the provided order")
            import numpy
            bvec = galsim.LVector(order,numpy.array(bvec))

        GSObject.__init__(self, galsim.SBShapelet(sigma, bvec))

    def getSigma(self):
        return self.SBProfile.getSigma()
    def getOrder(self):
        return self.SBProfile.getBVec().order
    def getBVec(self):
        return self.SBProfile.getBVec().array
    def getPQ(self,p,q):
        return self.SBProfile.getBVec().getPQ(p,q)
    def getNM(self,N,m):
        return self.SBProfile.getBVec().getPQ((N+m)/2,(N-m)/2)

    # Note: Since SBProfiles are officially immutable, these create a new
    # SBProfile object for this GSObject.  This is of course inefficient, but not
    # outrageously so, since the SBShapelet constructor is pretty minimalistic, and 
    # presumably anyone who cares about efficiency would not be using these functions.
    # They would create the Shapelet with the right bvec from the start.
    def setSigma(self,sigma):
        bvec = self.SBProfile.getBVec()
        GSObject.__init__(self, galsim.SBShapelet(sigma, bvec))
    def setOrder(self,order):
        curr_bvec = self.SBProfile.getBVec()
        curr_order = curr_bvec.order
        if curr_order == order: return
        # Preserve the existing values as much as possible.
        sigma = self.SBProfile.getSigma()
        if curr_order > order:
            bvec = galsim.LVector(order, curr_bvec.array[0:galsim.LVectorSize(order)])
        else:
            import numpy
            a = numpy.zeros(galsim.LVectorSize(order))
            a[0:len(curr_bvec.array)] = curr_bvec.array
            bvec = galsim.LVector(order,a)
        GSObject.__init__(self, galsim.SBShapelet(sigma, bvec))
    def setBVec(self,bvec):
        sigma = self.SBProfile.getSigma()
        order = self.SBProfile.getBVec().order
        bvec_size = galsim.LVectorSize(order)
        if len(bvec) != bvec_size:
            raise ValueError("bvec is the wrong size for the Shapelet order")
        import numpy
        bvec = galsim.LVector(order,numpy.array(bvec))
        GSObject.__init__(self, galsim.SBShapelet(sigma, bvec))
    def setPQ(self,p,q,re,im=0.):
        sigma = self.SBProfile.getSigma()
        bvec = self.SBProfile.getBVec().copy()
        bvec.setPQ(p,q,re,im)
        GSObject.__init__(self, galsim.SBShapelet(sigma, bvec))
    def setNM(self,N,m,re,im=0.):
        self.setPQ((N+m)/2,(N-m)/2,re,im)

    def setFlux(self, flux):
        # More efficient to change the bvector rather than add a transformation layer above 
        # the SBShapelet, which is what the normal setFlux method does.
        self.scaleFlux(flux/self.getFlux())

    def scaleFlux(self, fluxRatio):
        # More efficient to change the bvector rather than add a transformation layer above 
        # the SBShapelet, which is what the normal setFlux method does.
        sigma = self.SBProfile.getSigma()
        bvec = self.SBProfile.getBVec() * fluxRatio
        GSObject.__init__(self, galsim.SBShapelet(sigma, bvec))

    def applyRotation(self, theta):
        if not isinstance(theta, galsim.Angle):
            raise TypeError("Input theta should be an Angle")
        sigma = self.SBProfile.getSigma()
        bvec = self.SBProfile.getBVec().copy()
        bvec.rotate(theta)
        GSObject.__init__(self, galsim.SBShapelet(sigma, bvec))

    def applyDilation(self, scale):
        sigma = self.SBProfile.getSigma() * scale
        bvec = self.SBProfile.getBVec()
        GSObject.__init__(self, galsim.SBShapelet(sigma, bvec))

    def applyMagnification(self, mu):
        sigma = self.SBProfile.getSigma() * np.sqrt(mu)
        bvec = self.SBProfile.getBVec() * mu
        GSObject.__init__(self, galsim.SBShapelet(sigma, bvec))

    def fitImage(self, image, center=None, normalization='flux'):
        """Fit for a shapelet decomposition of a given image

        The optional normalization parameter mirrors the parameter in the GSObject `draw` method.
        If the fitted shapelet is drawn with the same normalization value as was used when it 
        was fit, then the resulting image should be an approximate match to the original image.

        For example:

            image = ...
            shapelet = galsim.Shapelet(sigma, order)
            shapelet.fitImage(image,normalization='sb')
            shapelet.draw(image=image2, dx=image.scale, normalization='sb')

        Then image2 and image should be as close to the same as possible for the given
        sigma and order.  Increasing the order can improve the fit, as can having sigma match
        the natural scale size of the image.  However, it should be noted that some images
        are not well fit by a shapelet for any (reasonable) order.

        @param image          The Image for which to fit the shapelet decomposition
        @param center         The position in pixels to use for the center of the decomposition.
                              [Default: use the image center (`image.bounds.trueCenter()`)]
        @param normalization  The normalization to assume for the image. 
                              (Default `normalization = "flux"`)
        """
        if not center:
            center = image.bounds.trueCenter()
        # convert from PositionI if necessary
        center = galsim.PositionD(center.x,center.y)

        if not normalization.lower() in ("flux", "f", "surface brightness", "sb"):
            raise ValueError(("Invalid normalization requested: '%s'. Expecting one of 'flux', "+
                              "'f', 'surface brightness' or 'sb'.") % normalization)

        sigma = self.SBProfile.getSigma()
        bvec = self.SBProfile.getBVec().copy()

        galsim.ShapeletFitImage(sigma, bvec, image, center)

        if normalization.lower() == "flux" or normalization.lower() == "f":
            bvec /= image.scale**2

        # SBShapelet, like all SBProfiles, is immutable, so we need to reinitialize with a 
        # new Shapelet object.
        GSObject.__init__(self, galsim.SBShapelet(sigma, bvec))



