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
        return galsim.Add(self, other)

    # op+= converts this into the equivalent of an Add object
    def __iadd__(self, other):
        GSObject.__init__(self, galsim.SBAdd([self.SBProfile, other.SBProfile]))
        self.__class__ = galsim.Add
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
        import numpy as np
        self.SBProfile.applyScale(scale)
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
        import numpy as np
        self.SBProfile.applyScale(np.sqrt(mu))
       
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
    def _draw_setup_image(self, image, dx, wmult, add_to_image, dx_is_dk=False):

        # Make sure the type of wmult is correct and has a valid value
        if type(wmult) != float:
            wmult = float(wmult)
        if wmult <= 0:
            raise ValueError("Invalid wmult <= 0 in draw command")

        # Save the input value, since we'll need to make a new dx (in case image is None)
        if dx_is_dk: dk = dx

        # Check dx value and adjust if necessary
        if dx is None:
            if image is not None and image.scale > 0.:
                if dx_is_dk:
                    # dx = 2pi / (N*dk)
                    dk = image.scale
                    import numpy as np
                    dx = 2.*np.pi/( np.max(image.array.shape) * image.scale )
                else:
                    dx = image.scale
            else:
                dx = self.SBProfile.nyquistDx()
                if dx_is_dk:
                    dk = self.stepK()
        elif dx <= 0:
            dx = self.SBProfile.nyquistDx()
            if dx_is_dk:
                dk = self.stepK()
        elif type(dx) != float:
            if dx_is_dk:
                dk = float(dx)
                if image is not None:
                    import numpy as np
                    dx = 2.*np.pi/( np.max(image.array.shape) * dk )
                else:
                    dx = self.SBProfile.nyquistDx()
            else:
                dx = float(dx)
        # At this point dx is really dx, not dk.

        # Make image if necessary
        if image is None:
            # Can't add to image if none is provided.
            if add_to_image:
                raise ValueError("Cannot add_to_image if image is None")
            N = self.SBProfile.getGoodImageSize(dx,wmult)
            image = galsim.ImageF(N,N)

        # Resize the given image if necessary
        elif not image.getBounds().isDefined():
            # Can't add to image if need to resize
            if add_to_image:
                raise ValueError("Cannot add_to_image if image bounds are not defined")
            N = self.SBProfile.getGoodImageSize(dx,wmult)
            bounds = galsim.BoundsI(1,N,1,N)
            image.resize(bounds)
            image.setZero()

        # Else use the given image as is
        else:
            # Clear the image if we are not adding to it.
            if not add_to_image:
                image.setZero()

        # Set the image scale
        if dx_is_dk:
            image.setScale(dk)
        else:
            image.setScale(dx)

        return image

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
        image = self._draw_setup_image(image,dx,wmult,add_to_image)

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
            gain /= image.scale**2

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
        image = self._draw_setup_image(image,dx,wmult,add_to_image)

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
            gain /= image.scale**2

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

    def drawK(self, re=None, im=None, dk=None, gain=1., add_to_image=False):
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
                if re.scale != im.scale:
                    raise ValueError("re and im do not have the same input scale")
            if re.getBounds().isDefined() or im.getBounds().isDefined():
                if re.getBounds() != im.getBounds():
                    raise ValueError("re and im do not have the same defined bounds")

        # Make sure images are setup correctly
        re = self._draw_setup_image(re,dk,1.0,add_to_image,dx_is_dk=True)
        im = self._draw_setup_image(im,dk,1.0,add_to_image,dx_is_dk=True)

        # wmult isn't really used by drawK, but we need to provide it.
        wmult = 1.0
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


