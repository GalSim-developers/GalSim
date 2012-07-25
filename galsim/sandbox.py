import os
import collections
import numpy as np
import galsim
import utilities
import descriptors

ALIAS_THRESHOLD = 0.005 # Matches hard coded value in src/SBProfile.cpp. TODO: bring these together

class GSObject(object):
    """@brief Base class for defining the interface with which all GalSim Objects access their
    shared methods and attributes, particularly those from the C++ SBProfile classes.
    """
    _data = {} # Used for storing GalSim object parameter data, accessed by their descriptors
    _SBProfile = None  # Private attribute used by the SBProfile property to store (and rebuild if
                       # necessary) the C++ layer SBProfile object for which GSObjects are a container

    # Then we define the .SBProfile attribute to actually be a property, with getter and setter
    # functions that provide access to the data stored in _SBProfile.  If the latter is None, for
    # example after a change in the object parameters (see, e.g., the SimpleParam descriptor), then
    # the SBProfile is re-initialized.
    #
    # Note that this requires ALL classes derived from GSObject to define a _SBInitialize() method. 
    def _get_SBProfile(self):
        if self._SBProfile is None:
            self._SBInitialize()
        return self._SBProfile
    def _set_SBProfile(self, value):
        self._SBProfile = value
    SBProfile = property(_get_SBProfile, _set_SBProfile)

    def __init__(self, SBProfile):
        self.SBProfile = SBProfile  # This guarantees that all GSObjects have an SBProfile

    # Make op+ of two GSObjects work to return an Add object
    def __add__(self, other):
        return Add(self,other)

    # op+= converts this into the equivalent of an Add object
    def __iadd__(self, other):
        GSObject.__init__(self, galsim.SBAdd(self.SBProfile, other.SBProfile))
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
        """@brief Returns a copy of an object

           This preserves the original type of the object, so if the caller is a
           Gaussian (for example), the copy will also be a Gaussian, and can thus call
           the methods that are not in GSObject, but are in Gaussian (e.g. getSigma).
        """
        sbp = self.SBProfile.__class__(self.SBProfile)
        ret = GSObject(sbp)
        ret.__class__ = self.__class__
        return ret

    # Now define direct access to all SBProfile methods via calls to self.SBProfile.method_name()
    #
    def maxK(self):
        """@brief Returns value of k beyond which aliasing can be neglected.
        """
        return self.SBProfile.maxK()

    def nyquistDx(self):
        """@brief Returns Image pixel spacing that does not alias maxK.
        """
        return self.SBProfile.nyquistDx()

    def stepK(self):
        """@brief Returns sampling in k space necessary to avoid folding of image in x space.
        """
        return self.SBProfile.stepK()

    def hasHardEdges(self):
        """@brief Returns True if there are any hard edges in the profile.
        """
        return self.SBProfile.hasHardEdges()

    def isAxisymmetric(self):
        """@brief Returns True if axially symmetric: affects efficiency of evaluation.
        """
        return self.SBProfile.isAxisymmetric()

    def isAnalyticX(self):
        """@brief Returns True if real-space values can be determined immediately at any position
        without requiring a Discrete Fourier Transform.
        """
        return self.SBProfile.isAnalyticX()

    # This method does not seem to be wrapped from C++
    # def isAnalyticK(self):
    # return self.SBProfile.isAnalyticK()

    def centroid(self):
        """@brief Returns the (x, y) centroid of an object as a Position.
        """
        return self.SBProfile.centroid()

    def getFlux(self):
        """@brief Returns the flux of the object.
        """
        return self.SBProfile.getFlux()

    def xValue(self, position):
        """@brief Returns the value of the object at a chosen 2D position in real space.
        
        As in SBProfile, this function assumes all are real-valued.  xValue() may not be
        implemented for derived classes (e.g. SBConvolve) that require an Discrete Fourier
        Transform to determine real space values.  In this case, an SBError will be thrown at the
        C++ layer (raises a RuntimeError in Python).
        
        @param position  A 2D galsim.PositionD/I instance giving the position in real space.
        """
        return self.SBProfile.xValue(position)

    def kValue(self, position):
        """@brief Returns the value of the object at a chosen 2D position in k space.

        @param position  A 2D galsim.PositionD/I instance giving the position in k space.
        """
        return self.SBProfile.kValue(position)

    def scaleFlux(self, fluxRatio):
        """@brief Multiply the flux of the object by fluxRatio
           
        After this call, the caller's type will be a GSObject.
        This means that if the caller was a derived type that had extra methods beyond
        those defined in GSObject (e.g. getSigma for a Gaussian), then these methods
        are no longer available.
        """
        self.SBProfile.scaleFlux(fluxRatio)
        self.__class__ = GSObject

    def setFlux(self, flux):
        """@brief Set the flux of the object.
           
        After this call, the caller's type will be a GSObject.
        This means that if the caller was a derived type that had extra methods beyond
        those defined in GSObject (e.g. getSigma for a Gaussian), then these methods
        are no longer available.
        """
        self.SBProfile.setFlux(flux)
        self.__class__ = GSObject

    def applyTransformation(self, ellipse):
        """@brief Apply a galsim.ellipse.Ellipse distortion to this object.
           
        Ellipse objects can be initialized in a variety of ways (see documentation of this
        class for details).

        Note: if the ellipse includes a dilation, then this transformation will 
        not be flux-conserving.  It conserves surface brightness instead.
        Thus, the flux will increase by the increase in area = dilation^2.

        After this call, the caller's type will be a GSObject.
        This means that if the caller was a derived type that had extra methods beyond
        those defined in GSObject (e.g. getSigma for a Gaussian), then these methods
        are no longer available.
        """
        if not isinstance(ellipse, galsim.Ellipse):
            raise TypeError("Argument to applyTransformation must be a galsim.Ellipse!")
        self.SBProfile.applyTransformation(ellipse._ellipse)
        self.__class__ = GSObject
 
    def applyDilation(self, scale):
        """@brief Apply a dilation of the linear size by the given scale.

        Scales the linear dimensions of the image by the factor scale.
        e.g. half_light_radius <-- half_light_radius * scale

        This operation preserves flux.
        See applyMagnification for a version that preserves surface brightness, and thus 
        changes the flux.

        After this call, the caller's type will be a GSObject.
        This means that if the caller was a derived type that had extra methods beyond
        those defined in GSObject (e.g. getSigma for a Gaussian), then these methods
        are no longer available.
        """
        import math
        flux = self.getFlux()
        self.applyTransformation(galsim.Ellipse(math.log(scale)))
        self.setFlux(flux)

    def applyMagnification(self, scale):
        """@brief Apply a magnification by the given scale, scaling the linear size by scale
        and the flux by scale^2.  
        
        Scales the linear dimensions of the image by the factor scale.
        e.g. half_light_radius <-- half_light_radius * scale

        This operation preserves surface brightness, which means that the flux is scales 
        with the change in area.  
        See applyDilation for a version that preserves flux.

        After this call, the caller's type will be a GSObject.
        This means that if the caller was a derived type that had extra methods beyond
        those defined in GSObject (e.g. getSigma for a Gaussian), then these methods
        are no longer available.
        """
        import math
        self.applyTransformation(galsim.Ellipse(math.log(scale)))

       
    def applyShear(self, *args, **kwargs):
        """@brief Apply a shear to this object, where arguments are either a galsim.shear.Shear, or
        arguments that will be used to initialize one.

        After this call, the caller's type will be a GSObject.
        This means that if the caller was a derived type that had extra methods beyond
        those defined in GSObject (e.g. getSigma for a Gaussian), then these methods
        are no longer available.
        """
        if len(args) == 1:
            if kwargs:
                raise TypeError("Error, gave both unnamed and named arguments to applyShear!")
            if not isinstance(args[0], galsim.Shear):
                raise TypeError("Error, unnamed argument to applyShear is not a Shear!")
            self.SBProfile.applyShear(args[0]._shear)
        elif len(args) > 1:
            raise TypeError("Error, too many unnamed arguments to applyShear!")
        else:
            shear = galsim.Shear(**kwargs)
            self.SBProfile.applyShear(shear._shear)
        self.__class__ = GSObject

    def applyRotation(self, theta):
        """@brief Apply a rotation theta (Angle object, +ve anticlockwise) to this object.
           
        After this call, the caller's type will be a GSObject.
        This means that if the caller was a derived type that had extra methods beyond
        those defined in GSObject (e.g. getSigma for a Gaussian), then these methods
        are no longer available.
        """
        if not isinstance(theta, galsim.Angle):
            raise TypeError("Input theta should be an Angle")
        self.SBProfile.applyRotation(theta)
        self.__class__ = GSObject
        
    def applyShift(self, dx, dy):
        """@brief Apply a (dx, dy) shift to this object.
           
        After this call, the caller's type will be a GSObject.
        This means that if the caller was a derived type that had extra methods beyond
        those defined in GSObject (e.g. getSigma for a Gaussian), then these methods
        are no longer available.
        """
        self.SBProfile.applyShift(dx, dy)
        self.__class__ = GSObject

    # Also add methods which create a new GSObject with the transformations applied...
    #
    def createTransformed(self, ellipse):
        """@brief Returns a new GSObject by applying a galsim.ellipse.Ellipse transformation 
        (shear, dilate, and/or shift).

        Note that Ellipse objects can be initialized in a variety of ways (see documentation 
        of this class for details).
        """
        if not isinstance(ellipse, galsim.Ellipse):
            raise TypeError("Argument to createTransformed must be a galsim.ellipse.Ellipse!")
        ret = self.copy()
        ret.applyTransformation(ellipse)
        return ret

    def createDilated(self, scale):
        """@brief Returns a new GSObject by applying a dilation of the linear size by the 
        given scale.
        
        Scales the linear dimensions of the image by the factor scale.
        e.g. half_light_radius <-- half_light_radius * scale

        This operation preserves flux.  
        See createMagnified for a version that preserves surface brightness, and thus 
        changes the flux.
        """
        import math
        ret = self.copy()
        flux = self.getFlux()
        ret.applyTransformation(galsim.Ellipse(math.log(scale)))
        ret.setFlux(flux)
        return ret

    def createMagnified(self, scale):
        """@brief Returns a new GSObject by applying a magnification by the given scale,
        scaling the linear size by scale and the flux by scale^2.  

        Scales the linear dimensions of the image by the factor scale.
        e.g. half_light_radius <-- half_light_radius * scale

        This operation preserves surface brightness, which means that the flux
        is also scaled by a factor of scale^2.
        See createDilated for a version that preserves flux.
        """
        import math
        ret = self.copy()
        ret.applyTransformation(galsim.Ellipse(math.log(scale)))
        return ret

    def createSheared(self, *args, **kwargs):
        """@brief Returns a new GSObject by applying a shear, where arguments are either a
        galsim.shear.Shear or keyword arguments that can be used to create one.
        """
        ret = self.copy()
        ret.applyShear(*args, **kwargs)
        return ret

    def createRotated(self, theta):
        """@brief Returns a new GSObject by applying a rotation theta (Angle object, +ve
        anticlockwise).
        """
        if not isinstance(theta, galsim.Angle):
            raise TypeError("Input theta should be an Angle")
        ret = self.copy()
        ret.applyRotation(theta)
        return ret
        
    def createShifted(self, dx, dy):
        """@brief Returns a new GSObject by applying a (dx, dy) shift.
        """
        ret = self.copy()
        ret.applyShift(dx, dy)
        return ret

    def draw(self, image=None, dx=None, gain=1., wmult=1, normalization="flux", add_to_image=False):
        """@brief Draws an Image of the object, with bounds optionally set by an input Image.

        @param image  If provided, this will be the image on which to draw the profile.
                      If image=None, then an automatically-sized image will be created.
                      (Default = None)
        @param dx     If provided, use this as the pixel scale for the image.
                      If dx is None and image != None, then take the provided image's pixel scale.
                      If dx is None and image == None, then use pi/maxK()
                      (Default = None)
        @param gain   The number of ADU to place on the image per photon.  (Default = 1)
        @param wmult  A factor by which to make the intermediate images larger than 
                      they are normally made.  The size is normally automatically chosen 
                      to reach some preset accuracy targets (see include/galsim/SBProfile.h); 
                      however, if you see strange artifacts in the image, you might try using 
                      wmult > 1.  This will take longer of course, but it will produce more 
                      accurate images, since they will have less "folding" in Fourier space.
                      (Default = 1.)
        @param normalization  Two options for the normalization:
                              "flux" or "f" means that the sum of the output pixels is normalized
                                     to be equal to the total flux.  (Modulo any flux that
                                     falls off the edge of the image of course.)
                              "surface brightness" or "sb" means that the output pixels sample
                                     the surface brightness distribution at each location.
                              (Default = "flux")
        @param add_to_image  Whether to add flux to the existing image rather than clear out
                             anything in the image before shooting.
                             (Default = False)
        @returns      The drawn image.
        """
        # Raise an exception immediately if the normalization type is not recognized
        if not normalization.lower() in ("flux", "f", "surface brightness", "sb"):
            raise ValueError(("Invalid normalization requested: '%s'. Expecting one of 'flux', "+
                              "'f', 'surface brightness' or 'sb'.") % normalization)
        # Raise an exception here since C++ is picky about the input types
        if type(wmult) != int:
            raise TypeError("Input wmult should be an int")
        if type(gain) != float:
            gain = float(gain)
        if dx is None: 
            dx = 0.
        if type(dx) != float:
            dx = float(dx)

        if image == None:
            # Can't add to image if none is provided.
            if add_to_image:
                raise ValueError("Cannot add_to_image if image is None")

            image = self.SBProfile.draw(dx, gain, wmult)

            # In this case, the draw command may set dx automatically, so we need to 
            # adjust the flux after the fact.  But this is ok, since add_to_image is
            # invalid in this case.
            if normalization.lower() == "flux" or normalization.lower() == "f":
                dx = image.getScale()
                image *= dx*dx

        else :
            
            # Set dx based on the image if not provided something else.
            if dx <= 0.:
                dx = image.getScale()

            # Clear the image if we are not adding to it.
            if not add_to_image:
                image.setZero()

            # SBProfile draw command uses surface brightness normalization.  So if we
            # want flux normalization, we need to scale the flux by dx^2
            if normalization.lower() == "flux" or normalization.lower() == "f":
                gain *= dx**2

            self.SBProfile.draw(image, dx, gain, wmult)
         
        return image

    def drawShoot(self, image, n_photons=0., dx=None, gain=1., uniform_deviate=None,
                  normalization="flux", noise=0., poisson_flux=True, add_to_image=False):
        """@brief Draw an image of the object by shooting individual photons drawn from the 
        surface brightness profile of the object.

        @param image  The image on which to draw the profile.
                      Note: Unlike for the regular draw command, image is a required
                      parameter.  drawShoot will not make the image for you.
        @param n_photons    If provided, the number of photons to use.
                            If not provided, use as many photons as necessary to end up with
                            an image with the correct poisson shot noise for the object's flux.
                            For positive definite profiles, this is equivalent to n_photons = flux.
                            However, some profiles need more than this because some of the shot
                            photons are negative (usually due to interpolants).
                            (Default = 0)
        @param dx     If provided, use this as the pixel scale for the image.
                      If dx is None then use the provided image's pixel scale.
                      (Default = None)
        @param gain  The number of ADU to place on the image per photon.  (Default = 1)
        @param uniform_deviate  If provided, a UniformDeviate to use for the random numbers
                                If uniform_deviate=None, one will be automatically created, 
                                using the time as a seed.
                                (Default = None)
        @param normalization  Two options for the normalization:
                              "flux" or "f" means that the sum of the output pixels is normalized
                                     to be equal to the total flux.  (Modulo any flux that
                                     falls off the edge of the image of course.)
                              "surface brightness" or "sb" means that the output pixels sample
                                     the surface brightness distribution at each location.
                              (Default = "flux")
        @param noise  If provided, the allowed extra noise in each pixel.
                      This is only relevant if n_photons=0, so the number of photons is being 
                      automatically calculated.  In that case, if the image noise is 
                      dominated by the sky background, you can get away with using fewer
                      shot photons than the full n_photons = flux.  Essentially each shot photon
                      can have a flux > 1, which increases the noise in each pixel.
                      The noise parameter specifies how much extra noise per pixel is allowed 
                      because of this approximation.  A typical value for this might be
                      noise = sky_level / 100 where sky_level is the flux per pixel 
                      due to the sky.  If the natural number of photons produces less noise 
                      than this value for all pixels, we lower the number of photons to bring 
                      the resultant noise up to this value.  If the natural value produces 
                      more noise than this, we accept it and just use the natural value.
                      Note that this uses a "variance" definition of noise, not a "sigma" 
                      definition.
                      (Default = 0.)
        @param poisson_flux  Whether to allow total object flux scaling to vary according to 
                             Poisson statistics for n_photons samples.
                             (Default = True)
        @param add_to_image  Whether to add flux to the existing image rather than clear out
                             anything in the image before shooting.
                             (Default = False)
                              
        @returns  The tuple (image, added_flux), where image is the input with drawn photons 
                  added and added_flux is the total flux of photons that landed inside the image 
                  bounds.

        The second part of the return tuple may be useful as a sanity check that you have
        provided a large enough image to catch most of the flux.  For example:
        @code
        image, added_flux = obj.drawShoot(image)
        assert added_flux > 0.99 * obj.getFlux()
        @endcode
        However, the appropriate threshold will depend things like whether you are 
        keeping poisson_flux=True, how high the flux is, how big your images are relative to
        the size of your object, etc.

        The input image must have defined boundaries and pixel scale.  The photons generated by
        the drawShoot() method will be binned into the target image.  The input image will be 
        cleared before drawing in the photons by default, unless the keyword add_to_image is 
        set to True.  Scale and location of the image pixels will not be altered. 

        It is important to remember that the image produced by drawShoot() represents the object
        as convolved with the square image pixel.  So when using drawShoot() instead of draw(),
        you should not convolve with a Pixel.  This will produce the equivalent image (for very 
        large n_photons) as draw() produces when the same object is convolved with Pixel(xw=dx) 
        when drawing onto an image with pixel scale dx.
        """

        # Raise an exception immediately if the normalization type is not recognized
        if not normalization.lower() in ("flux", "f", "surface brightness", "sb"):
            raise ValueError(("Invalid normalization requested: '%s'. Expecting one of 'flux', "+
                              "'f', 'surface brightness' or 'sb'.") % normalization)
        # Raise an exception here since C++ is picky about the input types
        if image is None:
            raise TypeError("drawShoot requires the image to be provided.")

        if type(n_photons) != float:
            # if given an int, just convert it to a float
            n_photons = float(n_photons)
        if dx is None: 
            dx = 0.
        if type(dx) != float:
            dx = float(dx)
        if type(gain) != float:
            gain = float(gain)
        if type(noise) != float:
            noise = float(noise)
        if uniform_deviate == None:
            uniform_deviate = galsim.UniformDeviate()

        # Check that either n_photons is set to something or flux is set to something
        if n_photons == 0. and self.getFlux() == 1.:
            import warnings
            msg = "Warning: drawShoot for object with flux == 1, but n_photons == 0.\n"
            msg += "This will only shoot a single photon."
            warnings.warn(msg)

        # Clear the image if we are not adding to it.
        if not add_to_image:
            image.setZero()

        # Set dx based on the image if not provided something else.
        if dx <= 0.:
            dx = image.getScale()

        # SBProfile draw command uses surface brightness normalization.  So if we
        # want flux normalization, we need to scale the flux by dx^2
        if normalization.lower() == "flux" or normalization.lower() == "f":
            gain *= dx**2
            
        added_flux = self.SBProfile.drawShoot(
                image, n_photons, uniform_deviate, dx, gain, noise, poisson_flux)

        return image, added_flux


class RadialProfile(GSObject):
    """@brief Intermediate base class that defines some parameters shared by all "radial profile"
    GSObjects.

    The radial profile GSObjects are characterized by:
      * one or more size parameters, e.g. sigma (for the Gaussian), half_light_radius (all objects),
        from which one only must be chosen for initialization
      * a flux
      * optional parameters describing the radial profile, but not directly related to the object's
        apparent size

    Currently, the RadialProfile objects are:
    Airy, DeVaucouleurs, Exponential, Gaussian, Moffat, Sersic

    Although only one size parameter must be chosen for initializing RadialProfile objects (giving
    more than one will raise a TypeError exception), subsequently all the size parameters defined
    for that object can be accessed as attributes.  If one of these size attributes is assigned to
    a new value, all the other sizes and the underlying SBProfile description of the profile itself
    will be updated to match.

    All RadialProfile GSObject classes share the half_light_radius size specification.
    """

    # All RadialProfile objects have a flux
    flux = descriptors.FluxParam()

    # All RadialProfile objects share a half_light_radius, so we can define this in the intermediate
    # base class
    half_light_radius = descriptors.SimpleParam(
        "half_light_radius", default=None,
        doc="Half light radius, kept consistent with the other size attributes.")
    
    def _parse_sizes(self, **kwargs):
        """
        Convenience function to parse input size parameters within the derived class __init__
        method.  Raises an exception if more than one input parameter kwarg is set != None.
        """
        size_set = False
        for name, value in kwargs.iteritems():
            if value != None:
                if size_set is True:
                    raise TypeError("Cannot specify more than one size parameter for this object.")
                else:
                    self.__setattr__(name, value)
                    size_set = True

class Gaussian1(RadialProfile):

    # --- Initialization of any additional size parameter descriptors ---
    sigma = descriptors.GetSetScaleParam(
        name="sigma", root_name="half_light_radius",
        factor=1 / 1.1774100225154747, # factor = 1 / sqrt[2ln(2)]
        doc="Scale radius sigma, kept consistent with the other size attributes.")
    
    fwhm = descriptors.GetSetScaleParam(
        name="fwhm", root_name="half_light_radius", factor=2., # strange but, it turns out, true...
        doc="FWHM, kept consistent with the other size attributes.")

    # --- Define the function used to (re)-initialize the contained SBProfile as necessary ---
    # *** Note a function of this name and similar content MUST be defined for all GSObjects! ***
    def _SBInitialize(self):
        GSObject.__init__(self, galsim.SBGaussian(half_light_radius=self.half_light_radius,
                                                  flux=self.flux))
    
    # --- Public Class methods ---
    def __init__(self, half_light_radius=None, sigma=None, fwhm=None, flux=1.):

        # Use the RadialProfile._parse_sizes() method to initialize size parameters
        RadialProfile._parse_sizes(
            self, half_light_radius=half_light_radius, sigma=sigma, fwhm=fwhm)

        # Set the flux
        self.flux = flux

        # Then build the SBProfile
        self._SBInitialize()




