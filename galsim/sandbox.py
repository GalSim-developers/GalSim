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
                       # necessary) the C++ layer SBProfile object for which GSObjects are a
                       # container

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
    """Intermediate base class that defines some parameters shared by all "radial profile"
    GSObjects.

    The radial profile GSObjects are characterized by:
      * one or more size parameters, e.g. sigma (for the Gaussian), half_light_radius (all objects),
        from which one only must be chosen for initialization
      * an optional flux parameter [default = 1]
      * zero or more additional parameters describing the radial profile, but not directly** related
        to the object's apparent size

    This intermediate base class sets up the parameter descriptors for the half_light_radius and
    flux params common to all derived objects.  Additional parameters should be defined in the
    class scopes for the derived RadialProfile objects.

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
        "half_light_radius", default=None, group="size",
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
        if size_set is False:
            raise TypeError("Must specify at least one size parameter for this object.")


class Gaussian(RadialProfile):
    """GalSim Gaussian, which has an SBGaussian in the SBProfile attribute.

    For more details of the Gaussian Surface Brightness profile, please see the SBGaussian
    documentation produced by doxygen.

    Initialization
    --------------
    A Gaussian can be initialized using one (and only one) of three possible size parameters

        half_light_radius
        sigma
        fwhm

    and an optional flux parameter [default flux = 1].

    Example:
    >>> gauss_obj = Gaussian(flux=3., sigma=1.)
    >>> gauss_obj.half_light_radius
    1.1774100225154747
    >>> gauss_obj.half_light_radius = 1.
    >>> gauss_obj.sigma
    0.8493218002880191

    Attempting to initialize with more than one size parameter is ambiguous, and will raise a
    TypeError exception.

    Methods
    -------
    The Gaussian is a GSObject, and inherits all of the GSObject methods (draw, drawShoot,
    applyShear etc.) and operator bindings.
    """

    # Initialization of additional size parameter descriptors beyond the half_light_radius inherited
    # by all RadialProfile GSObjects
    sigma = descriptors.GetSetScaleParam(
        name="sigma", root_name="half_light_radius", group="size",
        factor=1./1.1774100225154747, # factor = 1 / sqrt[2ln(2)]
        doc="Scale radius sigma, kept consistent with the other size attributes.")

    fwhm = descriptors.GetSetScaleParam(
        name="fwhm", root_name="half_light_radius", factor=2., # strange but, it turns out, true...
        group="size", doc="FWHM, kept consistent with the other size attributes.")

    # --- Defining the function used to (re)-initialize the contained SBProfile as necessary ---
    # *** Note a function of this name and similar content MUST be defined for all GSObjects! ***
    def _SBInitialize(self):
        GSObject.__init__(
            self, galsim.SBGaussian(half_light_radius=self.half_light_radius, flux=self.flux))
    
    # --- Public Class methods ---
    def __init__(self, half_light_radius=None, sigma=None, fwhm=None, flux=1.):

        # Use the RadialProfile._parse_sizes() method to initialize size parameters
        RadialProfile._parse_sizes(
            self, half_light_radius=half_light_radius, sigma=sigma, fwhm=fwhm)

        # Set the flux
        self.flux = flux

        # Then build the SBProfile
        self._SBInitialize()


class Sersic(RadialProfile):
    """GalSim Sersic, which has an SBSersic in the SBProfile attribute.

    For more details of the Sersic Surface Brightness profile, please see the SBSersic documentation
    produced by doxygen.

    Initialization
    --------------
    A Sersic is initialized with n, the Sersic index of the profile, and the half light radius size
    parameter half_light_radius.  A flux parameter is optional [default flux = 1].

    Example:
    >>> sersic_obj = Sersic(n=3.5, half_light_radius=2.5, flux=40.)
    >>> sersic_obj.half_light_radius
    2.5
    >>> sersic_obj.n
    3.5

    Methods
    -------
    The Sersic is a GSObject, and inherits all of the GSObject methods (draw, drawShoot,
    applyShear etc.) and operator bindings.
    """

    # Define the descriptor for the sersic index n
    n = descriptors.SimpleParam(
        "n", group="required", default=None, doc="Sersic index.")

    # --- Defining the function used to (re)-initialize the contained SBProfile as necessary ---
    # *** Note a function of this name and similar content MUST be defined for all GSObjects! ***
    def _SBInitialize(self):
        GSObject.__init__(
            self, galsim.SBSersic(self.n, half_light_radius=self.half_light_radius, flux=self.flux))

    # --- Public Class methods ---
    def __init__(self, n, half_light_radius, flux=1.):

        # Set the Sersic index
        self.n = n

        # Use the RadialProfile._parse_sizes() method to initialize size parameters
        RadialProfile._parse_sizes(self, half_light_radius=half_light_radius)

        # Set the flux
        self.flux = flux

        # Then build the SBProfile
        self._SBInitialize()


class Exponential(RadialProfile):
    """GalSim Exponential, which has an SBExponential in the SBProfile attribute.

    For more details of the Exponential Surface Brightness profile, please see the SBExponential
    documentation produced by doxygen.

    Initialization
    --------------
    An Exponential can be initialized using one (and only one) of two possible size parameters

        half_light_radius
        scale_radius

    and an optional flux parameter [default flux = 1].

    Example:
    >>> exp_obj = Exponential(flux=3., scale_radius=5.)
    >>> exp_obj.half_light_radius
    8.391734950083302
    >>> exp_obj.half_light_radius = 1.
    >>> exp_obj.scale_radius
    0.5958243473776976

    Attempting to initialize with more than one size parameter is ambiguous, and will raise a
    TypeError exception.

    Methods
    -------
    The Exponential is a GSObject, and inherits all of the GSObject methods (draw, drawShoot,
    applyShear etc.) and operator bindings.
    """
    
    # Beyond the half light radius, the additional size parameter for Exponential objects is the
    # scale_radius
    #
    # Constant scaling factor not analytic, but can be calculated by iterative solution of:
    #  (re / r0) = ln[(re / r0) + 1] + ln(2)
    scale_radius=descriptors.GetSetScaleParam(
        "scale_radius", root_name="half_light_radius", factor=1./1.6783469900166605,
        doc="scale_radius, kept consistent with the other size attributes.")

    # --- Defining the function used to (re)-initialize the contained SBProfile as necessary ---
    # *** Note a function of this name and similar content MUST be defined for all GSObjects! ***
    def _SBInitialize(self):
        GSObject.__init__(
            self, galsim.SBExponential(half_light_radius=self.half_light_radius, flux=self.flux))
 
    # --- Public Class methods ---
    def __init__(self, half_light_radius=None, scale_radius=None, flux=1.):

        # Use the RadialProfile._parse_sizes() method to initialize size parameters
        RadialProfile._parse_sizes(
            self, half_light_radius=half_light_radius, scale_radius=scale_radius)

        # Set the flux
        self.flux = flux

        # Then build the SBProfile
        self._SBInitialize()


class DeVaucouleurs(RadialProfile):
    """GalSim DeVaucouleurs, which has an SBDeVaucouleurs in the SBProfile attribute.

    For more details of the DeVaucouleurs Surface Brightness profile, please see the
    SBDeVaucouleurs documentation produced by doxygen.

    Initialization
    --------------
    A DeVaucouleurs is initialized with the half light radius size parameter half_light_radius and
    an optional flux parameter [default flux = 1].

    Example:
    >>> dvc_obj = DeVaucouleurs(half_light_radius=2.5, flux=40.)
    >>> dvc_obj.half_light_radius
    2.5
    >>> dvc_obj.flux
    40.0

    Methods
    -------
    The DeVaucouleurs is a GSObject, and inherits all of the GSObject methods (draw, drawShoot,
    applyShear etc.) and operator bindings.
    """

    # --- Defining the function used to (re)-initialize the contained SBProfile as necessary ---
    # *** Note a function of this name and similar content MUST be defined for all GSObjects! ***
    def _SBInitialize(self):
        GSObject.__init__(
            self, galsim.SBDeVaucouleurs(
                half_light_radius=self.half_light_radius, flux=self.flux))

    # --- Public Class methods ---
    def __init__(self, half_light_radius=None, flux=1.):

        # Use the RadialProfile._parse_sizes() method to initialize size parameters
        RadialProfile._parse_sizes(self, half_light_radius=half_light_radius)

        # Set the flux
        self.flux = flux

        # Then build the SBProfile
        self._SBInitialize() 


class Airy(RadialProfile):
    """GalSim Airy, which has an SBAiry in the SBProfile attribute.

    For more details of the Airy Surface Brightness profile, please see the SBAiry documentation
    produced by doxygen.

    Initialization
    --------------
    An Airy can be initialized using one (and only one) of two possible size parameters

        lam_over_D
        half_light_radius

    an optional obscuration parameter [default obscuration=0.] and an optional flux parameter
    [default flux = 1].  However, the half_light_radius size parameter can currently only be used
    if obscuration = 0.

    Example:
    >>> airy_obj = Airy(flux=3., lam_over_D=2.)
    >>> airy_obj.half_light_radius
    1.0696642954485294
    >>> airy_obj.half_light_radius = 1.
    >>> airy_obj.lam_over_D
    1.8697454972649754

    Attempting to initialize with more than one size parameter is ambiguous, and will raise a
    TypeError exception.

    Methods
    -------
    The Airy is a GSObject, and inherits all of the GSObject methods (draw, drawShoot, applyShear
    etc.) and operator bindings.
    """

    # Define the descriptor for the obscuration parameter
    obscuration = descriptors.SimpleParam(
        "obscuration", group="optional", default=0.,
        doc="Linear radial central obscuration for the obscured Airy.")

    # Then define the descriptor for the basic, underlying size parameter for the Airy, Lambda / D
    lam_over_D = descriptors.SimpleParam(
        "lam_over_D", group="size", default=None, doc="Lambda / D.")

    # Then we set up the other size descriptors.  These need to be a little more complex in their
    # execution than a typical RadialProfile, and involve a redefinition of the default
    # half_light_radius descriptor it provides

    # First we do the half_light_radius, for which we only have an easy scaling if obscuration=0.
    def _get_half_light_radius(self):
        if self.obscuration == 0.:
            # For an unobscured Airy, we have the following factor which can be derived using the
            # integral result given in the Wikipedia page (http://en.wikipedia.org/wiki/Airy_disk),
            # solved for half total flux using the free online tool Wolfram Alpha.
            # At www.wolframalpha.com:
            # Type "Solve[BesselJ0(x)^2+BesselJ1(x)^2=1/2]" ... and divide the result by pi
            return self.lam_over_D * 0.5348321477242647
        else:
            # In principle can find the half light radius as a function of lam_over_D and
            # obscuration too, but it will be much more involved
            raise NotImplementedError(
                "Half light radius calculation not implemented for Airy objects with non-zero "+
                "obscuration.")

    def _set_half_light_radius(self, value):
        if self.obscuration == 0.:
            # See _get_half_light_radius above for provenance of scaling factor
            self.lam_over_D = value / 0.5348321477242647
        else:
            raise NotImplementedError(
                "Half light radius support not implemented for Airy objects with non-zero "+
                "obscuration.")

    # Then we define the half_light_radius descriptor with ref. to these getter/setter functions
    half_light_radius = descriptors.GetSetFuncParam(
        getter=_get_half_light_radius, setter=_set_half_light_radius,
        doc="Half light radius, implemented for Airy function objects with obscuration=0.")

    # Now FWHM...
    def _get_fwhm(self):
        if self.obscuration == 0.:
            # As above, FWHM only easy to calculate for unobscured Airy
            return self.lam_over_D * 1.028993969962188
        else:
            # In principle can find the half light radius as a function of lam_over_D and
            # obscuration too, but it will be much more involved
            raise NotImplementedError(
                "FWHM calculation not implemented for Airy objects with non-zero obscuration.")

    def _set_fwhm(self, value):
        if self.obscuration == 0.:
            # As above, FWHM only easy to calculate for unobscured Airy
            self.lam_over_D = value / 1.028993969962188
        else:
            # In principle can find the half light radius as a function of lam_over_D and
            # obscuration too, but it will be much more involved
            raise NotImplementedError(
                "FWHM support not implemented for Airy objects with non-zero obscuration.")

    # Then we define the fwhm descriptor with reference to these getter/setter functions
    fwhm = descriptors.GetSetFuncParam(
        getter=_get_fwhm, setter=_set_fwhm,
        doc="FWHM, implemented for Airy function objects with obscuration=0.")

    # --- Defining the function used to (re)-initialize the contained SBProfile as necessary ---
    # *** Note a function of this name and similar content MUST be defined for all GSObjects! ***
    def _SBInitialize(self):
        GSObject.__init__(
            self, galsim.SBAiry(
                lam_over_D=self.lam_over_D, obscuration=self.obscuration, flux=self.flux))

    # --- Public Class methods ---
    def __init__(self, lam_over_D=lam_over_D, half_light_radius=None, obscuration=0., flux=1.):

        # Set obscuration. The latter must be set before the sizes to raise NotImplementedError
        # expections if half_light_radius is used with obscuration!=0.
        self.obscuration = obscuration

        # Use the RadialProfile._parse_sizes() method to initialize size parameters
        RadialProfile._parse_sizes(self, lam_over_D=lam_over_D, half_light_radius=half_light_radius)

        # Set the flux
        self.flux = flux

        # Then build the SBProfile
        self._SBInitialize()


class Moffat(RadialProfile):
    """@brief GalSim Moffat, which has an SBMoffat in the SBProfile attribute.

    For more details of the Moffat Surface Brightness profile, please see the SBMoffat
    documentation produced by doxygen.

    Initialization
    --------------
    A Moffat is initialized with a slope parameter beta, one (and only one) of three possible size
    parameters

        scale_radius
        half_light_radius
        fwhm

    an optional truncation radius parameter trunc [default trunc = 0., indicating no truncation] and
    a flux parameter [default flux = 1].

    Example:
    >>> moffat_obj = Moffat(beta=3., scale_radius=3., flux=0.5)
    >>> moffat_obj.half_light_radius
    1.9307827587167474
    >>> moffat_obj.half_light_radius = 1.
    >>> moffat_obj.scale_radius
    1.5537739740300376

    Attempting to initialize with more than one size parameter is ambiguous, and will raise a
    TypeError exception.

    Methods
    -------
    The Moffat is a GSObject, and inherits all of the GSObject methods (draw, drawShoot,
    applyShear etc.) and operator bindings.
    """

    # Define the descriptors for the Moffat slope parameter beta, and the truncation radius trunc
    beta = descriptors.SimpleParam(
        "beta", group="required", default=None, doc="Moffat profile slope parameter beta.")
    
    trunc = descriptors.SimpleParam(
        "trunc", group="optional", default=0.,
        doc="Truncation radius for Moffat in physical units.")

    # Then we set up the size descriptors.  These need to be a little more complex in their
    # execution than a typical RadialProfile, and involve a redefinition of the default
    # half_light_radius descriptor it provides.  Details below.

    # First we define a hidden storage variable to recall how the size parameter was last set: 
    _last_size_set_was_half_light_radius = False

    # Getter and setter functions for the scale_radius descriptor.
    # If the half light radius was the last size set then the value in _data["half_light_radius"]
    # will be None, so scale_radius needs to be got from self.SBProfile.getScaleRadius.
    def _get_scale_radius(self):
        if self._last_size_set_was_half_light_radius is True:
            return self.SBProfile.getScaleRadius()
        else:
            return self._data["scale_radius"]
        
    # Set the scale radius, then update the _last_size_set_was_half_light_radius switch AND the
    # _SBProfile.  The latter is rebuilt as necessary on first access after changes in param values.
    def _set_scale_radius(self, value):
        self._data["scale_radius"] = value
        self._data["half_light_radius"] = None
        self._last_size_set_was_half_light_radius = False
        self._SBProfile = None  # Make sure that the ._SBProfile storage is emptied too

    # Then we define the scale_radius descriptor with reference to these getter/setter functions
    scale_radius = descriptors.GetSetFuncParam(
        getter=_get_scale_radius, setter=_set_scale_radius, group="size",
        doc="Moffat scale radius parameter, kept updated with the other size attributes.")

    # Getter and setter functions for the half_light_radius descriptor
    # These are both defined in close analogy to the scale_radius, having mirror/inverse behaviour
    def _get_half_light_radius(self):
        if self._last_size_set_was_half_light_radius is True:
            return self._data["half_light_radius"]
        else:
            return self.SBProfile.getHalfLightRadius()

    def _set_half_light_radius(self, value):
        self._data["half_light_radius"] = value
        self._data["scale_radius"] = None
        self._last_size_set_was_half_light_radius = True
        self._SBProfile = None

    # Then we define the half_light_radius descriptor with ref. to these getter/setter functions
    half_light_radius = descriptors.GetSetFuncParam(
        getter=_get_half_light_radius, setter=_set_half_light_radius, group="size",
        doc="Half light radius, kept updated with the other size attributes.")

    # Getter and setter functions for the fwhm
    # The FWHM can be expressed in terms of the scale_radius and beta
    def _get_fwhm(self):
        return self.scale_radius * 2. * np.sqrt(2.**(1. / self.beta) - 1.)

    def _set_fwhm(self, value):
        self.scale_radius = 0.5 * value / np.sqrt(2.**(1. / self.beta) - 1.)

    # Then define the fwhm descriptor with reference to these getter/setter functions
    fwhm = descriptors.GetSetFuncParam(
        getter=_get_fwhm, setter=_set_fwhm,
        doc="FWHM, kept updated with the other size attributes.")

    # --- Defining the function used to (re)-initialize the contained SBProfile as necessary ---
    # *** Note a function of this name and similar content MUST be defined for all GSObjects! ***
    def _SBInitialize(self):
        # Initialize the GSObject differently depending on whether the HLR was set last.
        if self._last_size_set_was_half_light_radius is True:
            GSObject.__init__(
                self, galsim.SBMoffat(
                    self.beta, half_light_radius=self.half_light_radius, trunc=self.trunc,
                    flux=self.flux))
        else:
            GSObject.__init__(
                self, galsim.SBMoffat(
                    self.beta, scale_radius=self.scale_radius, trunc=self.trunc, flux=self.flux))

    # --- Public Class methods ---
    def __init__(self, beta, scale_radius=None, half_light_radius=None,  fwhm=None, trunc=0.,
                 flux=1.):
        
        # Set the beta and truncation parameters
        self.beta = beta
        self.trunc = trunc

        # Use the RadialProfile._parse_sizes() method to initialize size parameters
        RadialProfile._parse_sizes(
            self, scale_radius=scale_radius, fwhm=fwhm, half_light_radius=half_light_radius)

        # Set the flux
        self.flux = flux

        # Then build the SBProfile
        self._SBInitialize()


class DoubleGaussian(GSObject):
    """Double Gaussian, which is the sum of two Gaussian profiles and has an SBAdd in the SBProfile
    attribute.

    For more details of the Gaussian Surface Brightness profile, please see the SBGaussian
    documentation produced by doxygen.

    Initialization
    --------------
    Each component of the DoubleGaussian is initialized using a flux parameter (flux1 and flux2),
    and one of three possible size parameters

        half_light_radius1
        sigma1
        fwhm1

    (for the first component) and
  
        half_light_radius2
        sigma2
        fwhm2

    (for the second component).

    Example:
    >>> dgauss_obj = Gaussian(flux1=3., flux2=1., sigma1=1., sigma2=0.5)
    >>> dgauss_obj.half_light_radius1
    1.1774100225154747
    >>> dgauss_obj.half_light_radius1 = 1.
    >>> dgauss_obj.sigma1
    0.8493218002880191

    Attempting to initialize with more than one size parameter for each component is ambiguous,
    and will raise a TypeError exception.

    Methods
    -------
    The DoubleGaussian is a GSObject, and inherits all of the GSObject methods (draw, drawShoot,
    applyShear etc.) and operator bindings.
    """

    flux1 = descriptors.SimpleParam(
        "flux1", group="optional", default=None,
        doc="Flux for the first of the two Gaussian components of the DoubleGaussian.")

    flux2 = descriptors.SimpleParam(
        "flux1", group="optional", default=None,
        doc="Flux for the second of the two Gaussian components of the DoubleGaussian.")

    half_light_radius1 = descriptors.SimpleParam(
        "half_light_radius1", group="optional",
        doc="Half light radius for the first of the two Gaussian components of the "+
        "DoubleGaussian, kept updated with the other size attributes.")

    half_light_radius2 = descriptors.SimpleParam(
        "half_light_radius2", group="optional",
        doc="Half light radius for the second of the two Gaussian components of the "+
        "DoubleGaussian, kept updated with the other size attributes.")

    sigma1 = descriptors.GetSetScaleParam(
        "sigma1", root_name="half_light_radius1", group="optional",
        factor=1./1.1774100225154747, # factor = 1 / sqrt[2ln(2)]
        doc="Scale radius sigma for the first of the two Gaussian components of the "+
        "DoubleGaussian, kept updated with the other size attributes.")

    sigma2 = descriptors.GetSetScaleParam(
        "sigma2", root_name="half_light_radius2", group="optional",
        factor=1./1.1774100225154747, # factor = 1 / sqrt[2ln(2)]d v
        doc="Scale radius sigma for the second of the two Gaussian components of the "+
        "DoubleGaussian, kept updated with the other size attributes.")

    fwhm1 = descriptors.GetSetScaleParam(
        name="fwhm1", root_name="half_light_radius1", factor=2., # strange but true...
        group="optional", doc="FWHM for the first of the two Gaussian components of the "+
        "DoubleGaussian, kept consistent with the other size attributes.")

    fwhm2 = descriptors.GetSetScaleParam(
        name="fwhm2", root_name="half_light_radius2", factor=2., # strange but true...
        group="optional", doc="FWHM for the second of the two Gaussian components of the "+
        "DoubleGaussian, kept consistent with the other size attributes.")

    def _parse_sizes(self, **kwargs):
        """
        Convenience function to parse input size parameters within the derived class __init__
        method.  Raises an exception if more than one input parameter kwarg is set != None.
        """
        size_set = False
        for name, value in kwargs.iteritems():
            if value != None:
                if size_set is True:
                    raise TypeError(
                        "Cannot specify more than one size parameter for each component of the "+
                        "DoubleGaussian.")
                else:
                    self.__setattr__(name, value)
                    size_set = True
        if size_set is False:
            raise TypeError("Must specify at least one size parameter for each component of the "+
                            "DoubleGaussian.")

    # --- Defining the function used to (re)-initialize the contained SBProfile as necessary ---
    # *** Note a function of this name and similar content MUST be defined for all GSObjects! ***
    def _SBInitialize(self):
        sblist = [galsim.Gaussian(sigma=self.sigma1, flux=self.flux1),
                  galsim.Gaussian(sigma=self.sigma2, flux=self.flux2)]
        GSObject.__init__(self, galsim.Add(sblist))

    # --- Public Class methods ---
    def __init__(self, flux1, flux2, sigma1=None, sigma2=None, fwhm1=None, fwhm2=None,
                 half_light_radius1=None, half_light_radius2=None):

        # Parse both sets of size parameters using the DoubleGaussian's modified _parse_sizes method
        self._parse_sizes(half_light_radius1=half_light_radius1, sigma1=sigma1, fwhm1=fwhm1)
        self._parse_sizes(half_light_radius2=half_light_radius2, sigma2=sigma2, fwhm2=fwhm2)

        # Set the fluxes
        self.flux1 = flux1
        self.flux2 = flux2

        # Then build the SBProfile
        self._SBInitialize()


class AtmosphericPSF(RadialProfile):
    """Base class for long exposure Kolmogorov PSF.

    Initialization
    --------------
    @code
    atmospheric_psf = galsim.AtmosphericPSF(lam_over_r0, interpolantxy=None, oversampling=1.5)
    @endcode

    Initialized atmospheric_psf as a galsim.AtmosphericPSF() instance.

    @param lam_over_r0     lambda / r0 in the physical units adopted (user responsible for 
                           consistency), where r0 is the Fried parameter. The FWHM of the Kolmogorov
                           PSF is ~0.976 lambda/r0 (e.g., Racine 1996, PASP 699, 108). Typical 
                           values for the Fried parameter are on the order of 10 cm for most 
                           observatories and up to 20 cm for excellent sites. The values are 
                           usually quoted at lambda = 500 nm and r0 depends on wavelength as
                           [r0 ~ lambda^(-6/5)].
    @param fwhm            FWHM of the Kolmogorov PSF.
                           Either fwhm or lam_over_r0 (and only one) must be specified.
    @param oversampling    optional oversampling factor for the SBInterpolatedImage table 
                           [default = 1.5], setting oversampling < 1 will produce aliasing in the 
                           PSF (not good).
    """

    # First we define the size parameters for the AtmosphericPSF:
    # The basic, underlying size parameter lambda / r0
    lam_over_r0 = descriptors.SimpleParam(
        "lam_over_r0", group="size", default=None,
        doc="lam_over_r0, kept consistent with the other size attributes.")

    # The FWHM of the Kolmogorov PSF is ~0.976 lambda/r0 (e.g., Racine 1996, PASP 699, 108)
    fwhm = descriptors.GetSetScaleParam(
        "fwhm", root_name="lam_over_r0", factor=0.976, group="size",
        doc="FWHM, kept consistent with the other size attributes.")

    # Getter and setter functions for the half_light_radius descriptor (raising a
    # NotImplementedError exception for this not-yet-implemented).  Note this overrides the
    # half_light_radius inherited from the RadialProfile base class 
    def _get_half_light_radius(self):
        raise NotImplementedError(
            "Half light radius calculation not yet implemented for AtmosphericPSF objects.")
    def _set_half_light_radius(self, value):
        raise NotImplementedError(
            "Half light radius support not yet implemented for AtmosphericPSF objects.")
    
    # Then we define the half_light_radius descriptor with ref. to these getter/setter functions
    half_light_radius = descriptors.GetSetFuncParam(
        getter=_get_half_light_radius, setter=_set_half_light_radius, group="size",
        doc="Half light radius, access will raise a NotImplementedError exception!")

    # Then the optional parameters interpolant and oversampling
    interpolant = descriptors.SimpleParam(
        "interpolant", default=None, group="optional", doc="Real space interpolant instance (2D).")
    oversampling = descriptors.SimpleParam(
        "oversampling", default=1.5, group="optional",
        doc="Oversampling factor for the creation of the SBInterpolatedImage lookup table.")

    # --- Defining the function used to (re)-initialize the contained SBProfile as necessary ---
    # *** Note a function of this name and similar content MUST be defined for all GSObjects! ***
    def _SBInitialize(self):
        
        # Set the lookup table sample rate via FWHM / 2 / oversampling (BARNEY: is this enough??)
        dx_lookup = .5 * self.fwhm / self.oversampling

        # Fold at 10 times the FWHM
        stepk_kolmogorov = np.pi / (10. * self.fwhm)

        # Odd array to center the interpolant on the centroid. Might want to pad this later to
        # make a nice size array for FFT, but for typical seeing, arrays will be very small.
        npix = 1 + 2 * (np.ceil(np.pi / stepk_kolmogorov)).astype(int)
        atmoimage = galsim.atmosphere.kolmogorov_psf_image(array_shape=(npix, npix), dx=dx_lookup, 
                                                           lam_over_r0=self.lam_over_r0)
        # Run checks on the interpolant and build default if None
        if self.interpolant is None:
            lan5 = galsim.Lanczos(5, conserve_flux=True, tol=1e-4)
            self.interpolant = galsim.InterpolantXY(lan5)
        else:
            if isinstance(self.interpolant, galsim.InterpolantXY) is False:
                raise RuntimeError('Specified interpolant is not an InterpolantXY!')

        # Then initialize the SBProfile
        GSObject.__init__(
            self, galsim.SBInterpolatedImage(atmoimage, self.interpolant, dx=dx_lookup))

    # --- Public Class methods ---
    def __init__(self, lam_over_r0=None, fwhm=None, interpolant=None, oversampling=1.5):

        # Initialize the interpolant and oversampling parameters
        self.interpolant = interpolant
        self.oversampling = oversampling
        
        # Use the RadialProfile._parse_sizes() method to initialize size parameters
        RadialProfile._parse_sizes(self, lam_over_r0=lam_over_r0, fwhm=fwhm)

        # Then build the SBProfile
        self._SBInitialize()
        

class RealGalaxy(GSObject):
    """@brief Class describing real galaxies from some training dataset.

    This class uses a catalog describing galaxies in some training data to read in data about
    realistic galaxies that can be used for simulations based on those galaxies.  Also included in
    the class is additional information that might be needed to make or interpret the simulations,
    e.g., the noise properties of the training data.

    Initialization
    --------------
    @code
    real_galaxy = galsim.RealGalaxy(real_galaxy_catalog, index = None, ID = None, ID_string = None,
                                    random = False, uniform_deviate = None, interpolant = None)
    @endcode

    This initializes real_galaxy with three SBInterpolatedImage objects (one for the deconvolved
    galaxy, and saved versions of the original HST image and PSF). Note that there are multiple
    keywords for choosing a galaxy; exactly one must be set.  In future we may add more such
    options, e.g., to choose at random but accounting for the non-constant weight factors
    (probabilities for objects to make it into the training sample).

    @param real_galaxy_catalog  A RealGalaxyCatalog object with basic information about where to
                                find the data, etc.
    @param index                Index of the desired galaxy in the catalog.
    @param ID                   Object ID for the desired galaxy in the catalog.
    @param random               If true, then just select a completely random galaxy from the
                                catalog.
    @param uniform_deviate      A uniform deviate to use for selecting a random galaxy (optional)
    @param interpolant          optional keyword for specifying the
                                real-space interpolation scheme
                                [default = galsim.InterpolantXY(galsim.Lanczos(5, 
                                           conserve_flux=True, tol=1.e-4))].
    """

    # Define the parameters that need to be set as SimpleParams to define the RealGalaxy
    real_galaxy_catalog = descriptors.SimpleParam(
        "real_galaxy_catalog", default=None, group="required",
        doc="RealGalaxyCatalog object with basic information about where to find data for each "+
        "RealGalaxy instance.")
    index = descriptors.SimpleParam(
        "index", default=None, group="optional", doc="Index of the desired galaxy in the catalog.")
    ID = descriptors.SimpleParam(
        "ID", default=None, group="optional",
        doc="Object ID for the desired galaxy in the catalog.")
    random = descriptors.SimpleParam(
        "random", default=False, group="optional", doc="Whether galaxy selected at random.")
    uniform_deviate = descriptors.SimpleParam(
        "uniform_deviate", default=None, group="optional",
        doc="Uniform deviate to use for random galaxy selection.")
    interpolant = descriptors.SimpleParam(
        "interpolant", default=None, group="optional", doc="Real space interpolant instance (2D).")

    # --- Defining the function used to (re)-initialize the contained SBProfile as necessary ---
    # *** Note a function of this name and similar content MUST be defined for all GSObjects! ***
    def _SBInitialize(self):

        import pyfits
        # Code block below will be for galaxy selection; not all are currently implemented.  Each
        # option must return an index within the real_galaxy_catalog.
        if self.index != None:
            if (self.ID != None or self.random == True):
                raise RuntimeError('Too many methods for selecting a galaxy!')
        elif self.ID != None:
            if (self.random == True):
                raise RuntimeError('Too many methods for selecting a galaxy!')
            self.index = real_galaxy_catalog.get_index_for_id(ID)
        elif self.random == True:
            if self.uniform_deviate == None:
                self.uniform_deviate = galsim.UniformDeviate()
            self.index = int(real_galaxy_catalog.n * self.uniform_deviate()) 
            # this will round down, to get index in range [0, n-1]
        else:
            raise RuntimeError('No method specified for selecting a galaxy!')
        if self.random == False and self.uniform_deviate != None:
            import warnings
            msg = "Warning: uniform_deviate supplied, but random selection method was not chosen!"
            warnings.warn(msg)

        # read in the galaxy, PSF images; for now, rely on pyfits to make I/O errors. Should
        # consider exporting this code into fits.py in some function that takes a filename and HDU,
        # and returns an ImageView
        gal_image = self.real_galaxy_catalog.getGal(self.index)
        PSF_image = self.real_galaxy_catalog.getPSF(self.index)

        # choose proper interpolant
        if self.interpolant != None and isinstance(self.interpolant, galsim.InterpolantXY) == False:
            raise RuntimeError('Specified interpolant is not an InterpolantXY!')
        elif self.interpolant == None:
            lan5 = galsim.Lanczos(5, conserve_flux=True, tol=1.e-4) # copied from Shera.py!
            self.interpolant = galsim.InterpolantXY(lan5)

        # read in data about galaxy from FITS binary table; store as normal attributes of RealGalaxy
        # and save any other relevant information
        self.catalog_file = self.real_galaxy_catalog.filename
        self.pixel_scale = float(self.real_galaxy_catalog.pixel_scale[self.index])
        # note: will be adding more parameters here about noise properties etc., but let's be basic
        # for now

        self.original_image = galsim.SBInterpolatedImage(
            gal_image, self.interpolant, dx=self.pixel_scale)
        self.original_PSF = galsim.SBInterpolatedImage(
            PSF_image, self.interpolant, dx=self.pixel_scale)
        self.original_PSF.setFlux(1.0)
        psf_inv = galsim.SBDeconvolve(self.original_PSF)
        GSObject.__init__(self, galsim.SBConvolve([self.original_image, psf_inv]))

    # --- Public Class methods ---
    def __init__(self, real_galaxy_catalog, index=None, ID=None, random=False,
                 uniform_deviate=None, interpolant=None):

        # Set the values of the defining params based on the inputs
        self.real_galaxy_catalog = real_galaxy_catalog
        self.index = index
        self.ID = ID
        self.random = random
        self.uniform_deviate = uniform_deviate
        self.interpolant = interpolant

        # Then build the SBProfile
        self._SBInitialize()
     
