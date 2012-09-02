import os
import collections
import numpy as np
import galsim
import utilities

ALIAS_THRESHOLD = 0.005 # Matches hard coded value in src/SBProfile.cpp. TODO: bring these together

class GSObject(object):
    """@brief Base class for defining the interface with which all GalSim Objects access their
    shared methods and attributes, particularly those from the C++ SBProfile classes.
    """

    # --- Initialization ---
    def __init__(self, SBProfile):
        self.SBProfile = SBProfile  # This guarantees that all GSObjects have an SBProfile
    
    # Make op+ of two GSObjects work to return an Add object
    def __add__(self, other):
        return Add(self, other)

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
        # Re-initialize a return GSObject with self's SBProfile
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

    def scaleFlux(self, flux_ratio):
        """@brief Multiply the flux of the object by fluxRatio
           
        After this call, the caller's type will be a GSObject.
        This means that if the caller was a derived type that had extra methods beyond
        those defined in GSObject (e.g. getSigma for a Gaussian), then these methods
        are no longer available.
        """
        self.SBProfile.scaleFlux(flux_ratio)
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
        old_flux = self.getFlux()
        self.applyTransformation(galsim.Ellipse(np.log(scale)))
        self.setFlux(old_flux) # conserve flux

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
        self.applyTransformation(galsim.Ellipse(np.log(scale)))
       
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
            ellipse = galsim.Ellipse(args[0])
        elif len(args) > 1:
            raise TypeError("Error, too many unnamed arguments to applyShear!")
        else:
            ellipse = galsim.Ellipse(galsim.Shear(**kwargs))
        self.applyTransformation(ellipse)

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
        ellipse = galsim.Ellipse(x_shift=dx, y_shift=dy)
        self.applyTransformation(ellipse)

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
        ret = self.copy()
        old_flux = self.getFlux()
        ret.applyTransformation(galsim.Ellipse(np.log(scale)))
        ret.setFlux(old_flux)
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
        ret = self.copy()
        ret.applyTransformation(galsim.Ellipse(np.log(scale)))
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
        assert added_flux > 0.99 * obj.flux
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


# --- Now defining the derived classes ---
#
# All derived classes inherit the GSObject method interface, but therefore have a "has a" 
# relationship with the C++ SBProfile class rather than an "is a" one...
#
# The __init__ method is usually simple and all the GSObject methods & attributes are inherited.
# 
class Gaussian(GSObject):
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
    >>> gauss_obj.getHalfLightRadius()
    1.1774100225154747
    >>> gauss_obj = Gaussian(flux=3, half_light_radius=1.)
    >>> gauss_obj.getSigma()
    0.8493218002880191

    Attempting to initialize with more than one size parameter is ambiguous, and will raise a
    TypeError exception.

    Methods
    -------
    The Gaussian is a GSObject, and inherits all of the GSObject methods (draw, drawShoot,
    applyShear etc.) and operator bindings.
    """
    
    # Initialization parameters of the object, with type information
    _req_params = {}
    _opt_params = { "flux" : float }
    _size_params = { "sigma" : float, "half_light_radius" : float, "fwhm" : float }
    
    # --- Public Class methods ---
    def __init__(self, half_light_radius=None, sigma=None, fwhm=None, flux=1.):
        # Initialize the SBProfile
        GSObject.__init__(
            self, galsim.SBGaussian(
                sigma=sigma, half_light_radius=half_light_radius, fwhm=fwhm, flux=flux))
 
    def getSigma(self):
        """@brief Return the sigma scale length for this Gaussian profile.
        """
        return self.SBProfile.getSigma()
    
    def getFWHM(self):
        """@brief Return the FWHM for this Gaussian profile.
        """
        return self.SBProfile.getSigma() * 2.3548200450309493 # factor = 2 sqrt[2ln(2)]
 
    def getHalfLightRadius(self):
        """@brief Return the half light radius for this Gaussian profile.
        """
        return self.SBProfile.getSigma() * 1.1774100225154747 # factor = sqrt[2ln(2)]


class Moffat(GSObject):
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
    >>> moffat_obj.getHalfLightRadius()
    1.9307827587167474
    >>> moffat_obj = Moffat(beta=3., half_light_radius=1., flux=0.5)
    >>> moffat_obj.getScaleRadius()
    1.5537739740300376

    Attempting to initialize with more than one size parameter is ambiguous, and will raise a
    TypeError exception.

    Methods
    -------
    The Moffat is a GSObject, and inherits all of the GSObject methods (draw, drawShoot,
    applyShear etc.) and operator bindings.
    """

    # Initialization parameters of the object, with type information
    _req_params = { "beta" : float }
    _opt_params = { "trunc" : float , "flux" : float }
    _size_params = { "scale_radius" : float, "half_light_radius" : float, "fwhm" : float }

    # --- Public Class methods ---
    def __init__(self, beta, scale_radius=None, half_light_radius=None,  fwhm=None, trunc=0.,
                 flux=1.):
        GSObject.__init__(
            self, galsim.SBMoffat(
                beta, scale_radius=scale_radius, half_light_radius=half_light_radius, fwhm=fwhm,
                trunc=trunc, flux=flux))

    def getBeta(self):
        """@brief Return the beta parameter for this Moffat profile.
        """
        return self.SBProfile.getBeta()

    def getScaleRadius(self):
        """@brief Return the scale radius for this Moffat profile.
        """
        return self.SBProfile.getScaleRadius()
        
    def getFWHM(self):
        """@brief Return the FWHM for this Moffat profile.
        """
        return self.SBProfile.getFWHM()

    def getHalfLightRadius(self):
        """@brief Return the half light radius for this Moffat profile.
        """
        return self.SBProfile.getHalfLightRadius()


class AtmosphericPSF(GSObject):
    """Base class for long exposure Kolmogorov PSF.

    Initialization
    --------------
    @code
    atmospheric_psf = galsim.AtmosphericPSF(lam_over_r0, interpolant=None, oversampling=1.5)
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
    @param interpolant     optional keyword for specifying the interpolation scheme [default =
                           galsim.InterpolantXY(galsim.Quintic(tol=1.e-4))]
    @param oversampling    optional oversampling factor for the SBInterpolatedImage table 
                           [default = 1.5], setting oversampling < 1 will produce aliasing in the 
                           PSF (not good).
    @param flux            total flux of the profile [default flux=1.]
    """

    # Initialization parameters of the object, with type information
    _req_params = {}
    _opt_params = { "flux" : float , "oversampling" : float }
    _size_params = { "lam_over_r0" : float , "fwhm" : float }

    # --- Public Class methods ---
    def __init__(self, lam_over_r0=None, fwhm=None, interpolant=None, oversampling=1.5, flux=1.):

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
            if isinstance(interpolant, galsim.InterpolantXY) is False:
                raise RuntimeError('Specified interpolant is not an InterpolantXY!')
            self.interpolant = interpolant

        # Then initialize the SBProfile
        GSObject.__init__(
            self, galsim.SBInterpolatedImage(atmoimage, self.interpolant, dx=dx_lookup))

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
    """GalSim Airy, which has an SBAiry in the SBProfile attribute.

    For more details of the Airy Surface Brightness profile, please see the SBAiry documentation
    produced by doxygen.

    Initialization
    --------------
    An Airy can be initialized using one size parameter lam_over_D, an optional obscuration
    parameter [default obscuration=0.] and an optional flux parameter [default flux = 1].  The
    half light radius or FWHM can subsequently be calculated using the getHalfLightRadius() method
    or getFWHM(), respectively, if obscuration = 0.

    Example:
    >>> airy_obj = Airy(flux=3., lam_over_D=2.)
    >>> airy_obj.getHalfLightRadius()
    1.0696642954485294

    Attempting to initialize with more than one size parameter is ambiguous, and will raise a
    TypeError exception.

    Methods
    -------
    The Airy is a GSObject, and inherits all of the GSObject methods (draw, drawShoot, applyShear
    etc.) and operator bindings.
    """
    
    # Initialization parameters of the object, with type information
    _req_params = {}
    _opt_params = { "flux" : float , "obscuration" : float }
    _size_params = { "lam_over_D" : float }

    # --- Public Class methods ---
    def __init__(self, lam_over_D, obscuration=0., flux=1.):
        GSObject.__init__(
            self, galsim.SBAiry(lam_over_D=lam_over_D, obscuration=obscuration, flux=flux))

    def getHalfLightRadius(self):
        """Return the half light radius of this Airy profile (only supported for obscuration = 0.).
        """
        if self.SBProfile.getObscuration() == 0.:
            # For an unobscured Airy, we have the following factor which can be derived using the
            # integral result given in the Wikipedia page (http://en.wikipedia.org/wiki/Airy_disk),
            # solved for half total flux using the free online tool Wolfram Alpha.
            # At www.wolframalpha.com:
            # Type "Solve[BesselJ0(x)^2+BesselJ1(x)^2=1/2]" ... and divide the result by pi
            return self.SBProfile.getLamOverD() * 0.5348321477242647
        else:
            # In principle can find the half light radius as a function of lam_over_D and
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
            # In principle can find the FWHM as a function of lam_over_D and obscuration too,
            # but it will be much more involved...!
            raise NotImplementedError("FWHM calculation not implemented for Airy "+
                                      "objects with non-zero obscuration.")

    def getLamOverD(self):
        """Return the lam_over_D parameter of this Airy profile.
        """
        return self.SBProfile.getLamOverD()


class Kolmogorov(GSObject):
    """@brief GalSim Kolmogorov, which has an SBKolmogorov in the SBProfile attribute.
       
    Represents a long exposure Kolmogorov PSF.

    Initialization
    --------------
    @code
    psf = galsim.Kolmogorov(lam_over_r0, flux=1.)
    @endcode

    Initialized psf as a galsim.Kolmogorov() instance.

    @param lam_over_r0        lambda / r0 in the physical units adopted (user responsible for 
                              consistency), where r0 is the Fried parameter. The FWHM of the
                              Kolmogorov PSF is ~0.976 lambda/r0 (e.g., Racine 1996, PASP 699, 108).
                              Typical values for the Fried parameter are on the order of 10 cm for
                              most observatories and up to 20 cm for excellent sites. The values are
                              usually quoted at lambda = 500 nm and r0 depends on wavelength as
                              [r0 ~ lambda^(-6/5)].
    @param fwhm               FWHM of the Kolmogorov PSF.
    @param half_light_radius  Half-light radius of the Kolmogorov PSF.
                              One of lam_over_r0, fwhm and half_light_radius (and only one) 
                              must be specified.
    @param flux               Optional flux value [default = 1].
    """
    
    # The FWHM of the Kolmogorov PSF is ~0.976 lambda/r0 (e.g., Racine 1996, PASP 699, 108).
    # In SBKolmogorov.cpp we refine this factor to 0.975865
    _fwhm_factor = 0.975865
    # Similarly, SBKolmogorov calculates the relation between lambda/r0 and half-light radius
    _hlr_factor = 0.554811

    # Initialization parameters of the object, with type information
    _req_params = {}
    _opt_params = { "flux" : float }
    _size_params = { "lam_over_r0" : float, "fwhm" : float, "half_light_radius" : float }

    # --- Public Class methods ---
    def __init__(self, lam_over_r0=None, fwhm=None, half_light_radius=None, flux=1.):

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

        GSObject.__init__(self, galsim.SBKolmogorov(lam_over_r0=lam_over_r0, flux=flux))

    def getLamOverR0(self):
        """Return the lam_over_r0 parameter of this Kolmogorov profile.
        """
        return self.SBProfile.getLamOverR0()
    
    def getFWHM(self):
        """Return the FWHM of this Kolmogorov profile
        """
        return self.SBProfile.getLamOverR0() * Kolmogorov._fwhm_factor

    def getHalfLightRadius(self):
        """Return the half light radius of this Kolmogorov profile
        """
        return self.SBProfile.getLamOverR0() * Kolmogorov._hlr_factor


class OpticalPSF(GSObject):
    """@brief Class describing aberrated PSFs due to telescope optics, which has an
    SBInterpolatedImage in the SBProfile attribute.

    Input aberration coefficients are assumed to be supplied in units of incident light wavelength,
    and correspond to the conventions adopted here:
    http://en.wikipedia.org/wiki/Optical_aberration#Zernike_model_of_aberrations

    Initialization
    --------------
    @code
    optical_psf = galsim.OpticalPSF(lam_over_D, defocus=0., astig1=0., astig2=0., coma1=0.,
                                        coma2=0., spher=0., circular_pupil=True, obscuration=0.,
                                        interpolant=None, oversampling=1.5, pad_factor=1.5)
    @endcode

    Initializes optical_psf as a galsim.OpticalPSF() instance.

    @param lam_over_D      lambda / D in the physical units adopted (user responsible for 
                           consistency).
    @param defocus         defocus in units of incident light wavelength.
    @param astig1          first component of astigmatism (like e1) in units of incident light
                           wavelength.
    @param astig2          second component of astigmatism (like e2) in units of incident light
                           wavelength.
    @param coma1           coma along x in units of incident light wavelength.
    @param coma2           coma along y in units of incident light wavelength.
    @param spher           spherical aberration in units of incident light wavelength.
    @param circular_pupil  adopt a circular pupil? Alternative is square.
    @param obscuration     linear dimension of central obscuration as fraction of pupil linear 
                           dimension, [0., 1.) [default = 0.].
    @param interpolant     optional keyword for specifying the interpolation scheme [default =
                           galsim.InterpolantXY(galsim.Quintic(tol=1.e-4))].
    @param oversampling    optional oversampling factor for the SBInterpolatedImage table 
                           [default = 1.5], setting oversampling < 1 will produce aliasing in the 
                           PSF (not good).
    @param pad_factor      additional multiple by which to zero-pad the PSF image to avoid folding
                           compared to what would be required for a simple Airy [default = 1.5].
                           Note that pad_factor may need to be increased for stronger aberrations,
                           i.e. those larger than order unity.
    @param flux            total flux of the profile [default flux=1.].
    """

    # Initialization parameters of the object, with type information
    _req_params = {}
    _opt_params = {
        "defocus" : float ,
        "astig1" : float ,
        "astig2" : float ,
        "coma1" : float ,
        "coma2" : float ,
        "spher" : float ,
        "circular_pupil" : bool ,
        "obscuration" : float ,
        "oversampling" : float ,
        "pad_factor" : float ,
        "flux" : float }
    _size_params = { "lam_over_D" : float }

    # --- Public Class methods ---
    def __init__(self, lam_over_D, defocus=0., astig1=0., astig2=0., coma1=0., coma2=0., spher=0.,
                 circular_pupil=True, obscuration=0., interpolant=None, oversampling=1.5,
                 pad_factor=1.5, flux=1.):

        # Currently we load optics, noise etc in galsim/__init__.py, but this might change (???)
        import galsim.optics
        
        # Choose dx for lookup table using Nyquist for optical aperture and the specified
        # oversampling factor
        dx_lookup = .5 * lam_over_D / oversampling
        
        # Use a similar prescription as SBAiry to set Airy stepK and thus reference unpadded image
        # size in physical units
        stepk_airy = min(
            ALIAS_THRESHOLD * .5 * np.pi**3 * (1. - obscuration) / lam_over_D,
            np.pi / 5. / lam_over_D)
        
        # Boost Airy image size by a user-specifed pad_factor to allow for larger, aberrated PSFs,
        # also make npix always *odd* so that opticalPSF lookup table array is correctly centred:
        npix = 1 + 2 * (np.ceil(pad_factor * (np.pi / stepk_airy) / dx_lookup)).astype(int)
        
        # Make the psf image using this dx and array shape
        optimage = galsim.optics.psf_image(
            lam_over_D=lam_over_D, dx=dx_lookup, array_shape=(npix, npix), defocus=defocus,
            astig1=astig1, astig2=astig2, coma1=coma1, coma2=coma2, spher=spher,
            circular_pupil=circular_pupil, obscuration=obscuration, flux=flux)
        
        # If interpolant not specified on input, use a high-ish n lanczos
        if interpolant == None:
            quintic = galsim.Quintic(tol=1.e-4)
            self.interpolant = galsim.InterpolantXY(quintic)
        else:
            if isinstance(self.interpolant, galsim.InterpolantXY) is False:
                raise RuntimeError('Specified interpolant is not an InterpolantXY!')
            self.interpolant = interpolant
            
        # Initialize the SBProfile
        GSObject.__init__(
            self, galsim.SBInterpolatedImage(optimage, self.interpolant, dx=dx_lookup))

        # The above procedure ends up with a larger image than we really need, which
        # means that the default stepK value will be smaller than we need.  
        # Thus, we call the function calculateStepK() to refine the value.
        self.SBProfile.calculateStepK()
        self.SBProfile.calculateMaxK()


class Pixel(GSObject):
    """@brief GalSim Pixel, which has an SBBox in the SBProfile attribute.

    Initialization
    --------------
    A Pixel is initialized with an x dimension width xw, an optional y dimension width (if
    unspecifed yw=xw is assumed) and an optional flux parameter [default flux = 1].

    Methods
    -------
    The Pixel is a GSObject, and inherits all of the GSObject methods (draw, drawShoot, applyShear
    etc.) and operator bindings.
    """

    # Initialization parameters of the object, with type information
    _req_params = {}
    _opt_params = { "yw" : float , "flux" : float }
    _size_params = { "xw" : float }

    # --- Public Class methods ---
    def __init__(self, xw, yw=None, flux=1.):
        if yw is None:
            yw = xw
        GSObject.__init__(self, galsim.SBBox(xw=xw, yw=yw, flux=flux))

    def getXWidth(self):
        """@brief Return the width of the pixel in the x dimension.
        """
        return self.SBProfile.getXWidth()

    def getYWidth(self):
        """@brief Return the width of the pixel in the y dimension.
        """
        return self.SBProfile.getYWidth()


class Sersic(GSObject):
    """GalSim Sersic, which has an SBSersic in the SBProfile attribute.

    For more details of the Sersic Surface Brightness profile, please see the SBSersic documentation
    produced by doxygen.

    Initialization
    --------------
    A Sersic is initialized with n, the Sersic index of the profile, and the half light radius size
    parameter half_light_radius.  A flux parameter is optional [default flux = 1].

    Example:
    >>> sersic_obj = Sersic(n=3.5, half_light_radius=2.5, flux=40.)
    >>> sersic_obj.getHalfLightRadius()
    2.5
    >>> sersic_obj.getN()
    3.5

    Methods
    -------
    The Sersic is a GSObject, and inherits all of the GSObject methods (draw, drawShoot,
    applyShear etc.) and operator bindings.
    """

    # Initialization parameters of the object, with type information
    _req_params = { "n" : float }
    _opt_params = { "flux" : float }
    _size_params = { "half_light_radius" : float }

    # --- Public Class methods ---
    def __init__(self, n, half_light_radius, flux=1.):
        GSObject.__init__(
            self, galsim.SBSersic(n, half_light_radius=half_light_radius, flux=flux))

    def getN(self):
        """@brief Return the Sersic index for this profile.
        """
        return self.SBProfile.getN()

    def getHalfLightRadius(self):
        """@brief Return the half light radius for this Sersic profile.
        """
        return self.SBProfile.getHalfLightRadius()


class Exponential(GSObject):
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
    >>> exp_obj.getHalfLightRadius()
    8.391734950083302
    >>> exp_obj = Exponential(flux=3., half_light_radius=1.)
    >>> exp_obj.getScaleRadius()
    0.5958243473776976

    Attempting to initialize with more than one size parameter is ambiguous, and will raise a
    TypeError exception.

    Methods
    -------
    The Exponential is a GSObject, and inherits all of the GSObject methods (draw, drawShoot,
    applyShear etc.) and operator bindings.
    """

    # Initialization parameters of the object, with type information
    _req_params = {}
    _opt_params = { "flux" : float }
    _size_params = { "scale_radius" : float , "half_light_radius" : float }

    # --- Public Class methods ---
    def __init__(self, half_light_radius=None, scale_radius=None, flux=1.):
        GSObject.__init__(
            self, galsim.SBExponential(
                half_light_radius=half_light_radius, scale_radius=scale_radius, flux=flux))

    def getScaleRadius(self):
        """@brief Return the scale radius for this Exponential profile.
        """
        return self.SBProfile.getScaleRadius()

    def getHalfLightRadius(self):
        """@brief Return the half light radius for this Exponential profile.
        """
        # Factor not analytic, but can be calculated by iterative solution of equation:
        #  (re / r0) = ln[(re / r0) + 1] + ln(2)
        return self.SBProfile.getScaleRadius() * 1.6783469900166605


class DeVaucouleurs(GSObject):
    """GalSim DeVaucouleurs, which has an SBDeVaucouleurs in the SBProfile attribute.

    For more details of the DeVaucouleurs Surface Brightness profile, please see the
    SBDeVaucouleurs documentation produced by doxygen.

    Initialization
    --------------
    A DeVaucouleurs is initialized with the half light radius size parameter half_light_radius and
    an optional flux parameter [default flux = 1].

    Example:
    >>> dvc_obj = DeVaucouleurs(half_light_radius=2.5, flux=40.)
    >>> dvc_obj.getHalfLightRadius()
    2.5
    >>> dvc_obj.getFlux()
    40.0

    Methods
    -------
    The DeVaucouleurs is a GSObject, and inherits all of the GSObject methods (draw, drawShoot,
    applyShear etc.) and operator bindings.
    """

    # Initialization parameters of the object, with type information
    _req_params = {}
    _opt_params = { "flux" : float }
    _size_params = { "half_light_radius" : float }

    # --- Public Class methods ---
    def __init__(self, half_light_radius=None, flux=1.):
        GSObject.__init__(
            self, galsim.SBDeVaucouleurs(half_light_radius=half_light_radius, flux=flux))

    def getHalfLightRadius(self):
        """@brief Return the half light radius for this DeVaucouleurs profile.
        """
        return self.SBProfile.getHalfLightRadius()

     
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
    @param flux                 Total flux, if None then original flux in galaxy is adopted without
                                change [default flux = None].
    """

    # Initialization parameters of the object, with type information
    _req_params = {}
    _opt_params = { "index" : int , "flux" : float }
    _size_params = {}

    # --- Public Class methods ---
    def __init__(self, real_galaxy_catalog, index=None, ID=None, random=False,
                 uniform_deviate=None, interpolant=None, flux=None):

        import pyfits

        # Code block below will be for galaxy selection; not all are currently implemented.  Each
        # option must return an index within the real_galaxy_catalog.        
        use_index = None # using -1 here for 'safety' actually indexes in Python without complaint
        if index != None:
            if (ID != None or random == True):
                raise RuntimeError('Too many methods for selecting a galaxy!')
            use_index = index
        elif ID != None:
            if (random == True):
                raise RuntimeError('Too many methods for selecting a galaxy!')
            use_index = real_galaxy_catalog.get_index_for_id(ID)
        elif random == True:
            if uniform_deviate == None:
                uniform_deviate = galsim.UniformDeviate()
            use_index = int(real_galaxy_catalog.n * uniform_deviate()) 
            # this will round down, to get index in range [0, n-1]
        else:
            raise RuntimeError('No method specified for selecting a galaxy!')
        if random == False and uniform_deviate != None:
            import warnings
            msg = "Warning: uniform_deviate supplied, but random selection method was not chosen!"
            warnings.warn(msg)

        # read in the galaxy, PSF images; for now, rely on pyfits to make I/O errors. Should
        # consider exporting this code into fits.py in some function that takes a filename and HDU,
        # and returns an ImageView

        gal_image = real_galaxy_catalog.getGal(use_index)
        PSF_image = real_galaxy_catalog.getPSF(use_index)

        # choose proper interpolant
        if interpolant == None:
            lan5 = galsim.Lanczos(5, conserve_flux=True, tol=1.e-4) # copied from Shera.py!
            self.interpolant = galsim.InterpolantXY(lan5)
        else:
            if not isinstance(interpolant, galsim.InterpolantXY):
                raise RuntimeError('Specified interpolant is not an InterpolantXY!')
            self.interpolant = interpolant

        # read in data about galaxy from FITS binary table; store as normal attributes of RealGalaxy

        # save any other relevant information as instance attributes
        self.catalog_file = real_galaxy_catalog.filename
        self.index = use_index
        self.pixel_scale = float(real_galaxy_catalog.pixel_scale[use_index])
        # note: will be adding more parameters here about noise properties etc., but let's be basic
        # for now

        # save the original image and PSF too
        self.original_image = galsim.SBInterpolatedImage(
            gal_image, self.interpolant, dx=self.pixel_scale)
        self.original_PSF = galsim.SBInterpolatedImage(
            PSF_image, self.interpolant, dx=self.pixel_scale)
        
        if flux != None:
            self.original_image.setFlux(flux)
            self.original_image.__class__ = galsim.SBTransform # correctly reflect SBProfile change
        self.original_PSF.setFlux(1.0)
        self.original_PSF.__class__ = galsim.SBTransform # correctly reflect SBProfile change

        # Calculate the PSF "deconvolution" kernel
        psf_inv = galsim.SBDeconvolve(self.original_PSF)
        # Initialize the SBProfile attribute
        GSObject.__init__(self, galsim.SBConvolve([self.original_image, psf_inv]))

    def getHalfLightRadius(self):
        raise NotImplementedError("Half light radius calculation not implemented for RealGalaxy "
                                   +"objects.")

#
# --- Compound GSObect classes: Add and Convolve ---

class Add(GSObject):
    """@brief Base class for defining the python interface to the SBAdd C++ class.
    """
    
    # --- Public Class methods ---
    def __init__(self, *args):

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
            GSObject.__init__(self, galsim.SBAdd(SBList))
        elif len(args) >= 2:
            # >= 2 arguments.  Convert to a list of SBProfiles
            SBList = [obj.SBProfile for obj in args]
            GSObject.__init__(self, galsim.SBAdd(SBList))

class Convolve(GSObject):
    """@brief A class for convolving 2 or more GSObjects.

    The objects to be convolved may be provided either as multiple unnamed arguments
    (e.g. Convolve(psf,gal,pix)) or as a list (e.g. Convolve[psf,gal,pix]).
    Any number of objects may be provided using either syntax.  (Even 0 or 1, although
    that doesn't really make much sense.)
   
    The convolution will normally be done using discrete Fourier transforms of 
    each of the component profiles, multiplying them together, and then transforming
    back to real space.
   
    The stepK used for the k-space image will be (Sum 1/stepK()^2)^(-1/2)
    where the sum is over all the components being convolved.  Since the size of 
    the convolved image scales roughly as the quadrature sum of the components,
    this should be close to Pi/Rmax where Rmax is the radius that encloses
    all but (1-alias_threshold) of the flux in the final convolved image..
    
    The maxK used for the k-space image will be the minimum of the maxK calculated for
    each component.  Since the k-space images are multiplied, if one of them is 
    essentially zero beyond some k value, then that will be true of the final image
    as well.
    
    There is also an option to do the convolution as integrals in real space.
    To do this, use the optional keyword argument real_space=True.
    Currently, the real-space integration is only enabled for 2 profiles.
    (Aside from the trivial implementaion for 1 profile.)  If you try to use it 
    for more than 2 profiles, an exception will be raised.
    
    The real-space convolution is normally slower than the DFT convolution.
    The exception is if both component profiles have hard edges.  e.g. a truncated
    Moffat with a Pixel.  In that case, the maxK for each component is quite large
    since the ringing dies off fairly slowly.  So it can be quicker to use 
    real-space convolution instead.  Also, real-space convolution tends to be more
    accurate in this case as well.

    If you do not specify either True or False explicitly, then we check if 
    there are 2 profiles, both of which have hard edges.  In this case, we 
    automatically use real-space convolution.  In all other cases, the 
    default is to use the DFT algorithm.
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
        # The only kwarg we're looking for is real_space, which can be True or False
        # (default if omitted is None), which specifies whether to do the convolution
        # as an integral in real space rather than as a product in fourier space.
        # If the parameter is omitted (or explicitly given as None I guess), then
        # we will usually do the fourier method.  However, if there are 2 components
        # _and_ both of them have hard edges, then we use real-space convolution.
        real_space = kwargs.pop("real_space", None)
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
        GSObject.__init__(self, galsim.SBConvolve(SBList, real_space=real_space))


class Deconvolve(GSObject):
    """@brief Base class for defining the python interface to the SBDeconvolve C++ class.
    """
    # --- Public Class methods ---
    def __init__(self, farg):
        if isinstance(farg, GSObject):
            self.farg = farg
            GSObject.__init__(self, galsim.SBDeconvolve(self.farg.SBProfile))
        else:
            raise TypeError("Argument farg must be a GSObject.")


