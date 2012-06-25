import os
import collections
import numpy as np
import galsim
import utilities

ALIAS_THRESHOLD = 0.005 # Matches hard coded value in src/SBProfile.cpp. TODO: bring these together

class GSObject:
    """@brief Base class for defining the interface with which all GalSim Objects access their
    shared methods and attributes, particularly those from the C++ SBProfile classes.
    """
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
        """@brief Apply a galsim.Ellipse distortion to this object.
           
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
        
    def applyShear(self, *args, **kwargs):
        """@brief Apply a shear to this object, where arguments are either a galsim.Shear, or
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
                raise TypeError("Error, unnamed argument to applyShear is not a galsim.Shear!")
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
        """@brief Returns a new GSObject by applying a galsim.Ellipse transformation (shear, dilate,
        and/or shift).

        Note that Ellipse objects can be initialized in a variety of ways (see documentation of this
        class for details).
        """
        if not isinstance(ellipse, galsim.Ellipse):
            raise TypeError("Argument to createTransformed must be a galsim.Ellipse!")
        ret = self.copy()
        ret.applyTransformation(ellipse)
        return ret

    def createSheared(self, *args, **kwargs):
        """@brief Returns A new GSObject by applying a shear, where arguments are either a
        galsim.Shear or keyword arguments that can be used to create one.
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

    def draw(self, image=None, dx=0., wmult=1, normalization="flux"):
        """@brief Draws an Image of the object, with bounds optionally set by an input Image.

        @param image  If provided, this will be the image on which to draw the profile.
                      If image=None, then an automatically-sized image will be created.
                      (Default = None)
        @param dx     If provided, use this as the pixel scale for the image.
                      If dx <= 0. and image != None, then take the provided image's pixel scale.
                      If dx <= 0. and image == None, then use pi/maxK()
                      (Default = 0.)
        @param normalization  Two options for the normalization:
                              "flux" or "f" means that the sum of the output pixels is normalized
                                     to be equal to the total flux.  (Modulo any flux that
                                     falls off the edge of the image of course.)
                              "surface brightness" or "sb" means that the output pixels sample
                                     the surface brightness distribution at each location.
                              (Default = "flux")
        @param wmult  A factor by which to make the intermediate images larger than 
                      they are normally made.  The size is normally automatically chosen 
                      to reach some preset accuracy targets (see include/galsim/SBProfile.h); 
                      however, if you see strange artifacts in the image, you might try using 
                      wmult > 1.  This will take longer of course, but it will produce more 
                      accurate images, since they will have less "folding" in Fourier space.
                      (Default = 1.)
        @returns      The drawn image.
        """
        # Raise an exception immediately if the normalization type is not recognized
        if not normalization.lower() in ("flux", "f", "surface brightness", "sb"):
            raise ValueError(("Invalid normalization requested: '%s'. Expecting one of 'flux', "+
                              "'f', 'surface brightness' or 'sb'.") % normalization)
        # Raise an exception here since C++ is picky about the input types
        if type(wmult) != int:
            raise TypeError("Input wmult should be an int")
        if type(dx) != float:
            raise Warning("Input dx not a float, converting...")
            dx = float(dx)
        if image == None:
            image = self.SBProfile.draw(dx=dx, wmult=wmult)
        else :
            if dx <= 0.:
                dx = image.getScale()
            self.SBProfile.draw(image, dx=dx, wmult=wmult)

        if normalization.lower() == "flux" or normalization.lower() == "f":
            dx = image.getScale()
            image *= dx*dx
        return image

    def drawShoot(self, image, N=0., ud=None, normalization="flux", noise=0., poisson_flux=True):
        """@brief Draw an image of the object by shooting individual photons drawn from the 
        surface brightness profile of the object.

        @param image  The image on which to draw the profile.
                      Note: Unlike for the regular draw command, image is a required
                      parameter.  drawShoot will not make the image for you.
        @param N      If provided, the number of photons to use.
                      If not provided, use as many photons as necessary to end up with
                      an image with the correct poisson shot noise for the object's flux.
                      For positive definite profiles, this is equivalent to N = flux.
                      However, some profiles need more than this because some of the shot
                      photons are negative (usually due to interpolants).
                      (Default = 0.)
        @param ud     If provided, a UniformDeviate to use for the random numbers
                      If ud=None, one will be automatically created, using the time as a seed.
                      (Default = None)
        @param normalization  Two options for the normalization:
                              "flux" or "f" means that the sum of the output pixels is normalized
                                     to be equal to the total flux.  (Modulo any flux that
                                     falls off the edge of the image of course.)
                              "surface brightness" or "sb" means that the output pixels sample
                                     the surface brightness distribution at each location.
                              (Default = "flux")
        @param noise  If provided, the allowed extra noise in each pixel.
                      This is only relevant if N=0, so the number of photons is being 
                      automatically calculated.  In that case, if the image noise is 
                      dominated by the sky background, you can get away with using fewer
                      shot photons than the full N = flux.  Essentially each shot photon
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
        @param  poisson_flux  Whether to allow total object flux scaling to vary according to 
                              Poisson statistics for N samples.
                              (Default = True)
                              
        @returns  The tuple (image, added_flux), where image is the input with drawn photons 
                  added and added_flux is the total flux of photons that landed inside the image 
                  bounds.

        The input image must have defined boundaries and pixel scale.  The photons generated by
        the drawShoot() method will be binned into the target image.  The input image will be 
        cleared before drawing in the photons.  Scale and location of the image pixels will not 
        be altered. 

        It is important to remember that the image produced by drawShoot() represents the object
        as convolved with the square image pixel.  So when using drawShoot() instead of draw(),
        you should not convolve with a Pixel.  This will produce the equivalent image (for very 
        large N) as draw() produces when the same object is convolved with Pixel(xw=dx) when 
        drawing onto an image with pixel scale dx.
        """
        # Raise an exception immediately if the normalization type is not recognized
        if not normalization.lower() in ("flux", "f", "surface brightness", "sb"):
            raise ValueError(("Invalid normalization requested: '%s'. Expecting one of 'flux', "+
                              "'f', 'surface brightness' or 'sb'.") % normalization)
        # Raise an exception here since C++ is picky about the input types
        if image is None:
            raise TypeError("drawShoot requires the image to be provided.")

        if type(N) != float:
            # if given an int, just convert it to a float
            N = float(N)
        if type(noise) != float:
            noise = float(noise)
        if ud == None:
            ud = galsim.UniformDeviate()

        added_flux = self.SBProfile.drawShoot(image, N, ud, noise, poisson_flux)

        if normalization.lower() == "flux" or normalization.lower() == "f":
            dx = image.getScale()
            image *= dx*dx

        return image, added_flux
         

# Now define some of the simplest derived classes, those which are otherwise empty containers for
# SBPs...
#
# Gaussian class inherits the GSObject method interface, but therefore has a "has a" relationship 
# with the C++ SBProfile class rather than an "is a"... The __init__ method is very simple and all
# the GSObject methods & attributes are inherited.
# 
# In particular, the SBGaussian is now an attribute of the Gaussian, an attribute named 
# "SBProfile", which can be queried for type as desired.
#
class Gaussian(GSObject):
    """@brief GalSim Gaussian, which has an SBGaussian in the SBProfile attribute.
    """
    def __init__(self, half_light_radius=None, sigma=None, fwhm=None, flux=1.):
        GSObject.__init__(self, galsim.SBGaussian(half_light_radius=half_light_radius, 
                                                  fwhm=fwhm, sigma=sigma, flux=flux))
        
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
    """
    def __init__(self, beta, fwhm=None, scale_radius=None, half_light_radius=None,
                 trunc=0., flux=1.):
        GSObject.__init__(self, galsim.SBMoffat(beta, fwhm=fwhm, scale_radius=scale_radius,
                                                half_light_radius=half_light_radius, trunc=trunc,
                                                flux=flux))
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
    

class Sersic(GSObject):
    """@brief GalSim Sersic, which has an SBSersic in the SBProfile attribute.
    """
    def __init__(self, n, half_light_radius, flux=1.):
        GSObject.__init__(self, galsim.SBSersic(n, half_light_radius=half_light_radius, flux=flux))

    def getN(self):
        """@brief Return the Sersic index for this profile.
        """
        return self.SBProfile.getN()

    def getHalfLightRadius(self):
        """@brief Return the half light radius for this Sersic profile.
        """
        return self.SBProfile.getHalfLightRadius()


class Exponential(GSObject):
    """@brief GalSim Exponential, which has an SBExponential in the SBProfile attribute.
    """
    def __init__(self, half_light_radius=None, scale_radius=None, flux=1.):
        GSObject.__init__(self, galsim.SBExponential(half_light_radius=half_light_radius,
                                                     scale_radius=scale_radius, flux=flux))

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
    """@brief GalSim De-Vaucouleurs, which has an SBDeVaucouleurs in the SBProfile attribute.
    """
    def __init__(self, half_light_radius=None, flux=1.):
        GSObject.__init__(self, galsim.SBDeVaucouleurs(half_light_radius=half_light_radius,
                                                       flux=flux))

    def getHalfLightRadius(self):
        """@brief Return the half light radius for this DeVaucouleurs profile.
        """
        return self.SBProfile.getHalfLightRadius()


class Airy(GSObject):
    """@brief GalSim Airy, which has an SBAiry in the SBProfile attribute.
    """
    def __init__(self, lam_over_D, obscuration=0., flux=1.):
        GSObject.__init__(self, galsim.SBAiry(lam_over_D=lam_over_D, obscuration=obscuration,
                                              flux=flux))

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


class Pixel(GSObject):
    """@brief GalSim Pixel, which has an SBBox in the SBProfile attribute.
    """
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


class OpticalPSF(GSObject):
    """@brief Class describing aberrated PSFs due to telescope optics.

    Input aberration coefficients are assumed to be supplied in units of incident light wavelength,
    and correspond to the conventions adopted here:
    http://en.wikipedia.org/wiki/Optical_aberration#Zernike_model_of_aberrations

    Initialization
    --------------
    @code
    optical_psf = galsim.OpticalPSF(lam_over_D, defocus=0., astig1=0., astig2=0., coma1=0.,
                                        coma2=0., spher=0., circular_pupil=True, obscuration=0.,
                                        interpolantxy=None, oversampling=1.5, pad_factor=1.5)
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
    @param circular_pupil  adopt a circular pupil?
    @param obscuration     linear dimension of central obscuration as fraction of pupil linear 
                           dimension, [0., 1.) [default = 0.]
    @param interpolantxy   optional keyword for specifying the interpolation scheme [default =
                           galsim.InterpolantXY(galsim.Lanczos(5, conserve_flux=True, tol=1.e-4))].
    @param oversampling    optional oversampling factor for the SBInterpolatedImage table 
                           [default = 1.5], setting oversampling < 1 will produce aliasing in the 
                           PSF (not good).
    @param pad_factor      additional multiple by which to zero-pad the PSF image to avoid folding
                           compared to what would be required for a simple Airy [default = 1.5].
                           Note that padFactor may need to be increased for stronger aberrations,
                           i.e. those larger than order unity. 
    """
    def __init__(self, lam_over_D, defocus=0., astig1=0., astig2=0., coma1=0., coma2=0., spher=0.,
                 circular_pupil=True, obscuration=0., interpolantxy=None, oversampling=1.5,
                 pad_factor=1.5):
        # Currently we load optics, noise etc in galsim/__init__.py, but this might change (???)
        import galsim.optics
        # Choose dx for lookup table using Nyquist for optical aperture and the specified
        # oversampling factor
        dx_lookup = .5 * lam_over_D / oversampling
        # Use a similar prescription as SBAiry to set Airy stepK and thus reference unpadded image
        # size in physical units
        stepk_airy = min(ALIAS_THRESHOLD * .5 * np.pi**3 * (1. - obscuration) / lam_over_D,
                         np.pi / 5. / lam_over_D)
        # Boost Airy image size by a user-specifed pad_factor to allow for larger, aberrated PSFs,
        # also make npix always *odd* so that opticalPSF lookup table array is correctly centred:
        npix = 1 + 2 * (np.ceil(pad_factor * (np.pi / stepk_airy) / dx_lookup)).astype(int)
        # Make the psf image using this dx and array shape
        optimage = galsim.optics.psf_image(lam_over_D=lam_over_D, dx=dx_lookup,
                                           array_shape=(npix, npix), defocus=defocus, astig1=astig1,
                                           astig2=astig2, coma1=coma1, coma2=coma2, spher=spher,
                                           circular_pupil=circular_pupil, obscuration=obscuration)
        # If interpolant not specified on input, use a high-ish n lanczos
        if interpolantxy == None:
            lan5 = galsim.Lanczos(5, conserve_flux=True, tol=1.e-4)
            self.Interpolant2D = galsim.InterpolantXY(lan5)
        else:
            self.Interpolant2D = interpolantxy
        GSObject.__init__(self, galsim.SBInterpolatedImage(optimage, self.Interpolant2D,
                                                           dx=dx_lookup))
    def getHalfLightRadius(self):
        # The half light radius is a complex function for aberrated optical PSFs, so just give
        # up gracelessly...
        raise NotImplementedError("Half light radius calculation not implemented for OpticalPSF "
                                   +"objects.")

class AtmosphericPSF(GSObject):
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
                           usually quoted at lambda = 500 nm and r0 depends weakly on wavelength
                           [r0 ~ lambda^(-6/5)].
    @param oversampling    optional oversampling factor for the SBInterpolatedImage table 
                           [default = 1.5], setting oversampling < 1 will produce aliasing in the 
                           PSF (not good).
    """
    def __init__(self, lam_over_r0, interpolantxy=None, oversampling=1.5):
        # The FWHM of the Kolmogorov PSF is ~0.976 lambda/r0 (e.g., Racine 1996, PASP 699, 108).
        fwhm = 0.976 * lam_over_r0
        dx_lookup = .5 * fwhm / oversampling
        # Fold at 10 times the FWHM
        stepk_kolmogorov = np.pi / (10. * fwhm)
        # Odd array to center the interpolant on the centroid. Might want to pad this later to
        # make a nice size array for FFT, but for typical seeing, arrays will be very small.
        npix = 1 + 2 * (np.ceil(np.pi / stepk_kolmogorov)).astype(int)
        atmoimage = galsim.atmosphere.kolmogorov_psf_image(array_shape=(npix, npix), dx=dx_lookup, 
                                                           lam_over_r0=lam_over_r0)
        if interpolantxy == None:
            lan5 = galsim.Lanczos(5, conserve_flux=True, tol=1e-4)
            self.Interpolant2D = galsim.InterpolantXY(lan5)
        GSObject.__init__(self, galsim.SBInterpolatedImage(atmoimage, self.Interpolant2D, 
                                                           dx=dx_lookup))
    def getHalfLightRadius(self):
        # TODO: This seems like it would not be impossible to calculate
        raise NotImplementedError("Half light radius calculation not yet implemented for "+
                                   "Atmospheric PSF objects (could be though).")
        
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
    def __init__(self, real_galaxy_catalog, index = None, ID = None, random = False,
                 uniform_deviate = None, interpolant = None):

        import pyfits

        # Code block below will be for galaxy selection; not all are currently implemented.  Each
        # option must return an index within the real_galaxy_catalog.
        use_index = -1
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
        if interpolant != None and isinstance(interpolant, galsim.InterpolantXY) == False:
            raise RuntimeError('Specified interpolant is not an InterpolantXY!')
        elif interpolant == None:
            lan5 = galsim.Lanczos(5, conserve_flux=True, tol=1.e-4) # copied from Shera.py!
            self.Interpolant2D = galsim.InterpolantXY(lan5)
        else:
            self.Interpolant2D = interpolant

        # read in data about galaxy from FITS binary table; store as members of RealGalaxy

        # save any other relevant information
        self.catalog_file = real_galaxy_catalog.filename
        self.index = use_index
        self.pixel_scale = float(real_galaxy_catalog.pixel_scale[use_index])
        # note: will be adding more parameters here about noise properties etc., but let's be basic
        # for now

        self.original_image = galsim.SBInterpolatedImage(gal_image, self.Interpolant2D, dx =
                                                         self.pixel_scale)
        self.original_PSF = galsim.SBInterpolatedImage(PSF_image, self.Interpolant2D,
                                                         dx=self.pixel_scale)
        self.original_PSF.setFlux(1.0)
        psf_inv = galsim.SBDeconvolve(self.original_PSF)

        GSObject.__init__(self, galsim.SBConvolve([self.original_image, psf_inv]))

    def getHalfLightRadius(self):
        raise NotImplementedError("Half light radius calculation not implemented for RealGalaxy "
                                   +"objects.")

class Add(GSObject):
    """@brief Base class for defining the python interface to the SBAdd C++ class.
    """
    def __init__(self, *args):
        # This is a workaround for the fact that Python doesn't allow multiple constructors.
        # So check the number and type of the arguments here in the single __init__ method.
        if len(args) == 0:
            # No arguments.  Start with none and add objects later with add(obj)
            GSObject.__init__(self, galsim.SBAdd())
        elif len(args) == 1:
            # 1 argment.  Should be either a GSObject or a list of GSObjects
            if isinstance(args[0], GSObject):
                # If single argument is a GSObject, then use the SBAdd for a single SBProfile.
                GSObject.__init__(self, galsim.SBAdd(args[0].SBProfile))
            else:
                # Otherwise, should be a list of GSObjects
                SBList = [obj.SBProfile for obj in args[0]]
                GSObject.__init__(self, galsim.SBAdd(SBList))
        elif len(args) == 2:
            # 2 arguments.  Should both be GSObjects.
            GSObject.__init__(self, galsim.SBAdd(args[0].SBProfile,args[1].SBProfile))
        else:
            # > 2 arguments.  Convert to a list of SBProfiles
            SBList = [obj.SBProfile for obj in args]
            GSObject.__init__(self, galsim.SBAdd(SBList))

    def add(self, obj, scale=1.):
        self.SBProfile.add(obj.SBProfile, scale)


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
    where the sum is over all teh components being convolved.  Since the size of 
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
    def __init__(self, *args, **kwargs):
        # Check kwargs first
        # The only kwarg we're looking for is real_space, which can be True or False
        # (default if omitted is None), which specifies whether to do the convolution
        # as an integral in real space rather than as a product in fourier space.
        # If the parameter is omitted (or explicitly given as None I guess), then
        # we will usually do the fourier method.  However, if there are 2 components
        # _and_ both of them have hard edges, then we use real-space convolution.
        real_space = kwargs.pop("real_space",None)

        if kwargs:
            raise TypeError(
                "Convolve constructor got unexpected keyword argument(s): %s"%kwargs.keys())

        # If 1 argument, check if it is a list:
        if len(args) == 1 and isinstance(args[0],list):
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

        if len(args) == 0:
            GSObject.__init__(self, galsim.SBConvolve(real_space=real_space))
        elif len(args) == 1:
            GSObject.__init__(self, galsim.SBConvolve(args[0].SBProfile,real_space=real_space))
        elif len(args) == 2:
            GSObject.__init__(self, galsim.SBConvolve(
                    args[0].SBProfile,args[1].SBProfile,real_space=real_space))
        else:
            # > 2 arguments.  Convert to a list of SBProfiles
            SBList = [obj.SBProfile for obj in args]
            GSObject.__init__(self, galsim.SBConvolve(SBList,real_space=real_space))

    def add(self, obj):
        self.SBProfile.add(obj.SBProfile)


class Deconvolve(GSObject):
    """@brief Base class for defining the python interface to the SBDeconvolve C++ class.
    """
    def __init__(self, farg):
        # the single argument should be one of our base classes
        GSObject.__init__(self, galsim.SBDeconvolve(farg.SBProfile))


# Now we define a dictionary containing all the GSobject subclass names as keys, referencing a
# nested dictionary containing the names of their required parameters (not including size), size
# specification parameters (one of which only must be set), and optional parameters, stored as a
# tuple of string names in each case.
#
# This is useful for I/O, and as a reference.
#
# NOTE TO DEVELOPERS: This dict should be kept updated to reflect changes in parameter names or new
#                     objects.
#
object_param_dict = {"Gaussian":       { "required" : (),
                                         "size" :     ("half_light_radius", "sigma", "fwhm",),
                                         "optional" : ("flux",) },
                     "Moffat":         { "required" : ("beta",),
                                         "size"     : ("half_light_radius", "scale_radius", 
                                                       "fwhm",),
                                         "optional" : ("trunc", "flux",) },
                     "Sersic":         { "required" : ("n",) ,
                                         "size"     : ("half_light_radius",),
                                         "optional" : ("flux",) },
                     "Exponential":    { "required" : (),
                                         "size"     : ("half_light_radius", "scale_radius"),
                                         "optional" : ("flux",) },
                     "DeVaucouleurs":  { "required" : (),
                                         "size"     : ("half_light_radius",),
                                         "optional" : ("flux",) },
                     "Airy":           { "required" : () ,
                                         "size"     : ("D",) ,
                                         "optional" : ("obs", "flux",)},
                     "Pixel":          { "required" : ("xw", "yw",),
                                         "size"     : (),
                                         "optional" : ("flux",) },
                     "OpticalPSF":     { "required" : (),
                                         "size"     : ("lam_over_D",),
                                         "optional" : ("defocus", "astig1", "astig2", "coma1",
                                                       "coma2", "spher", "circular_pupil",
                                                       "interpolantxy", "dx", "oversampling",
                                                       "pad_factor") },
                     "DoubleGaussian": { "required" : (), 
                                         "size"     : ("sigma1, sigma2, fwhm1, fwhm2",), 
                                         "optional" : () },
                     "AtmosphericPSF": { "required" : (),
                                         "size"     : ("lam_over_r0",),
                                         "optional" : ("dx", "oversampling") } }


class AttributeDict(object):
    """@brief Dictionary class that allows for easy initialization and refs to key values via
    attributes.

    NOTE: Modified a little from Jim's bot.git AttributeDict class  (Jim, please review!) so that...

    ...Tab completion now works in ipython (it didn't with the bot.git version on my build) since
    attributes are actually added to __dict__.
    
    HOWEVER this means I have redefined the __dict__ attribute to be a collections.defaultdict()
    so that Jim's previous default attrbiute behaviour is also replicated.

    I prefer this, as a newbie who uses ipython and the tab completion function to aid development,
    but does it potentially break something down the line or add undesirable behaviour? (I guessed
    not, since collections.defaultdict objects have all the usual dict() methods, but I cannot be
    sure.)
    """
    def __init__(self):
        object.__setattr__(self, "__dict__", collections.defaultdict(AttributeDict))

    def __getattr__(self, name):
        return self.__dict__[name]

    def __setattr__(self, name, value):
        self.__dict__[name] = value

    def merge(self, other):
        self.__dict__.update(other.__dict__)

    def _write(self, output, prefix=""):
        for k, v in self.__dict__.iteritems():
            if isinstance(v, AttributeDict):
                v._write(output, prefix="{0}{1}.".format(prefix, k))
            else:
                output.append("{0}{1} = {2}".format(prefix, k, repr(v)))

    def __nonzero__(self):
        return not not self.__dict__

    def __repr__(self):
        output = []
        self._write(output, "")
        return "\n".join(output)

    __str__ = __repr__

    def __len__(self):
        return len(self.__dict__)


class Config(AttributeDict):
    """@brief Config class that is basically a renamed AttributeDict, and allows for easy
    initialization and refs to key values via attributes.
    """
    def __init__(self):
        AttributeDict.__init__(self)



