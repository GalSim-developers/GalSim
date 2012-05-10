import os
import numpy as np
import galsim

ALIAS_THRESHOLD = 0.005 # Matches hard coded value in src/SBProfile.cpp. TODO: bring these together

class GSObject:
    """Base class for defining the interface with which all GalSim Objects access their shared 
    methods and attributes, particularly those from the C++ SBProfile classes.
    """
    def __init__(self, SBProfile):
        self.SBProfile = SBProfile  # This guarantees that all GSObjects have an SBProfile

    # Make op+ of two GSObjects work to return an Add object
    def __add__(self, other):
        return Add(self,other)

    # op+= converts this into the equivalent of an Add object
    def __iadd__(self, other):
        GSObject.__init__(self, galsim.SBAdd(self.SBProfile, other.SBProfile))
        return self

    # Make op* and op*= work to adjust the flux of an object
    def __imul__(self, other):
        self.setFlux(other * self.getFlux())
        return self

    def __mul__(self, other):
        ret = self.copy()
        ret *= other
        return ret

    def __rmul__(self, other):
        ret = self.copy()
        ret *= other
        return ret

    # Make a copy of an object
    def copy(self):
        return GSObject(self.SBProfile.duplicate())

    # Now define direct access to all SBProfile methods via calls to self.SBProfile.method_name()
    #
    # TODO: add method-specific docstrings later if we go for this overall layout
    def maxK(self):
        maxk = self.SBProfile.maxK()
        return maxk

    def nyquistDx(self):
        return self.SBProfile.nyquistDx()

    def stepK(self):
        return self.SBProfile.stepK()

    def isAxisymmetric(self):
        return self.SBProfile.isAxisymmetric()

    def isAnalyticX(self):
        return self.SBProfile.isAnalyticX()

    # This method does not seem to be wrapped from C++
    # def isAnalyticK(self):
    # return self.SBProfile.isAnalyticK()

    def centroid(self):
        return self.SBProfile.centroid()

    def setFlux(self, flux=1.):
        self.SBProfile.setFlux(flux)
        return

    def getFlux(self):
        return self.SBProfile.getFlux()

    def applyDistortion(self, ellipse):
        """Apply a galsim.Ellipse distortion to this object.
        """
        GSObject.__init__(self, self.SBProfile.distort(ellipse))
        
    def applyShear(self, g1, g2):
        """Apply a (g1,g2) shear to this object.
        """
        e1, e2 = _g1g2_to_e1e2(g1, g2)
        GSObject.__init__(self, self.SBProfile.distort(galsim.Ellipse(e1, e2)))

    def applyRotation(self, theta):
        """Apply a rotation theta (Angle object, +ve anticlockwise) to this object.
        """
        if not isinstance(theta, galsim.Angle):
            raise TypeError("Input theta should be an Angle")
        GSObject.__init__(self, self.SBProfile.rotate(theta))
        
    def applyShift(self, dx, dy):
        """Apply a (dx, dy) shift to this object.
        """
        GSObject.__init__(self, self.SBProfile.shift(dx, dy))

    # Also add methods which create a new GSObject with the transformations applied...
    #
    def createDistorted(self, ellipse):
        """Create a new GSObject by applying a galsim.Ellipse distortion.
        """
        return GSObject(self.SBProfile.distort(ellipse))

    def createSheared(self, g1, g2):
        """Create a new GSObject by applying a (g1, g2) shear.
        """
        e1, e2 = _g1g2_to_e1e2(g1, g2)
        return GSObject(self.SBProfile.distort(galsim.Ellipse(e1,e2)))

    def createRotated(self, theta):
        """Create a new GSObject by applying a rotation theta (Angle object, +ve anticlockwise).
        """
        if not isinstance(theta, galsim.Angle):
            raise TypeError("Input theta should be an Angle")
        return GSObject(self.SBProfile.rotate(theta))
        
    def createShifted(self, dx, dy):
        """Create a new GSObject by applying a (dx, dy) shift.
        """
        return GSObject(self.SBProfile.shift(dx, dy))

    def draw(self, image=None, dx=0., wmult=1):
    # Raise an exception here since C++ is picky about the input types
        if type(wmult) != int:
            raise TypeError("Input wmult should be an int")
        if type(dx) != float:
            raise Warning("Input dx not a float, converting...")
            dx = float(dx)
        if image is None:
            return self.SBProfile.draw(dx=dx, wmult=wmult)
        else :
            self.SBProfile.draw(image, dx=dx, wmult=wmult)
            return image

    # Did not define all the other draw operations that operate on images inplace, would need to
    # work out slightly different return syntax for that in Python

    def shoot(self):
        raise NotImplementedError("Sorry, photon shooting coming soon!")


# Define "hidden" convenience function for going from (g1, g2) -> (e1, e2), used by two methods
# in the GSObject class:
def _g1g2_to_e1e2(g1, g2):
    """Convenience function for going from (g1, g2) -> (e1, e2), used by two methods in the 
    GSObject class.
    """
    # SBProfile expects an e1,e2 distortion, rather than a shear,
    # so we need to convert:
    # e = (a^2-b^2) / (a^2+b^2)
    # g = (a-b) / (a+b)
    # b/a = (1-g)/(1+g)
    # e = (1-(b/a)^2) / (1+(b/a)^2)
    import math
    gsq = g1*g1 + g2*g2
    if gsq > 0.:
        g = math.sqrt(gsq)
        boa = (1-g) / (1+g)
        e = (1 - boa*boa) / (1 + boa*boa)
        e1 = g1 * (e/g)
        e2 = g2 * (e/g)
        return e1, e2
    elif gsq == 0.:
        return 0., 0.
    else:
        raise ValueError("Input |g|^2 < 0, cannot convert.")


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
    """GalSim Gaussian, which has an SBGaussian in the SBProfile attribute.
    """
    def __init__(self, flux=1., half_light_radius=None, sigma=None, fwhm=None):
        GSObject.__init__(self, galsim.SBGaussian(flux=flux, half_light_radius=half_light_radius, 
                                                  sigma=sigma, fwhm=fwhm))

    # Hmmm, these Gaussian-specific methods do not appear to be wrapped yet (will add issue to 
    # myself for this)... when they are, uncomment below:
    # def getSigma(self):
    #     return self.SBProfile.getSigma()
    #
    # def setSigma(self, sigma):
    #     return self.SBProfile.setSigma(sigma)


class Moffat(GSObject):
    """GalSim Moffat, which has an SBMoffat in the SBProfile attribute.
    """
    def __init__(self, beta, truncationFWHM=2., flux=1.,
                 half_light_radius=None, scale_radius=None, fwhm=None):
        GSObject.__init__(self, galsim.SBMoffat(beta, truncationFWHM=truncationFWHM, flux=flux,
                          half_light_radius=half_light_radius, scale_radius=scale_radius, fwhm=fwhm))
    # As for the Gaussian currently only the base layer SBProfile methods are wrapped
    # def getBeta(self):
    #     return self.SBProfile.getBeta()
    # ...etc.


class Sersic(GSObject):
    """GalSim Sersic, which has an SBSersic in the SBProfile attribute.
    """
    def __init__(self, n, flux=1., half_light_radius=None):
        GSObject.__init__(self, galsim.SBSersic(n, flux=flux, half_light_radius=half_light_radius))
    # Ditto!


class Exponential(GSObject):
    """GalSim Exponential, which has an SBExponential in the SBProfile attribute.
    """
    def __init__(self, flux=1., half_light_radius=None, scale_radius=None):
        GSObject.__init__(self, galsim.SBExponential(flux=flux, half_light_radius=half_light_radius,
                                                     scale_radius=scale_radius))
    # Ditto!


class Airy(GSObject):
    """GalSim Airy, which has an SBAiry in the SBProfile attribute.
    """
    def __init__(self, D=1., obs=0., flux=1.):
        GSObject.__init__(self, galsim.SBAiry(D=D, obs=obs, flux=flux))
    # Ditto!


class Pixel(GSObject):
    """GalSim Pixel, which has an SBBox in the SBProfile attribute.
    """
    def __init__(self, xw=None, yw=None, flux=1.):
        GSObject.__init__(self, galsim.SBBox(xw=xw, yw=yw, flux=flux))
    # Ditto!


class OpticalPSF(GSObject):
    """@brief Class describing aberrated PSFs due to telescope optics.

    Input aberration coefficients are assumed to be supplied in units of incident light wavelength,
    and correspond to the conventions adopted here:
    http://en.wikipedia.org/wiki/Optical_aberration#Zernike_model_of_aberrations

    Initialization
    --------------
    >>> optical_psf = galsim.OpticalPSF(lod=1., defocus=0., astig1=0., astig2=0., coma1=0., 
                                        coma2=0., spher=0., circular_pupil=True, interpolantxy=None,
                                        dx=1., oversampling=2., pad_factor=2)

    Initializes optical_psf as a galsim.optics() instance.

    @param lod             lambda / D in the physical units adopted (user responsible for 
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
    @param obs             add a central obstruction due to secondary mirror?
    @param interpolantxy   optional keyword for specifying the interpolation scheme [default =
                           galsim.InterpolantXY(galsim.Lanczos(5, True, 1.e-4))].
    @param oversampling    optional oversampling factor for the SBInterpolatedImage table 
                           [default = 2.], setting oversampling < 1 will produce aliasing in the 
                           PSF (not good).
    @param pad_factor      additional multiple by which to zero-pad the PSF image to avoid folding
                           compared to what would be required for a simple Airy [default = 2]. Note
                           that padFactor may need to be increased for stronger aberrations, i.e.
                           those larger than order unity. 
    """
    def __init__(self, lam_over_D, defocus=0., astig1=0., astig2=0., coma1=0., coma2=0., spher=0.,
                 circular_pupil=True, obs=None, interpolantxy=None, oversampling=2., pad_factor=2):
        # Currently we load optics, noise etc in galsim/__init__.py, but this might change (???)
        import galsim.optics
        # Use the same prescription as SBAiry to set dx, maxK, Airy stepK and thus image size
        self.maxk = 2. * np.pi / lam_over_D
        dx = .5 * lam_over_D / oversampling
        if obs == None:
            stepk_airy = min(ALIAS_THRESHOLD * .5 * np.pi**3 / lam_over_D,
                             np.pi / 5. / lam_over_D)
        else:
            raise NotImplementedError('Secondary mirror obstruction not yet implemented')
        # TODO: check that the above still makes sense even for large aberrations, probably not...
        npix = np.ceil(2. * pad_factor * self.maxk / stepk_airy).astype(int)
        optimage = galsim.optics.psf_image(array_shape=(npix, npix), defocus=defocus,
                                           astig1=astig1, astig2=astig2, coma1=coma1, coma2=coma2,
                                           spher=spher, circular_pupil=circular_pupil, obs=obs,
                                           kmax=self.maxk, dx=dx)
        # If interpolant not specified on input, use a high-ish lanczos
        if interpolantxy == None:
            l5 = galsim.Lanczos(5, True, 1.e-4) # Conserve flux=True and 1.e-4 copied from Shera.py!
            self.Interpolant2D = galsim.InterpolantXY(l5)
        GSObject.__init__(self, galsim.SBInterpolatedImage(optimage, self.Interpolant2D, dx=dx))


class RealGalaxy(GSObject):
    """@brief Class describing real galaxies from some training dataset.

    This class uses a catalog describing galaxies in some training data to read in data about
    realistic galaxies that can be used for simulations based on those galaxies.  Also included in
    the class is additional information that might be needed to make or interpret the simulations,
    e.g., the noise properties of the training data.

    Initialization
    --------------
    >>> real_galaxy = galsim.RealGalaxy(real_galaxy_catalog, index = None, ID = None, ID_string =
                                        None, random = False, interpolantxy = None)

    Initializes real_galaxy with two SBInterpolatedImage objects (one for the galaxy and one for the
    PSF). Note that there are multiple keywords for choosing a galaxy; exactly one must be set.  In
    future we may add more such options, e.g., to choose at random but accounting for the
    non-constant weight factors (probabilities for objects to make it into the training sample).

    @param real_galaxy_catalog  A RealGalaxyCatalog object with basic information about where to
                                find the data, etc.
    @param index                Index of the desired galaxy in the catalog.
    @param ID                   Object ID for the desired galaxy in the catalog.
    @param ID_string            ID string for the desired galaxy in the catalog.
    @param random               If true, then just select a completely random galaxy from the
                                catalog.
    @param interpolantxy        optional keyword for specifying the interpolation scheme [default = 
                                galsim.InterpolantXY(galsim.Lanczos(5, True, 1.e-4))].
    """
    def __init__(self, real_galaxy_catalog, index = None, ID = None, ID_string = None, random =
                                False, interpolantxy = None):

        # Code block below will be for galaxy selection;
        # implement exactly one for now, then gradually others.  All of them need to return an index
        # within the real_galaxy_catalog.
        use_index = -1
        if index != None:
            use_index = index
        elif ID != None:
            raise NotImplementedError('Selecting galaxy based on its ID not implemented')
        elif ID_string != None:
            raise NotImplementedError('Selecting galaxy based on its ID string not implemented')
        elif random == True:
            raise NotImplementedError('Selecting a random galaxy not implemented')
        else:
            raise RuntimeError('No method specified for selecting a galaxy!')

        # read in the galaxy, PSF images
        hdu_list = pyfits.open(os.path.join(real_galaxy_catalog.imagedir,
                                            real_galaxy_catalog.gal_filename[use_index]))
        n_hdu = len(hdu_list)
        ## note, we assume that the file containing the image has n_hdu HDUs containing
        ## images in it
        if real_galaxy_catalog.gal_hdu[use_index] > n_hdu:
            raise RuntimeError('No HDU corresponding to the one that should contain this galaxy!')
        use_hdu_list = pyfits.HDUList()
        use_hdu_list.append(hdu_list[real_galaxy_catalog.gal_hdu[use_index]])
        gal_image = galsim.fits.read(use_hdu_list)

        if (real_galaxy_catalog.gal_filename[use_index] !=real_galaxy_catalog.PSF_filename[use_index]):
            hdu_list = pyfits.open(os.path.join(real_galaxy_catalog.imagedir,
                                                real_galaxy_catalog.PSF_filename[use_index]))
            n_hdu = len(hdu_list)
        if real_galaxy_catalog.PSF_hdu[use_index] > n_hdu:
            raise RuntimeError('No HDU corresponding to the one that should contain this PSF!')
        use_hdu_list = pyfits.HDUList()
        use_hdu_list.append(hdu_list[real_galaxy_catalog.PSF_hdu[use_index]])
        PSF_image = galsim.fits.read(use_hdu_list)

        # choose proper interpolant
        if interpolantxy == None:
            l5 = galsim.Lanczos(5, True, 1.e-4) # Conserve flux=True and 1.e-4 copied from Shera.py!
            self.Interpolant2D = galsim.InterpolantXY(l5)

        # read in data about galaxy from FITS binary table; store as members of RealGalaxy

        # save any other relevant information
        self.catalog_file = real_galaxy_catalog.filename
        self.index = use_index
        self.pixel_scale = real_galaxy_catalog.pixel_scale[use_index]
        # note: will be adding more parameters here about noise properties etc., but let's be basic
        # for now

        self.PSF = galsim.SBInterpolatedImage(PSF_image, self.Interpolant2D, dx=self.pixel_scale)
        GSObject.__init__(self, galsim.SBInterpolatedImage(gal_image, self.Interpolant2D,
                                                           dx=self.pixel_scale))

class Add(GSObject):
    """Base class for defining the python interface to the SBAdd C++ class.
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
    """Base class for defining the python interface to the SBConvolve C++ class.
    """
    def __init__(self, *args):
        # This is a workaround for the fact that Python doesn't allow multiple constructors.
        # So check the number and type of the arguments here in the single __init__ method.
        if len(args) == 0:
            # No arguments.  Start with none and add objects later with add(obj)
            GSObject.__init__(self, galsim.SBConvolve())
        elif len(args) == 1:
            # 1 argment.  Should be either a GSObject or a list of GSObjects
            if isinstance(args[0], GSObject):
                # If single argument is a GSObject, then use the SBConvolve for a single SBProfile.
                GSObject.__init__(self, galsim.SBConvolve(args[0].SBProfile))
            else:
                # Otherwise, should be a list of GSObjects
                SBList = [obj.SBProfile for obj in args[0]]
                GSObject.__init__(self, galsim.SBConvolve(SBList))
        elif len(args) == 2:
            # 2 arguments.  Should both be GSObjects.
            GSObject.__init__(self, galsim.SBConvolve(args[0].SBProfile,args[1].SBProfile))
        else:
            # > 2 arguments.  Convert to a list of SBProfiles
            SBList = [obj.SBProfile for obj in args]
            GSObject.__init__(self, galsim.SBConvolve(SBList))

    def add(self, obj):
        self.SBProfile.add(obj.SBProfile)

