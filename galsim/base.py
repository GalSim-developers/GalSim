import os
import collections
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

        galsim.Ellipse instances can be generated via

        >>> ellipse = galsim.Ellipse(e1, e2)

        where the ellipticities follow the convention |e| = (a^2 - b^2)/(a^2 + b^2).
        """
        GSObject.__init__(self, self.SBProfile.distort(ellipse))
        
    def applyShear(self, g1, g2):
        """Apply a (g1, g2) shear to this object, where |g| = (a-b)/(a+b).
        """
        e1, e2 = g1g2_to_e1e2(g1, g2)
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

        galsim.Ellipse instances can be generated via

        >>> ellipse = galsim.Ellipse(e1, e2)

        where the ellipticities follow the convention |e| = (a^2 - b^2)/(a^2 + b^2).
        """
        return GSObject(self.SBProfile.distort(ellipse))

    def createSheared(self, g1, g2):
        """Create a new GSObject by applying a (g1, g2) shear, where |g| = (a-b)/(a+b).
        """
        e1, e2 = g1g2_to_e1e2(g1, g2)
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


# Define "convenience function for going from (g1, g2) -> (e1, e2), used by two methods
# in the GSObject class and by one function in real.py:
def g1g2_to_e1e2(g1, g2):
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


class DeVaucouleurs(GSObject):
    """GalSim De-Vaucouleurs, which has an SBDeVaucouleurs in the SBProfile attribute.
    """
    def __init__(self, flux=1., half_light_radius=None):
        GSObject.__init__(self, galsim.SBDeVaucouleurs(flux=flux, 
                                                       half_light_radius=half_light_radius))
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
        if yw is None:
            yw = xw
        GSObject.__init__(self, galsim.SBBox(xw=xw, yw=yw, flux=flux))
    # Ditto!

class OpticalPSF(GSObject):
    """@brief Class describing aberrated PSFs due to telescope optics.

    Input aberration coefficients are assumed to be supplied in units of incident light wavelength,
    and correspond to the conventions adopted here:
    http://en.wikipedia.org/wiki/Optical_aberration#Zernike_model_of_aberrations

    Initialization
    --------------
    >>> optical_psf = galsim.OpticalPSF(lam_over_D, defocus=0., astig1=0., astig2=0., coma1=0., 
                                        coma2=0., spher=0., circular_pupil=True, interpolantxy=None,
                                        dx=1., oversampling=2., pad_factor=2)

    Initializes optical_psf as a galsim.OpticalPSF() instance.

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
            lan5 = galsim.Lanczos(5, conserve_flux=True, tol=1.e-4) # copied from Shera.py!
            self.Interpolant2D = galsim.InterpolantXY(lan5)
        GSObject.__init__(self, galsim.SBInterpolatedImage(optimage, self.Interpolant2D, dx=dx))


class RealGalaxy(GSObject):
    """@brief Class describing real galaxies from some training dataset.

    This class uses a catalog describing galaxies in some training data to read in data about
    realistic galaxies that can be used for simulations based on those galaxies.  Also included in
    the class is additional information that might be needed to make or interpret the simulations,
    e.g., the noise properties of the training data.

    Initialization
    --------------
    real_galaxy = galsim.RealGalaxy(real_galaxy_catalog, index = None, ID = None, ID_string =
                                    None, random = False, uniform_deviate = None, interpolant = None)

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
                                [default = galsim.InterpolantXY(galsim.Lanczos(5, True, 1.e-4))].
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
            raise NotImplementedError('Selecting galaxy based on its ID not implemented')
        elif random == True:
            if uniform_deviate == None:
                uniform_deviate = galsim.UniformDeviate()
            use_index = int(real_galaxy_catalog.n * uniform_deviate()) # this will round down, to get index in
                                                                           # range [0, n-1]
        else:
            raise RuntimeError('No method specified for selecting a galaxy!')
        if random == False and uniform_deviate != None:
            import warnings
            message = "Warning: uniform_deviate supplied, but random selection method was not chosen!"
            warnings.warn(message)

        # read in the galaxy, PSF images; for now, rely on pyfits to make I/O errors. Should
        # consider exporting this code into fits.py in some function that takes a filename and HDU,
        # and returns an ImageView
        gal_image_numpy = pyfits.getdata(os.path.join(real_galaxy_catalog.imagedir,
                                                      real_galaxy_catalog.gal_filename[use_index]),
                                         real_galaxy_catalog.gal_hdu[use_index])
        gal_image = galsim.ImageViewD(np.ascontiguousarray(gal_image_numpy.astype(np.float64)))
        PSF_image_numpy = pyfits.getdata(os.path.join(real_galaxy_catalog.imagedir,
                                                      real_galaxy_catalog.PSF_filename[use_index]),
                                         real_galaxy_catalog.PSF_hdu[use_index])
        PSF_image = galsim.ImageViewD(np.ascontiguousarray(PSF_image_numpy.astype(np.float64)))

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
    """@brief Base class for defining the python interface to the SBConvolve C++ class.
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

class Deconvolve(GSObject):
    """Base class for defining the python interface to the SBDeconvolve C++ class.
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
                                         "optional" : ("truncationFWHM", "flux",) },
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
                                         "optional" : () } }


class AttributeDict(object):
    """Dictionary class that allows for easy initialization and refs to key values via attributes.

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
    """Config class that is basically a renamed AttributeDict, and allows for easy initialization
    and refs to key values via attributes.
    """
    def __init__(self):
        AttributeDict.__init__(self)



