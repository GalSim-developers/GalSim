import galsim

class GSObject:
    """Base class for defining the interface with which all GalSim Objects access their shared 
    methods and attributes, particularly those from the C++ SBProfile classes.
    """
    def __init__(self, SBProfile):
        self.SBProfile = SBProfile  # This guarantees that all GSObjects have an SBProfile

    # Now define direct access to all SBProfile methods via calls to self.SBProfile.method_name()
    #
    # ...Do we want to do this?  Barney is not sure... Surely most of these are pretty stable at,
    # the SBP level but this scheme would demand that changes to SBProfile are kept updated here.
    #
    # The alternative is for these methods to always be accessed from the top level 
    # via Whatever.SBProfile.method(), which I guess makes it explicit what is going on, but
    # is starting to get clunky...
    #
    # Will add method specific docstrings later if we go for this overall layout
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
        
    def applyShear(self, e1, e2):
        """Apply an (e1, e2) shear to this object.
        """
        GSObject.__init__(self, self.SBProfile.distort(galsim.Ellipse(e1, e2)))

    def applyRotation(self, theta):
        """Apply an angular rotation theta [radians, +ve anticlockwise] to this object.
        """
        GSObject.__init__(self, self.SBProfile.rotate(theta))
        
    def applyShift(self, dx, dy):
        """Apply a (dx, dy) shift to this object.
        """
        GSObject.__init__(self, self.SBProfile.shift(dx, dy))

    # Barney: not sure about the below, kind of wanted not to have to let the user deal with
    # GSObject instances... Might need to reconsider this scheme.
    #
    # Keeping them here as commented placeholders.
    #
    #def createDistorted(self, ellipse):
    #    return GSObject(self.SBProfile.distort(ellipse))
        
    #def createSheared(self, e1, e2):
    #    return GSObject(self.SBProfile.distort(galsim.Ellipse(e1, e2)))

    #def createRotated(self, theta):
    #    return GSObject(self.SBProfile.rotate(theta))
        
    #def createShifted(self, dx, dy):
    #    return GSObject(self.SBProfile.shift(dx, dy))
            
    def draw(self, dx=0., wmult=1):
    # Raise an exception here since C++ is picky about the input types
        if type(wmult) != int:
            raise TypeError("Input wmult should be an int")
        if type(dx) != float:
            raise Warning("Input dx not a float, converting...")
            dx = float(dx)
        return self.SBProfile.draw(dx=dx, wmult=wmult)

    # Did not define all the other draw operations that operate on images inplace, would need to
    # work out slightly different return syntax for that in Python

    def shoot(self):
        raise NotImplementedError("Sorry, photon shooting coming soon!")


# Now define some of the simplest derived classes, those which are otherwise empty containers for
# SBPs...

# Gaussian class inherits the GSObject method interface, but therefore has a "has a" relationship 
# with the C++ SBProfile class rather than an "is a"... The __init__ method is very simple and all
# the GSObject methods & attributes are inherited.
# 
# In particular, the SBGaussian is now an attribute of the Gaussian, an attribute named 
# "SBProfile", which can be queried for type as desired.
class Gaussian(GSObject):
    """GalSim Gaussian, which has an SBGaussian in the SBProfile attribute.
    """
    def __init__(self, flux=1., sigma=1.):
        GSObject.__init__(self, galsim.SBGaussian(flux=flux, sigma=sigma))

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
    def __init__(self, beta, truncationFWHM=2., flux=1., re=1.):
        GSObject.__init__(self, galsim.SBMoffat(beta, truncationFWHM=truncationFWHM, flux=flux,
                          re=re))
    # As for the Gaussian currently only the base layer SBProfile methods are wrapped
    # def getBeta(self):
    #     return self.SBProfile.getBeta()
    # ...etc.

class Sersic(GSObject):
    """GalSim Sersic, which has an SBSersic in the SBProfile attribute.
    """
    def __init__(self, n, flux=1., re=1.):
        GSObject.__init__(self, galsim.SBSersic(n, flux=flux, re=re))
    # Ditto!


class Exponential(GSObject):
    """GalSim Exponential, which has an SBExponential in the SBProfile attribute.
    """
    def __init__(self, flux=1., r0=1.):
        GSObject.__init__(self, galsim.SBExponential(flux=flux, r0=r0))
    # Ditto!


class Airy(GSObject):
    """GalSim Airy, which has an SBAiry in the SBProfile attribute.
    """
    def __init__(self, D=1., obs=0., flux=1.):
        GSObject.__init__(self, galsim.SBAiry(D=D, obs=obs, flux=flux))
    # Ditto!

class GSAdd(GSObject):
    """Base class for defining the python interface to the SBAdd C++ class.
    """
    def __init__(self, SBAdd):
        GSObject.__init__(self, SBAdd)
        
    def add(self, profile, scale=1.):
        self.SBProfile.add(profile, scale)

class GSConvolve(GSObject):
    """Base class for defining the python interface to the SBConvolve C++ class.
    """
    def __init__(self, GSObjList):
        SBList = []
        for obj in GSObjList:
            SBList.append(obj.SBProfile)
        GSObject.__init__(self, galsim.SBConvolve(SBList))
