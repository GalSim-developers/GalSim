import galsim

class GSBase:
    """Base class for defining the interface with which all GalSim Objects access their shared 
    methods and attributes, particularly those from the C++ SBProfile classes.
    """
    def __init__(self, SBProfile):
        self.SBProfile = SBProfile

    #Define direct access to all SBProfile methods via calls to self.SBProfile.method_name()

    # ...Do we want to do this?  Barney is not sure... Surely most of these are pretty stable,
    # but this scheme would demand that changes to SBProfile are kept updated here.
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

#    This method does not seem to be wrapped
#    def isAnalyticK(self):
#        return self.SBProfile.isAnalyticK()

    def centroid(self):
        return self.SBProfile.centroid()

    def setFlux(self, flux=1.):
        self.SBProfile.setFlux(flux)
        return

    def getFlux(self):
        return self.SBProfile.getFlux()

    def distort(self, ellipse):
        self.SBProfile.distort(ellipse)
        return

    def shear(self, e1, e2):
        self.SBProfile.distort(galsim.Ellipse(e1, e2))
        return

    def rotate(self, theta):
        self.SBProfile.rotate(theta)
        return

    def shift(self, dx, dy):
        self.SBProfile.shift(dx, dy)
        return    

    def draw(self, dx=0., wmult=1):
        if type(wmult) != int:
            raise TypeError("Input wmult should be an int")
        if type(dx) != float:
            raise Warning("Input dx not a float, converting...")
            dx = float(dx)
        return self.SBProfile.draw(dx=0., wmult=1)

    def plainDraw(self, image, dx=0., wmult=1.):
        image, flux = self.SBPRofile.plainDraw(image, dx=dx, wmult=wmult)
        return image, flux

    def fourierDraw(self, image, dx=0., wmult=1.):
        image, flux = self.SBPRofile.fourierDraw(image, dx=dx, wmult=wmult)
        return flux

    def drawK(self, real, imag, dk=0., wmult=1.):
        real, imag = self.SBProfile.drawK(real, imag, dk=0., wmult=1.)
        return real, imag

    def plainDrawK(self, real, imag, dk=0., wmult=1.):
        real, imag = self.SBProfile.plainDrawK(real, imag, dk=0., wmult=1.)
        return real, imag

    def fourierDrawK(self, real, imag, dk=0., wmult=1.):
        real, image = self.SBProfile.plainDrawK(real, imag, dk=0., wmult=1.)
        return real, imag

    def shoot(self):
        raise NotImplementedError("Sorry, photon shooting coming soon!")
 
