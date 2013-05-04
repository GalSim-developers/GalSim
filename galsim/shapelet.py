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
"""@file shapelet.py 

Shapelet is a GSObject that implements a shapelet decomposition of a profile.
"""

import galsim
from galsim import GSObject

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
        import numpy
        sigma = self.SBProfile.getSigma() * numpy.sqrt(mu)
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



