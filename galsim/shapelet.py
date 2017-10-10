# Copyright (c) 2012-2017 by the GalSim developers team on GitHub
# https://github.com/GalSim-developers
#
# This file is part of GalSim: The modular galaxy image simulation toolkit.
# https://github.com/GalSim-developers/GalSim
#
# GalSim is free software: redistribution and use in source and binary forms,
# with or without modification, are permitted provided that the following
# conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions, and the disclaimer given in the accompanying LICENSE
#    file.
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions, and the disclaimer given in the documentation
#    and/or other materials provided with the distribution.
#
"""@file shapelet.py

Shapelet is a GSObject that implements a shapelet decomposition of a profile.
"""

import numpy as np

import galsim
from galsim import GSObject
from . import _galsim
from ._galsim import LVector, ShapeletSize

class Shapelet(GSObject):
    """A class describing polar shapelet surface brightness profiles.

    This class describes an arbitrary profile in terms of a shapelet decomposition.  A shapelet
    decomposition is an eigenfunction decomposition of a 2-d function using the eigenfunctions
    of the 2-d quantum harmonic oscillator.  The functions are Laguerre polynomials multiplied
    by a Gaussian.  See Bernstein & Jarvis, 2002 or Massey & Refregier, 2005 for more detailed
    information about this kind of decomposition.  For this class, we follow the notation of
    Bernstein & Jarvis.

    The decomposition is described by an overall scale length, `sigma`, and a vector of
    coefficients, `b`.  The `b` vector is indexed by two values, which can be either (p,q) or (N,m).
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

        >>> shapelet = galsim.Shapelet(sigma, order)

    2. Make a Shapelet instance using a given vector for the b_pq values.

        >>> order = 2
        >>> bvec = [ 1, 0, 0, 0.2, 0.3, -0.1 ]
        >>> shapelet = galsim.Shapelet(sigma, order, bvec)

    We use the following order for the coefficients, where the subscripts are in terms of p,q.

    [ b00  Re(b10)  Im(b10)  Re(b20)  Im(b20)  b11  Re(b30)  Im(b30)  Re(b21)  Im(b21) ... ]

    i.e. we progressively increase N, and for each value of N, we start with m=N and go down to
    m=0 or 1 as appropriate.  And since m=0 is intrinsically real, it only requires one spot
    in the list.

    @param sigma        The scale size in the standard units (usually arcsec).
    @param order        The order of the shapelet decomposition.  This is the maximum
                        N=p+q included in the decomposition.
    @param bvec         The initial vector of coefficients.  [default: None, which means to use
                        all zeros]
    @param gsparams     An optional GSParams argument.  See the docstring for GSParams for
                        details. [default: None]

    Fitting an Image
    ----------------

    There is also a factory function that measures the shapelet decomposition of a given
    image

        >>> shapelet = galsim.Shapelet.fit(sigma, order, image)

    Attributes
    ----------

    After construction, the `sigma`, `order`, and `bvec` are available as attributes.

    Methods and Properties
    ----------------------

    In addition to the usual GSObject methods, Shapelet has the following access methods and
    properties:

        >>> sigma = shapelet.sigma
        >>> order = shapelet.order
        >>> bvec = shapelet.bvec
        >>> b_pq = shapelet.getPQ(p,q)      # Get b_pq.  Returned as tuple (re, im) (even if p==q).
        >>> b_Nm = shapelet.getNM(N,m)      # Get b_Nm.  Returned as tuple (re, im) (even if m=0).

    Furthermore, there are specializations of the rotate() and expand() methods that let
    them be performed more efficiently than the usual GSObject implementation.
    """
    _req_params = { "sigma" : float, "order" : int }
    _opt_params = {}
    _single_params = []
    _takes_rng = False

    def __init__(self, sigma, order, bvec=None, gsparams=None):
        # Make sure order and sigma are the right type:
        order = int(order)
        sigma = float(sigma)

        # Make bvec if necessary
        if bvec is None:
            bvec = LVector(order, _depr_warn=False)
        else:
            bvec_size = Shapelet.size(order)
            if len(bvec) != bvec_size:
                raise ValueError("bvec is the wrong size for the provided order")
            bvec = LVector(order,np.array(bvec), _depr_warn=False)

        self._sbp = _galsim.SBShapelet(sigma, bvec, gsparams)
        self._gsparams = gsparams

    def getSigma(self):
        from .deprecated import depr
        depr("shapelet.getSigma()", 1.5, "shapelet.sigma")
        return self.sigma

    def getOrder(self):
        from .deprecated import depr
        depr("shapelet.getOrder()", 1.5, "shapelet.order")
        return self.order

    def getBVec(self):
        from .deprecated import depr
        depr("shapelet.getBVec()", 1.5, "shapelet.bvec")
        return self.bvec

    @property
    def sigma(self): return self._sbp.getSigma()
    @property
    def order(self): return self._sbp.getBVec().order
    @property
    def bvec(self): return self._sbp.getBVec().array

    def getPQ(self,p,q):
        return self._sbp.getBVec().getPQ(p,q)
    def getNM(self,N,m):
        return self._sbp.getBVec().getPQ((N+m)//2,(N-m)//2)

    # These act directly on the bvector, so they may be a bit more efficient than the
    # regular methods in GSObject
    def rotate(self, theta):
        if not isinstance(theta, galsim.Angle):
            raise TypeError("Input theta should be an Angle")
        bvec = self._sbp.getBVec().copy()
        bvec.rotate(theta)
        return Shapelet(self.sigma, self.order, bvec.array)

    def expand(self, scale):
        sigma = self.sigma * scale
        return Shapelet(sigma, self.order, self.bvec * scale**2)

    def dilate(self, scale):
        sigma = self.sigma * scale
        return Shapelet(sigma, self.order, self.bvec)

    def __eq__(self, other):
        return (isinstance(other, galsim.Shapelet) and
                self.sigma == other.sigma and
                self.order == other.order and
                np.array_equal(self.bvec, other.bvec) and
                self._gsparams == other._gsparams)

    def __hash__(self):
        return hash(("galsim.Shapelet", self.sigma, self.order, tuple(self.bvec),
                     self._gsparams))

    def __repr__(self):
        return 'galsim.Shapelet(sigma=%r, order=%r, bvec=%r, gsparams=%r)'%(
                self.sigma, self.order, self.bvec, self._gsparams)

    def __str__(self):
        return 'galsim.Shapelet(sigma=%s, order=%s, bvec=%s)'%(self.sigma, self.order, self.bvec)

    @classmethod
    def size(cls, order):
        """Calculate the size of a shapelet vector for a given order.

        Equal to (order+1) * (order+2) // 2
        """
        return _galsim.ShapeletSize(order)

    @classmethod
    def fit(cls, sigma, order, image, center=None, normalization='flux', gsparams=None):
        """Fit for a shapelet decomposition of a given image.

        The optional `normalization` parameter mirrors the parameter of the InterpolatedImage
        class.  The following sequence should produce drawn images that are approximate matches to
        the original image:

            >>> image = [...]
            >>> shapelet = galsim.FitShapelet(sigma, order, image, normalization='sb')
            >>> im2 = shapelet.drawImage(image=im2, scale=image.scale, method='sb')
            >>> shapelet = galsim.FitShapelet(sigma, order, image, normalization='flux')
            >>> im3 = shapelet.drawImage(image=im3, scale=image.scale, method='no_pixel')

        Then `im2` and `im3` should be as close as possible to `image` for the given `sigma` and
        `order`.  Increasing the order can improve the fit, as can having `sigma` match the natural
        scale size of the image.  However, it should be noted that some images are not well fit by
        a shapelet for any (reasonable) order.

        @param sigma        The scale size in the standard units (usually arcsec).
        @param order        The order of the shapelet decomposition.  This is the maximum
                            N=p+q included in the decomposition.
        @param image        The Image for which to fit the shapelet decomposition
        @param center       The position in pixels to use for the center of the decomposition.
                            [default: image.true_center]
        @param normalization  The normalization to assume for the image.
                            [default: "flux"]
        @param gsparams     An optional GSParams argument.  See the docstring for GSParams for
                            details. [default: None]

        @returns the fitted Shapelet profile
        """
        if not center:
            center = image.true_center
        # convert from PositionI if necessary
        center = galsim.PositionD(center.x,center.y)

        if not normalization.lower() in ("flux", "f", "surface brightness", "sb"):
            raise ValueError(("Invalid normalization requested: '%s'. Expecting one of 'flux', "+
                                "'f', 'surface brightness' or 'sb'.") % normalization)

        bvec = LVector(order, _depr_warn=False)

        if image.wcs is not None and not image.wcs.isPixelScale():
            # TODO: Add ability for ShapeletFitImage to take jacobian matrix.
            raise NotImplementedError("Sorry, cannot (yet) fit a shapelet model to an image "+
                                        "with a non-trivial WCS.")

        _galsim.ShapeletFitImage(sigma, bvec, image._image, image.scale, center)

        if normalization.lower() == "flux" or normalization.lower() == "f":
            bvec /= image.scale**2

        return Shapelet(sigma, order, bvec.array, gsparams)


_galsim.SBShapelet.__getinitargs__ = lambda self: (
        self.getSigma(), self.getBVec(), self.getGSParams())
_galsim.SBShapelet.__getstate__ = lambda self: None
_galsim.SBShapelet.__repr__ = lambda self: 'galsim._galsim.SBShapelet(%r, %r, %r)'%(
        self.getSigma(), self.getBVec(), self.getGSParams())

_galsim.LVector.__getinitargs__ = lambda self: (self.order, self.array, False)
_galsim.LVector.__repr__ = lambda self: 'galsim._galsim.LVector(%r, %r, False)'%(
        self.order, self.array)
_galsim.LVector.__eq__ = lambda self, other: repr(self) == repr(other)
_galsim.LVector.__ne__ = lambda self, other: not self.__eq__(other)
_galsim.LVector.__hash__ = lambda self: hash(repr(self))

# Give deprecation warnings if user uses LVector.
orig_LVector_init = _galsim.LVector.__init__
def new_LVector_init(self, order=0, array=None, _depr_warn=True):
    if _depr_warn:
        from .deprecated import depr
        depr("LVector", 1.5, "",
            "LVector is an implementation detail of Shapelet that users should not need to use "+
            "directly.  It will be removed in version 2.0.")
    if array is None:
        orig_LVector_init(self, order)
    else:
        orig_LVector_init(self, order, array)
_galsim.LVector.__init__ = new_LVector_init



