# Copyright (c) 2012-2021 by the GalSim developers team on GitHub
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
import numpy as np

from .gsobject import GSObject
from .gsparams import GSParams
from .position import PositionD
from .image import Image
from .utilities import lazy_property
from . import _galsim
from .errors import GalSimValueError, GalSimIncompatibleValuesError, GalSimNotImplementedError


class Shapelet(GSObject):
    r"""A class describing polar shapelet surface brightness profiles.

    This class describes an arbitrary profile in terms of a shapelet decomposition.  A shapelet
    decomposition is an eigenfunction decomposition of a 2-d function using the eigenfunctions
    of the 2-d quantum harmonic oscillator.  The functions are Laguerre polynomials multiplied
    by a Gaussian.  See Bernstein & Jarvis, 2002 or Massey & Refregier, 2005 for more detailed
    information about this kind of decomposition.  For this class, we follow the notation of
    Bernstein & Jarvis.

    The decomposition is described by an overall scale length, ``sigma``, and a vector of
    coefficients, ``b``.  The ``b`` vector is indexed by two values, which can be either (p,q) or
    (N,m).  In terms of the quantum solution of the 2-d harmonic oscillator, p and q are the number
    of quanta with positive and negative angular momentum (respectively).  Then, N=p+q, m=p-q.

    The 2D image is given by (in polar coordinates):

    .. math::
        I(r,\theta) = \frac{1}{\sigma^2} \sum_{pq} b_{pq} \psi_{pq}(r/\sigma, \theta)

    where :math:`\psi_{pq}` are the shapelet eigenfunctions, given by:

    .. math::
        \psi_pq(r,\theta) = \frac{(-)^q}{\sqrt{\pi}} \sqrt{\frac{q!}{p!}}
                            r^m \exp(i m \theta) \exp(-r^2/2) L_q^{(m)}(r^2)

    and :math:`L_q^{(m)}(x)` are generalized Laguerre polynomials.

    The coeffients :math:`b_{pq}` are in general complex.  However, we require that the resulting
    :math:`I(r,\theta)` be purely real, which implies that :math:`b_{pq} = b_{qp}^*`
    (where :math:`{}^*` means complex conjugate).
    This further implies that :math:`b_{pp}` (i.e. :math:`b_{pq}` with :math:`p==q`) is real.

    1. Make a blank Shapelet instance with all :math:`b_{pq} = 0.`::

        >>> shapelet = galsim.Shapelet(sigma, order)

    2. Make a Shapelet instance using a given vector for the :math:`b_{pq}` values.::

        >>> order = 2
        >>> bvec = [ 1, 0, 0, 0.2, 0.3, -0.1 ]
        >>> shapelet = galsim.Shapelet(sigma, order, bvec)

    We use the following order for the coefficients, where the subscripts are in terms of p,q.

    [ b00  Re(b10)  Im(b10)  Re(b20)  Im(b20)  b11  Re(b30)  Im(b30)  Re(b21)  Im(b21) ... ]

    i.e. we progressively increase N, and for each value of N, we start with m=N and go down to
    m=0 or 1 as appropriate.  And since m=0 is intrinsically real, it only requires one spot
    in the list.

    Parameters:
        sigma:      The scale size in the standard units (usually arcsec).
        order:      The order of the shapelet decomposition.  This is the maximum
                    N=p+q included in the decomposition.
        bvec:       The initial vector of coefficients.  [default: None, which means to use
                    all zeros]
        gsparams:   An optional `GSParams` argument. [default: None]
    """
    _req_params = { "sigma" : float, "order" : int }

    _has_hard_edges = False
    _is_axisymmetric = False
    _is_analytic_x = True
    _is_analytic_k = True

    def __init__(self, sigma, order, bvec=None, gsparams=None):
        # Make sure order and sigma are the right type:
        self._order = int(order)
        self._sigma = float(sigma)
        bvec_size = self.size(order)
        self._gsparams = GSParams.check(gsparams)

        # Make bvec if necessary
        if bvec is None:
            self._bvec = np.empty(bvec_size, dtype=float)
        else:
            if len(bvec) != bvec_size:
                raise GalSimIncompatibleValuesError(
                    "bvec is the wrong size for the provided order", bvec=bvec, order=order)
            self._bvec = np.ascontiguousarray(bvec, dtype=float)

    @lazy_property
    def _sbp(self):
        _bvec = self._bvec.__array_interface__['data'][0]
        return _galsim.SBShapelet(self._sigma, self._order, _bvec, self.gsparams._gsp)

    @classmethod
    def size(cls, order):
        """The size of the shapelet vector.
        """
        return (order+1)*(order+2)//2;

    @property
    def sigma(self):
        """The scale size, sigma.
        """
        return self._sigma

    @property
    def order(self):
        """The shapelet order.
        """
        return self._order

    @property
    def bvec(self):
        """The vector of shapelet coefficients
        """
        return self._bvec

    def getPQ(self,p,q):
        """Return the (p,q) coefficient.

        Parameters:
            p:      The p index to get.
            q:      The q index to get.

        Returns:
            a tuple (Re(b_pq), Im(b_pq))
        """
        pq = (p+q)*(p+q+1)//2 + 2*min(p,q)
        if p == q:
            return self._bvec[pq], 0
        elif p > q:
            return self._bvec[pq], self._bvec[pq+1]
        else:
            return self._bvec[pq], -self._bvec[pq+1]

    def getNM(self,N,m):
        """Return the coefficient according to N,m rather than p,q where N=p+q and m=p-q.

        Parameters:
            N:      The value of N=p+q to get.
            m:      The value of m=p-q to get.

        Returns:
            a tuple (Re(b_pq), Im(b_pq))
        """
        return self.getPQ((N+m)//2,(N-m)//2)

    def __eq__(self, other):
        return (self is other or
                (isinstance(other, Shapelet) and
                 self.sigma == other.sigma and
                 self.order == other.order and
                 np.array_equal(self.bvec, other.bvec) and
                 self.gsparams == other.gsparams))

    def __hash__(self):
        return hash(("galsim.Shapelet", self.sigma, self.order, tuple(self.bvec), self.gsparams))

    def __repr__(self):
        return 'galsim.Shapelet(sigma=%r, order=%r, bvec=%r, gsparams=%r)'%(
                self.sigma, self.order, self.bvec, self.gsparams)

    def __str__(self):
        return 'galsim.Shapelet(sigma=%s, order=%s, bvec=%s)'%(self.sigma, self.order, self.bvec)

    def __getstate__(self):
        d = self.__dict__.copy()
        d.pop('_sbp',None)
        return d

    def __setstate__(self, d):
        self.__dict__ = d

    @property
    def _maxk(self):
        return self._sbp.maxK()

    @property
    def _stepk(self):
        return self._sbp.stepK()

    @property
    def _centroid(self):
        return PositionD(self._sbp.centroid())

    @property
    def _flux(self):
        return self._sbp.getFlux()

    @property
    def _positive_flux(self):
        return self._sbp.getPositiveFlux()

    @property
    def _negative_flux(self):
        return self._sbp.getNegativeFlux()

    @lazy_property
    def _flux_per_photon(self):
        return self._calculate_flux_per_photon()

    @property
    def _max_sb(self):
        return self._sbp.maxSB()

    def _xValue(self, pos):
        return self._sbp.xValue(pos._p)

    def _kValue(self, kpos):
        return self._sbp.kValue(kpos._p)

    def _drawReal(self, image, jac=None, offset=(0.,0.), flux_scaling=1.):
        _jac = 0 if jac is None else jac.__array_interface__['data'][0]
        dx,dy = offset
        self._sbp.draw(image._image, image.scale, _jac, dx, dy, flux_scaling)

    def _drawKImage(self, image, jac=None):
        _jac = 0 if jac is None else jac.__array_interface__['data'][0]
        self._sbp.drawK(image._image, image.scale, _jac)

    @classmethod
    def fit(cls, sigma, order, image, center=None, normalization='flux', gsparams=None):
        """Fit for a shapelet decomposition of a given image.

        The optional ``normalization`` parameter mirrors the parameter of the `InterpolatedImage`
        class.  The following sequence should produce drawn images that are approximate matches to
        the original image::

            >>> image = [...]
            >>> shapelet = galsim.FitShapelet(sigma, order, image, normalization='sb')
            >>> im2 = shapelet.drawImage(image=im2, scale=image.scale, method='sb')
            >>> shapelet = galsim.FitShapelet(sigma, order, image, normalization='flux')
            >>> im3 = shapelet.drawImage(image=im3, scale=image.scale, method='no_pixel')

        Then ``im2`` and ``im3`` should be as close as possible to ``image`` for the given ``sigma``
        and ``order``.  Increasing the order can improve the fit, as can having ``sigma`` match the
        natural scale size of the image.  However, it should be noted that some images are not well
        fit by a shapelet for any (reasonable) order.

        Parameters:
            sigma:          The scale size in the standard units (usually arcsec).
            order:          The order of the shapelet decomposition.  This is the maximum
                            N=p+q included in the decomposition.
            image:          The `Image` for which to fit the shapelet decomposition
            center:         The position in pixels to use for the center of the decomposition.
                            [default: image.true_center]
            normalization:  The normalization to assume for the image.
                            [default: "flux"]
            gsparams:       An optional `GSParams` argument. [default: None]

        Returns:
            the fitted Shapelet profile
        """
        if not center:
            center = image.true_center
        # convert from PositionI if necessary
        center = PositionD(center)

        if not normalization.lower() in ("flux", "f", "surface brightness", "sb"):
            raise GalSimValueError("Invalid normalization requested.", normalization,
                                   ('flux', 'f', 'surface brightneess', 'sb'))

        ret = Shapelet(sigma, order, bvec=None, gsparams=gsparams)

        if image.wcs is not None and not image.wcs._isPixelScale:
            # TODO: Add ability for ShapeletFitImage to take jacobian matrix.
            raise GalSimNotImplementedError("Sorry, cannot (yet) fit a shapelet model to an image "
                                            "with a non-trivial WCS.")

        # Make it double precision if it is not.
        image = Image(image, dtype=np.float64, copy=False)

        _bvec = ret._bvec.__array_interface__['data'][0]
        _galsim.ShapeletFitImage(ret._sigma, ret._order, _bvec,
                                 image._image, image.scale, center._p)

        if normalization.lower() == "flux" or normalization.lower() == "f":
            ret._bvec /= image.scale**2

        return ret
