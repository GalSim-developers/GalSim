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

from .image import Image
from . import _galsim
from .errors import GalSimValueError

class BaseCDModel(object):
    """Base class for the most generic, i.e. no with symmetries or distance scaling relationships
    assumed, pixel boundary charge deflection model (as per Antilogus et al 2014).
    """

    def __init__(self, a_l, a_r, a_b, a_t):
        """Initialize a generic CDModel (charge deflection model).

        Usually this class will not be instantiated directly, but there is nothing to prevent you
        from doing so.  Each of the input a_l, a_r, a_b & a_t matrices must have the same shape and
        be odd-dimensioned.

        The model implemented here is described in Antilogus et al. (2014). The effective border
        of a pixel shifts to an extent proportional to the flux in a pixel at separation (dx,dy)
        and a coefficient a(dx,dy). Contributions of all neighbouring pixels are superposed. Border
        shifts are calculated for each (l=left, r=right (=positive x), b=bottom, t=top (=pos. y))
        border and the resulting change in flux in a pixel is the shift times the mean of its flux
        and the flux in the pixel on the opposite side of the border (caveat: in Antilogus et al.
        2014 the sum is used instead of the mean, making the a(dx,dy) a factor of 2 smaller).

        The parameters of the model are the a_l/r/b/t matrices, whose entry at (dy,dx) gives the
        respective shift coefficient. Note that for a realistic model, the matrices have a number
        of symmetries, as described in Antilogus et al. (2014). Use derived classes like PowerLawCD
        to have a model that automatically fulfills the symmetry conditions.

        Note that there is a gain factor included in the coefficients. When the a_* are measured
        from flat fields according to eqn. 4.10 in Antilogus et. al (2014) and applied to images
        that have the same gain as the flats, the correction is as intended. If the gain in the
        images is different, this can be accounted for with the gain_ratio parameter when calling
        applyForward or applyBackward.

        Parameters:
            a_l:    NumPy array containing matrix of deflection coefficients of left pixel border
            a_r:    NumPy array containing matrix of deflection coefficients of right pixel border
            a_b:    NumPy array containing matrix of deflection coefficients of bottom pixel border
            a_t:    NumPy array containing matrix of deflection coefficients of top pixel border
        """
        # Some basic sanity checking
        if (a_l.shape[0] % 2 != 1):
            raise GalSimValueError("Input array must be odd-dimensioned", a_l.shape)
        for a in (a_l, a_r, a_b, a_t):
            if a.shape[0] != a.shape[1]:
                raise GalSimValueError("Input array is not square", a.shape)
            if a.shape[0] != a_l.shape[0]:
                raise GalSimValueError("Input arrays not all the same dimensions", a.shape)
        # Save the relevant dimension and the matrices storing deflection coefficients
        self.n = a_l.shape[0] // 2
        if (self.n < 1):
            raise GalSimValueError("Input arrays must be at least 3x3", a_l.shape)

        self.a_l = Image(a_l, dtype=np.float64, make_const=True)
        self.a_r = Image(a_r, dtype=np.float64, make_const=True)
        self.a_b = Image(a_b, dtype=np.float64, make_const=True)
        self.a_t = Image(a_t, dtype=np.float64, make_const=True)

    def applyForward(self, image, gain_ratio=1.):
        """Apply the charge deflection model in the forward direction.

        Returns an image with the forward charge deflection transformation applied.  The input image
        is not modified, but its WCS is included in the returned image.

        Parameters:
            gain_ratio: Ratio of gain_image/gain_flat when shift coefficients were derived from
                        flat fields; default value is 1., which assumes the common case that your
                        flat and science images have the same gain value
        """
        ret = image.copy()
        _galsim._ApplyCD(
            ret._image, image._image, self.a_l._image, self.a_r._image, self.a_b._image,
            self.a_t._image, int(self.n), float(gain_ratio))
        return ret

    def applyBackward(self, image, gain_ratio=1.):
        """Apply the charge deflection model in the backward direction (accurate to linear order).

        Returns an image with the backward charge deflection transformation applied.  The input
        image is not modified, but its WCS is included in the returned image.

        Parameters:
            gain_ratio: Ratio of gain_image/gain_flat when shift coefficients were derived from
                        flat fields; default value is 1., which assumes the common case that your
                        flat and science images have the same gain value
        """
        retimage = self.applyForward(image, gain_ratio=-gain_ratio)
        return retimage

    def __repr__(self):
        return 'galsim.cdmodel.BaseCDModel(array(%r),array(%r),array(%r),array(%r))'%(
                self.a_l.array.tolist(), self.a_r.array.tolist(),
                self.a_b.array.tolist(), self.a_t.array.tolist())

    # Quick and dirty.  Just check reprs are equal.
    def __eq__(self, other): return self is other or repr(self) == repr(other)
    def __ne__(self, other): return not self.__eq__(other)
    def __hash__(self): return hash(repr(self))

# The _modelShiftCoeffX functions are used by the PowerLawCD class
def _modelShiftCoeffR(x, y, r0, t0, rx, tx, r, t, alpha):
    """Calculate the model shift coeff of right pixel border as a function of int pixel position
    (x, y).
    """
    # Invoke symmetry
    if y < 0: return _modelShiftCoeffR(x, -y, r0, t0, rx, tx, r, t, alpha)
    if x < 0: return -_modelShiftCoeffR(1 - x, y, r0, t0, rx, tx, r, t, alpha)
    # Invoke special immediate neighbour cases
    if x == 0 and y == 0: return -r0
    if x == 1 and y == 0: return +r0
    if x == 0 and y == 1: return -rx
    if x == 1 and y == 1: return +rx
    # Then, for remainder, apply power law model
    rr = np.sqrt((float(x) - .5)**2 + float(y)**2)
    cc = (x - 0.5) / rr # projection onto relevant axis
    return cc * r * rr**(-alpha)

def _modelShiftCoeffL(x, y, r0, t0, rx, tx, r, t, alpha):
    """Calculate the model shift coeff of left pixel border as a function of int pixel
    position (x, y).

    Equivalent to ``-_modelShiftCoeffR(x+1, y, *args)``.
    """
    return -_modelShiftCoeffR(x+1, y, r0, t0, rx, tx, r, t, alpha)

def _modelShiftCoeffT(x, y, r0, t0, rx, tx, r, t, alpha):
    """Calculate the model shift coeff of top pixel border as a function of int pixel
    position (x, y).
    """
    # Invoke symmetry
    if x < 0: return _modelShiftCoeffT(-x, y, r0, t0, rx, tx, r, t, alpha)
    if y < 0: return -_modelShiftCoeffT(x, 1 - y, r0, t0, rx, tx, r, t, alpha)
    # Invoke special immediate neighbour cases
    if x == 0 and y == 0: return -t0
    if x == 0 and y == 1: return +t0
    if x == 1 and y == 0: return -tx
    if x == 1 and y == 1: return +tx
    # Then, for remainder, apply power law model
    rr = np.sqrt((float(y) - .5)**2 + float(x)**2)
    cc = (y - 0.5) / rr # projection onto relevant axis
    return cc * t * rr**(-alpha)

def _modelShiftCoeffB(x, y, r0, t0, rx, tx, r, t, alpha):
    """Calculate the model shift coeff of bottom pixel border as a function of int pixel
    position (x, y)

    Equivalent to ``-_modelShiftCoeffT(x, y+1, *args)``.
    """
    return -_modelShiftCoeffT(x, y+1, r0, t0, rx, tx, r, t, alpha)

class PowerLawCD(BaseCDModel):
    """Class for parametrizing charge deflection coefficient strengths as a power law in distance
    from affected pixel border.
    """

    def __init__(self, n, r0, t0, rx, tx, r, t, alpha):
        """Initialize a power-law charge deflection model.

        The deflections from charges in the six pixels directly neighbouring a pixel border are
        modelled independently by the parameters ``r0``, ``t0`` (directly adjacent to borders
        between two pixels in the same row=y / column=x) and ``rx``, ``tx`` (pixels on the corner
        of pixel borders).

        Deflections due to charges further away are modelled as a power-law::

            a = A * numpy.sin(theta) * (r_distance)**(-alpha)

        where A is a power-law amplitude (``r`` for a_l / a_b and ``t`` for a_b / a_t), theta is
        the angle between the pixel border line and the line from border center to the other pixel
        center.

        Sign conventions are such that positive ``r0``, ``t0``, ``rx``, ``tx``, ``r``, ``t``
        correspond to physical deflection of equal charges (this is also how the theta above is
        defined).

        Parameters:
            n:      Maximum separation [pix] out to which charges contribute to deflection
            r0:     a_l(0,-1)=a_r(0,+1) deflection coefficient along x direction
            t0:     a_b(-1,0)=a_t(+1,0) deflection coefficient along y direction
            rx:     a_l(-1,-1)=a_r(+1,+1) diagonal contribution to deflection along x direction
            tx:     a_b(-1,-1)=a_t(+1,+1) diagonal contribution to deflection along y direction
            r:      Power-law amplitude for contribution to deflection along x from further away
            t:      Power-law amplitude for contribution to deflection along y from further away
            alpha:  Power-law exponent for deflection from further away
        """
        n = int(n)
        # First define x and y coordinates in a square grid of ints of shape (2n + 1) * (2n + 1)
        x, y = np.meshgrid(np.arange(2 * n + 1) - n, np.arange(2 * n + 1) - n)

        # prepare a_* matrices
        a_l = np.zeros((2 * n + 1, 2 * n + 1), dtype=np.float64)
        a_r = np.zeros((2 * n + 1, 2 * n + 1), dtype=np.float64)
        a_b = np.zeros((2 * n + 1, 2 * n + 1), dtype=np.float64)
        a_t = np.zeros((2 * n + 1, 2 * n + 1), dtype=np.float64)

        # Fill with power law model (slightly clunky loop but not likely a big time sink)
        # See https://github.com/GalSim-developers/GalSim/pull/592#discussion_r17027766 for a
        # discussion of the speeding up possibilities / timing results for this loop
        for ix in np.arange(0, 2 * n + 1):

            for iy in np.arange(0, 2 * n + 1):

                if(ix<2*n): # need to keep the other elements zero for flux conservation
                    a_l[iy, ix] = _modelShiftCoeffL(ix-n, iy-n, r0, t0, rx, tx, r, t, alpha)
                if(ix>0):
                    a_r[iy, ix] = _modelShiftCoeffR(ix-n, iy-n, r0, t0, rx, tx, r, t, alpha)
                if(iy<2*n):
                    a_b[iy, ix] = _modelShiftCoeffB(ix-n, iy-n, r0, t0, rx, tx, r, t, alpha)
                if(iy>0):
                    a_t[iy, ix] = _modelShiftCoeffT(ix-n, iy-n, r0, t0, rx, tx, r, t, alpha)

        BaseCDModel.__init__(self, a_l, a_r, a_b, a_t)
