# Copyright (c) 2012-2014 by the GalSim developers team on GitHub
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
import galsim


class BaseCDModel(object):
    """Base class for the most generic, i.e. no with symmetries or distance scaling relationships
    assumed, pixel boundary charge deflection model (as per, e.g. Antilogus et al 2014).
    """

    def __init__(self, a_l, a_r, a_b, a_t):
        """Initialize a generic CDModel (charge deflection model) as described

        Usually this class will not be instantiated directly, but there is nothing to prevent you
        from doing so.  Each of the input a_l, a_r, a_b & a_t matrices must have the same shape and
        be odd-dimensioned.
        
        The model implemented here is described in Antilogus et al. (2014). The effective border
        of a pixel shifts to an extent proportional to the flux in a pixel at separation (dx,dy) 
        and a coefficient a(dx,dy). Contributions of all neighbouring pixels are superposed. Border
        shifts are calculated for each (l=left, r=right (=positive x), b=bottom, t=top (=pos. y)) 
        border and the resulting change in flux in a pixel is the shift times the mean of its flux
        and the flux in the pixel on the opposite side of the border.
        
        The parameters of the model are the a_l/r/b/t matrices, whose entry at (dy,dx) gives the
        respective shift coefficient. Note that for a realistic model, the matrices have a number
        of symmetries, as described in Antilogus et al. (2014). Use derived classes like PowerLawCD
        to have a model that automatically fulfills the symmetry conditions.

        @param a_l  Array containing matrix of deflection coefficients of left pixel border
        @param a_r  Array containing matrix of deflection coefficients of right pixel border
        @param a_b  Array containing matrix of deflection coefficients of bottom pixel border
        @param a_t  Array containing matrix of deflection coefficients of top pixel border
        """
        # Some basic sanity checking
        if (a_l.shape[0] % 2 != 1):
            raise ValueError("Input array must be odd-dimensioned")
        for a in (a_l, a_r, a_b, a_t):
            if a.shape[0] != a.shape[1]:
                raise ValueError("Input array is not square")
            if a.shape[0] != a_l.shape[0]:
                raise ValueError("Input arrays not all the same dimensions")
        # Save the relevant dimension and the matrices storing deflection coefficients
        self.n = a_l.shape[0] / 2
        self.a_l = a_l
        self.a_r = a_r
        self.a_b = a_b
        self.a_t = a_t
        # Also save all these arrays in flattened format as Image instance (dtype=float) for easy
        # passing to C++ via Python wrapping code
        self._a_l_flat = galsim.Image(
            np.reshape(a_l.flatten(), (1, np.product(a_l.shape))), dtype=np.float64,
            make_const=True)
        self._a_r_flat = galsim.Image(
            np.reshape(a_r.flatten(), (1, np.product(a_r.shape))), dtype=np.float64,
            make_const=True)
        self._a_b_flat = galsim.Image(
            np.reshape(a_b.flatten(), (1, np.product(a_b.shape))), dtype=np.float64,
            make_const=True)
        self._a_t_flat = galsim.Image(
            np.reshape(a_t.flatten(), (1, np.product(a_t.shape))), dtype=np.float64,
            make_const=True)
            
        # Also save inverse this once, otherwise we'll run into a const cast later
        self._a_l_flat_inv = galsim.Image(
            np.reshape((-a_l).flatten(), (1, np.product(a_l.shape))), dtype=np.float64,
            make_const=True)
        self._a_r_flat_inv = galsim.Image(
            np.reshape((-a_r).flatten(), (1, np.product(a_r.shape))), dtype=np.float64,
            make_const=True)
        self._a_b_flat_inv = galsim.Image(
            np.reshape((-a_b).flatten(), (1, np.product(a_b.shape))), dtype=np.float64,
            make_const=True)
        self._a_t_flat_inv = galsim.Image(
            np.reshape((-a_t).flatten(), (1, np.product(a_t.shape))), dtype=np.float64,
            make_const=True)

    def applyForward(self, image):
        """Apply the charge deflection model in the forward direction.

        Returns an image with the forward charge deflection transformation applied.  The input image
        is not modified, but its WCS is included in the returned image.
        """
        retimage = galsim.Image(
            image=image.image.applyCD(
                self._a_l_flat.image, self._a_r_flat.image,
                self._a_b_flat.image, self._a_t_flat.image, self.n),
            wcs=image.wcs)
        return retimage

    def applyBackward(self, image):
        """Apply the charge deflection model in the backward direction (accurate to linear order).

        Returns an image with the backward charge deflection transformation applied.  The input
        image is not modified, but its WCS is included in the returned image.
        """
        retimage = galsim.Image(
            image=image.image.applyCD(
                self._a_l_flat_inv.image, self._a_r_flat_inv.image,
                self._a_b_flat_inv.image, self._a_t_flat_inv.image, self.n),
            wcs=image.wcs)
        return retimage


def _modelShiftCoeffR(x, y, r0, t0, rx, tx, r, t, alpha):
    """Calculate the model shift coeff of right pixel border as a function of int pixel position
    (x, y)
    """
    if not isinstance(x, (int, long)):
        raise ValueError("Input x coordinate must be an int or long")
    if not isinstance(y, (int, long)):
        raise ValueError("Input x coordinate must be an int or long")
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
    position (x, y)

    Equal to -_modelShiftCoeffR(x+1, y, *args)
    """
    return -_modelShiftCoeffR(x+1, y, r0, t0, rx, tx, r, t, alpha)

def _modelShiftCoeffT(x, y, r0, t0, rx, tx, r, t, alpha):
    """Calculate the model shift coeff of top pixel border as a function of int pixel
    position (x, y)
    """
    if not isinstance(x, (int, long)):
        raise ValueError("Input x coordinate must be an int or long")
    if not isinstance(y, (int, long)):
        raise ValueError("Input x coordinate must be an int or long")
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

    Equal to -_modelShiftCoeffT(x, y+1, *args)
    """
    return -_modelShiftCoeffT(x, y+1, r0, t0, rx, tx, r, t, alpha)

class PowerLawCD(BaseCDModel):
    """Class for parametrizing charge deflection coefficient strengths as a power law in distance
    from affected pixel border
    """

    def __init__(self, n, r0, t0, rx, tx, r, t, alpha):
        """Initialize a power-law charge deflection model.
        
        The deflections from charges in the six pixels directly neighbouring a pixel border are 
        modelled independently by the parameters r0, t0 (directly adjacent to borders between 
        two pixels in the same row=y / column=x) and rx, tx (pixels on the corner of pixel borders)
        
        Deflections due to charges further away are modelled as a power-law,
          a = A * sin(theta) * r^(-alpha)
        where A is a power-law amplitude (r for a_l / a_b and t for a_b / a_t), theta is the angle
        between pixel border line and line from border center to other pixel center.
        
        Sign convention is such that positive r0,t0,rx,tx,r,t correspond to physical deflection of
        equal charges (this is also how the theta above is defined).
        
        @param n      Maximum separation [pix] out to which charges contribute to deflection
        @param r0     a_l(0,-1)=a_r(0,+1) deflection coefficient along x direction
        @param t0     a_b(-1,0)=a_t(+1,0) deflection coefficient along y direction
        @param rx     a_l(-1,-1)=a_r(+1,+1) diagonal contribution to deflection along x direction
        @param tx     a_b(-1,-1)=a_t(+1,+1) diagonal contribution to deflection along y direction
        @param r      power-law amplitude for contribution to deflection along x from further away
        @param t      power-law amplitude for contribution to deflection along y from further away
        @param alpha  power-law exponent for deflection from further away
        
        """
        # First define x and y coordinates in a square grid of shape (2n + 1) * (2n + 1)
        x, y = np.meshgrid(np.arange(2 * n + 1) - n, np.arange(2 * n + 1) - n)

        # prepare a_* matrices
        a_l = np.zeros((2 * n + 1, 2 * n + 1))
        a_r = np.zeros((2 * n + 1, 2 * n + 1))
        a_b = np.zeros((2 * n + 1, 2 * n + 1))
        a_t = np.zeros((2 * n + 1, 2 * n + 1))

        # fill with power law model (slightly clunky loop but not likely a big time sink)
        for ix in np.arange(0, 2*n + 1):

            for iy in np.arange(0, 2*n + 1):
	        if(ix<2*n): # need to keep the other elements zero for flux conservation
                  a_l[iy, ix] = _modelShiftCoeffL(ix-n, iy-n, r0, t0, rx, tx, r, t, alpha)
                if(ix>0):
                  a_r[iy, ix] = _modelShiftCoeffR(ix-n, iy-n, r0, t0, rx, tx, r, t, alpha)
                if(iy<2*n):
                  a_b[iy, ix] = _modelShiftCoeffB(ix-n, iy-n, r0, t0, rx, tx, r, t, alpha)
                if(iy>0):
                  a_t[iy, ix] = _modelShiftCoeffT(ix-n, iy-n, r0, t0, rx, tx, r, t, alpha)

        BaseCDModel.__init__(self, a_l, a_r, a_b, a_t)
