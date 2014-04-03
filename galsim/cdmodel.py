# Copyright 2012-2014 The GalSim developers:
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

import numpy as np

class BaseCDModel(object):
    """Base class for the most generic, i.e. no with symmetries or distance scaling relationships
    assumed, pixel boundary charge deflection model (as per, e.g. Antilogus et al 2014).
    """

    def __init__(self, a_l, a_r, a_b, a_t):
        """Initialize a generic CDModel (charge deflection model) as described

        Usually this class will not be instantiated directly, but there is nothing to prevent you
        from doing so.  Each of the input a_l, a_r, a_b & a_t matrices must have the same shape and
        be odd-dimensioned.

        @param a_l  Array containing matrix of leftward deflection coefficients
        @param a_r  Array containing matrix of rightward deflection coefficients
        @param a_b  Array containing matrix of downward deflection coefficients
        @param a_t  Array containing matrix of upward deflection coefficients
        """
        # Some basic sanity checking
        if (a_l.shape[0] % 2 != 1):
            raise ValueError("Input array must be odd-dimensioned")
        for a in (a_l, a_r, a_b, a_t):
            if a.shape[0] != a.shape[1]:
                raise ValueError("Input array is not square")
            if a.shape[0] != a_l.shape[0]:
                raise ValueError("Input arrays not all the same dimensions")
        self.n = a_l.shape[0] / 2
        self.a_l = a_l
        self.a_r = a_r
        self.a_b = a_b
        self.a_t = a_t
        # Save all these arrays in the contiguous, format.  This assumes that the input arrays were
        # ordered as per usual in NumPy, i.e. array[y, x]
        self._a_lrbt = np.concatenate((a_l.flatten(), a_r.flatten(), a_b.flatten(), a_t.flatten()))

    def applyForward(self, image):
        """Apply the charge deflection model in the forward direction
        """
        return image.applyCD(self.n, self._a_lrbt)

    def applyBackward(self, image):
        """Apply the charge deflection model in the backward direction (to linear order)
        """
        return image.applyCD(self.n, -self._a_lrbt)


    class PowerLawCD(BaseCDModel):

        @staticmethod
        def _modelShiftCoeffR(x, y, r0, t0, rx, tx, r, t, alpha):
            """Calculate the rightward model shift coeff as a function of int pixel position x, y
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

        @staticmethod
        def _modelShiftCoeffL(x, y, r0, t0, rx, tx, r, t, alpha):
            """Calculate the leftward model shift coeff as a function of int pixel position x, y

            Equal to -_modelShiftCoeffR(*args)
            """
            return -_modelShiftCoeffR(x, y, r0, t0, rx, tx, r, t, alpha)

        def __init__(self, n, r0, t0, rx, tx, r, t, alpha):
            """Initialize a power-law charge deflection model.
            """
            # First define x and y coordinates in a square grid of shape (2n + 1) * (2n + 1)
            x, y = np.meshgrid(np.arange(2 * n + 1) - n, np.arange(2 * n + 1) - n)
            # Do special cases first




