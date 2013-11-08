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
"""@file chromatic.py
Definitions for GalSim classes implementing wavelength-dependence.

This file extends the base GalSim classes by allowing them to be wavelength dependent.  This allows
one to implement wavelength-dependent PSFs or galaxies with color gradients.
"""

import galsim

class ChromaticObject(object):
    """Base class for defining wavelength dependent objects.
    """
    def draw(self, wave, throughput, image=None):
        """Draws an Image of a chromatic object as observed through a bandpass filter.

        @param wave        Wavelengths in nanometers describing bandpass filter.  For now these are
                           assumed to be linearly spaced.

        @param throughput  Dimensionless throughput at each wavelength, i.e. the probability that any
                           particular photon at the corresponding wavelength is detected.

        @returns           The drawn image.
        """

        #Assume that wave is linear, and compute dwave.
        dwave = wave[1] - wave[0]

        #Initialize Image from first wavelength.
        prof = self.evaluateAtWavelength(wave[0]) * throughput[0]
        image = prof.draw(image=image)

        #And now build it up at remaining wavelengths
        for w, tp in zip(wave, throughput)[1:]:
            prof = self.evaluateAtWavelength(w) * tp
            prof.draw(image=image, add_to_image=True)

        return image

class ChromaticBaseObject(ChromaticObject):
    def __init__(self, gsobj, wave, flambda, **kwargs):
        self.wave = wave
        self.flambda = flambda
        self.gsobj = gsobj(**kwargs)

    def applyShear(self, shear):
        self.gsobj.applyShear(shear)

    def evaluateAtWavelength(self, wave):
        import numpy as np
        tp = np.interp(wave, self.wave, self.flambda)*wave
        return self.gsobj*tp

class ChromaticAdd(ChromaticObject):
    def __init__(self, objlist):
        self.objlist = objlist

    def evaluateAtWavelength(self, wave):
        return galsim.Add([obj.evaluateAtWavelength(wave)
                           if hasattr(obj, 'evaluateAtWavelength')
                           else obj
                           for obj in self.objlist])

class ChromaticConvolve(ChromaticObject):
    def __init__(self, objlist):
        self.objlist = objlist

    def evaluateAtWavelength(self, wave):
        return galsim.Convolve([obj.evaluateAtWavelength(wave)
                                if hasattr(obj, 'evaluateAtWavelength')
                                else obj
                                for obj in self.objlist])

class ChromaticShiftAndScale(ChromaticObject):
    def __init__(self, gsobj,
                 centering_fn=None, sizing_fn=None,
                 **kwargs):
        self.gsobj = gsobj(**kwargs)
        self.centering_fn = centering_fn
        self.sizing_fn = sizing_fn

    def applyShear(self, shear):
        self.gsobj.applyShear(shear)

    def evaluateAtWavelength(self, wave):
        PSF = self.gsobj.copy()
        size = self.sizing_fn(wave)
        shift = self.centering_fn(wave)
        print 'galsim, ', wave, size, shift
        PSF.applyDilation(size)
        PSF.applyShift(shift)
        return PSF
