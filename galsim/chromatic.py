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

        Draws the image wavelength by wavelength using a Riemann sum.

        @param wave        Wavelengths in nanometers describing bandpass filter.  For now these are
                           assumed to be linearly spaced.

        @param throughput  Dimensionless throughput at each wavelength, i.e. the probability that any
                           particular photon at the corresponding wavelength is detected.

        @returns           The drawn image.
        """

        #Assume that wave is linear, and compute constant dwave.
        dwave = wave[1] - wave[0]

        #Initialize Image from first wavelength.
        prof = self.evaluateAtWavelength(wave[0]) * throughput[0] * dwave
        image = prof.draw(image=image)

        #And now build it up at remaining wavelengths
        for w, tp in zip(wave, throughput)[1:]:
            prof = self.evaluateAtWavelength(w) * tp * dwave
            prof.draw(image=image, add_to_image=True)

        return image

class ChromaticBaseObject(ChromaticObject):
    """Construct chromatic versions of the galsim.base objects.

    This class extends the base GSObjects in basy.py by adding SEDs.  Useful to consistently generate
    images through different filters, or, with the ChromaticAdd class, to construct
    multi-component galaxies, each with a different SED. For example, a bulge+disk galaxy could be
    constructed:

    >>> bulge_wave, bulge_photons = user_function_to_get_bulge_spectrum()
    >>> disk_wave, disk_photons = user_function_to_get_disk_spectrum()
    >>> bulge = galsim.ChromaticBaseObject(galsim.Sersic, bulge_wave, bulge_photons,
                                           n=4, half_light_radius=1.0)
    >>> disk = galsim.ChromaticBaseObject(galsim.Sersic, disk_wave, disk_photons,
                                          n=1, half_light_radius=2.0)
    >>> gal = galsim.ChromaticAdd([bulge, disk])

    Notice that positional and keyword arguments which apply to the specific base class being
    generalized (e.g. n=4, half_light_radius = 1.0 for the bulge component) are passed to
    ChromaticBaseObject after the base object type and SED.

    The SED is specified with a wavelength array, and a photon array.  At present the wavelength
    array is assumed to be linear.  The photon array specifies the distribution of source photons
    over wavelength, i.e it is proportional to f_lambda * lambda.  Flux normalization is set
    such that the Riemann sum over the wavelength and photon array is equal to the total number of
    photons in an image with infinite area.
    """
    def __init__(self, gsobj, wave, photons, *args, **kwargs):
        """Initialize ChromaticBaseObject.

        @param gsobj    One of the GSObjects defined in base.py.  Possibly works with other GSObjects
                        too.
        @param wave     Wavelength array in nanometers for SED.
        @param photons  Photon array in photons per nanometers.
        @param args
        @param kwargs   Additional positional and keyword arguments are forwarded to the gsobj
                        constructor.
        """
        self.wave = wave
        self.photons = photons
        self.gsobj = gsobj(*args, **kwargs)

    def applyShear(self, *args, **kwargs):
        self.gsobj.applyShear(*args, **kwargs)

    def evaluateAtWavelength(self, wave):
        """
        @param wave  Wavelength in nanometers.
        @returns     GSObject for profile at specified wavelength
        """
        import numpy as np
        p = np.interp(wave, self.wave, self.photons)
        return p * self.gsobj

class ChromaticAdd(ChromaticObject):
    """Add ChromaticObjects and/or GSObjects together.  GSObjects are treated as having flat spectra.
    """
    def __init__(self, objlist):
        self.objlist = objlist

    def evaluateAtWavelength(self, wave):
        """
        @param wave  Wavelength in nanometers.
        @returns     GSObject for profile at specified wavelength
        """
        return galsim.Add([obj.evaluateAtWavelength(wave)
                           if hasattr(obj, 'evaluateAtWavelength')
                           else obj
                           for obj in self.objlist])

class ChromaticConvolve(ChromaticObject):
    """Convolve ChromaticObjects and/or GSObjects together.  GSObjects are treated as having flat
    spectra.
    """
    def __init__(self, objlist):
        self.objlist = objlist

    def evaluateAtWavelength(self, wave):
        """
        @param wave  Wavelength in nanometers.
        @returns     GSObject for profile at specified wavelength
        """
        return galsim.Convolve([obj.evaluateAtWavelength(wave)
                                if hasattr(obj, 'evaluateAtWavelength')
                                else obj
                                for obj in self.objlist])

class ChromaticShiftAndDilate(ChromaticObject):
    """Class representing chromatic profiles whose wavelength dependence consists of shifting and
    dilating a fiducial profile.

    By simply shifting and dilating a fiducial PSF, a number of wavelength-dependent effects can be
    effected.  For instance, differential chromatic refraction is just shifting the PSF center as a
    function of wavelength.  The wavelength-dependence of seeing, and the wavelength-dependence of
    the diffraction limit are dilations.  This class can compactly represent all of these effects.
    See tests/test_chromatic.py for an example.
    """
    def __init__(self, gsobj,
                 shift_fn=None, dilate_fn=None,
                 **kwargs):
        """
        @param gsobj      Fiducial galsim.base profile to shift and dilate.
        @param shift_fn   Function that takes wavelength in nanometers and returns a
                          galsim.Position object, or parameters which can be transformed into a
                          galsim.Position object (dx, dy).
        @param dilate_fn  Function that takes wavelength in nanometers and returns a dilation
                          scale factor.
        """
        self.gsobj = gsobj(**kwargs)
        self.shift_fn = shift_fn
        self.dilate_fn = dilate_fn

    def applyShear(self, *args, **kwargs):
        self.gsobj.applyShear(*args, **kwargs)

    def evaluateAtWavelength(self, wave):
        """
        @param wave  Wavelength in nanometers.
        @returns     GSObject for profile at specified wavelength
        """
        PSF = self.gsobj.copy()
        PSF.applyDilation(self.dilate_fn(wave))
        PSF.applyShift(self.shift_fn(wave))
        return PSF
