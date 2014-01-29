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
"""@file sed.py
Simple spectral energy distribution class.  Used by galsim.chromatic.py
"""

import numpy

import galsim

class SED(object):
    """Very simple SED container object."""
    def __init__(self, wave=None, flambda=None, fnu=None, fphotons=None):
        """ Initialize SED with a wavelength array and a flux density array.  The flux density
        array can be represented in one of three ways.

        @param wave     Array of wavelengths at which the SED is sampled.
        @param flambda  Array of flux density samples.  Units proprotional to erg/nm
        @param fnu      Array of flux density samples.  Units proprotional to erg/Hz
        @param fphotons Array of photon density samples.  Units proportional to photons/nm
        """
        self.wave = numpy.array(wave, dtype=numpy.float)
        self.redshift = 0.0
        # could be more careful here with factors of hbar, c, etc...
        if flambda is not None:
            self.fphotons = numpy.array(flambda * wave, dtype=numpy.float)
        elif fnu is not None:
            self.fphotons = numpy.array(fnu / wave, dtype=numpy.float)
        elif fphotons is not None:
            self.fphotons = numpy.array(fphotons, dtype=numpy.float)

        self.needs_new_interp = True

    def __call__(self, wave, force_new_interp=False):
        """ Uses a galsim.LookupTable to interpolate the photon density at the requested wavelength.
        The LookupTable is cached for future use.

        @param force_new_interp     Force rebuild of LookupTable.

        @returns photon density, Units proportional to photons/nm
        """
        interp = self._get_interp(force_new_interp=force_new_interp)
        return interp(wave)

    def _get_interp(self, force_new_interp=False):
        # Return SED LookupTable, rebuild if requested.
        if force_new_interp or self.needs_new_interp:
            self.interp = galsim.LookupTable(self.wave * (1.0 + self.redshift), self.fphotons)
            self.needs_new_interp=False
        return self.interp

    def __mul__(self, factor):
        # SED's can be multiplied by scalars.
        ret = self.copy()
        ret.fphotons *= factor
        ret.needs_new_interp = True
        return ret

    def __rmul__(self, factor):
        # SED's can be multiplied by scalars.
        ret = self.copy()
        ret.fphotons *= factor
        ret.needs_new_interp = True
        return ret

    def __imul__(self, factor):
        # SED's can be multiplied by scalars.
        self.fphotons *= factor
        self.needs_new_interp = True
        return self

    def __add__(self, other):
        # SED's can be added together, with a couple of caveats:
        # 1) The resulting SED will be defined on the wavelength range set by the overlap of
        #    the (possibly redshifted!) wavelength ranges of the two SED operands.
        # 2) The wavelength sampling of the resulting SED will be set to the union of the
        #    (possibly redshifted!) wavelength samplings of the SED operands.
        # 3) The redshift of the resulting SED will be set to 0.0 regardless of the redshifts of the
        #    SED operands.
        # These ensure that SED.__add__ is commutative.

        # Find overlapping wavelength interval
        bluelim = max([self.wave[0] * (1.0 + self.redshift),
                       other.wave[0] * (1.0 + other.redshift)])
        redlim = min([self.wave[-1] * (1.0 + self.redshift),
                      other.wave[-1] * (1.0 + other.redshift)])
        # Unionize wavelengths
        wave = set(self.wave * (1.0 + self.redshift)).union(other.wave * (1.0 + other.redshift))
        wave = numpy.array(list(wave))
        wave.sort()
        # Clip to overlap region
        wave = wave[(wave >= bluelim) & (wave <= redlim)]
        # Evaluate sum on new wavelength array
        fphotons = self(wave) + other(wave)

        ret = SED(wave=wave, fphotons=fphotons)
        return ret

    def copy(self):
        cls = self.__class__
        ret = cls.__new__(cls)
        for k, v in self.__dict__.iteritems():
            ret.__dict__[k] = copy.deepcopy(v) # need deepcopy for copying self.interp
        return ret

    def setNormalization(self, base_wavelength, normalization):
        """ Set photon density normalization at specified wavelength.  Note that this
        normalization is *relative* to the flux of the chromaticized GSObject.

        @param base_wavelength    The wavelength, in nanometers, at which the normalization will
                                  be set.
        @param normalization      The target *relative* normalization in photons / nm.
        """
        current_fphoton = self(base_wavelength)
        self.fphotons *= normalization/current_fphoton
        self.needs_new_interp = True

    def setFlux(self, bandpass, flux_norm):
        """ Set flux of SED when observed through given bandpass.  Note that the final number
        of counts drawn into an image is a function of both the SED and the chromaticized
        GSObject's flux attribute.

        @param bandpass   A galsim.Bandpass object defining a filter bandpass.
        @param flux_norm  Desired *relative* flux contribution from the SED.
        """
        current_flux = self.getFlux(bandpass)
        multiplier = flux_norm/current_flux
        self.fphotons *= multiplier
        self.needs_new_interp = True

    def setRedshift(self, redshift):
        """ Scale the wavelength axis of the SED.
        """
        self.redshift = redshift
        self.needs_new_interp=True

    def getFlux(self, bandpass):
        interp = self._get_interp()
        return galsim.integ.int1d(lambda w:bandpass(w)*interp(w), bandpass.bluelim, bandpass.redlim)
