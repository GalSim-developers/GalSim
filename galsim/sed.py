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
"""@file sed.py
Simple spectral energy distribution class.  Used by galsim/chromatic.py
"""

import copy

import numpy as np

import galsim

class SED(object):
    """Simple SED object to represent the spectral energy distributions of stars and galaxies.

    SEDs are callable, returning the flux in photons/nm as a function of wavelength in nm.

    SEDs are immutable; all transformative SED methods return *new* SEDs, and leave their
    originating SEDs unaltered.

    SEDs have `blue_limit` and `red_limit` attributes, which may be set to `None` in the case that
    the SED is defined by a python function or lambda `eval` string.  SEDs are considered undefined
    outside of this range, and __call__ will raise an exception if a flux is requested outside of
    this range.

    SEDs may be multiplied by scalars or scalar functions of wavelength.

    SEDs may be added together.  The resulting SED will only be defined on the wavelength
    region where both of the operand SEDs are defined. `blue_limit` and `red_limit` will be reset
    accordingly.

    The input parameter, spec, may be one of several possible forms:
    1. a regular python function (or an object that acts like a function)
    2. a galsim.LookupTable
    3. a file from which a LookupTable can be read in
    4. a string which can be evaluated into a function of `wave`
       via `eval('lambda wave : '+spec)
       e.g. spec = '0.8 + 0.2 * (wave-800)`

    The argument of `spec` will be the wavelength in either nanometers (default) or
    Angstroms depending on the value of `wave_type`.  The output should be the flux density at
    that wavelength.  (Note we use wave rather than lambda, since lambda is a python reserved
    word.)

    The argument `wave_type` specifies the units to assume for wavelength and must be one of
    'nm', 'nanometer', 'nanometers', 'A', 'Ang', 'Angstrom', or 'Angstroms'. Text case here
    is unimportant.  If these wavelength options are insufficient, please submit an issue to
    the GalSim github issues page: https://github.com/GalSim-developers/GalSim/issues

    The argument `flux_type` specifies the type of spectral density and must be one of:
    1. 'flambda':  `spec` is proportional to erg/nm
    2. 'fnu':      `spec` is proportional to erg/Hz
    3. 'fphotons': `spec` is proportional to photons/nm

    Note that the `wave_type` and `flux_type` parameters do not propagate into other methods of
    `SED`.  For instance, SED.__call__ assumes its input argument is in nanometers and returns
    flux proportional to photons/nm.


    @param spec          Function defining the spectrum at each wavelength.  See above for
                         valid options for this parameter.
    @param flux_type     String specifying what type of spectral density `spec` represents.  See
                         above for valid options for this parameter.
    @param wave_type     String specifying units for wavelength input to `spec`.

    """
    def __init__(self, spec, wave_type='nm', flux_type='flambda'):
        if wave_type.lower() in ['nm', 'nanometer', 'nanometers']:
            wave_factor = 1.0
        elif wave_type.lower() in ['a', 'ang', 'angstrom', 'angstroms']:
            wave_factor = 10.0
        else:
            raise ValueError("Unknown wave_type `{}` in SED.__init__".format(wave_type))

        if isinstance(spec, (str, unicode)):
            import os
            if os.path.isfile(spec):
                spec = galsim.LookupTable(file=spec)
            else:
                spec = eval('lambda wave : ' + spec)

        if isinstance(spec, galsim.LookupTable):
            self.blue_limit = spec.x_min / wave_factor
            self.red_limit = spec.x_max / wave_factor
        else:
            self.blue_limit = None
            self.red_limit = None

        if flux_type == 'flambda':
            self.fphotons = lambda w: spec(np.array(w) * wave_factor) * w
        elif flux_type == 'fnu':
            self.fphotons = lambda w: spec(np.array(w) * wave_factor) / w
        elif flux_type == 'fphotons':
            self.fphotons = lambda w: spec(np.array(w) * wave_factor)
        else:
            raise ValueError("Unknown flux_type `{}` in SED.__init__".format(flux_type))
        self.redshift = 0

    def _wavelength_intersection(self, other):
        blue_limit = self.blue_limit
        if other.blue_limit is not None:
            if blue_limit is None:
                blue_limit = other.blue_limit
            else:
                blue_limit = max([blue_limit, other.blue_limit])

        red_limit = self.red_limit
        if other.red_limit is not None:
            if red_limit is None:
                red_limit = other.red_limit
            else:
                red_limit = min([red_limit, other.red_limit])

        return blue_limit, red_limit

    def __call__(self, wave):
        """ Return photon density at wavelength `wave`.

        Note that outside of the wavelength range defined by the `blue_limit` and `red_limit`
        attributes, the SED is considered undefined, and this method will raise an exception if a
        flux at a wavelength outside the defined range is requested.

        @param   wave  Wavelength in nanometers at which to evaluate the SED.
        @returns       Photon density, Units proportional to photons/nm
        """
        if hasattr(wave, '__iter__'): # Only iterables respond to min(), max()
            wmin = min(wave)
            wmax = max(wave)
        else: # python scalar
            wmin = wave
            wmax = wave
        if self.blue_limit is not None:
            if wmin < self.blue_limit:
                raise ValueError("Wavelength ({0}) is bluer than SED blue limit ({1})"
                                 .format(wmin, self.blue_limit))
        if self.red_limit is not None:
            if wmax > self.red_limit:
                raise ValueError("Wavelength ({0}) redder than SED red limit ({1})"
                                 .format(wmax, self.red_limit))
        return self.fphotons(wave)

    def __mul__(self, other):
        if isinstance(other, galsim.GSObject):
            return galsim.Chromatic(other, self)
        # SEDs can be multiplied by scalars or functions (callables)
        ret = self.copy()
        if hasattr(other, '__call__'):
            ret.fphotons = lambda w: self.fphotons(w) * other(w)
        else:
            ret.fphotons = lambda w: self.fphotons(w) * other
        return ret

    def __rmul__(self, other):
        return self*other

    def __div__(self, other):
        # SEDs can be divided by scalars or functions (callables)
        ret = self.copy()
        if hasattr(other, '__call__'):
            ret.fphotons = lambda w: self.fphotons(w) / other(w)
        else:
            ret.fphotons = lambda w: self.fphotons(w) / other
        return ret

    def __rdiv__(self, other):
        # SEDs can be divided by scalars or functions (callables)
        ret = self.copy()
        if hasattr(other, '__call__'):
            ret.fphotons = lambda w: other(w) / self.fphotons(w)
        else:
            ret.fphotons = lambda w: other / self.fphotons(w)
        return ret

    def __truediv__(self, other):
        return self.__div__(other)

    def __rtruediv__(self, other):
        return self.__rdiv__(other)

    def __add__(self, other):
        # Add together two SEDs, with the following two caveats:
        # 1) The resulting SED will be defined on the wavelength range set by the overlap of the
        #    wavelength ranges of the two SED operands.
        # 2) The SED `redshift` attribute will be reset to zero.
        # This ensures that SED addition is commutative.

        # Find overlapping wavelength interval
        blue_limit, red_limit = self._wavelength_intersection(other)
        ret = self.copy()
        ret.blue_limit = blue_limit
        ret.red_limit = red_limit
        ret.fphotons = lambda w: self(w) + other(w)
        ret.redshift = 0
        return ret

    def __sub__(self, other):
        # Subtract two SEDs, with the caveat that the resulting SED will be defined on the
        # wavelength range set by the overlap of the wavelength ranges of the two SED operands.

        # Find overlapping wavelength interval
        return self.__add__(-1.0 * other)

    def copy(self):
        cls = self.__class__
        ret = cls.__new__(cls)
        for k, v in self.__dict__.iteritems():
            ret.__dict__[k] = copy.deepcopy(v) # need deepcopy for copying self.fphotons
        return ret

    def withFluxDensity(self, target_flux_density, wavelength):
        """ Return a new SED with flux density set to `target_flux_density` at wavelength
        `wavelength`.  Note that this normalization is *relative* to the `flux` attribute of the
        chromaticized GSObject.

        @param target_flux_density   The target *relative* normalization in photons / nm.
        @param wavelength   The wavelength, in nanometers, at which flux density will be set.
        @returns   New normalized SED.
        """
        current_fphotons = self(wavelength)
        factor = target_flux_density / current_fphotons
        ret = self.copy()
        ret.fphotons = lambda w: self.fphotons(w) * factor
        return ret

    def withFlux(self, target_flux, bandpass):
        """ Return a new SED with flux through the Bandpass `bandpass` set to `target_flux`. Note
        that this normalization is *relative* to the `flux` attribute of the chromaticized GSObject.

        @param target_flux  Desired *relative* flux normalization of the SED.
        @param bandpass   A galsim.Bandpass object defining a filter bandpass.
        @returns   New normalized SED.
        """
        current_flux = self.calculateFlux(bandpass)
        norm = target_flux/current_flux
        ret = self.copy()
        ret.fphotons = lambda w: self.fphotons(w) * norm
        return ret

    def atRedshift(self, redshift):
        """ Return a new SED with redshifted wavelengths.

        @param redshift
        @returns Redshifted SED.
        """
        ret = self.copy()
        wave_factor = (1.0 + redshift) / (1.0 + self.redshift)
        ret.fphotons = lambda w: self.fphotons(w / wave_factor)
        ret.blue_limit = self.blue_limit * wave_factor
        ret.red_limit = self.red_limit * wave_factor
        ret.redshift = redshift
        return ret

    def calculateFlux(self, bandpass):
        """ Return the SED flux through a bandpass.

        @param bandpass   galsim.Bandpass object representing a filter, or None for bolometric
                          flux (over defined wavelengths).
        @returns   Flux through bandpass.
        """
        if bandpass is None: # do bolometric flux
            if self.blue_limit is None:
                blue_limit = 0.0
            else:
                blue_limit = self.blue_limit
            if self.red_limit is None:
                red_limit = 1.e11 # = infinity in int1d
            else:
                red_limit = self.red_limit
            return galsim.integ.int1d(self.fphotons, blue_limit, red_limit)
        else: # do flux through bandpass
            if hasattr(bandpass, 'wave_list'):
                x = bandpass.wave_list
                return np.trapz(bandpass(x) * self.fphotons(x), x)
            else:
                return galsim.integ.int1d(lambda w: bandpass(w)*self.fphotons(w),
                                          bandpass.blue_limit, bandpass.red_limit)
