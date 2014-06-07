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
"""@file sed.py
Simple spectral energy distribution class.  Used by galsim/chromatic.py
"""

import copy

import numpy as np

import galsim
import utilities

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

    The input parameter, `spec`, may be one of several possible forms:
    1. a regular python function (or an object that acts like a function)
    2. a LookupTable
    3. a file from which a LookupTable can be read in
    4. a string which can be evaluated into a function of `wave`
       via `eval('lambda wave : '+spec)
       e.g. spec = '0.8 + 0.2 * (wave-800)`

    The argument of `spec` will be the wavelength in either nanometers (default) or Angstroms
    depending on the value of `wave_type`.  The output should be the flux density at that
    wavelength.  (Note we use `wave` rather than `lambda`, since `lambda` is a python reserved
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
    @param wave_type     String specifying units for wavelength input to `spec`. [default: 'nm']
    @param flux_type     String specifying what type of spectral density `spec` represents.  See
                         above for valid options for this parameter. [default: 'flambda']

    """
    def __init__(self, spec, wave_type='nm', flux_type='flambda'):
        # Figure out input wavelength type
        if wave_type.lower() in ['nm', 'nanometer', 'nanometers']:
            wave_factor = 1.0
        elif wave_type.lower() in ['a', 'ang', 'angstrom', 'angstroms']:
            wave_factor = 10.0
        else:
            raise ValueError("Unknown wave_type '{0}'".format(wave_type))

        # Figure out input flux density type
        if isinstance(spec, basestring):
            import os
            if os.path.isfile(spec):
                spec = galsim.LookupTable(file=spec, interpolant='linear')
            else:
                origspec = spec
                # Don't catch ArithmeticErrors when testing to see if the the result of `eval()`
                # is valid since `spec = '1./(wave-700)'` will generate a ZeroDivisionError (which
                # is a subclass of ArithmeticError) despite being a valid spectrum specification,
                # while `spec = 'blah'` where `blah` is undefined generates a NameError and is not
                # a valid spectrum specification.
                # Are there any other types of errors we should trap here?
                try:
                    spec = eval('lambda wave : ' + spec)   # This can raise SyntaxError
                    spec(700)   # This can raise NameError or ZeroDivisionError
                except ArithmeticError:
                    pass
                except:
                    raise ValueError(
                        "String spec must either be a valid filename or something that "+
                        "can eval to a function of wave. Input provided: {0}".format(origspec))

        if isinstance(spec, galsim.LookupTable):
            self.blue_limit = spec.x_min / wave_factor
            self.red_limit = spec.x_max / wave_factor
            self.wave_list = np.array(spec.getArgs())/wave_factor
        else:
            self.blue_limit = None
            self.red_limit = None
            self.wave_list = np.array([], dtype=float)

        if flux_type == 'flambda':
            self.fphotons = lambda w: spec(np.array(w) * wave_factor) * w
        elif flux_type == 'fnu':
            self.fphotons = lambda w: spec(np.array(w) * wave_factor) / w
        elif flux_type == 'fphotons':
            self.fphotons = lambda w: spec(np.array(w) * wave_factor)
        else:
            raise ValueError("Unknown flux_type '{0}'".format(flux_type))
        self.redshift = 0

        # Hack to avoid (LookupTable.x_max * 10) / 10.0 > LookupTable.x_max due to roundoff
        # error.
        if len(self.wave_list) > 0.0:
            self.wave_list[0] = self.wave_list[0] + 0.0000001
            self.wave_list[-1] = self.wave_list[-1] - 0.0000001

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

        @param wave     Wavelength in nanometers at which to evaluate the SED.

        @returns the photon density in units of photons/nm
        """
        if hasattr(wave, '__iter__'): # Only iterables respond to min(), max()
            wmin = min(wave)
            wmax = max(wave)
        else: # python scalar
            wmin = wave
            wmax = wave
        if self.blue_limit is not None:
            if wmin < self.blue_limit:
                raise ValueError("Requested wavelength ({0}) is bluer than blue_limit ({1})"
                                 .format(wmin, self.blue_limit))
        if self.red_limit is not None:
            if wmax > self.red_limit:
                raise ValueError("Requested wavelength ({0}) is redder than red_limit ({1})"
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
        # Add together two SEDs, with the following caveats:
        # 1) The resulting SED will be defined on the wavelength range set by the overlap of the
        #    wavelength ranges of the two SED operands.
        # 2) If both SEDs maintain a `wave_list` attribute, then the new `wave_list` will be
        #    the union of the old `wave_list`s in the intersecting region.
        # 3) The resulting SED `redshift` attribute will be set to zero.
        # This ensures that SED addition is commutative.

        # Find overlapping wavelength interval
        blue_limit, red_limit = self._wavelength_intersection(other)
        ret = self.copy()
        ret.blue_limit = blue_limit
        ret.red_limit = red_limit
        ret.fphotons = lambda w: self(w) + other(w)
        ret.redshift = 0
        if len(self.wave_list) > 0 and len(other.wave_list) > 0:
            wave_list = np.union1d(self.wave_list, other.wave_list)
            wave_list = wave_list[wave_list <= red_limit]
            wave_list = wave_list[wave_list >= blue_limit]
            ret.wave_list = wave_list
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

        @param target_flux_density  The target *relative* normalization in photons / nm.
        @param wavelength           The wavelength, in nm, at which flux density will be set.

        @returns the new normalized SED.
        """
        current_fphotons = self(wavelength)
        factor = target_flux_density / current_fphotons
        ret = self.copy()
        ret.fphotons = lambda w: self.fphotons(w) * factor
        return ret

    def withFlux(self, target_flux, bandpass):
        """ Return a new SED with flux through the Bandpass `bandpass` set to `target_flux`. Note
        that this normalization is *relative* to the `flux` attribute of the chromaticized GSObject.

        @param target_flux  The desired *relative* flux normalization of the SED.
        @param bandpass     A Bandpass object defining a filter bandpass.

        @returns the new normalized SED.
        """
        current_flux = self.calculateFlux(bandpass)
        norm = target_flux/current_flux
        ret = self.copy()
        ret.fphotons = lambda w: self.fphotons(w) * norm
        return ret

    def withMagnitude(self, target_magnitude, bandpass):
        """ Return a new SED with `bandpass` magnitude set to `target_magnitude`. Note that this
        normalization is *relative* to the `flux` attribute of the chromaticized GSObject.

        @param target_magnitude  The desired *relative* magnitude of the SED.
        @param bandpass          A Bandpass object defining a filter bandpass.

        @returns the new normalized SED.
        """
        current_magnitude = self.calculateMagnitude(bandpass)
        norm = 10**(-0.4*(target_magnitude - current_magnitude))
        ret = self.copy()
        ret.fphotons = lambda w: self.fphotons(w) * norm
        return ret

    def atRedshift(self, redshift):
        """ Return a new SED with redshifted wavelengths.

        @param redshift

        @returns the redshifted SED.
        """
        ret = self.copy()
        wave_factor = (1.0 + redshift) / (1.0 + self.redshift)
        ret.fphotons = lambda w: self.fphotons(w / wave_factor)
        ret.wave_list = self.wave_list * wave_factor
        if ret.blue_limit is not None:
            ret.blue_limit = self.blue_limit * wave_factor
        if ret.red_limit is not None:
            ret.red_limit = self.red_limit * wave_factor
        ret.redshift = redshift
        return ret

    def calculateFlux(self, bandpass):
        """ Return the SED flux through a Bandpass `bandpass`.

        @param bandpass   A Bandpass object representing a filter, or None for bolometric
                          flux (over defined wavelengths).

        @returns the flux through the bandpass.
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
            if len(bandpass.wave_list) > 0:
                x = np.union1d(bandpass.wave_list, self.wave_list)
                x = x[(x <= bandpass.red_limit) & (x >= bandpass.blue_limit)]
                return np.trapz(bandpass(x) * self.fphotons(x), x)
            else:
                return galsim.integ.int1d(lambda w: bandpass(w)*self.fphotons(w),
                                          bandpass.blue_limit, bandpass.red_limit)

    def calculateMagnitude(self, bandpass):
        """ Return the SED magnitude through a Bandpass `bandpass`.

        @param bandpass   A Bandpass object representing a filter, or None for bolometric
                          magnitude (over defined wavelengths).

        @returns the bandpass magnitude.
        """
        current_flux = self.calculateFlux(bandpass)
        return -2.5 * np.log10(current_flux) - bandpass.getZeroPoint()

    def thin(self, rel_err=1.e-4, preserve_range=False):
        """ If the SED was initialized with a LookupTable or from a file (which internally creates a
        LookupTable), then remove tabulated values while keeping the integral over the set of
        tabulated values still accurate to `rel_err`.

        @param rel_err            The relative error allowed in the integral over the SED
                                  [default: 1.e-4]
        @param preserve_range     Should the original range (`blue_limit` and `red_limit`) of the
                                  SED be preserved? (True) Or should the ends be trimmed to
                                  include only the region where the integral is significant? (False)
                                  [default: False]

        @returns the thinned SED.
        """
        if len(self.wave_list) > 0:
            x = self.wave_list
            f = self(x)
            newx, newf = utilities.thin_tabulated_values(x, f, rel_err=rel_err,
                                                         preserve_range=preserve_range)
            return SED(galsim.LookupTable(newx, newf, interpolant='linear'),
                       flux_type='fphotons')


    def calculateDCRMomentShifts(self, bandpass, **kwargs):
        """ Calculates shifts in first and second moments of PSF due to differential chromatic
        refraction (DCR).

        @param bandpass             Bandpass through which object is being imaged.
        @param zenith_angle         Angle from object to zenith, expressed as an Angle
        @param parallactic_angle    Parallactic angle, i.e. the position angle of the zenith,
                                    measured from North through East.  [default: 0]
        @param obj_coord            Celestial coordinates of the object being drawn as a
                                    CelestialCoord. [default: None]
        @param zenith_coord         Celestial coordinates of the zenith as a CelestialCoord.
                                    [default: None]
        @param HA                   Hour angle of the object as an Angle. [default: None]
        @param latitude             Latitude of the observer as an Angle. [default: None]
        @param pressure             Air pressure in kiloPascals.  [default: 69.328 kPa]
        @param temperature          Temperature in Kelvins.  [default: 293.15 K]
        @param H2O_pressure         Water vapor pressure in kiloPascals.  [default: 1.067 kPa]

        @returns a tuple.  The first element is the vector of DCR first moment shifts, and the
                 second element is the 2x2 matrix of DCR second (central) moment shifts.
        """
        if 'zenith_angle' in kwargs:
            zenith_angle = kwargs.pop('zenith_angle')
            parallactic_angle = kwargs.pop('parallactic_angle', 0.0*galsim.degrees)
        elif 'obj_coord' in kwargs:
            obj_coord = kwargs.pop('obj_coord')
            if 'zenith_coord' in kwargs:
                zenith_coord = kwargs.pop('zenith_coord')
                zenith_angle, parallactic_angle = galsim.dcr.zenith_parallactic_angles(
                    obj_coord=obj_coord, zenith_coord=zenith_coord)
            else:
                if 'HA' not in kwargs or 'latitude' not in kwargs:
                    raise TypeError("calculateDCRMomemntShifts requires either zenith_coord or "+
                                    "(HA, latitude) when obj_coord is specified!")
                HA = kwargs.pop('HA')
                latitude = kwargs.pop('latitude')
                zenith_angle, parallactic_angle = galsim.dcr.zenith_parallactic_angles(
                    obj_coord=obj_coord, HA=HA, latitude=latitude)
        else:
            raise TypeError(
                "Need to specify zenith_angle and parallactic_angle in calculateDCRMomentShifts!")
        # Any remaining kwargs will get forwarded to galsim.dcr.get_refraction
        # Check that they're valid
        for kw in kwargs.keys():
            if kw not in ['temperature', 'pressure', 'H2O_pressure']:
                raise TypeError("Got unexpected keyword in ChromaticAtmosphere: {0}".format(kw))
        # Now actually start calculating things.
        flux = self.calculateFlux(bandpass)
        if len(bandpass.wave_list) > 0:
            x = np.union1d(bandpass.wave_list, self.wave_list)
            x = x[(x <= bandpass.red_limit) & (x >= bandpass.blue_limit)]
            R = galsim.dcr.get_refraction(x, zenith_angle, **kwargs) / galsim.radians
            photons = self.fphotons(x)
            throughput = bandpass(x)
            Rbar = np.trapz(throughput * photons * R, x) / flux
            V = np.trapz(throughput * photons * (R-Rbar)**2, x) / flux
        else:
            weight = lambda w: bandpass(w) * self.fphotons(w)
            Rbar_kernel = lambda w: galsim.dcr.get_refraction(w, zenith_angle, **kwargs)
            Rbar = galsim.integ.int1d(lambda w: weight(w) * Rbar_kernel(w),
                                      bandpass.blue_limit, bandpass.red_limit)
            V_kernel = lambda w: (galsim.dcr.get_refraction(w, zenith_angle, **kwargs) - Rbar)**2
            V = galsim.integ.int1d(lambda w: weight(w) * V_kernel(w),
                                   bandpass.blue_limit, bandpass.red_limit)
        # Rbar and V are computed above assuming that the parallactic angle is 0.  Hence we
        # need to rotate our frame by the parallactic angle to get the desired output.
        rot = np.matrix([[np.cos(parallactic_angle.rad()), -np.sin(parallactic_angle.rad())],
                         [np.sin(parallactic_angle.rad()), np.cos(parallactic_angle.rad())]])
        Rbar = rot * Rbar * np.matrix([0,1]).T
        V = rot * np.matrix([[0, 0], [0, V]]) * rot.T
        return Rbar, V
