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
from astropy import units
from astropy import constants

from .gsobject import GSObject
from .table import LookupTable, _LookupTable
from . import utilities
from . import integ
from . import dcr
from .utilities import WeakMethod, lazy_property, combine_wave_list, basestring
from .errors import GalSimError, GalSimValueError, GalSimRangeError, GalSimSEDError
from .errors import GalSimIncompatibleValuesError

class SED(object):
    """Object to represent the spectral energy distributions of stars and galaxies.

    SEDs are callable, usually returning the flux density in photons/nm/cm^2/s as a function of
    wavelength, though SEDs are also used by GalSim to track dimensionless wavelength-dependent
    normalizations, and may thus also return dimensionless values.  By default, the above wavelength
    used by __call__ is nanometers, but it's possible to use other units via the astropy.units
    module (at least, if the SED keyword argument ``fast=False``, see below).  For instance,::

        >>> sed = galsim.SED(...)
        >>> from astropy import units as u
        >>> assert sed(500) == sed(5000 * u.AA)  # 500 nm == 5000 Angstroms

    The python type of the return value depends on the type of the input wavelength(s).  A scalar
    input wavelength yields a scalar flux density, a tuple yields a tuple, a list yields a list, and
    a numpy.ndarray yields a numpy.ndarray.  A scalar astropy.units.Quantity yields a python scalar,
    and a vector astropy.units.Quantity yields a numpy.ndarray.

    SEDs are immutable; all transformative SED methods return *new* SEDs, and leave their
    originating SEDs unaltered.

    SEDs have ``blue_limit`` and ``red_limit`` attributes, which indicate the range over which the
    SED is defined.  An exception will be raised if the flux density or normalization is requested
    outside of this range.  Note that ``blue_limit`` and ``red_limit`` are always in nanometers and
    in the observed frame when ``redshift != 0``.

    SEDs may be multiplied by scalars or scalar functions of wavelength.  In particular, an SED
    multiplied by a `Bandpass` will yield the appropriately filtered SED.  Two SEDs may be
    multiplied together if at least one of them represents a dimensionless normalization.

    SEDs may be added together if they are at the same redshift.  The resulting SED will only be
    defined on the wavelength region where both of the operand SEDs are defined. ``blue_limit`` and
    ``red_limit`` will be reset accordingly.

    The input parameter, ``spec``, may be one of several possible forms:

    1. a regular python function (or an object that acts like a function)
    2. a `LookupTable`
    3. a file from which a `LookupTable` can be read in
    4. a string which can be evaluated to a function of ``wave`` via ``eval('lambda wave:'+spec)``,
       e.g.::

            spec = '0.8 + 0.2 * (wave-800)'

    5. a python scalar (only possible for dimensionless SEDs)

    The argument of ``spec`` should be the wavelength in units specified by ``wave_type``, which
    should be an instance of ``astropy.units.Unit`` of equivalency class ``astropy.units.spectral``,
    or one of the case-insensitive aliases 'nm', 'nanometer', 'nanometers', 'A', 'Ang', 'Angstrom',
    or 'Angstroms'.  Note that ``astropy.units.spectral`` includes not only units with dimensions
    of length, but also frequency, energy, or wavenumber.

    The return value of ``spec`` should be a spectral density with units specified by ``flux_type``,
    which should be an instance of ``astropy.units.Unit`` of equivalency class
    ``astropy.units.spectral_density``, or one of the case-insensitive aliases:

    1. 'flambda':  erg/wave_type/cm^2/s, where wave_type is as above.
    2. 'fnu':      erg/Hz/cm^2/s
    3. 'fphotons': photons/wave_type/cm^2/s, where wave_type is as above.
    4. '1':        dimensionless

    Note that the ``astropy.units.spectral_density`` class includes units with dimensions of
    [energy/time/area/unit-wavelength], [energy/time/area/unit-frequency],
    [photons/time/area/unit-wavelength], and so on.

    Finally, the optional ``fast`` keyword option is used to specify when unit and dimension changes
    are executed, particularly for SEDs specified by a `LookupTable`.  If ``fast=True``, the
    default, then the input units/dimensions may be converted to an internal working unit before
    interpolation in wavelength is performed.  Alternatively, ``fast=False`` implies that
    interpolation should take place in the native units of the input ``spec``, and subsequently flux
    density converted to photons/cm^2/s/nm afterwards.  Generally, the former option is faster, but
    may be less accurate since interpolation and dimensionality conversion do not commute.  One
    consequence of using ``fast=True`` is that __call__ can not accept an ``astropy.units.Quantity``
    in this case.

    Parameters:
        spec:        Function defining the z=0 spectrum at each wavelength.  See above for
                     valid options for this parameter.
        wave_type:   String or astropy.unit specifying units for wavelength input to ``spec``.
        flux_type:   String or astropy.unit specifying what type of spectral density or
                     dimensionless normalization ``spec`` represents.  See above for valid options
                     for this parameter.
        redshift:    Optionally shift the spectrum to the given redshift. [default: 0]
        fast:        Convert units on initialization instead of on __call__. [default: True]
    """
    # We'll use these multiple times below, and they are ridiculously slow to construct,
    # so just make them once at the class level.
    _fphotons_base = units.astrophys.photon/(units.s * units.cm**2)
    _flambda_base = units.erg/(units.s * units.cm**2)
    _fphotons = _fphotons_base / units.nm
    _flambda = _flambda_base / units.nm
    _fnu = units.erg / (units.s * units.Hz * units.cm**2)
    _spec_nm = units.spectral_density(1*units.nm)
    _c = constants.c.to('nm/s').value
    _h = constants.h.to('erg s').value
    _dimensionless = units.dimensionless_unscaled

    def __init__(self, spec, wave_type, flux_type, redshift=0., fast=True,
                 _blue_limit=0.0, _red_limit=np.inf, _wave_list=None, _spectral=None):
        self._flux_type = flux_type  # Need to save the original for repr
        # Parse the various options for wave_type
        if isinstance(wave_type, str):
            if wave_type.lower() in ('nm', 'nanometer', 'nanometers'):
                self.wave_type = 'nm'
                self.wave_factor = 1.
            elif wave_type.lower() in ('a', 'ang', 'angstrom', 'angstroms'):
                self.wave_type = 'Angstrom'
                self.wave_factor = 10.
            else:
                raise GalSimValueError("Unknown wave_type", wave_type, ('nm', 'Angstrom'))
        else:
            self.wave_type = wave_type
            try:
                self.wave_factor = (1*units.nm).to(self.wave_type).value
                if self.wave_factor == 1.:
                    self.wave_type = 'nm'
                elif abs(self.wave_factor-10.) < 2.e-15:  # This doesn't come out exactly 10.
                    self.wave_type = 'Angstrom'
                    self.wave_factor = 10.
            except units.UnitConversionError:
                self.wave_factor = None

        # Parse the various options for flux_type
        self.flux_factor = None
        if isinstance(flux_type, str):
            if flux_type.lower() == 'flambda':
                self.flux_type = 'flambda'
                self.spectral = True
                self.flux_factor = 1. / (SED._h * SED._c)
            elif flux_type.lower() == 'fphotons':
                self.spectral = True
                if self.wave_factor is not None:
                    self.flux_type = 'fphotons'
                    self.flux_factor = self.wave_factor
                else:
                    self.flux_type = SED._fphotons
            elif flux_type.lower() == 'fnu':
                self.spectral = True
                if self.wave_factor is not None:
                    self.flux_type = 'fnu'
                    self.flux_factor = self.wave_factor / SED._h
                else:
                    self.flux_type = SED._fnu
            elif flux_type == '1':
                self.flux_type = '1'
                self.spectral = False
            else:
                raise GalSimValueError("Unknown flux_type", flux_type,
                                       ('flambda', 'fnu', 'fphotons', '1'))
        else:
            self.flux_type = flux_type
            self.spectral = self.check_spectral()
            if not self.spectral and not self.check_dimensionless():
                raise GalSimValueError(
                    "Flux_type must be equivalent to a spectral density or dimensionless.",
                    flux_type)
            try:
                if self.wave_factor and self.spectral:
                    self.flux_factor = (1*self.flux_type).to(SED._fphotons).value
                    self.flux_type = 'fphotons'
            except units.UnitConversionError:
                try:
                    self.flux_factor = (1*self.flux_type).to(SED._flambda).value
                    self.flux_factor /= SED._h * SED._c * self.wave_factor
                    self.flux_type = 'flambda'
                except units.UnitConversionError:
                    try:
                        self.flux_factor = (1*self.flux_type).to(SED._fnu).value
                        self.flux_factor *= self.wave_factor / SED._h
                        self.flux_type = 'fnu'
                    except units.UnitConversionError:
                        self.wave_type = units.Unit(self.wave_type)

        self.redshift = redshift
        self.fast = fast

        # Convert string input into a real function (possibly a LookupTable)
        self._orig_spec = spec  # Save this for pickling
        self._initialize_spec()

        # Setup the wave_list, red_limit, blue_limit
        if _wave_list is not None:
            self.wave_list = _wave_list
            self.blue_limit = float(_blue_limit)
            self.red_limit = float(_red_limit)
        elif isinstance(self._spec, LookupTable):
            self.wave_list = np.array(self._spec.getArgs())
            if self.wave_factor:
                self.wave_list *= (1.0 + self.redshift) / self.wave_factor
            else:
                self.wave_list = (self.wave_list*self.wave_type).to(units.nm, units.spectral()).value
                self.wave_list *= (1.0 + self.redshift)
            self.blue_limit = float(np.min(self.wave_list))
            self.red_limit = float(np.max(self.wave_list))
        else:
            self.blue_limit = 0.0
            self.red_limit = np.inf
            self.wave_list = np.array([], dtype=float)

        # Define the appropriate functions to call
        self._setup_funcs()

    def _setup_funcs(self):
        # Set up the various functions we use to do the right calculation based on which
        # wave type and/or flux type we have for _spec.
        # The astropy unit functions are horribly slow, so we want to avoid them as much as
        # possible.  If the wave_type and flux_type are one of the simpler (and most common)
        # types, then we have custom functions that do the necessary conversions directly.
        if self.wave_factor == 1:
            self._get_native_waves = WeakMethod(self._get_native_waves_trivial)
        elif self.wave_factor:
            self._get_native_waves = WeakMethod(self._get_native_waves_fast)
        else:
            self._get_native_waves = WeakMethod(self._get_native_waves_slow)

        if self.redshift == 0.:
            self._get_rest_native_waves = self._get_native_waves
        elif self.wave_factor:
            self._get_rest_native_waves = WeakMethod(self._get_rest_native_waves_fast)
        else:
            self._get_rest_native_waves = WeakMethod(self._get_rest_native_waves_slow)

        if self.flux_type == 'fphotons':
            #assert self.flux_factor is not None
            self._flux_to_photons = WeakMethod(self._flux_to_photons_fphot)
        elif self.flux_type == 'flambda':
            #assert self.flux_factor is not None
            self._flux_to_photons = WeakMethod(self._flux_to_photons_flam)
        elif self.flux_type == 'fnu':
            #assert self.flux_factor is not None
            self._flux_to_photons = WeakMethod(self._flux_to_photons_fnu)
        else:
            self._flux_to_photons = WeakMethod(self._flux_to_photons_slow)

        if self.fast:
            self._call = WeakMethod(self._call_fast)
        else:
            self._call = WeakMethod(self._call_slow)

    # Here are the definitions for the various functions we can use depending on the wave_type
    # and flux_type (cf. _setup_funcs).
    def _get_native_waves_trivial(self, wave):
        return wave

    def _get_native_waves_fast(self, wave):
        return np.asarray(wave) * self.wave_factor

    def _get_native_waves_slow(self, wave):
        return (wave * units.nm).to(self.wave_type, units.spectral()).value

    def _get_rest_native_waves_fast(self, wave):
        return np.asarray(wave) * (self.wave_factor / (1.0+self.redshift))

    def _get_rest_native_waves_slow(self, wave):
        return (wave / (1.0+self.redshift) * units.nm).to(self.wave_type, units.spectral()).value

    def _flux_to_photons_fphot(self, flux_native, wave_native):
        return flux_native * self.flux_factor

    def _flux_to_photons_flam(self, flux_native, wave_native):
        return flux_native * wave_native * self.flux_factor

    def _flux_to_photons_fnu(self, flux_native, wave_native):
        return flux_native / wave_native * self.flux_factor

    def _flux_to_photons_slow(self, flux_native, wave_native):
        return (flux_native * self.flux_type).to(
                SED._fphotons, units.spectral_density(wave_native * self.wave_type)).value


    def _initialize_spec(self):
        # Turn the input _orig_spec into a real function _spec.
        # The function cannot be pickled, so will need to do this in getstate as well as init.

        self._const = False
        if isinstance(self._orig_spec, (int, float)):
            if not self.dimensionless:
                raise GalSimSEDError("Attempt to set spectral SED using float or integer.", self)
            self._const = True
            self._spec = lambda w: float(self._orig_spec)
        elif isinstance(self._orig_spec, basestring):
            isfile, filename = utilities.check_share_file(self._orig_spec, 'SEDs')
            if isfile:
                self._spec = LookupTable.from_file(filename, interpolant='linear')
            else:
                # Don't catch ArithmeticErrors when testing to see if the the result of `eval()`
                # is valid since `spec = '1./(wave-700)'` will generate a ZeroDivisionError (which
                # is a subclass of ArithmeticError) despite being a valid spectrum specification,
                # while `spec = 'blah'` where `blah` is undefined generates a NameError and is not
                # a valid spectrum specification.
                # Are there any other types of errors we should trap here?
                try:
                    self._spec = utilities.math_eval('lambda wave : ' + self._orig_spec)
                    test_value = self._spec(700.0)
                except ArithmeticError:
                    test_value = 0
                except Exception as e:
                    raise GalSimValueError(
                        "String spec must either be a valid filename or something that "
                        "can eval to a function of wave.\n"
                        "Caught error: {0}".format(e), self._orig_spec)
                from numbers import Real
                if not isinstance(test_value, Real):
                    raise GalSimValueError("The given SED function did not return a valid number "
                                           "at test wavelength %s: got %s"%(700.0, test_value),
                                           self._orig_spec)

        else:
            self._spec = self._orig_spec

    def check_spectral(self):
        """Return boolean indicating if SED has units compatible with a spectral density."""
        return self.flux_type.is_equivalent(SED._fphotons, SED._spec_nm)

    def check_dimensionless(self):
        """Return boolean indicating if SED is dimensionless."""
        if self.flux_type.is_equivalent(SED._dimensionless):
            self._flux_type = '1'
            # The astropy.units.dimensionless_unscaled object isn't properly reprable.
            # So switch to using '1' in these cases.
            return True
        else:
            return False

    @property
    def dimensionless(self):  # for convenience
        """Whether the object is dimensionless (rather than spectral).
        """
        return not self.spectral

    def _rest_nm_to_photons(self, wave):
        wave_native = self._get_native_waves(wave)
        flux_native = self._spec(wave_native)
        return self._flux_to_photons(flux_native, wave_native)

    def _rest_nm_to_dimensionless(self, wave):
        return self._spec(self._get_native_waves(wave))

    def _check_bounds(self, wave):
        if hasattr(wave, '__iter__'):
            wmin = np.min(wave)
            wmax = np.max(wave)
        else:
            wmin = wmax = wave

        extrapolation_slop = 1.e-6 # allow a small amount of extrapolation
        if wmin < self.blue_limit - extrapolation_slop:
            raise GalSimRangeError("Requested wavelength is bluer than blue_limit.",
                                   wave, self.blue_limit, self.red_limit)
        if wmax > self.red_limit + extrapolation_slop:
            raise GalSimRangeError("Requested wavelength is redder than red_limit.",
                                   wave, self.blue_limit, self.red_limit)

    @lazy_property
    def _fast_spec(self):
        # Create a fast version of self._spec by constructing a LookupTable on self.wave_list
        if self.wave_factor == 1. and self.flux_factor == 1.:
            return self._spec
        else:
            if len(self.wave_list) == 0:
                if self.spectral:
                    return WeakMethod(self._rest_nm_to_photons)
                else:
                    return WeakMethod(self._rest_nm_to_dimensionless)
            else:
                x = self.wave_list / (1.0 + self.redshift)
                if self.spectral:
                    f = self._rest_nm_to_photons(x)
                else:
                    f = self._rest_nm_to_dimensionless(x)
                return _LookupTable(x, f, interpolant='linear')

    def _call_fast(self, wave):
        """Return either flux in photons / sec / cm^2 / nm, or dimensionless normalization.

        Assumes that self._spec has already been transformed to accept correct wavelength units and
        yield correct flux units.

        Parameters:
            wave:   Wavelength in nanometers.

        Returns:
            Flux or normalization.
        """
        self._check_bounds(wave)
        return self._fast_spec(np.asarray(wave) / (1.0 + self.redshift))

    def _call_slow(self, wave):
        """Return flux in photons / sec / cm^2 / nm or dimensionless normalization.

        Uses self._spec that has not been pre-transformed for desired units, instead does all unit
        conversions inside this method.

        Parameters:
            wave:   Wavelength.  If not an astropy.units.Quantity, then assumed units are
                    nanometers.

        Returns:
            Flux.
        """
        wave_in = wave
        # Convert wave to nanometers if needed.
        if isinstance(wave, units.Quantity):
            wave = wave.to(units.nm, units.spectral()).value

        self._check_bounds(wave)

        # Figure out rest-frame wave_type wavelength array for query to self._spec.
        rest_wave_native = self._get_rest_native_waves(wave)

        out = self._spec(rest_wave_native)

        # Manipulate output units
        if self.spectral:
            out = self._flux_to_photons(out, rest_wave_native)
        return out

    def __call__(self, wave):
        """Return photon flux density or dimensionless normalization at wavelength ``wave``.

        Note that outside of the wavelength range defined by the ``blue_limit`` and ``red_limit``
        attributes, the SED is considered undefined, and this method will raise an exception if a
        wavelength outside the defined range is passed as an argument.

        Parameters:
            wave:   Wavelength in nanometers at which to evaluate the SED. May be a scalar,
                    a numpy.array, or an astropy.units.Quantity

        Returns:
            photon flux density in units of photons/nm/cm^2/s if self.spectral, or
            dimensionless normalization if self.dimensionless.
        """
        return self._call(wave)

    def _mul_sed(self, other):
        """Equivalent to self * other when other is an SED, but no sanity checks."""
        # There should only be one SED with a non-trivial redshift, so adding them
        # should always give us the right net redshift to use.
        redshift = self.redshift + other.redshift

        fast = self.fast and other.fast

        wave_list, blue_limit, red_limit = combine_wave_list(self, other)
        if fast:
            zfactor1 = (1.+redshift) / (1.+self.redshift)
            zfactor2 = (1.+redshift) / (1.+other.redshift)
            spec = lambda w: self._fast_spec(w * zfactor1) * other._fast_spec(w * zfactor2)
        else:
            spec = lambda w: self(w * (1.+redshift)) * other(w * (1.+redshift))
        _spectral = self.spectral or other.spectral
        return SED(spec, 'nm', 'fphotons', redshift=redshift, fast=fast,
                   _blue_limit=blue_limit, _red_limit=red_limit, _wave_list=wave_list,
                   _spectral=_spectral)

    def _mul_bandpass(self, other):
        """Equivalent to self * other when other is a Bandpass"""
        wave_list, blue_limit, red_limit = combine_wave_list(self, other)
        zfactor = (1.0+self.redshift) / other.wave_factor
        if self.fast:
            if (isinstance(self._fast_spec, LookupTable)
                and not self._fast_spec.x_log
                and not self._fast_spec.f_log
                and self._fast_spec.interpolant == 'linear'):
                x = wave_list / (1.0 + self.redshift)
                f = self._fast_spec(x) * other._tp(x*zfactor)
                spec = _LookupTable(x, f, 'linear')
            else:
                spec = lambda w: self._fast_spec(w) * other._tp(w*zfactor)
        else:
            spec = lambda w: self(w*(1.0+self.redshift)) * other._tp(w*zfactor)
        return SED(spec, 'nm', 'fphotons', redshift=self.redshift, fast=self.fast,
                   _blue_limit=blue_limit, _red_limit=red_limit, _wave_list=wave_list,
                   _spectral=self.spectral)


    def _mul_scalar(self, other):
        """Equivalent to self * other when other is a scalar"""
        # If other is a scalar and self._spec a LookupTable, then remake that LookupTable.
        if isinstance(self._spec, LookupTable):
            wave_type = self.wave_type
            flux_type = self.flux_type
            x = self._spec.getArgs()
            f = np.array(self._spec.getVals()) * other
            spec = _LookupTable(x, f, x_log=self._spec.x_log, f_log=self._spec.f_log,
                                interpolant=self._spec.interpolant)
        elif self._const:
            spec = self._spec(42.0) * other
            wave_type = 'nm'
            flux_type = '1'
        else:
            wave_type = 'nm'
            flux_type = 'fphotons' if self.spectral else '1'
            if self.fast:
                spec = lambda w: self._fast_spec(w) * other
            else:
                spec = lambda w: self(w*(1.0+self.redshift)) * other
        return SED(spec, wave_type, flux_type, redshift=self.redshift, fast=self.fast,
                   _blue_limit=self.blue_limit, _red_limit=self.red_limit,
                   _wave_list=self.wave_list,
                   _spectral=self.spectral)


    def __mul__(self, other):
        """Multiply the SED by something.

        There are several possibilities:

        1. SED * SED -> SED (at least one must be dimensionless)
        2. SED * GSObject -> ChromaticObject
        3. SED * Bandpass -> SED (treating throughput similarly to dimensionless SED)
        4. SED * callable function -> SED (treating function as dimensionless SED)
        5. SED * scalar -> SED
        """
        from .transform import Transform
        from .bandpass import Bandpass
        # Watch out for 5 types of `other`:
        # 1.  SED: Check that not both spectral densities.
        # 2.  GSObject: return a ChromaticObject().
        # 3.  Bandpass: return an SED, but carefully propagate blue/red limit and wave_list.
        # 4.  Callable: return an SED
        # 5.  Scalar: return an SED
        #
        # Additionally, check for shortcuts when self._const

        # Product of two SEDs
        if isinstance(other, SED):
            if self.spectral and other.spectral:
                raise GalSimIncompatibleValuesError(
                    "Cannot multiply two spectral densities together.", self_sed=self, other=other)

            if other._const:
                return self._mul_scalar(other._spec(42.0))  # const, so can eval anywhere.
            elif self._const:
                return other._mul_scalar(self._spec(42.0))
            else:
                return self._mul_sed(other)

        # Product of SED and achromatic GSObject is a `ChromaticTransformation`.
        elif isinstance(other, GSObject):
            return Transform(other, flux_ratio=self)

        # Product of SED and Bandpass is (filtered) SED.  The `redshift` attribute is retained.
        elif isinstance(other, Bandpass):
            return self._mul_bandpass(other)

        # Product of SED with generic callable is also a (filtered) SED, with retained `redshift`.
        elif hasattr(other, '__call__'):
            if self.fast:
                spec = lambda w: self._fast_spec(w) * other(w*(1.0+self.redshift))
            else:
                spec = lambda w: self(w*(1.0+self.redshift)) * other(w*(1.0+self.redshift))
            flux_type = 'fphotons' if self.spectral else '1'
            return SED(spec, 'nm', flux_type, redshift=self.redshift, fast=self.fast,
                       _blue_limit=self.blue_limit, _red_limit=self.red_limit,
                       _wave_list=self.wave_list,
                       _spectral=self.spectral)

        elif isinstance(other, (int, float)):
            return self._mul_scalar(other)

        else:
            raise TypeError("Cannot multiply an SED by %s"%(other))

    def __rmul__(self, other):
        return self*other

    def __div__(self, other):
        # Enable division by scalars or dimensionless callables (including dimensionless SEDs.)
        if isinstance(other, SED) and other.spectral:
            raise GalSimSEDError("Cannot divide by spectral SED.", other)
        if hasattr(other, '__call__'):
            spec = lambda w: self(w * (1.0 + self.redshift)) / other(w * (1.0 + self.redshift))
        elif isinstance(self._spec, LookupTable):
            # If other is not a function, then there is no loss of accuracy by applying the
            # factor directly to the LookupTable, if that's what we are using.
            # Make sure to keep the same properties about the table, flux_type, wave_type.
            x = self._spec.getArgs()
            f = [ val / other for val in self._spec.getVals() ]
            spec = _LookupTable(x, f, x_log=self._spec.x_log, f_log=self._spec.f_log,
                                interpolant=self._spec.interpolant)
        else:
            spec = lambda w: self(w * (1.0 + self.redshift)) / other

        return SED(spec, flux_type=self.flux_type, wave_type=self.wave_type,
                   redshift=self.redshift, fast=self.fast,
                   _wave_list=self.wave_list,
                   _blue_limit=self.blue_limit, _red_limit=self.red_limit)

    __truediv__ = __div__

    def __add__(self, other):
        # Add together two SEDs, with the following caveats:
        # 1) The SEDs must have the same redshift.
        # 2) The resulting SED will be defined on the wavelength range set by the overlap of the
        #    wavelength ranges of the two SED operands.
        # 3) The new `wave_list` will be the union of the operand `wave_list`s in the intersecting
        #    region, even if one or both of the `wave_list`s are empty.
        # These conditions ensure that SED addition is commutative.

        if self.redshift != other.redshift:
            raise GalSimIncompatibleValuesError(
                "Can only add SEDs with same redshift.", self_sed=self, other=other)

        if self.dimensionless and other.dimensionless:
            flux_type = '1'
            _spectral = False
        elif self.spectral and other.spectral:
            flux_type = 'fphotons'
            _spectral = True
        else:
            raise GalSimIncompatibleValuesError(
                "Cannot add SEDs with incompatible dimensions.", self_sed=self, other=other)

        wave_list, blue_limit, red_limit = combine_wave_list(self, other)

        # If both SEDs are `fast`, and both `_fast_spec`s are LookupTables, then make a new
        # LookupTable instead and preserve picklability.
        # First need to make sure self._fast_spec and other._fast_spec are initialized.  Can do this
        # by evaluating them at a good wavelength.  blue_limit should work.
        self(blue_limit)
        other(blue_limit)
        if (self.fast
                and other.fast
                and isinstance(self._fast_spec, LookupTable)
                and isinstance(other._fast_spec, LookupTable)
                and not self._fast_spec.x_log
                and not other._fast_spec.x_log
                and not self._fast_spec.f_log
                and not other._fast_spec.f_log
                and self._fast_spec.interpolant == 'linear'
                and other._fast_spec.interpolant == 'linear'):
            x = wave_list / (1.0 + self.redshift)
            f = self._fast_spec(x) + other._fast_spec(x)
            spec = _LookupTable(x, f, interpolant='linear')
        else:
            spec = lambda w: self(w*(1.0+self.redshift)) + other(w*(1.0+self.redshift))

        return SED(spec, wave_type='nm', flux_type=flux_type,
                   redshift=self.redshift, fast=self.fast, _wave_list=wave_list,
                   _blue_limit=blue_limit, _red_limit=red_limit,
                   _spectral=_spectral)

    def __sub__(self, other):
        # Subtract two SEDs, with the same caveats as adding two SEDs.
        return self.__add__(-1.0 * other)

    def withFluxDensity(self, target_flux_density, wavelength):
        """Return a new `SED` with flux density set to ``target_flux_density`` at wavelength
        ``wavelength``.

        See `ChromaticObject` docstring for information about how `SED` normalization affects
        `ChromaticObject` normalization.

        Parameters:
            target_flux_density:    The target normalization in photons/nm/cm^2/s.
            wavelength:             The wavelength, in nm, at which the flux density will be set.

        Returns:
            the new normalized SED.
        """
        if self.dimensionless:
            raise GalSimSEDError("Cannot set flux density of dimensionless SED.", self)
        if isinstance(wavelength, units.Quantity):
            wavelength_nm = wavelength.to(units.nm, units.spectral())
            current_flux_density = self._call(wavelength_nm.value)
        else:
            wavelength_nm = wavelength * units.nm
            current_flux_density = self._call(wavelength)
        if isinstance(target_flux_density, units.Quantity):
            target_flux_density = target_flux_density.to(
                    SED._fphotons, units.spectral_density(wavelength_nm)).value
        factor = target_flux_density / current_flux_density
        return self * factor

    def withFlux(self, target_flux, bandpass):
        """Return a new `SED` with flux through the `Bandpass` ``bandpass`` set to ``target_flux``.

        See `ChromaticObject` docstring for information about how `SED` normalization affects
        `ChromaticObject` normalization.

        Parameters:
            target_flux:    The desired flux normalization of the SED.
            bandpass:       A `Bandpass` object defining a filter bandpass.

        Returns:
            the new normalized `SED`.
        """
        current_flux = self.calculateFlux(bandpass)
        norm = target_flux/current_flux
        return self * norm

    def withMagnitude(self, target_magnitude, bandpass):
        """Return a new `SED` with magnitude through the `Bandpass` ``bandpass`` set to
        ``target_magnitude``.

        Note that this requires ``bandpass`` to have been assigned a zeropoint using
        `Bandpass.withZeropoint`.  See `ChromaticObject` docstring for information about how `SED`
        normalization affects `ChromaticObject` normalization.

        Parameters:
            target_magnitude:   The desired magnitude of the `SED`.
            bandpass:           A `Bandpass` object defining a filter bandpass.

        Returns:
            the new normalized `SED`.
        """
        if bandpass.zeropoint is None:
            raise GalSimError("Cannot call SED.withMagnitude on this bandpass, because it does "
                              "not have a zeropoint.  See Bandpass.withZeropoint()")
        current_magnitude = self.calculateMagnitude(bandpass)
        norm = 10**(-0.4*(target_magnitude - current_magnitude))
        return self * norm

    def atRedshift(self, redshift):
        """Return a new `SED` with redshifted wavelengths.

        Parameters:
            redshift:   The redshift for the returned `SED`

        Returns:
            the redshifted `SED`.
        """
        if redshift == self.redshift:
            return self
        if redshift <= -1:
            raise GalSimRangeError("Invalid redshift", redshift, -1.)
        zfactor = (1.0 + redshift) / (1.0 + self.redshift)
        wave_list = self.wave_list * zfactor
        blue_limit = self.blue_limit * zfactor
        red_limit = self.red_limit * zfactor

        return SED(self._spec, self.wave_type, self.flux_type, redshift, self.fast,
                   _wave_list=wave_list, _blue_limit=blue_limit, _red_limit=red_limit)

    def calculateFlux(self, bandpass):
        """Return the flux (photons/cm^2/s) of the `SED` through the `Bandpass` bandpass.

        Parameters:
            bandpass:   A `Bandpass` object representing a filter, or None to compute the
                        bolometric flux.  For the bolometric flux the integration limits will be
                        set to (0, infinity), which implies that the `SED` needs to be evaluable
                        over this entire range.

        Returns:
            the flux through the bandpass.
        """
        from . import integ
        if self.dimensionless:
            raise GalSimSEDError("Cannot calculate flux of dimensionless SED.", self)
        if len(bandpass.wave_list) > 0 or len(self.wave_list) > 0:
            slop = 1e-6 # nm
            if (self.blue_limit > bandpass.blue_limit + slop
                    or self.red_limit < bandpass.red_limit - slop):
                raise GalSimRangeError("Bandpass is not completely within defined wavelength "
                                       "range for this SED.",
                                       (bandpass.blue_limit, bandpass.red_limit),
                                       self.blue_limit, self.red_limit)
            wmin = max(self.blue_limit, bandpass.blue_limit)
            wmax = min(self.red_limit, bandpass.red_limit)
            if self.fast and isinstance(self._fast_spec, LookupTable):
                wf = 1./(1.+self.redshift) / bandpass.wave_factor
                ff = 1./bandpass.wave_factor
                wmin *= bandpass.wave_factor
                wmax *= bandpass.wave_factor
                return self._fast_spec.integrate_product(bandpass._tp, wmin, wmax, wf) * ff
            else:
                w, _, _ = combine_wave_list(self, bandpass)
                if not self.fast and self.flux_type != 'fphotons':
                    # When not fast, the SED definition is not linear between the wave_list
                    # points, so this can be slightly inaccurate if the waves are too far apart.
                    # Add in 100 uniformly spaced points to achieve relative accurace ~few e-6.
                    w = np.union1d(w, np.linspace(w[0], w[-1], 100))
                return _LookupTable(w,bandpass(w),'linear').integrate_product(self)
        else:
            return integ.int1d(lambda w: bandpass(w)*self(w),
                               bandpass.blue_limit, bandpass.red_limit)

    def calculateMagnitude(self, bandpass):
        """Return the `SED` magnitude through a `Bandpass` ``bandpass``.

        Note that this requires ``bandpass`` to have been assigned a zeropoint using
        `Bandpass.withZeropoint`.

        Parameters:
            bandpass:     A `Bandpass` object representing a filter, or None to compute the
                          bolometric magnitude.  For the bolometric magnitude the integration
                          limits will be set to (0, infinity), which implies that the `SED` needs
                          to be evaluable over this entire range.

        Returns:
            the bandpass magnitude.
        """
        if self.dimensionless:
            raise GalSimSEDError("Cannot calculate magnitude of dimensionless SED.", self)
        if bandpass.zeropoint is None:
            raise GalSimError("Cannot do this calculation for a bandpass without an assigned "
                              "zeropoint")
        flux = self.calculateFlux(bandpass)
        return -2.5 * np.log10(flux) + bandpass.zeropoint

    def thin(self, rel_err=1.e-4, trim_zeros=True, preserve_range=True, fast_search=True):
        """Remove some tabulated values while keeping the integral over the set of tabulated values
        still accurate to ``rel_err``.

        This is only relevant if the `SED` was initialized with a `LookupTable` or from a file
        (which internally creates a `LookupTable`).

        Parameters:
            rel_err:          The relative error allowed in the integral over the `SED`
                              [default: 1.e-4]
            trim_zeros:       Remove redundant leading and trailing points where f=0?  (The last
                              leading point with f=0 and the first trailing point with f=0 will
                              be retained).  Note that if both trim_leading_zeros and
                              preserve_range are True, then the only the range of ``x`` *after*
                              zero trimming is preserved.  [default: True]
            preserve_range:   Should the original range (``blue_limit`` and ``red_limit``) of the
                              `SED` be preserved? (True) Or should the ends be trimmed to
                              include only the region where the integral is significant? (False)
                              [default: True]
            fast_search:      If set to True, then the underlying algorithm will use a
                              relatively fast O(N) algorithm to select points to include in the
                              thinned approximation.  If set to False, then a slower O(N^2)
                              algorithm will be used.  We have found that the slower algorithm
                              tends to yield a thinned representation that retains fewer samples
                              while still meeting the relative error requirement.
                              [default: True]

        Returns:
            the thinned `SED`.
        """
        if len(self.wave_list) > 0:
            rest_wave_native = self._get_rest_native_waves(self.wave_list)
            spec_native = self._spec(rest_wave_native)

            # Note that this is thinning in native units, not nm and photons/nm.
            newx, newf = utilities.thin_tabulated_values(
                    rest_wave_native, spec_native, rel_err=rel_err,
                    trim_zeros=trim_zeros, preserve_range=preserve_range, fast_search=fast_search)

            newspec = _LookupTable(newx, newf, interpolant='linear')
            return SED(newspec, self.wave_type, self.flux_type, redshift=self.redshift,
                       fast=self.fast)
        else:
            return self

    def calculateDCRMomentShifts(self, bandpass, **kwargs):
        """Calculates shifts in first and second moments of PSF due to differential chromatic
        refraction (DCR).

        I.e., equations (1) and (2) from Plazas and Bernstein (2012):

        http://arxiv.org/abs/1204.1346).

        Parameters:
            bandpass:           `Bandpass` through which object is being imaged.
            zenith_angle:       `Angle` from object to zenith
            parallactic_angle:  Parallactic angle, i.e. the position angle of the zenith,
                                measured from North through East.  [default: 0]
            obj_coord:          Celestial coordinates of the object being drawn as a
                                `CelestialCoord`. [default: None]
            zenith_coord:       Celestial coordinates of the zenith as a `CelestialCoord`.
                                [default: None]
            HA:                 Hour angle of the object as an `Angle`. [default: None]
            latitude:           Latitude of the observer as an `Angle`. [default: None]
            pressure:           Air pressure in kiloPascals.  [default: 69.328 kPa]
            temperature:        Temperature in Kelvins.  [default: 293.15 K]
            H2O_pressure:       Water vapor pressure in kiloPascals.  [default: 1.067 kPa]

        Returns:
            a tuple:

            - The first element is the vector of DCR first moment shifts
            - The second element is the 2x2 matrix of DCR second (central) moment shifts.
        """
        from .dcr import parse_dcr_angles
        if self.dimensionless:
            raise GalSimSEDError("Cannot calculate DCR shifts of dimensionless SED.", self)

        zenith_angle, parallactic_angle, kwargs = parse_dcr_angles(**kwargs)

        # Any remaining kwargs will get forwarded to galsim.dcr.get_refraction
        # Check that they're valid
        for kw in kwargs:
            if kw not in ('temperature', 'pressure', 'H2O_pressure'):
                raise (TypeError("Got unexpected keyword in calculateDCRMomentShifts: {0}"
                                 .format(kw)))

        # Now actually start calculating things.
        flux = self.calculateFlux(bandpass)
        if len(self.wave_list) > 0 or len(bandpass.wave_list) > 0:
            w, _, _ = combine_wave_list(self, bandpass)
            bp = _LookupTable(w,bandpass(w),'linear')
            R = lambda w: dcr.get_refraction(w, zenith_angle, **kwargs)
            Rbar = bp.integrate_product(lambda w: self(w) * R(w)) / flux
            V = bp.integrate_product(lambda w: self(w) * (R(w)-Rbar)**2) / flux
        else:
            weight = lambda w: bandpass(w) * self(w)
            Rbar_kernel = lambda w: dcr.get_refraction(w, zenith_angle, **kwargs)
            Rbar = integ.int1d(lambda w: weight(w) * Rbar_kernel(w),
                               bandpass.blue_limit, bandpass.red_limit)
            Rbar /= flux
            V_kernel = lambda w: (dcr.get_refraction(w, zenith_angle, **kwargs) - Rbar)**2
            V = integ.int1d(lambda w: weight(w) * V_kernel(w),
                            bandpass.blue_limit, bandpass.red_limit)
            V /= flux
        # Rbar and V are computed above assuming that the parallactic angle is 0.  Hence we
        # need to rotate our frame by the parallactic angle to get the desired output.
        sinp, cosp = parallactic_angle.sincos()
        rot = np.array([[cosp, -sinp], [sinp, cosp]])
        Rbar = Rbar * rot.dot(np.array([0,1]))
        V = rot.dot(np.array([[0, 0], [0, V]])).dot(rot.T)
        return Rbar, V

    def calculateSeeingMomentRatio(self, bandpass, alpha=-0.2, base_wavelength=500):
        """Calculates the relative size of a PSF compared to the monochromatic PSF size at
        wavelength ``base_wavelength``.

        Parameters:
            bandpass:           `Bandpass` through which object is being imaged.
            alpha:              Power law index for wavelength-dependent seeing.  [default:
                                -0.2, the prediction for Kolmogorov turbulence]
            base_wavelength:    Reference wavelength in nm from which to compute the relative
                                PSF size.  [default: 500]

        Returns:
            the ratio of the PSF second moments to the second moments of the reference PSF.
        """
        if self.dimensionless:
            raise GalSimSEDError("Cannot calculate seeing moment ratio of dimensionless SED.", self)
        flux = self.calculateFlux(bandpass)
        if len(self.wave_list) > 0 or len(bandpass.wave_list) > 0:
            # With three things multiplied together, we can't rely on integrate_product
            # being completely accurate if the waves are spaced too far apart, especially with
            # a power law being one of the factors.
            # So make sure to include a uniform density of points along with the native sed and
            # bandpass points. The error goes like dx**3, so 100 points should give relative
            # errors of order ~few e-6.
            w, _, _ = combine_wave_list([self, bandpass])
            w = np.union1d(w, np.linspace(w[0], w[-1], 100))
            bp = _LookupTable(w,bandpass(w),'linear')
            return bp.integrate_product(lambda w: self(w) * (w/base_wavelength)**(2*alpha)) / flux
        else:
            weight = lambda w: bandpass(w) * self(w)
            kernel = lambda w: (w/base_wavelength)**(2*alpha)
            return integ.int1d(lambda w: weight(w) * kernel(w),
                               bandpass.blue_limit, bandpass.red_limit) / flux

    @lazy_property
    def _cache_deviate(self):
        return dict()

    def sampleWavelength(self, nphotons, bandpass, rng=None, npoints=None):
        """Sample a number of random wavelength values from the `SED`, possibly as observed through
        a `Bandpass` bandpass.

        Parameters:
            nphotons:    Number of samples (photons) to randomly draw.
            bandpass:    A `Bandpass` object representing a filter, or None to sample over the full
                         `SED` wavelength range.
            rng:         If provided, a random number generator that is any kind of `BaseDeviate`
                         object. If ``rng`` is None, one will be automatically created from the
                         system. [default: None]
            npoints:     Number of points `DistDeviate` should use for its internal interpolation
                         tables. [default: None, which uses the `DistDeviate` default]
        """
        from .random import DistDeviate
        nphotons=int(nphotons)

        key = (bandpass,npoints)
        if key in self._cache_deviate:
            dev = self._cache_deviate[key]
        else:
            if bandpass is None:
                sed = self
            else:
                sed = self._mul_bandpass(bandpass)

            if isinstance(sed._fast_spec, LookupTable):
                dev = DistDeviate(function=sed._fast_spec, npoints=npoints)
            else:
                xmin = sed.blue_limit / (1.+self.redshift)
                xmax = sed.red_limit / (1.+self.redshift)
                dev = DistDeviate(function=sed._fast_spec, x_min=xmin, x_max=xmax,
                                  npoints=npoints)
            self._cache_deviate[key] = dev

        # Reset the deviate explicitly
        if rng is not None: dev.reset(rng)

        ret = np.empty(nphotons)
        dev.generate(ret)

        if self.redshift != 0:
            ret *= (1. + self.redshift)
            # Rarely, with the redshift round trip, this can produce wavelengths < blue_limit.
            # If this happens, set those values equal to blue_limit.
            # I'm not sure if the red limit overrun can happen (we didn't see any in the use case
            # that noticed the blue overruns), but it seems prudent to also correct any of these
            # that may occur too.  Plus it's not noticeably slower using clip to do both at once.
            if bandpass is not None:
                np.clip(ret, bandpass.blue_limit, bandpass.red_limit, out=ret)

        return ret

    def __eq__(self, other):
        return (self is other or
                (isinstance(other, SED) and
                 self._orig_spec == other._orig_spec and
                 self.fast == other.fast and
                 self.wave_type == other.wave_type and
                 self.flux_type == other.flux_type and
                 self.redshift == other.redshift and
                 self.red_limit == other.red_limit and
                 self.blue_limit == other.blue_limit and
                 np.array_equal(self.wave_list,other.wave_list)))

    def __ne__(self, other): return not self.__eq__(other)

    def __hash__(self):
        # Cache this in case self._orig_spec or self.wave_list is long.
        if not hasattr(self, '_hash'):
            self._hash = hash(("galsim.SED", self._orig_spec, self.wave_type, self.flux_type,
                               self.redshift, self.fast, self.blue_limit, self.red_limit,
                               tuple(self.wave_list)))
        return self._hash

    def __repr__(self):
        outstr = ('galsim.SED(%r, wave_type=%r, flux_type=%r, redshift=%r, fast=%r,'
                  ' _wave_list=%r, _blue_limit=%r, _red_limit=%s)')%(
                      self._orig_spec, self.wave_type, self._flux_type, self.redshift, self.fast,
                      self.wave_list, self.blue_limit,
                      "float('inf')" if self.red_limit == np.inf else repr(self.red_limit))
        return outstr

    def __str__(self):
        orig_spec = repr(self._orig_spec)
        if len(orig_spec) > 80:
            orig_spec = str(self._orig_spec)
        return 'galsim.SED(%s, redshift=%s)'%(orig_spec, self.redshift)

    def __getstate__(self):
        d = self.__dict__.copy()
        if not isinstance(d['_spec'], LookupTable):
            del d['_spec']
        d.pop('_fast_spec',None)
        del d['_call']
        del d['_get_native_waves']
        del d['_get_rest_native_waves']
        del d['_flux_to_photons']
        return d

    def __setstate__(self, d):
        self.__dict__ = d
        if '_spec' not in d:
            self._initialize_spec()
        self._setup_funcs()
