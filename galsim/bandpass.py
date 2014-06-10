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
"""@file bandpass.py
Very simple implementation of a filter bandpass.  Used by galsim.chromatic.
"""

import numpy as np

import galsim
import utilities

class Bandpass(object):
    """Simple bandpass object, which models the transmission fraction of incident light as a
    function of wavelength, for either an entire optical path (e.g., atmosphere, reflecting and
    refracting optics, filters, CCD quantum efficiency), or some intermediate piece thereof.
    Bandpasses representing individual components may be combined through the `*` operator to form
    a new Bandpass object representing the composite optical path.

    Bandpasses are callable, returning dimensionless throughput as a function of wavelength in nm.

    Bandpasses are immutable; all transformative methods return *new* Bandpasses, and leave their
    originating Bandpasses unaltered.

    Bandpasses require `blue_limit` and `red_limit` attributes, which may either be explicitly set
    at initialization, or are inferred from the initializing galsim.LookupTable or 2-column file.

    Outside of the wavelength interval between `blue_limit` and `red_limit`, the throughput is
    returned as zero, regardless of the `throughput` input parameter.

    Bandpasses may be multiplied by other Bandpasses, functions, or scalars.

    You can calculate the effective wavelength of a Bandpass using the method
    getEffectiveWavelength().  We use throughput-weighted average wavelength (which is independent
    of any SED) as our definition for effective wavelength.

    For Bandpasses defined using a LookupTable, a numpy.array of wavelengths, `wave_list`, defining
    the table is maintained.  Bandpasses defined as products of two other Bandpasses will define
    their `wave_list` as the union of multiplicand `wave_list`s, although limited to the range
    between the new product `blue_limit` and `red_limit`.  (This implementation detail may affect
    the choice of integrator used to draw ChromaticObjects.)

    The input parameter, throughput, may be one of several possible forms:
    1. a regular python function (or an object that acts like a function)
    2. a galsim.LookupTable
    3. a file from which a LookupTable can be read in
    4. a string which can be evaluated into a function of `wave`
       via `eval('lambda wave : '+throughput)`
       e.g. throughput = '0.8 + 0.2 * (wave-800)'

    The argument of `throughput` will be the wavelength in either nanometers (default) or
    Angstroms depending on the value of `wave_type`.  The output should be the dimensionless
    throughput at that wavelength.  (Note we use `wave` rather than `lambda`, since `lambda` is a
    python reserved word.)

    The argument `wave_type` specifies the units to assume for wavelength and must be one of
    'nm', 'nanometer', 'nanometers', 'A', 'Ang', 'Angstrom', or 'Angstroms'. Text case here
    is unimportant.  If these wavelength options are insufficient, please submit an issue to
    the GalSim github issues page: https://github.com/GalSim-developers/GalSim/issues

    Note that the `wave_type` parameter does not propagate into other methods of `Bandpass`.
    For instance, Bandpass.__call__ assumes its input argument is in nanometers.

    The optional `zeropoint` argument can be used to specify the magnitude zeropoint for the
    bandpass.  This input parameter can also take a variety of possible forms:
    1. a number, which will be the zeropoint
    2. a galsim.SED.  In this case, the zeropoint is set such that the magnitude of the supplied
       SED through the bandpass is 0.0
    3. the string 'AB' or constant None.  In this (default) case, use an AB zeropoint.
    4. the string 'Vega'.  Use a Vega zeropoint.
    5. the string 'ST'.  Use a HST STmag zeropoint.
    For 3, 4, and 5, the SEDs whose magnitudes are to be calculated are assumed to have units
    of precisely erg/s/cm^2/nm, instead of simply units proportional to erg/nm, as would
    otherwise be assumed.

    @param throughput   Function defining the throughput at each wavelength.  See above for
                        valid options for this parameter.
    @param blue_limit   Hard cut off of bandpass on the blue side. [default: None, but required
                        if throughput is not a LookupTable or file.  See above.]
    @param red_limit    Hard cut off of bandpass on the red side. [default: None, but required
                        if throughput is not a LookupTable or file.  See above.]
    @param wave_type    The units to use for the wavelength argument of the `throughput`
                        function. See above for details. [default: 'nm']
    @param zeropoint    Zeropoint for this bandpass.
    """
    def __init__(self, throughput, blue_limit=None, red_limit=None, wave_type='nm',
                 zeropoint=None, _wave_list=None):
        # Note that `_wave_list` acts as a private construction variable that overrides the way that
        # `wave_list` is normally constructed (see `Bandpass.__mul__` below)

        # Figure out input throughput type.
        tp = throughput  # For brevity within this function
        if isinstance(tp, basestring):
            import os
            if os.path.isfile(tp):
                tp = galsim.LookupTable(file=tp, interpolant='linear')
            else:
                # Evaluate the function somewhere to make sure it is valid before continuing on.
                if red_limit is not None:
                    test_wave = red_limit
                elif blue_limit is not None:
                    test_wave = blue_limit
                else:
                    # If neither `blue_limit` nor `red_limit` is defined, then the Bandpass should
                    # be able to be evaluated at any wavelength, so check.
                    test_wave = 700
                try:
                    tp = eval('lambda wave : ' + tp)
                    tp(test_wave)
                except:
                    raise ValueError(
                        "String throughput must either be a valid filename or something that "+
                        "can eval to a function of wave. Input provided: {0}".format(throughput))

        # Figure out wavelength type
        if wave_type.lower() in ['nm', 'nanometer', 'nanometers']:
            wave_factor = 1.0
        elif wave_type.lower() in ['a', 'ang', 'angstrom', 'angstroms']:
            wave_factor = 10.0
        else:
            raise ValueError("Unknown wave_type '{0}'".format(wave_type))

        # Assign blue and red limits of bandpass
        if isinstance(tp, galsim.LookupTable):
            if blue_limit is None:
                blue_limit = tp.x_min
            if red_limit is None:
                red_limit = tp.x_max
        else:
            if blue_limit is None or red_limit is None:
                raise AttributeError(
                    "red_limit and blue_limit are required if throughput is not a LookupTable.")

        if blue_limit > red_limit:
            raise ValueError("blue_limit must be less than red_limit")
        self.blue_limit = blue_limit / wave_factor
        self.red_limit = red_limit / wave_factor

        # Sanity check blue/red limit and create self.wave_list
        if isinstance(tp, galsim.LookupTable):
            self.wave_list = np.array(tp.getArgs())/wave_factor
            # Make sure that blue_limit and red_limit are within LookupTable region of support.
            if self.blue_limit < (tp.x_min/wave_factor):
                raise ValueError("Cannot set blue_limit to be less than throughput "
                                 + "LookupTable.x_min")
            if self.red_limit > (tp.x_max/wave_factor):
                raise ValueError("Cannot set red_limit to be greater than throughput "
                                 + "LookupTable.x_max")
            # Make sure that blue_limit and red_limit are part of wave_list.
            if self.blue_limit not in self.wave_list:
                np.insert(self.wave_list, 0, self.blue_limit)
            if self.red_limit not in self.wave_list:
                np.insert(self.wave_list, -1, self.red_limit)
        else:
            self.wave_list = np.array([], dtype=np.float)

        # Manual override!  Be careful!
        if _wave_list is not None:
            self.wave_list = _wave_list

        self.func = lambda w: tp(np.array(w) * wave_factor)

        # Hack to avoid (LookupTable.x_max * 10) / 10.0 > LookupTable.x_max due to roundoff
        # error.
        if len(self.wave_list) > 0.0:
            self.wave_list[0] = self.wave_list[0] + 0.0000001
            self.wave_list[-1] = self.wave_list[-1] - 0.0000001

        self.zeropoint = zeropoint

    def __mul__(self, other):
        blue_limit = self.blue_limit
        red_limit = self.red_limit
        wave_list = self.wave_list

        if isinstance(other, (Bandpass, galsim.SED)):
            if len(other.wave_list) > 0:
                wave_list = np.union1d(wave_list, other.wave_list)
            blue_limit = max([self.blue_limit, other.blue_limit])
            red_limit = min([self.red_limit, other.red_limit])
            wave_list = wave_list[(wave_list >= blue_limit) & (wave_list <= red_limit)]

        # preserve type if possible.
        if type(self) == type(other):
            ret = self.copy()
            ret.blue_limit = blue_limit
            ret.red_limit = red_limit
            ret.wave_list = wave_list
            ret.func = lambda w: other(w)*self(w)
            if hasattr(ret, 'effective_wavelength'):
                del ret.effective_wavelength # this will get lazily recomputed when needed
            if hasattr(ret, '_zeropoint'):
                del ret._zeropoint # recalculate this too
            # Use default AB magnitudes for Bandpasses constructed via multiplication
            ret.zeropoint = None
        # if not possible, then demote to Bandpass
        elif hasattr(other, '__call__'):
            ret = Bandpass(lambda w: other(w)*self(w),
                           blue_limit=blue_limit, red_limit=red_limit,
                           _wave_list=wave_list)
        else:
            ret = Bandpass(lambda w: other*self(w),
                           blue_limit=blue_limit, red_limit=red_limit,
                           _wave_list=wave_list)

        return ret

    def __rmul__(self, other):
        return self*other

    # Doesn't check for divide by zero, so be careful.
    def __div__(self, other):
        blue_limit = self.blue_limit
        red_limit = self.red_limit
        wave_list = self.wave_list

        if isinstance(other, galsim.Bandpass):
            if len(other.wave_list) > 0:
                wave_list = np.union1d(wave_list, other.wave_list)
            blue_limit = max([self.blue_limit, other.blue_limit])
            red_limit = min([self.red_limit, other.red_limit])
            wave_list = wave_list[(wave_list >= blue_limit) & (wave_list <= red_limit)]

        # preserve type if possible.
        if type(self) == type(other):
            ret = self.copy()
            ret.blue_limit = blue_limit
            ret.red_limit = red_limit
            ret.wave_list = wave_list
            ret.func = lambda w: self(w)/other(w)
            if hasattr(ret, 'effective_wavelength'):
                del ret.effective_wavelength # this will get lazily recomputed when needed
            if hasattr(ret, '_zeropoint'):
                del ret._zeropoint # recalculate this too
            # Use default AB magnitudes for Bandpasses constructed via multiplication
            ret.zeropoint = None
        # if not possible, then demote to Bandpass
        elif hasattr(other, '__call__'):
            ret = Bandpass(lambda w: self(w)/other(w),
                           blue_limit=blue_limit, red_limit=red_limit,
                           _wave_list=wave_list)
        else:
            ret = Bandpass(lambda w: self(w)/other,
                           blue_limit=blue_limit, red_limit=red_limit,
                           _wave_list=wave_list)

        return ret

    # Doesn't check for divide by zero, so be careful.
    def __rdiv__(self, other):
        blue_limit = self.blue_limit
        red_limit = self.red_limit
        wave_list = self.wave_list

        if isinstance(other, galsim.Bandpass):
            if len(other.wave_list) > 0:
                wave_list = np.union1d(wave_list, other.wave_list)
            blue_limit = max([self.blue_limit, other.blue_limit])
            red_limit = min([self.red_limit, other.red_limit])
            wave_list = wave_list[(wave_list >= blue_limit) & (wave_list <= red_limit)]

        # preserve type if possible.
        if type(self) == type(other):
            ret = self.copy()
            ret.blue_limit = blue_limit
            ret.red_limit = red_limit
            ret.wave_list = wave_list
            ret.func = lambda w: other(w)/self(w)
            if hasattr(ret, 'effective_wavelength'):
                del ret.effective_wavelength # this will get lazily recomputed when needed
            if hasattr(ret, '_zeropoint'):
                del ret._zeropoint # recalculate this too
            # Use default AB magnitudes for Bandpasses constructed via multiplication
            ret.zeropoint = None
        # if not possible, then demote to Bandpass
        elif hasattr(other, '__call__'):
            ret = Bandpass(lambda w: other(w)/self(w),
                           blue_limit=blue_limit, red_limit=red_limit,
                           _wave_list=wave_list)
        else:
            ret = Bandpass(lambda w: other/self(w),
                           blue_limit=blue_limit, red_limit=red_limit,
                           _wave_list=wave_list)

        return ret

    # Doesn't check for divide by zero, so be careful.
    def __truediv__(self, other):
        return __div__(self, other)

    # Doesn't check for divide by zero, so be careful.
    def __rtruediv__(self, other):
        return __rdiv__(self, other)

    def copy(self):
        import copy
        return copy.deepcopy(self)

    def __call__(self, wave):
        """ Return dimensionless throughput of bandpass at given wavelength in nanometers.

        Note that outside of the wavelength range defined by the `blue_limit` and `red_limit`
        attributes, the throughput is assumed to be zero.

        @param wave   Wavelength in nanometers.

        @returns the dimensionless throughput.
        """
        # figure out what we received, and return the same thing
        # option 1: a NumPy array
        if isinstance(wave, np.ndarray):
            wgood = (wave >= self.blue_limit) & (wave <= self.red_limit)
            ret = np.zeros(wave.shape, dtype=np.float)
            np.place(ret, wgood, self.func(wave[wgood]))
            return ret
        # option 2: a tuple
        elif isinstance(wave, tuple):
            return tuple([self.func(w) if (w >= self.blue_limit and w <= self.red_limit) else 0.0
                          for w in wave])
        # option 3: a list
        elif isinstance(wave, list):
            return [self.func(w) if (w >= self.blue_limit and w <= self.red_limit) else 0.0
                    for w in wave]
        # option 4: a single value
        else:
            return self.func(wave) if (wave >= self.blue_limit and wave <= self.red_limit) else 0.0

    def getEffectiveWavelength(self):
        """ Calculate, store, and return the effective wavelength for this bandpass.  We define
        the effective wavelength as the throughput-weighted average wavelength, which is
        SED-independent.  Units are nanometers.
        """
        if not hasattr(self, 'effective_wavelength'):
            if len(self.wave_list) > 0:
                f = self.func(self.wave_list)
                self.effective_wavelength = (np.trapz(f * self.wave_list, self.wave_list) /
                                             np.trapz(f, self.wave_list))
            else:
                self.effective_wavelength = (galsim.integ.int1d(lambda w: self.func(w) * w,
                                                                self.blue_limit, self.red_limit)
                                             / galsim.integ.int1d(self.func,
                                                                  self.blue_limit, self.red_limit))
        return self.effective_wavelength

    def getZeroPoint(self):
        """ Calculate and return the magnitude zeropoint for this bandpass.
        """
        if not hasattr(self, '_zeropoint'):
            # If zeropoint is None (default), or set to a string 'AB', then use AB magnitudes
            if (self.zeropoint is None or
                (isinstance(self.zeropoint, basestring) and self.zeropoint.upper()=='AB')):
                AB_source = 3631e-23 # 3631 Jy in units of erg/s/Hz/cm^2
                c = 2.99792458e17 # speed of light in nm/s
                AB_flambda = AB_source * c / self.wave_list**2
                AB_sed = galsim.SED(galsim.LookupTable(self.wave_list, AB_flambda))
                flux = AB_sed.calculateFlux(self)
                self._zeropoint = 2.5 * np.log10(flux)
            # If zeropoint.upper() is 'ST', then use HST STmags:
            # http://www.stsci.edu/hst/acs/analysis/zeropoints
            elif (isinstance(self.zeropoint, basestring) and self.zeropoint.upper()=='ST'):
                ST_flambda = 3.63e-8 # erg/s/cm^2/nm
                ST_sed = galsim.SED(galsim.LookupTable(self.wave_list, ST_flambda))
                flux = ST_sed.calculateFlux(self)
                self._zeropoint = 2.5 * np.log10(flux)
            # If zeropoint.upper() is 'VEGA', then load vega spectrum stored in repository,
            # and use that for zeropoint spectrum.
            elif (isinstance(self.zeropoint, basestring) and self.zeropoint.upper()=='VEGA'):
                import os
                vegafile = os.path.join(galsim.meta_data.share_dir, "vega.txt")
                sed = galsim.SED(vegafile)
                flux = sed.calculateFlux(self)
                self._zeropoint = 2.5 * np.log10(flux)
            # If `self.zeropoint` is an `SED`, then compute the SED flux through the bandpass, and
            # use this to create a magnitude zeropoint.
            elif isinstance(self.zeropoint, galsim.SED):
                flux = self.zeropoint.calculateFlux(self)
                self._zeropoint = 2.5 * np.log10(flux)
            # If zeropoint is a number, then use that
            elif isinstance(self.zeropoint, (float, int)):
                self._zeropoint = self.zeropoint
            # But if zeropoint is none of these, raise an exception.
            else:
                raise ValueError(
                    "Don't know how to handle zeropoint of type: {0}".format(type(self.zeropoint)))
        return self._zeropoint

    def truncate(self, blue_limit=None, red_limit=None, relative_throughput=None):
        """Return a bandpass with its wavelength range truncated.

        If the bandpass was initialized with a galsim.LookupTable or from a file (which internally
        creates a galsim.LookupTable), then

        This function truncate the range of the bandpass either explicitly (with `blue_limit` or
        `red_limit` or both) or automatically, just trimming off leading and trailing wavelength
        ranges where the relative throughput is less than some amount (`relative_throughput`).

        This second option using relative_throughpt is only available for bandpasses initialized
        with a LookupTable or from a file, not when using a regular python funciton or a string
        evaluation.

        This function does not remove any intermediate wavelength ranges, but see thin() for
        a method that can thin out the intermediate values.

        @param blue_limit       Truncate blue side of bandpass here. [default: None]
        @param red_limit        Truncate red side of bandpass here. [default: None]
        @param relative_throughput  Truncate leading or trailing wavelengths that are below
                                this relative throughput level.  (See above for details.)
                                [default: None]

        @returns the truncated Bandpass.
        """
        if blue_limit is None:
            blue_limit = self.blue_limit
        if red_limit is None:
            red_limit = self.red_limit
        if len(self.wave_list) > 0:
            wave = np.array(self.wave_list)
            tp = self.func(wave)
            if relative_throughput is not None:
                w = (tp >= tp.max()*relative_throughput).nonzero()
                blue_limit = max([min(wave[w]), blue_limit])
                red_limit = min([max(wave[w]), red_limit])
        elif relative_throughput is not None:
            raise ValueError(
                "Can only truncate with relative_throughput argument if throughput is "
                + "a LookupTable")
        # preserve type
        ret = self.copy()
        ret.blue_limit = blue_limit
        ret.red_limit = red_limit
        if hasattr(ret, 'effective_wavelength'):
            del ret.effective_wavelength
        if hasattr(ret, '_zeropoint'):
            del ret._zeropoint
        return ret

    def thin(self, rel_err=1.e-4, preserve_range=False):
        """Thin out the internal wavelengths of a Bandpass that uses a LookupTable.

        If the bandpass was initialized with a LookupTable or from a file (which internally
        creates a LookupTable), this function removes tabulated values while keeping the integral
        over the set of tabulated values still accurate to the given relative error.

        That is, the integral of the bandpass function is preserved to a relative precision
        of `rel_err`, while eliminating as many internal wavelength values as possible.  This
        process will usually help speed up integrations using this bandpass.  You should weigh
        the speed improvements against your fidelity requirements for your particular use
        case.

        @param rel_err            The relative error allowed in the integral over the throughput
                                  function. [default: 1.e-4]
        @param preserve_range     Should the original range (`blue_limit` and `red_limit`) of the
                                  Bandpass be preserved? (True) Or should the ends be trimmed to
                                  include only the region where the integral is significant? (False)
                                  [default: False]

        @returns the thinned Bandpass.
        """
        if len(self.wave_list) > 0:
            x = self.wave_list
            f = self(x)
            newx, newf = utilities.thin_tabulated_values(x, f, rel_err=rel_err,
                                                         preserve_range=preserve_range)
            # preserve type
            ret = self.copy()
            ret.func = galsim.LookupTable(newx, newf, interpolant='linear')
            ret.blue_limit = np.min(newx) - 0.0000001
            ret.red_limit = np.max(newx) + 0.0000001
            ret.wave_list = np.array(newx)
            if hasattr(ret, 'effective_wavelength'):
                del ret.effective_wavelength
            if hasattr(ret, '_zeropoint'):
                del ret._zeropoint
            return ret
