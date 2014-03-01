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
"""@file bandpass.py
Very simple implementation of a filter bandpass.  Used by galsim.chromatic.
"""

import numpy

import galsim

class Bandpass(object):
    """Simple bandpass object, which models the transmission fraction of incident light as a
    function of wavelength, for either an entire optical path (e.g., atmosphere, reflecting and
    refracting optics, filters, CCD quantum efficiency), or some intermediate piece thereof.
    Bandpasses representing individual components may be combined through the `*` operator to form
    a new Bandpass object representing the complete optical path.

    Bandpasses are callable, returning dimensionless throughput as a function of wavelength in nm.

    Bandpasses are immutable; all transformative methods return *new* Bandpasses, and leave their
    originating Bandpasses unaltered.

    Bandpasses require `blue_limit` and `red_limit` attributes, which may either be explicitly set
    at initialization, or are inferred from the initializing galsim.LookupTable or 2-column file.

    Outside of the wavelength interval between `blue_limit` and `red_limit`, the throughput is
    returned as zero, regardless of the `throughput` input parameter.

    Bandpasses may be multiplied by other Bandpasses, functions, or scalars.

    A Bandpass.effective_wavelength will be computed upon construction.  We use throughput-weighted
    average wavelength (which is independent of any SED) as our definition for effective wavelength.

    For Bandpasses defined using a LookupTable, a list of wavelengths, `wave_list`, defining the
    table is maintained.  Bandpasses defined as products of two other Bandpasses will define
    their `wave_list` as the union of multiplicand `wave_list`s, although limited to the range
    between the new product `blue_limit` and `red_limit`.  (This implementation detail may affect
    the choice of integrator used to draw `ChromaticObject`s.)

    The input parameter, throughput, may be one of several possible forms:
    1. a regular python function (or an object that acts like a function)
    2. a galsim.LookupTable
    3. a file from which a LookupTable can be read in
    4. a string which can be evaluated into a function of `wave`
       via `eval('lambda wave : '+throughput)`
       e.g. throughput = '0.8 + 0.2 * (wave-800)'

    The argument of `throughput` will be the wavelength in either nanometers (default) or
    Angstroms depending on the value of `wave_type`.  The output should be the dimensionless
    throughput at that wavelength.  (Note we use wave rather than lambda, since lambda is a
    python reserved word.)

    The argument `wave_type` specifies the units to assume for wavelength and must be one of
    'nm', 'nanometer', 'nanometers', 'A', 'Ang', 'Angstrom', or 'Angstroms'. Text case here
    is unimportant.  If these wavelength options are insufficient, please submit an issue to
    the GalSim github issues page: https://github.com/GalSim-developers/GalSim/issues

    Note that the `wave_type` parameter does not propagate into other methods of `Bandpass`.
    For instance, Bandpass.__call__ assumes its input argument is in nanometers.

    @param throughput    Function defining the throughput at each wavelength.  See above for
                         valid options for this parameter.
    @param blue_limit    Hard cut off of bandpass on the blue side.  This is optional if the
                         throughput is a LookupTable or a file, but is required if the
                         throughput is a function.
    @param red_limit     Hard cut off of bandpass on the red side.  This is optional if the
                         throughput is a LookupTable or a file, but is required if the
                         throughput is a function.

    """
    def __init__(self, throughput, wave_type='nm', blue_limit=None, red_limit=None, _wave_list=None):
        # Note that _wave_list acts a private construction variable that overrides the way that
        # `wave_list` is normally constructed (see Bandpass.__mul__ below)

        # Figure out input throughput type.
        tp = throughput  # For brevity within this function
        if isinstance(tp, (str, unicode)):
            import os
            if os.path.isfile(tp):
                tp = galsim.LookupTable(file=tp)
            else:
                tp = eval('lambda wave : ' + tp)

        # Figure out wavelength type
        if wave_type.lower() in ['nm', 'nanometer', 'nanometers']:
            wave_factor = 1.0
        elif wave_type.lower() in ['a', 'ang', 'angstrom', 'angstroms']:
            wave_factor = 10.0
        else:
            raise ValueError("Unknown wave_type `{0}` in SED.__init__".format(wave_type))

        # Assign blue and red limits of bandpass
        if blue_limit is None:
            if not isinstance(tp, galsim.LookupTable):
                raise AttributeError("blue_limit is required if throughput is not a LookupTable.")
            blue_limit = tp.x_min
        if red_limit is None:
            if not isinstance(tp, galsim.LookupTable):
                raise AttributeError("red_limit is required if throughput is not a LookupTable.")
            red_limit = tp.x_max
        self.blue_limit = blue_limit / wave_factor
        self.red_limit = red_limit / wave_factor

        # Sanity check blue/red limit and create self.wave_list
        if isinstance(tp, galsim.LookupTable):
            self.wave_list = [w/wave_factor for w in tp.getArgs()]
            # Make sure that blue_limit and red_limit are within LookupTable region of support.
            if self.blue_limit < (tp.x_min/wave_factor):
                raise ValueError("Cannot set Bandpass `blue_limit` to be less than"
                                 + "`LookupTable.x_min`")
            if self.red_limit > (tp.x_max/wave_factor):
                raise ValueError("Cannot set Bandpass `red_limit` to be greater than"
                                 + "`LookupTable.x_max`")
            # Make sure that blue_limit and red_limit are part of wave_list.
            if self.blue_limit not in self.wave_list:
                self.wave_list.insert(0, self.blue_limit)
            if self.red_limit not in self.wave_list:
                self.wave_list.append(self.red_limit)
        else:
            self.wave_list = []

        # Manual override!  Be careful!
        if _wave_list is not None:
            self.wave_list = _wave_list

        self.func = lambda w: tp(numpy.array(w) * wave_factor)

        # Evaluate and store bandpass effective wavelength, which we define as the
        # throughput-weighted average wavelength, independent of any SED.  Units are nanometers.
        if self.wave_list != []:
            f = self.func(self.wave_list)
            self.effective_wavelength = (numpy.trapz(f * self.wave_list, self.wave_list) /
                                         numpy.trapz(f, self.wave_list))
        else:
            self.effective_wavelength = (galsim.integ.int1d(lambda w: self.func(w) * w,
                                                            self.blue_limit, self.red_limit)
                                         / galsim.integ.int1d(self.func,
                                                              self.blue_limit, self.red_limit))

    def __mul__(self, other):
        blue_limit = self.blue_limit
        red_limit = self.red_limit
        wave_list = set(self.wave_list)

        if isinstance(other, galsim.Bandpass):
            blue_limit = max([self.blue_limit, other.blue_limit])
            red_limit = min([self.red_limit, other.red_limit])
            if other.wave_list != []:
                wave_list = wave_list.union(other.wave_list)

        wave_list = list(wave_list)
        wave_list = [w for w in wave_list if w >= blue_limit and w <= red_limit]
        wave_list.sort()

        if hasattr(other, '__call__'):
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
        wave_list = set(self.wave_list)
        if isinstance(other, galsim.Bandpass):
            blue_limit = max([self.blue_limit, other.blue_limit])
            red_limit = min([self.red_limit, other.red_limit])
            if other.wave_list != []:
                wave_list = wave_list.union(other.wave_list)
        if hasattr(other, '__call__'):
            ret = Bandpass(lambda w: self(w)/other(w), blue_limit=blue_limit, red_limit=red_limit)
        else:
            ret = Bandpass(lambda w: self(w)/other, blue_limit=blue_limit, red_limit=red_limit)
        wave_list = list(wave_list)
        wave_list = [w for w in wave_list if w >= blue_limit and w <= red_limit]
        wave_list.sort()
        ret.wave_list = wave_list
        return ret

    # Doesn't check for divide by zero, so be careful.
    def __rdiv__(self, other):
        blue_limit = self.blue_limit
        red_limit = self.red_limit
        wave_list = set(self.wave_list)
        if isinstance(other, galsim.Bandpass):
            blue_limit = max([self.blue_limit, other.blue_limit])
            red_limit = min([self.red_limit, other.red_limit])
            if other.wave_list != []:
                wave_list = wave_list.union(other.wave_list)
        if hasattr(other, '__call__'):
            ret = Bandpass(lambda w: other(w)/self(w), blue_limit=blue_limit, red_limit=red_limit)
        else:
            ret = Bandpass(lambda w: other/self(w), blue_limit=blue_limit, red_limit=red_limit)
        wave_list = list(wave_list)
        wave_list = [w for w in wave_list if w >= blue_limit and w <= red_limit]
        wave_list.sort()
        ret.wave_list = wave_list
        return ret

    # Doesn't check for divide by zero, so be careful.
    def __truediv__(self, other):
        return __div__(self, other)

    # Doesn't check for divide by zero, so be careful.
    def __rtruediv__(self, other):
        return __rdiv__(self, other)

    def __call__(self, wave):
        """ Return dimensionless throughput of bandpass at given wavelength in nanometers.

        Note that outside of the wavelength range defined by the `blue_limit` and `red_limit`
        attributes, the throughput is assumed to be zero.

        @param wave   Wavelength in nanometers.
        @returns      Dimensionless throughput.
        """
        # figure out what we received, and return the same thing
        # option 1: a Numpy array
        if isinstance(wave, numpy.ndarray):
            wgood = (wave >= self.blue_limit) & (wave <= self.red_limit)
            ret = numpy.zeros(wave.shape, dtype=numpy.float)
            numpy.place(ret, wgood, self.func(wave[wgood]))
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

    def createTruncated(self, relative_throughput=None, blue_limit=None, red_limit=None):
        """ Return a bandpass with its wavelength range truncated.

        @param blue_limit             Truncate blue side of bandpass here.
        @param red_limit              Truncate red side of bandpass here.
        @param relative_throughput    If the bandpass was initialized with a galsim.LookupTable or
                                      from a file (which internally creates a galsim.LookupTable),
                                      then truncate leading and trailing wavelength ranges where the
                                      relative throughput is less than this amount.  Do not remove
                                      any intermediate wavelength ranges.  This option is not
                                      available for bandpasses initialized with a function or
                                      `eval` string.
        @returns   The truncated Bandpass.
        """
        if blue_limit is None:
            blue_limit = self.blue_limit
        if red_limit is None:
            red_limit = self.red_limit
        if hasattr(self, 'wave_list'):
            wave = numpy.array(self.wave_list)
            tp = self.func(wave)
            if relative_throughput is not None:
                w = (tp >= tp.max()*relative_throughput).nonzero()
                blue_limit = max([min(wave[w]), blue_limit])
                red_limit = min([max(wave[w]), red_limit])
            w = (wave >= blue_limit) & (wave <= red_limit)
            return Bandpass(galsim.LookupTable(wave[w], tp[w]))
        else:
            if relative_throughput is not None:
                raise ValueError("relative_throughput only available if Bandpass is specified as"
                                 +" a galsim.LookupTable")
            return Bandpass(self.func, blue_limit=blue_limit, red_limit=red_limit)

    def createThinned(self, step):
        """ If the bandpass was initialized with a galsim.LookupTable or from a file (which
        internally creates a galsim.LookupTable), then thin the tabulated wavelengths and
        throughput by keeping only every `step`th index.  Always keep the first and last index,
        however, to maintain the same blue_limit and red_limit.

        @param step     Factor by which to thin the tabulated Bandpass wavelength and throughput
                        arrays.
        @returns  The thinned Bandpass.
        """
        # if isinstance(self.func, galsim.LookupTable):
        if hasattr(self, 'wave_list'):
            wave = self.wave_list[::step]
            # maintain the same red_limit, even if it breaks the step size a bit.
            if wave[-1] != self.wave_list[-1]:
                wave.append(self.wave_list[-1])
            throughput = self.func(wave)
            return Bandpass(galsim.LookupTable(wave, throughput))
