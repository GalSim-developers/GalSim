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
    """Simple bandpass object.

    Bandpasses are callable, returning dimensionless throughput as a function of wavelength in nm.

    Bandpasses are immutable; all transformative methods return *new* Bandpasses, and leave their
    originating Bandpasses unaltered.

    Bandpasses require `blue_limit` and `red_limit` attributes, which may either be explicitly set
    at initialization, or are inferred from the initializing galsim.LookupTable or 2-column file.

    Bandpases are only defined between `blue_limit` and `red_limit`.  Requesting a throughput value
    outside of this range raises an exception.

    Bandpasses may be multiplied by other Bandpasses, functions, or scalars.

    Products of two Bandpasses are defined only on the overlapping wavelengths for which their
    multiplicands are defined, with `blue_limit` and `red_limit` updated to match.

    A Bandpass.effective_wavelength will be computed upon construction.  We use throughput-weighted
    average wavelength (which is independent of any SED) as our definition for effective wavelength.
    """
    def __init__(self, throughput, wave_type='nm', blue_limit=None, red_limit=None):
        """Very simple Bandpass filter object.  This object is callable, returning dimensionless
        throughput as a function of wavelength in nanometers.

        The input parameter, throughput, may be one of several possible forms:
        1. a regular python function (or an object that acts like a function)
        2. a galsim.LookupTable
        3. a file from which a LookupTable can be read in
        4. a string which can be evaluated into a function of `wave`
           via `eval('lambda wave : '+throughput)
           e.g. throughput = '0.8 + 0.2 * (wave-800)`

        The argument of `throughput` will be the wavelength in either nanometers (default) or
        Angstroms depending on the value of `wave_type`.  The output should be the dimensionless
        throughput at that wavelength.  (Note we use wave rather than lambda, since lambda is a
        python reserved word.)

        The argument `wave_type` specifies the units to assume for wavelength and must be one of
        'nm', 'nanometer', 'nanometers', 'A', 'Ang', 'Angstrom', or 'Angstroms'. Text case here
        is unimportant.

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
        tp = throughput  # For brevity within this function
        if isinstance(tp, (str, unicode)):
            import os
            if os.path.isfile(tp):
                tp = galsim.LookupTable(file=tp)
            else:
                tp = eval('lambda wave : ' + tp)
        if wave_type.lower() in ['nm', 'nanometer', 'nanometers']:
            wave_factor = 1.0
        elif wave_type.lower() in ['a', 'ang', 'angstrom', 'angstroms']:
            wave_factor = 10.0
        else:
            raise ValueError("Unknown wave_type `{}` in SED.__init__".format(wave_type))
        if blue_limit is None:
            if not isinstance(tp, galsim.LookupTable):
                raise AttributeError("blue_limit is required if throughput is not a LookupTable.")
            blue_limit = tp.x_min
        if red_limit is None:
            if not isinstance(tp, galsim.LookupTable):
                raise AttributeError("red_limit is required if throughput is not a LookupTable.")
            red_limit = tp.x_max
        if isinstance(tp, galsim.LookupTable):
            self.wave_list = [w/wave_factor for w in tp.getArgs()]
        else:
            self.wave_list = []
        self.func = lambda w: tp(numpy.array(w) * wave_factor)
        self.blue_limit = blue_limit / wave_factor
        self.red_limit = red_limit / wave_factor
        # We define bandpass effective wavelength as the throughput-weighted average wavelength,
        # independent of any SED.  Units are nanometers.
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
        if hasattr(other, '__call__'):
            ret = Bandpass(lambda w: other(w)*self(w), blue_limit=blue_limit, red_limit=red_limit)
        else:
            ret = Bandpass(lambda w: other*self(w), blue_limit=blue_limit, red_limit=red_limit)
        wave_list = list(wave_list)
        wave_list = [w for w in wave_list if w >= blue_limit and w <= red_limit]
        wave_list.sort()
        ret.wave_list = wave_list
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
        attributes, the Bandpass is considered undefined, and this method will raise an exception
        if a throughput at a wavelength outside the defined range is requested.

        @param wave   Wavelength in nanometers.
        @returns      Dimensionless throughput.
        """
        if hasattr(wave, '__iter__'): # Only iterables respond to min(), max()
            wmin = min(wave)
            wmax = max(wave)
        else: # python scalar
            wmin = wave
            wmax = wave
        if wmin < self.blue_limit:
            raise ValueError("Wavelength ({}) is bluer than Bandpass blue limit ({})"
                             .format(wmin, self.blue_limit))
        if wmax > self.red_limit:
            raise ValueError("Wavelength ({}) is redder than Bandpass red limit ({})"
                             .format(wmax, self.red_limit))
        return self.func(wave)

    def truncate(self, relative_throughput=None, blue_limit=None, red_limit=None):
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
#        if isinstance(self.func, galsim.LookupTable):
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

    def thin(self, step):
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
