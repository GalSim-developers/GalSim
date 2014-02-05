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

    Bandpasses require `blue_limit` and `red_limit` attributes, which may either be explicity set
    at initialization, or are inferred from the initializing galsim.LookupTable or 2-column file.

    Bandpases are only defined between `blue_limit` and `red_limit`.  Requesting a throughput value
    outside of this range raises an exception.

    Bandpasses may be multiplied by other Bandpasses, functions, or scalars.

    Products of two Bandpasses are defined only on the overlapping wavelengths for which their
    multiplicands are defined, with `blue_limit` and `red_limit` updated to match.
    """
    def __init__(self, throughput, blue_limit=None, red_limit=None):
        """Very simple Bandpass filter object.  This object is callable, returning dimensionless
        throughput as a function of wavelength in nanometers.

        The input parameter, throughput, may be one of several possible forms:
        1. a regular python function (or an object that acts like a function)
        2. a galsim.LookupTable
        3. a file from which a LookupTable can be read in
        4. a string which can be evaluated into a function of `wave`
           via `eval('lambda wave : '+throughput)
           e.g. throughput = '0.8 + 0.2 * (wave-800)`

        The argument of the function will be the wavelength in nanometers, and the output should be
        the dimensionless throughput at that wavelength.  (Note we use wave rather than lambda,
        since lambda is a python reserved word.)

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
        if isinstance(tp, str):
            import os
            if os.path.isfile(tp):
                tp = galsim.LookupTable(file=tp)
            else:
                tp = eval('lambda wave : ' + tp)
        if blue_limit is None:
            if not isinstance(tp, galsim.LookupTable):
                raise AttributeError("blue_limit is required if throughput is not a LookupTable.")
            blue_limit = tp.x_min
        if red_limit is None:
            if not isinstance(tp, galsim.LookupTable):
                raise AttributeError("red_limit is required if throughput is not a LookupTable.")
            red_limit = tp.x_max
        self.func = tp
        self.blue_limit = blue_limit
        self.red_limit = red_limit

    def __mul__(self, other):
        blue_limit = self.blue_limit
        red_limit = self.red_limit
        if isinstance(other, galsim.Bandpass):
            blue_limit = max([self.blue_limit, other.blue_limit])
            red_limit = min([self.red_limit, other.red_limit])
        if hasattr(other, '__call__'):
            return Bandpass(lambda w: other(w)*self(w), blue_limit=blue_limit, red_limit=red_limit)
        else:
            return Bandpass(lambda w: other*self(w), blue_limit=blue_limit, red_limit=red_limit)

    def __rmul__(self, other):
        return self*other

    # Doesn't check for divide by zero, so be careful.
    def __div__(self, other):
        blue_limit = self.blue_limit
        red_limit = self.red_limit
        if isinstance(other, galsim.Bandpass):
            blue_limit = max([self.blue_limit, other.blue_limit])
            red_limit = min([self.red_limit, other.red_limit])
        if hasattr(other, '__call__'):
            return Bandpass(lambda w: self(w)/other(w), blue_limit=blue_limit, red_limit=red_limit)
        else:
            return Bandpass(lambda w: self(w)/other, blue_limit=blue_limit, red_limit=red_limit)

    # Doesn't check for divide by zero, so be careful.
    def __rdiv__(self, other):
        blue_limit = self.blue_limit
        red_limit = self.red_limit
        if isinstance(other, galsim.Bandpass):
            blue_limit = max([self.blue_limit, other.blue_limit])
            red_limit = min([self.red_limit, other.red_limit])
        if hasattr(other, '__call__'):
            return Bandpass(lambda w: other(w)/self(w), blue_limit=blue_limit, red_limit=red_limit)
        else:
            return Bandpass(lambda w: other/self(w), blue_limit=blue_limit, red_limit=red_limit)

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
        if wave < self.blue_limit:
            raise ValueError("Wavelength out of range for Bandpass")
        if wave > self.red_limit:
            raise ValueError("Wavelength out of range for Bandpass")
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
        if isinstance(self.func, galsim.LookupTable):
            tp = numpy.array(self.func.getVals())
            wave = numpy.array(self.func.getArgs())
            if relative_throughput is not None:
                w = (tp >= tp.max()*relative_throughput).nonzero()
                blue_limit = max([min(wave[w]), blue_limit])
                red_limit = min([max(wave[w]), red_limit])
            w = (wave >= blue_limit) & (wave <= red_limit)
            return Bandpass(galsim.LookupTable(wave[w], tp[w]))
        else:
            if relative_throughput is not None:
                raise ValueError("relative_throughput only available for galsim.LookupTable")
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
        if isinstance(self.func, galsim.LookupTable):
            wave = self.func.getArgs()[::step]
            throughput = self.func.getVals()[::step]
            # maintain the same red_limit, even if it breaks the step size a bit.
            if throughput[-1] != self.func.x_max:
                throughput.append(self.func.x_max)
            return Bandpass(galsim.LookupTable(wave, throughput))
