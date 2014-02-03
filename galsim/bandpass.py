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
    """Very simple Bandpass filter object.  This object is callable, returning dimensionless
    throughput as a function of wavelength in nanometers.
    """
    def __init__(self, wave, throughput):
        """ Create a bandpass filter object.

        @param wave          Wavelength array in nanometers.
        @param throughput    Congruent dimensionless throughput array.
        """
        self.wave = numpy.array(wave)
        self.throughput = numpy.array(throughput)
        self.blue_limit = self.wave[0]
        self.red_limit = self.wave[-1]
        self.interp = galsim.LookupTable(wave, throughput)

    def __call__(self, wave):
        """ Return dimensionless throughput of bandpass at given wavelength in nanometers.
        @param wave   Wavelength in nanometers.
        @returns      Dimensionless throughput.
        """
        return self.interp(wave)

    def truncate(self, relative_throughput=None, blue_limit=None, red_limit=None):
        """ Return a bandpass with its wavelength range truncated.

        @param   blue_limit   Truncate blue side of bandpass here.
        @param   red_limit    Truncate red side of bandpass here.
        @param   relative_throughput     Truncate leading and trailing wavelength ranges where the
                                         relative throughput is less than this amount.  Do not
                                         remove any intermediate wavelength ranges.
        """
        if blue_limit is None:
            blue_limit = self.blue_limit
        if red_limit is None:
            red_limit = self.red_limit
        if relative_throughput is not None:
            mx = self.throughput.max()
            w = (self.throughput > mx*relative_throughput).nonzero()
            blue_limit = max([min(self.wave[w]), blue_limit])
            red_limit = min([max(self.wave[w]), red_limit])
        w = (self.wave >= blue_limit) & (self.wave <= red_limit)
        return Bandpass(self.wave[w], self.throughput[w])
