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
"""@file wcs.py 
All the classes to implement different WCS transformations for GalSim Images.
"""


class BaseWCS(object):
    """The base class for all other kinds of WCS transformations.  It doesn't really
    do much except provide a type for isinstance(wcs,BaseWCS) queries.
    """    
    def __init__(self):
        raise TypeError("BaseWCS is an abstract base class.  It cannot be instantiated.")


class PixelScale(BaseWCS):
    """This is the simplest possible WCS transformation.  It only involves a unit conversion
    from pixels to arcsec (or whatever units you want to take for your sky coordinate system).

    The conversion functions are:

        u = x * scale
        v = y * scale

    Initialization
    --------------
    A PixelScale object is initialized with the command:

        wcs = galsim.PixelScale(scale)

    @param scale        The pixel scale, typically in units of arcsec/pixel.
    """
    _req_params = { "scale" : float }
    _opt_params = {}
    _single_params = []
    _takes_rng = False
    _takes_logger = False

    def __init__(self, scale):
        self.scale = scale

    def toSky(self, chip_pos):
        """Convert from chip coordinates to sky coordinates

        @param chip_pos     The chip coordinates as a galsim.PositionD
        @returns sky_pos    The sky coordinates as a galsim.PositionD
        """
        return chip_pos * self.scale

    def toChip(self, sky_pos):
        """Convert from sky coordinates to chip coordinates

        @param sky_pos      The sky coordinates as a galsim.PositionD
        @returns chip_pos   The chip coordinates as a galsim.PositionD
        """
        return sky_pos / self.scale

    def pixelArea(self, chip_pos=None, sky_pos=None):
        """Return the area of a pixel in arcsec**2 (or in whatever units you are using for 
        sky coordinates).

        For compatibility with other WCS classes, this takes chip_pos or sky_pos, but
        both are ignored by the PixelScale version of this function.

        @param sky_pos      The sky coordinates as a galsim.PositionD (ignored)
        @param chip_pos     The chip coordinates as a galsim.PositionD (ignored)
        @returns            The pixel area in arcsec**2
        """
        return self.scale**2

    def linearScale(self, chip_pos=None, sky_pos=None):
        """Return a reasonable estimate of the linear scale of the transformation

        I'm not sure yet what this really means for wcs's other than PixelScale, 
        but there are some places where we need a linear scale factor, so this is it.
        I think for other WCS types, this will just be the sqrt of the pixelArea().

        For compatibility with other WCS classes, this takes chip_pos or sky_pos, but
        both are ignored by the PixelScale version of this function.

        @param sky_pos      The sky coordinates as a galsim.PositionD (ignored)
        @param chip_pos     The chip coordinates as a galsim.PositionD (ignored)
        @returns            The pixel area in arcsec**2
        """
        return self.scale

    def applyTo(self, profile, chip_pos=None, sky_pos=None):
        """Apply the appropriate transformation to convert a profile in sky coordinates
        to the corresponding profile in chip coordinates.

        For compatibility with other WCS classes, this takes chip_pos or sky_pos, but
        both are ignored by the PixelScale version of this function.

        @param profile      The profile to be converted from sky coordinates to chip coordinates.
        @param sky_pos      The sky coordinates as a galsim.PositionD (ignored)
        @param chip_pos     The chip coordinates as a galsim.PositionD (ignored)
        """
        profile.applyDilation(1./self.scale)

    def applyInverseTo(self, profile, chip_pos=None, sky_pos=None):
        """Apply the appropriate transformation to convert a profile in chip coordinates
        back to the corresponding profile in sky coordinates.

        For compatibility with other WCS classes, this takes chip_pos or sky_pos, but
        both are ignored by the PixelScale version of this function.

        @param profile      The profile to be converted from chip coordinates to sky coordinates.
        @param sky_pos      The sky coordinates as a galsim.PositionD (ignored)
        @param chip_pos     The chip coordinates as a galsim.PositionD (ignored)
        """
        profile.applyDilation(self.scale)

    def isVariable(self):
        """Return whether this WCS solution has a variable function."""
        return False

    def local(self, chip_pos=None, sky_pos=None):
        """Return the local linear approximation of this WCS at a given point.

        Since a PixelScale is already linear, this just returns itself.

        For compatibility with other WCS classes, this takes chip_pos or sky_pos, but
        both are ignored by the PixelScale version of this function.

        @param sky_pos      The sky coordinates as a galsim.PositionD (ignored)
        @param chip_pos     The chip coordinates as a galsim.PositionD (ignored)
        @returns local_wcs  A WCS object with wcs.isVariable() == False
        """
        return self

    def copy(self):
        return PixelScale(self.scale)

    def __eq__(self, other):
        if not isinstance(other, PixelScale): return False
        else: return self.scale == other.scale

    def __ne__(self, other):
        return not self.__eq__(other)


