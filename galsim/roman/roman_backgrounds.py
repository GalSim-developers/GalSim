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
"""
@file roman_backgrounds.py

Part of the Roman Space Telescope module.  This file includes any routines needed to define the
background level, for which the main contribution is zodiacal light.
"""

import numpy as np
import os


def getSkyLevel(bandpass, world_pos=None, exptime=None, epoch=2025, date=None):
    """
    Get the expected sky level for a Roman ST observation due to zodiacal light for this bandpass
    and position.

    This routine requires Bandpass objects that were loaded by galsim.roman.getBandpasses().  That
    routine will have stored tables containing the sky background as a function of position on the
    sky for that bandpass.  This routine then interpolates between the values in those tables to
    arbitrary positions on the sky.

    The numbers that are stored in the Bandpass object ``bandpass`` are background level in units of
    e-/m^2/s/arcsec^2.  To get rid of the m^2, this routine multiplies by the total effective
    collecting area in m^2.  Multiplying by the exposure time gives a result in e-/arcsec^2.  The
    result can either be multiplied by the approximate pixel area to get e-/pix, or the result can
    be used with wcs.makeSkyImage() to make an image of the sky that properly includes the actual
    pixel area as a function of position on the detector.

    The source of the tables that are being interpolated is Chris Hirata's publicly-available Roman
    exposure time calculator (ETC):

        http://www.tapir.caltech.edu/~chirata/web/software/space-etc/

    It nominally returns photons/m^2/s/arcsec^2, but the input bandpasses used internally by the ETC
    code include the quantum efficiency, to effectively convert to e-/m^2/s/arcsec^2.  Note that in
    general results will depend on the adopted model for zodiacal light, and these are uncertain at
    the ~10% level.

    Positions should be specified with the ``world_pos`` keyword, which must be a CelestialCoord
    object.  If no ``world_pos`` is supplied, then the routine will use a default position that
    looks sensibly away from the sun.

    Parameters:
        bandpass:   A Bandpass object.
        world_pos:  A position, given as a CelestialCoord object.  If None, then the routine
                    will use an ecliptic longitude of 90 degrees with respect to the sun
                    position (as a fair compromise between 0 and 180), and an ecliptic latitude
                    of 30 degrees with respect to the sun position (decently out of the plane
                    of the Earth-sun orbit). [default: None]
        exptime:    Exposure time in seconds.  If None, use the default Roman exposure time.
                    [default: None]
        epoch:      The epoch to be used for estimating the obliquity of the ecliptic when
                    converting ``world_pos`` to ecliptic coordinates.  This keyword is only used
                    if ``date`` is None, otherwise ``date`` is used to determine the ``epoch``.
                    [default: 2025]
        date:       The date of the observation, provided as a python datetime object.  If None,
                    then the conversion to ecliptic coordinates assumes the sun is at ecliptic
                    coordinates of (0,0), as it is at the vernal equinox. [default: None]

    Returns:
        the expected sky level in e-/arcsec^2.
    """
    from .. import degrees, radians, CelestialCoord
    from .. import GalSimValueError
    from . import diameter, obscuration

    if exptime is None:
        from . import exptime

    # Check for cached sky level information for this filter.  If not, raise exception
    if not hasattr(bandpass, '_sky_level'):
        raise GalSimValueError("Only bandpasses returned from galsim.roman.getBandpasses() are "
                               "allowed here!", bandpass)

    # Check for proper type for position, and extract the ecliptic coordinates.
    if world_pos is None:
        # Use our defaults for the case of unspecified position.
        ecliptic_lat = 30.*degrees
        ecliptic_lon = 90.*degrees
    else:
        if not isinstance(world_pos, CelestialCoord):
            raise TypeError("world_pos must be supplied as a CelestialCoord.")
        if date is not None:
            epoch = date.year
        ecliptic_lon, ecliptic_lat = world_pos.ecliptic(epoch=epoch, date=date)

    # Check the position in our table, and make sure to take advantage of the latitude / longitude
    # symmetries:
    # The table only includes positive values of latitude, because there is symmetry about zero.  So
    # we take the absolute value of the input ecliptic latitude.
    # The table only includes longitude in the range [0, 180] because there is symmetry in that a
    # negative longitude in the range[-180, 0] should have the same sky level as at the positive
    # value of longitude (given that the Sun is at 0).
    ecliptic_lon = ecliptic_lon.wrap()
    ecliptic_lon = abs(ecliptic_lon.rad)*radians
    ecliptic_lat = abs(ecliptic_lat.rad)*radians
    sin_ecliptic_lat = np.sin(ecliptic_lat)

    # Take the lookup table, and turn negative numbers (indicating failure because of proximity to
    # sun) to large positive values so that we can identify them as bad after interpolation.
    max_sky = np.max(bandpass._sky_level)
    sky_level = bandpass._sky_level.copy()
    sky_level[sky_level<0] = 1.e6

    # Interpolate in 2d on the table.
    s = sky_level.reshape(46,42).transpose()
    xlat = sin_ecliptic_lat*41
    xlon = abs(ecliptic_lon.wrap() / degrees)/4.
    ilat = int(xlat)
    ilon = int(xlon)
    xlat -= ilat
    xlon -= ilon
    sky_val = (s[ilat, ilon] * (1.-xlat)*(1.-xlon) +
               s[ilat, ilon+1] * (1.-xlat)*xlon +
               s[ilat+1, ilon] * xlat*(1.-xlon) +
               s[ilat+1, ilon+1] * xlat*xlon)

    # If the result is too large, then raise an exception: we should not look at this position!
    if sky_val > max_sky:
        raise GalSimValueError("world_pos is too close to sun. Would not observe here.", world_pos)

    # Now, convert to the right units, and return.  (See docstring for explanation.)
    # First, multiply by the effective collecting area in m^2.
    eff_area = 0.25 * np.pi * diameter**2 * (1. - obscuration**2)
    sky_val *= eff_area
    # Multiply by exposure time.
    sky_val *= exptime

    # The result is now the sky level in e-/arcsec^2.
    return sky_val
