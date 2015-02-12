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
"""
@file wfirst_backgrounds.py

Part of the WFIRST module.  This file includes any routines needed to define the background level,
for which the main contribution is zodiacal light.
"""

import galsim
import numpy as np
import os

def getSkyLevel(bandpass, position=None, e_lat=None, e_lon=None, exp_time=None):
    """
    Get the expected sky level for a WFIRST observation due to zodiacal light for this bandpass and
    position.

    This routine requires Bandpass objects that were loaded by galsim.wfirst.getBandpasses().  That
    routine will have stored tables containing the sky background as a function of position on the
    sky for that bandpass.  This routine then interpolates between the values in those tables to
    arbitrary positions on the sky.

    The numbers that are stored in the Bandpass object `bandpass` are background level in units of
    e-/m^2/s/arcsec^2.  To get rid of the m^2, this routine multiplies by the total effective
    collecting area in m^2, and to get rid of the arcsec^2 it multiplies by the pixel area in
    arcsec^2.  This will give a result in e-/s/pix.  If the user has passed in an exposure time, then
    the results are returned in e-/pix, otherwise the results are returned in e-/pix/s and the user
    must multiply by the exposure time later on themselves.

    The source of the tables that are being interpolated is Chris Hirata's publicly-available WFIRST
    exposure time calculator (ETC):

        http://www.tapir.caltech.edu/~chirata/web/software/space-etc/

    It nominally returns photons/m^2/s/arcsec^2, but the input bandpasses used internally by the ETC
    code include the quantum efficiency, to effectively convert to e-/m^2/s/arcsec^2.

    Positions can be specified either with a keyword `position`, which must be a CelestialCoord
    object, or with both `e_lat` and `e_lon` for the ecliptic coordinates (as Angle instances).
    Without any of those values supplied, the routine will use a default position that looks
    sensibly away from the sun.

    @param bandpass     A Bandpass object.
    @param position     A position, given as a CelestialCoord object.  If None, and `e_lat` and
                        `e_lon` are not specified, then the routine will use an ecliptic longitude
                        of 90 degrees (as a fair compromise between 0 and 180), and an ecliptic
                        latitude of 30 degrees (decently out of the plane of the Earth-sun orbit).
                        [default: None]
    @param e_lat        Ecliptic latitude, given as an Angle instance.  Can only be specified if
                        `position` is not specified, and if specified, then `e_lon` must also be
                        specified. [default: None]
    @param e_lon        Ecliptic longitude, given as an Angle instance.  Can only be specified if
                        `position` is not specified, and if specified, then `e_lat` must also be
                        specified. [default: None]
    @param exp_time     Exposure time in seconds.  If None, the routine will return the sky level in
                        e-/pix/s instead of in e-/pix.  [default: None]

    @returns the expected sky level in e-/pix/s, unless `exp_time` was used, in which case the
    results are in e-/pix.
    """
    # Check for cached sky level information for this filter.  If not, raise exception
    if not hasattr(bandpass, '_sky_level'):
        raise RuntimeError("This bandpass does not have stored sky level information!")

    # Check for proper type for position
    if position is not None:
        if not isinstance(position, galsim.CelestialCoord):
            raise ValueError("Position must be supplied as a CelestialCoord!")
        if e_lat is not None or e_lon is not None:
            raise ValueError("Position is specified in too many ways!")
    if e_lat is not None and not isinstance(e_lat, galsim.Angle):
        raise ValueError("Ecliptic latitude must be supplied as an Angle!")
    if e_lon is not None and not isinstance(e_lon, galsim.Angle):
        raise ValueError("Ecliptic longitude must be supplied as an Angle!")
    if (e_lat is None and e_lon is not None) or (e_lat is not None and e_lon is None):
        raise ValueError("Ecliptic latitude and longitude must be simultaneously specified!")

    # Check the position, and make sure to take advantage of the latitude / longitude symmetries:
    # The table only includes positive values of latitude, because there is symmetry about zero.  So
    # we take the absolute value of the input ecliptic latitude.
    # The table only includes longitude in the range [0, 180] because there is symmetry in that a
    # negative longitude in the range[-180, 0] should have the same sky level as at the positive
    # value of longitude (given that the Sun is at 0).
    if position is None:
        if e_lat is None:
            ecliptic_lat = 30.
            ecliptic_lon = 90.
        else:
            ecliptic_lat = e_lat / galsim.degrees
            ecliptic_lon = e_lon / galsim.degrees
    else:
        ecliptic_lon, ecliptic_lat = position.ecliptic()
        ecliptic_lon = ecliptic_lon / galsim.degrees
        ecliptic_lat = ecliptic_lat / galsim.degrees
        if ecliptic_lon > 180.:
            ecliptic_lon -= 360.
            ecliptic_lon = abs(ecliptic_lon)
        if ecliptic_lat < 0.:
            ecliptic_lat = abs(ecliptic_lat)
    sin_ecliptic_lat = np.sin(np.pi*ecliptic_lat/180.)

    # Take the lookup table, and turn negative numbers (indicating failure because of proximity to
    # sun) to large positive values so that we can identify them as bad after interpolation.
    max_sky = np.max(bandpass._sky_level)
    sky_level = bandpass._sky_level.copy()
    sky_level[sky_level<0] = 1.e6

    # Interpolate!  We could do a full 2d interpolation, but this seems a bit like overkill.  We're
    # going to first interpolate in latitude, then in longitude.
    # Equal spacing in sin(latitude).
    all_sin_lat_vals = np.sin(np.pi*bandpass._ecliptic_lat/180.)
    sin_lat_vals = np.sort(np.array(list(set(all_sin_lat_vals))))
    n_sl = len(sin_lat_vals)
    d_sl = sin_lat_vals[1]-sin_lat_vals[0]
    # Handle edge cases:
    if sin_ecliptic_lat <= sin_lat_vals[0]:
        sky_table_1d = sky_level[all_sin_lat_vals == sin_lat_vals[0]]
    elif sin_ecliptic_lat >= sin_lat_vals[n_sl-1]:
        sky_table_1d = sky_level[all_sin_lat_vals == sin_lat_vals[n_sl-1]]
    else:
        interp_val = (sin_ecliptic_lat - sin_lat_vals[0])/d_sl
        lower_ind = int(np.floor(interp_val))
        frac = interp_val - lower_ind
        sky_table_1d = (1.-frac)*sky_level[all_sin_lat_vals == sin_lat_vals[lower_ind]] \
            + frac*sky_level[all_sin_lat_vals == sin_lat_vals[lower_ind+1]]
    # Now we have a table to interpolate on the set of galactic longitude values.
    long_vals = np.sort(np.array(list(set(bandpass._ecliptic_lon))))
    n_l = len(long_vals)
    d_l = long_vals[1]-long_vals[0]
    # Handle edge cases:
    if ecliptic_lon <= long_vals[0]:
        sky_val = sky_table_1d[0]
    elif ecliptic_lon >= long_vals[n_l-1]:
        sky_val = sky_table_1d[n_l-1]
    else:
        interp_val = (ecliptic_lon - long_vals[0])/d_l
        lower_ind = int(np.floor(interp_val))
        frac = interp_val - lower_ind
        sky_val = (1.-frac)*sky_table_1d[lower_ind] + frac*sky_table_1d[lower_ind]

    # If the result is too large, then raise an exception: we should not look at this position!
    if sky_val > max_sky:
        raise ValueError("Position is too close to sun!  Would not observe here.")

    # Now, convert to the right units, and return.  (See docstring for explanation.)
    # First, multiply by the effective collecting area in m^2.
    eff_area = 0.25 * np.pi * galsim.wfirst.diameter**2 * (1. - galsim.wfirst.obscuration**2)
    sky_val *= eff_area
    # Then multiply by pixel area in arcsec^2.
    sky_val *= galsim.wfirst.pixel_scale**2
    # Optionally multiply by exposure time (if it was given).
    if exp_time is not None:
        sky_val *= exp_time

    # The result is now in e-/pix (if exposure time was given), or e-/pix/s (if exposure time was
    # not given).
    return sky_val
