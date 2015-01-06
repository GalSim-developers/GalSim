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
@file wfirst_wcs.py

Part of the WFIRST module.  This file includes any routines needed to define and use the WFIRST WCS.
"""
import galsim
import galsim.wfirst
import numpy as np
import os

# Basic WFIRST reference info, with lengths in mm.
pixel_size_mm = 0.01
focal_length = 18500.
n_sip = 5 # Number of SIP coefficients used, where arrays are n_sip x n_sip in dimension

# Version-related information, for reference back to material provided by Jeff Kruk.
optics_design_ver = "4.2.2"
prog_version = "0.2"

# Information about center points of the SCAs in the WFI focal plane coordinate system (f1, f2)
# coordinates.  These are rotated by an angle theta_fpa with respect to the payload axes, as
# projected onto the sky. Note that the origin is at the center of the payload axes, on the
# telescope boresight, not centered on the FPA.  Since the SCAs are 1-indexed, these arrays have a
# non-used entry with index 0.  i.e., the maximum SCA is 18, and its value is in sca_xc_mm[18].
# Units are mm.
sca_xc_mm = np.array([0., -21.690, -21.690, -21.690, -65.070, -65.070, -65.070, -108.450, -108.450,
                      -108.450,  21.690,  21.690,  21.690,  65.070,  65.070,  65.070, 108.450,
                      108.450, 108.450])
sca_yc_mm = np.array([0., 199.250, 149.824, 242.630, 188.250, 138.824, 231.630, 164.923, 115.498,
                      208.304, 199.250, 149.824, 242.630, 188.250, 138.824, 231.630, 164.923,
                      115.498, 208.304])
# Nominal center of FPA in this coordinate frame, in mm and as an angle.
fpa_xc_mm = 0.0
fpa_yc_mm = 199.250
xc_fpa = np.arctan(fpa_xc_mm/focal_length)*galsim.radians
yc_fpa = np.arctan(fpa_yc_mm/focal_length)*galsim.radians

# The next array contains rotation offsets of individual SCA Y axis relative to FPA f2 axis. Same
# sign convention as theta_fpa. These represent mechanical installation deviations from perfect
# alignment and are ideally zero. These will be measured during focal plane integration and
# testing.
sca_rot = np.zeros_like(sca_xc_mm)

# Rotation of focal plane axes relative to payload axes: positive CCW rotation of the f2 axis
# relative to -Y_pl, and of f1 relative to +X_pl.
theta_fpa = 32.5*galsim.degrees

# File with SIP coefficients.
sip_filename = os.path.join(galsim.meta_data.share_dir, 'sip_422.txt')

def getWCS(PA, ra=None, dec=None, pos=None, PA_is_FPA=False, as_header=False):
    """
    This routine gets a list of WCS, one for each of the WFIRST SCAs.  Since the WFIRST SCAs are
    labeled 1-18, the zeroth list item is simply None.  Use of this routine requires that GalSim be
    able to access some software that can handle TAN-SIP style WCS (either Astropy, starlink.Ast,
    WCSTools).

    The user must specify a position for the center of the focal plane array (either as (ra, dec),
    or a CelestialCoord `pos`) and the orientation.

    Some notes on coordinate systems for WFIRST.  There are three coordinate systems that are used,
    all as projected onto the sky.

    Observatory coordinate system: +X_obs points along the boresight into the sky, +Z_obs points
    towards the Sun in the absence of a roll offset, +Y_obs makes a right-handed system

    Payload coordinate system: +X_pl points along -Y_obs, +Y_pl points along +Z_obs, +Z_pl points
    along -X_obs (back towards observer).

    WFI focal plane array (FPA) coordinate system: This is defined by a left-handed system f1, f2,
    that is rotated by an angle `theta_fpa` with respect to the payload axes.  `theta_fpa` is a
    positive CCW rotation of the f2 axis relative to -Y_pl, and of f1 relative to +X_pl.  In terms
    of focal plane geometry, if +Y_fp is pointing North, then SCAs 3 and 12 will be at highest
    declination, 8 and 17 at the lowest.  +Y_fp is aligned with the short axis of the focal plane
    array.

    So, for reference, if the boresight is pointed at RA=90, DEC=0 on March 21st (Sun at vernal
    equinox), then  +X_obs points at (RA,DEC)=(90,0), +Y_obs points North, and +Z_obs points at the
    Sun.  The payload coordinates are +X_pl points South, -Y_pl points East.  Finally, the FPA
    coordinate system is defined by +f2 being at a position angle 90+theta_fpa east of North.  If
    the observatory +Y axis is at a position angle `pa_obsy` East of North, then the focal plane
    (+f2) is at a position angle pa_fpa = pa_obsy + 90 + theta_fpa.

    @param PA        Position angle of the observatory Y axis, unless `PA_is_FPA=True`, in which
                     case it's the position angle of the FPA.  Must be provided as a galsim.Angle.
    @param ra        Right ascension of the center of the FPA, as a galsim.Angle.  Must be provided
                     with `dec`.
    @param dec       Declination of the center of the FPA, as a galsim.Angle.  Must be provided
                     with `ra`.
    @param pos       A galsim.CelestialCoord indicating the center of the FPA, as an alternative to
                     providing `ra` and `dec`.
    @param PA_is_FPA If True, then the position angle that was provided was the PA of the focal
                     plane array, not the observatory. [default: False]
    @param as_header If True, then instead of returning a list of WCS objects, return a list of
                     galsim.FitsHeader objects defining the WCS.
    @returns a list of WCS or FitsHeader objects for each SCA.
    """
    # Enforce type for PA
    if not isinstance(PA, galsim.Angle):
        raise TypeError("Position angle must be a galsim.Angle!")

    # Parse input position
    use_ra, use_dec = _parse_input_position(ra, dec, pos)

    # Compute position angle of FPA f2 axis, where positive corresponds to the angle east of North.
    if PA_is_FPA:
        pa_fpa = PA
        pa_obsy = PA - 90.*galsim.degrees - theta_fpa
    else:
        pa_obsy = PA
        pa_fpa = PA + 90.*galsim.degrees + theta_fpa
    cos_pa = np.cos(pa_fpa.rad())
    sin_pa = np.sin(pa_fpa.rad())

    # Figure out tangent-plane positions for FPA center:
    xc_fpa_tp, yc_fpa_tp = _det_to_tangplane_positions(xc_fpa, yc_fpa)

    # Note, this routine reads in the coeffs.
    a_sip, b_sip = _parse_sip_file(sip_filename)

    # Loop over SCAs:
    wcs_list = []
    for i_sca in range(galsim.wfirst.n_sca+1):
        if i_sca == 0:
            wcs_list.append(None)
            continue

        # Set up the header.
        header = {}
        header = galsim.FitsHeader(header=header)
        # Populate some necessary variables in the FITS header that are always the same, regardless of
        # input and SCA number.
        _populate_required_fields(header)

        # And populate some things that just depend on the overall locations or other input, not on
        # the SCA.
        header['RA_TARG'] = (use_ra / galsim.degrees, "right ascension of the target (deg) (J2000)")
        header['DEC_TARG'] = (use_dec / galsim.degrees, "declination of the target (deg) (J2000)")
        header['PA_OBSY'] = (pa_obsy / galsim.degrees, "position angle of observatory Y axis (deg)")
        header['PA_FPA'] = (pa_fpa / galsim.degrees, "position angle of FPA Y axis (deg)")

        # Finally do all the SCA-specific stuff.
        header['SCA_NUM'] = (i_sca, "SCA number (1 - 18)")

        # Set the position of center of this SCA in focal plane angular coordinates.
        sca_xc_fpa = np.arctan(sca_xc_mm[i_sca]/focal_length)*galsim.radians
        sca_yc_fpa = np.arctan(sca_yc_mm[i_sca]/focal_length)*galsim.radians

        # Figure out tangent plane positions after distortion, and subtract off those for FPA center
        # (calculated in header).
        sca_xc_tp, sca_yc_tp = _det_to_tangplane_positions(sca_xc_fpa, sca_yc_fpa)
        sca_xc_tp_f = sca_xc_tp - xc_fpa_tp
        sca_yc_tp_f = sca_yc_tp - yc_fpa_tp

        # Convert to polar coordinates in tangent plane: 
        # Define theta, phi as in Calabretta & Greisen 2002 A&A 395 1077, fig 3
        # and eqns 12-15
        rtheta = np.sqrt(sca_xc_tp_f.rad()**2 + sca_yc_tp_f.rad()**2)*galsim.radians
        theta = np.arctan(1./rtheta.rad())*galsim.radians
        phi = np.arctan2(sca_xc_tp_f.rad(), -sca_yc_tp_f.rad())*galsim.radians
        phi_p = 180.*galsim.degrees - pa_fpa
        cos_delta_phi = np.cos(phi.rad() - phi_p.rad())
        sin_delta_phi = np.sin(phi.rad() - phi_p.rad())
        cos_theta = np.cos(theta.rad())
        sin_theta = np.sin(theta.rad())
        cos_dec = np.cos(use_dec.rad())
        sin_dec = np.sin(use_dec.rad())

        # Rotate sca_xc_tp[i_sca], sca_yc_tp[i_sca] by pa_fpa
        # Add pos_targ when implemented
        crval1 = np.arctan2(-sin_delta_phi*cos_theta,
                             sin_theta*cos_dec - cos_theta*sin_dec*cos_delta_phi)*galsim.radians
        crval1 += use_ra
        header['CRVAL1'] = (crval1 / galsim.degrees, "first axis value at reference pixel")
        crval2 = np.arcsin(sin_theta*sin_dec + cos_theta*cos_dec*cos_delta_phi)*galsim.radians
        header['CRVAL2'] = (crval2 / galsim.degrees, "second axis value at reference pixel")

        # Compute CD coefficients: extract the linear terms from the a_sip, b_sip arrays.
        a10 = a_sip[i_sca,1,0]
        a11 = a_sip[i_sca,0,1]
        b10 = b_sip[i_sca,1,0]
        b11 = b_sip[i_sca,0,1]

        # Rotate by pa_fpa, plus rotational offsets of each SCA.
        pa_tot = pa_fpa + sca_rot[i_sca]*galsim.degrees
        cos_pa = np.cos(pa_tot.rad())
        sin_pa = np.sin(pa_tot.rad())
        header['CD1_1'] = (cos_pa * a10 + sin_pa * b10, "partial of first axis coordinate w.r.t. x")
        header['CD1_2'] = (cos_pa * a11 + sin_pa * b11, "partial of first axis coordinate w.r.t. y")
        header['CD2_1'] = (-sin_pa * a10 + cos_pa * b10, "partial of second axis coordinate w.r.t. x")
        header['CD2_2'] = (-sin_pa * a11 + cos_pa * b11, "partial of second axis coordinate w.r.t. y")

        for i in range(n_sip):
            for j in range(n_sip):
                if i+j >= 2 and i+j < n_sip:
                    sipstr = "A_%d_%d"%(i,j)
                    header[sipstr] = a_sip[i_sca,i,j]
                    sipstr = "B_%d_%d"%(i,j)
                    header[sipstr] = b_sip[i_sca,i,j]

        if not as_header:
            # Make a WCS object.  If we let it try all WCS types, then it will claim to work for
            # AstropyWCS, but it will throw exceptions when doing routine calculations.  Since we
            # know we have a TAN-SIP WCS we restrict the list of WCS to try to those that we know
            # should work for TAN-SIP.
            galsim.wcs.fits_wcs_types = [galsim.PyAstWCS, galsim.WcsToolsWCS]
            wcs = galsim.FitsWCS(header=header)
            wcs_list.append(wcs)
        else:
            wcs_list.append(header)

    return wcs_list

def findSCA(wcs_list, ra=None, dec=None, pos=None, include_border=False):
    """
    This is a subroutine to take a list of WCS (one per SCA) from galsim.wfirst.getWCS() and query
    which SCA a particular real-world coordinate would be located on.  If the position is not
    located on any of the SCAs, the result will be None.

    The position should be specified in a way similar to how it is specified for
    galsim.wfirst.getWCS().

    Depending on what the use wants to do with the results, they may wish to use the
    `include_border` keyword.  This keyword determines whether or not to include an additional
    border corresponding to half of the gaps between SCAs.  For example, if a user is drawing a
    single image they may wish to only know whether a given position falls onto an SCA, and if so,
    which one (ignoring everything in the gaps).  In contrast, a user who plans to make a sequence
    of dithered images might find it most useful to know whether the position is either on an SCA or
    close enough that in a typical dither sequence it might appear on the SCA at some point.  Use of
    `include_border` switches between these scenarios.

    @param ra               Right ascension of the sky position of interest, as a galsim.Angle.
                            Must be provided with `dec`.
    @param dec              Declination of the sky position of interest, as a galsim.Angle.  Must be
                            provided with `ra`.
    @param pos              A galsim.CelestialCoord indicating the sky position of interest, as an
                            alternative to providing `ra` and `dec`.
    @param include_border   If True, then include the half-border around SCA to cover the gap
                            between each sensor. [default: False]
    @returns an integer value of the SCA on which the position falls, or None if the position is not
             on any SCA.

    """
    if len(wcs_list) != galsim.wfirst.n_sca+1:
        raise ValueError("wcs_list should be a list of length %d output by"
                         " galsim.wfirst.getWCS!"%(galsim.wfirst.n_sca+1))

    # Parse input position
    use_ra, use_dec = _parse_input_position(ra, dec, pos)

    # Set up the minimum and maximum pixel values, depending on whether or not to include the
    # border.
    min_x_pix, max_x_pix, min_y_pix, max_y_pix = _calculate_minmax_pix(include_border)

    sca = None
    for i_sca in range(1, galsim.wfirst.n_sca+1):
        wcs = wcs_list[i_sca]
        image_pos = wcs.toImage(galsim.CelestialCoord(use_ra, use_dec))
        if image_pos.x >= min_x_pix[i_sca] and image_pos.x <= max_x_pix[i_sca] and \
                image_pos.y >= min_y_pix[i_sca] and image_pos.y <= max_y_pix[i_sca]:
            sca = i_sca
            break

    return sca

def _calculate_minmax_pix(include_border=False):
    """
    This is a helper routine to calculate the minimum and maximum pixel values that should be
    considered within an SCA, possibly including the complexities of including 1/2 of the gap
    between SCAs.  In that case it depends on the detailed geometry of the WFIRST focal plane.

    @param include_border   A boolean value that determines whether to include 1/2 of the gap
                            between SCAs as part of the SCA itself.  [default: False]
    @returns a tuple of NumPy arrays for the minimum x pixel value, maximum x pixel value, minimum y
             pixel value, and maximum y pixel value for each SCA.
    """
    # First, set up the default (no border).
    # The minimum and maximum pixel values are (1, n_pix).
    min_x_pix = np.ones(galsim.wfirst.n_sca+1)
    max_x_pix = min_x_pix + galsim.wfirst.n_pix - 1.0
    min_y_pix = min_x_pix.copy()
    max_y_pix = max_x_pix.copy()

    # Then, calculate the half-gaps, grouping together SCAs whenever possible.
    if include_border:
        # Currently, the configuration in the focal plane is such that all the horizontal chip gaps
        # are the same, but that won't always be the case, so for the sake of generality we only
        # group together those that are forced to be the same.
        #
        # Negative side of 1/2/3, same as positive side of 10/11/12
        border_mm = abs(sca_xc_mm[1]-sca_xc_mm[10])-galsim.wfirst.n_pix*pixel_size_mm
        half_border_pix = int(0.5*border_mm / pixel_size_mm)
        min_x_pix[1:4] -= half_border_pix
        max_x_pix[10:13] += half_border_pix

        # Positive side of 1/2/3 and 13/14/15, same as negative side of 10/11/12, 4/5/6
        border_mm = abs(sca_xc_mm[1]-sca_xc_mm[4])-galsim.wfirst.n_pix*pixel_size_mm
        half_border_pix = int(0.5*border_mm / pixel_size_mm)
        max_x_pix[1:4] += half_border_pix
        max_x_pix[13:16] += half_border_pix
        min_x_pix[10:13] -= half_border_pix
        min_x_pix[4:7] -= half_border_pix

        # Positive side of 4/5/6, 16/17/18, 7/8/9, same as negative side of 13/14/15, 7/8/9,
        # 16/17/18
        border_mm = abs(sca_xc_mm[7]-sca_xc_mm[4])-galsim.wfirst.n_pix*pixel_size_mm
        half_border_pix = int(0.5*border_mm / pixel_size_mm)
        max_x_pix[4:10] += half_border_pix
        max_x_pix[16:19] += half_border_pix
        min_x_pix[7:10] -= half_border_pix
        min_x_pix[13:19] -= half_border_pix

        # In the vertical direction, the gaps vary, with the gap between one pair of rows being
        # significantly larger than between the other pair of rows.  The reason for this has to do
        # with asymmetries in the electronics that stick out from the top and bottom of the SCAs,
        # and choices in which way to arrange each SCA to maximize the usable space in the focal
        # plane.

        # Top of 2/5/8/11/14/17, same as bottom of 1/4/7/10/13/16 and 2/5/8/11/14/17
        border_mm = abs(sca_yc_mm[1]-sca_yc_mm[2])-galsim.wfirst.n_pix*pixel_size_mm
        half_border_pix = int(0.5*border_mm / pixel_size_mm)
        list_1 = np.linspace(1,16,6).astype(int)
        list_2 = list_1 + 1
        list_3 = list_1 + 2
        min_y_pix[list_1] -= half_border_pix
        min_y_pix[list_2] -= half_border_pix
        max_y_pix[list_2] += half_border_pix

        # Top of 1/4/7/10/13/16, same as bottom of 3/6/9/12/15/18 and top of same
        border_mm = abs(sca_yc_mm[1]-sca_yc_mm[3])-galsim.wfirst.n_pix*pixel_size_mm
        half_border_pix = int(0.5*border_mm / pixel_size_mm)
        min_y_pix[list_3] -= half_border_pix
        max_y_pix[list_1] += half_border_pix
        max_y_pix[list_3] += half_border_pix

    return min_x_pix, max_x_pix, min_y_pix, max_y_pix

def _parse_input_position(ra, dec, pos):
    if ra is not None:
        if dec is None:
            raise ValueError("Must provide (RA, dec) pair!")
        if not isinstance(ra, galsim.Angle) or not isinstance(dec, galsim.Angle):
            raise TypeError("(RA, dec) pair must be galsim.Angles")
        if pos is not None:
            raise ValueError("Can provide either pos or (RA, dec), not both!")
    if pos is not None:
        if ra is not None or dec is not None:
            raise ValueError("Can provide either pos or (RA, dec), not both!")
        ra = pos.ra
        dec = pos.dec
    if ra is None:
        raise ValueError("Must provide either pos or (RA, dec)!")
    return ra, dec

def _populate_required_fields(header):
    """
    Utility routine to do some of the basics for the WCS headers for WFIRST that don't require any
    interesting calculation.
    """
    header['EQUINOX'] = (2000.0, "equinox of celestial coordinate system")
    header['WCSAXES'] = (2, "number of World Coordinate System axes")
    header['A_ORDER'] = 4
    header['B_ORDER'] = 4
    header['WCSNAME'] = 'wfiwcs_'+optics_design_ver+'_'+prog_version
    header['CRPIX1'] = (galsim.wfirst.n_pix/2, "x-coordinate of reference pixel")
    header['CRPIX2'] = (galsim.wfirst.n_pix/2, "y-coordinate of reference pixel")
    header['CTYPE1'] = ("RA---TAN-SIP", "coordinate type for the first axis")
    header['CTYPE2'] = ("DEC--TAN-SIP", "coordinate type for the second axis")
    header['SIMPLE'] = 'True'
    header['BITPIX'] = 16
    header['NAXIS'] = 0
    header['EXTEND'] = 'True'
    header['BZERO'] = 32768
    header['BSCALE'] = 1

def _parse_sip_file(file):
    """
    Utility routine to parse the file with the SIP coefficients and hand back some arrays to be used
    for later calculations.
    """
    if not os.path.exists(file):
        raise RuntimeError("Error, cannot find file that should have WFIRST SIP"
                           " coefficients: %s"%file)

    # Parse the file, which comes from wfi_wcs_sip_gen_0.1.c provided by Jeff Kruk.
    data = np.loadtxt(file, usecols=[0, 3, 4, 5, 6, 7]).transpose()

    a_sip = np.zeros((galsim.wfirst.n_sca+1, n_sip, n_sip))
    b_sip = np.zeros((galsim.wfirst.n_sca+1, n_sip, n_sip))
    for i_sca in range(1, galsim.wfirst.n_sca+1):
        i_sca_m1 = i_sca - 1
        # Take the data for this SCA
        use_data = data[:, data[0,:]==i_sca_m1]
        # Split it into a and b-type coefficients
        a_data = use_data[:, 0:n_sip]
        b_data = use_data[:, n_sip:]
        # Assign the data to our master array of coefficients
        a_sip[i_sca,:,:] = a_data[1:,:].transpose()
        b_sip[i_sca,:,:] = b_data[1:,:].transpose()

    return a_sip, b_sip

def _det_to_tangplane_positions(x_in, y_in):
    """
    Helper routine to convert (x_in, y_in) focal plane coordinates to tangent plane coordinates
    (x_out, y_out).  If (x_in, y_in) are measured focal plane positions of an object, with the
    origin at the telescope boresight, then we can define a radius as

        r = sqrt(x_in^2 + y_in^2)

    The optical distortion model relies on the following definitions:

        dist = a2*r^2 + a1*r + a0

    with true (tangent plane) coordinates given by

        (x_out, y_out) = (x_in, y_in)/(1 + dist).

    Note that the coefficients given in this routine go in the order {a0, a1, a2}.

    """
    img_dist_coeff = np.array([-1.4976e-02, 3.7053e-03, 3.0186e-02])
    if not isinstance(x_in, galsim.Angle) or not isinstance(y_in, galsim.Angle):
        raise ValueError("Input x_in and y_in are not galsim.Angles.")
    # The optical distortion model is defined in terms of separations in *degrees*.
    r_sq = (x_in/galsim.degrees)**2 + (y_in/galsim.degrees)**2
    r = np.sqrt(r_sq)
    dist_fac = 1. + img_dist_coeff[0] + img_dist_coeff[1]*r + img_dist_coeff[2]*r_sq
    return x_in/dist_fac, y_in/dist_fac
