# Copyright (c) 2012-2023 by the GalSim developers team on GitHub
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
@file roman_wcs.py

Part of the Roman Space Telescope module.  This file includes any routines needed to define and use
the Roman WCS.  Current version is consistent with Roman wide-field channel optical design version
7.6.8, generated during Phase A and presented at the Roman System Requirements Review and Mission
Definition Review.
"""
import numpy as np
import os
import coord
import datetime

from . import n_sca, n_pix, max_sun_angle
from .. import meta_data
from .. import GSFitsWCS, FitsHeader
from .. import PositionD
from .. import BoundsI
from .. import GalSimRangeError, GalSimError


# Basic Roman reference info, with lengths in mm.
pixel_size_mm = 0.01
focal_length = 18714
pix_scale = (pixel_size_mm/focal_length)*coord.radians
n_sip = 5 # Number of SIP coefficients used, where arrays are n_sip x n_sip in dimension

# Version-related information, for reference back to material provided by Jeff Kruk.
tel_name = "Roman"
instr_name = "WFC"
optics_design_ver = "20210204"
prog_version = "d2"

# Information about center points of the SCAs in the WFI focal plane coordinate system (f1, f2)
# coordinates.  These are rotated by an angle theta_fpa with respect to the payload axes, as
# projected onto the sky.  The origin is centered on the telescope boresight, but can be related to
# the center of the FPA by subtracting fpa_xc_mm and fpa_yc_mm.
#
# Since the SCAs are 1-indexed, these arrays have a non-used entry with index 0.  i.e., the maximum
# SCA is 18, and its value is in sca_xc_mm[18].  Units are mm.
infile = os.path.join(meta_data.share_dir, 'roman', 'sca_positions_20210204.txt')
sca_data = np.loadtxt(infile).transpose()
sca_xc_mm = sca_data[3,:]
sca_yc_mm = sca_data[4,:]
sca_xc_mm = np.insert(sca_xc_mm, 0, 0)
sca_yc_mm = np.insert(sca_yc_mm, 0, 0)
sca_crval_u_deg = sca_data[5,:]
sca_crval_v_deg = sca_data[6,:]
sca_crval_u_deg = np.insert(sca_crval_u_deg, 0, 0)
sca_crval_v_deg = np.insert(sca_crval_v_deg, 0, 0)
# Nominal center of FPA from the payload axis in this coordinate system, in mm and as an angle
# (neglecting distortions - to be included later).
fpa_xc_mm = 0.0
fpa_yc_mm = 160.484
xc_fpa = np.arctan(fpa_xc_mm/focal_length)*coord.radians
yc_fpa = np.arctan(fpa_yc_mm/focal_length)*coord.radians

# The next array contains rotation offsets of individual SCA Y axis relative to FPA f2 axis. Same
# sign convention as theta_fpa. These represent mechanical installation deviations from perfect
# alignment and are ideally zero. These will be measured during focal plane integration and
# testing.
sca_rot = np.zeros_like(sca_xc_mm)

# Rotation of WFI local axes relative to payload axes: this is expressed as a CCW rotation
# relative to observatory +Z direction.
theta_fpa = 120.0*coord.degrees

# File with SIP coefficients.
sip_filename = os.path.join(meta_data.share_dir, 'roman', 'sip_20210204.txt')

def getWCS(world_pos, PA=None, date=None, SCAs=None, PA_is_FPA=False):
    """
    This routine returns a dict containing a WCS for each of the Roman SCAs (Sensor Chip Array, the
    equivalent of a chip in an optical CCD).  The Roman SCAs are labeled 1-18, so these numbers are
    used as the keys in the dict.  Alternatively the user can request a subset of the SCAs using the
    ``SCAs`` option.  The basic instrument parameters used to create the WCS correspond to those in
    Cycle 6, which includes some significant updates from Cycle 5, including a 90 degree rotation of
    the focal plane axes relative to the payload axes, and two rows of SCAs are swapped.

    The user must specify a position for observation, at which the center of the focal plane array
    will point.  This must be supplied as a CelestialCoord ``world_pos``.  In general, only certain
    positions are observable on certain dates, and for a given position there is an optimal position
    angle for the observatory (with the solar panels pointed as directly towards the sun as
    possible).  Users who are knowledgable about these details may choose to supply a position angle
    as ``PA``, either for the observatory or for the focal plane (using ``PA_is_FPA`` to indicate
    this).  But otherwise, the routine will simply choose the optimal position angle for a given
    date.

    To fully understand all possible inputs and outputs to this routine, users may wish to consult
    the diagram on the GalSim wiki,
    https://github.com/GalSim-developers/GalSim/wiki/GalSim-Roman-module-diagrams

    Parameters:
        world_pos:      A `galsim.CelestialCoord` indicating the position to observe at the center
                        of the focal plane array (FPA).  Note that if the given position is not
                        observable on the given date, then the routine will raise an exception.
        PA:             A `galsim.Angle` representing the position angle of the observatory +Y
                        axis, unless ``PA_is_FPA=True``, in which case it's the position angle of
                        the FPA.  For users to do not care about this, then leaving this as None
                        will result in the routine using the supplied ``date`` and ``world_pos`` to
                        select the optimal orientation for the observatory.  Note that if a user
                        supplies a ``PA`` value, the routine does not check whether this orientation
                        is actually allowed.  [default: None]
        date:           The date of the observation, as a python datetime object.  If None, then the
                        vernal equinox in 2025 will be used.  [default: None]
        PA_is_FPA:      If True, then the position angle that was provided was the PA of the focal
                        plane array, not the observatory. [default: False]
        SCAs:           A single number or iterable giving the SCAs for which the WCS should be
                        obtained.  If None, then the WCS is calculated for all SCAs.
                        [default: None]

    Returns:
        A dict of WCS objects for each SCA.
    """
    # First just parse the input quantities.
    date, SCAs, pa_fpa, pa_obsy = _parse_WCS_inputs(world_pos, PA, date, PA_is_FPA, SCAs)

    # Further gory details on coordinate systems, for developers: Observatory coordinate system is
    # defined such that +X_obs points along the boresight into the sky, +Z_obs points towards the
    # Sun in the absence of a roll offset (i.e., roll offset = 0 defines the optimal position angle
    # for the observatory), +Y_obs makes a right-handed system.
    #
    # The x,y axes of each SCA are shown in the figure mapping_v210503.pdf in the devel/roman
    # directory.  Some are 180 rotated with respect to others.
    # The data in sip_filename give the coordinate transformation from each SCA's image x,y
    # coordinates to the WFI Local coordinate system.  +Y in this system points away from the
    # center of the observatory.  And +X is to the right if +Y is up.
    #
    # These coordinates are rotated by an angle theta_fpa CCW from observatory +Z.

    # Note, this routine reads in the coeffs.  We don't use them until later, but read them in for
    # all SCAs at once.
    a_sip, b_sip = _parse_sip_file(sip_filename)

    # Loop over SCAs:
    wcs_dict = {}
    for i_sca in SCAs:
        # Set up the header.
        header = []
        # Populate some necessary variables in the FITS header that are always the same, regardless of
        # input and SCA number.
        _populate_required_fields(header)

        # And populate some things that just depend on the overall locations or other input, not on
        # the SCA.
        header.extend([
            ('RA_TARG', world_pos.ra / coord.degrees,
                        "right ascension of the target (deg) (J2000)"),
            ('DEC_TARG', world_pos.dec / coord.degrees,
                         "declination of the target (deg) (J2000)"),
            ('PA_OBSY', pa_obsy / coord.degrees, "position angle of observatory Y axis (deg)"),
            ('PA_FPA', pa_fpa / coord.degrees, "position angle of FPA Y axis (deg)"),
            ('SCA_NUM', i_sca, "SCA number (1 - 18)"),
        ])

        # Leave phi_p at 180 (0 if dec_targ==-90), so that tangent plane axes remain oriented along
        # celestial coordinates. In other words, phi_p is the angle of the +Y axis in the tangent
        # plane, which is of course pi if we're measuring these phi angles clockwise from the -Y
        # axis.  Note that this quantity is not used in any calculations at all, but for consistency
        # with the WCS code that comes from the Roman project office, we calculate this quantity
        # and put it in the FITS header.
        if world_pos.dec / coord.degrees > -90.:
            phi_p = np.pi*coord.radians
        else:
            phi_p = 0.*coord.radians

        # Get position of SCA center given the center of the FPA and the orientation angle of the
        # focal plane.
        crval, u, v = _get_sca_center_pos(i_sca, world_pos, pa_fpa)

        # Compute the position angle of the local pixel Y axis.
        # This requires projecting local North onto the detector axes.
        # Start by adding any SCA-unique rotation relative to FPA axes:
        sca_tp_rot = pa_fpa + sca_rot[i_sca]*coord.degrees

        # Go some reasonable distance from crval in the +y direction.  Say, 1 degree.
        plus_y = world_pos.deproject(u, v + 1*coord.degrees, projection='gnomonic')
        # Find the angle between this point, crval and due north.
        north = coord.CelestialCoord(0.*coord.degrees, 90.*coord.degrees)
        pa_sca = sca_tp_rot - crval.angleBetween(plus_y, north)

        # Compute CD coefficients: extract the linear terms from the a_sip, b_sip arrays.  These
        # linear terms are stored in the SIP arrays for convenience.
        a10 = a_sip[i_sca,1,0]
        a11 = a_sip[i_sca,0,1]
        b10 = b_sip[i_sca,1,0]
        b11 = b_sip[i_sca,0,1]

        # Rotate by pa_fpa.
        cos_pa_sca = np.cos(pa_sca)
        sin_pa_sca = np.sin(pa_sca)

        header.extend([
            ('CRVAL1', crval.ra / coord.degrees, "first axis value at reference pixel"),
            ('CRVAL2', crval.dec / coord.degrees, "second axis value at reference pixel"),
            ('CD1_1', cos_pa_sca * a10 + sin_pa_sca * b10,
                      "partial of first axis coordinate w.r.t. x"),
            ('CD1_2', cos_pa_sca * a11 + sin_pa_sca * b11,
                      "partial of first axis coordinate w.r.t. y"),
            ('CD2_1', -sin_pa_sca * a10 + cos_pa_sca * b10,
                      "partial of second axis coordinate w.r.t. x"),
            ('CD2_2', -sin_pa_sca * a11 + cos_pa_sca * b11,
                      "partial of second axis coordinate w.r.t. y"),
            ('ORIENTAT', pa_sca / coord.degrees, "position angle of image y axis (deg. e of n)"),
            ('LONPOLE', phi_p / coord.degrees, "Native longitude of celestial pole"),
        ])
        for i in range(n_sip):
            for j in range(n_sip):
                if i+j >= 2 and i+j < n_sip:
                    sipstr = "A_%d_%d"%(i,j)
                    header.append( (sipstr, a_sip[i_sca,i,j]) )
                    sipstr = "B_%d_%d"%(i,j)
                    header.append( (sipstr,  b_sip[i_sca,i,j]) )

        header = FitsHeader(header)
        wcs = GSFitsWCS(header=header)
        # Store the original header as an attribute of the WCS.  This ensures that we have all the
        # extra keywords for whenever an image with this WCS is written to file.
        wcs.header = header
        wcs_dict[i_sca]=wcs

    return wcs_dict

def convertCenter(world_pos, SCA, PA=None, date=None, PA_is_FPA=False, tol=0.5*coord.arcsec):
    """
    This is a simple helper routine that takes an input position ``world_pos`` that is meant to
    correspond to the position of the center of an SCA, and tells where the center of the focal
    plane array should be.  The goal is to provide a position that can be used as an input to
    getWCS(), which wants the center of the focal plane array.

    The results of the calculation are deterministic if given a fixed position angle (PA).  If it's
    not given one, it will try to determine the best one for this location and date, like getWCS()
    does.

    Because of distortions varying across the focal plane, this routine has to iteratively correct
    its initial result based on empirical tests.  The ``tol`` kwarg can be used to adjust how
    careful it will be, but it always does at least one iteration.

    To fully understand all possible inputs and outputs to this routine, users may wish to consult
    the diagram on the GalSim wiki,
    https://github.com/GalSim-developers/GalSim/wiki/GalSim-Roman-module-diagrams

    Parameters:
        world_pos:  A galsim.CelestialCoord indicating the position to observe at the center of the
                    given SCA.  Note that if the given position is not observable on
                    the given date, then the routine will raise an exception.
        SCA:        A single number giving the SCA for which the center should be located at
                    ``world_pos``.
        PA:         galsim.Angle representing the position angle of the observatory +Y axis, unless
                    ``PA_is_FPA=True``, in which case it's the position angle of the FPA.  For
                    users to do not care about this, then leaving this as None will result in the
                    routine using the supplied ``date`` and ``world_pos`` to select the optimal
                    orientation for the observatory.  Note that if a user supplies a ``PA`` value,
                    the routine does not check whether this orientation is actually allowed.
                    [default: None]
        date:       The date of the observation, as a python datetime object.  If None, then the
                    vernal equinox in 2025 will be used.  [default: None]
        PA_is_FPA:  If True, then the position angle that was provided was the PA of the focal
                    plane array, not the observatory. [default: False]
        tol:        Tolerance for errors due to distortions, as a galsim.Angle.
                    [default: 0.5*galsim.arcsec]

    Returns:
        A CelestialCoord object indicating the center of the focal plane array.
    """
    if not isinstance(SCA, int):
        raise TypeError("Must pass in an int corresponding to the SCA")
    if not isinstance(tol, coord.Angle):
        raise TypeError("tol must be a galsim.Angle")
    use_SCA = SCA
    # Parse inputs appropriately.
    _, _, pa_fpa, _ = _parse_WCS_inputs(world_pos, PA, date, PA_is_FPA, [SCA])

    # Now pretend world_pos was the FPA center and we want to find the location of this SCA:
    _, u, v = _get_sca_center_pos(use_SCA, world_pos, pa_fpa)
    # The (u, v) values give an offset, and we can invert this.
    fpa_cent = world_pos.deproject(-u, -v, projection='gnomonic')
    # This is only approximately correct, especially for detectors that are far from the center of
    # the FPA, because of distortions etc.  We can do an iterative correction.
    # For the default value of 'tol', typically just 1-2 iterations are needed.
    shift_val = 1000.0 # arcsec
    while shift_val > tol/coord.arcsec:
        test_wcs = getWCS(fpa_cent, PA, date, use_SCA, PA_is_FPA)[use_SCA]
        im_cent_pos = PositionD(n_pix/2, n_pix/2)
        test_sca_pos = test_wcs.toWorld(im_cent_pos)
        delta_ra = np.cos(world_pos.dec)*(world_pos.ra-test_sca_pos.ra)
        delta_dec = world_pos.dec-test_sca_pos.dec
        shift_val = np.abs(world_pos.distanceTo(test_sca_pos)/coord.arcsec)
        fpa_cent = coord.CelestialCoord(fpa_cent.ra + delta_ra, fpa_cent.dec + delta_dec)

    return fpa_cent

def findSCA(wcs_dict, world_pos, include_border=False):
    """
    This is a subroutine to take a dict of WCS (one per SCA) from galsim.roman.getWCS() and query
    which SCA a particular real-world coordinate would be located on.  The position (``world_pos``)
    should be specified as a galsim.CelestialCoord.  If the position is not located on any of the
    SCAs, the result will be None.  Note that if ``wcs_dict`` does not include all SCAs in it, then
    it's possible the position might lie on one of the SCAs that was not included.

    Depending on what the user wants to do with the results, they may wish to use the
    ``include_border`` keyword.  This keyword determines whether or not to include an additional
    border corresponding to half of the gaps between SCAs.  For example, if a user is drawing a
    single image they may wish to only know whether a given position falls onto an SCA, and if so,
    which one (ignoring everything in the gaps).  In contrast, a user who plans to make a sequence
    of dithered images might find it most useful to know whether the position is either on an SCA or
    close enough that in a small dither sequence it might appear on the SCA at some point.  Use of
    ``include_border`` switches between these scenarios.

    Parameters:
        wcs_dict:        The dict of WCS's output from galsim.roman.getWCS().
        world_pos:       A galsim.CelestialCoord indicating the sky position of interest.
        include_border:  If True, then include the half-border around SCA to cover the gap
                         between each sensor. [default: False]

    Returns:
        an integer value of the SCA on which the position falls, or None if the position is not
        on any SCA.

    """
    # Sanity check args.
    if not isinstance(wcs_dict, dict):
        raise TypeError("wcs_dict should be a dict containing WCS output by galsim.roman.getWCS.")

    if not isinstance(world_pos, coord.CelestialCoord):
        raise TypeError("Position on the sky must be given as a galsim.CelestialCoord.")

    # Set up the minimum and maximum pixel values, depending on whether or not to include the
    # border.  We put it immediately into a galsim.BoundsI(), since the routine returns xmin, xmax,
    # ymin, ymax:
    xmin, xmax, ymin, ymax = _calculate_minmax_pix(include_border)
    bounds_list = [ BoundsI(x1,x2,y1,y2) for x1,x2,y1,y2 in zip(xmin,xmax,ymin,ymax) ]

    sca = None
    for i_sca in wcs_dict:
        wcs = wcs_dict[i_sca]
        image_pos = wcs.toImage(world_pos)
        if bounds_list[i_sca].includes(image_pos):
            sca = i_sca
            break

    return sca

def _calculate_minmax_pix(include_border=False):
    """
    This is a helper routine to calculate the minimum and maximum pixel values that should be
    considered within an SCA, possibly including the complexities of including 1/2 of the gap
    between SCAs.  In that case it depends on the detailed geometry of the Roman focal plane.

    Parameters:
        include_border:     A boolean value that determines whether to include 1/2 of the gap
                            between SCAs as part of the SCA itself.  [default: False]

    Returns:
        a tuple of NumPy arrays for the minimum x pixel value, maximum x pixel value, minimum y
        pixel value, and maximum y pixel value for each SCA.
    """
    # First, set up the default (no border).
    # The minimum and maximum pixel values are (1, n_pix).
    min_x_pix = np.ones(n_sca+1).astype(int)
    max_x_pix = min_x_pix + n_pix - 1
    min_y_pix = min_x_pix.copy()
    max_y_pix = max_x_pix.copy()

    # Then, calculate the half-gaps, grouping together SCAs whenever possible.
    if include_border:
        # Currently, the configuration in the focal plane is such that all the horizontal chip gaps
        # are the same, but that won't always be the case, so for the sake of generality we only
        # group together those that are forced to be the same.
        #
        # We figure out the borders based on the FPA coordinates, but when we apply the
        # adjustments to min/max x/y, we take into account the orientations of the SCAs.
        # cf. mapping_v210503.pdf in the devel/roman directory.
        #
        # Positive side of 1/2/3, same as negative side of 10/11/12
        border_mm = (sca_xc_mm[10]-sca_xc_mm[1])-n_pix*pixel_size_mm
        # assert statement help ensure that these pairings continue to work if positions
        # are updated in the future.
        assert 0 < border_mm < 5
        half_border_pix = int(0.5*border_mm / pixel_size_mm)
        max_x_pix[ [3,10,11] ] += half_border_pix
        min_x_pix[ [1,2,12] ] -= half_border_pix

        # Negative side of 1/2/3 and 13/14/15, same as positive side of 4/5/6 and 10/11/12
        border_mm = (sca_xc_mm[1]-sca_xc_mm[4])-n_pix*pixel_size_mm
        assert 0 < border_mm < 5
        half_border_pix = int(0.5*border_mm / pixel_size_mm)
        min_x_pix[ [3,4,5,10,11,15] ] -= half_border_pix
        max_x_pix[ [1,2,6,12,13,14] ] += half_border_pix

        # Negative side of 4/5/6 and 16/17/18, same as positive side of 7/8/9 and 13/14/15
        # Also add this same chip gap to the outside chips.  Neg side of 7/8/9, pos 16/17/18.
        border_mm = (sca_xc_mm[4]-sca_xc_mm[7])-n_pix*pixel_size_mm
        assert 0 < border_mm < 5
        half_border_pix = int(0.5*border_mm / pixel_size_mm)
        min_x_pix[ [6,7,8,13,14,18] ] -= half_border_pix
        max_x_pix[ [4,5,9,15,16,17] ] += half_border_pix

        # In the vertical direction, the gaps vary, with the gap between one pair of rows being
        # significantly larger than between the other pair of rows.  The reason for this has to do
        # with asymmetries in the electronics that stick out from the top and bottom of the SCAs,
        # and choices in which way to arrange each SCA to maximize the usable space in the focal
        # plane.

        # Top of 2/5/8/11/14/17, same as bottom of 1/4/7/10/13/16.
        # Also use this for top of top row: 1/4/7/10/13/16.
        border_mm = (sca_yc_mm[1]-sca_yc_mm[2])-n_pix*pixel_size_mm
        # One of the two vertical borders is larger.  Just test them both at <10.
        # (In the 20210204 setup, this one is the wider border.)
        assert border_mm < 10
        half_border_pix = int(0.5*border_mm / pixel_size_mm)
        list_1 = np.arange(1,18,3)
        list_2 = list_1 + 1
        list_3 = list_1 + 2
        max_y_pix[list_1] += half_border_pix
        min_y_pix[list_2] -= half_border_pix
        min_y_pix[list_1] -= half_border_pix

        # Top of 3/6/9/12/15/18, same as bottom of 2/5/8/11/14/17.
        # Also use this for bottom of bottom row: 3/6/9/12/15/18.
        border_mm = (sca_yc_mm[2]-sca_yc_mm[3])-n_pix*pixel_size_mm
        assert border_mm < 10
        half_border_pix = int(0.5*border_mm / pixel_size_mm)
        max_y_pix[list_2] += half_border_pix
        min_y_pix[list_3] -= half_border_pix
        max_y_pix[list_3] += half_border_pix

    return min_x_pix, max_x_pix, min_y_pix, max_y_pix

def _populate_required_fields(header):
    """
    Utility routine to do populate some of the basic fields for the WCS headers for Roman that
    don't require any interesting calculation.
    """
    header.extend([
        ('EQUINOX', 2000.0, "equinox of celestial coordinate system"),
        ('WCSAXES', 2, "number of World Coordinate System axes"),
        ('A_ORDER', 4),
        ('B_ORDER', 4),
        ('WCSNAME', 'wfiwcs_'+optics_design_ver+'_'+prog_version),
        ('CRPIX1', n_pix/2, "x-coordinate of reference pixel"),
        ('CRPIX2', n_pix/2, "y-coordinate of reference pixel"),
        ('CTYPE1', "RA---TAN-SIP", "coordinate type for the first axis"),
        ('CTYPE2', "DEC--TAN-SIP", "coordinate type for the second axis"),
        ('SIMPLE', 'True'),
        ('BITPIX', 16),
        ('NAXIS', 0),
        ('EXTEND', 'True'),
        ('BZERO', 0),
        ('BSCALE', 1),
        ('TELESCOP', tel_name, "telescope used to acquire data"),
        ('INSTRUME', instr_name, "identifier for instrument used to acquire data"),
    ])

def _parse_sip_file(file):
    """
    Utility routine to parse the file with the SIP coefficients and hand back some arrays to be used
    for later calculations.
    """
    if not os.path.exists(file):
        raise OSError("Cannot find file that should have Roman SIP coefficients: %s"%file)

    # Parse the file, generated by make_sip_file.py in devel/roman directory.
    data = np.loadtxt(file, usecols=[0, 3, 4, 5, 6, 7]).transpose()

    a_sip = np.zeros((n_sca+1, n_sip, n_sip))
    b_sip = np.zeros((n_sca+1, n_sip, n_sip))
    for i_sca in range(1, n_sca+1):
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

def _get_sca_center_pos(i_sca, world_pos, pa_fpa):
    """
    This helper routine calculates the center position for a given SCA ``sca`` given the position of
    the center of the focal plane array ``world_pos`` and an orientation angle for the observation.
    It is used by getWCS() and other routines.
    """
    # Go from the tangent plane position of the SCA center, to the actual celestial coordinate,
    # using `world_pos` as the center point of the tangent plane projection.  This celestial
    # coordinate for the SCA center is `crval`, which goes into the WCS as CRVAL1, CRVAL2.
    cos_pa = np.cos(pa_fpa)
    sin_pa = np.sin(pa_fpa)
    u = sca_crval_u_deg[i_sca] * cos_pa - sca_crval_v_deg[i_sca] * sin_pa
    v = sca_crval_u_deg[i_sca] * sin_pa + sca_crval_v_deg[i_sca] * cos_pa
    u = u * coord.degrees
    v = v * coord.degrees
    crval = world_pos.deproject(u, v, projection='gnomonic')
    return crval, u, v

def _parse_SCAs(SCAs):
    # This is a helper routine to parse the input SCAs (single number or iterable) and put it into a
    # convenient format.  It is used in roman_wcs.py.
    #
    # Check which SCAs are to be done.  Default is all (and they are 1-indexed).
    all_SCAs = np.arange(1, n_sca + 1, 1)
    # Later we will use the list of selected SCAs to decide which ones we're actually going to do
    # the calculations for.  For now, just check for invalid numbers.
    if SCAs is not None:
        # Make sure SCAs is iterable.
        if not hasattr(SCAs, '__iter__'):
            SCAs = [SCAs]
        # Then check for reasonable values.
        if min(SCAs) <= 0 or max(SCAs) > n_sca:
            raise GalSimRangeError("Invalid SCA.", SCAs, 1, n_sca)
        # Check for uniqueness.  If not unique, make it unique.
        SCAs = list(set(SCAs))
    else:
        SCAs = all_SCAs
    return SCAs

def _parse_WCS_inputs(world_pos, PA, date, PA_is_FPA, SCAs):
    """
    This routine parses the various input options to getWCS() and returns what the routine needs to
    do its job.  The reason to pull this out is so other helper routines can use it.
    """

    # Parse input position
    if not isinstance(world_pos, coord.CelestialCoord):
        raise TypeError("Position on the sky must be given as a galsim.CelestialCoord!")

    # Get the date. (Vernal equinox in 2025, taken from
    # http://www.astropixels.com/ephemeris/soleq2001.html, if none was supplied.)
    if date is None:
        date = datetime.datetime(2025,3,20,9,2,0)

    # Are we allowed to look here?
    if not allowedPos(world_pos, date):
        raise GalSimError("Error, Roman cannot look at this position on this date!")

    # If position angle was not given, then get the optimal one:
    if PA is None:
        PA_is_FPA = False
        PA = bestPA(world_pos, date)
    else:
        # Just enforce type
        if not isinstance(PA, coord.Angle):
            raise TypeError("Position angle must be a galsim.Angle!")

    # Check which SCAs are to be done using a helper routine in the galsim.roman module.
    SCAs = _parse_SCAs(SCAs)

    # Compute position angle of FPA f2 axis, where positive corresponds to the angle east of North.
    if PA_is_FPA:
        pa_fpa = PA
        pa_obsy = PA - theta_fpa
    else:
        pa_obsy = PA
        pa_fpa = PA + theta_fpa

    return date, SCAs, pa_fpa, pa_obsy

def allowedPos(world_pos, date):
    """
    This routine can be used to check whether Roman would be allowed to look at a particular
    position (``world_pos``) on a given ``date``.   This is determined by the angle of this position
    relative to the Sun.

    In general, Roman can point at angles relative to the Sun in the range 90+/-36 degrees.
    Obviously, pointing too close to the Sun would result in overly high sky backgrounds.  It is
    less obvious why Roman cannot look at a spot directly opposite from the Sun (180 degrees on the
    sky).  The reason is that the observatory is aligned such that if the observer is looking at
    some sky position, the solar panels are oriented at 90 degrees from that position.  So it's
    always optimal for the observatory to be pointing at an angle of 90 degrees relative to the
    Sun.  It is also permitted to look within 36 degrees of that optimal position.

    Parameters:
        world_pos:      A galsim.CelestialCoord indicating the position at which the observer
                        wishes to look.
        date:           A python datetime object indicating the desired date of observation.

    Returns:
        True or False, indicating whether it is permitted to look at this position on this date.
    """
    # Find the Sun's location on the sky on this date.
    lam = coord.util.sun_position_ecliptic(date)
    sun = coord.CelestialCoord.from_ecliptic(lam, 0*coord.radians, date.year)

    # Find the angle between that and the supplied position
    angle_deg = abs(world_pos.distanceTo(sun)/coord.degrees)

    # Check if it's within tolerance.
    min_ang = 90. - max_sun_angle
    max_ang = 90. + max_sun_angle
    return min_ang <= angle_deg <= max_ang

def bestPA(world_pos, date):
    """
    This routine determines the best position angle for the observatory for a given observation date
    and position on the sky.

    The best/optimal position angle is determined by the fact that the solar panels are at 90
    degrees to the position being observed, and it is best to have those facing the Sun as directly
    as possible.  Note that if a given ``world_pos`` is not actually observable on the given
    ``date``, then this routine will return None.

    Parameters:
        world_pos:      A galsim.CelestialCoord indicating the position at which the observer
                        wishes to look.
        date:           A python datetime object indicating the desired date of observation.

    Returns:
        the best position angle for the observatory, as a galsim.Angle, or None if the position
        is not observable.
    """
    # First check for observability.
    if not allowedPos(world_pos, date):
        return None

    # Find the location of the sun on this date.  +X_observatory points out into the sky, towards
    # world_pos, while +Z is in the plane of the sky pointing towards the sun as much as possible.
    lam = coord.util.sun_position_ecliptic(date)
    sun = coord.CelestialCoord.from_ecliptic(lam, 0*coord.radians, date.year)
    # Now we do a projection onto the sky centered at world_pos to find the (u, v) for the Sun.
    sun_tp_x, sun_tp_y = world_pos.project(sun, 'gnomonic')

    # We want to rotate around by 90 degrees to find the +Y obs direction.  Specifically, we want
    # (+X, +Y, +Z)_obs to form a right-handed coordinate system.
    y_obs_tp_x, y_obs_tp_y = -sun_tp_y, sun_tp_x
    y_obs = world_pos.deproject(y_obs_tp_x, y_obs_tp_y, 'gnomonic')

    # Finally the observatory position angle is defined by the angle between +Y_observatory and the
    # celestial north pole.  It is defined as position angle east of north.
    north = coord.CelestialCoord(y_obs.ra, 90.*coord.degrees)
    obs_pa = world_pos.angleBetween(y_obs, north)
    return obs_pa
