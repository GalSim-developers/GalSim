# Copyright (c) 2012-2017 by the GalSim developers team on GitHub
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

import numpy as np
import warnings
import galsim

try:
    with warnings.catch_warnings():
        # At least one of these imports matplotlib at the top level, which on some systems
        # emits warnings.  Ignore them.
        warnings.filterwarnings("ignore",'.*Matplotlib.*')
        import lsst.pex.logging as pexLog
        import lsst.daf.base as dafBase
        import lsst.afw.geom as afwGeom
        import lsst.afw.cameraGeom as cameraGeom
        import lsst.afw.image as afwImage
        import lsst.afw.image.utils as afwImageUtils
        import lsst.meas.astrom as measAstrom
        from lsst.afw.cameraGeom import PUPIL, PIXELS, TAN_PIXELS, FOCAL_PLANE
        from lsst.obs.lsstSim import LsstSimMapper
except ImportError:
    raise ImportError("You cannot use the LSST module.\n"
                      "You either do not have the LSST stack installed,\n"
                      "or you have it installed, but have not set it up.\n"
                      "------------\n"
                      "To install the LSST stack, follow the instructions at:\n\n"
                      "\thttps://confluence.lsstcorp.org/display/SIM/Catalogs+and+MAF\n\n"
                      "NOTE: Make sure that you use the same version of python to install\n"
                      "the LSST stack as you do to build GalSim (or vice-versa).  See\n"
                      "\thttps://github.com/GalSim-developers/GalSim/wiki/Building-GalSim-with-the-LSST-stack\n"
                      "for more details\n\n"
                      "------------\n"
                      "If you have installed the stack, run\n\n"
                      "If you installed from source:\n"
                      "\tsource $LSST_HOME/loadLSST.bash\n"
                      "\tsetup obs_lsstSim -t sims\n\n"
                      "If you installed the binary (conda) distribution:\n"
                      "\tsource eups-setups.sh\n"
                      "\tsetup obs_lsstSim\n\n"
                      "These commands setup the LSST package management system EUPS "
                      "and then tell EUPS to setup the package that contains the "
                      "model of the LSST camera.  For an explanation of what EUPS "
                      "is and how it works, see\n"
                      "\thttps://developer.lsst.io/build-ci/eups_tutorial.html\n")


warnings.filterwarnings('always', '.*LsstWCS*')


__all__ = ["LsstCamera", "LsstWCS", "_nativeLonLatFromRaDec"]

def _nativeLonLatFromRaDec(ra, dec, raPointing, decPointing):
    """
    Convert the RA and Dec of a star into `native' longitude and latitude.

    Native longitude and latitude are defined as what RA and Dec would be
    if the celestial pole were at the location where the telescope is pointing.
    The transformation is achieved by rotating the vector pointing to the RA
    and Dec being transformed once about the x axis and once about the z axis.
    These are the Euler rotations referred to in Section 2.3 of

    Calabretta and Greisen (2002), A&A 395, p. 1077

    Note: while ra and dec can be numpy.arrays, raPointing and decPointing
    must be floats (you cannot transform for more than one pointing at once)

    @param ra           The RA of the star being transformed in radians
    @param dec          The Dec of the star being transformed in radians
    @param raPointing   The RA at which the telescope is pointing in radians
    @param decPointing  The Dec at which the telescope is pointing in radians

    @returns a tuple (lon, lat), the longitude and latitude in radians.
    """

    x = -1.0*np.cos(dec)*np.sin(ra)
    y = np.cos(dec)*np.cos(ra)
    z = np.sin(dec)

    alpha = decPointing - 0.5*np.pi
    beta = raPointing

    ca=np.cos(alpha)
    sa=np.sin(alpha)
    cb=np.cos(beta)
    sb=np.sin(beta)

    v2 = np.dot(np.array([
                          [1.0, 0.0, 0.0],
                          [0.0,  ca,  sa],
                          [0.0, -sa,  ca]
                          ]),
                   np.dot(np.array([[ cb,  sb, 0.0],
                                    [-sb,  cb, 0.0],
                                    [0.0, 0.0, 1.0]
                                    ]), np.array([x,y,z])))

    cc = np.sqrt(v2[0]*v2[0]+v2[1]*v2[1])
    latOut = np.arctan2(v2[2], cc)

    _y = v2[1]/np.cos(latOut)
    _ra_raw = np.arccos(_y)

    # control for _y=1.0, -1.0 but actually being stored as just outside
    # the bounds of -1.0<=_y<=1.0 because of floating point error
    if hasattr(_ra_raw, '__len__'):
        _ra = np.array([rr if not np.isnan(rr)
                           else 0.5*np.pi*(1.0-np.sign(yy))
                           for rr, yy in zip(_ra_raw, _y)])
    else:
        if np.isnan(_ra_raw):
            if np.sign(_y)<0.0:
                _ra = np.pi
            else:
                _ra = 0.0
        else:
            _ra = _ra_raw

    _x = -np.sin(_ra)

    if type(_ra) is np.float64:
        if np.isnan(_ra):
            lonOut = 0.0
        elif ( (np.abs(_x)>1.0e-9 and np.sign(_x)!=np.sign(v2[0])) or
               (np.abs(_y)>1.0e-9 and np.sign(_y)!=np.sign(v2[1])) ):
            lonOut = 2.0*np.pi-_ra
        else:
            lonOut = _ra
    else:
        _lonOut = [2.0*np.pi-rr if ( (np.abs(xx)>1.0e-9 and np.sign(xx)!=np.sign(v2_0)) or
                                     (np.abs(yy)>1.0e-9 and np.sign(yy)!=np.sign(v2_1)) )
                                else rr
                                for rr, xx, yy, v2_0, v2_1 in zip(_ra, _x, _y, v2[0], v2[1])]

        lonOut = np.array([0.0 if np.isnan(ll) else ll for ll in _lonOut])

    return lonOut, latOut


class LsstCamera(object):
    # A class variable, so we only have to build it once.
    # And only if we need it (it is actually built the first time LsstCamera.__init__ is called).
    _camera = None

    """
    This class characterizes the entire LSST Camera.  It uses a pointing and a rotation
    angle to construct transformations between RA, Dec and pixel positions on each of the
    chips in the camera.

    Note: Each chip on the camera has its own origin in pixel coordinates.  When you ask this
    class to transform RA, Dec into pixel coordinates, it will return you coordinate values as
    well as the name of the chip on which those coordinate values are valid.

    The convention for rotation_angle is:

            rotation_angle = 0 degrees means north is in the +x direction
            (in pixel coordinates) and east is in the +y direction

            rotation_angle = 90 degrees means north is +y and east is -x

            rotation_angle = -90 degrees means north is -y and east is +x

            rotation_angle = 180 degrees means north is -x and east is -y

    @param origin           A CelestialCoord indicating the direction the telescope is pointing
    @param rotation_angle   An angle indicating the orientation of the camera with
                            respect to the sky.
    """
    def __init__(self, origin, rotation_angle):
        # this line prevents the camera mapper from printing harmless warnings to
        # stdout (which, as of 5 November 2015, happens every time you instantiate
        # the camera below)
        pexLog.Log.getDefaultLog().setThresholdFor("CameraMapper", pexLog.Log.FATAL)

        # Make the camera if we haven't yet.
        if LsstCamera._camera is None:
            LsstCamera._camera = LsstSimMapper().camera

        # _pixel_system_dict will be a dictionary of chip pixel coordinate systems
        # keyed to chip names
        self._pixel_system_dict = {}

        # _tan_pixel_system_dict will be a dictionary of chip tan pixel coordinate
        # systems
        self._tan_pixel_system_dict = {}

        # _pupil_system_dict will be a dictionary of chip pupil coordinate systems
        # keyed to chip names
        self._pupil_system_dict = {}

        self._pointing = origin
        self._rotation_angle = rotation_angle
        self._cos_rot = np.cos(self._rotation_angle/galsim.radians)
        self._sin_rot = np.sin(self._rotation_angle/galsim.radians)
        self._cos_dec = np.cos(self._pointing.dec/galsim.radians)
        self._sin_dec = np.sin(self._pointing.dec/galsim.radians)


    @property
    def pointing(self):
        """
        A galsim.CelestialCoord characterizing the direction the camera is pointing
        """
        return self._pointing


    @property
    def rotation_angle(self):
        """
        A galsim.Angle object characterizing the rotation of the camera with respect
        to the sky.  See the docstring for LsstCamera for the meanings of this angle.
        """
        return self._rotation_angle


    def pupilCoordsFromPoint(self, point):
        """
        Convert from RA, Dec into coordinates on the pupil

        @param point    A CelestialCoord (or a list of CelestialCoords) indicating
                        the positions to be transformed

        @return (x,y), the coordinates on the pupil in radians
        """

        if not hasattr(point, '__len__'):
            ra = point.ra
            dec = point.dec
        else:
            ra = np.array([pp.ra for pp in point])
            dec = np.array([pp.dec for pp in point])

        return self.pupilCoordsFromFloat(ra, dec)


    def pupilCoordsFromFloat(self, ra, dec):
        """
        Convert from RA, Dec into coordinates on the pupil

        @param ra       The RA in radians.  Can be a list.
        @param dec      The declination in radians.  Can be a list.

        @returns (x,y), the coordinates on the pupil in radians
        """

        xx, yy = self._pointing.project_rad(ra, dec, projection='gnomonic')
        factor = galsim.arcsec/galsim.radians
        xx *= factor
        yy *= factor
        return xx*self._cos_rot - yy*self._sin_rot, xx*self._sin_rot + yy*self._cos_rot


    def raDecFromPupilCoords(self, x, y):
        """
        Convert pupil coordinates in radians to RA, Dec in radians

        @param x    The x pupil coordinate in radians
        @param y    The y pupil coordinate in radians

        @returns (ra, dec) in radians as lists of floats (or individual floats if only one
                 set of x, y was passed in)
        """

        x_g = x*self._cos_rot + y*self._sin_rot
        y_g = -1.0*x*self._sin_rot + y*self._cos_rot

        factor = galsim.radians/galsim.arcsec
        x_g *= factor
        y_g *= factor

        return self._pointing.deproject_rad(x_g, y_g, projection='gnomonic')


    def _get_chip_name_from_afw_point_list(self, point_list):
        """
        @param point_list   A list of afwGeom.Point2D objects corresponding to pupil coordinates
                            (in radians)

        @returns a list of chip names where those points fall
        """
        det_list = self._camera.findDetectorsList(point_list, PUPIL)

        chip_name_list = []

        for pt, det in zip(point_list, det_list):
            if len(det)==0 or np.isnan(pt.getX()) or np.isnan(pt.getY()):
                chip_name_list.append(None)
            else:
                names = [dd.getName() for dd in det]
                if len(names)>1:
                    raise RuntimeError("This method does not know how to deal with cameras " +
                                       "where points can be on multiple detectors.  " +
                                       "Override LSSTWCS._get_chip_name_from_afw_point_list " +
                                       "to add this.")
                elif len(names)==0:
                    chip_name_list.append(None)
                else:
                    chip_name_list.append(names[0])

        return chip_name_list


    def _get_afw_pupil_coord_list_from_point(self, point):
        """
        @param point    A CelestialCoord (or a list of CelestialCoords) corresponding to RA, Dec
                        on the sky

        @returns a list of afwGeom.Point2D objects correspdonding to pupil coordinates in radians
                 of point
        """

        x_pupil, y_pupil = self.pupilCoordsFromPoint(point)

        if hasattr(x_pupil, '__len__'):
            camera_point_list = [afwGeom.Point2D(x, y) for x,y in zip(x_pupil, y_pupil)]
        else:
            camera_point_list = [afwGeom.Point2D(x_pupil, y_pupil)]

        return camera_point_list


    def _get_afw_pupil_coord_list_from_float(self, ra, dec):
        """
        @param ra       The RA in radians (can be a list)
        @param dec      The declination in radians (can be a list)

        @returns a list of afwGeom.Point2D objects correspdonding to pupil coordinates in radians
                 of point
        """

        x_pupil, y_pupil = self.pupilCoordsFromFloat(ra, dec)

        if hasattr(x_pupil, '__len__'):
            camera_point_list = [afwGeom.Point2D(x, y) for x,y in zip(x_pupil, y_pupil)]
        else:
            camera_point_list = [afwGeom.Point2D(x_pupil, y_pupil)]

        return camera_point_list


    def chipNameFromPoint(self, point):
        """
        Take a point on the sky and find the chip which sees it

        @param point    A CelestialCoord (or a list of CelestialCoords) indicating
                        the positions to be transformed

        @returns the name of the chip (or a list of the names of the chips) on which
                 those points fall
        """

        camera_point_list = self._get_afw_pupil_coord_list_from_point(point)

        chip_name_list = self._get_chip_name_from_afw_point_list(camera_point_list)


        if len(camera_point_list)==1:
            return chip_name_list[0]
        else:
            return chip_name_list


    def chipNameFromFloat(self, ra, dec):
        """
        Take a point on the sky and find the chip which sees it

        @param ra       The RA in radians (can be a list)
        @param dec      The declintation in radians (can be a list)

        @returns the name of the chip (or a list of the names of the chips) on which
                 those points fall
        """

        camera_point_list = self._get_afw_pupil_coord_list_from_float(ra, dec)

        chip_name_list = self._get_chip_name_from_afw_point_list(camera_point_list)

        if len(camera_point_list)==1:
            return chip_name_list[0]
        else:
            return chip_name_list


    def _pixel_coord_from_point_and_name(self, point_list, name_list):
        """
        @param point_list   A list of afwGeom.Point2D objects corresponding to pupil coordinates
                            (in radians)
        @param name_list    A list of chip names

        @return (x_list, y_list), lists of x and y pixel coordinates

        Note: these coordinates are only valid on the chips named in name_list
        """

        x_pix = []
        y_pix = []
        for name, pt in zip(name_list, point_list):
            if name is None:
                x_pix.append(np.nan)
                y_pix.append(np.nan)
                continue

            if name not in self._pixel_system_dict:
                self._pixel_system_dict[name] = self._camera[name].makeCameraSys(PIXELS)

            cp = self._camera.makeCameraPoint(pt, PUPIL)
            detPoint = self._camera.transform(cp, self._pixel_system_dict[name])
            x_pix.append(detPoint.getPoint().getX())
            y_pix.append(detPoint.getPoint().getY())

        return np.array(x_pix), np.array(y_pix)


    def _tan_pixel_coord_from_point_and_name(self, point_list, name_list):
        """
        @param point_list   A list of afwGeom.Point2D objects corresponding to pupil coordinates
                            (in radians)
        @param name_list    A list of chip names

        @returns (x_list, y_list), lists of x and y tan_pixel coordinates

        Note: these coordinates are only valid on the chips named in name_list
        """

        x_pix = []
        y_pix = []
        for name, pt in zip(name_list, point_list):
            if name is None:
                x_pix.append(np.nan)
                y_pix.append(np.nan)
                continue

            if name not in self._pixel_system_dict:
                self._tan_pixel_system_dict[name] = self._camera[name].makeCameraSys(TAN_PIXELS)

            cp = self._camera.makeCameraPoint(pt, PUPIL)
            detPoint = self._camera.transform(cp, self._tan_pixel_system_dict[name])
            x_pix.append(detPoint.getPoint().getX())
            y_pix.append(detPoint.getPoint().getY())

        return np.array(x_pix), np.array(y_pix)


    def pixelCoordsFromPoint(self, point):
        """
        Take a point on the sky and transform it into pixel coordinates

        Note: if the point specified does not fall on a chip, the
        returned coordinates will be numpy.NaN and the returned
        chip name will be None (not the string 'None'; an actual None).

        @param point    A CelestialCoord (or a list of CelestialCoords) to be transformed

        @returns (x_list, y_list, chip_names), the coordinates in radians and the names of the
                 chips on which x and y are reckoned
        """

        camera_point_list = self._get_afw_pupil_coord_list_from_point(point)
        chip_name_list = self._get_chip_name_from_afw_point_list(camera_point_list)
        xx, yy = self._pixel_coord_from_point_and_name(camera_point_list, chip_name_list)

        if len(xx)==1:
            return xx[0], yy[0], chip_name_list[0]
        else:
            return np.array(xx), np.array(yy), chip_name_list


    def pixelCoordsFromFloat(self, ra, dec):
        """
        Take a point on the sky and transform it into pixel coordinates

        Note: if the point specified does not fall on a chip, the
        returned coordinates will be numpy.NaN and the returned
        chip name will be None (not the string 'None'; an actual None).

        @param ra       The RA in radians (can be a list)
        @param dec      The declination in radians (can be a list)

        @returns (x_list, y_list, chip_names), the coordinates in radians and the names of the
                 chips on which x and y are reckoned
        """

        camera_point_list = self._get_afw_pupil_coord_list_from_float(ra, dec)
        chip_name_list = self._get_chip_name_from_afw_point_list(camera_point_list)
        xx, yy = self._pixel_coord_from_point_and_name(camera_point_list, chip_name_list)

        if len(xx)==1:
            return xx[0], yy[0], chip_name_list[0]
        else:
            return np.array(xx), np.array(yy), chip_name_list


    def pupilCoordsFromPixelCoords(self, x, y, chip_name):
        """
        Convert pixel coordinates on a specific chip into pupil coordinates (in radians)

        @param x            The x pixel coordinate (can be a list)
        @param y            The y pixel coordinate (can be a list)
        @param chip_name    The name of the chip on which x and y were measured (can be a list)

        @returns (x_list, y_list), the (x,y) coordinations in radians
        """

        if not hasattr(x, '__len__'):
            x_list = [x]
            y_list = [y]
            chip_name_list = [chip_name]
        else:
            x_list = x
            y_list = y
            chip_name_list = chip_name

        x_pupil = []
        y_pupil = []
        for xx, yy, name in zip(x_list, y_list, chip_name_list):
            if name is None or name=='None':
                x_pupil.append(np.NaN)
                y_pupil.append(np.NaN)
                continue

            if name not in self._pixel_system_dict:
                self._pixel_system_dict[name] = self._camera[name].makeCameraSys(PIXELS)

            if name not in self._pupil_system_dict:
                self._pupil_system_dict[name] = self._camera[name].makeCameraSys(PUPIL)

            pt = self._camera.transform(self._camera.makeCameraPoint(
                                            afwGeom.Point2D(xx, yy),
                                            self._pixel_system_dict[name]),
                                        self._pupil_system_dict[name]).getPoint()

            x_pupil.append(pt.getX())
            y_pupil.append(pt.getY())


        if len(x_pupil)==1:
            return x_pupil[0], y_pupil[0]
        else:
            return np.array(x_pupil), np.array(y_pupil)


    def pupilCoordsFromTanPixelCoords(self, x, y, chip_name):
        """
        Convert tan_pixel coordinates on a specific chip into pupil coordinates (in radians)

        @param x            The x tan_pixel coordinate (can be a list)
        @param y            The y tan_pixel coordinate (can be a list)
        @param chip_name    The name of the chip on which x and y were measured (can be a list)

        @returns (x_list, y_list), the (x,y) pupil coordinates in radians
        """

        if not hasattr(x, '__len__'):
            x_list = [x]
            y_list = [y]
            chip_name_list = [chip_name]
        else:
            x_list = x
            y_list = y
            chip_name_list = chip_name

        x_pupil = []
        y_pupil = []
        for xx, yy, name in zip(x_list, y_list, chip_name_list):
            if name is None or name=='None':
                x_pupil.append(np.NaN)
                y_pupil.append(np.NaN)
                continue

            if name not in self._tan_pixel_system_dict:
                self._tan_pixel_system_dict[name] = self._camera[name].makeCameraSys(TAN_PIXELS)

            if name not in self._pupil_system_dict:
                self._pupil_system_dict[name] = self._camera[name].makeCameraSys(PUPIL)

            pt = self._camera.transform(self._camera.makeCameraPoint(
                                               afwGeom.Point2D(xx, yy),
                                               self._tan_pixel_system_dict[name]),
                                        self._pupil_system_dict[name]).getPoint()

            x_pupil.append(pt.getX())
            y_pupil.append(pt.getY())


        if len(x_pupil)==1:
            return x_pupil[0], y_pupil[0]
        else:
            return np.array(x_pupil), np.array(y_pupil)



    def raDecFromPixelCoords(self, x, y, chip_name):
        """
        Convert pixel coordinates into RA, Dec

        @param x            The x pixel coordinate (can be a list)
        @param y            The y pixel coordinate (can be a list)
        @param chip_name    The name of the chip on which x and y are measured (can be a list)

        @returns (ra, dec) in radians
        """

        x_pupil, y_pupil = self.pupilCoordsFromPixelCoords(x, y, chip_name)
        return self.raDecFromPupilCoords(x_pupil, y_pupil)


    def raDecFromTanPixelCoords(self, x, y, chip_name):
        """
        Convert tan_pixel coordinates into RA, Dec

        @param x            The x tan_pixel coordinate (can be a list)
        @param y            The y tan_pixel coordinate (can be a list)
        @param chip_name    The name of the chip on which x and y are measured (can be a list)

        @returns (ra, dec) in radians
        """

        x_pupil, y_pupil = self.pupilCoordsFromTanPixelCoords(x, y, chip_name)
        return self.raDecFromPupilCoords(x_pupil, y_pupil)


class LsstWCS(galsim.wcs.CelestialWCS):
    """
    This class characterizes the WCS for a single chip on the LSST Camera.
    It uses an instantiation of the class LsstCamera to handle the transformations
    between RA, Dec and pixel coordinates.

    Note: the pixel coordinates calculated by this class are relative to the origin
    of the chip specified by self.chip_name.  You can get valid pixel coordinates for
    an RA, Dec pair that does not actually fall on the chip.  The returned coordinates
    will just exceed the chip bounds in that case (most chips have 4000 pixels in both
    the x and y directions; it is possible to get pixel coordinates like (5000, 6000), etc.).
    To find which chip an RA, Dec point lies on, use the methods in the LsstCamera class.

    The convention for rotation_angle is:

            rotation_angle = 0 degrees means north is in the +x direction
            (in pixel coordinates) and east is in the +y direction

            rotation_angle = 90 degrees means north is +y and east is -x

            rotation_angle = -90 degrees means north is -y and east is +x

            rotation_angle = 180 degrees means north is -x and east is -y

    Note: The origin attribute denotes the point on the sky at which the center of the entire
          LSST field of view is pointing.  It does not (and often won't) have to fall on the chip
          specified by chip_name

    @param pointing         A CelestialCoord indicating the point at which the telescope is
                            pointing.
    @param rotation_angle   An angle indicating the orientation of the camera with respect to the
                            sky.  (See above for the conventions.)
    @param chip_name        A string indicating the name of the chip to which this WCS corresponds.
                            Valid formats for chip_name are
                                R:i,j S:m,n
                            where i,j,m,n are integers.  R denotes the raft (a 3x3 block of chips).
                            S denotes the chip within the raft.  Chip names can be found using the
                            chipNameFromFloat and chipNameFromPoint methods of the class LsstCamera.
    @param camera           An optional instantiation of LsstCamera that you want to use for this
                            WCS.  Before actually assigning camera to this WCS, LsstWCS will verify
                            that camera was created with the pointing and rotation_angle that you
                            say you want for this WCS.  If that is not true, LsstWCS will go ahead
                            and create a new LsstCamera instantiation for this WCS.  This option
                            exists in case you want to create multiple LsstWCSs using the same
                            camera but do not want to incur the overhead of instantiating a camera
                            for each WCS.
    """

    def __init__(self, pointing, rotation_angle, chip_name, camera=None):
        self._pointing = pointing
        self._rotation_angle = rotation_angle
        self._chip_name = chip_name
        self._camera = camera
        self._initialize()

    @property
    def pointing(self):
        """
        A galsim.CelestialCoord representing the point at which the center of the camera
        pointed.
        """
        return self._pointing


    @property
    def rotation_angle(self):
        """
        A galsim.Angle object representing the rotation of the camera with respect to the sky
        """
        return self._rotation_angle


    @property
    def chip_name(self):
        """
        A string indicating the name of the chip for which this WCS is valid
        """
        return self._chip_name

    @property
    def camera(self):
        """
        An instantiation of LsstCamera characterizing the camera used for this WCS
        """
        return self._camera


    def _initialize(self):
        """
        Setup the LsstCamera that does all of the calculations for this
        WCS

        (This is separate from __init__() so that it can be used in pickling)
        """

        if (self._camera is None or
            self._camera.pointing!=self._pointing or
            self._camera.rotation_angle!=self._rotation_angle):

            if self._camera is not None:
               # changing the text of this warning will necessitate a change in the
               # test_passing_camera_by_hand method in test_lsst.py
               warnings.warn("The camera you passed to LsstWCS does not have the same\n"
                             "pointing and rotation angle as you asked for for this WCS.\n"
                             "LsstWCS is creating a new camera with the pointing and\n"
                             "rotation angle you specified in the constructor for LsstWCS.")

            self._camera = LsstCamera(self._pointing, self._rotation_angle)

        if self._chip_name not in self._camera._camera:
            raise RuntimeError("%s is not a valid chip_name for an LsstWCS" % self._chip_name)

        self._detector = self._camera._camera[self._chip_name]

        self.origin = galsim.PositionD(0,0)


    def _newOrigin(self, origin):
        newWcs = LsstWCS(self._pointing, self._rotation_angle,
                         self._chip_name, camera=self._camera)
        newWcs.origin = origin
        return newWcs


    def _xy(self, ra, dec):
        """
        @param ra       The RA in radians (can be a list)
        @param dec      The declination in radians (can be a list)

        @returns (x,y)
        """
        camera_point_list = self._camera._get_afw_pupil_coord_list_from_float(ra, dec)
        xx, yy = self._camera._pixel_coord_from_point_and_name(
                    camera_point_list, [self._chip_name]*len(camera_point_list))

        if len(xx)==1:
            return xx[0], yy[0]
        else:
            return xx, yy


    def _radec(self, x, y):
        """
        @param x    The x pixel coordinate on this chip (can be a list)
        @param y    The y pixel coordinate on this chip (can be a list)

        @returns (ra,dec) in radians
        """

        if hasattr(x, '__len__'):
            chip_name = [self._chip_name]*len(x)
        else:
            chip_name = self._chip_name

        return self._camera.raDecFromPixelCoords(x, y, chip_name)


    def _getTanPixelBounds(self):
        """
        Return the minimum and maximum values of x and y in TAN_PIXELS
        coordinates (pixel coordinates without any distoration due to
        optics applied).

        output order is: xmin, xmax, ymin, ymax
        """

        tanPixelSystem = self._detector.makeCameraSys(TAN_PIXELS)
        xPixMin = None
        xPixMax = None
        yPixMin = None
        yPixMax = None
        cornerPointList = self._detector.getCorners(FOCAL_PLANE)
        for cornerPoint in cornerPointList:
            cameraPoint = self._camera._camera.transform(
                               self._detector.makeCameraPoint(cornerPoint, FOCAL_PLANE),
                               tanPixelSystem).getPoint()

            xx = cameraPoint.getX()
            yy = cameraPoint.getY()
            if xPixMin is None or xx<xPixMin:
                xPixMin = xx
            if xPixMax is None or xx>xPixMax:
                xPixMax = xx
            if yPixMin is None or yy<yPixMin:
                yPixMin = yy
            if yPixMax is None or yy>yPixMax:
                yPixMax = yy

        return xPixMin, xPixMax, yPixMin, yPixMax


    def getTanWcs(self):
        """
        Return a WCS which approximates the focal plane as perfectly flat
        (i.e. it ignores optical distortions that the telescope may impose on the image)

        The output is an instantiation of lsst.afw.image's TanWcs class
        representing the WCS of the detector as if there were no optical
        distortions imposed by the telescope.
        """

        xTanPixMin, xTanPixMax, yTanPixMin, yTanPixMax = self._getTanPixelBounds()


        xPixList = []
        yPixList = []
        nameList = []

        # dx and dy are set somewhat heuristically.
        # Setting them eqal to 0.1*(max-min) lead to errors on the order of 0.7 arcsec in the WCS.
        # Here we use 0.5*(max-min), which is good to ~1 mas.

        dx = 0.5*(xTanPixMax-xTanPixMin)
        dy = 0.5*(yTanPixMax-yTanPixMin)
        for xx in np.arange(xTanPixMin, xTanPixMax+0.5*dx, dx):
            for yy in np.arange(yTanPixMin, yTanPixMax+0.5*dx, dx):
                xPixList.append(xx)
                yPixList.append(yy)
                nameList.append(self._chip_name)

        raList, decList = self._camera.raDecFromTanPixelCoords(np.array(xPixList),
                                                               np.array(yPixList),
                                                               nameList)

        raPointing = self._camera._pointing.ra/galsim.radians
        decPointing = self._camera._pointing.dec/galsim.radians

        camera_point_list = self._camera._get_afw_pupil_coord_list_from_float(
                raPointing, decPointing)

        crPix1, crPix2 = self._camera._tan_pixel_coord_from_point_and_name(
                camera_point_list, [self._chip_name])

        lonList, latList = _nativeLonLatFromRaDec(raList, decList, raPointing, decPointing)

        # convert from native longitude and latitude to intermediate world coordinates
        # according to equations (12), (13), (54) and (55) of
        #
        # Calabretta and Greisen (2002), A&A 395, p. 1077
        #
        radiusList = 180.0/(np.tan(latList)*np.pi)
        uList = radiusList*np.sin(lonList)
        vList = -radiusList*np.cos(lonList)

        # Now that we have converted into the intermediate world coordinates u and v
        # solve for the coefficients [c1, c2, c3, c4] that best (in the least-squares
        # sense) satisfy
        #
        # u = c1 * deltaX + c2 * deltaY
        # v = c3 * deltaX + c4 * deltaY
        #
        # where (x, y) is a position on the focal plain in pixel coordinates and
        # deltaX = (x - crPix1) and deltaY = (y - crPix2)

        delta_xList = xPixList - crPix1[0]
        delta_yList = yPixList - crPix2[0]

        bVector = np.array([
                           (delta_xList*uList).sum(),
                           (delta_yList*uList).sum(),
                           (delta_xList*vList).sum(),
                           (delta_yList*vList).sum()
                           ])

        offDiag = (delta_yList*delta_xList).sum()
        xsq = np.power(delta_xList,2).sum()
        ysq = np.power(delta_yList,2).sum()

        aMatrix = np.array([
                           [xsq, offDiag, 0.0, 0.0],
                           [offDiag, ysq, 0.0, 0.0],
                           [0.0, 0.0, xsq, offDiag],
                           [0.0, 0.0, offDiag, ysq]
                           ])

        coeffs = np.linalg.solve(aMatrix, bVector)

        fitsHeader = dafBase.PropertyList()
        fitsHeader.set("RADESYS", "ICRS")
        fitsHeader.set("EQUINOX", 2000.0)
        fitsHeader.set("CRVAL1", np.degrees(raPointing))
        fitsHeader.set("CRVAL2", np.degrees(decPointing))
        fitsHeader.set("CRPIX1", crPix1[0]+1) # the +1 is because LSST uses 0-indexed images
        fitsHeader.set("CRPIX2", crPix2[0]+1) # FITS files use 1-indexed images
        fitsHeader.set("CTYPE1", "RA---TAN")
        fitsHeader.set("CTYPE2", "DEC--TAN")
        fitsHeader.setDouble("CD1_1", coeffs[0])
        fitsHeader.setDouble("CD1_2", coeffs[1])
        fitsHeader.setDouble("CD2_1", coeffs[2])
        fitsHeader.setDouble("CD2_2", coeffs[3])
        tanWcs = afwImage.cast_TanWcs(afwImage.makeWcs(fitsHeader))

        return tanWcs


    def getTanSipWcs(self, order=3, skyToleranceArcSec=0.001, pixelTolerance=0.01):
        """
        Take an afw Detector and approximate its pixel-to-(Ra,Dec) transformation
        with a TAN-SIP WCs.

        Definition of the TAN-SIP WCS can be found in Shupe and Hook (2008)
        http://fits.gsfc.nasa.gov/registry/sip/SIP_distortion_v1_0.pdf

        @param order                The order of the SIP polynomials to be fit to the
                                    optical distortions [default: 3]
        @param skyToleranceArcSec   The maximum allowed error in the fitted world coordinates (in
                                    arcseconds).  [default: 0.001]
        @param pixelTolerance       The maximum allowed error in the fitted pixel coordinates.
                                    [default: 0.02]

        @returns tanSipWcs, an instantiation of lsst.afw.image's TanWcs class representing the WCS
                 of the detector with optical distortions parametrized by the SIP polynomials.
        """

        bbox = self._detector.getBBox()

        tanWcs = self.getTanWcs()

        mockExposure = afwImage.ExposureF(bbox.getMaxX(), bbox.getMaxY())
        mockExposure.setWcs(tanWcs)
        mockExposure.setDetector(self._detector)

        distortedWcs = afwImageUtils.getDistortedWcs(mockExposure.getInfo())

        tanSipWcs = measAstrom.approximateWcs(
                distortedWcs, bbox, order=order,
                skyTolerance=skyToleranceArcSec*afwGeom.arcseconds,
                pixelTolerance=pixelTolerance)

        return tanSipWcs


    def _writeHeader(self, header, bounds):

        tanSipWcs = self.getTanSipWcs()
        tanSipHeader = tanSipWcs.getFitsMetadata()
        header["GS_WCS"] = ("lsst.LsstWCS", "GalSim WCS name")
        for name in tanSipHeader.getOrderedNames():
            header[name] = tanSipHeader.get(name)

        header["RApoint"] = self._camera._pointing.ra/galsim.radians
        header["DECpoint"] = self._camera._pointing.dec/galsim.radians
        header["ROT"] = self._camera._rotation_angle/galsim.radians
        header["CNAME"] = self._chip_name

        return header


    @staticmethod
    def _readHeader(header):
        pointing = galsim.CelestialCoord(header.get("RApoint")*galsim.radians,
                                         header.get("DECpoint")*galsim.radians)

        rot = header.get("ROT")*galsim.radians
        return LsstWCS(pointing, rot, header.get("CNAME"))


    def copy(self):
        other = LsstWCS(self._camera._pointing, self._camera._rotation_angle, self._chip_name)
        other = other._newOrigin(self.origin)
        return other


    def __eq__(self, other):
        return (isinstance(other, LsstWCS) and
                other.origin == self.origin and
                other._camera._pointing == self._camera._pointing and
                other._camera._rotation_angle == self._camera._rotation_angle and
                other._chip_name == self._chip_name)


    def __repr__(self):
        return ("galsim.lsst.LsstWCS(galsim.CelestialCoord(%e*galsim.radians, %e*galsim.radians), "
                "%e*galsim.radians, '%s')" % (self._camera._pointing.ra/galsim.radians,
                                              self._camera._pointing.dec/galsim.radians,
                                              self._camera._rotation_angle/galsim.radians,
                                              self._chip_name))


    def __str__(self):
        return self.__repr__()


    def __hash__(self):
        return hash(self.__repr__())


    def __getstate__(self):
        output_dict = {}
        output_dict['pointing'] = self._pointing
        output_dict['rotation_angle'] = self._rotation_angle
        output_dict['chip_name'] = self._chip_name
        output_dict['origin'] = self.origin
        return output_dict


    def __setstate__(self, input_dict):
        self._camera = None
        self._pointing = input_dict['pointing']
        self._rotation_angle = input_dict['rotation_angle']
        self._chip_name = input_dict['chip_name']
        self._initialize()
        self.origin = input_dict['origin']
