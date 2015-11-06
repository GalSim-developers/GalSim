import numpy as np
import galsim

try:
    import lsst.pex.logging as pexLog
    import lsst.afw.geom as afwGeom
    import lsst.afw.cameraGeom as cameraGeom
    from lsst.afw.cameraGeom import PUPIL, PIXELS
    from lsst.obs.lsstSim import LsstSimMapper
except ImportError:
    raise ImportError("You cannot use the LSST module.\n"
                      "You either do not have the LSST stack installed,\n"
                      "or you have it installed, but have not set it up.\n"
                      "------------\n"
                      "To install the LSST stack, follow the instructions at:\n\n"
                      "https://confluence.lsstcorp.org/display/SIM/Catalogs+and+MAF\n\n"
                      "NOTE: you must build the stack with the python you are using to\n"
                      "run GalSim.  This means that when the stack asks you if you want\n"
                      "it to install Anaconda for you , you MUST SAY NO.\n\n"
                      "------------\n"
                      "If you have installed the stack, run\n\n"
                      "source $LSST_HOME/loadLSST.bash\n"
                      "setup obs_lsstSim -t sims\n")


__all__ = ["LsstCamera", "LsstWCS"]

class LsstCamera(object):

    def __init__(self, origin, rotation_angle):
        """
        inputs
        ------------
        origin is a CelestialCoord indicating the direction the telescope is pointing

        rotation_angle is an angle indicating the orientation of the camera with
        respect to the sky.  The convention for rotation_angle is:

        rotation_angle = 0 degrees means north is in the +y direction on the camera and east is -x

        rotation_angle = 90 degrees means north is -x and east is -y

        rotation_angle = -90 degrees means north is +x and east is +y

        rotation_angle = 180 degrees means north is -y and east is +x

        Note that in the above, x and y return to coordinates on the pupil.  These are
        rotated 90 degrees with respect to coordinates on the camera (pixel coordinates)
        because of the LSST Data Management convention that the x-direction in pixel
        coordinates must be oriented along the direction of serial readout.
        """

        # this line prevents the camera mapper from printing harmless warnings to
        # stdout (which, as of 5 November 2015, happens every time you instantiate
        # the camera below)
        pexLog.Log.getDefaultLog().setThresholdFor("CameraMapper", pexLog.Log.FATAL)

        self._camera = LsstSimMapper().camera

        # _pixel_system_dict will be a dictionary of chip pixel coordinate systems
        # keyed to chip names
        self._pixel_system_dict = {}

        # _pupil_system_dict will be a dictionary of chip pupil coordinate systems
        # keyed to chip names
        self._pupil_system_dict = {}

        self._pointing = origin
        self._rotation_angle = rotation_angle
        self._cos_rot = np.cos(self._rotation_angle/galsim.radians)
        self._sin_rot = np.sin(self._rotation_angle/galsim.radians)
        self._cos_dec = np.cos(self._pointing.dec/galsim.radians)
        self._sin_dec = np.sin(self._pointing.dec/galsim.radians)


    def pupilCoordsFromPoint(self, point):
        """
        Convert from RA, Dec into coordinates on the pupil

        inputs
        ------------
        point is a CelestialCoord (or a list of CelestialCoords) indicating
        the positions to be transformed

        outputs
        ------------
        The x and y coordinates on the pupil in radians
        """

        if not hasattr(point, '__len__'):
            pt = self._pointing.project(point, projection='gnomonic')
            x = pt.x
            y = pt.y
        else:
            x = []
            y = []
            for pp in point:
                pt = self._pointing.project(pp, projection='gnomonic')
                x.append(pt.x)
                y.append(pt.y)
            x = np.array(x)
            y = np.array(y)

        return (x*self._cos_rot - y*self._sin_rot)*galsim.arcsec, \
               (x*self._sin_rot + y*self._cos_rot)*galsim.arcsec


    def pupilCoordsFromFloat(self, ra, dec):
        """
        Convert from RA, Dec into coordinates on the pupil

        Note: this method just reimplements the PALPY method Ds2tp
        with some minor adjustments for sign conventions.

        inputs
        ------------
        ra is in radians.  Can be a list.

        dec is in radians.  Can be a list.

        outputs
        ------------
        The x and y coordinates on the pupil in radians
        """

        dra = ra - self._pointing.ra/galsim.radians
        cradif = np.cos(dra)
        sradif = np.sin(dra)
        sdec = np.sin(dec)
        cdec = np.cos(dec)
        denom = sdec * self._sin_dec + cdec * self._cos_dec * cradif
        xx = cdec * sradif/denom
        yy = (sdec * self._cos_dec - cdec * self._sin_dec * cradif)/denom
        xx *= -1.0
        return xx*self._cos_rot - yy*self._sin_rot, xx*self._sin_rot + yy*self._cos_rot


    def raDecFromPupilCoords(self, x, y):
        """
        Convert pupil coordinates in radians to RA, Dec in radians

        Note: this method just reimplements the PALPY method Dtp2s
        with some minor adjustments for sign conventsion

        inputs
        ------------
        x is the x pupil coordinate in radians

        y is the y pupil coordinate in radians

        ouputs
        ------------
        RA and Dec in radians as lists of floats (or individual floats if only one
        set of x, y was passed in)
        """

        x_g = x*self._cos_rot + y*self._sin_rot
        y_g = -1.0*x*self._sin_rot + y*self._cos_rot

        x_g *= -1.0

        denom = self._cos_dec - y_g * self._sin_dec
        d = np.arctan2(x_g, denom) + self._pointing.ra/galsim.radians
        ra = d%(2.0*np.pi)
        dec = np.arctan2(self._sin_dec + y_g * self._cos_dec, np.sqrt(x_g*x_g + denom*denom))

        return ra, dec


    def _get_chip_name_from_afw_point_list(self, point_list):
        """
        inputs
        ------------
        point_list is a list of afwGeom.Point2D objects corresponding to pupil coordinates (in radians)

        outputs
        ------------
        a list of chip names where those points fall
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
                                       "Override LSSTWCS._get_chip_name to add this.")
                elif len(names)==0:
                    chip_name_list.append(None)
                else:
                    chip_name_list.append(names[0])

        return chip_name_list


    def _get_afw_pupil_coord_list_from_point(self, point):
        """
        inputs
        -------------
        point is a CelestialCoord (or a list of CelestialCoords) corresponding to RA, Dec
        on the sky

        outputs
        -------------
        a list of afwGeom.Point2D objects correspdonding to pupil coordinates in radians
        of point
        """

        x_pupil, y_pupil = self.pupilCoordsFromPoint(point)

        if hasattr(x_pupil, '__len__'):
            camera_point_list = [afwGeom.Point2D(x/galsim.radians, y/galsim.radians) for x,y in zip(x_pupil, y_pupil)]
        else:
            camera_point_list = [afwGeom.Point2D(x_pupil/galsim.radians, y_pupil/galsim.radians)]

        return camera_point_list


    def _get_afw_pupil_coord_list_from_float(self, ra, dec):
        """
        inputs
        -------------
        ra is in radians (can be a list)

        dec is in radians (can be a list)

        outputs
        -------------
        a list of afwGeom.Point2D objects correspdonding to pupil coordinates in radians
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

        inputs
        ------------
        point is a CelestialCoord (or a list of CelestialCoords) indicating
        the positions to be transformed

        outputs
        ------------
        the name of the chip (or a list of the names of the chips) on which
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

        inputs
        ------------
        ra is in radians (can be a list)

        dec is in radians (can be a list)

        outputs
        ------------
        the name of the chip (or a list of the names of the chips) on which
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
        inputs
        ------------
        point_list is a list of afwGeom.Point2D objects corresponding to pupil coordinates (in radians)

        name_list is a list of chip names

        outputs
        ------------
        a list of x pixel coordinates

        a list of y pixel coordinates

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


    def pixelCoordsFromPoint(self, point):
        """
        Take a point on the sky and transform it into pixel coordinates

        inputs
        ------------
        point is a CelestialCoord (or a list of CelestialCoords) to be
        transformed

        outputs
        ------------
        a list of x pixel coordinates

        a list of y pixel coordinates

        a list of the names of the chips on which x and y are reckoned
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

        inputs
        ------------
        ra is in radians (can be a list)

        dec is in radians (can be a list)

        outputs
        ------------
        a list of x pixel coordinates

        a list of y pixel coordinates

        a list of the names of the chips on which x and y are reckoned
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

        inputs
        ------------
        x is the x pixel coordinate (it can be a list)

        y is the y pixel coordinate (it can be a list)

        chip_name is the name of the chip on which x and y were
        measured (it can be a list)

        outputs
        ------------
        a list of x pupil coordinates in radians

        a list of y pupil coordinates in radians
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
                self._pixel_system_dict[name] = self._camera[name].makeCameraSys(PIXEL)

            if name not in self._pupil_system_dict:
                self._pupil_system_dict[name] = self._camera[name].makeCameraSys(PUPIL)

            pt = self._camera.transform(self._camera.makeCameraPoint(afwGeom.Point2D(xx, yy), self._pixel_system_dict[name]),
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

        inputs
        ------------
        x is the x pixel coordinate (can be a list)

        y is the y pixel coordinate (can be a list)

        chip_name is the name of the chip on which x and y are measured (can be a list)

        outputs
        ------------
        ra is in radians

        dec is in radians
        """

        x_pupil, y_pupil = self.pupilCoordsFromPixelCoords(x, y, chip_name)
        return self.raDecFromPupilCoords(x_pupil, y_pupil)


class LsstWCS(galsim.wcs.CelestialWCS):

    def __init__(self, origin, rotation_angle, chip_name):
        """
        inputs
        ------------
        origin is a CelestialCoord indicating the point at which the telescope
        is pointing

        rotation_angle is an angle indicating the orientation of the camera with
        respect to the sky.  The convention for rotation_angle is:

            rotation_angle = 0 degrees means north is in the +y direction on the camera and east is -x

            rotation_angle = 90 degrees means north is -x and east is -y

            rotation_angle = -90 degrees means north is +x and east is +y

            rotation_angle = 180 degrees means north is -y and east is +x

            Note that in the above, x and y return to coordinates on the pupil.  These are
            rotated 90 degrees with respect to coordinates on the camera (pixel coordinates)
            because of the LSST Data Management convention that the x-direction in pixel
            coordinates must be oriented along the direction of serial readout.

        chip_name is a string indicating the name of the chip to which this WCS corresponds

            valid formats for chip_name are

            R:i,j S:m,n

            where i,j,m,n are integers.  R denotes the raft (a 3x3 block of chips).
            S denotes the chip within the raft.

            chip_names can be found using the chipNameFromFloat and chipNameFromPoint
            methods of the class LsstCamera

            Note: origin denotes the point on the sky at which the center of the entire
            LSST field of view is pointing.  It does not (and often won't) have to fall
            on the chip specified by chip_name
        """

        self._camera = LsstCamera(origin, rotation_angle)
        self._chip_name = chip_name
        if self._chip_name not in self._camera._camera:
            raise RuntimeError("%s is not a valid chip_name for an LsstWCS" % chip_name)


    def _xy(self, ra, dec):
        """
        inputs
        ------------
        ra is in radians (can be a list)

        dec is in radians (can be a list)

        outputs
        ------------
        a list of x pixel coordinates on the chip specified for this WCS

        a list of y pixel coordinates on the chip specified for this WCS
        """
        camera_point_list = self._camera._get_afw_pupil_coord_list_from_float(ra, dec)
        xx, yy = self._camera._pixel_coord_from_point_and_name(camera_point_list,
                                                               [self._chip_name]*len(camera_point_list))

        if len(xx)==1:
            return xx[0], yy[0]
        else:
            return xx, yy
