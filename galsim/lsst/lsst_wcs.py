import numpy as np
import galsim

__all__ = ["LSSTWCS"]

class LSSTWCS(galsim.wcs.CelestialWCS):

    def __init__(self, origin, rotation_angle):
        """
        inputs
        ------------
        origin is a CelestialCoord indicating the direction the telescope is pointing

        rotation_angle is an angle indicating the orientation of the camera with
        respect to the sky.  The convention for rotation_angle is:

        rotSkyPos = 0 degrees means north is in the +y direction on the camera and east is -x

        rotSkyPos = 90 degrees means north is -x and east is -y

        rotSkyPos = -90 degrees means north is +x and east is +y

        rotSkyPos = 180 degrees means north is -y and east is +x

        Note that in the above, x and y return to coordinates on the pupil.  These are
        rotated 90 degrees with respect to coordinates on the camera (pixel coordinates)
        because of the LSST Data Management convention that the x-direction in pixel
        coordinates must be oriented along the direction of serial readout.
        """

        self._pointing = origin
        self._rotation_angle = rotation_angle
        self.cos_rot = np.cos(self._rotation_angle/galsim.radians)
        self.sin_rot = np.sin(self._rotation_angle/galsim.radians)


    def _get_pupil_coordinates(self, point):
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

        x *= -1.0
        return (x*self.cos_rot - y*self.sin_rot)*galsim.arcsec, (x*self.sin_rot + y*self.cos_rot)*galsim.arcsec
