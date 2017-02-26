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

from __future__ import print_function
import unittest
import numpy as np
import warnings
import os
import galsim
import sys
from galsim_test_helpers import funcname
from galsim.celestial import CelestialCoord

from galsim_test_helpers import *

have_lsst_stack = True

try:
    from galsim.lsst import LsstCamera, LsstWCS
except ImportError as ee:
    #if __name__ == '__main__':
        #raise
    # make sure that you are failing because the stack isn't there,
    # rather than because of some bug in lsst_wcs.py
    if "You cannot use the LSST module" in str(ee):
        global ee_message
        ee_message = str(ee)
        have_lsst_stack = False
    else:
        raise

if have_lsst_stack:
    try:
        import lsst.afw.geom as afwGeom
    except ImportError:
        have_lsst_stack = False


if have_lsst_stack:
    warnings.warn("Some installations of the LSST stack have trouble with "
                  "multiprocessing nosetests.\n"
                  "If `scons tests` freezes up, you may need to run "
                  "`scons tests -j1` instead.")

if sys.version_info < (2,7):
    # skipIf requires Python 2.7, so for 2.6, just make a decorator that skips manually.
    def skipIf(f, cond):
        def f2(*args, **kwargs):
            if cond: return
            else: return f(*args, **kwargs)
        return f2
else:
    skipIf = unittest.skipIf

@skipIf(not have_lsst_stack, "LSST stack not installed")
class NativeLonLatTest(unittest.TestCase):

    @timer
    def testNativeLonLat(self):
        """
        Test that nativeLonLatFromRaDec works by considering stars and pointings
        at intuitive locations
        """
        from galsim.lsst.lsst_wcs import _nativeLonLatFromRaDec

        raList = [0.0, 0.0, 0.0, 1.5*np.pi]
        decList = [0.5*np.pi, 0.5*np.pi, 0.0, 0.0]

        raPointList = [0.0, 1.5*np.pi, 1.5*np.pi, 0.0]
        decPointList = [0.0, 0.0,0.0, 0.0]

        lonControlList = [np.pi, np.pi, 0.5*np.pi, 1.5*np.pi]
        latControlList = [0.0, 0.0, 0.0, 0.0]

        for rr, dd, rp, dp, lonc, latc in \
        zip(raList, decList, raPointList, decPointList, lonControlList, latControlList):
            lon, lat = _nativeLonLatFromRaDec(rr, dd, rp, dp)
            self.assertAlmostEqual(lon, lonc, 10)
            self.assertAlmostEqual(lat, latc, 10)

    @timer
    def testNativeLongLatComplicated(self):
        """
        Test that nativeLongLatFromRaDec works by considering stars and pointings
        at non-intuitive locations.
        """
        from galsim.lsst.lsst_wcs import _nativeLonLatFromRaDec

        rng = np.random.RandomState(42)
        nPointings = 10
        raPointingList = rng.random_sample(nPointings)*2.0*np.pi
        decPointingList = rng.random_sample(nPointings)*np.pi - 0.5*np.pi

        nStars = 10
        for raPointing, decPointing in zip(raPointingList, decPointingList):
            raList = rng.random_sample(nStars)*2.0*np.pi
            decList = rng.random_sample(nStars)*np.pi - 0.5*np.pi
            for raRad, decRad in zip(raList, decList):

                sinRa = np.sin(raRad)
                cosRa = np.cos(raRad)
                sinDec = np.sin(decRad)
                cosDec = np.cos(decRad)

                # the three dimensional position of the star
                controlPosition = np.array([-cosDec*sinRa, cosDec*cosRa, sinDec])

                # calculate the rotation matrices needed to transform the
                # x, y, and z axes into the local x, y, and z axes
                # (i.e. the axes with z lined up with raPointing, decPointing)
                alpha = 0.5*np.pi - decPointing
                ca = np.cos(alpha)
                sa = np.sin(alpha)
                rotX = np.array([[1.0, 0.0, 0.0],
                                 [0.0, ca, sa],
                                 [0.0, -sa, ca]])

                cb = np.cos(raPointing)
                sb = np.sin(raPointing)
                rotZ = np.array([[cb, -sb, 0.0],
                                 [sb, cb, 0.0],
                                 [0.0, 0.0, 1.0]])

                # rotate the coordinate axes into the local basis
                xAxis = np.dot(rotZ, np.dot(rotX, np.array([1.0, 0.0, 0.0])))
                yAxis = np.dot(rotZ, np.dot(rotX, np.array([0.0, 1.0, 0.0])))
                zAxis = np.dot(rotZ, np.dot(rotX, np.array([0.0, 0.0, 1.0])))

                # calculate the local longitude and latitude of the star
                lon, lat = _nativeLonLatFromRaDec(raRad, decRad, raPointing, decPointing)
                cosLon = np.cos(lon)
                sinLon = np.sin(lon)
                cosLat = np.cos(lat)
                sinLat = np.sin(lat)

                # the x, y, z position of the star in the local coordinate basis
                transformedPosition = np.array([-cosLat*sinLon,
                                                   cosLat*cosLon,
                                                   sinLat])

                # convert that position back into the un-rotated bases
                testPosition = transformedPosition[0]*xAxis + \
                               transformedPosition[1]*yAxis + \
                               transformedPosition[2]*zAxis

                # assert that testPosition and controlPosition should be equal
                np.testing.assert_array_almost_equal(controlPosition, testPosition, decimal=10)

    @timer
    def testNativeLonLatVector(self):
        """
        Test that _nativeLonLatFromRaDec works by considering stars and pointings
        at intuitive locations (make sure it works in a vectorized way; we do this
        by performing a bunch of tansformations passing in ra and dec as numpy arrays
        and then comparing them to results computed in an element-wise way)
        """
        from galsim.lsst.lsst_wcs import _nativeLonLatFromRaDec

        raPoint = np.radians(145.0)
        decPoint = np.radians(-35.0)

        nSamples = 100
        rng = np.random.RandomState(42)
        raList = rng.random_sample(nSamples)*2.0*np.pi
        decList = rng.random_sample(nSamples)*np.pi - 0.5*np.pi

        lonList, latList = _nativeLonLatFromRaDec(raList, decList, raPoint, decPoint)

        for rr, dd, lon, lat in zip(raList, decList, lonList, latList):
            lonControl, latControl = _nativeLonLatFromRaDec(rr, dd, raPoint, decPoint)
            self.assertAlmostEqual(lat, latControl, 10)
            if np.abs(np.abs(lat) - 0.5*np.pi)>1.0e-9:
                self.assertAlmostEqual(lon, lonControl, 10)


@skipIf(not have_lsst_stack, "LSST stack not installed")
class LsstCameraTestClass(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        if not have_lsst_stack:
            # skip doesn't apply to setUpClass.  cf. https://github.com/nose-devs/nose/issues/946
            return

        # these are taken from the header of the
        # galsim_afwCameraGeom_data.txt file generated by
        # GalSim/devel/external/generate_galsim_lsst_camera_validation.py
        cls.raPointing = 112.064181578541
        cls.decPointing = -33.015167519966
        cls.rotation = 27.0

        pointing = CelestialCoord(cls.raPointing*galsim.degrees, cls.decPointing*galsim.degrees)
        cls.camera = LsstCamera(pointing, cls.rotation*galsim.degrees)

    @timer
    def test_attribute_exceptions(self):
        """
        Test that exceptions are raised when you try to set attributes
        """

        with self.assertRaises(AttributeError) as context:
            self.camera.pointing = galsim.CelestialCoord(34.0*galsim.degrees, 18.0*galsim.degrees)

        with self.assertRaises(AttributeError) as context:
            self.camera.rotation_angle = 56.0*galsim.degrees

    @timer
    def test_pupil_coordinates(self):
        """
        Test the conversion between (RA, Dec) and pupil coordinates.
        Results are checked against the routine provided by PALPY.
        """

        def palpyPupilCoords(star, pointing):
            """
            This is just a copy of the PALPY method Ds2tp, which
            I am taking to be the ground truth for projection from
            a sphere onto the tangent plane

            inputs
            ------------
            star is a CelestialCoord corresponding to the point being projected

            pointing is a CelestialCoord corresponding to the pointing of the
            'telescope'

            outputs
            ------------
            The x and y coordinates in the focal plane (radians)
            """

            ra = star.ra/galsim.radians
            dec = star.dec/galsim.radians
            ra_pointing = pointing.ra/galsim.radians
            dec_pointing = pointing.dec/galsim.radians

            cdec = np.cos(dec)
            sdec = np.sin(dec)
            cdecz = np.cos(dec_pointing)
            sdecz = np.sin(dec_pointing)
            cradif = np.cos(ra - ra_pointing)
            sradif = np.sin(ra - ra_pointing)

            denom = sdec * sdecz + cdec * cdecz * cradif
            xx = cdec * sradif/denom
            yy = (sdec * cdecz - cdec * sdecz * cradif)/denom
            return xx, yy


        rng = np.random.RandomState(42)
        n_pointings = 10
        ra_pointing_list = rng.random_sample(n_pointings)*2.0*np.pi
        dec_pointing_list = 0.5*(rng.random_sample(n_pointings)-0.5)*np.pi
        rotation_angle_list = rng.random_sample(n_pointings)*2.0*np.pi

        radians_to_arcsec = 3600.0*np.degrees(1.0)

        for ra, dec, rotation in zip(ra_pointing_list, dec_pointing_list, rotation_angle_list):

            pointing = CelestialCoord(ra*galsim.radians, dec*galsim.radians)
            camera = LsstCamera(pointing, rotation*galsim.radians)

            dra_list = (rng.random_sample(100)-0.5)*0.5
            ddec_list = (rng.random_sample(100)-0.5)*0.5

            star_list = np.array([CelestialCoord((ra+dra)*galsim.radians,
                                                (dec+ddec)*galsim.radians)
                                 for dra, ddec in zip(dra_list, ddec_list)])

            xTest, yTest = camera.pupilCoordsFromPoint(star_list)
            xControl = []
            yControl = []
            for star in star_list:
                xx, yy = palpyPupilCoords(star, pointing)
                xx *= -1.0
                xControl.append(xx*np.cos(rotation) - yy*np.sin(rotation))
                yControl.append(yy*np.cos(rotation) + xx*np.sin(rotation))

            xControl = np.array(xControl)
            yControl = np.array(yControl)

            np.testing.assert_array_almost_equal((xTest*radians_to_arcsec) -
                                                 (xControl*radians_to_arcsec),
                                                 np.zeros(len(xControl)),  7)

            np.testing.assert_array_almost_equal((yTest*radians_to_arcsec) -
                                                 (yControl*radians_to_arcsec),
                                                 np.zeros(len(yControl)), 7)

    @timer
    def test_pupil_coordinates_from_floats(self):
        """
        Test that the method which converts floats into pupil coordinates agrees with the method
        that converts CelestialCoords into pupil coordinates
        """

        raPointing = 113.0
        decPointing = -25.6
        rot = 82.1
        pointing = CelestialCoord(raPointing*galsim.degrees, decPointing*galsim.degrees)
        camera = LsstCamera(pointing, rot*galsim.degrees)

        arcsec_per_radian = 180.0*3600.0/np.pi
        rng = np.random.RandomState(33)
        raList = (rng.random_sample(100)-0.5)*20.0+raPointing
        decList = (rng.random_sample(100)-0.5)*20.0+decPointing
        pointingList = []
        for rr, dd in zip(raList, decList):
            pointingList.append(CelestialCoord(rr*galsim.degrees, dd*galsim.degrees))

        control_x, control_y = camera.pupilCoordsFromPoint(pointingList)
        test_x, test_y = camera.pupilCoordsFromFloat(np.radians(raList), np.radians(decList))

        np.testing.assert_array_almost_equal((test_x - control_x)*arcsec_per_radian,
                                             np.zeros(len(test_x)), 10)


        np.testing.assert_array_almost_equal((test_y - control_y)*arcsec_per_radian,
                                             np.zeros(len(test_y)), 10)

    @timer
    def test_ra_dec_from_pupil_coords(self):
        """
        Test that the method which converts from pupil coordinates back to RA, Dec works
        """

        rng = np.random.RandomState(55)
        n_samples = 100
        raList = (rng.random_sample(n_samples)-0.5)*1.0 + np.radians(self.raPointing)
        decList = (rng.random_sample(n_samples)-0.5)*1.0 + np.radians(self.decPointing)

        x_pupil, y_pupil = self.camera.pupilCoordsFromFloat(raList, decList)

        ra_test, dec_test = self.camera.raDecFromPupilCoords(x_pupil, y_pupil)

        np.testing.assert_array_almost_equal(np.cos(raList), np.cos(ra_test), 10)
        np.testing.assert_array_almost_equal(np.sin(raList), np.sin(ra_test), 10)
        np.testing.assert_array_almost_equal(np.cos(decList), np.cos(dec_test), 10)
        np.testing.assert_array_almost_equal(np.sin(decList), np.sin(dec_test), 10)

    @timer
    def test_pupil_coords_from_pixel_coords(self):
        """
        Test the conversion from pixel coordinates back into pupil coordinates
        """

        rng = np.random.RandomState(88)
        n_samples = 100
        raList = (rng.random_sample(n_samples)-0.5)*np.radians(1.5) + \
                  np.radians(self.raPointing)

        decList = (rng.random_sample(n_samples)-0.5)*np.radians(1.5) + \
                   np.radians(self.decPointing)

        x_pup_control, y_pup_control = self.camera.pupilCoordsFromFloat(raList, decList)

        camera_point_list = self.camera._get_afw_pupil_coord_list_from_float(raList, decList)

        chip_name_possibilities = ('R:0,1 S:1,1', 'R:0,3 S:0,2', 'R:4,2 S:2,2', 'R:3,4 S:0,2')

        chip_name_list = [chip_name_possibilities[ii]
                          for ii in rng.randint(0,4,n_samples)]

        x_pix_list, y_pix_list = \
        self.camera._pixel_coord_from_point_and_name(camera_point_list, chip_name_list)

        x_pup_test, y_pup_test = \
        self.camera.pupilCoordsFromPixelCoords(x_pix_list, y_pix_list, chip_name_list)

        np.testing.assert_array_almost_equal(x_pup_test, x_pup_control, 10)
        np.testing.assert_array_almost_equal(y_pup_test, y_pup_control, 10)

        # test one at a time
        for xx, yy, name, x_control, y_control in \
            zip(x_pix_list, y_pix_list, chip_name_list, x_pup_control, y_pup_control):

            x_test, y_test = self.camera.pupilCoordsFromPixelCoords(xx, yy, name)
            self.assertAlmostEqual(x_test, x_control, 10)
            self.assertAlmostEqual(y_test, y_control, 10)


        # test that NaNs are returned if chip_name is None or 'None'
        chip_name_list = ['None'] * len(x_pix_list)
        x_pup_test, y_pup_test = \
        self.camera.pupilCoordsFromPixelCoords(x_pix_list, y_pix_list, chip_name_list)

        for xp, yp in zip(x_pup_test, y_pup_test):
            self.assertTrue(np.isnan(xp))
            self.assertTrue(np.isnan(yp))

        chip_name_list = [None] * len(x_pix_list)
        x_pup_test, y_pup_test = \
        self.camera.pupilCoordsFromPixelCoords(x_pix_list, y_pix_list, chip_name_list)

        for xp, yp in zip(x_pup_test, y_pup_test):
            self.assertTrue(np.isnan(xp))
            self.assertTrue(np.isnan(yp))

    @timer
    def test_rotation_angle_pupil_coordinate_convention(self):
        """
        Test the convention on how rotation angle affects the orientation of north
        on the focal plane (in pupil coordinates) by calculating the puipil
        coordinates of positions slightly displaced from the center of the camera.
        """

        ra = 30.0
        dec = 0.0
        delta = 0.001

        pointing = CelestialCoord(ra*galsim.degrees, dec*galsim.degrees)
        north = CelestialCoord(ra*galsim.degrees, (dec+delta)*galsim.degrees)
        east = CelestialCoord((ra+delta)*galsim.degrees, dec*galsim.degrees)

        camera = LsstCamera(pointing, 0.0*galsim.degrees)
        x_0, y_0 = camera.pupilCoordsFromPoint(pointing)
        x_n, y_n = camera.pupilCoordsFromPoint(north)
        x_e, y_e = camera.pupilCoordsFromPoint(east)
        self.assertAlmostEqual(0.0, np.degrees(x_0), 7)
        self.assertAlmostEqual(0.0, np.degrees(y_0), 7)
        self.assertAlmostEqual(0.0, np.degrees(x_n), 7)
        self.assertGreater(np.degrees(y_n), 1.0e-4)
        self.assertLess(np.degrees(x_e), -1.0e-4)
        self.assertAlmostEqual(np.degrees(y_e), 0.0, 7)

        camera = LsstCamera(pointing, 90.0*galsim.degrees)
        x_n, y_n = camera.pupilCoordsFromPoint(north)
        x_e, y_e = camera.pupilCoordsFromPoint(east)
        self.assertLess(np.degrees(x_n), -1.0e-4)
        self.assertAlmostEqual(np.degrees(y_n), 0.0, 7)
        self.assertAlmostEqual(np.degrees(x_e), 0.0, 7)
        self.assertLess(np.degrees(y_e), -1.0e-4)

        camera = LsstCamera(pointing, -90.0*galsim.degrees)
        x_n, y_n = camera.pupilCoordsFromPoint(north)
        x_e, y_e = camera.pupilCoordsFromPoint(east)
        self.assertGreater(np.degrees(x_n), 1.0e-4)
        self.assertAlmostEqual(np.degrees(y_n), 0.0, 7)
        self.assertAlmostEqual(np.degrees(x_e), 0.0, 7)
        self.assertGreater(np.degrees(y_e), 1.0e-4)

        camera = LsstCamera(pointing, 180.0*galsim.degrees)
        x_n, y_n = camera.pupilCoordsFromPoint(north)
        x_e, y_e = camera.pupilCoordsFromPoint(east)
        self.assertAlmostEqual(np.degrees(x_n), 0, 7)
        self.assertLess(np.degrees(y_n), -1.0e-4)
        self.assertGreater(np.degrees(x_e), 1.0e-4)
        self.assertAlmostEqual(np.degrees(y_e), 0.0, 7)

    @timer
    def test_rotation_angle_pixel_coordinate_convention(self):
        """
        Test the convention on how rotation angle affects the orientation of north
        on the focal plane (in pixel coordinates) by calculating the pixel
        coordinates of positions slightly displaced from the center of the camera.
        """

        ra = 30.0
        dec = 0.0
        delta = 0.001

        pointing = CelestialCoord(ra*galsim.degrees, dec*galsim.degrees)
        north = CelestialCoord(ra*galsim.degrees, (dec+delta)*galsim.degrees)
        east = CelestialCoord((ra+delta)*galsim.degrees, dec*galsim.degrees)

        camera = LsstCamera(pointing, 0.0*galsim.degrees)
        x_0, y_0, name = camera.pixelCoordsFromPoint(pointing)
        x_n, y_n, name = camera.pixelCoordsFromPoint(north)
        x_e, y_e, name = camera.pixelCoordsFromPoint(east)
        self.assertGreater(x_n-x_0, 10.0)
        self.assertAlmostEqual(y_n-y_0, 0.0, 7)
        self.assertAlmostEqual(x_e-x_0, 0.0, 7)
        self.assertGreater(y_e-y_0, 10.0)

        camera = LsstCamera(pointing, 90.0*galsim.degrees)
        x_0, y_0, name = camera.pixelCoordsFromPoint(pointing)
        x_n, y_n, name = camera.pixelCoordsFromPoint(north)
        x_e, y_e, name = camera.pixelCoordsFromPoint(east)
        self.assertAlmostEqual(x_n-x_0, 0.0, 7)
        self.assertGreater(y_n-y_0, 10.0)
        self.assertLess(x_e-x_0, -10.0)
        self.assertAlmostEqual(y_e-y_0, 0.0, 7)

        camera = LsstCamera(pointing, -90.0*galsim.degrees)
        x_0, y_0, name = camera.pixelCoordsFromPoint(pointing)
        x_n, y_n, name = camera.pixelCoordsFromPoint(north)
        x_e, y_e, name = camera.pixelCoordsFromPoint(east)
        self.assertAlmostEqual(x_n-x_0, 0.0, 7)
        self.assertLess(y_n-y_0, -10.0)
        self.assertGreater(x_e-x_0, 10.0)
        self.assertAlmostEqual(y_e-y_0, 0.0, 7)

        camera = LsstCamera(pointing, 180.0*galsim.degrees)
        x_0, y_0, name = camera.pixelCoordsFromPoint(pointing)
        x_n, y_n, name = camera.pixelCoordsFromPoint(north)
        x_e, y_e, name = camera.pixelCoordsFromPoint(east)
        self.assertLess(x_n-x_0, -10.0)
        self.assertAlmostEqual(y_n-y_0, 0.0, 7)
        self.assertAlmostEqual(x_e-x_0, 0.0, 7)
        self.assertLess(y_e-y_0, -10.0)


@skipIf(not have_lsst_stack, "LSST stack not installed")
class LsstWcsTestCase(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        if not have_lsst_stack:
            # skip doesn't apply to setUpClass.  cf. https://github.com/nose-devs/nose/issues/946
            return

        # these are taken from the header of the
        # galsim_afwCameraGeom_forced_data.txt file generated by
        # GalSim/devel/external/generate_galsim_lsst_camera_validation.py
        cls.raPointing = 112.064181578541 * galsim.degrees
        cls.decPointing = -33.015167519966 * galsim.degrees
        cls.rotation = 27.0 * galsim.degrees
        cls.chip_name = 'R:0,1 S:1,2'

        cls.pointing = CelestialCoord(cls.raPointing, cls.decPointing)
        cls.wcs = LsstWCS(cls.pointing, cls.rotation, cls.chip_name)

    @timer
    def test_constructor(self):
        """
        Just make sure that the constructor for LsstWCS runs, and that it throws an error
        when you specify a nonsense chip.
        """

        pointing = CelestialCoord(112.0*galsim.degrees, -39.0*galsim.degrees)
        rotation = 23.1*galsim.degrees

        wcs1 = LsstWCS(pointing, rotation, 'R:1,1 S:2,2')

        with self.assertRaises(RuntimeError) as context:
            wcs2 = LsstWCS(pointing, rotation, 'R:1,1 S:3,3')
        self.assertEqual(context.exception.args[0],
                         "R:1,1 S:3,3 is not a valid chip_name for an LsstWCS")

    @timer
    def test_attribute_exceptions(self):
        """
        Test that exceptions are raised when you try to re-assign LsstWCS attributes
        """

        with self.assertRaises(AttributeError) as context:
            self.wcs.pointing = CelestialCoord(22.0*galsim.degrees, -17.0*galsim.degrees)

        with self.assertRaises(AttributeError) as context:
            self.wcs.rotation_angle = 23.0*galsim.degrees

        with self.assertRaises(AttributeError) as context:
            self.wcs.chip_name = 'R:4,4 S:1,1'

    @timer
    def test_tan_wcs(self):
        """
        Test method to return a Tan WCS by generating a bunch of pixel coordinates
        in the undistorted TAN-PIXELS coordinate system.  Then, use sims_coordUtils
        to convert those pixel coordinates into RA and Dec.  Compare these to the
        RA and Dec returned by the WCS.  Demand agreement to witin 0.001 arcseconds.

        Note: if you use a bigger camera, it is possible to have disagreements of
        order a few milliarcseconds.
        """

        xPixList = []
        yPixList = []

        tanWcs = self.wcs.getTanWcs()
        wcsRa = []
        wcsDec = []
        for xx in np.arange(0.0, 4001.0, 1000.0):
            for yy in np.arange(0.0, 4001.0, 1000.0):
                xPixList.append(xx)
                yPixList.append(yy)

                pt = afwGeom.Point2D(xx ,yy)
                skyPt = tanWcs.pixelToSky(pt).getPosition()
                wcsRa.append(skyPt.getX())
                wcsDec.append(skyPt.getY())

        wcsRa = np.radians(np.array(wcsRa))
        wcsDec = np.radians(np.array(wcsDec))

        xPixList = np.array(xPixList)
        yPixList = np.array(yPixList)

        raTest, decTest = \
        self.wcs._camera.raDecFromTanPixelCoords(xPixList, yPixList,
                                                [self.wcs._chip_name]*len(xPixList))

        for rr1, dd1, rr2, dd2 in zip(raTest, decTest, wcsRa, wcsDec):
            pp = CelestialCoord(rr1*galsim.radians, dd1*galsim.radians)

            dist = \
            pp.distanceTo(CelestialCoord(rr2*galsim.radians, dd2*galsim.radians))/galsim.arcsec

            msg = 'error in tanWcs was %e arcsec' % dist
            self.assertLess(dist, 0.001, msg=msg)

    @timer
    def test_tan_sip_wcs(self):
        """
        Test that getTanSipWcs works by fitting a TAN WCS and a TAN-SIP WCS to
        the a detector with distortions and verifying that the TAN-SIP WCS better approximates
        the truth.
        """

        arcsec_per_radian = 180.0*3600.0/np.pi

        tanWcs = self.wcs.getTanWcs()
        tanSipWcs = self.wcs.getTanSipWcs()

        tanWcsRa = []
        tanWcsDec = []
        tanSipWcsRa = []
        tanSipWcsDec = []

        xPixList = []
        yPixList = []
        for xx in np.arange(0.0, 4001.0, 1000.0):
            for yy in np.arange(0.0, 4001.0, 1000.0):
                xPixList.append(xx)
                yPixList.append(yy)

                pt = afwGeom.Point2D(xx ,yy)
                skyPt = tanWcs.pixelToSky(pt).getPosition()
                tanWcsRa.append(skyPt.getX())
                tanWcsDec.append(skyPt.getY())

                skyPt = tanSipWcs.pixelToSky(pt).getPosition()
                tanSipWcsRa.append(skyPt.getX())
                tanSipWcsDec.append(skyPt.getY())

        tanWcsRa = np.radians(np.array(tanWcsRa))
        tanWcsDec = np.radians(np.array(tanWcsDec))

        tanSipWcsRa = np.radians(np.array(tanSipWcsRa))
        tanSipWcsDec = np.radians(np.array(tanSipWcsDec))

        xPixList = np.array(xPixList)
        yPixList = np.array(yPixList)

        raTest, decTest = \
        self.wcs._camera.raDecFromPixelCoords(xPixList, yPixList,
                                             [self.wcs._chip_name]*len(xPixList))

        for rrTest, ddTest, rrTan, ddTan, rrSip, ddSip in \
        zip(raTest, decTest, tanWcsRa, tanWcsDec, tanSipWcsRa, tanSipWcsDec):

            pp = CelestialCoord(rrTest*galsim.radians, ddTest*galsim.radians)

            distTan = \
            pp.distanceTo(CelestialCoord(rrTan*galsim.radians, ddTan*galsim.radians))/galsim.arcsec

            distSip = \
            pp.distanceTo(CelestialCoord(rrSip*galsim.radians, ddSip*galsim.radians))/galsim.arcsec

            msg = 'error in TAN WCS %e arcsec; error in TAN-SIP WCS %e arcsec' % (distTan, distSip)
            self.assertLess(distSip, 0.001, msg=msg)
            self.assertGreater(distTan-distSip, 1.0e-10, msg=msg)

    @timer
    def test_round_trip(self):
        """
        Test writing out an image with an LsstWCS, reading it back in, and comparing
        the resulting pixel -> ra, dec mappings
        """

        path, filename = os.path.split(__file__)

        im0 = galsim.Image(int(4000), int(4000), wcs=self.wcs)

        outputFile = os.path.join(path,'scratch_space','lsst_roundtrip_img.fits')
        if os.path.exists(outputFile):
            os.unlink(outputFile)
        im0.write(outputFile)

        im1 = galsim.fits.read(outputFile)

        xPix = []
        yPix = []
        pixPts = []
        for xx in range(0, 4000, 100):
            for yy in range(0, 4000, 100):
                xPix.append(xx)
                yPix.append(yy)
                pixPts.append(galsim.PositionI(xx, yy))

        xPix = np.array(xPix)
        yPix = np.array(yPix)

        ra_control, dec_control = self.wcs._radec(xPix, yPix)
        for rr, dd, pp in zip(ra_control, dec_control, pixPts):
            ra_dec_test = im1.wcs.toWorld(pp)
            self.assertAlmostEqual(rr, ra_dec_test.ra/galsim.radians, 12)
            self.assertAlmostEqual(dd, ra_dec_test.dec/galsim.radians, 12)

        if os.path.exists(outputFile):
            os.unlink(outputFile)

    @timer
    def test_eq(self):
        """
        Test that __eq__ works for LsstWCS
        """

        wcs1 = LsstWCS(self.pointing, self.rotation, self.chip_name)
        self.assertEqual(self.wcs, wcs1)

        new_origin = galsim.PositionI(9, 9)
        wcs1 = wcs1._newOrigin(new_origin)
        self.assertNotEqual(self.wcs, wcs1)

        other_pointing = CelestialCoord(1.9*galsim.degrees, -34.0*galsim.degrees)
        wcs2 = LsstWCS(other_pointing, self.rotation, self.chip_name)
        self.assertNotEqual(self.wcs, wcs2)

        wcs3 = LsstWCS(self.pointing, 112.0*galsim.degrees, self.chip_name)
        self.assertNotEqual(self.wcs, wcs3)

        wcs4 = LsstWCS(self.pointing, self.rotation, 'R:2,2 S:2,2')
        self.assertNotEqual(self.wcs, wcs4)

    @timer
    def test_copy(self):
        """
        Test that copy() works
        """

        pointing = CelestialCoord(64.82*galsim.degrees, -16.73*galsim.degrees)
        rotation = 116.8*galsim.degrees
        chip_name = 'R:1,2 S:2,2'
        wcs0 = LsstWCS(pointing, rotation, chip_name)
        wcs0 = wcs0._newOrigin(galsim.PositionI(112, 4))
        wcs1 = wcs0.copy()
        self.assertEqual(wcs0, wcs1)

        wcs0 = wcs0._newOrigin(galsim.PositionI(66, 77))
        self.assertNotEqual(wcs0, wcs1)

    @timer
    def test_pickling(self):
        """
        Test that LsstWCS can be pickled and un-pickled
        """

        path, filename = os.path.split(__file__)
        file_name = os.path.join(path,'scratch_space','pickle_LsstWCS.txt')

        import pickle

        with open(file_name, 'w') as output_file:
             pickle.dump(self.wcs, output_file)

        with open(file_name, 'r') as input_file:
            wcs1 = pickle.load(input_file)

        self.assertEqual(self.wcs, wcs1)

        if os.path.exists(file_name):
            os.unlink(file_name)

    @timer
    def test_passing_camera_by_hand(self):
        """
        Test that you can pass a camera from one WCS to another
        """

        with warnings.catch_warnings(record=True) as ww:
            wcs1 = LsstWCS(self.pointing, self.rotation, chip_name='R:0,1 S:1,1',
                           camera=self.wcs.camera)

        self.assertEqual(len(ww), 0)


        # verify that, if the camera does not have the pointing or rotation angle you want,
        # a new camera will be instantiated
        with warnings.catch_warnings(record=True) as ww:
            wcs1 = LsstWCS(galsim.CelestialCoord(0.0*galsim.degrees, 0.0*galsim.degrees),
                           self.rotation, chip_name='R:0,1 S:1,1', camera=self.wcs.camera)

        expected_message = "The camera you passed to LsstWCS does not have the same\n" \
                           "pointing and rotation angle as you asked for for this WCS.\n" \
                           "LsstWCS is creating a new camera with the pointing and\n" \
                           "rotation angle you specified in the constructor for LsstWCS."
        self.assertEqual(str(ww[0].message), expected_message)

        with warnings.catch_warnings(record=True) as ww:
            wcs1 = LsstWCS(self.pointing, 49.0*galsim.degrees,
                           chip_name='R:0,1 S:1,1', camera=self.wcs.camera)

        expected_message = "The camera you passed to LsstWCS does not have the same\n" \
                           "pointing and rotation angle as you asked for for this WCS.\n" \
                           "LsstWCS is creating a new camera with the pointing and\n" \
                           "rotation angle you specified in the constructor for LsstWCS."
        self.assertEqual(str(ww[0].message), expected_message)


if __name__ == "__main__":
    if have_lsst_stack:
        unittest.main()
    else:
        print(ee_message)
