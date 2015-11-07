from __future__ import with_statement
import unittest
import numpy as np
import os
import galsim
from galsim.lsst import LsstCamera, LsstWCS, _nativeLonLatFromRaDec
from galsim.celestial import CelestialCoord
import lsst.afw.geom as afwGeom

def haversine(long1, lat1, long2, lat2):
    """
    Return the angular distance between two points in radians

    inputs
    ------------
    long1 is the longitude of point 1 in radians

    lat1 is the latitude of point 1 in radians

    long2 is the longitude of point 2 in radians

    lat2 is the latitude of point 2 in radians

    outputs
    ------------
    the angular separation between points 1 and 2 in radians

    From http://en.wikipedia.org/wiki/Haversine_formula
    """
    t1 = np.sin(lat2/2.-lat1/2.)**2
    t2 = np.cos(lat1)*np.cos(lat2)*np.sin(long2/2. - long1/2.)**2
    return 2*np.arcsin(np.sqrt(t1 + t2))


class NativeLonLatTest(unittest.TestCase):

    def testNativeLonLat(self):
        """
        Test that nativeLonLatFromRaDec works by considering stars and pointings
        at intuitive locations
        """

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


    def testNativeLongLatComplicated(self):
        """
        Test that nativeLongLatFromRaDec works by considering stars and pointings
        at non-intuitive locations.
        """

        np.random.seed(42)
        nPointings = 10
        raPointingList = np.random.random_sample(nPointings)*2.0*np.pi
        decPointingList = np.random.random_sample(nPointings)*np.pi - 0.5*np.pi

        nStars = 10
        for raPointing, decPointing in zip(raPointingList, decPointingList):
            raList = np.random.random_sample(nStars)*2.0*np.pi
            decList = np.random.random_sample(nStars)*np.pi - 0.5*np.pi
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



    def testNativeLonLatVector(self):
        """
        Test that _nativeLonLatFromRaDec works by considering stars and pointings
        at intuitive locations (make sure it works in a vectorized way; we do this
        by performing a bunch of tansformations passing in ra and dec as numpy arrays
        and then comparing them to results computed in an element-wise way)
        """

        raPoint = np.radians(145.0)
        decPoint = np.radians(-35.0)

        nSamples = 100
        np.random.seed(42)
        raList = np.random.random_sample(nSamples)*2.0*np.pi
        decList = np.random.random_sample(nSamples)*np.pi - 0.5*np.pi

        lonList, latList = _nativeLonLatFromRaDec(raList, decList, raPoint, decPoint)

        for rr, dd, lon, lat in zip(raList, decList, lonList, latList):
            lonControl, latControl = _nativeLonLatFromRaDec(rr, dd, raPoint, decPoint)
            self.assertAlmostEqual(lat, latControl, 10)
            if np.abs(np.abs(lat) - 0.5*np.pi)>1.0e-9:
                self.assertAlmostEqual(lon, lonControl, 10)


class LsstCameraTestClass(unittest.TestCase):

    @classmethod
    def setUpClass(cls):

        # these are taken from the header of the
        # galsim_afwCameraGeom_data.txt file generated by
        # GalSim/devel/external/generate_galsim_lsst_camera_validation.py
        cls.raPointing = 112.064181578541
        cls.decPointing = -33.015167519966
        cls.rotation = 27.0

        cls.validation_msg = "The LSST Camera outputs are no longer consistent\n" \
                             + "with the LSST Stack.  Contact Scott Daniel at scottvalscott@gmail.com\n" \
                             + "to make sure you have the correct version\n" \
                             + "\nYou can also try re-creating the test validation data\n" \
                             + "using the script GalSim/devel/external/generate_galsim_lsst_camera_validation.py"

        pointing = CelestialCoord(cls.raPointing*galsim.degrees, cls.decPointing*galsim.degrees)
        cls.camera = LsstCamera(pointing, cls.rotation*galsim.degrees)

        path, filename = os.path.split(__file__)
        file_name = os.path.join(path, 'random_data', 'galsim_afwCameraGeom_data.txt')
        dtype = np.dtype([('ra', np.float), ('dec', np.float), ('chipName', str, 100),
                           ('xpix', np.float), ('ypix', np.float),
                           ('xpup', np.float), ('ypup', np.float)])

        cls.camera_data = np.genfromtxt(file_name, dtype=dtype, delimiter='; ')


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
            return xx*galsim.radians, yy*galsim.radians


        np.random.seed(42)
        n_pointings = 10
        ra_pointing_list = np.random.random_sample(n_pointings)*2.0*np.pi
        dec_pointing_list = 0.5*(np.random.random_sample(n_pointings)-0.5)*np.pi
        rotation_angle_list = np.random.random_sample(n_pointings)*2.0*np.pi

        for ra, dec, rotation in zip(ra_pointing_list, dec_pointing_list, rotation_angle_list):

            pointing = CelestialCoord(ra*galsim.radians, dec*galsim.radians)
            camera = LsstCamera(pointing, rotation*galsim.radians)

            dra_list = (np.random.random_sample(100)-0.5)*0.5
            ddec_list = (np.random.random_sample(100)-0.5)*0.5

            star_list = np.array([CelestialCoord((ra+dra)*galsim.radians, (dec+ddec)*galsim.radians)
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

            np.testing.assert_array_almost_equal((xTest/galsim.arcsec) - (xControl/galsim.arcsec), np.zeros(len(xControl)),  7)
            np.testing.assert_array_almost_equal((yTest/galsim.arcsec) - (yControl/galsim.arcsec), np.zeros(len(yControl)), 7)


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
        np.random.seed(33)
        raList = (np.random.random_sample(100)-0.5)*20.0+raPointing
        decList = (np.random.random_sample(100)-0.5)*20.0+decPointing
        pointingList = []
        for rr, dd in zip(raList, decList):
            pointingList.append(CelestialCoord(rr*galsim.degrees, dd*galsim.degrees))

        control_x, control_y = camera.pupilCoordsFromPoint(pointingList)
        test_x, test_y = camera.pupilCoordsFromFloat(np.radians(raList), np.radians(decList))

        np.testing.assert_array_almost_equal((test_x - control_x/galsim.radians)*arcsec_per_radian,
                                             np.zeros(len(test_x)), 10)


        np.testing.assert_array_almost_equal((test_y - control_y/galsim.radians)*arcsec_per_radian,
                                             np.zeros(len(test_y)), 10)


    def test_ra_dec_from_pupil_coords(self):
        """
        Test that the method which converts from pupil coordinates back to RA, Dec works
        """

        np.random.seed(55)
        n_samples = 100
        raList = (np.random.random_sample(n_samples)-0.5)*1.0 + np.radians(self.raPointing)
        decList = (np.random.random_sample(n_samples)-0.5)*1.0 + np.radians(self.decPointing)

        x_pupil, y_pupil = self.camera.pupilCoordsFromFloat(raList, decList)

        ra_test, dec_test = self.camera.raDecFromPupilCoords(x_pupil, y_pupil)

        np.testing.assert_array_almost_equal(np.cos(raList), np.cos(ra_test), 10)
        np.testing.assert_array_almost_equal(np.sin(raList), np.sin(ra_test), 10)
        np.testing.assert_array_almost_equal(np.cos(decList), np.cos(dec_test), 10)
        np.testing.assert_array_almost_equal(np.sin(decList), np.sin(dec_test), 10)


    def test_get_chip_name(self):
        """
        Test the method which associates positions on the sky with names of chips
        """

        # test case of a mapping a single location
        for rr, dd, control_name in \
            zip(self.camera_data['ra'], self.camera_data['dec'], self.camera_data['chipName']):

            point = CelestialCoord(rr*galsim.degrees, dd*galsim.degrees)
            test_name = self.camera.chipNameFromPoint(point)

            try:
                if control_name != 'None':
                    self.assertEqual(test_name, control_name)
                else:
                    self.assertEqual(test_name, None)
            except AssertionError as aa:
                print 'triggering error: ',aa.args[0]
                raise AssertionError(self.validation_msg)

        # test case of mapping a list of celestial coords
        point_list = []
        for rr, dd in zip(self.camera_data['ra'], self.camera_data['dec']):
            point_list.append(CelestialCoord(rr*galsim.degrees, dd*galsim.degrees))

        test_name_list = self.camera.chipNameFromPoint(point_list)
        for test_name, control_name in zip(test_name_list, self.camera_data['chipName']):
            try:
                if control_name != 'None':
                    self.assertEqual(test_name, control_name)
                else:
                    self.assertEqual(test_name, None)
            except AssertionError as aa:
                print 'triggering error: ',aa.args[0]
                raise AssertionError(self.validation_msg)


    def test_get_chip_name_from_float(self):
        """
        Test the method which associates positions on the sky (in terms of floats) with names of chips
        """

        # test case of a mapping a single location
        for rr, dd, control_name in \
            zip(self.camera_data['ra'], self.camera_data['dec'], self.camera_data['chipName']):

            test_name = self.camera.chipNameFromFloat(np.radians(rr), np.radians(dd))

            try:
                if control_name != 'None':
                    self.assertEqual(test_name, control_name)
                else:
                    self.assertEqual(test_name, None)
            except AssertionError as aa:
                print 'triggering error: ',aa.args[0]
                raise AssertionError(self.validation_msg)

        # test case of mapping a list of celestial coords
        test_name_list = self.camera.chipNameFromFloat(np.radians(self.camera_data['ra']), np.radians(self.camera_data['dec']))
        for test_name, control_name in zip(test_name_list, self.camera_data['chipName']):
            try:
                if control_name != 'None':
                    self.assertEqual(test_name, control_name)
                else:
                    self.assertEqual(test_name, None)
            except AssertionError as aa:
                print 'triggering error: ',aa.args[0]
                raise AssertionError(self.validation_msg)


    def test_pixel_coords_from_point(self):
        """
        Test method that goes from CelestialCoord to pixel coordinates
        """

        # test one at a time
        for rr, dd, x_control, y_control, name_control in \
            zip(self.camera_data['ra'], self.camera_data['dec'],
                self.camera_data['xpix'], self.camera_data['ypix'], self.camera_data['chipName']):

            point = CelestialCoord(rr*galsim.degrees, dd*galsim.degrees)
            x_test, y_test, name_test = self.camera.pixelCoordsFromPoint(point)
            try:
                if not np.isnan(x_test):
                    self.assertAlmostEqual(x_test, x_control, 6)
                    self.assertAlmostEqual(y_test, y_control, 6)
                    self.assertEqual(name_test, name_control)
                else:
                    self.assertTrue(np.isnan(x_control))
                    self.assertTrue(np.isnan(y_control))
                    self.assertTrue(np.isnan(y_test))
                    self.assertIsNone(name_test)
            except AssertionError as aa:
                print 'triggering error: ',aa.args[0]
                raise AssertionError(self.validation_msg)

        # test lists
        pointing_list = []
        for rr, dd in zip(self.camera_data['ra'], self.camera_data['dec']):
            pointing_list.append(CelestialCoord(rr*galsim.degrees, dd*galsim.degrees))

        x_test, y_test, name_test_0 = self.camera.pixelCoordsFromPoint(pointing_list)

        name_test = np.array([nn if nn is not None else 'None' for nn in name_test_0])

        try:
            np.testing.assert_array_almost_equal(x_test, self.camera_data['xpix'], 6)
            np.testing.assert_array_almost_equal(y_test, self.camera_data['ypix'], 6)
            np.testing.assert_array_equal(name_test, self.camera_data['chipName'])
        except AssertionError as aa:
            print 'triggering error: ',aa.args[0]
            raise AssertionError(self.validation_msg)


    def test_pixel_coords_from_float(self):
        """
        Test method that goes from floats of RA, Dec to pixel coordinates
        """

        # test one at a time
        for rr, dd, x_control, y_control, name_control in \
            zip(self.camera_data['ra'], self.camera_data['dec'],
                self.camera_data['xpix'], self.camera_data['ypix'], self.camera_data['chipName']):

            x_test, y_test, name_test = self.camera.pixelCoordsFromFloat(np.radians(rr), np.radians(dd))
            try:
                if not np.isnan(x_test):
                    self.assertAlmostEqual(x_test, x_control, 6)
                    self.assertAlmostEqual(y_test, y_control, 6)
                    self.assertEqual(name_test, name_control)
                else:
                    self.assertTrue(np.isnan(x_control))
                    self.assertTrue(np.isnan(y_control))
                    self.assertTrue(np.isnan(y_test))
                    self.assertIsNone(name_test)
            except AssertionError as aa:
                print 'triggering error: ',aa.args[0]
                raise AssertionError(self.validation_msg)

        # test lists
        pointing_list = []
        for rr, dd in zip(self.camera_data['ra'], self.camera_data['dec']):
            pointing_list.append(CelestialCoord(rr*galsim.degrees, dd*galsim.degrees))

        x_test, y_test, name_test_0 = self.camera.pixelCoordsFromFloat(np.radians(self.camera_data['ra']), np.radians(self.camera_data['dec']))

        name_test = np.array([nn if nn is not None else 'None' for nn in name_test_0])

        try:
            np.testing.assert_array_almost_equal(x_test, self.camera_data['xpix'], 6)
            np.testing.assert_array_almost_equal(y_test, self.camera_data['ypix'], 6)
            np.testing.assert_array_equal(name_test, self.camera_data['chipName'])
        except AssertionError as aa:
            print 'triggering error: ',aa.args[0]
            raise AssertionError(self.validation_msg)


    def test_pupil_coords_from_pixel_coords(self):
        """
        Test the conversion from pixel coordinates back into pupil coordinates
        """

        np.random.seed(88)
        n_samples = 100
        raList = (np.random.random_sample(n_samples)-0.5)*np.radians(1.5) + np.radians(self.raPointing)
        decList = (np.random.random_sample(n_samples)-0.5)*np.radians(1.5) + np.radians(self.decPointing)

        x_pup_control, y_pup_control = self.camera.pupilCoordsFromFloat(raList, decList)

        camera_point_list = self.camera._get_afw_pupil_coord_list_from_float(raList, decList)

        chip_name_possibilities = ('R:0,1 S:1,1', 'R:0,3 S:0,2', 'R:4,2 S:2,2', 'R:3,4 S:0,2')

        chip_name_list = [chip_name_possibilities[ii] for ii in np.random.random_integers(0,3,n_samples)]
        x_pix_list, y_pix_list = self.camera._pixel_coord_from_point_and_name(camera_point_list, chip_name_list)

        x_pup_test, y_pup_test = self.camera.pupilCoordsFromPixelCoords(x_pix_list, y_pix_list, chip_name_list)

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
        x_pup_test, y_pup_test = self.camera.pupilCoordsFromPixelCoords(x_pix_list, y_pix_list, chip_name_list)
        for xp, yp in zip(x_pup_test, y_pup_test):
            self.assertTrue(np.isnan(xp))
            self.assertTrue(np.isnan(yp))

        chip_name_list = [None] * len(x_pix_list)
        x_pup_test, y_pup_test = self.camera.pupilCoordsFromPixelCoords(x_pix_list, y_pix_list, chip_name_list)
        for xp, yp in zip(x_pup_test, y_pup_test):
            self.assertTrue(np.isnan(xp))
            self.assertTrue(np.isnan(yp))


    def test_ra_dec_from_pixel_coordinates(self):
        """
        Test the method that converts from pixel coordinates back to RA, Dec
        """

        ra_test, dec_test = self.camera.raDecFromPixelCoords(self.camera_data['xpix'], self.camera_data['ypix'], self.camera_data['chipName'])

        for rt, dt, rc, dc, name in \
            zip(ra_test, dec_test, np.radians(self.camera_data['ra']), np.radians(self.camera_data['dec']), self.camera_data['chipName']):

            if name != 'None':
                self.assertAlmostEqual(np.cos(rt), np.cos(rc))
                self.assertAlmostEqual(np.sin(rt), np.sin(rc))
                self.assertAlmostEqual(np.cos(dt), np.cos(dc))
                self.assertAlmostEqual(np.sin(dt), np.sin(dc))
            else:
                self.assertTrue(np.isnan(rt))
                self.assertTrue(np.isnan(dt))


        np.random.seed(99)
        n_samples = 100
        raList = (np.random.random_sample(n_samples)-0.5)*np.radians(1.5) + np.radians(self.raPointing)
        decList = (np.random.random_sample(n_samples)-0.5)*np.radians(1.5) + np.radians(self.decPointing)

        x_pup_control, y_pup_control = self.camera.pupilCoordsFromFloat(raList, decList)

        camera_point_list = self.camera._get_afw_pupil_coord_list_from_float(raList, decList)

        chip_name_possibilities = ('R:0,1 S:1,1', 'R:0,3 S:0,2', 'R:4,2 S:2,2', 'R:3,4 S:0,2')

        chip_name_list = [chip_name_possibilities[ii] for ii in np.random.random_integers(0,3,n_samples)]
        x_pix_list, y_pix_list = self.camera._pixel_coord_from_point_and_name(camera_point_list, chip_name_list)

        ra_test, dec_test = self.camera.raDecFromPixelCoords(x_pix_list, y_pix_list, chip_name_list)
        np.testing.assert_array_almost_equal(np.cos(ra_test), np.cos(raList), 10)
        np.testing.assert_array_almost_equal(np.sin(ra_test), np.sin(raList), 10)
        np.testing.assert_array_almost_equal(np.cos(dec_test), np.cos(decList), 10)
        np.testing.assert_array_almost_equal(np.sin(dec_test), np.sin(decList), 10)


        # test one at a time
        for xx, yy, name, ra_control, dec_control in \
            zip(x_pix_list, y_pix_list, chip_name_list, raList, decList):

            ra_test, dec_test = self.camera.raDecFromPixelCoords(xx, yy, name)
            self.assertAlmostEqual(np.cos(ra_test), np.cos(ra_control), 10)
            self.assertAlmostEqual(np.sin(ra_test), np.sin(ra_control), 10)
            self.assertAlmostEqual(np.cos(dec_test), np.cos(dec_control), 10)
            self.assertAlmostEqual(np.sin(dec_test), np.sin(dec_control), 10)


class LsstWcsTestCase(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # these are taken from the header of the
        # galsim_afwCameraGeom_forced_data.txt file generated by
        # GalSim/devel/external/generate_galsim_lsst_camera_validation.py
        cls.raPointing = 112.064181578541
        cls.decPointing = -33.015167519966
        cls.rotation = 27.0
        cls.chip_name = 'R:0,1 S:1,2'

        cls.validation_msg = "The LSST WCS outputs are no longer consistent\n" \
                             + "with the LSST Stack.  Contact Scott Daniel at scottvalscott@gmail.com\n" \
                             + "to make sure you have the correct version\n" \
                             + "\nYou can also try re-creating the test validation data\n" \
                             + "using the script GalSim/devel/external/generate_galsim_lsst_camera_validation.py"

        pointing = CelestialCoord(cls.raPointing*galsim.degrees, cls.decPointing*galsim.degrees)
        cls.wcs = LsstWCS(pointing, cls.rotation*galsim.degrees, cls.chip_name)

        dtype = np.dtype([('ra', np.float), ('dec', np.float), ('xpix', np.float), ('ypix', np.float)])

        path, filename = os.path.split(__file__)
        file_name = os.path.join(path, 'random_data', 'galsim_afwCameraGeom_forced_data.txt')
        cls.wcs_data = np.genfromtxt(file_name, dtype=dtype, delimiter='; ')


    def test_constructor(self):
        """
        Just make sure that the constructor for LsstWCS runs, and that it throws an error when you specify
        a nonsense chip.
        """

        pointing = CelestialCoord(112.0*galsim.degrees, -39.0*galsim.degrees)
        rotation = 23.1*galsim.degrees

        wcs1 = LsstWCS(pointing, rotation, 'R:1,1 S:2,2')

        with self.assertRaises(RuntimeError) as context:
            wcs2 = LsstWCS(pointing, rotation, 'R:1,1 S:3,3')
        self.assertEqual(context.exception.args[0],
                         "R:1,1 S:3,3 is not a valid chip_name for an LsstWCS")


    def test_xy(self):
        """
        Test that the conversion from RA, Dec to pixel coordinates works
        """

        # test one-at-a-time use case
        for rr, dd, x_control, y_control in \
            zip(np.radians(self.wcs_data['ra']), np.radians(self.wcs_data['dec']), self.wcs_data['xpix'], self.wcs_data['ypix']):

            x_test, y_test = self.wcs._xy(rr, dd)
            try:
                self.assertAlmostEqual(x_test, x_control, 6)
                self.assertAlmostEqual(y_test, y_control, 6)
            except AssertionError as aa:
                print 'triggering error: ',aa.args[0]
                raise AssertionError(self.validation_msg)

        # test list use case
        x_test, y_test = self.wcs._xy(np.radians(self.wcs_data['ra']), np.radians(self.wcs_data['dec']))
        try:
            np.testing.assert_array_almost_equal(x_test, self.wcs_data['xpix'], 6)
            np.testing.assert_array_almost_equal(y_test, self.wcs_data['ypix'], 6)
        except AssertionError as aa:
            print 'triggering error: ',aa.args[0]
            raise AssertionError(self.validation_msg)


    def test_radec(self):
        """
        Test that the conversion from pixel coordinates to RA, Dec orks
        """

        # test one-at-a-time use case
        for ra_control, dec_control, xx, yy in \
            zip(np.radians(self.wcs_data['ra']), np.radians(self.wcs_data['dec']), self.wcs_data['xpix'], self.wcs_data['ypix']):

            ra_test, dec_test = self.wcs._radec(xx, yy)
            try:
                self.assertAlmostEqual(np.cos(ra_test), np.cos(ra_control), 10)
                self.assertAlmostEqual(np.sin(ra_test), np.sin(ra_control), 10)
                self.assertAlmostEqual(np.cos(dec_test), np.cos(dec_control), 10)
                self.assertAlmostEqual(np.sin(dec_test), np.sin(dec_control), 10)
            except AssertionError as aa:
                print 'triggering error: ',aa.args[0]
                raise AssertionError(self.validation_msg)


        # test list inputs
        ra_test, dec_test = self.wcs._radec(self.wcs_data['xpix'], self.wcs_data['ypix'])
        try:
            np.testing.assert_array_almost_equal(np.cos(ra_test), np.cos(np.radians(self.wcs_data['ra'])))
            np.testing.assert_array_almost_equal(np.sin(ra_test), np.sin(np.radians(self.wcs_data['ra'])))
            np.testing.assert_array_almost_equal(np.cos(dec_test), np.cos(np.radians(self.wcs_data['dec'])))
            np.testing.assert_array_almost_equal(np.sin(dec_test), np.sin(np.radians(self.wcs_data['dec'])))
        except AssertionError as aa:
            print 'triggering error: ',aa.args[0]
            raise AssertionError(self.validation_msg)


    def test_tan_wcs(self):
        """
        Test method to return a Tan WCS by generating a bunch of pixel coordinates
        in the undistorted TAN-PIXELS coordinate system.  Then, use sims_coordUtils
        to convert those pixel coordinates into RA and Dec.  Compare these to the
        RA and Dec returned by the WCS.  Demand agreement to witin 0.001 arcseconds.

        Note: if you use a bigger camera, it is possible to have disagreements of
        order a few milliarcseconds.
        """

        arcsec_per_radian = 180.0*3600.0/np.pi

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

        raTest, decTest = self.wcs._camera.raDecFromTanPixelCoords(xPixList, yPixList,
                                                                   [self.wcs._chip_name]*len(xPixList))

        distanceList = arcsec_per_radian*haversine(raTest, decTest, wcsRa, wcsDec)
        maxDistance = distanceList.max()

        msg = 'maxError in tanWcs was %e ' % maxDistance
        self.assertLess(maxDistance, 0.001, msg=msg)


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

        raTest, decTest = self.wcs._camera.raDecFromPixelCoords(xPixList, yPixList,
                                                                [self.wcs._chip_name]*len(xPixList))

        tanDistanceList = arcsec_per_radian*haversine(raTest, decTest, tanWcsRa, tanWcsDec)
        tanSipDistanceList = arcsec_per_radian*haversine(raTest, decTest, tanSipWcsRa, tanSipWcsDec)

        maxDistanceTan = tanDistanceList.max()
        maxDistanceTanSip = tanSipDistanceList.max()

        msg = 'max error in TAN WCS %e; in TAN-SIP %e' % (maxDistanceTan, maxDistanceTanSip)
        self.assertLess(maxDistanceTanSip, 0.001, msg=msg)
        self.assertGreater(maxDistanceTan-maxDistanceTanSip, 1.0e-10, msg=msg)


    def test_round_trip(self):
        """
        Test writing out an image with an LsstWCS, reading it back in, and comparing
        the resulting pixel -> ra, dec mappings
        """

        path, filename = os.path.split(__file__)

        pointing = CelestialCoord(64.82*galsim.degrees, -16.73*galsim.degrees)
        rotation = 116.8*galsim.degrees
        chip_name = 'R:1,2 S:2,2'
        wcs0 = LsstWCS(pointing, rotation, chip_name)
        im0 = galsim.Image(int(4000), int(4000), wcs=wcs0)

        outputFile = os.path.join(path,'scratch_space','lsst_junk_img.fits')
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

        ra_control, dec_control = wcs0._radec(xPix, yPix)
        for rr, dd, pp in zip(ra_control, dec_control, pixPts):
            ra_dec_test = im1.wcs.toWorld(pp)
            self.assertAlmostEqual(rr, ra_dec_test.ra/galsim.radians, 12)
            self.assertAlmostEqual(dd, ra_dec_test.dec/galsim.radians, 12)

        if os.path.exists(outputFile):
            os.unlink(outputFile)
