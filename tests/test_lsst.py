import unittest
import numpy as np
import galsim
from galsim.lsst import LSSTWCS
from galsim.celestial import CelestialCoord


class WcsTestClass(unittest.TestCase):

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
            wcs = LSSTWCS(pointing, rotation*galsim.radians)

            dra_list = (np.random.random_sample(100)-0.5)*0.5
            ddec_list = (np.random.random_sample(100)-0.5)*0.5

            star_list = np.array([CelestialCoord((ra+dra)*galsim.radians, (dec+ddec)*galsim.radians)
                                 for dra, ddec in zip(dra_list, ddec_list)])

            xTest, yTest = wcs._get_pupil_coordinates(star_list)
            xControl = []
            yControl = []
            for star in star_list:
                xx, yy = palpyPupilCoords(star, pointing)
                xControl.append(xx*np.cos(rotation) - yy*np.sin(rotation))
                yControl.append(yy*np.cos(rotation) + xx*np.sin(rotation))

            xControl = np.array(xControl)
            yControl = np.array(yControl)

            np.testing.assert_array_almost_equal(xTest/galsim.arcsec, xControl/galsim.arcsec, 7)
            np.testing.assert_array_almost_equal(yTest/galsim.arcsec, yControl/galsim.arcsec, 7)

