"""
This script generates the galsim_afwCameraGeom_data.txt file against
which the transformations from RA, Dec to pixel position in lsst_wcs.py
are validated.

To run it, you will need to install the entire LSST Simulations stack
(not just the packages required by the galsim/lsst module
"""


from __future__ import with_statement

import numpy as np
from lsst.sims.utils import ObservationMetaData
from lsst.sims.coordUtils import chipNameFromRaDec, pixelCoordsFromRaDec, pupilCoordsFromRaDec
from lsst.sims.coordUtils import observedFromICRS
from lsst.obs.lsstSim import LsstSimMapper

if __name__ == "__main__":
    camera = LsstSimMapper().camera


    ra = 112.0
    dec = -33.0
    rot = 27.0
    mjd = 53850.0
    epoch = 2000.0
    obs = ObservationMetaData(unrefractedRA=ra, unrefractedDec=dec,
                              rotSkyPos=rot, mjd=mjd)


    ra_p, dec_p = observedFromICRS(np.array([ra]), np.array([dec]),
                                   obs_metadata=obs, epoch=epoch)


    np.random.seed(42)
    raList = 3.0*(np.random.random_sample(20)-0.5) + ra_p[0]
    decList = 3.0*(np.random.random_sample(20)-0.5) + dec_p[0]

    chipNameList = chipNameFromRaDec(raList, decList, obs_metadata=obs,
                                     epoch=epoch, camera=camera)

    pixelCoordsList = pixelCoordsFromRaDec(raList, decList, obs_metadata=obs, epoch=epoch, camera=camera)
    pupilCoordsList = pupilCoordsFromRaDec(raList, decList, obs_metadata=obs, epoch=epoch)

    print pixelCoordsList.shape

    with open('galsim_afwCameraGeom_data.txt', 'w') as output:
        output.write('# pointing: ra %.12f degrees dec %.12f degrees rot %e degrees\n' % (ra_p[0], dec_p[0], rot))
        output.write('# columns: ra; dec; chip name; x_pixel; y_pixel; x_pupil (rad); y_pupil (rad)\n')
        for rr, dd, name, pixpt, puppt in \
        zip(raList, decList, chipNameList, pixelCoordsList.transpose(), pupilCoordsList.transpose()):
            output.write('%.12f; %.12f; %s; %.12f; %.12f; %.12f; %.12f\n'
            % (rr, dd, name, pixpt[0], pixpt[1], puppt[0], puppt[1]))

    # now generate pixel coordinates which are forced to be reckoned on a specific chip
    chip_name = 'R:0,1 S:1,2'
    forcedPixelCoordsList = pixelCoordsFromRaDec(raList, decList, obs_metadata=obs, epoch=epoch,
                                                 camera=camera, chipNames=[chip_name]*len(raList))

    with open('galsim_afwCameraGeom_forced_data.txt', 'w') as output:
        output.write('# pointing: ra %.12f degrees dec %.12f degrees rot %e degrees\n' % (ra_p[0], dec_p[0], rot))
        output.write('# chip %s\n' % chip_name)
        output.write('# columns: ra; dec; x_pixel; y_pixel;\n')
        for rr, dd, pixpt in zip(raList, decList, forcedPixelCoordsList.transpose()):
            output.write('%.12f; %.12f; %.12f; %.12f\n' % (rr, dd, pixpt[0], pixpt[1]))
