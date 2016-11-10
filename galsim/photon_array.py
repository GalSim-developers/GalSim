# Copyright (c) 2012-2016 by the GalSim developers team on GitHub
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
from ._galsim import PhotonArray
import galsim

def assignPhotonAngles(self, fratio, obscuration, seed):
    """
    Assigns arrival directions at the focal plane for photons, drawing from a uniform
    brightness distribution between the obscuration angle and the angle of the FOV defined
    by the f-ratio of the telescope.  The angles are expressed in terms of slopes dx/dz
    and dy/dz.

    @param fratio               The f-ratio of the telescope (1.2 for LSST)
    @param obscuration_angle    The angular radius of the central obscuration (deg)
    @param seed                 Random number seed (optional)
    """

    if obscuration < 0 or obscuration > 1:
        raise ValueError("The obscuration fraction must be between 0 and 1.")

    if fratio < 0:
        raise ValueError("The f-ratio must be positive.")
        
    dxdz = self.getDXDZArray()
    dydz = self.getDYDZArray()
    n_photons = len(dxdz)

    fov_angle = np.arctan(0.5 / fratio)  # radians
    obscuration_angle = obscuration * fov_angle

    if seed is None:
        ud = galsim.UniformDeviate()
    else:
        ud = galsim.UniformDeviate(seed)

    # Generate azimuthal angles for the photons
    # Set up a loop to fill the array of azimuth angles for now
    # (The array is initialized below but there's no particular need to do this.)
    phi = np.zeros(n_photons)

    for i in np.arange(n_photons):
        phi[i] = ud() * 2 * np.pi 

    # Generate inclination angles for the photons, which are uniform in sin(theta) between
    # the sine of the obscuration angle and the sine of the FOV radius
    sintheta = np.zeros(n_photons)

    for i in np.arange(n_photons):
        sintheta[i] = np.sin(obscuration_angle) + (np.sin(fov_angle) - \
                      np.sin(obscuration_angle))*ud()

    # Assign the directions to the arrays. In this class the convention for the
    # zero of phi does not matter but it might if the obscuration dependent on
    # phi
    costheta = np.sqrt(1. - np.square(sintheta))
    dxdz[:] = costheta * np.sin(phi)
    dydz[:] = costheta * np.cos(phi)

PhotonArray.assignPhotonAngles = assignPhotonAngles
