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
import galsim
import os

def getPSF(approximate_spider=False, gsparams=None, **kwargs):
    """
    Get a PSF for LSST.  Currently does only the basics, keeps a lot of information locally that
    should eventually go in __init__.py.  Does not attempt to use camera geometry to propagate
    through the orientation of the spider.  No chromatic info yet.  Aberrations should be provided
    by the user, will be passed to OpticalPSF.
    """
    diam = 8.36 # m (clear area)
    obscuration = 5.1/diam # linear obscuration
    pupil_plane_im = os.path.join(galsim.meta_data.share_dir, "lsst_spider_2048.fits")
    pupil_plane_scale = diam / 2048. # for a 2048 x 2048 image
    nstruts = 4 # when doing approximate calculation

    if approximate_spider:
        psf = galsim.OpticalPSF(lam=500., diam=diam, obscuration=obscuration, nstruts=nstruts,
                                gsparams=gsparams, **kwargs)
    else:
        psf = galsim.OpticalPSF(lam=500., diam=diam, obscuration=obscuration,
                                pupil_plane_im=pupil_plane_im, pupil_plane_scale=pupil_plane_scale,
                                gsparams=gsparams, **kwargs)
    return psf
