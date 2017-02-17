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
"""@file optics.py
Module containing deprecated optical PSF generation routines.
"""

# Import the deprecated optics module here, so that you can still do
#     >>> from galsim.optics import wavefront
# instead of requiring
#     >>> from galsim.deprecated.optics import wavefront
#
# Note that in the main galsim namespace, galsim.OpticalPSF points to galsim.phase_psf.OpticalPSF
# and not galsim.deprecated.optics.OpticalPSF.

from .deprecated.optics import (OpticalPSF, wavefront, wavefront_image, psf, psf_image,
        otf, otf_image, mtf, mtf_image, ptf, ptf_image)
