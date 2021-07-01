# Copyright (c) 2012-2021 by the GalSim developers team on GitHub
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

# We used to support legacy pyfits in addition to astropy.io.fits.  We still call
# astropy.io.fits pyfits in the code, but we have removed the legacy compatibility hacks.

import sys
import astropy.io.fits as pyfits

if 'PyPy' in sys.version:  # pragma: no cover
    # As of astropy version 4.2.1, the memmap stuff didn't work with PyPy, since it
    # needed getrefcount, which PyPy doesn't have.
    pyfits.Conf.use_memmap = False
