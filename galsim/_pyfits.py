# Copyright (c) 2012-2018 by the GalSim developers team on GitHub
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

# Make it so we can use either pyfits or astropy.io.fits as pyfits.

try:
    import astropy.io.fits as pyfits
    # astropy started their versioning over at 0.  (Understandably.)
    # To make this seamless with pyfits versions, we add 4 to the astropy version.
    from astropy import version as astropy_version
    pyfits_version = str( (4 + astropy_version.major) + astropy_version.minor/10.)
    pyfits_str = 'astropy.io.fits'
except ImportError:
    import pyfits
    pyfits_version = pyfits.__version__
    pyfits_str = 'pyfits'


