# Copyright (c) 2012-2023 by the GalSim developers team on GitHub
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
"""
GalSim: The modular galaxy image simulation toolkit

GalSim is open-source software for simulating images of astronomical objects
(stars, galaxies) in a variety of ways.  The bulk of the calculations are
carried out in C++, and the user interface is in python.  In addition, the code
can operate directly on "config" files, for those users who prefer not to work
in python.  The impetus for the software package was a weak lensing community
data challenge, called GREAT3:

https://github.com/barnabytprowe/great3-public

However, the code has numerous additional capabilities beyond those needed for
the challenge, and has been useful for a number of projects that needed to
simulate high-fidelity galaxy images with accurate sizes and shears.

For an overview of GalSim workflow and python tools, please see the file
`doc/GalSim_Quick_Reference.pdf` in the GalSim repository.  A guide to using
the configuration files to generate simulations, a FAQ for installation issues,
a link to Doxygen-generated documentation, and other useful references can be
found on the GalSim wiki,

https://github.com/GalSim-developers/GalSim/wiki

If you experience any issues with the software, or if you would like to file
a feature request, please do so on the main github site for GalSim,

https://github.com/GalSim-developers/GalSim

Finally, if you have questions about how to do something in GalSim, you may
ask a question on StackOverflow,

http://stackoverflow.com/

Use the galsim tag to flag it as a question about GalSim.



Copyright (c) 2012-2023 by the GalSim developers team on GitHub
https://github.com/GalSim-developers

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this
   list of conditions and the following disclaimer.
2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.

This software is made available to you on an ``as is'' basis with no
representations or warranties, express or implied, including but not
limited to any warranty of performance, merchantability, fitness for a
particular purpose, commercial utility, non-infringement or title.
Neither the authors nor the organizations providing the support under
which the work was developed will be liable to you or any third party
with respect to any claim arising from your further development of the
software or any products related to or derived from the software, or for
lost profits, business interruption, or indirect special or consequential
damages of any kind.
"""
import os

# The version is stored in _version.py as recommended here:
# http://stackoverflow.com/questions/458550/standard-way-to-embed-version-into-python-package
from ._version import __version__, __version_info__

# Define the current code version, in addition to the hidden attribute, to be consistent with
# previous GalSim versions that indicated the version number in this way.
version = __version__

# Tell people where the headers and lib are in case they want to use and link to the
# C++ layer directly.  (Basically copied from TreeCorr's __init__.py.)
galsim_dir = os.path.dirname(__file__)
include_dir = os.path.join(galsim_dir,'include')
from . import _galsim
lib_file = os.path.abspath(_galsim.__file__)

# Import things from other files we want to be in the galsim namespace
# More or less the order here is to import things after other modules they depend on.

# First some basic building blocks that don't usually depend on anything else
from .errors import *
from .position import *
from .bounds import *
from .angle import *
from .celestial import *
from .shear import *
from .catalog import *
from .table import *

# Image
from .image import *

# WCS
from .wcs import *
from .fits import *
from .fitswcs import *

# Noise
from .random import *
from .noise import *
from .correlatednoise import *

# PhotonArray
from .photon_array import *

# GSObject
from .gsobject import *
from .gsparams import *
from .gaussian import *
from .moffat import *
from .airy import *
from .kolmogorov import *
from .vonkarman import *
from .box import *
from .exponential import *
from .sersic import *
from .spergel import *
from .deltafunction import *
from .shapelet import *
from .inclined import *
from .knots import *

from .sum import *
from .convolve import *
from .transform import *
from .fouriersqrt import *

from .interpolant import *
from .interpolatedimage import *
from .real import *
from .galaxy_sample import *

from .phase_screens import *
from .second_kick import *
from .phase_psf import *

# Chromatic
from .sed import *
from .bandpass import *
from .chromatic import *

# Lensing stuff
from .lensing_ps import *
from .nfw_halo import *

# Detector effects
from .sensor import *
from . import detectors  # Everything here is a method of Image, so nothing to import by name.
from .utilities import set_omp_threads, get_omp_threads  # These we bring into the main scope.

# Deprecated functionality
from . import deprecated

# Packages we intentionally keep separate.  E.g. requires galsim.fits.read(...)
from . import fits
from . import config
from . import integ
from . import bessel
from . import pse
from . import hsm
from . import dcr
from . import meta_data
from . import cdmodel
from . import utilities
from . import fft
from . import download_cosmos
from . import zernike
