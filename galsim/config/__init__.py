# Copyright (c) 2012-2015 by the GalSim developers team on GitHub
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


# These have the basic config functionality that gets imported into galsim.config scope
from .process import *
from .input import *
from .output import *
from .extra import *
from .image import *
from .stamp import *
from .noise import *
from .wcs import *
from .gsobject import *
from .value import *

# These implement specific types and features that get registered into the main config
# apparatus.  The functions themselves are not available at galsim.config scope.


import extra_psf
import extra_weight
import extra_badpix
import extra_truth

import image_scattered
import image_tiled

import gsobject_ring
import gsobject_realgalaxy
import gsobject_cosmosgalaxy
