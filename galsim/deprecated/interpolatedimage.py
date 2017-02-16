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

import galsim
from galsim.deprecated import depr

def InterpolantXY(arg):
    """A deprecated function for converting a 1d interpolant into a 2d interpolant.

    This is no longer needed.
    """
    depr('InterpolatedXY', 1.3, 'the 1-d Interpolant by itself')
    return arg

galsim.InterpolantXY = InterpolantXY

# Also make Interpolant2d an alias for Interpolant in case anyone does
# isinstance(interp, galsim.Interpolant2d)

galsim.Interpolant2d = galsim.Interpolant

