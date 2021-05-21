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

from ..errors import GalSimDeprecationWarning

def depr(f, v, s1, s2=None):
    """A helper function for emitting a GalSimDeprecationWarning.

    Example usage:

    1. Normally, you would simply warn that a deprecated function has a new preferred syntax:

            depr('applyShear', 1.1, 'obj = obj.shear(..)')

    2. You can add extra information if you want to point out something about the new syntax:

            depr('draw', 1.1, "drawImage(..., method='no_pixel')",
                 'Note: drawImage has different args than draw did.  '
                 'Read the docs for the method keywords carefully.')

    3. If the deprecated function has no replacement, you can use '' for the first string.

            depr('calculateCovarianceMatrix', 1.3, '',
                 'This functionality has been removed. If you have a need for it, please open '
                 'an issue requesting the functionality.')
    """
    import warnings
    s = str(f)+' has been deprecated since GalSim version '+str(v)+'.'
    if s1:
        s += '  Use '+s1+' instead.'
    if s2:
        s += '  ' + s2
    warnings.warn(s, GalSimDeprecationWarning)

from . import correlatednoise
from . import randwalk
from . import integ
