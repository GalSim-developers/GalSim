# Copyright (c) 2012-2014 by the GalSim developers team on GitHub
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

# Note: By default python2.7 ignores DeprecationWarnings.  Apparently they are really
#       for python system deprecations.  I think GalSim deprecations are more appropriately
#       considered UserWarnings, which are not ignored by default.
class GalSimDeprecationWarning(UserWarning):
    def __init__(self, s):
        super(GalSimDeprecationWarning, self).__init__(self, s)

def depr(f, v, s1, s2=None):
    import warnings
    s = str(f)+' has been deprecated since GalSim version '+str(v)+'.'
    if s1:
        s += '  Use '+s1+' instead.'
    if s2:
        s += '  ' + s2
    warnings.warn(s, GalSimDeprecationWarning)

from . import base
from . import bounds
from . import chromatic
from . import correlatednoise
from . import image
from . import noise
from . import random
from . import shapelet
from . import shear
