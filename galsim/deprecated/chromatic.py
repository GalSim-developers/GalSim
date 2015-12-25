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

import galsim
from galsim.deprecated import depr

def Chromatic_draw(self, *args, **kwargs):
    """A deprecated synonym for obj.drawImage(method='no_pixel')
    """
    depr('draw', 1.1, "drawImage(..., method='no_pixel'",
            'Note: drawImage has different args than draw did.  '+
            'Read the docs for the method keywords carefully.')
    normalization = kwargs.pop('normalization','f')
    if normalization in ['flux','f']:
        return self.drawImage(*args, method='no_pixel', **kwargs)
    else:
        return self.drawImage(*args, method='sb', **kwargs)

galsim.ChromaticObject.draw = Chromatic_draw
