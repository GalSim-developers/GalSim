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

def HMS_Angle(str):
    """Deprecated function equivalent to Angle.hms"""
    depr("HMS_Angle", 1.5, "Angle.from_hms")
    return galsim.angle.parse_dms(str) * galsim.hours

def DMS_Angle(str):
    """Deprecated function equivalent to Angle.dms"""
    depr("DMS_Angle", 1.5, "Angle.from_dms")
    return galsim.angle.parse_dms(str) * galsim.degrees

galsim.HMS_Angle = HMS_Angle
galsim.DMS_Angle = DMS_Angle

