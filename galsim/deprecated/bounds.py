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

def Bounds_setXMin(self, xmin):
    """Deprecated method for setting the value of xmin.
    """
    depr('setXMin',1.1,
         'bounds = galsim.'+self.__class__.__name__+'(xmin,bounds.xmax,bounds.ymin,bounds.ymax)')
    self._setXMin(xmin)

def Bounds_setXMax(self, xmax):
    """Deprecated method for setting the value of xmax.
    """
    depr('setXMax',1.1,
         'bounds = galsim.'+self.__class__.__name__+'(bounds.xmin,xmax,bounds.ymin,bounds.ymax)')
    self._setXMax(xmax)

def Bounds_setYMin(self, ymin):
    """Deprecated method for setting the value of ymin.
    """
    depr('setYMin',1.1,
         'bounds = galsim.'+self.__class__.__name__+'(bounds.xmin,bounds.xmax,ymin,bounds.ymax)')
    self._setYMin(ymin)

def Bounds_setYMax(self, ymax):
    """Deprecated method for setting the value of ymax.
    """
    depr('setYMax',1.1,
         'bounds = galsim.'+self.__class__.__name__+'(bounds.xmin,bounds.xmax,bounds.ymin,ymax)')
    self._setYMax(ymax)

def Bounds_addBorder(self, border):
    """Deprecated name for the current withBorder.
    """
    depr('addBorder', 1.3, 'withBorder')
    return self.withBorder(border)

def Bounds_getXMin(self):
    depr('bounds.getXMin()', 1.5, 'bounds.xmin')
    return self.xmin

def Bounds_getXMax(self):
    depr('bounds.getXMax()', 1.5, 'bounds.xmax')
    return self.xmax

def Bounds_getYMin(self):
    depr('bounds.getYMin()', 1.5, 'bounds.ymin')
    return self.ymin

def Bounds_getYMax(self):
    depr('bounds.getYMax()', 1.5, 'bounds.ymax')
    return self.ymax

for Class in (galsim._galsim.BoundsD, galsim._galsim.BoundsI):
    Class.setXMin = Bounds_setXMin
    Class.setXMax = Bounds_setXMax
    Class.setYMin = Bounds_setYMin
    Class.setYMax = Bounds_setYMax
    Class.addBorder = Bounds_addBorder
    Class.getXMin = Bounds_getXMin
    Class.getXMax = Bounds_getXMax
    Class.getYMin = Bounds_getYMin
    Class.getYMax = Bounds_getYMax

del Class    # cleanup public namespace

