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

def PA_getXArray(self):
    """Deprecated method for getting x values"""
    depr('getXArray', 1.5, 'photon_array.x')
    return self.x

def PA_getYArray(self):
    """Deprecated method for getting y values"""
    depr('getYArray', 1.5, 'photon_array.y')
    return self.y

def PA_getFluxArray(self):
    """Deprecated method for getting flux values"""
    depr('getFluxArray', 1.5, 'photon_array.flux')
    return self.flux

def PA_getDXDZArray(self):
    """Deprecated method for getting dxdz values"""
    depr('getDXDZArray', 1.5, 'photon_array.dxdz')
    return self.dxdz

def PA_getDYDZArray(self):
    """Deprecated method for getting dydz values"""
    depr('getDYDZArray', 1.5, 'photon_array.dydz')
    return self.dydz

def PA_getWavelengthArray(self):
    """Deprecated method for getting wavelength values"""
    depr('getWavelengthArray', 1.5, 'photon_array.wavelength')
    return self.wavelength

def PA_getX(self,i):
    """Deprecated method for getting x values"""
    depr('getX(i)', 1.5, 'photon_array.x[i]')
    return self.x[i]

def PA_getY(self, i):
    """Deprecated method for getting y values"""
    depr('getY(i)', 1.5, 'photon_array.y[i]')
    return self.y[i]

def PA_getFlux(self, i):
    """Deprecated method for getting flux values"""
    depr('getFlux(i)', 1.5, 'photon_array.flux[i]')
    return self.flux[i]

def PA_getDXDZ(self, i):
    """Deprecated method for getting dxdz values"""
    depr('getDXDZ(i)', 1.5, 'photon_array.dxdz[i]')
    return self.dxdz[i]

def PA_getDYDZ(self, i):
    """Deprecated method for getting dydz values"""
    depr('getDYDZ(i)', 1.5, 'photon_array.dydz[i]')
    return self.dydz[i]

def PA_getWavelength(self, i):
    """Deprecated method for getting wavelength values"""
    depr('getWavelength(i)', 1.5, 'photon_array.wavelength[i]')
    return self.wavelength[i]

def PA_setPhoton(self, i, x, y, flux):
    """Deprecated method for setting x,y,flux values"""
    depr('setPhoton(i,x,y,flux)', 1.5,
         'pa.x[i] = x; pa.y[i] = y; pa.flux[i] = flux')
    self.x[i] = x
    self.y[i] = y
    self.flux[i] = flux
    
galsim.PhotonArray.getXArray = PA_getXArray
galsim.PhotonArray.getYArray = PA_getYArray
galsim.PhotonArray.getFluxArray = PA_getFluxArray
galsim.PhotonArray.getDXDZArray = PA_getDXDZArray
galsim.PhotonArray.getDYDZArray = PA_getDYDZArray
galsim.PhotonArray.getWavelengthArray = PA_getWavelengthArray

galsim.PhotonArray.getX = PA_getX
galsim.PhotonArray.getY = PA_getY
galsim.PhotonArray.getFlux = PA_getFlux
galsim.PhotonArray.getDXDZ = PA_getDXDZ
galsim.PhotonArray.getDYDZ = PA_getDYDZ
galsim.PhotonArray.getWavelength = PA_getWavelength

galsim.PhotonArray.setPhoton = PA_setPhoton
