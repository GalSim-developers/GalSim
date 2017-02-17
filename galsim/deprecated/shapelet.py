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

def LVectorSize(order):
    """A deprecated synonym for ShapeletSize"""
    depr('LVectorSize', 1.1, 'ShapeletSize')
    return galsim.ShapeletSize(order)

def Shapelet_setSigma(self,sigma):
    """Deprecated method to change the value of sigma"""
    depr('setSigma',1.1,'shapelet = galsim.Shapelet(sigma, order, ...)')
    galsim.GSObject.__init__(self, galsim._galsim.SBShapelet(sigma, self.SBProfile.getBVec()))

def Shapelet_setOrder(self,order):
    """Deprecated method to change the order"""
    depr('setOrder',1.1,'shapelet = galsim.Shapelet(sigma, order, ...)')
    if self.order == order: return
    # Preserve the existing values as much as possible.
    if self.order > order:
        bvec = galsim.shapelet.LVector(order, self.bvec[0:galsim.ShapeletSize(order)])
    else:
        import numpy
        a = numpy.zeros(galsim.ShapeletSize(order))
        a[0:len(self.bvec)] = self.bvec
        bvec = galsim.shapelet.LVector(order,a)
    galsim.GSObject.__init__(self, galsim._galsim.SBShapelet(self.sigma, bvec))

def Shapelet_setBVec(self,bvec):
    """Deprecated method to change the bvec"""
    depr('setBVec',1.1,'shapelet = galsim.Shapelet(sigma, order, bvec=bvec)')
    bvec_size = galsim.ShapeletSize(self.order)
    if len(bvec) != bvec_size:
        raise ValueError("bvec is the wrong size for the Shapelet order")
    import numpy
    bvec = galsim.shapelet.LVector(self.order,numpy.array(bvec))
    galsim.GSObject.__init__(self, galsim._galsim.SBShapelet(self.sigma, bvec))

def Shapelet_setPQ(self,p,q,re,im=0.):
    """Deprecated method to change a single element (p,q)"""
    depr('setPQ',1.1,'bvec with correct values in the constructor')
    bvec = self.SBProfile.getBVec().copy()
    bvec.setPQ(p,q,re,im)
    galsim.GSObject.__init__(self, galsim._galsim.SBShapelet(self.sigma, bvec))

def Shapelet_setNM(self,N,m,re,im=0.):
    """Deprecated method to change a single element (N,m)"""
    depr('setNM',1.1,'bvec with correct values in the constructor')
    bvec = self.SBProfile.getBVec().copy()
    bvec.setPQ((N+m)//2,(N-m)//2,re,im)
    galsim.GSObject.__init__(self, galsim._galsim.SBShapelet(self.sigma, bvec))

def Shapelet_fitImage(self, image, center=None, normalization='flux'):
    """A deprecated method that is roughly equivalent to
    self = galsim.FitShapelet(self.sigma, self.order, image)
    """
    depr('fitImage', 1.1, 'galsim.FitShapelet')
    new_obj = galsim.FitShapelet(self.sigma, self.order, image, center, normalization)
    bvec = new_obj.SBProfile.getBVec()
    galsim.GSObject.__init__(self, galsim._galsim.SBShapelet(self.sigma, bvec))

galsim.LVectorSize = LVectorSize
galsim.Shapelet.setSigma = Shapelet_setSigma
galsim.Shapelet.setOrder = Shapelet_setOrder
galsim.Shapelet.setBVec = Shapelet_setBVec
galsim.Shapelet.setPQ = Shapelet_setPQ
galsim.Shapelet.setNM = Shapelet_setNM
galsim.Shapelet.fitImage = Shapelet_fitImage
