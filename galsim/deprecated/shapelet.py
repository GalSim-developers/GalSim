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
    self._sbp = galsim._galsim.SBShapelet(sigma, self._sbp.getBVec())

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
    self._sbp = galsim._galsim.SBShapelet(self.sigma, bvec)

def Shapelet_setBVec(self,bvec):
    """Deprecated method to change the bvec"""
    depr('setBVec',1.1,'shapelet = galsim.Shapelet(sigma, order, bvec=bvec)')
    bvec_size = galsim.ShapeletSize(self.order)
    if len(bvec) != bvec_size:
        raise ValueError("bvec is the wrong size for the Shapelet order")
    import numpy
    bvec = galsim.shapelet.LVector(self.order,numpy.array(bvec))
    self._sbp = galsim._galsim.SBShapelet(self.sigma, bvec)

def Shapelet_setPQ(self,p,q,re,im=0.):
    """Deprecated method to change a single element (p,q)"""
    depr('setPQ',1.1,'bvec with correct values in the constructor')
    bvec = self._sbp.getBVec().copy()
    bvec.setPQ(p,q,re,im)
    self._sbp = galsim._galsim.SBShapelet(self.sigma, bvec)

def Shapelet_setNM(self,N,m,re,im=0.):
    """Deprecated method to change a single element (N,m)"""
    depr('setNM',1.1,'bvec with correct values in the constructor')
    bvec = self._sbp.getBVec().copy()
    bvec.setPQ((N+m)//2,(N-m)//2,re,im)
    self._sbp = galsim._galsim.SBShapelet(self.sigma, bvec)

def Shapelet_fitImage(self, image, center=None, normalization='flux'):
    """A deprecated method that is roughly equivalent to
    self = galsim.FitShapelet(self.sigma, self.order, image)
    """
    depr('fitImage', 1.1, 'galsim.FitShapelet')
    new_obj = galsim.FitShapelet(self.sigma, self.order, image, center, normalization)
    bvec = new_obj._sbp.getBVec()
    self._sbp = galsim._galsim.SBShapelet(self.sigma, bvec)

def FitShapelet(sigma, order, image, center=None, normalization='flux', gsparams=None):
    "Deprecated function equivalent to Shapelet.fit"
    depr("FitShapelet", 1.5, 'galsim.Shapelet.fit(...)')
    return galsim.Shapelet.fit(sigma, order, image, center, normalization, gsparams)

def ShapeletSize(order):
    "Deprecated function equivalent to Shapelet.size"
    depr("ShapeletSize", 1.5, 'galsim.Shapelet.size(order)')
    return galsim.Shapelet.size(order)

galsim.LVectorSize = LVectorSize
galsim.ShapeletSize = ShapeletSize
galsim.FitShapelet = FitShapelet
galsim.Shapelet.setSigma = Shapelet_setSigma
galsim.Shapelet.setOrder = Shapelet_setOrder
galsim.Shapelet.setBVec = Shapelet_setBVec
galsim.Shapelet.setPQ = Shapelet_setPQ
galsim.Shapelet.setNM = Shapelet_setNM
galsim.Shapelet.fitImage = Shapelet_fitImage
