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

def Shear_setE1E2(self, e1=0.0, e2=0.0):
    """Deprecated method for setting the Shear to have some e1,e2"""
    depr('setE1E2',1.1,'shear = galsim.Shear(e1=e1,e2=e2)')
    new_shear = galsim.Shear(e1=e1, e2=e2)
    self._g = new_shear._g

def Shear_setEBeta(self, e=0.0, beta=None):
    """Deprecated method for setting the Shear to have some e,beta"""
    depr('setEBeta',1.1,'shear = galsim.Shear(e=e,beta=beta)')
    new_shear = galsim.Shear(e=e, beta=beta)
    self._g = new_shear._g

def Shear_setEta1Eta2(self, eta1=0.0, eta2=0.0):
    """Deprecated method for setting the Shear to have some eta1,eta2"""
    depr('setEta1Eta2',1.1,'shear = galsim.Shear(eta1=eta1,eta2=eta2)')
    new_shear = galsim.Shear(eta1=eta1, eta2=eta2)
    self._g = new_shear._g

def Shear_setEtaBeta(self, eta=0.0, beta=None):
    """Deprecated method for setting the Shear to have some eta,beta"""
    depr('setEtaBeta',1.1,'shear = galsim.Shear(e=e,beta=beta)')
    new_shear = galsim.Shear(eta=eta, beta=beta)
    self._g = new_shear._g

def Shear_setG1G2(self, g1=0.0, g2=0.0):
    """Deprecated method for setting the Shear to have some g1,g2"""
    depr('setG1G2',1.1,'shear = galsim.Shear(g1=g1,g2=g2)')
    new_shear = galsim.Shear(g1=g1, g2=g2)
    self._g = new_shear._g

galsim.Shear.setE1E2 = Shear_setE1E2
galsim.Shear.setEBeta = Shear_setEBeta
galsim.Shear.setEta1Eta2 = Shear_setEta1Eta2
galsim.Shear.setEtaBeta = Shear_setEtaBeta
galsim.Shear.setG1G2 = Shear_setG1G2

