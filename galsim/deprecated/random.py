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

def GaussianDeviate_setMean(self, mean):
    """Deprecated method to set the mean.
    """
    depr('setMean', 1.1, 'rng = galsim.GaussianDeviate(rng, mean=mean, sigma=rng.sigma)')
    self._setMean(mean)

def GaussianDeviate_setSigma(self, sigma):
    """Deprecated method to set sigma.
    """
    depr('setSigma', 1.1, 'rng = galsim.GaussianDeviate(rng, mean=rng.mean, sigma=sigma)')
    self._setSigma(sigma)

def GaussianDeviate_getMean(self):
    depr('gd.getMean', 1.5, 'gd.mean')
    return self.mean

def GaussianDeviate_getSigma(self):
    depr('gd.getSigma', 1.5, 'gd.sigma')
    return self.sigma

galsim._galsim.GaussianDeviate.setMean = GaussianDeviate_setMean
galsim._galsim.GaussianDeviate.setSigma = GaussianDeviate_setSigma
galsim._galsim.GaussianDeviate.getMean = GaussianDeviate_getMean
galsim._galsim.GaussianDeviate.getSigma = GaussianDeviate_getSigma

def BinomialDeviate_setN(self, N):
    """Deprecated method to set N.
    """
    depr('setN', 1.1, 'rng = galsim.BinomialDeviate(rng, N=N, p=rng.p)')
    self._setN(N)

def BinomialDeviate_setP(self, p):
    """Deprecated method to set p.
    """
    depr('setP', 1.1, 'rng = galsim.BinomialDeviate(rng, N=rng.N, p=p)')
    self._setP(p)

def BinomialDeviate_getN(self):
    depr('bd.getN', 1.5, 'bd.n')
    return self.n

def BinomialDeviate_getP(self):
    depr('bd.getP', 1.5, 'bd.p')
    return self.p

galsim._galsim.BinomialDeviate.setN = BinomialDeviate_setN
galsim._galsim.BinomialDeviate.setP = BinomialDeviate_setP
galsim._galsim.BinomialDeviate.getN = BinomialDeviate_getN
galsim._galsim.BinomialDeviate.getP = BinomialDeviate_getP

def PoissonDeviate_setMean(self, mean):
    """Deprecated method to set the mean.
    """
    depr('setMean', 1.1, 'rng = galsim.PoissonDeviate(rng, mean=mean)')
    self._setMean(mean)

def PoissonDeviate_getMean(self):
    depr('pd.getMean', 1.5, 'pd.mean')
    return self.mean

galsim._galsim.PoissonDeviate.setMean = PoissonDeviate_setMean
galsim._galsim.PoissonDeviate.getMean = PoissonDeviate_getMean

def WeibullDeviate_setA(self, a):
    """Deprecated method to set a.
    """
    depr('setA', 1.1, 'rng = galsim.WeibullDeviate(rng, a=a, b=rng.b)')
    self._setA(a)

def WeibullDeviate_setB(self, b):
    """Deprecated method to set b.
    """
    depr('setB', 1.1, 'rng = galsim.WeibullDeviate(rng, a=rng.a, b=b)')
    self._setB(b)

def WeibullDeviate_getA(self):
    depr('wd.getA', 1.5, 'wd.a')
    return self.a

def WeibullDeviate_getB(self):
    depr('wd.getB', 1.5, 'wd.b')
    return self.b

galsim._galsim.WeibullDeviate.setA = WeibullDeviate_setA
galsim._galsim.WeibullDeviate.setB = WeibullDeviate_setB
galsim._galsim.WeibullDeviate.getA = WeibullDeviate_getA
galsim._galsim.WeibullDeviate.getB = WeibullDeviate_getB

def GammaDeviate_setK(self, k):
    """Deprecated method to set k.
    """
    depr('setK', 1.1, 'rng = galsim.GammaDeviate(rng, k=k, theta=rng.theta)')
    self._setK(k)

def GammaDeviate_setTheta(self, theta):
    """Deprecated method to set theta.
    """
    depr('setTheta', 1.1, 'rng = galsim.GammaDeviate(rng, k=rng.k, theta=theta)')
    self._setTheta(theta)

def GammaDeviate_getK(self):
    depr('gd.getK', 1.5, 'gd.k')
    return self.k

def GammaDeviate_getTheta(self):
    depr('gd.getTheta', 1.5, 'gd.theta')
    return self.theta

galsim._galsim.GammaDeviate.setK = GammaDeviate_setK
galsim._galsim.GammaDeviate.setTheta = GammaDeviate_setTheta
galsim._galsim.GammaDeviate.getK = GammaDeviate_getK
galsim._galsim.GammaDeviate.getTheta = GammaDeviate_getTheta

def Chi2Deviate_setN(self, n):
    """Deprecated method to set n.
    """
    depr('setN', 1.1, 'rng = galsim.Chi2Deviate(rng, n=n)')
    self._setN(n)

def Chi2Deviate_getN(self):
    depr('cd.getN', 1.5, 'cd.n')
    return self.n

galsim._galsim.Chi2Deviate.setN = Chi2Deviate_setN
galsim._galsim.Chi2Deviate.getN = Chi2Deviate_getN

