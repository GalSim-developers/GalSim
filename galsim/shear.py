# Copyright 2012, 2013 The GalSim developers:
# https://github.com/GalSim-developers
#
# This file is part of GalSim: The modular galaxy image simulation toolkit.
#
# GalSim is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# GalSim is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with GalSim.  If not, see <http://www.gnu.org/licenses/>
#
"""@file shear.py 
Redefinition of the Shear class at the Python layer.
"""

from . import _galsim

class Shear(object):
    """A class to represent shears in a variety of ways.

    The python Shear class (galsim.Shear) can be initialized in a variety of ways to represent shape
    distortions.  A shear is an operation that transforms a circle into an ellipse with
    minor-to-major axis ratio b/a, with position angle beta, while conserving the area.  Given the
    multiple definitions of ellipticity, we have multiple definitions of shear:

    reduced shear |g| = (a - b)/(a + b)
    distortion |e| = (a^2 - b^2)/(a^2 + b^2)
    conformal shear eta, with a/b = exp(eta)
    minor-to-major axis ratio q = b/a

    These can be thought of as a magnitude and a real-space position angle beta, or as two
    components, e.g., `g1` and `g2`, with

    `g1` = |g| cos(2*`beta`)
    `g2` = |g| sin(2*`beta`)

    Note: `beta` is _not_ the phase of a complex valued shear.  Rather, the complex shear is 
    `g1` + i `g2` = `g` exp(2 i `beta`).  Likewise for `eta` or `e`.  The phase of the complex value
    is twice `beta`.

    The following are all examples of valid calls to initialize a Shear object:
    
        >>> s = galsim.Shear()                    # empty constructor sets ellipticity/shear to zero
        >>> s = galsim.Shear(g1=0.05, g2=0.05)
        >>> s = galsim.Shear(g1=0.05)             # assumes g2=0
        >>> s = galsim.Shear(e1=0.05, e2=0.05)
        >>> s = galsim.Shear(e2=0.05)             # assumes e1=0
        >>> s = galsim.Shear(eta1=0.07, eta2=-0.1)
        >>> s = galsim.Shear(eta=0.05, beta=45.0*galsim.degrees)
        >>> s = galsim.Shear(g=0.05, beta=0.25*numpy.pi*galsim.radians)
        >>> s = galsim.Shear(e=0.3, beta=30.0*galsim.degrees)
        >>> s = galsim.Shear(q=0.5, beta=0.0*galsim.radians)

    There can be no mixing and matching, e.g., specifying `g1` and `e2`.  It is permissible to only
    specify one of two components, with the other assumed to be zero.  If a magnitude such as `e`, 
    `g`, `eta`, or `q` is specified, then `beta` is also required to be specified.  It is possible 
    to initialize a Shear with zero reduced shear by specifying no args or kwargs, i.e. 
    galsim.Shear().  The galsim.Shear contains a C++ CppShear object, and operations on Shear rely 
    on wrapped methods of the CppShear.
    """
    def __init__(self, *args, **kwargs):
        import numpy as np

        # unnamed arg has to be a _CppShear
        if len(args) == 1:
            if isinstance(args[0], _galsim._CppShear):
                self._shear = args[0]
            else:
                raise TypeError("Unnamed argument to initialize Shear must be a _CppShear!")
        elif len(args) > 1:
            raise TypeError("Too many unnamed arguments to initialize Shear: %d"%len(args))
        else:

            # There is no valid set of >2 keyword arguments, so raise an exception in this case:
            if len(kwargs) > 2:
                raise TypeError(
                    "Shear constructor received >2 keyword arguments: %s"%kwargs.keys())

            # Since there are no args, check the keyword args: if a component of e, g, or eta, then
            # require that the other component is zero if not set, and don't allow specification of
            # mixed pairs like e1 and g2.  Also, require also a position angle if we didn't get
            # g1/g2, e1/e2, or eta1/eta2

            # first case: an empty constructor (no args/kwargs)
            if not kwargs:
                use_shear = _galsim._CppShear(0.0, 0.0)
            elif 'g1' in kwargs or 'g2' in kwargs:
                g1 = kwargs.pop('g1', 0.)
                g2 = kwargs.pop('g2', 0.)
                g = np.sqrt(g1**2 + g2**2)
                if g < 1:
                    use_shear = _galsim._CppShear(g1, g2)
                else:
                    raise ValueError("Requested shear exceeds 1: %f"%g)
            elif 'e1' in kwargs or 'e2' in kwargs:
                e1 = kwargs.pop('e1', 0.)
                e2 = kwargs.pop('e2', 0.)
                e = np.sqrt(e1**2 + e2**2)
                if e < 1:
                    use_shear = _galsim._CppShear()
                    use_shear.setE1E2(e1, e2)
                else:
                    raise ValueError("Requested distortion exceeds 1: %s"%e)
            elif 'eta1' in kwargs or 'eta2' in kwargs:
                eta1 = kwargs.pop('eta1', 0.)
                eta2 = kwargs.pop('eta2', 0.)
                use_shear = _galsim._CppShear()
                use_shear.setEta1Eta2(eta1, eta2)
            elif 'g' in kwargs:
                if 'beta' not in kwargs:
                    raise TypeError(
                        "Shear constructor requires position angle when g is specified!") 
                beta = kwargs.pop('beta')
                if not isinstance(beta, _galsim.Angle):
                    raise TypeError(
                        "The position angle that was supplied is not an Angle instance!")
                g = kwargs.pop('g')
                if g > 1 or g < 0:
                    raise ValueError("Requested |shear| is outside [0,1]: %f"%g)
                g1 = np.cos(2.*beta.rad())*g
                g2 = np.sin(2.*beta.rad())*g
                use_shear = _galsim._CppShear(g1, g2)
            elif 'e' in kwargs:
                if 'beta' not in kwargs:
                    raise TypeError(
                        "Shear constructor requires position angle when e is specified!")
                beta = kwargs.pop('beta')
                if not isinstance(beta, _galsim.Angle):
                    raise TypeError(
                        "The position angle that was supplied is not an Angle instance!")
                e = kwargs.pop('e')
                if e > 1 or e < 0:
                    raise ValueError("Requested distortion is outside [0,1]: %f"%e)
                use_shear = _galsim._CppShear()
                use_shear.setEBeta(e, beta)
            elif 'eta' in kwargs:
                if 'beta' not in kwargs:
                    raise TypeError(
                        "Shear constructor requires position angle when eta is specified!")
                beta = kwargs.pop('beta')
                if not isinstance(beta, _galsim.Angle):
                    raise TypeError(
                        "The position angle that was supplied is not an Angle instance!")
                eta = kwargs.pop('eta')
                if eta < 0:
                    raise ValueError("Requested eta is below 0: %f"%e)
                use_shear = _galsim._CppShear()
                use_shear.setEtaBeta(eta, beta)
            elif 'q' in kwargs:
                if 'beta' not in kwargs:
                    raise TypeError(
                        "Shear constructor requires position angle when q is specified!")
                beta = kwargs.pop('beta')
                if not isinstance(beta, _galsim.Angle):
                    raise TypeError(
                        "The position angle that was supplied is not an Angle instance!")
                q = kwargs.pop('q')
                if q <= 0 or q > 1:
                    raise ValueError("Cannot use requested axis ratio of %f!"%q)
                use_shear = _galsim._CppShear()
                eta = -np.log(q)
                use_shear.setEtaBeta(eta, beta)
            elif 'beta' in kwargs:
                raise TypeError("beta provided to Shear constructor, but not g/e/eta/q")

            # check for the case where there are 1 or 2 kwargs that are not valid ones for
            # initialization a Shear
            if kwargs:
                raise TypeError(
                    "Shear constructor got unexpected extra argument(s): %s"%kwargs.keys())

            self._shear = use_shear

    # below, we propagate through all the methods from C++

    # define all the methods for setting shear values
    def setE1E2(self, e1=0.0, e2=0.0): self._shear.setE1E2(e1, e2)
    def setEBeta(self, e=0.0, beta=None): self._shear.setEBeta(e, beta)
    def setEta1Eta2(self, eta1=0.0, eta2=0.0): self._shear.setEta1Eta2(eta1, eta2)
    def setEtaBeta(self, eta=0.0, beta=None): self._shear.setEtaBeta(eta, beta)
    def setG1G2(self, g1=0.0, g2=0.0): self._shear.setG1G2(g1, g2)
    # define all the methods to get shear values
    def getE1(self): return self._shear.getE1()
    def getE2(self): return self._shear.getE2()
    def getE(self): return self._shear.getE()
    def getESq(self): return self._shear.getESq()
    def getBeta(self): return self._shear.getBeta()
    def getEta(self): return self._shear.getEta()
    def getG(self): return self._shear.getG()
    def getG1(self): return self._shear.getG1()
    def getG2(self): return self._shear.getG2()
    # make it possible to access g, e, etc. of some Shear object called name using name.g, name.e
    e1 = property(getE1)
    e2 = property(getE2)
    e = property(getE)
    esq = property(getESq)
    beta = property(getBeta)
    eta = property(getEta)
    g = property(getG)
    g1 = property(getG1)
    g2 = property(getG2)
    # define all the various operators on Shear objects
    def __neg__(self): return Shear(-self._shear)
    # order of operations: shear by other._shear, then by self._shear
    def __add__(self, other): return Shear(self._shear + other._shear)
    # order of operations: shear by -other._shear, then by self._shear
    def __sub__(self, other): return Shear(self._shear - other._shear)
    def __iadd__(self, other):
        self._shear += other._shear
        return self
    def __isub__(self, other):
        self._shear -= other._shear
        return self
    def rotationWith(self, other): return self._shear.rotationWith(other)
    def __eq__(self, other): return self._shear == other._shear
    def __ne__(self, other): return self._shear != other._shear
    def fwd(self, p): return self._shear.fwd(p)
    def inv(self, p): return self._shear.inv(p)
    def getMatrix(self): return self._shear.getMatrix()

    def __repr__(self):
        return (self.__class__.__name__+"(g1="+str(self.getG1())+", g2="+str(self.getG2())+")")

    def __str__(self):
        return ("("+str(self.getG1())+", "+str(self.getG2())+")")

