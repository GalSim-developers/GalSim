# Copyright (c) 2012-2018 by the GalSim developers team on GitHub
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
"""@file shear.py
Redefinition of the Shear class at the Python layer.
"""

import numpy as np

from .angle import Angle, _Angle, radians
from .errors import GalSimRangeError, GalSimIncompatibleValuesError

class Shear(object):
    """A class to represent shears in a variety of ways.

    The python Shear class (galsim.Shear) can be initialized in a variety of ways to represent shape
    distortions.  A shear is an operation that transforms a circle into an ellipse with
    minor-to-major axis ratio b/a, with position angle beta, while conserving the area (see
    below for a discussion of the implications of this choice).  Given the multiple definitions of
    ellipticity, we have multiple definitions of shear:

    reduced shear |g| = (a - b)/(a + b)
    distortion |e| = (a^2 - b^2)/(a^2 + b^2)
    conformal shear eta, with a/b = exp(eta)
    minor-to-major axis ratio q = b/a

    These can be thought of as a magnitude and a real-space position angle beta, or as two
    components, e.g., g1 and g2, with

    g1 = |g| cos(2*beta)
    g2 = |g| sin(2*beta)

    Note: beta is _not_ the phase of a complex valued shear.  Rather, the complex shear is
    g1 + i g2 = g exp(2 i beta).  Likewise for eta or e.  The phase of the complex value is 2 beta.

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
    galsim.Shear().

    In addition, for use cases where extreme efficiency is required, you can skip all the
    normal sanity checks and branches in the regular Shear constructor by using a leading
    underscore with the complex shear (g1 + 1j * g2).

        >>> s = galsim._Shear(0.05 + 0.03j)  # Equivalent to galsim.Shear(g1=0.05, g2=0.03)

    Since we have defined a Shear as a transformation that preserves area, this means that it is not
    a precise description of what happens during the process of weak lensing.  The coordinate
    transformation that occurs during the actual weak lensing process is such that if a galaxy is
    sheared by some `(gamma_1, gamma_2)`, and then sheared by `(-gamma_1, -gamma_2)`, it will in the
    end return to its original shape, but will have changed in area due to the magnification, `mu =
    1/((1.-kappa)**2 - (gamma_1**2 + gamma_2**2))`, which is not equal to one for non-zero shear
    even for convergence `kappa=0`.  Application of a Shear using the GSObject.shear() method does
    not include this area change.  To properly incorporate the effective change in area due to
    shear, it is necessary to either (a) define the Shear object, use the shear() method, and
    separately use the magnify() method, or (b) use the lens() method that simultaneously magnifies
    and shears.
    """
    def __init__(self, *args, **kwargs):

        # There is no valid set of >2 keyword arguments, so raise an exception in this case:
        if len(kwargs) > 2:
            raise TypeError(
                "Shear constructor received >2 keyword arguments: %s"%kwargs.keys())

        if len(args) > 1:
            raise TypeError(
                "Shear constructor received >1 non-keyword arguments: %s"%args)

        # If a component of e, g, or eta, then require that the other component is zero if not set,
        # and don't allow specification of mixed pairs like e1 and g2.
        # Also, require a position angle if we didn't get g1/g2, e1/e2, or eta1/eta2

        # Unnamed arg must be a complex shear
        if len(args) == 1:
            self._g = args[0]
            if not isinstance(self._g, complex):
                raise TypeError("Non-keyword argument to Shear must be complex g1 + 1j * g2")

        # Empty constructor means shear == (0,0)
        elif not kwargs:
            self._g = 0j

        # g1,g2
        elif 'g1' in kwargs or 'g2' in kwargs:
            g1 = kwargs.pop('g1', 0.)
            g2 = kwargs.pop('g2', 0.)
            self._g = g1 + 1j * g2
            if abs(self._g) > 1.:
                raise GalSimRangeError("Requested shear exceeds 1.", self._g, 0., 1.)

        # e1,e2
        elif 'e1' in kwargs or 'e2' in kwargs:
            e1 = kwargs.pop('e1', 0.)
            e2 = kwargs.pop('e2', 0.)
            absesq = e1**2 + e2**2
            if absesq > 1.:
                raise GalSimRangeError("Requested distortion exceeds 1.",np.sqrt(absesq), 0., 1.)
            self._g = (e1 + 1j * e2) * self._e2g(absesq)

        # eta1,eta2
        elif 'eta1' in kwargs or 'eta2' in kwargs:
            eta1 = kwargs.pop('eta1', 0.)
            eta2 = kwargs.pop('eta2', 0.)
            eta = eta1 + 1j * eta2
            abseta = abs(eta)
            self._g = eta * self._eta2g(abseta)

        # g,beta
        elif 'g' in kwargs:
            if 'beta' not in kwargs:
                raise GalSimIncompatibleValuesError(
                    "Shear constructor requires beta when g is specified.",
                    g=kwargs['g'], beta=None)
            beta = kwargs.pop('beta')
            if not isinstance(beta, Angle):
                raise TypeError("beta must be an Angle instance.")
            g = kwargs.pop('g')
            if g > 1 or g < 0:
                raise GalSimRangeError("Requested |shear| is outside [0,1].",g, 0., 1.)
            self._g = g * np.exp(2j * beta.rad)

        # e,beta
        elif 'e' in kwargs:
            if 'beta' not in kwargs:
                raise GalSimIncompatibleValuesError(
                    "Shear constructor requires beta when e is specified.",
                    e=kwargs['e'], beta=None)
            beta = kwargs.pop('beta')
            if not isinstance(beta, Angle):
                raise TypeError("beta must be an Angle instance.")
            e = kwargs.pop('e')
            if e > 1 or e < 0:
                raise GalSimRangeError("Requested distortion is outside [0,1].", e, 0., 1.)
            self._g = self._e2g(e**2) * e * np.exp(2j * beta.rad)

        # eta,beta
        elif 'eta' in kwargs:
            if 'beta' not in kwargs:
                raise GalSimIncompatibleValuesError(
                    "Shear constructor requires beta when eta is specified.",
                    eta=kwargs['eta'], beta=None)
            beta = kwargs.pop('beta')
            if not isinstance(beta, Angle):
                raise TypeError("beta must be an Angle instance.")
            eta = kwargs.pop('eta')
            if eta < 0:
                raise GalSimRangeError("Requested eta is below 0.", eta, 0.)
            self._g = self._eta2g(eta) * eta * np.exp(2j * beta.rad)

        # q,beta
        elif 'q' in kwargs:
            if 'beta' not in kwargs:
                raise GalSimIncompatibleValuesError(
                    "Shear constructor requires beta when q is specified.",
                    q=kwargs['q'], beta=None)
            beta = kwargs.pop('beta')
            if not isinstance(beta, Angle):
                raise TypeError("beta must be an Angle instance.")
            q = kwargs.pop('q')
            if q <= 0 or q > 1:
                raise GalSimRangeError("Cannot use requested axis ratio.", q, 0., 1.)
            eta = -np.log(q)
            self._g = self._eta2g(eta) * eta * np.exp(2j * beta.rad)

        elif 'beta' in kwargs:
            raise GalSimIncompatibleValuesError(
                "beta provided to Shear constructor, but not g/e/eta/q",
                beta=kwargs['beta'], e=None, g=None, q=None, eta=None)

        # check for the case where there are 1 or 2 kwargs that are not valid ones for
        # initializing a Shear
        if kwargs:
            raise TypeError(
                "Shear constructor got unexpected extra argument(s): %s"%kwargs.keys())

    @property
    def g1(self): return self._g.real

    @property
    def g2(self): return self._g.imag
    @property
    def g(self): return abs(self._g)
    @property
    def beta(self): return _Angle(0.5 * np.angle(self._g))

    @property
    def shear(self): return self._g

    @property
    def e1(self): return self._g.real * self._g2e(self.g**2)
    @property
    def e2(self): return self._g.imag * self._g2e(self.g**2)
    @property
    def e(self): return self.g * self._g2e(self.g**2)
    @property
    def esq(self): return self.e**2

    @property
    def eta1(self): return self._g.real * self._g2eta(self.g)
    @property
    def eta2(self): return self._g.imag * self._g2eta(self.g)
    @property
    def eta(self): return self.g * self._g2eta(self.g)

    # Helpers to convert between different conventions
    # Note: These return the scale factor by which to multiply.  Not the final value.
    def _g2e(self, absgsq):
        return 2. / (1.+absgsq)

    def _e2g(self, absesq):
        if absesq > 1.e-4:
            #return (1. - np.sqrt(1.-absesq)) / absesq
            return 1. / (1. + np.sqrt(1.-absesq))
        else:
            # Avoid numerical issues near e=0 using Taylor expansion
            return 0.5 + absesq*(0.125 + absesq*(0.0625 + absesq*0.0390625))

    def _g2eta(self, absg):
        if absg > 1.e-4:
            return 2.*np.arctanh(absg)/absg
        else:
            # This doesn't have as much trouble with accuracy, but have to avoid absg=0,
            # so might as well Taylor expand for small values.
            absgsq = absg * absg
            return 2. + absgsq*((2./3.) + absgsq*0.4)

    def _eta2g(self, abseta):
        if abseta > 1.e-4:
            return np.tanh(0.5*abseta)/abseta
        else:
            absetasq = abseta * abseta
            return 0.5 + absetasq*((-1./24.) + absetasq*(1./240.))

    # define all the various operators on Shear objects
    def __neg__(self): return _Shear(-self._g)

    # order of operations: shear by other._shear, then by self._shear
    def __add__(self, other):
        return _Shear((self._g + other._g) / (1. + self._g.conjugate() * other._g))

    # order of operations: shear by -other._shear, then by self._shear
    def __sub__(self, other): return self + (-other)

    def __eq__(self, other):
        return self is other or (isinstance(other, Shear) and self._g == other._g)
    def __ne__(self, other): return not self.__eq__(other)

    def getMatrix(self):
        """Return the matrix that tells how this shear acts on a position vector:

        If a field is sheared by some shear, s, then the position (x,y) -> (x',y')
        according to:

        [ x' ] = S [ x ]
        [ y' ]     [ y ]

        where S = s.getMatrix().

        Specifically, the matrix is S = (1-g^2)^(-1/2) [ 1+g1 ,  g2  ]
                                                       [  g2  , 1-g1 ]
        """
        return np.array([[ 1.+self.g1,  self.g2   ],
                         [  self.g2  , 1.-self.g1 ]]) / np.sqrt(1.-self.g**2)

    def rotationWith(self, other):
        """Return the rotation angle associated with the addition of two shears.

        The effect of two shears is not just a single net shear.  There is also a rotation
        associated with it.  This is easiest to understand in terms of the matrix representations:

        If s3 = s1 + s2, and the corresponding shear matrices are S1,S2,S3, then S3 R = S1 S2,
        where R is a rotation matrix:

        R = [ cos(theta) , -sin(theta) ]
            [ sin(theta) ,  cos(theta) ]

        and theta is the return value (as a galsim.Angle) from s1.rotationWith(s2).
        """
        # Save a little time by only working on the first column.
        S3 = self.getMatrix().dot(other.getMatrix()[:,:1])
        R = (-(self + other)).getMatrix().dot(S3)
        theta = np.arctan2(R[1,0], R[0,0])
        return theta * radians

    def __repr__(self):
        return 'galsim.Shear(%r)'%(self.shear)

    def __str__(self):
        return 'galsim.Shear(g1=%s,g2=%s)'%(self.g1,self.g2)

    def __hash__(self): return hash(self._g)

def _Shear(g):
    """Equivalent to Shear(shear), but without the overhead of the normal sanity checks and other
    options for how to specify the shear.

    @param g        The complex shear g1 + 1j * g2.

    @returns a galsim.Shear instance
    """
    ret = Shear.__new__(Shear)
    ret._g = g
    return ret
