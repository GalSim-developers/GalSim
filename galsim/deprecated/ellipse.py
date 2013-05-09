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
"""@file ellipse.py
A few adjustments to the Ellipse class at the Python layer.
"""

import galsim


class Ellipse(object):
    """A class to represent ellipses in a variety of ways.

    The galsim Ellipse class (galsim.Ellipse) represents a shear (shape distortion), dilation,
    and/or centroid shift.

    The python Ellipse class can be initialized in a variety of ways.  Unnamed arguments must be a
    galsim.Shear object for shape distortion (see galsim.shear.Shear for doxygen documentation), a
    float for dilation (via the mu parameter), and/or a Position for centroid shift.  Some examples
    are listed below.

    Keyword arguments can be used to set parameters of the shape distortion the same as for the
    Shear class; or the parameter `dilation` can be used for re-sizing; or the parameters
    `x_shift` and `y_shift` can be used for centroid shifts.  The galsim.Ellipse contains a C++
    CppEllipse object, and operations on Ellipse rely on wrapped methods of the CppEllipse.

    The following are all examples of valid calls to initialize a Ellipse object:
    
        >>> s = galsim.Shear(g1=0.05, g2=0.05)
        >>> shift = galsim.PositionD(0.0, 0.2)
        >>> ell = galsim.Ellipse()            # an empty ellipse, i.e. no shear, dilation, shift
        >>> ell = galsim.Ellipse(s)           # represents shearing by s only
        >>> ell = galsim.Ellipse(shear = s)   # same as previous, but with keyword explicitly named
        >>> ell = galsim.Ellipse(s, shift)       # shear and shift
        >>> ell = galsim.Ellipse(shift, s)       # can specify the arguments in any order
        >>> ell = galsim.Ellipse(s, y_shift=0.2) # same as previous, specifying the y shift directly
        >>> ell = galsim.Ellipse(mu=0.0, shear=s)         # no dilation, but shear by s
        >>> ell = galsim.Ellipse(shift, g1=0.05, g2=0.05) # arguments can be used to specify a shear
        >>> ell = galsim.Ellipse(mu=0.5, g=0.5, beta=45.0*galsim.degrees) # dilation, shear via
                                                                          # keyword argument
"""
    def __init__(self, *args, **kwargs):
        use_mu = None
        use_shear = None
        use_shift = None

        # check unnamed args: can have a Shear, float, and/or Position
        if len(args) > 0:
            # very special case: if it is given a wrapped C++ Ellipse
            if len(args) == 1 and isinstance(args[0], galsim._galsim._CppEllipse):
                self._ellipse = args[0]
            # there are args that are not a C++ Ellipse, so we have to process them by checking for
            # one of the allowed types
            else:
                for this_arg in args:
                    if isinstance(this_arg, galsim.Shear):
                        if use_shear is not None:
                            raise TypeError("Ellipse received >1 unnamed Shear arguments!")
                        use_shear = this_arg
                    elif isinstance(this_arg, float):
                        if use_mu is not None:
                            raise TypeError("Ellipse received >1 unnamed float/double arguments!")
                        use_mu = this_arg
                    elif isinstance(this_arg, galsim._galsim.PositionD):
                        if use_shift is not None:
                            raise TypeError("Ellipse received >1 unnamed Position arguments!")
                        use_shift = this_arg
                    else:
                        raise TypeError(
                            "Ellipse received an unnamed argument of a type that is not permitted!")

        # If no args, check kwargs: we start by checking for dilation or shifts, because there is a
        # limited set of allowed keyword arguments to specify those, whereas for shear there are
        # many allowed keyword arguments (see documentation for Shear)
        if use_mu is None:
            use_mu = kwargs.pop('mu', 0.0)
        if use_shift is None:
            x_shift = kwargs.pop('x_shift', 0.0)
            y_shift = kwargs.pop('y_shift', 0.0)
            use_shift = galsim._galsim.PositionD(x_shift, y_shift)
        if use_shear is None:
            use_shear = kwargs.pop('shear', None)
            if use_shear is None:
                if kwargs:
                    use_shear = galsim.Shear(**kwargs)
                else:
                    use_shear = galsim.Shear()
            else:
                if not isinstance(use_shear, galsim.Shear):
                    raise TypeError("Shear passed to Ellipse constructor was not a Shear!")
            # if shear was passed using the 'shear' keyword, then we should not allow any other args
                if kwargs:
                    raise TypeError("Keyword arguments to Ellipse not permitted: %s"%kwargs.keys())
        else:
            # if shear was specified as an unnamed argument, make sure there aren't also keyword
            # arguments that specify shear
            if kwargs:
                raise TypeError("Keyword arguments to Ellipse not permitted: %s"%kwargs.keys())     

        self._ellipse = galsim._galsim._CppEllipse(s = use_shear._shear, mu = use_mu, p = use_shift)

    # below, we propagate through all the methods from C++ #

    # define all the various operators on Ellipse objects
    def __neg__(self): return Ellipse(-self._ellipse)
    def __add__(self, other): return Ellipse(self._ellipse + other._ellipse)
    def __sub__(self, other): return Ellipse(self._ellipse - other._ellipse)
    def __iadd__(self, other): self._ellipse += other._ellipse
    def __isub__(self, other): self._ellipse -= other._ellipse
    def __eq__(self, other): return self._ellipse == other._ellipse
    def __ne__(self, other): return self._ellipse != other._ellipse
    def reset(self, s, mu, p): self._ellipse.reset(s._shear, mu, p)
    def fwd(self, p): return self._ellipse.fwd(p)
    def inv(self, p): return self._ellipse.inv(p)
    # methods for setting values
    def setS(self, s): self._ellipse.setS(s._shear)
    def setMu(self, mu): self._ellipse.setMu(mu)
    def setX0(self, p): self._ellipse.setX0(p)
    # methods for getting values
    def getS(self): return galsim.Shear(self._ellipse.getS())
    def getMu(self): return self._ellipse.getMu()
    def getX0(self): return self._ellipse.getX0()
    def getMajor(self): return self._ellipse.getMajor()
    def getMinor(self): return self._ellipse.getMinor()
    def getBeta(self): return self._ellipse.getBeta()
    def range(self): return self._ellipse.range()
    def getMatrix(self): return self._ellipse.getMatrix()
    # or access values directly
    s = property(getS)
    mu = property(getMu)
    x0 = property(getX0)
    major = property(getMajor)
    minor = property(getMinor)
    beta = property(getBeta)
    range = property(range)

    def __repr__(self):
        shear = self.getS()  # extract the e1 and e2 from the Shear instance
        x0 = self.getX0()    # extract the x0 and y0 from a Position instance
        return (
            self.__class__.__name__+"(g1="+str(shear.getG1())+", g2="+str(shear.getG2())+
            ", mu="+str(self.getMu())+", x_shift="+str(x0.x)+", y_shift="+str(x0.y)+")")

    def __str__(self):
        shear = self.getS()  # extract the e1 and e2 from the Shear instance
        x0 = self.getX0()    # extract the x0 and y0 from a Position instance
        return (
            "("+str(shear.getG1())+", "+str(shear.getG2())+", "+str(self.getMu())+", "
            +str(x0.x)+", "+str(x0.y)+")")

