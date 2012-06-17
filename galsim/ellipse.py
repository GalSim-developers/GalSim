"""@file ellipse.py @brief A few adjustments to the Ellipse class at the Python layer.
"""

from . import _galsim
import galsim

"""
@brief A class to represent ellipses in a variety of ways.

The Ellipse class represents a shear (shape distortion), dilation, and/or centroid shift.

The python Ellipse class can be initialized in a variety of ways.  Unnamed arguments must be a Shear
object for shape distortion, a float for dilation, and/or a Position for centroid shift.  Keyword
arguments can be used to set parameters of the shape distortion the same as for the Shear class; or
the parameter "dilation" can be used for re-sizing; or the parameters "x_shift" and "y_shift" can be
used for centroid shifts.

The following are all examples of valid calls to initialize a Ellipse object:
@code
s = galsim.Shear(g1=0.05, g2=0.05)
shift = galsim.PositionD(0.0, 0.2)
ell = galsim.Ellipse(s)
ell2 = galsim.Ellipse(s, shift)
ell3 = galsim.Ellipse(s, y_shift = 0.2)
ell4 = galsim.Ellipse(dilation = 0.0, shear = s)
@endcode
"""
class Ellipse:
    def __init__(self, *args, **kwargs):
        use_dil = None
        use_shear = None
        use_shift = None

        # check unnamed args: can have a Shear, float, and/or Position
        if len(args) > 0:
            # very special case: if it is given a wrapped C++ Ellipse
            if len(args) == 1 and isinstance(args[0], _galsim._Ellipse):
                self._ellipse = args[0]
            else:
                for this_arg in args:
                    if isinstance(this_arg, galsim.Shear):
                        if use_shear != None:
                            raise TypeError("Ellipse received >1 unnamed Shear arguments!")
                        use_shear = this_arg
                    elif isinstance(this_arg, float):
                        if use_dil != None:
                            raise TypeError("Ellipse received >1 unnamed float/double arguments!")
                        use_dil = this_arg
                    elif isinstance(this_arg, _galsim.PositionD):
                        if use_shift != None:
                            raise TypeError("Ellipse received >1 unnamed Position arguments!")
                        use_shift = this_arg
                    else:
                        raise TypeError(
                            "Ellipse received an unnamed argument of a type that is not permitted!")

        # if no args, check kwargs: if one is shear, then use that
        #                   look for dilation
        #                   look for x_shift, y_shift
        #                   if anything is left, pass to Shear constructor
        if use_dil is None:
            use_dil = kwargs.pop('dilation', 0.0)
        if use_shift is None:
            x_shift = kwargs.pop('x_shift', 0.0)
            y_shift = kwargs.pop('y_shift', 0.0)
            use_shift = _galsim.PositionD(x_shift, y_shift)
        if use_shear is None:
            use_shear = kwargs.pop('shear', None)
            if use_shear is None:
                if kwargs:
                    use_shear = galsim.Shear(**kwargs)
                else:
                    use_shear = galsim.Shear()
            else:
                if not isinstance(use_shear, galsim.Shear):
                    raise TypeError("Shear passed to Ellipse constructor was not a galsim.Shear!")
                # if shear was passed in some way, then we should not allow any other args
                if kwargs:
                    raise TypeError("Keyword arguments to Ellipse not permitted: %s"%kwargs.keys())

        self._ellipse = _galsim._Ellipse(s = use_shear._shear, mu = use_dil, p = use_shift)

    #### propagate through all the methods from C++
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
        return (self.__class__.__name__+"(g1="+str(shear.getG1())+", g2="+str(shear.getG2())+
                ", mu="+str(self.getMu())+", x="+str(x0.x)+", y="+str(x0.y)+")")

    def __str__(self):
        shear = self.getS()  # extract the e1 and e2 from the Shear instance
        x0 = self.getX0()    # extract the x0 and y0 from a Position instance
        return ("("+str(shear.getG1())+", "+str(shear.getG2())+", "+str(self.getMu())+", "
                +str(x0.x)+", "+str(x0.y)+")")

