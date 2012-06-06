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
        import numpy as np

        use_dil = None
        use_shear = None
        use_shift = None

        # check unnamed args: can have a Shear, float, and/or Position
        if len(args) > 0:
            for this_arg in args:
                if isinstance(this_arg, galsim.Shear):
                    if use_shear != None:
                        raise RuntimeError("Ellipse received two unnamed Shear arguments!")
                    use_shear = this_arg
                elif isinstance(this_arg, float) or isinstance(this_arg, double):
                    if use_dil != None:
                        raise RuntimeError("Ellipse received two unnamed float/double arguments!")
                    use_dil = this_arg
                elif isinstance(this_arg, _galsim.PositionD):
                    if use_shift != None:
                        raise RuntimeError("Ellipse received two unnamed Position arguments!")
                    use_shift = this_arg
                else:
                    raise RuntimeError(
                        "Ellipse received an unnamed argument of a type that is not permitted!")

        # if no args, check kwargs: if one is shear, then use that
        #                   look for dilation
        #                   look for x_shift, y_shift
        #                   if anything is left, pass to Shear constructor
        if use_dil == None:
            use_dil = kwargs.pop('dilation', None)
        if use_shift == None:
            x_shift = kwargs.pop('x_shift', None)
            y_shift = kwargs.pop('y_shift', None)
            if x_shift != None or y_shift != None:
                if x_shift == None:
                    x_shift = 0.0
                if y_shift == None:
                    y_shift = 0.0
                use_shift = _galsim.PositionD(x_shift, y_shift)
        if use_shear == None:
            use_shear = kwargs.pop('shear', None)
            if use_shear == None:
                if kwargs:
                    for key in kwargs:
                        print "Args: %s %s"%(key, kwargs[key])
                    use_shear = galsim.Shear(kwargs)
                else:
                    use_shear = galsim.Shear(g1 = 0.0, g2 = 0.0)

        # make sure something was specified!
        if use_shear == None:
            use_shear = galsim.Shear(g1 = 0.0, g2 = 0.0)
        if use_dil == None:
            use_dil = 0.0
        if use_shift == None:
            use_shift = _galsim.PositionD(0.0, 0.0)

        self.Ellipse = _galsim._Ellipse(s = use_shear, mu = use_dil, p = use_shift)

    #### propagate through all the methods from C++
    # define all the various operators on Ellipse objects
    def __neg__(self):
        return -self.Ellipse
    def __add__(self, other):
        return self.Ellipse + other.Ellipse
    def __sub__(self, other):
        return self.Ellipse - other.Ellipse
    def __iadd__(self, other):
        self.Ellipse += other.Ellipse
    def __isub__(self, other):
        self.Ellipse -= other.Ellipse
    def __eq__(self, other):
        return self.Ellipse == other.Ellipse
    def __ne__(self, other):
        return self.Ellipse != other.Ellipse
    def reset(self, s, mu, p):
        self.Ellipse.reset(s, mu, p)
    def fwd(self, p):
        return self.Ellipse.fwd(p)
    def inv(self, p):
        return self.Ellipse.inv(p)
    # methods for setting values
    def setS(self, s):
        self.Ellipse.setS(s)
    def setMu(self, mu):
        self.Ellipse.setMu(mu)
    def setX0(self, p):
        self.Ellipse.setX0(p)
    # methods for getting values
    def getS(self):
        return self.Ellipse.getS()
    def getMu(self):
        return self.Ellipse.getMu()
    def getX0(self):
        return self.Ellipse.getX0()
    def getMajor(self):
        return self.Ellipse.getMajor()
    def getMinor(self):
        return self.Ellipse.getMinor()
    def getBeta(self):
        return self.Ellipse.getBeta()
    def range(self):
        return self.Ellipse.range()
    def getMatrix(self):
        return self.Ellipse.getMatrix()
    # or access values directly
    s = property(getS)
    mu = property(getMu)
    x0 = property(getX0)
    major = property(getMajor)
    minor = property(getMinor)
    beta = property(getBeta)
    range = property(range)

def Ellipse_repr(self):
    shear = self.getS()  # extract the e1 and e2 from the Shear instance
    x0 = self.getX0()    # extract the x0 and y0 from a Position instance
    return (self.__class__.__name__+"(e1="+str(shear.getE1())+", e2="+str(shear.getE2())+
            ", mu="+str(self.getMu())+", x="+str(x0.x)+", y="+str(x0.y)+")")

def Ellipse_str(self):
    shear = self.getS()  # extract the e1 and e2 from the Shear instance
    x0 = self.getX0()    # extract the x0 and y0 from a Position instance
    return ("("+str(shear.getE1())+", "+str(shear.getE2())+", "+str(self.getMu())+", "+str(x0.x)+
            ", "+str(x0.y)+")")

Ellipse.__repr__ = Ellipse_repr
Ellipse.__str__ = Ellipse_str

