"""@file ellipse.py @brief A few adjustments to the Ellipse class at the Python layer.
"""

from . import _galsim

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
shift = galsim.Position<double>(0.0, 0.2)
ell = galsim.Ellipse(s)
ell2 = galsim.Ellipse(s, shift)
ell3 = galsim.Ellipse(s, y_shift = 0.2)
ell4 = galsim.Ellipse(dilation = 0.0, shear = s)
@endcode
"""
class Ellipse:
    def __init__(self, *args, **kwargs):
        import numpy as np

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
                elif isinstance(this_arg, galsim.Position):
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
            use_dil = kwargs.pop('dilation')
        if use_shift == None:
            x_shift = kwargs.pop('x_shift')
            y_shift = kwargs.pop('y_shift')
            if x_shift != None or y_shift != None:
                if x_shift == None:
                    x_shift = 0.0
                if y_shift == None:
                    y_shift = 0.0
                use_shift = galsim.Position<double>(x_shift, y_shift)
        if use_shear == None:
            use_shear = kwargs.pop('shear')
            if use_shear == None:
                use_shear = galsim.Shear(**kwargs)

        # make sure something was specified!
        if use_shear == None:
            use_shear = galsim.Shear(g1 = 0.0, g2 = 0.0)
        if use_dil == None:
            use_dil = 0.0
        if use_shift == None:
            use_shift = galsim.Position<double>(0.0, 0.0)

        return galsim._Ellipse(s = use_shear, mu = use_dil, p = use_shift)

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

_galsim._Ellipse.__repr__ = Ellipse_repr
_galsim._Ellipse.__str__ = Ellipse_str

