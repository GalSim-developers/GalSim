"""\file shear.py Redefinition of the Shear and Ellipse classes at the Python layer.
"""

from . import _galsim

class Shear():
"""
@brief A class to represent shears in a variety of ways.

The python Shear class can be initialized in a variety of ways to represent shape distortions. All
arguments must be named.  Given semi-major and semi-minor axes a and b, we can define multiple shape
measurements:

reduced shear g = (a^2 - b^2)/(a^2 + b^2)
distortion e = (a - b)/(a + b)
conformal shear eta, with a/b = exp(eta)
axis ratio q = b/a

These can be thought of as a magnitude and a position angle theta, or as two components e.g., g1 and
g2, with

g1 = g cos(2*theta)
g2 = g sin(2*theta)

The following are all examples of valid calls to initialize a Shear object:
@code
s = galsim.Shear(g1=0.05, g2=0.05)
s = galsim.Shear(g1=0.05) # assumes g2=0
s = galsim.Shear(e1=0.05, e2=0.05)
s = galsim.Shear(e2=0.05) # assumes e1=0
s = galsim.Shear(eta=0.05, theta=galsim.Angle(45.0*galsim.degrees))
s = galsim.Shear(q=0.5, theta=galsim.Angle(0.0*galsim.radians))
@endcode

There can be no mixing and matching, e.g., specifying g1 and e2.  It is permissible to only specify
one of two components, with the other assumed to be zero.  If a magnitude such as e, g, eta, or q is
specified, then theta is also required to be specified.

@returns A Shear object.
"""
    def __init__(self, *args, **kwargs):
        # make sure there are no unnamed args

        # check the named args: if a magnitude, then that's fine, and require also a position angle

        # check the named args: if a component of e, g, or eta, then require that the other
        # component is zero if not set, and don't allow specification of mixed pairs like e1 and g2

        # make sure there is no other random keyword arg provided


def Shear_repr(self):
    return (self.__class__.__name__+"(e1="+str(self.getE1())+", e2="+str(self.getE2())+")")

def Shear_str(self):
    return ("("+str(self.getE1())+", "+str(self.getE2())+")")

_galsim.Shear.__repr__ = Shear_repr
_galsim.Shear.__str__ = Shear_str

