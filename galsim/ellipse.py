"""@file ellipse.py @brief A few adjustments to the Ellipse class at the Python layer.
"""

from . import _galsim

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

_galsim.Ellipse.__repr__ = Ellipse_repr
_galsim.Ellipse.__str__ = Ellipse_str

