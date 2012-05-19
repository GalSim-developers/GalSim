"""@brief A few adjustments to the Shear class at the Python layer.
"""

# NOTE SHEAR HERE IS CURRENTLY THE E1/E2 DEFINITION, SHOULD BE RECITIFIED WITH ISSUE #134

from .import _galsim

def Shear_repr(self):
    return (self.__class__.__name__+"(e1="+str(self.getE1())+", e2="+str(self.getE2())+")")

def Shear_str(self):
    return ("(e1="+str(self.getE1())+", e2="+str(self.getE2())+")")

_galsim.Shear.__repr__ = Shear_repr
_galsim.Shear.__str__ = Shear_str

