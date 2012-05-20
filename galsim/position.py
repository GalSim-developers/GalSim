"""@file position.py @brief A few adjustments to the Position classes at the Python layer.
"""

from . import _galsim

def Position_repr(self):
    return self.__class__.__name__+"(x="+str(self.x)+", y="+str(self.y)+")"

def Position_str(self):
    return "("+str(self.x)+", "+str(self.y)+")"

for Class in (_galsim.PositionD, _galsim.PositionI):
    Class.__repr__ = Position_repr
    Class.__str__ = Position_str

del Class    # cleanup public namespace
