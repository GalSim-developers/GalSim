"""@file bounds.py @brief A few adjustments to the Bounds class at the Python layer.
"""

from . import _galsim

def Bounds_repr(self):
    return (self.__class__.__name__+"(xmin="+str(self.xMin)+", xmax="+str(self.xMax)+
            ", ymin="+str(self.yMin)+", ymax="+str(self.yMax)+")")

def Bounds_str(self):
    return "("+str(self.xMin)+", "+str(self.xMax)+", "+str(self.yMin)+", "+str(self.yMax)+")"

for Class in (_galsim.BoundsD, _galsim.BoundsI):
    Class.__repr__ = Bounds_repr
    Class.__str__ = Bounds_str

del Class    # cleanup public namespace
