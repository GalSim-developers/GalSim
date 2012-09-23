"""@file bounds.py @brief A few adjustments to the Bounds class at the Python layer.
"""

from . import _galsim

def Bounds_repr(self):
    return (self.__class__.__name__+"(xmin="+str(self.xMin)+", xmax="+str(self.xMax)+
            ", ymin="+str(self.yMin)+", ymax="+str(self.yMax)+")")

def Bounds_str(self):
    return "("+str(self.xMin)+", "+str(self.xMax)+", "+str(self.yMin)+", "+str(self.yMax)+")"

def Position_repr(self):
    return (self.__class__.__name__+"(x="+str(self.x)+", y="+str(self.y)+")")

def Position_str(self):
    return "("+str(self.x)+", "+str(self.y)+")"


# Some functions to enable pickling of Position and Bounds
def Bounds_getinitargs(self):
    return self.xmin, self.xmax, self.ymin, self.ymax

def Position_getinitargs(self):
    return self.x, self.y

for Class in (_galsim.BoundsD, _galsim.BoundsI):
    Class.__repr__ = Bounds_repr
    Class.__str__ = Bounds_str
    Class.__getinitargs__ = Bounds_getinitargs

for Class in (_galsim.PositionD, _galsim.PositionI):
    Class.__repr__ = Position_repr
    Class.__str__ = Position_str
    Class.__getinitargs__ = Position_getinitargs


del Class    # cleanup public namespace
