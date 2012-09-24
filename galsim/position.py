"""@file position.py
A few adjustments to the Position classes at the Python layer.
"""

from . import _galsim

def Position_repr(self):
    return self.__class__.__name__+"(x="+str(self.x)+", y="+str(self.y)+")"

def Position_str(self):
    return "("+str(self.x)+", "+str(self.y)+")"

for Class in (_galsim.PositionD, _galsim.PositionI):
    Class.__repr__ = Position_repr
    Class.__str__ = Position_str
    Class.__doc__ = """A class for representing 2D positions on the plane.

    PositionD describes positions with floating point values in `x` and `y`.
    PositionI described positions with integer values in `x` and `y`.

    Initialization
    --------------

    For the float-valued position class, an example init:

        >>> pos = galsim.PositionD(x=0.5, y=-0.5)

    And for the integer-valued position class, an example init:

        >>> pos = galsim.PositionI(x=45, y=13)

    Attributes
    ----------
    For an instance `pos` as instantiated above, `pos.x` and `pos.y` store the x and y values of the
    position.
    """

del Class    # cleanup public namespace
