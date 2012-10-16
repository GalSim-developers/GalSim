"""@file position.py
A few adjustments to the Position classes at the Python layer.
"""

from . import _galsim

def Position_repr(self):
    return self.__class__.__name__+"(x="+str(self.x)+", y="+str(self.y)+")"

def Position_str(self):
    return "("+str(self.x)+", "+str(self.y)+")"

def Position_getinitargs(self):
    return self.x, self.y

for Class in (_galsim.PositionD, _galsim.PositionI):
    Class.__repr__ = Position_repr
    Class.__str__ = Position_str
    Class.__getinitargs__ = Position_getinitargs
    Class.__doc__ = """A class for representing 2D positions on the plane.

    PositionD describes positions with floating point values in `x` and `y`.
    PositionI described positions with integer values in `x` and `y`.

    Initialization
    --------------

    For the float-valued position class, example inits include:

        >>> pos = galsim.PositionD(x=0.5, y=-0.5)
        >>> pos = galsim.PositionD(0.5, -0.5)

    And for the integer-valued position class, example inits include:

        >>> pos = galsim.PositionI(x=45, y=13)
        >>> pos = galsim.PositionI(45, 13)

    Attributes
    ----------
    For an instance `pos` as instantiated above, `pos.x` and `pos.y` store the x and y values of the
    position.

    Arithmetic
    ----------
    Most arithmetic that makes sense for a position is allowed:

        >>> pos1 + pos2
        >>> pos1 - pos2
        >>> pos * x
        >>> pos / x
        >>> -pos
        >>> pos1 += pos2
        >>> pos1 -= pos2
        >>> pos *= x
        >>> pos -= x

    Note though that the types generally need to match.  For example, you cannot multiply
    a PositionI by a float or add a PositionI to a PositionD.
    """

del Class    # cleanup public namespace
