"""
Inject pickle and copy support into Boost.Python classes where appropriate.
"""

from . import _galsim

def memberof(cls):
    def closure(func):
        setattr(cls, func.func_name, func)
        return getattr(cls, func.func_name)
    return closure

def constructAngleFromRadians(value):
    """
    Named constructor for Angle needed for unpickling.
    """
    return _galsim.Angle(value * _galsim.radians)
constructAngleFromRadians.__safe_for_unpickling__ = True

@memberof(_galsim.Angle)
def __reduce__(self):
    return (constructAngleFromRadians, (self / _galsim.radians,))

@memberof(_galsim.PositionD)
def __reduce__(self):
    return (_galsim.PositionD, (self.x, self.y))

@memberof(_galsim.PositionI)
def __reduce__(self):
    return (_galsim.PositionI, (self.x, self.y))

@memberof(_galsim.BoundsD)
def __reduce__(self):
    return (_galsim.BoundsD, (self.xMin, self.xMax, self.yMin, self.yMax))

@memberof(_galsim.BoundsI)
def __reduce__(self):
    return (_galsim.BoundsI, (self.xMin, self.xMax, self.yMin, self.yMax))
