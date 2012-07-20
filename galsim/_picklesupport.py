"""
Inject pickle and copy support into Boost.Python classes where appropriate.
"""

from . import _galsim

def memberof(*classes):
    def closure(func):
        for cls in classes:
            setattr(cls, func.func_name, func)
            getattr(cls, func.func_name)
    return closure

def unpickleAngle(value):
    """
    Named constructor for Angle needed for unpickling.
    """
    return _galsim.Angle(value * _galsim.radians)
unpickleAngle.__safe_for_unpickling__ = True

@memberof(_galsim.Angle)
def __reduce__(self):
    return (unpickleAngle, (self / _galsim.radians,))

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

@memberof(*(_galsim.ImageView.values() + _galsim.ConstImageView.values()))
def __reduce__(self):
    return (type(self), (self.array, self.xMin, self.xMax))

@memberof(*_galsim.Image.values())
def __reduce__(self):
    return (type(self), (self.view(),))
