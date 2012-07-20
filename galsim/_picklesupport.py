"""
Inject pickle and copy support into Boost.Python classes where appropriate.
"""

from . import _galsim
import numpy

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

def unpickleShear(e1, e2):
    """
    Named constructor for _Shear needed for unpickling.  We can't use the actual constructor
    because the internal representation uses e1,e2 while the constructor takes g1,g2, and
    that means things wouldn't round-trip exactly due to roundoff error.
    """
    s = _galsim._Shear()
    s.setE1E2(e1, e2)
    return s
unpickleShear.__safe_for_unpickling__ = True

@memberof(_galsim._Shear)
def __reduce__(self):
    return (unpickleShear, (self.getE1(), self.getE2()))

@memberof(*(_galsim.ImageView.values() + _galsim.ConstImageView.values()))
def __reduce__(self):
    return (type(self), (self.array, self.xMin, self.xMax))

@memberof(*_galsim.Image.values())
def __reduce__(self):
    return (type(self), (self.view(),))

@memberof(_galsim.PhotonArray)
def __reduce__(self):
    size = len(self)
    x = numpy.zeros(size, dtype=float)
    y = numpy.zeros(size, dtype=float)
    flux = numpy.zeros(size, dtype=float)
    for i in xrange(size):
        x[i] = self.getX(i)
        y[i] = self.getY(i)
        flux[i] = self.getFlux(i)
    return (_galsim.PhotonArray, (x, y, flux))

def unpickleBaseDeviate(state):
    r = _galsim.BaseDeviate(lseed=0)
    r.readState(state)
    return r
unpickleBaseDeviate.__safe_for_unpickling__ = True

@memberof(_galsim.BaseDeviate)
def __reduce__(self):
    return (unpickleBaseDeviate, (self.writeState(),))

@memberof(_galsim.GaussianDeviate)
def __reduce__(self):
    return (_galsim.GaussianDeviate, (_galsim.BaseDeviate(self), self.getMean(), self.getSigma()))

@memberof(_galsim.BinomialDeviate)
def __reduce__(self):
    return (_galsim.BinomialDeviate, (_galsim.BaseDeviate(self), self.getN(), self.getP()))

@memberof(_galsim.PoissonDeviate)
def __reduce__(self):
    return (_galsim.PoissonDeviate, (_galsim.BaseDeviate(self), self.getMean()))

@memberof(_galsim.CCDNoise)
def __reduce__(self):
    return (_galsim.CCDNoise, (_galsim.BaseDeviate(self), self.getGain(), self.getReadNoise()))

@memberof(_galsim.WeibullDeviate)
def __reduce__(self):
    return (_galsim.WeibullDeviate, (_galsim.BaseDeviate(self), self.getA(), self.getB()))

@memberof(_galsim.GammaDeviate)
def __reduce__(self):
    return (_galsim.GammaDeviate, (_galsim.BaseDeviate(self), self.getAlpha(), self.getBeta()))

@memberof(_galsim.Chi2Deviate)
def __reduce__(self):
    return (_galsim.Chi2Deviate, (_galsim.BaseDeviate(self), self.getN()))
