import galsim

def __str__(self):
    angle_rad = self.rad()
    return str(angle_rad)+" radians"

def __repr__(self):
    angle_rad = self.rad()
    return str(angle_rad)+" * galsim.radians"

galsim.Angle.__str__ = __str__
galsim.Angle.__repr__ = __repr__
