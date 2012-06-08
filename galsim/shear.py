"""\file shear.py Redefinition of the Shear and Ellipse classes at the Python layer.
"""

from . import _galsim

"""
@brief A class to represent shears in a variety of ways.

The python Shear class can be initialized in a variety of ways to represent shape distortions. All
arguments must be named.  Given semi-major and semi-minor axes a and b, we can define multiple shape
measurements:

reduced shear g = (a^2 - b^2)/(a^2 + b^2)
distortion e = (a - b)/(a + b)
conformal shear eta, with a/b = exp(eta)
axis ratio q = b/a

These can be thought of as a magnitude and a position angle beta, or as two components e.g., g1 and
g2, with

g1 = g cos(2*beta)
g2 = g sin(2*beta)

The following are all examples of valid calls to initialize a Shear object:
@code
s = galsim.Shear(g1=0.05, g2=0.05)
s = galsim.Shear(g1=0.05) # assumes g2=0
s = galsim.Shear(e1=0.05, e2=0.05)
s = galsim.Shear(e2=0.05) # assumes e1=0
s = galsim.Shear(eta=0.05, beta=galsim.Angle(45.0*galsim.degrees))
s = galsim.Shear(q=0.5, beta=galsim.Angle(0.0*galsim.radians))
@endcode

There can be no mixing and matching, e.g., specifying g1 and e2.  It is permissible to only specify
one of two components, with the other assumed to be zero.  If a magnitude such as e, g, eta, or q is
specified, then beta is also required to be specified.
"""
class Shear:
    def __init__(self, **kwargs):
        # note, no *args because we're not allowing Shear to be initialized with unnamed args!
        import numpy as np

        # check the named args: if a component of e, g, or eta, then require that the other
        # component is zero if not set, and don't allow specification of mixed pairs like e1 and g2
        # check the named args: require also a position angle if we didn't get g1/g2, e1/e2, or
        # eta1/eta2
        if not kwargs:
            raise RuntimeError("No keywords given to initialize Shear!")
        if len(kwargs) > 2:
            raise RuntimeError(
                "Shear constructor received too many keyword arguments (max 2): %s"%kwargs.keys())
        g1 = kwargs.pop('g1', None)
        g2 = kwargs.pop('g2', None)
        e1 = kwargs.pop('e1', None)
        e2 = kwargs.pop('e2', None)
        eta1 = kwargs.pop('eta1', None)
        eta2 = kwargs.pop('eta2', None)
        beta = kwargs.pop('beta', None)
        g = kwargs.pop('g', None)
        e = kwargs.pop('e', None)
        eta = kwargs.pop('eta', None)
        q = kwargs.pop('q', None)
        # make sure there is no other random keyword arg provided
        if kwargs:
            raise RuntimeError(
                "Shear constructor got unexpected argument(s): %s"%kwargs.keys())

        # Now go through the possibilities for combinations of args
        use_shear = None
        if g1 != None or g2 != None:
            if g1 == None:
                g1 = 0.0
            if g2 == None:
                g2 = 0.0
            g = np.sqrt(g1**2 + g2**2)
            if g < 1:
                use_shear = _galsim._Shear(g1, g2)
            else:
                raise ValueError("Requested shear exceeds 1: %f"%g)
        elif e1 != None or e2 != None:
            if use_shear != None:
                raise RuntimeError("Tried to initialize Shear in too many ways!")
            if e1 == None:
                e1 = 0.0
            if e2 == None:
                e2 = 0.0
            e = np.sqrt(e1**2 + e2**2)
            if e < 1:
                use_shear = _galsim._Shear()
                use_shear.setE1E2(e1, e2)
            else:
                raise ValueError("Requested distortion exceeds 1: %s"%e)
        elif eta1 != None or eta2 != None:
            if use_shear != None:
                raise RuntimeError("Tried to initialize Shear in too many ways!")
            if eta1 == None:
                eta1 = 0.0
            if eta2 == None:
                eta2 = 0.0
            use_shear = _galsim._Shear()
            use_shear.setEta1Eta2(eta1, e2)
        # from here on, we need a magnitude and position angle, so check the PA
        elif beta == None:
            raise RuntimeError(
                "Shear constructor did not get 2 components, OR a magnitude and position angle!")
        elif not isinstance(beta, _galsim.Angle) :
            raise RuntimeError("The position angle that was supplied is not an Angle instance!")
        elif g != None:
            if use_shear != None:
                raise RuntimeError("Tried to initialize Shear in too many ways!")
            if abs(g) > 1:
                raise ValueError("Requested shear exceeds 1: %f"%g)
            g1 = np.cos(2.0*np.pi)*g
            g2 = np.sin(2.0*np.pi)*g
            use_shear = _galsim._Shear(g1, g2)
        elif e != None:
            if use_shear != None:
                raise RuntimeError("Tried to initialize Shear in too many ways!")
            if abs(e) > 1:
                raise ValueError("Requested distortion exceeds 1: %f"%e)
            use_shear = _galsim._Shear()
            use_shear.setEBeta(e, beta)
        elif eta != None:
            if use_shear != None:
                raise RuntimeError("Tried to initialize Shear in too many ways!")
            use_shear = _galsim._Shear()
            use_shear.setEtaBeta(eta, beta)
        elif q != None:
            if use_shear != None:
                raise RuntimeError("Tried to initialize Shear in too many ways!")
            if q <= 0 or q > 1:
                raise ValueError("Cannot use requested axis ratio of %f!"%q)
            use_shear = _galsim._Shear()
            eta = -np.log(q)
            use_shear.setEtaBeta(eta, beta)

        self.Shear = use_shear

    #### propagate through all the methods from C++
    # define all the methods for setting shear values
    def setE1E2(self, e1=0.0, e2=0.0): return self.Shear.setE1E2(e1, e2)
    def setEBeta(self, e=0.0, beta=None): return self.Shear.setEBeta(e, beta)
    def setEta1Eta2(self, eta1=0.0, eta2=0.0): return self.Shear.setEta1Eta2(eta1, eta2)
    def setEtaBeta(self, eta=0.0, beta=None): return self.Shear.setEtaBeta(eta, beta)
    def setG1G2(self, g1=0.0, g2=0.0): return self.Shear.setG1G2(g1, g2)
    # define all the methods to get shear values
    def getE1(self): return self.Shear.getE1()
    def getE2(self): return self.Shear.getE2()
    def getE(self): return self.Shear.getE()
    def getESq(self): return self.Shear.getESq()
    def getBeta(self): return self.Shear.getBeta()
    def getEta(self): return self.Shear.getEta()
    def getG(self): return self.Shear.getG()
    # make it possible to access g, e, etc. of some Shear object called name using name.g, name.e
    e1 = property(getE1)
    e2 = property(getE2)
    e = property(getE)
    esq = property(getESq)
    beta = property(getBeta)
    eta = property(getEta)
    g = property(getG)
    # define all the various operators on Shear objects
    def __neg__(self): return -self.Shear
    def __add__(self, other): return self.Shear + other.Shear
    def __sub__(self, other): return self.Shear - other.Shear
    def __iadd__(self, other): self.Shear += other.Shear
    def __isub__(self, other): self.Shear -= other.Shear
    def rotationWith(self, other): return self.Shear.rotationWith(other)
    def __eq__(self, other): return self.Shear == other.Shear
    def __ne__(self, other): return self.Shear != other.Shear
    def __mul__(self, val): return self.Shear * val
    def __rmul__(self, val): return self.Shear * val
    def __div__(self, val): return self.Shear / val
    def __imul__(self, val): self.Shear *= val
    def __idiv__(self, val): self.Shear /= val
    def fwd(self, p): return self.Shear.fwd(p)
    def inv(self, p): return self.Shear.inv(p)
    def getMatrix(self, a, b, c): self.Shear.getMatrix(a, b, c)

def Shear_repr(self):
    return (self.__class__.__name__+"(e1="+str(self.getE1())+", e2="+str(self.getE2())+")")

def Shear_str(self):
    return ("("+str(self.getE1())+", "+str(self.getE2())+")")

Shear.__repr__ = Shear_repr
Shear.__str__ = Shear_str

