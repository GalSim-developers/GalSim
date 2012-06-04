from . import machinery
from . import generators
from .. import base

class GSObjectNode(machinery.NodeBase):
    flux = generators.GeneratableField(default=1., required=False)

    target = None  # GSObject class or factory function used by apply; should be set by derived classes.

    def apply(self, row):
        """
        Given a catalog row (sequence of numbers or strings convertible to numbers),
        construct a GSObject by evaluating the configuration values, drawing random
        numbers, and/or inserting catalog values.

        Must be called after finish().
        """
        kwds = generators.makeDict(self, row, self.fields.keys())
        return self.target(**kwds)

class EllipticalObjectNode(GSObjectNode):
    g1 = generators.GeneratableField(default=None, required=False)
    g2 = generators.GeneratableField(default=None, required=False)

    def apply(self, row):
        # could remove this implementation if we add ellipticity kwds to GSObject constructors
        kwds = generators.makeDict(self, row, self.fields.keys())
        g1 = kwds.pop("g1", 0.)
        g2 = kwds.pop("g2", 0.)
        result = self.target(**kwds)
        if g1 != 0. or g2 != 0.:
            result.applyShear(g1=g1, g2=g2)
        return result

class MoffatNode(EllipticalObjectNode):
    beta = generators.GeneratableField(default=None, required=False)
    truncationFWHM = generators.GeneratableField(default=2., required=False)
    fwhm = generators.GeneratableField(default=None, required=False)
    half_light_radius = generators.GeneratableField(default=None, required=False)    
    scale_radius = generators.GeneratableField(default=None, required=False)    

    target = base.Moffat

class DeVaucouleursNode(EllipticalObjectNode):
    half_light_radius = generators.GeneratableField(default=None, required=False)    

    target = base.DeVaucouleurs

class ExponentialNode(EllipticalObjectNode):
    half_light_radius = generators.GeneratableField(default=None, required=False)    
    scale_radius = generators.GeneratableField(default=None, required=False)    

    target = base.Exponential

class GaussianNode(EllipticalObjectNode):
    half_light_radius = generators.GeneratableField(default=None, required=False)
    sigma = generators.GeneratableField(default=None, required=False)
    fwhm = generators.GeneratableField(default=None, required=False)

    target = base.Gaussian

class SersicNode(EllipticalObjectNode):
    n = generators.GeneratableField(default=None, required=False)
    half_light_radius = generators.GeneratableField(default=None, required=False)

    target = base.Sersic

class PixelNode(GSObjectNode):
    xw = Field(default=None, required=False) # but see RootNode.finish()
    yw = Field(default=None, required=False)

# TODO: more PSF component nodes

class PSFNode(machinery.ListNodeBase):
    choices = (MoffatNode, PixelNode)
    context = {"Moffat": MoffatNode, "Pixel": PixelNode}

    def apply(self, row):
        components = [element.apply(row) for element in self]
        return base.Convolve(components)

class GalaxyNode(machinery.ListNodeBase):
    choices = (GaussianNode, ExponentialNode, DeVaucouleursNode, SersicNode)
    context = {"Gaussian": GaussianNode, "Exponential": ExponentialNode,
               "DeVaucouleurs": DeVaucouleursNode, "Sersic": SersicNode}

    def apply(self, row):
        components = [element.apply(row) for element in self]
        return base.Add(components)

class ShearNode(machinery.NodeBase):
    g1 = generators.GeneratableField(default=0., required=True)
    g2 = generators.GeneratableField(default=0., required=True)
    
class RootNode(machinery.NodeBase):
    dx = machinery.Field(float, default=1., required=True)
    PSF = machinery.Field(PSFNode, default=True)
    galaxy = machinery.Field(GalaxyNode, default=True)
    shear = machinery.Field(ShearNode, default=True)
    
    def finish(self):
        machinery.NodeBase.finish(self)
        for element in self.PSF:
            if isinstance(element, PixelNode):
                if element.xw is None: element.xw = self.dx
                if element.yw is None: element.yw = self.dx
