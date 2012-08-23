from . import machinery
from . import generators
from .. import base
from .. import _galsim

#-------------------------------------------------------------------------------------------------
# Config nodes for PSF and galaxy components

class GSObjectNode(machinery.NodeBase):
    flux = generators.GeneratableField(default=1.)
    target = None  # GSObject class returned by apply; updated by _update_gsobject_node_fields

    def apply(self, row):
        """
        Given a catalog row (sequence of numbers or strings convertible to numbers),
        construct a GSObject by evaluating the configuration values, drawing random
        numbers, and/or inserting catalog values.

        Must be called after finish().
        """
        kwds = generators.makeDict(self, row, self.fields.keys())
        return self.target(**kwds)


def _update_gsobject_node_fields(node, target):
    """Update the GSObjectNode class with the required fields for the target GSOjbect class.

    Takes a GSObjectNode class (the node input) and returns a new class with one
    generators.GeneratableField() attribute for each of the parameter descriptors of the GSObject
    supplied on input.
    """
    
    # First check that we're dealing with a GSObject in the input target class
    if not issubclass(target, base.GSObject):
        raise TypeError("Supplied target class must be a GSObject-derived class.")
        
    # Then introspect the class to find which parameter descriptors it uses, find out which ones are
    # GSObject parameter descriptors, and set these as appropriately-named attributes of the node:
    for keyname, entry in target.__dict__.iteritems():
        if entry.__module__ is "galsim.descriptors":
            node.__setattr__(keyname, generators.GeneratableField(default=None))

    node.target = target
    return node

#class EllipticalObjectNode(GSObjectNode):
#    g1 = generators.GeneratableField(default=None)
#    g2 = generators.GeneratableField(default=None)

#    def apply(self, row):
        # could remove this implementation if we add ellipticity kwds to GSObject constructors
        #kwds = generators.makeDict(self, row, self.fields.keys())
        #g1 = kwds.pop("g1", 0.)
        #g2 = kwds.pop("g2", 0.)
        #result = self.target(**kwds)
        #if g1 != 0. or g2 != 0.:
        #    result.applyShear(g1=g1, g2=g2)
        #return result

#class MoffatNode(EllipticalObjectNode):
    #beta = generators.GeneratableField(default=None)
    #truncationFWHM = generators.GeneratableField(default=2.)
    #fwhm = generators.GeneratableField(default=None)
    #half_light_radius = generators.GeneratableField(default=None)
    #scale_radius = generators.GeneratableField(default=None)
    #target = base.Moffat

#class DeVaucouleursNode(EllipticalObjectNode):
    #half_light_radius = generators.GeneratableField(default=None)
    #target = base.DeVaucouleurs

#class ExponentialNode(EllipticalObjectNode):
    #half_light_radius = generators.GeneratableField(default=None)
    #scale_radius = generators.GeneratableField(default=None)
    #target = base.Exponential

#class GaussianNode(EllipticalObjectNode):
    #half_light_radius = generators.GeneratableField(default=None)
    #sigma = generators.GeneratableField(default=None)
    #fwhm = generators.GeneratableField(default=None)
    #target = base.Gaussian

#class SersicNode(EllipticalObjectNode):
    #n = generators.GeneratableField(default=None)
    #half_light_radius = generators.GeneratableField(default=None)
    #target = base.Sersic

#class PixelNode(GSObjectNode):
    #xw = machinery.Field(default=None) # but see RootNode.finish()
    #yw = machinery.Field(default=None)

# TODO: more PSF component nodes


#-------------------------------------------------------------------------------------------------
# Config nodes for pixel noise generators

class NoiseNodeBase(machinery.NodeBase):
    """
    Base class for nodes that set the pixel noise to add to simulated images.
    """

    target = None  # random deviate class created by apply; should be set by derived classes

    def finish(self, **kwds):
        NodeBase.finish(self, **kwds)
        self.uniform = kwds["uniform"]

    def apply(self, row):
        kwds = generators.makeDict(self, row, self.fields.keys())
        return self.target(self.uniform, **kwds)

class CCDNoiseNode(NoiseNodeBase):
    gain = generators.GeneratableField(default=1.)
    readNoise = generators.GeneratableField(default=0.)
    target = _galsim.CCDNoise

class GaussianNoiseNode(NoiseNodeBase):
    mean = generators.GeneratableField(default=0.)
    sigma = generators.GeneratableField(default=1.)
    target = _galsim.GaussianDeviate

class PoissonNoiseNode(NoiseNodeBase):
    mean = generators.GeneratableField(default=1.)
    target = _galsim.PoissonDeviate

#-------------------------------------------------------------------------------------------------
# Config nodes to specify input catalog types and their properties

class InputCatNodeBase(machinery.NodeBase):

    def read(self):
        """
        Iterate over the catalog's records, yielding a sequence of fields for each row.
        """
        raise NotImplementedError()

class DummyInputCatNode(InputCatNodeBase):
    """
    A dummy input catalog with no columns and fixed size.  All parameters must
    be fixed or randomly generated.

    Setting the input_cat to None is equivalent to setting it to this class.
    """
    size = machinery.Field(type=int, default=None, doc="Number of postage stamps to generate")

    def read(self):
        for i in xrange(self.size):
            yield ()

class ASCIIInputCatNode(InputCatNodeBase):
    filename = machinery.Field(type=str, default=None)
    columns = machinery.Field(type=list, default=[])

    def finish(self, **kwds):
        for column in self.columns:
            if not isinstance(column, basestring):
                raise TypeError("ASCII column names must be strings; '%s' is not" % column)
        InputCatNodeBase.finish(self, **kwds)

    def read(self):
        with open(self.filename, 'r') as file:
            for line in file:
                yield line.split()

# TODO: FITS input catalogs

#-------------------------------------------------------------------------------------------------
# Config hierarchy definition for multi-postage-stamp image generation

class PostageStampRootNode(machinery.NodeBase):
    dx = machinery.Field(float, default=1.)

    def finish(self):
        for element in self.psf:
            if isinstance(element, PixelNode):
                if element.xw is None: element.xw = self.dx
                if element.yw is None: element.yw = self.dx
        machinery.NodeBase.finish(self)

    @machinery.nested
    class psf(machinery.ListNode):
        types = (MoffatNode, PixelNode)
        aliases = {"Moffat": MoffatNode, "Pixel": PixelNode}

        def apply(self, row):
            components = [element.apply(row) for element in self]
            return base.Convolve(components)

    @machinery.nested
    class galaxy(machinery.ListNode):
        types = (GaussianNode, ExponentialNode, DeVaucouleursNode, SersicNode)
        aliases = {"Gaussian": GaussianNode, "Exponential": ExponentialNode,
                   "DeVaucouleurs": DeVaucouleursNode, "Sersic": SersicNode}

        def apply(self, row):
            components = [element.apply(row) for element in self]
            return base.Add(components)

    @machinery.nested
    class shear(machinery.NodeBase):
        g1 = generators.GeneratableField(default=0.)
        g2 = generators.GeneratableField(default=0.)

    noise = machinery.Field(
        NoiseNodeBase, default=None,
        aliases={
            "CCDNoise": CCDNoiseNode,
            "GaussianNoise": GaussianNoiseNode,
            "PoissonNoise": PoissonNoiseNode,
            }
        )

    @machinery.nested
    class sky(machinery.NodeBase):
        value = generators.GeneratableField(default=0.)
        postsubtract = machinery.Field(type=bool, default=False)

    input_cat = machinery.Field(
        InputCatNodeBase, default=None,
        aliases={
            None: DummyInputCatNode,
            "ASCII": ASCIIInputCatNode,
            }
        )

