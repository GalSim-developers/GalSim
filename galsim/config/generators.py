"""
Field and node classes that generate scalars or extract them from catalogs.
"""

from .._galsim import GaussianDeviate

def makeDict(node, row, keys):
    """
    Evaluate generatable fields in the given node for a catalog row and return
    a dictionary of numeric values.  Any fields set to None will be ignored.

    node ---- node to extract values from
    row ----- input catalog row
    keys ---- which fields to extract into a dictionary (all must be GeneratableFields)

    This should be used as the access point for GeneratorNode.apply(), as it contains the check
    for scalars that are used as constants (and those of course don't have an apply method),
    and it also casts the result to the field type.

    Must be called after finish().
    """
    d = {}
    cls = type(cls)
    for key in keys:
        field = getattr(cls, key)
        value = getattr(node, key)
        if value is None:
            pass
        elif isinstance(value, GeneratorNode):
            d[key] = field.type(value.apply(row))
        else:
            d[key] = value
    return d

class GeneratorBase(NodeBase):
    """
    A base class for nodes that can generate a scalar value.
    """

    def apply(self, row):
        """
        Return the appropriate generated or catalog-extracted value.

        This should only be called by the makeDict function.
        """
        raise NotImplementedError()

class GeneratableField(Field):

    def __init__(self, type=float, default=None, required=False, doc=None):
        Field.__init__(self, type=type, default=default, required=required, doc=doc)

    def __set__(self, instance, value):
        if isinstance(value, GeneratorBase):
            self._update_node_path(instance, value, self.name)
            instance._data[self.name] = value
        else:
            Field.__set__(self, instance, value)

class GaussianRandom(GeneratorBase):
    """
    Generator that indicates that a value should be drawn from a Gaussian distribution.
    """

    __slots__ = ("generator",)

    mean = Field(float, default=0., required=True, doc="mean of Gaussian distribution")
    sigma = Field(float, default=1., required=True, doc="sigma of Gaussian distribution")

    def finish(self, **kwds):
        """
        Prepare the config hierarchy to be used to process a catalog.

        The keywords arguments must include a "uniform" key that contains a UniformDeviate
        instance.

        See NodeBase.finish() for additional documentation.
        """
        GeneratorBase.finish(self, **kwds)
        self.generator = GaussianDeviate(kwds["uniform"], self.mean, self.sigma)

    def apply(self, row):
        """
        Return the generated or catalog-extracted value.

        This should only be called by GeneratableField.apply().

        Must be called after finish().
        """
        return self.generator()

class FromCatalog(GeneratorBase):
    """
    Generator that indicates that a value should be extracted from the input catalog.
    """

    name = Field(str, default=None, required=True, doc="name of the catalog column to use")

    def finish(self, **kwds):
        """
        Prepare the config hierarchy to be used to process a catalog.

        The keywords arguments must include a "columns" key that contains a dict that
        maps column names to column indices (starting from zero).
        
        See NodeBase.finish() for additional documentation.
        """
        GeneratorBase.finish(self, **kwds)
        self.index = kwds["columns"][self.name]

    def apply(self, row):
        """
        Return the generated or catalog-extracted value.

        This should only be called by GeneratableField.apply().

        Must be called after finish().
        """
        return row[self.index]

# TODO: add other random distributions
