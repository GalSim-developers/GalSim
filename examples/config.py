import galsim.config.machinery as m
import numpy

class Node2(m.ConfigBase):
    """A node defined at the top level but intended to be used (and reused anywhere in the hierarchy)."""
    list0 = m.Field(list, doc="a list field that can have any kind of element (or even be None)")

    def _get_size(self): return len(self.list0)
    def _set_size(self, n): self.list0 = [None] * n
    size = property(_get_size, _set_size)

class Root(m.ConfigBase):
    str1 = m.Field(str, doc="a string field")
    int2 = m.Field(int, default=0, required=True, doc="an integer field")

    class Node1(m.ConfigBase):
        """A nested node in the config hierarchy."""
        float3 = m.Field(float, default=numpy.pi, doc="a float field")
    node1 = m.Field(Node1, default=True)

    node2 = m.Field(Node2)

    class Node3(m.ConfigBase):
        node2a = m.Field(Node2, default=True)
    node3 = m.Field(Node3, default=True)
