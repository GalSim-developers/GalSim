import numpy as np
import os
import sys

try:
    import galsim
except ImportError:
    path, filename = os.path.split(__file__)
    sys.path.append(os.path.abspath(os.path.join(path, "..")))
    import galsim
import galsim.config

class TestPluggableBase(galsim.config.machinery.NodeBase):
    """
    A base class for testing pluggable nodes; pluggables must subclass this class.
    """
    p1 = galsim.config.machinery.Field(int, default=3)

class TestPluggableA(TestPluggableBase):
    p2a = galsim.config.machinery.Field(float, default=None)
    
class TestPluggableB(TestPluggableBase):
    p2b = galsim.config.machinery.Field(str, default="foo")

    def __init__(self, *args, **kwds):
        self.p1 = 4    # change default for field in base class
        TestPluggableBase.__init__(self, *args, **kwds)

class TestConfigRoot(galsim.config.machinery.NodeBase):
    """
    Root of the test config tree.
    """

    r1 = galsim.config.machinery.Field(int, default=0.)
    r2 = galsim.config.machinery.Field(float, default=None)

    @galsim.config.machinery.nested
    class r3(galsim.config.machinery.NodeBase):
        """
        A test subnode with a fixed type.
        """
        a1 = galsim.config.machinery.Field(str, default="")
    
    @galsim.config.machinery.nested
    class r4(galsim.config.machinery.ListNodeBase):
        """
        A test subnode that's a list of floats.
        """
        types = (float,)

    r5 = galsim.config.machinery.Field(
        TestPluggableBase, default=None,
        aliases={
            "A": TestPluggableA,
            "B": TestPluggableB,
            None: TestPluggableBase
            },
        doc="A single-element pluggable node field"
        )

    @galsim.config.machinery.nested
    class r6(galsim.config.machinery.ListNodeBase):
        """
        A test subnode that's a list of pluggables.
        """
        types = (TestPluggableA, TestPluggableB)   # don't allow base class here
        aliases={
            "A": TestPluggableA,
            "B": TestPluggableB,
            }

    def __init__(self, *args, **kwds):
        self.a1.default = "bar"  # change default for field in nested node
        galsim.config.machinery.NodeBase.__init__(self, *args, **kwds)


def funcname():
    import inspect
    return inspect.stack()[1][3]

if __name__ == "__main__":
        pass
