import numpy as np
import os
import sys
import unittest

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

    r1 = galsim.config.machinery.Field(int, default=0)
    r2 = galsim.config.machinery.Field(float, default=None)

    @galsim.config.machinery.nested
    class r3(galsim.config.machinery.NodeBase):
        """
        A test subnode with a fixed type.
        """
        a1 = galsim.config.machinery.Field(str, default="")
    
    r4 = galsim.config.machinery.ListField(float, doc="A test subnode that's a list of floats.")

    r5 = galsim.config.machinery.Field(
        TestPluggableBase, default=None,
        aliases={
            "A": TestPluggableA,
            "B": TestPluggableB,
            None: TestPluggableBase
            },
        doc="A single-element pluggable node field"
        )

    r6 = galsim.config.machinery.ListField(
        types=(TestPluggableA, TestPluggableB),   # don't allow base class here
        aliases={
            "A": TestPluggableA,
            "B": TestPluggableB,
            },
        doc="A test subnode that's a list of pluggables."
        )        

    def __init__(self, *args, **kwds):
        galsim.config.machinery.NodeBase.__init__(self, *args, **kwds)
        self.r3.a1 = "bar"  # change default for field in nested node

class ConfigMachineryTestCase(unittest.TestCase):

    def setUp(self):
        self.root = TestConfigRoot()

    def testDefaults(self):
        self.assertEqual(self.root.r1, 0)
        self.assertEqual(self.root.r2, None)
        self.assertEqual(self.root.r3.a1, "bar")
        self.assertEqual(len(self.root.r4), 0)
        self.assertEqual(list(self.root.r4), [])
        self.assertEqual(type(self.root.r5), TestPluggableBase)
        self.assertEqual(len(self.root.r6), 0)
        self.assertEqual(list(self.root.r6), [])

    def testScalars(self):
        self.root.r1 = 5
        self.assertEqual(self.root.r1, 5)
        self.assertRaises(TypeError, setattr, self.root, "r1", 4.3)
        self.root.r2 = 3.14
        self.assertEqual(self.root.r2, 3.14)
        self.assertRaises(TypeError, setattr, self.root, "r2", 2)
        self.assertRaises(TypeError, setattr, self.root, "r2", "fifty")
        self.root.r3.a1 = "foo"
        self.assertEqual(self.root.r3.a1, "foo")

    def testList(self):
        self.assertEqual(len(self.root.r4), 0)
        self.root.r4[0] = 0.0
        self.root.r4[1] = 1.0
        self.assertEqual(len(self.root.r4), 2)
        self.assertEqual(self.root.r4[0], 0.0)
        self.assertEqual(self.root.r4[1], 1.0)
        self.root.r4.append(2.0)
        self.assertEqual(self.root.r4[-1], 2.0)
        self.root.r4.extend([3.0, 4.0])
        self.assertEqual(list(self.root.r4), [0.0, 1.0, 2.0, 3.0, 4.0])
        self.root.r4 = [2.1, 2.2]
        self.assertEqual(list(self.root.r4), [2.1, 2.2])
        

if __name__ == "__main__":
    unittest.main()
