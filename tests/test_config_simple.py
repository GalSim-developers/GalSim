
try:
    import galsim
except ImportError:
    path, filename = os.path.split(__file__)
    sys.path.append(os.path.abspath(os.path.join(path, "..")))
    import galsim
import galsim.config

class TestRootNode(galsim.config.machinery.NodeBase):
    """
    A simple test config hierarchy corresponding to the override file
    in tests/config_files/machinery2.py.
    """
    psf = galsim.config.machinery.Field(GSObjectNode, default=None)
    gal = galsim.config.machinery.Field(GSObjectNode, default=None)

def test_load():
    root = TestRootNode()
    root.load("tests/config_files/machinery2.py")
    # ... test that values were set properly when loading the config file
