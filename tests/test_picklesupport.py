import numpy as np
import os
import sys
import pyfits
import cPickle
import copy

try:
    import galsim
except ImportError:
    path, filename = os.path.split(__file__)
    sys.path.append(os.path.abspath(os.path.join(path, "..")))
    import galsim
import galsim.utilities

comparable_inputs = [galsim.Angle(3.20 * galsim.radians),
                     galsim.PositionI(2,3),
                     galsim.PositionD(2.2, 4.5),
                     galsim.BoundsI(2,3,7,8),
                     galsim.BoundsD(2.1, 4.3, 6.5, 9.1),
                     ]

def test_pickle_and_copy():
    for item in comparable_inputs:
        s = cPickle.dumps(item, protocol=2)
        out1 = cPickle.loads(s)
        assert item == out1
        out2 = copy.copy(item)
        assert item == out2
        out3 = copy.deepcopy(item)
        assert item == out3
        

if __name__ == "__main__":
    test_pickle_and_copy()
