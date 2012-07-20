import numpy as np
import os
import sys
import pyfits
import cPickle
import copy
import numpy

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
        out1 = cPickle.loads(cPickle.dumps(item, protocol=2))
        assert item == out1
        out2 = copy.copy(item)
        assert item == out2
        out3 = copy.deepcopy(item)
        assert item == out3

def test_image_pickle_and_copy():
    bounds = galsim.BoundsI(1,2,4,5)
    images = [galsim.ImageS(bounds), galsim.ImageI(bounds), galsim.ImageF(bounds), galsim.ImageD(bounds)]
    shape = images[0].array.shape
    images[0].array[:,:] = numpy.random.randint(0, 10, shape)
    images[1].array[:,:] = numpy.random.randint(0, 10, shape)
    images[2].array[:,:] = numpy.random.randn(*shape)
    images[3].array[:,:] = numpy.random.randn(*shape)
    for image in images:
        out1a = cPickle.loads(cPickle.dumps(image, protocol=2))
        assert (out1a.array == image.array).all()
        out2a = copy.copy(image)
        assert (out2a.array == image.array).all()
        assert out2a.getDataAddress() != image.getDataAddress()
        out3a = copy.deepcopy(image)
        assert (out3a.array == image.array).all()
        assert out3a.getDataAddress() != image.getDataAddress()
        view = image.view()
        out1b = cPickle.loads(cPickle.dumps(view, protocol=2))
        assert (out1b.array == view.array).all()
        out2b = copy.copy(view)
        assert (out2b.array == view.array).all()
        assert out2b.getDataAddress() == image.getDataAddress()
        out3b = copy.deepcopy(view)
        assert (out3b.array == view.array).all()
        assert out3b.getDataAddress() != image.getDataAddress()

if __name__ == "__main__":
    test_pickle_and_copy()
    test_image_pickle_and_copy()
