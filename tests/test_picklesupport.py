import numpy as np
import os
import sys
import pyfits
import cPickle
import copy
import numpy
import unittest

try:
    import galsim
except ImportError:
    path, filename = os.path.split(__file__)
    sys.path.append(os.path.abspath(os.path.join(path, "..")))
    import galsim
import galsim

class PickleTestCase(unittest.TestCase):

    def test_comparable(self):
        inputs = [galsim.Angle(3.20 * galsim.radians),
                  galsim.PositionI(2,3),
                  galsim.PositionD(2.2, 4.5),
                  galsim.BoundsI(2,3,7,8),
                  galsim.BoundsD(2.1, 4.3, 6.5, 9.1),
                  galsim._galsim._Shear(g1=0.2, g2=0.33),
                  ]
        for item in inputs:
            out1 = cPickle.loads(cPickle.dumps(item, protocol=2))
            self.assertEqual(item, out1)
            out2 = copy.copy(item)
            self.assertEqual(item, out2)
            out3 = copy.deepcopy(item)
            self.assertEqual(item, out3)

    def test_image(self):
        bounds = galsim.BoundsI(1,2,4,5)
        images = [galsim.ImageS(bounds), galsim.ImageI(bounds), galsim.ImageF(bounds), galsim.ImageD(bounds)]
        shape = images[0].array.shape
        images[0].array[:,:] = numpy.random.randint(0, 10, shape)
        images[1].array[:,:] = numpy.random.randint(0, 10, shape)
        images[2].array[:,:] = numpy.random.randn(*shape)
        images[3].array[:,:] = numpy.random.randn(*shape)
        for image in images:
            out1a = cPickle.loads(cPickle.dumps(image, protocol=2))
            self.assert_((out1a.array == image.array).all())
            out2a = copy.copy(image)
            self.assert_((out2a.array == image.array).all())
            self.assertNotEqual(out2a.getDataAddress(), image.getDataAddress())
            out3a = copy.deepcopy(image)
            self.assert_((out3a.array == image.array).all())
            self.assertNotEqual(out3a.getDataAddress(), image.getDataAddress())
            view = image.view()
            out1b = cPickle.loads(cPickle.dumps(view, protocol=2))
            self.assert_((out1b.array == view.array).all())
            out2b = copy.copy(view)
            self.assert_((out2b.array == view.array).all())
            self.assertEqual(out2b.getDataAddress(), image.getDataAddress())
            out3b = copy.deepcopy(view)
            self.assert_((out3b.array == view.array).all())
            self.assertNotEqual(out3b.getDataAddress(), image.getDataAddress())

    def test_photons(self):
        x = numpy.random.randn(3)
        y = numpy.random.randn(3)
        flux = numpy.arange(1, 4, dtype=float)
        photons = galsim.PhotonArray(x, y, flux)
        out1 = cPickle.loads(cPickle.dumps(photons, protocol=2))
        out2 = copy.copy(photons)
        out3 = copy.deepcopy(photons)
        for out in (out1, out2, out3):
            self.assertEqual(len(out), len(photons))
            for i in range(len(photons)):
                self.assertEqual(photons.getX(i), out.getX(i))
                self.assertEqual(photons.getY(i), out.getY(i))
                self.assertEqual(photons.getFlux(i), out.getFlux(i))

    def test_random(self):
        inputs = [galsim.GaussianDeviate(100, 0.5, 3.0),
                  galsim.BinomialDeviate(101, 8, 5),
                  galsim.PoissonDeviate(102, 12),
                  galsim.WeibullDeviate(104, 0.8, 0.7),
                  galsim.GammaDeviate(105, 0.9, 0.8),
                  galsim.Chi2Deviate(106, 50),
                  ]
        for item in inputs:
            out1 = cPickle.loads(cPickle.dumps(item, protocol=2))
            out2 = copy.copy(item)
            out3 = copy.deepcopy(item)
            seq0 = [item() for i in range(5)]
            seq1 = [out1() for i in range(5)]
            seq3 = [out3() for i in range(5)]
            self.assertEqual(seq0, seq1)
            self.assertEqual(seq0, seq3)
            # shallow copy means we have the same BaseDeviate, but different parameterized distributions
            self.assertEqual(item.writeState(), out2.writeState())
            self.assert_(item is not out2)

if __name__ == "__main__":
    unittest.main()
