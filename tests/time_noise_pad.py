# Copyright 2012, 2013 The GalSim developers:
# https://github.com/GalSim-developers
#
# This file is part of GalSim: The modular galaxy image simulation toolkit.
#
# GalSim is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# GalSim is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with GalSim.  If not, see <http://www.gnu.org/licenses/>
#
import numpy as np
import os
import sys

"""Unit tests for the InterpolatedImage class.
"""

try:
    import galsim
except ImportError:
    path, filename = os.path.split(__file__)
    sys.path.append(os.path.abspath(os.path.join(path, "..")))
    import galsim

def funcname():
    import inspect
    return inspect.stack()[1][3]

def test_corr_padding_cf():
    """Test for correlated noise padding of InterpolatedImage."""
    import time
    t1 = time.time()

    imgfile = 'blankimg.fits'
    orig_nx = 147
    orig_ny = 124
    orig_seed = 151241

    # Make an ImageCorrFunc
    cf = galsim.ImageCorrFunc(galsim.fits.read(imgfile))

    # first, make a noise image
    orig_img = galsim.ImageF(orig_nx, orig_ny)
    orig_img.setScale(1.)
    orig_img.addNoise(galsim.GaussianDeviate(1234))

    # make it into an InterpolatedImage padded with cf
    int_im = galsim.InterpolatedImage(orig_img, noise_pad=cf)

    # do it again with a particular seed
    int_im = galsim.InterpolatedImage(orig_img, rng = galsim.GaussianDeviate(orig_seed),
                                      noise_pad = cf)

    # repeat
    int_im = galsim.InterpolatedImage(orig_img, rng = galsim.GaussianDeviate(orig_seed),
                                      noise_pad = cf)

    t2 = time.time()
    print 'time for %s = %.2f'%(funcname(),t2-t1)


def test_corr_padding_im():
    """Test for correlated noise padding of InterpolatedImage."""
    import time
    t1 = time.time()

    imgfile = 'blankimg.fits'
    orig_nx = 147
    orig_ny = 124
    orig_seed = 151241

    # Make an Image
    im = galsim.fits.read(imgfile)

    # first, make a noise image
    orig_img = galsim.ImageF(orig_nx, orig_ny)
    orig_img.setScale(1.)
    orig_img.addNoise(galsim.GaussianDeviate(1234))

    # make it into an InterpolatedImage padded with im
    int_im = galsim.InterpolatedImage(orig_img, noise_pad=im)

    # do it again with a particular seed
    int_im = galsim.InterpolatedImage(orig_img, rng = galsim.GaussianDeviate(orig_seed),
                                      noise_pad = im)

    # repeat
    int_im = galsim.InterpolatedImage(orig_img, rng = galsim.GaussianDeviate(orig_seed),
                                      noise_pad = im)

    t2 = time.time()
    print 'time for %s = %.2f'%(funcname(),t2-t1)


def test_corr_padding_imgfile():
    """Test for correlated noise padding of InterpolatedImage."""
    import time
    t1 = time.time()

    imgfile = 'blankimg.fits'
    orig_nx = 147
    orig_ny = 124
    orig_seed = 151241

    # Make an Image

    # first, make a noise image
    orig_img = galsim.ImageF(orig_nx, orig_ny)
    orig_img.setScale(1.)
    orig_img.addNoise(galsim.GaussianDeviate(1234))

    # make it into an InterpolatedImage padded with imgfile
    int_im = galsim.InterpolatedImage(orig_img, noise_pad=imgfile)

    # do it again with a particular seed
    int_im = galsim.InterpolatedImage(orig_img, rng = galsim.GaussianDeviate(orig_seed),
                                      noise_pad = imgfile)

    # repeat
    int_im = galsim.InterpolatedImage(orig_img, rng = galsim.GaussianDeviate(orig_seed),
                                      noise_pad = imgfile)

    t2 = time.time()
    print 'time for %s = %.2f'%(funcname(),t2-t1)

if __name__ == "__main__":
    test_corr_padding_cf()
    test_corr_padding_im()
    test_corr_padding_imgfile()
