# Copyright (c) 2012-2015 by the GalSim developers team on GitHub
# https://github.com/GalSim-developers
#
# This file is part of GalSim: The modular galaxy image simulation toolkit.
# https://github.com/GalSim-developers/GalSim
#
# GalSim is free software: redistribution and use in source and binary forms,
# with or without modification, are permitted provided that the following
# conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions, and the disclaimer given in the accompanying LICENSE
#    file.
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions, and the disclaimer given in the documentation
#    and/or other materials provided with the distribution.
#
import numpy as np
import os
import sys

from galsim_test_helpers import *

try:
    import galsim
except ImportError:
    path, filename = os.path.split(__file__)
    sys.path.append(os.path.abspath(os.path.join(path, "..")))
    import galsim
import galsim.utilities


testshape = (512, 512)  # shape of image arrays for all tests
decimal = 6     # Last decimal place used for checking equality of float arrays, see
                # np.testing.assert_array_almost_equal(), low since many are ImageF

def test_roll2d_circularity():
    """Test both integer and float arrays are unchanged by full circular roll.
    """
    import time
    t1 = time.time()
    # Make heterogenous 2D array, integers first, test that a full roll gives the same as the inputs
    int_image = np.random.random_integers(low=0, high=1, size=testshape)
    np.testing.assert_array_equal(int_image,
                                  galsim.utilities.roll2d(int_image, int_image.shape),
                                  err_msg='galsim.utilities.roll2D failed int array circularity')
    # Make heterogenous 2D array, this time floats
    flt_image = np.random.random(size=testshape)
    np.testing.assert_array_equal(flt_image,
                                  galsim.utilities.roll2d(flt_image, flt_image.shape),
                                  err_msg='galsim.utilities.roll2D failed flt array circularity')
    t2 = time.time()
    print 'time for %s = %.2f'%(funcname(),t2-t1)

def test_roll2d_fwdbck():
    """Test both integer and float arrays are unchanged by unit forward and backward roll.
    """
    import time
    t1 = time.time()
    # Make heterogenous 2D array, integers first, test that a +1, -1 roll gives the same as initial
    int_image = np.random.random_integers(low=0, high=1, size=testshape)
    np.testing.assert_array_equal(int_image,
                                  galsim.utilities.roll2d(galsim.utilities.roll2d(int_image,
                                                                                  (+1, +1)),
                                                          (-1, -1)),
                                  err_msg='galsim.utilities.roll2D failed int array fwd/back test')
    # Make heterogenous 2D array, this time floats
    flt_image = np.random.random(size=testshape)
    np.testing.assert_array_equal(flt_image,
                                  galsim.utilities.roll2d(galsim.utilities.roll2d(flt_image,
                                                                                  (+1, +1)),
                                                          (-1, -1)),
                                  err_msg='galsim.utilities.roll2D failed flt array fwd/back test')
    t2 = time.time()
    print 'time for %s = %.2f'%(funcname(),t2-t1)

def test_roll2d_join():
    """Test both integer and float arrays are equivalent if rolling +1/-1 or -/+(shape[i/j] - 1).
    """
    import time
    t1 = time.time()
    # Make heterogenous 2D array, integers first
    int_image = np.random.random_integers(low=0, high=1, size=testshape)
    np.testing.assert_array_equal(galsim.utilities.roll2d(int_image, (+1, -1)),
                                  galsim.utilities.roll2d(int_image, (-(int_image.shape[0] - 1),
                                                                   +(int_image.shape[1] - 1))),
                                  err_msg='galsim.utilities.roll2D failed int array +/- join test')
    np.testing.assert_array_equal(galsim.utilities.roll2d(int_image, (-1, +1)),
                                  galsim.utilities.roll2d(int_image, (+(int_image.shape[0] - 1),
                                                                   -(int_image.shape[1] - 1))),
                                  err_msg='galsim.utilities.roll2D failed int array -/+ join test')
    # Make heterogenous 2D array, this time floats
    flt_image = np.random.random(size=testshape)
    np.testing.assert_array_equal(galsim.utilities.roll2d(flt_image, (+1, -1)),
                                  galsim.utilities.roll2d(flt_image, (-(flt_image.shape[0] - 1),
                                                                   +(flt_image.shape[1] - 1))),
                                  err_msg='galsim.utilities.roll2D failed flt array +/- join test')
    np.testing.assert_array_equal(galsim.utilities.roll2d(flt_image, (-1, +1)),
                                  galsim.utilities.roll2d(flt_image, (+(flt_image.shape[0] - 1),
                                                                   -(flt_image.shape[1] - 1))),
                                  err_msg='galsim.utilities.roll2D failed flt array -/+ join test')
    t2 = time.time()
    print 'time for %s = %.2f'%(funcname(),t2-t1)

def test_kxky():
    """Test that the basic properties of kx and ky are right.
    """
    import time
    t1 = time.time()
    kx, ky = galsim.utilities.kxky((4, 4))
    kxref = np.array([0., 0.25, -0.5, -0.25]) * 2. * np.pi
    kyref = np.array([0., 0.25, -0.5, -0.25]) * 2. * np.pi
    for i in xrange(4):
        np.testing.assert_array_almost_equal(kx[i, :], kxref, decimal=decimal,
                                             err_msg='failed kx equivalence on row i = '+str(i))
    for j in xrange(4):
        np.testing.assert_array_almost_equal(ky[:, j], kyref, decimal=decimal,
                                             err_msg='failed ky equivalence on row j = '+str(j))
    t2 = time.time()
    print 'time for %s = %.2f'%(funcname(),t2-t1)

def test_kxky_plusone():
    """Test that the basic properties of kx and ky are right...
    But increment testshape used in test_kxky by one to test both odd and even cases.
    """
    import time
    t1 = time.time()
    kx, ky = galsim.utilities.kxky((4 + 1, 4 + 1))
    kxref = np.array([0., 0.2, 0.4, -0.4, -0.2]) * 2. * np.pi
    kyref = np.array([0., 0.2, 0.4, -0.4, -0.2]) * 2. * np.pi
    for i in xrange(4 + 1):
        np.testing.assert_array_almost_equal(kx[i, :], kxref, decimal=decimal,
                                             err_msg='failed kx equivalence on row i = '+str(i))
    for j in xrange(4 + 1):
        np.testing.assert_array_almost_equal(ky[:, j], kyref, decimal=decimal,
                                             err_msg='failed ky equivalence on row j = '+str(j))
    t2 = time.time()
    print 'time for %s = %.2f'%(funcname(),t2-t1)

def test_check_all_contiguous():
    """Test all galsim.optics outputs are C-contiguous as required by the galsim.Image class.
    """
    import time
    t1 = time.time()
    #Check that roll2d outputs contiguous arrays whatever the input
    imcstyle = np.random.random(size=testshape)
    rolltest = galsim.utilities.roll2d(imcstyle, (+1, -1))
    assert rolltest.flags.c_contiguous
    imfstyle = np.random.random(size=testshape).T
    rolltest = galsim.utilities.roll2d(imfstyle, (+1, -1))
    assert rolltest.flags.c_contiguous
    # Check kx, ky
    kx, ky = galsim.utilities.kxky(testshape)
    assert kx.flags.c_contiguous
    assert ky.flags.c_contiguous
    t2 = time.time()
    print 'time for %s = %.2f'%(funcname(),t2-t1)

def test_interleave():
  # 1) Dummy image
  x_size, y_size = 16,10
  n1, n2 = 2, 5
  im_list = []
  for i in xrange(n1):
     for j in xrange(n2):
         x = np.arange(-x_size+2.*i/n1,x_size+2.*i/n1,2)
         y = np.arange(-y_size+2.*j/n2,y_size+2.*j/n2,2)
         X,Y = np.meshgrid(y,x)
         im = galsim.Image(100.*X+Y)
         im_list.append(im)

  im1 = interleave(im_list,n2,n1)
  x = np.arange(-x_size,x_size,2.0/n1)
  y = np.arange(-y_size,y_size,2.0/n2)
  X,Y = np.meshgrid(y,x)
  im2 = galsim.Image(100.*X+Y)

  print "Size of 'im1' = ", im1.array.shape
  print "Size of 'im2' = ", im2.array.shape

  #fig,ax = plt.subplots(1)
  #ax.hist(im1.array.reshape(n1*n2*x_size*y_size,1)-im2.array.reshape(n1*n2*x_size*y_size,1),20)
  #plt.show()
  np.testing.assert_array_almost_equal(im1.array,im2.array,decimal=11,err_msg="Interleave failed")

  # 2) With galsim Gaussian
  g = galsim.Gaussian(sigma=3.7,flux=1000.)
  gal = g#alsim.Convolve([g,galsim.Pixel(1.0)])
  im_list = []
  n = 1
  for j in xrange(n):
    for i in xrange(n):
      im = galsim.Image(16*n,16*n)
      gal.drawImage(image=im,scale=1.0,method='no_pixel',offset=galsim.PositionD(-(i+0.5)/n+0.5,-(j+0.5)/n+0.5))
      im_list.append(im)

  img = interleave(im_list,n,n)
  print "Size of the Gaussian = ", img.FindAdaptiveMom().moments_sigma/n, img.FindAdaptiveMom().moments_centroid, img.array.max()
  im = galsim.Image(16*n*n,16*n*n)
  g = galsim.Gaussian(sigma=3.7,flux=1000.*n*n)
  gal = g#alsim.Convolve([g,galsim.Pixel(1.0)])
  gal.drawImage(image=im,method='no_pixel',offset=galsim.PositionD(0.0,0.0),scale=1.0/n)
  print " must match this - ", im.FindAdaptiveMom().moments_sigma/n, im.FindAdaptiveMom().moments_centroid, im.array.max()
  print "Central part of the images:"
  print np.round(img.array[8*n*n-2:8*n*n+3,8*n*n-2:8*n*n+3],3)
  print np.round(im.array[8*n*n-2:8*n*n+3,8*n*n-2:8*n*n+3],3)
  np.testing.assert_array_equal(img.array,im.array,err_msg="Gaussian images dont match")

  # 3) With actual WFIRST PSFs
  import galsim.wfirst as wfirst
  filters = wfirst.getBandpasses(AB_zeropoint=True)
  base_size = 32
  n = 2
  img_size = n*base_size
  img = galsim.Image(img_size,img_size)
  superimg = galsim.Image(img_size,img_size)

  for filter_name, filter_ in filters.iteritems():
      im_list = []
      PSFs = wfirst.getPSF(SCAs=7,approximate_struts=True,wavelength=filter_)
      PSF = galsim.Convolve([PSFs[7],galsim.Pixel(wfirst.pixel_scale)])
      PSF.drawImage(image=superimg,scale=wfirst.pixel_scale/n,method='no_pixel')
      print "Max pix val = ", superimg.array.max()
      for j in xrange(n):
          for i in xrange(n):
              im = galsim.Image(base_size,base_size)
              offset = galsim.PositionD(-(i+0.5)/n+0.5,-(j+0.5)/n+0.5)
              PSF.drawImage(image=im,scale=wfirst.pixel_scale,method='no_pixel',offset=offset)
              im_list.append(im)

      img = interleave(im_list,n,n)
      np.testing.assert_array_almost_equal(img.array,n*n*superimg.array,decimal=6,err_msg='WFIRST PSFs disagree for '+filter_name)
      print "Test passed for "+filter_name

if __name__ == "__main__":
    test_roll2d_circularity()
    test_roll2d_fwdbck()
    test_roll2d_join()
    test_kxky()
    test_kxky_plusone()
    test_check_all_contiguous()
    test_interleave()
