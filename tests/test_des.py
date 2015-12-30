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

import numpy
import os
import sys
import galsim
import galsim.des

from galsim_test_helpers import *

try:
    import galsim
except ImportError:
    sys.path.append(os.path.abspath(os.path.join(path, "..")))
    import galsim


def test_meds():
    """
    Create two objects, each with two exposures. Save them to a MEDS file.
    Load the MEDS file. Compare the created objects with the one read by MEDS.
    """
    import time
    t1 = time.time()

    # initialise empty MultiExposureObject list
    objlist = []

    # we will be using 2 objects for testing, each with 2 cutouts
    n_obj_test = 2 
    n_cut_test = 2

    # set the image size
    box_size = 32

    # first obj
    img11 = galsim.Image(box_size, box_size, init_value=111)
    img12 = galsim.Image(box_size, box_size, init_value=112)
    seg11 = galsim.Image(box_size, box_size, init_value=121)
    seg12 = galsim.Image(box_size, box_size, init_value=122)
    wth11 = galsim.Image(box_size, box_size, init_value=131)
    wth12 = galsim.Image(box_size, box_size, init_value=132)
    dudx = 11.1; dudy = 11.2; dvdx = 11.3; dvdy = 11.4; x0 = 11.5; y0 = 11.6;
    wcs11  = galsim.AffineTransform(dudx, dudy, dvdx, dvdy, galsim.PositionD(x0, y0))
    dudx = 12.1; dudy = 12.2; dvdx = 12.3; dvdy = 12.4; x0 = 12.5; y0 = 12.6;
    wcs12  = galsim.AffineTransform(dudx, dudy, dvdx, dvdy, galsim.PositionD(x0, y0))


    # create lists
    images =   [img11, img12]
    weights =  [wth11, wth12]
    segs =     [seg11, seg12]
    wcs =      [wcs11, wcs12]

    # create object
    obj1 = galsim.des.MultiExposureObject(images=images, weights=weights, segs=segs, wcs=wcs, id=1)
    #print 'obj1 = ',obj1

    # second obj
    img21 = galsim.Image(box_size, box_size, init_value=211)
    img22 = galsim.Image(box_size, box_size, init_value=212)
    seg21 = galsim.Image(box_size, box_size, init_value=221)
    seg22 = galsim.Image(box_size, box_size, init_value=222)
    wth21 = galsim.Image(box_size, box_size, init_value=231)
    wth22 = galsim.Image(box_size, box_size, init_value=332)

    dudx = 21.1; dudy = 21.2; dvdx = 21.3; dvdy = 21.4; x0 = 21.5; y0 = 21.6;
    wcs21  = galsim.AffineTransform(dudx, dudy, dvdx, dvdy, galsim.PositionD(x0, y0))
    dudx = 22.1; dudy = 22.2; dvdx = 22.3; dvdy = 22.4; x0 = 22.5; y0 = 22.6;
    wcs22  = galsim.AffineTransform(dudx, dudy, dvdx, dvdy, galsim.PositionD(x0, y0))

    # create lists
    images =   [img21, img22]
    weights =  [wth21, wth22]
    segs =     [seg21, seg22]
    wcs =      [wcs21, wcs22]

    # create object
    # This time put the wcs in the image and get it there.
    img21.wcs = wcs21
    img22.wcs = wcs22
    obj2 = galsim.des.MultiExposureObject(images=images, weights=weights, segs=segs, id=2)
    #print 'obj2 = ',obj2

    # create an object list
    objlist = [obj1, obj2]

    # save objects to MEDS file
    filename_meds = 'output/test_meds.fits'
    #print 'file_name = ',filename_meds
    #print 'objlist = ',objlist
    galsim.des.write_meds(objlist, filename_meds, clobber=True)
    print 'wrote MEDS file %s ' % filename_meds

    # test functions in des_meds.py
    try:
        import meds
    except ImportError:
        print 'Failed to import meds.  Unable to do tests of meds file.'
        # Note that while there are no tests prior to this, the above still checks for 
        # syntax errors in the meds creation software, so it's still worth running as part
        # of the normal unit test runs.

    print 'reading %s' % filename_meds
    m = meds.MEDS(filename_meds)

    # get the catalog
    cat = m.get_cat()

    # get number of objects
    n_obj = len(cat)

    # check if the number of objects is correct
    numpy.testing.assert_equal(n_obj,n_obj_test,
                               err_msg="MEDS file has wrong number of objects")

    print 'number of objects is %d' % n_obj
    print 'testing if loaded images are the same as original images'
    
    # loop over objects and exposures - test get_cutout
    for iobj in range(n_obj):

        # check ID is correct
        numpy.testing.assert_equal(cat['id'][iobj], iobj+1,
                                   err_msg="MEDS file has wrong id for object %d"%iobj)

        # get number of cutouts and check if it's right
        n_cut = cat['ncutout'][iobj]
        numpy.testing.assert_equal(n_cut,n_cut_test,
                                   err_msg="MEDS file has wrong ncutout for object %d"%iobj)

        # loop over cutouts
        for icut in range(n_cut):

            # get the images etc to compare with originals
            img = m.get_cutout( iobj, icut, type='image')
            wth = m.get_cutout( iobj, icut, type='weight')
            seg = m.get_cutout( iobj, icut, type='seg')
            wcs_meds = m.get_jacobian(iobj, icut)
            wcs_array_meds= numpy.array( [ wcs_meds['dudrow'], wcs_meds['dudcol'],
                wcs_meds['dvdrow'], wcs_meds['dvdcol'], wcs_meds['row0'],
                wcs_meds['col0'] ] )


            # compare
            numpy.testing.assert_array_equal(img, objlist[iobj].images[icut].array,
                                             err_msg="MEDS cutout has wrong img for object %d"%iobj)
            numpy.testing.assert_array_equal(wth, objlist[iobj].weights[icut].array,
                                             err_msg="MEDS cutout has wrong wth for object %d"%iobj)
            numpy.testing.assert_array_equal(seg, objlist[iobj].segs[icut].array,
                                             err_msg="MEDS cutout has wrong seg for object %d"%iobj)
            wcs_orig = objlist[iobj].wcs[icut]
            wcs_array_orig = numpy.array(
                    [ wcs_orig.dudx, wcs_orig.dudy, wcs_orig.dvdx, wcs_orig.dvdy,
                      wcs_orig.origin.x, wcs_orig.origin.y ])
            numpy.testing.assert_array_equal(wcs_array_meds, wcs_array_orig,
                                             err_msg="MEDS cutout has wrong wcs for object %d"%iobj)

            print 'test passed get_cutout obj=%d icut=%d' % (iobj, icut)

    # loop over objects - test get_mosaic
    for iobj in range(n_obj):

        # get the mosaic to compare with originals
        img = m.get_mosaic( iobj, type='image')
        wth = m.get_mosaic( iobj, type='weight')
        seg = m.get_mosaic( iobj, type='seg')

        # get the concatenated images - create the true mosaic
        true_mosaic_img = numpy.concatenate([x.array for x in objlist[iobj].images],  axis=0)
        true_mosaic_wth = numpy.concatenate([x.array for x in objlist[iobj].weights], axis=0)
        true_mosaic_seg = numpy.concatenate([x.array for x in objlist[iobj].segs],    axis=0)

        # compare
        numpy.testing.assert_array_equal(true_mosaic_img, img,
                                         err_msg="MEDS mosaic has wrong img for object %d"%iobj)
        numpy.testing.assert_array_equal(true_mosaic_wth, wth,
                                         err_msg="MEDS mosaic has wrong wth for object %d"%iobj)
        numpy.testing.assert_array_equal(true_mosaic_seg, seg,
                                         err_msg="MEDS mosaic has wrong seg for object %d"%iobj)

        print 'test passed get_mosaic for obj=%d' % (iobj)

    t2 = time.time()
    print 'time for %s = %.2f'%(funcname(),t2-t1)


def test_meds_config():
    """
    Create a meds file from a config and compare with a manual creation.
    """
    import time
    t1 = time.time()

    # Some parameters:
    if __name__ == '__main__':
        nobj = 5
        n_per_obj = 8
    else:
        nobj = 5
        n_per_obj = 3
    file_name = 'output/test_meds.fits'
    stamp_size = 32
    pixel_scale = 0.26
    seed = 5757231
    g1 = -0.17
    g2 = 0.23

    # The config dict to write some images to a MEDS file
    config = {
        'gal' : { 'type' : 'Sersic',
                  'n' : 1.3,
                  'half_light_radius' : { 'type' : 'Sequence', 'first' : 1.7, 'step' : 0.2,
                                          'repeat' : n_per_obj },
                  'shear' : { 'type' : 'G1G2', 'g1' : g1, 'g2' : g2 },
                  'shift' : { 'type' : 'XY' , 'x' : 0.02, 'y' : 0.03}
                },
        'psf' : { 'type' : 'Moffat', 'beta' : 2.9, 'fwhm' : 0.7 },
        'image' : { 'pixel_scale' : pixel_scale,
                    'random_seed' : seed,
                    'size' : stamp_size },
        'output' : { 'type' : 'MEDS',
                     'nobjects' : nobj,
                     'nstamps_per_object' : n_per_obj,
                     'file_name' : file_name
                   }
    }

    import logging
    logging.basicConfig(format="%(message)s", level=logging.WARN, stream=sys.stdout)
    logger = logging.getLogger('test_meds_config')
    galsim.config.Process(config, logger=logger)

    # Now repeat, making a separate file for each
    config['gal']['half_light_radius'] = { 'type' : 'Sequence', 'first' : 1.7, 'step' : 0.2,
                                           'index_key' : 'file_num' }
    config['output'] = { 'type' : 'Fits',
                         'nfiles' : nobj,
                         'weight' : { 'hdu' : 1 },
                         'badpix' : { 'hdu' : 2 },
                         'psf' : { 'hdu' : 3 },
                         'dir' : 'output',
                         'file_name' : { 'type' : 'NumberedFile', 'root' : 'test_meds' }
                       }
    config['image'] = { 'type' : 'Tiled',
                        'nx_tiles' : 1,
                        'ny_tiles' : n_per_obj,
                        'pixel_scale' : pixel_scale,
                        'random_seed' : seed,
                        'stamp_size' : stamp_size 
                      }
    galsim.config.Process(config, logger=logger)

    # test functions in des_meds.py
    try:
        import meds
    except ImportError:
        print 'Failed to import meds.  Unable to do tests of meds file.'
        # Note that while there are no tests prior to this, the above still checks for 
        # syntax errors in the meds creation software, so it's still worth running as part
        # of the normal unit test runs.

    print 'reading %s' % file_name
    m = meds.MEDS(file_name)
    print 'number of objects is %d' % m.size
    assert m.size == nobj

    # get the catalog
    cat = m.get_cat()

    # loop over objects and exposures - test get_cutout
    for iobj in range(nobj):
        ref_file = os.path.join('output','test_meds%d.fits' % iobj)
        ref_im = galsim.fits.read(ref_file)

        meds_im_array = m.get_mosaic(iobj)

        alt_meds_file = os.path.join('output','test_alt_meds%d.fits' % iobj)
        alt_meds_im = galsim.Image(meds_im_array)
        alt_meds_im.write(alt_meds_file)

        numpy.testing.assert_array_equal(ref_im.array, meds_im_array,
                                         err_msg="config MEDS has wrong im for object %d"%iobj)

        meds_wt_array = m.get_mosaic(iobj, type='weight')
        ref_wt_im = galsim.fits.read(ref_file, hdu=1)
        numpy.testing.assert_array_equal(ref_wt_im.array, meds_wt_array,
                                         err_msg="config MEDS has wrong wt for object %d"%iobj)
        meds_seg_array = m.get_mosaic(iobj, type='seg')
        ref_seg_im = galsim.fits.read(ref_file, hdu=2)
        ref_seg_im = 1 - ref_seg_im  # The seg mag is 1 where badpix == 0
        numpy.testing.assert_array_equal(ref_seg_im.array, meds_seg_array,
                                         err_msg="config MEDS has wrong seg for object %d"%iobj)

    t2 = time.time()
    print 'time for %s = %.2f'%(funcname(),t2-t1)

def test_nan_fits():
    """Test reading in a FITS file that has NAN.0 entries in the header.

    This test is specifically in response to issue #602.
    """
    import warnings
    from galsim._pyfits import pyfits
    import time
    t1 = time.time()

    # The problematic file:
    file_name = "des_data/DECam_00158414_01.fits.fz"

    # These are the values we should be reading in:
    ref_bounds = galsim.BoundsI(xmin=1, xmax=2048, ymin=1, ymax=4096)
    ref_wcs = galsim.GSFitsWCS(_data = [
            'TPV',
            numpy.array([13423.2, 6307.333]),
            numpy.array([[-4.410051713005e-09, 7.286844513153e-05],
                   [-7.285161461796e-05, 3.936353853081e-09]]),
            galsim.CelestialCoord(1.1502513773465992 * galsim.radians,
                                  -0.9862866578241959 * galsim.radians),
            numpy.array(
                    [[[0.004336243600183, -0.01133740904139, 0.01202041999278, -0.004357212119479],
                      [1.013741474567, -0.01657049389296, 0.005805882078771, 0.0],
                      [0.008865811106037, -0.007472254968395, 0.0, 0.0],
                      [0.0008534196190617, 0.0, 0.0, 0.0]],
                     [[0.002619866608142, 0.9931356822158, 0.008771460618847, -0.003739430249945],
                      [-0.009422336649176, 0.01826140592329, -0.009387805146152, 0.0],
                      [-0.01066967054507, 0.007202907073747, 0.0, 0.0],
                      [-0.003683686751425, 0.0, 0.0, 0.0]]
                    ]),
            None, None])

    # First just read the file directly, not using galsim.fits.read
    fp = pyfits.open(file_name)
    try:
        data = fp[1].data
        print 'Able to read FITS file with NAN.0 without any problem.'
    except:
        print 'Running verify to fix the problematic FITS header.'
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore",category=pyfits.verify.VerifyWarning)
            fp[1].verify('fix')
        # This should work now.
        data = fp[1].data
    assert data.shape == ref_bounds.numpyShape()

    # Check a direct read of the header with GSFitsWCS
    header = fp[1].header
    wcs = galsim.GSFitsWCS(header=header)
    assert wcs == ref_wcs

    # Now read it with GalSim's fits.read function.
    # Reading this file will emit verification warnings, so we'll ignore those here for the 
    # test.  But the result should be a valid image.
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore",category=pyfits.verify.VerifyWarning)
        im = galsim.fits.read(file_name)

    assert im.bounds == ref_bounds
    assert im.wcs == ref_wcs

    t2 = time.time()
    print 'time for %s = %.2f'%(funcname(),t2-t1)

def test_psf():
    """Test the two kinds of PSF files we have in DES.
    """
    import time
    t1 = time.time()

    # The shapelet file to use for the tests
    data_dir = 'des_data'
    psfex_file = "DECam_00154912_12_psfcat.psf"
    fitpsf_file = "DECam_00154912_12_fitpsf.fits"
    wcs_file = "DECam_00154912_12_header.fits"

    wcs = galsim.FitsWCS(wcs_file, dir=data_dir)

    # We don't require that the files in example_data_dir have been downloaded.  If they 
    # haven't, then we just directly set the comparison values that we want here.
    example_data_dir = '../examples/des/des_data'
    cat_file = "DECam_00154912_12_cat.fits"
    image_file = "DECam_00154912_12.fits.fz"

    try:
        cat = galsim.Catalog(cat_file, hdu=2, dir=example_data_dir)
        size = numpy.array([ cat.getFloat(i,'FLUX_RADIUS') for i in range(cat.nobjects) ])
        mag = numpy.array([ cat.getFloat(i,'MAG_AUTO') for i in range(cat.nobjects) ])
        flags = numpy.array([ cat.getInt(i,'FLAGS') for i in range(cat.nobjects) ])
        index = numpy.array(range(cat.nobjects))
        xvals = numpy.array([ cat.getFloat(i,'X_IMAGE') for i in range(cat.nobjects) ])
        yvals = numpy.array([ cat.getFloat(i,'Y_IMAGE') for i in range(cat.nobjects) ])

        # Pick bright small objects as probable stars
        mask = (flags == 0) & (mag < 14) & (mag > 13) & (size > 2) & (size < 2.5)
        idx = numpy.argsort(size[mask])
        #print 'sizes = ',size[mask][idx].tolist()
        #print 'index = ',index[mask][idx].tolist()
        #print 'mag = ',mag[mask][idx].tolist()
        #print 'x = ',xvals[mask][idx].tolist()
        #print 'y = ',yvals[mask][idx].tolist()

        # This choice of a star is fairly isolated from neighbors, isn't too near an edge or a tape
        # bump, and doesn't have any noticeable image artifacts in its vicinity.
        x = xvals[mask][idx][27]
        y = yvals[mask][idx][27]
        print 'Using x,y = ',x,y
        image_pos = galsim.PositionD(x,y)
        print 'size, mag = ',size[mask][idx][27], mag[mask][idx][27]

        data = galsim.fits.read(image_file, dir=example_data_dir)
        b = galsim.BoundsI(int(x)-15, int(x)+16, int(y)-15, int(y)+16)
        data_stamp = data[b]

        header = galsim.fits.FitsHeader(image_file, dir=example_data_dir)
        sky_level = header['SKYBRITE']
        #print 'sky_level = ',sky_level
        data_stamp -= sky_level

        raw_meas = data_stamp.FindAdaptiveMom()
        print 'raw_meas = ',raw_meas
        #print 'pixel scale = ',data_stamp.wcs.minLinearScale(image_pos=image_pos)
        ref_size = raw_meas.moments_sigma
        ref_shape = raw_meas.observed_shape
        print 'ref size: ',ref_size
        print 'ref shape: ',ref_shape

    except OSError:
        x,y = 1195.64074707, 1276.63427734
        image_pos = galsim.PositionD(x,y)
        b = galsim.BoundsI(int(x)-15, int(x)+16, int(y)-15, int(y)+16)
        ref_size = 1.80668628216
        ref_shape = galsim.Shear(g1=0.022104322221,g2=-0.130925191715)

    # First the PSFEx model using the wcs_file to get the model is sky coordinates.
    psfex = galsim.des.DES_PSFEx(psfex_file, wcs_file, dir=data_dir)
    psf = psfex.getPSF(image_pos)
    #print 'psfex psf = ',psf

    # Draw the postage stamp image
    # Note: the PSF already includes the pixel response, so draw with method 'no_pixel'.
    stamp = psf.drawImage(wcs=wcs.local(image_pos), bounds=b, method='no_pixel')
    print 'wcs = ',wcs.local(image_pos)
    meas = stamp.FindAdaptiveMom()
    print 'meas = ',meas
    print 'pixel scale = ',stamp.wcs.minLinearScale(image_pos=image_pos)
    print 'cf sizes: ',ref_size, meas.moments_sigma
    print 'cf shapes: ',ref_shape, meas.observed_shape
    # The agreement for a single star is not great of course, not even 2 decimals.
    # Divide by 2 to get agreement at 2 dp.
    numpy.testing.assert_almost_equal(meas.moments_sigma/2, ref_size/2, decimal=2,
                                      err_msg="PSFEx size doesn't match")
    numpy.testing.assert_almost_equal(meas.observed_shape.g1/2, ref_shape.g1/2, decimal=2,
                                      err_msg="PSFEx shape.g1 doesn't match")
    numpy.testing.assert_almost_equal(meas.observed_shape.g2/2, ref_shape.g2/2, decimal=2,
                                      err_msg="PSFEx shape.g2 doesn't match")

    # Repeat without the wcs_file argument, so the model is in chip coordinates.
    psfex = galsim.des.DES_PSFEx(psfex_file, dir=data_dir)
    psf = psfex.getPSF(image_pos)
    #print 'psfex psf = ',psf

    # Draw the postage stamp image.  This time in image coords, so pixel_scale = 1.0.
    stamp = psf.drawImage(bounds=b, scale=1.0, method='no_pixel')
    meas = stamp.FindAdaptiveMom()
    numpy.testing.assert_almost_equal(meas.moments_sigma/2, ref_size/2, decimal=2,
                                      err_msg="no-wcs PSFEx size doesn't match")
    numpy.testing.assert_almost_equal(meas.observed_shape.g1/2, ref_shape.g1/2, decimal=2,
                                      err_msg="no-wcs PSFEx shape.g1 doesn't match")
    numpy.testing.assert_almost_equal(meas.observed_shape.g2/2, ref_shape.g2/2, decimal=2,
                                      err_msg="no-wcs PSFEx shape.g2 doesn't match")

    # Now the shapelet PSF model.  This model is already in sky coordinates, so no wcs_file needed.
    fitpsf = galsim.des.DES_Shapelet(fitpsf_file, dir=data_dir)
    psf = fitpsf.getPSF(image_pos)

    # Draw the postage stamp image
    # Again, the PSF already includes the pixel response.
    stamp = psf.drawImage(wcs=wcs.local(image_pos), bounds=b, method='no_pixel')
    meas = stamp.FindAdaptiveMom()
    numpy.testing.assert_almost_equal(meas.moments_sigma/2, ref_size/2, decimal=2,
                                      err_msg="Shapelet PSF size doesn't match")
    numpy.testing.assert_almost_equal(meas.observed_shape.g1/2, ref_shape.g1/2, decimal=2,
                                      err_msg="Shapelet PSF shape.g1 doesn't match")
    numpy.testing.assert_almost_equal(meas.observed_shape.g2/2, ref_shape.g2/2, decimal=2,
                                      err_msg="Shapelet PSF shape.g2 doesn't match")

    t2 = time.time()
    print 'time for %s = %.2f'%(funcname(),t2-t1)

if __name__ == "__main__":
    test_meds()
    test_meds_config()
    test_nan_fits()
    test_psf()

