# Copyright (c) 2012-2023 by the GalSim developers team on GitHub
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
from galsim._pyfits import pyfits

@timer
def test_meds():
    """
    Create two objects, each with three exposures. Save them to a MEDS file.
    Load the MEDS file. Compare the created objects with the one read by MEDS.
    """
    # initialise empty MultiExposureObject list
    objlist = []

    # we will be using 2 objects for testing, each with 3 cutouts
    n_obj_test = 2
    n_cut_test = 3

    # set the image size
    box_size = 32

    # first obj
    img11 = galsim.Image(box_size, box_size, init_value=111)
    img12 = galsim.Image(box_size, box_size, init_value=112)
    img13 = galsim.Image(box_size, box_size, init_value=113)
    seg11 = galsim.Image(box_size, box_size, init_value=121)
    seg12 = galsim.Image(box_size, box_size, init_value=122)
    seg13 = galsim.Image(box_size, box_size, init_value=123)
    wth11 = galsim.Image(box_size, box_size, init_value=131)
    wth12 = galsim.Image(box_size, box_size, init_value=132)
    wth13 = galsim.Image(box_size, box_size, init_value=133)
    psf11 = galsim.Image(box_size, box_size, init_value=141)
    psf12 = galsim.Image(box_size, box_size, init_value=142)
    psf13 = galsim.Image(box_size, box_size, init_value=143)
    dudx = 11.1; dudy = 11.2; dvdx = 11.3; dvdy = 11.4; x0 = 11.5; y0 = 11.6;
    wcs11  = galsim.AffineTransform(dudx, dudy, dvdx, dvdy, galsim.PositionD(x0, y0))
    dudx = 12.1; dudy = 12.2; dvdx = 12.3; dvdy = 12.4;
    wcs12  = galsim.JacobianWCS(dudx, dudy, dvdx, dvdy)
    wcs13  = galsim.PixelScale(13)


    # create lists
    images = [img11, img12, img13]
    weight = [wth11, wth12, wth13]
    seg =    [seg11, seg12, seg13]
    psf =    [psf11, psf12, psf13]
    wcs =    [wcs11, wcs12, wcs13]

    # create object
    obj1 = galsim.des.MultiExposureObject(images=images, weight=weight, seg=seg, psf=psf,
                                          wcs=wcs, id=1)

    # second obj
    img21 = galsim.Image(box_size, box_size, init_value=211)
    img22 = galsim.Image(box_size, box_size, init_value=212)
    img23 = galsim.Image(box_size, box_size, init_value=213)
    seg21 = galsim.Image(box_size, box_size, init_value=221)
    seg22 = galsim.Image(box_size, box_size, init_value=222)
    seg23 = galsim.Image(box_size, box_size, init_value=223)
    wth21 = galsim.Image(box_size, box_size, init_value=231)
    wth22 = galsim.Image(box_size, box_size, init_value=332)
    wth23 = galsim.Image(box_size, box_size, init_value=333)
    psf21 = galsim.Image(box_size, box_size, init_value=241)
    psf22 = galsim.Image(box_size, box_size, init_value=342)
    psf23 = galsim.Image(box_size, box_size, init_value=343)

    dudx = 21.1; dudy = 21.2; dvdx = 21.3; dvdy = 21.4; x0 = 21.5; y0 = 21.6;
    wcs21  = galsim.AffineTransform(dudx, dudy, dvdx, dvdy, galsim.PositionD(x0, y0))
    dudx = 22.1; dudy = 22.2; dvdx = 22.3; dvdy = 22.4;
    wcs22  = galsim.JacobianWCS(dudx, dudy, dvdx, dvdy)
    wcs23  = galsim.PixelScale(23)

    # create lists
    images = [img21, img22, img23]
    weight = [wth21, wth22, wth23]
    seg =    [seg21, seg22, seg23]
    psf =    [psf21, psf22, psf23]
    wcs =    [wcs21, wcs22, wcs23]

    # create object
    # This time put the wcs in the image and get it there.
    img21.wcs = wcs21
    img22.wcs = wcs22
    img23.wcs = wcs23
    obj2 = galsim.des.MultiExposureObject(images=images, weight=weight, seg=seg, psf=psf, id=2)

    obj3 = galsim.des.MultiExposureObject(images=images, id=3)

    # create an object list
    objlist = [obj1, obj2]

    # save objects to MEDS file
    filename_meds = 'output/test_meds.fits'
    galsim.des.WriteMEDS(objlist, filename_meds, clobber=True)

    bad1 = galsim.Image(32, 48, init_value=0)
    bad2 = galsim.Image(35, 35, init_value=0)
    bad3 = galsim.Image(48, 48, init_value=0)

    with assert_raises(TypeError):
        galsim.des.MultiExposureObject(images=img11)
    with assert_raises(galsim.GalSimValueError):
        galsim.des.MultiExposureObject(images=[])
    with assert_raises(galsim.GalSimValueError):
        galsim.des.MultiExposureObject(images=[bad1])
    with assert_raises(galsim.GalSimValueError):
        galsim.des.MultiExposureObject(images=[bad2])
    with assert_raises(galsim.GalSimValueError):
        galsim.des.MultiExposureObject(images=[img11,bad3])
    with assert_raises(TypeError):
        galsim.des.MultiExposureObject(images=images, weight=wth11)
    with assert_raises(galsim.GalSimValueError):
        galsim.des.MultiExposureObject(images=images, weight=[])
    with assert_raises(galsim.GalSimValueError):
        galsim.des.MultiExposureObject(images=[img11], weight=[bad3])
    with assert_raises(galsim.GalSimValueError):
        galsim.des.MultiExposureObject(images=[img11], psf=[bad1])
    with assert_raises(galsim.GalSimValueError):
        galsim.des.MultiExposureObject(images=[img11], psf=[bad2])
    with assert_raises(galsim.GalSimValueError):
        galsim.des.MultiExposureObject(images=[img11, img12], psf=[bad2, psf12])
    with assert_raises(TypeError):
        galsim.des.MultiExposureObject(images=images, wcs=wcs11)
    celestial_wcs = galsim.FitsWCS("DECam_00154912_12_header.fits", dir='des_data')
    with assert_raises(galsim.GalSimValueError):
        galsim.des.MultiExposureObject(images=[img11], wcs=[celestial_wcs])

    # Check the one with no psf, weight, etc.
    filename_meds2 = 'output/test_meds_image_only.fits'
    galsim.des.WriteMEDS([obj3], filename_meds2, clobber=True)


    # Note that while there are no tests prior to this, the above still checks for
    # syntax errors in the meds creation software, so it's still worth running as part
    # of the normal unit tests.
    # But for the rest of the tests, we'll use the meds module to make sure our code
    # stays in sync with any changes there.
    try:
        import meds
        # Meds will import this, so check for this too.
        import fitsio
    except ImportError:
        print('Failed to import either meds or fitsio.  Unable to do tests of meds file.')
        return

    # Run meds module's validate function
    try:
        meds.util.validate_meds(filename_meds)
        meds.util.validate_meds(filename_meds2)
    except AttributeError:
        print('Seems to be the wrong meds package.  Unable to do tests of meds file.')
        return

    m = meds.MEDS(filename_meds)

    # Check the image_info extension:
    ref_info = meds.util.get_image_info_dtype(1)
    info = m.get_image_info()
    print('info = ',info)
    for name, dt in ref_info:
        dt = numpy.dtype(dt)
        print(name, dt, info.dtype[name], dt.char, info.dtype[name].char)
        assert name in info.dtype.names, "column %s not present in image_info extension"%name
        # I think S and U for this purpose are equivalent.
        # But I'm finding S in the reference, and U in info.
        c = info.dtype[name].char
        c = 'S' if c == 'U' else c
        assert dt.char == c, "column %s is the wrong type"%name

    # Check the basic structure of the object_data extension
    cat = m.get_cat()
    ref_data = meds.util.get_meds_output_dtype(1)
    for tup in ref_data:
        # Some of these tuples have 3 items, not 2.  The last two are the full dtype tuple.
        name = tup[0]
        if len(tup) == 2:
            dt = tup[1]
        else:
            dt = tup[1:]
        dt = numpy.dtype(dt)
        print(name, dt, cat.dtype[name], dt.char, cat.dtype[name].char)
        assert name in cat.dtype.names, "column %s not present in object_data extension"%name
        assert dt.char == cat.dtype[name].char, "column %s is the wrong type"%name

    # Check that we have the right number of objects.
    n_obj = len(cat)
    print('number of objects is %d' % n_obj)
    numpy.testing.assert_equal(n_obj,n_obj_test,
                               err_msg="MEDS file has wrong number of objects")

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
            img = m.get_cutout(iobj, icut, type='image')
            wth = m.get_cutout(iobj, icut, type='weight')
            seg = m.get_cutout(iobj, icut, type='seg')
            psf = m.get_psf(iobj, icut)
            wcs_meds = m.get_jacobian(iobj, icut)
            # Note: col == x, row == y.
            wcs_array_meds= numpy.array(
                [ wcs_meds['dudcol'], wcs_meds['dudrow'],
                  wcs_meds['dvdcol'], wcs_meds['dvdrow'],
                  wcs_meds['col0'], wcs_meds['row0'] ] )

            # compare
            numpy.testing.assert_array_equal(img, objlist[iobj].images[icut].array,
                                             err_msg="MEDS cutout has wrong img for object %d"%iobj)
            numpy.testing.assert_array_equal(wth, objlist[iobj].weight[icut].array,
                                             err_msg="MEDS cutout has wrong wth for object %d"%iobj)
            numpy.testing.assert_array_equal(seg, objlist[iobj].seg[icut].array,
                                             err_msg="MEDS cutout has wrong seg for object %d"%iobj)
            numpy.testing.assert_array_equal(psf, objlist[iobj].psf[icut].array,
                                             err_msg="MEDS cutout has wrong psf for object %d"%iobj)
            wcs_orig = objlist[iobj].wcs[icut]
            wcs_array_orig = numpy.array(
                    [ wcs_orig.dudx, wcs_orig.dudy, wcs_orig.dvdx, wcs_orig.dvdy,
                      wcs_orig.origin.x, wcs_orig.origin.y ])
            numpy.testing.assert_array_equal(wcs_array_meds, wcs_array_orig,
                                             err_msg="MEDS cutout has wrong wcs for object %d"%iobj)

        # get the mosaic to compare with originals
        img = m.get_mosaic( iobj, type='image')
        wth = m.get_mosaic( iobj, type='weight')
        seg = m.get_mosaic( iobj, type='seg')
        # There is currently no get_mosaic option for the psfs.
        #psf = m.get_mosaic( iobj, type='psf')
        psf = numpy.concatenate([m.get_psf(iobj,icut) for icut in range(n_cut)], axis=0)

        # get the concatenated images - create the true mosaic
        true_mosaic_img = numpy.concatenate([x.array for x in objlist[iobj].images], axis=0)
        true_mosaic_wth = numpy.concatenate([x.array for x in objlist[iobj].weight], axis=0)
        true_mosaic_seg = numpy.concatenate([x.array for x in objlist[iobj].seg],    axis=0)
        true_mosaic_psf = numpy.concatenate([x.array for x in objlist[iobj].psf],    axis=0)

        # compare
        numpy.testing.assert_array_equal(true_mosaic_img, img,
                                         err_msg="MEDS mosaic has wrong img for object %d"%iobj)
        numpy.testing.assert_array_equal(true_mosaic_wth, wth,
                                         err_msg="MEDS mosaic has wrong wth for object %d"%iobj)
        numpy.testing.assert_array_equal(true_mosaic_seg, seg,
                                         err_msg="MEDS mosaic has wrong seg for object %d"%iobj)
        numpy.testing.assert_array_equal(true_mosaic_psf, psf,
                                         err_msg="MEDS mosaic has wrong psf for object %d"%iobj)



@timer
def test_meds_config(run_slow):
    """
    Create a meds file from a config and compare with a manual creation.
    """
    # Some parameters:
    if run_slow:
        nobj = 5
        n_per_obj = 8
    else:
        nobj = 2
        n_per_obj = 3
    file_name = 'output/test_meds.fits'
    stamp_size = 64
    pixel_scale = 0.26
    seed = 5757231
    g1 = -0.17
    g2 = 0.23

    # generate offsets that depend on the object num so can be easily reproduced
    # for testing below
    offset_x = '$ np.sin(999.*(@obj_num+1))'
    offset_y = '$ np.sin(998.*(@obj_num+1))'
    def get_offset(obj_num):
        return galsim.PositionD(np.sin(999.*(obj_num+1)),np.sin(998.*(obj_num+1)))

    # The config dict to write some images to a MEDS file
    config = {
        'gal' : { 'type' : 'Sersic',
                  'n' : 1.3,
                  'half_light_radius' : { 'type' : 'Sequence', 'first' : 0.7, 'step' : 0.1,
                                          'repeat' : n_per_obj },
                  'shear' : { 'type' : 'G1G2', 'g1' : g1, 'g2' : g2 },
                },
        'psf' : { 'type' : 'Moffat', 'beta' : 2.9, 'fwhm' : 0.7 },
        'image' : { 'pixel_scale' : pixel_scale,
                    'size' : stamp_size, 'random_seed' : seed },
        'output' : { 'type' : 'MEDS',
                     'nobjects' : nobj,
                     'nstamps_per_object' : n_per_obj,
                     'file_name' : file_name
                   }
    }

    import logging
    logging.basicConfig(format="%(message)s", level=logging.WARN, stream=sys.stdout)
    logger = logging.getLogger('test_meds_config')
    galsim.config.BuildFile(galsim.config.CopyConfig(config), logger=logger)

    # Add in badpix and offset so we run both with and without options.
    config = galsim.config.CleanConfig(config)
    config['image']['offset'] = { 'type' : 'XY' , 'x' : offset_x, 'y' : offset_y }
    config['output']['badpix'] = {}
    # These three are just added for coverage really.
    config['output']['weight'] = {}
    config['output']['psf'] = {}
    config['output']['meds_get_offset'] = {}
    galsim.config.BuildFile(galsim.config.CopyConfig(config), logger=logger)

    # Scattered image is invalid with MEDS output
    config = galsim.config.CleanConfig(config)
    config['image'] = {
        'type' : 'Scattered',
        'nobjects' : 20,
        'pixel_scale' : pixel_scale,
        'size' : stamp_size ,
    }
    with assert_raises(galsim.GalSimConfigError):
        galsim.config.BuildFile(galsim.config.CopyConfig(config), logger=logger)

    # Now repeat, making a separate file for each
    config = galsim.config.CleanConfig(config)
    config['gal']['half_light_radius'] = { 'type' : 'Sequence', 'first' : 0.7, 'step' : 0.1,
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
                        'offset' : { 'type' : 'XY' , 'x' : offset_x, 'y' : offset_y },
                        'stamp_size' : stamp_size,
                        'random_seed' : seed
                      }
    galsim.config.Process(galsim.config.CopyConfig(config), logger=logger)

    try:
        import meds
        import fitsio
    except ImportError:
        print('Failed to import either meds or fitsio.  Unable to do tests of meds file.')
        return

    try:
        m = meds.MEDS(file_name)
    except AttributeError:
        print('Seems to be the wrong meds package.  Unable to do tests of meds file.')
        return

    assert m.size == nobj

    # Test that the images made as meds mosaics match the ones written to the separate fits files.
    cat = m.get_cat()
    for iobj in range(nobj):
        ref_file = os.path.join('output','test_meds%d.fits' % iobj)
        ref_im = galsim.fits.read(ref_file)

        meds_im_array = m.get_mosaic(iobj)

        # Just for reference.  If you get an error, you can open this file with ds9.
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

        meds_psf_array = numpy.concatenate([m.get_psf(iobj,icut) for icut in range(n_per_obj)],
                                           axis=0)
        ref_psf_im = galsim.fits.read(ref_file, hdu=3)
        numpy.testing.assert_array_equal(ref_psf_im.array, meds_psf_array,
                                         err_msg="config MEDS has wrong psf for object %d"%iobj)

    # Check that the various positions and sizes are set correctly.
    info = m.get_image_info()
    for iobj in range(nobj):
        n_cut = cat['ncutout'][iobj]
        for icut in range(n_cut):

            # This should be stamp_size
            box_size = cat['box_size'][iobj]
            numpy.testing.assert_almost_equal(box_size, stamp_size)

            # cutout_row and cutout_col are the "zero-offset"
            # position of the object in the stamp. In this convention, the center
            # of the first pixel is at (0,0), call this meds_center.
            # This means cutout_row/col should be the same as meds_center + offset
            offset = get_offset(iobj*n_cut+icut)
            meds_center = galsim.PositionD( (box_size-1.)/2., (box_size-1.)/2. )
            cutout_row = cat['cutout_row'][iobj][icut]
            cutout_col = cat['cutout_col'][iobj][icut]
            print('cutout_row, cutout_col = ',cutout_col, cutout_row)
            numpy.testing.assert_almost_equal(cutout_col,
                                              (meds_center+offset).x)
            numpy.testing.assert_almost_equal(cutout_row,
                                              (meds_center+offset).y)

            # The col0 and row0 here should be the same.
            wcs_meds = m.get_jacobian(iobj, icut)
            numpy.testing.assert_almost_equal(wcs_meds['col0'],
                                              (meds_center+offset).x)
            numpy.testing.assert_almost_equal(wcs_meds['row0'],
                                              (meds_center+offset).y)


            # The centroid should be (roughly) at the nominal center + offset
            img = m.get_cutout(iobj, icut, type='image')
            x,y = numpy.meshgrid( range(img.shape[1]), range(img.shape[0]) )
            itot = numpy.sum(img)
            ix = numpy.sum(x*img)
            iy = numpy.sum(y*img)
            print('centroid = ',ix/itot, iy/itot)

            print('center + offset = ',meds_center + offset)
            numpy.testing.assert_almost_equal(ix/itot,
                                              (meds_center+offset).x, decimal=2)
            numpy.testing.assert_almost_equal(iy/itot,
                                              (meds_center+offset).y, decimal=2)

            # The orig positions are irrelevant and should be 0.
            orig_row = cat['orig_row'][iobj][icut]
            orig_col = cat['orig_col'][iobj][icut]
            orig_start_row = cat['orig_start_row'][iobj][icut]
            orig_start_col = cat['orig_start_col'][iobj][icut]
            numpy.testing.assert_almost_equal(orig_col, 0.)
            numpy.testing.assert_almost_equal(orig_row, 0.)
            numpy.testing.assert_almost_equal(orig_start_col, 0.)
            numpy.testing.assert_almost_equal(orig_start_row, 0.)

            # This should be also be 0.
            numpy.testing.assert_almost_equal(info['position_offset'], 0.)


@timer
def test_nan_fits():
    """Test reading in a FITS file that has NAN.0 entries in the header.

    This test is specifically in response to issue #602.
    """
    import warnings
    from galsim._pyfits import pyfits
    # Older pyfits versions don't have this, so just skip this test then.
    if not hasattr(pyfits, 'verify'): return

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
    with pyfits.open(file_name) as fp:
        try:
            data = fp[1].data
            print('Able to read FITS file with NAN.0 without any problem.')
        except:
            print('Running verify to fix the problematic FITS header.')
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore",category=pyfits.verify.VerifyWarning)
                fp[1].verify('fix')
            # This should work now.
            data = fp[1].data
        header = fp[1].header

    assert data.shape == ref_bounds.numpyShape()

    # Check a direct read of the header with GSFitsWCS
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


@timer
def test_psf():
    """Test the two kinds of PSF files we have in DES.
    """
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

        # This choice of a star is fairly isolated from neighbors, isn't too near an edge or a tape
        # bump, and doesn't have any noticeable image artifacts in its vicinity.
        x = xvals[mask][idx][27]
        y = yvals[mask][idx][27]
        print('Using x,y = ',x,y)
        image_pos = galsim.PositionD(x,y)
        print('size, mag = ',size[mask][idx][27], mag[mask][idx][27])

        data = galsim.fits.read(image_file, dir=example_data_dir)
        b = galsim.BoundsI(int(x)-15, int(x)+16, int(y)-15, int(y)+16)
        data_stamp = data[b]

        header = galsim.fits.FitsHeader(image_file, dir=example_data_dir)
        sky_level = header['SKYBRITE']
        data_stamp -= sky_level

        raw_meas = data_stamp.FindAdaptiveMom()
        print('raw_meas = ',raw_meas)
        ref_size = raw_meas.moments_sigma
        ref_shape = raw_meas.observed_shape
        print('ref size: ',ref_size)
        print('ref shape: ',ref_shape)

    except OSError:
        x,y = 1195.64074707, 1276.63427734
        image_pos = galsim.PositionD(x,y)
        b = galsim.BoundsI(int(x)-15, int(x)+16, int(y)-15, int(y)+16)
        ref_size = 1.80668628216
        ref_shape = galsim.Shear(g1=0.022104322221,g2=-0.130925191715)

    # First the PSFEx model using the wcs_file to get the model is sky coordinates.
    psfex = galsim.des.DES_PSFEx(psfex_file, wcs_file, dir=data_dir)
    psf = psfex.getPSF(image_pos)

    # The getLocalWCS function should return a local WCS
    assert psfex.getLocalWCS(image_pos).isLocal()

    # Draw the postage stamp image
    # Note: the PSF already includes the pixel response, so draw with method 'no_pixel'.
    stamp = psf.drawImage(wcs=wcs.local(image_pos), bounds=b, method='no_pixel')
    print('wcs = ',wcs.local(image_pos))
    meas = stamp.FindAdaptiveMom()
    print('meas = ',meas)
    print('pixel scale = ',stamp.wcs.minLinearScale(image_pos=image_pos))
    print('cf sizes: ',ref_size, meas.moments_sigma)
    print('cf shapes: ',ref_shape, meas.observed_shape)
    # The agreement for a single star is not great of course, not even 2 decimals.
    # Divide by 2 to get agreement at 2 dp.
    numpy.testing.assert_almost_equal(meas.moments_sigma/2, ref_size/2, decimal=2,
                                      err_msg="PSFEx size doesn't match")
    numpy.testing.assert_almost_equal(meas.observed_shape.g1/2, ref_shape.g1/2, decimal=2,
                                      err_msg="PSFEx shape.g1 doesn't match")
    numpy.testing.assert_almost_equal(meas.observed_shape.g2/2, ref_shape.g2/2, decimal=2,
                                      err_msg="PSFEx shape.g2 doesn't match")

    # Repeat without the wcs_file argument, so the model is in chip coordinates.
    # Also check the functionality where the file is already open.
    with pyfits.open(os.path.join(data_dir, psfex_file)) as hdu_list:
        psfex = galsim.des.DES_PSFEx(hdu_list[1])
    psf = psfex.getPSF(image_pos)

    # In this case, the getLocalWCS function won't return anything useful.
    assert psfex.getLocalWCS(image_pos) is None

    # Draw the postage stamp image.  This time in image coords, so pixel_scale = 1.0.
    stamp = psf.drawImage(bounds=b, scale=1.0, method='no_pixel')
    meas = stamp.FindAdaptiveMom()
    numpy.testing.assert_almost_equal(meas.moments_sigma/2, ref_size/2, decimal=2,
                                      err_msg="no-wcs PSFEx size doesn't match")
    numpy.testing.assert_almost_equal(meas.observed_shape.g1/2, ref_shape.g1/2, decimal=2,
                                      err_msg="no-wcs PSFEx shape.g1 doesn't match")
    numpy.testing.assert_almost_equal(meas.observed_shape.g2/2, ref_shape.g2/2, decimal=2,
                                      err_msg="no-wcs PSFEx shape.g2 doesn't match")

    with assert_raises(TypeError):
        # file_name must be a string.
        galsim.des.DES_PSFEx(psf, wcs=wcs_file, dir=data_dir)
    with assert_raises(galsim.GalSimError):
        # Cannot provide both image_file_name and wcs
        galsim.des.DES_PSFEx(psfex_file, image_file_name=wcs_file, wcs=wcs_file, dir=data_dir)
    with assert_raises(OSError):
        # This one doesn't exist.
        galsim.des.DES_PSFEx('nonexistant.psf', wcs=wcs_file, dir=data_dir)
    with assert_raises(OSError):
        # This one exists, but has invalid header parameters.
        galsim.des.DES_PSFEx('invalid_psfcat.psf', wcs=wcs_file, dir=data_dir)

    # Now the shapelet PSF model.  This model is already in sky coordinates, so no wcs_file needed.
    fitpsf = galsim.des.DES_Shapelet(os.path.join(data_dir,fitpsf_file))
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

    with assert_raises(galsim.GalSimBoundsError):
        fitpsf.getPSF(image_pos = galsim.PositionD(4000, 5000))


@timer
def test_psf_config():
    """Test building the two PSF types using the config layer.
    """
    data_dir = 'des_data'
    psfex_file = "DECam_00154912_12_psfcat.psf"
    fitpsf_file = "DECam_00154912_12_fitpsf.fits"
    wcs_file = "DECam_00154912_12_header.fits"

    image_pos = galsim.PositionD(123.45, 543.21)

    config = {
        'input' : {
            'des_shapelet' : { 'dir' : data_dir, 'file_name' : fitpsf_file },
            'des_psfex' : [
                { 'dir' : data_dir, 'file_name' : psfex_file },
                { 'dir' : data_dir, 'file_name' : psfex_file, 'image_file_name' : wcs_file },
            ]
        },

        'psf1' : { 'type' : 'DES_Shapelet' },
        'psf2' : { 'type' : 'DES_PSFEx', 'num' : 0 },
        'psf3' : { 'type' : 'DES_PSFEx', 'num' : 1 },
        'psf4' : { 'type' : 'DES_Shapelet', 'image_pos' : galsim.PositionD(567,789), 'flux' : 179,
                   'gsparams' : { 'folding_threshold' : 1.e-4 } },
        'psf5' : { 'type' : 'DES_PSFEx', 'image_pos' : galsim.PositionD(789,567), 'flux' : 388,
                   'gsparams' : { 'folding_threshold' : 1.e-4 } },
        'bad1' : { 'type' : 'DES_Shapelet', 'image_pos' : galsim.PositionD(5670,789) },

        # This would normally be set by the config processing.  Set it manually here.
        'image_pos' : image_pos,
    }

    galsim.config.ProcessInput(config)

    psf1a = galsim.config.BuildGSObject(config, 'psf1')[0]
    fitpsf = galsim.des.DES_Shapelet(fitpsf_file, dir=data_dir)
    psf1b = fitpsf.getPSF(image_pos)
    gsobject_compare(psf1a, psf1b)

    psf2a = galsim.config.BuildGSObject(config, 'psf2')[0]
    psfex0 = galsim.des.DES_PSFEx(psfex_file, dir=data_dir)
    psf2b = psfex0.getPSF(image_pos)
    gsobject_compare(psf2a, psf2b)

    psf3a = galsim.config.BuildGSObject(config, 'psf3')[0]
    psfex1 = galsim.des.DES_PSFEx(psfex_file, wcs_file, dir=data_dir)
    psf3b = psfex1.getPSF(image_pos)
    gsobject_compare(psf3a, psf3b)

    gsparams = galsim.GSParams(folding_threshold=1.e-4)
    psf4a = galsim.config.BuildGSObject(config, 'psf4')[0]
    psf4b = fitpsf.getPSF(galsim.PositionD(567,789),gsparams=gsparams).withFlux(179)
    gsobject_compare(psf4a, psf4b)

    # Insert a wcs for thes last one.
    config['wcs'] = galsim.FitsWCS(os.path.join(data_dir,wcs_file))
    config = galsim.config.CleanConfig(config)
    galsim.config.ProcessInput(config)
    psfex2 = galsim.des.DES_PSFEx(psfex_file, dir=data_dir, wcs=config['wcs'])
    psf5a = galsim.config.BuildGSObject(config, 'psf5')[0]
    psf5b = psfex2.getPSF(galsim.PositionD(789,567),gsparams=gsparams).withFlux(388)
    gsobject_compare(psf5a, psf5b)

    del config['image_pos']
    galsim.config.RemoveCurrent(config)
    with assert_raises(galsim.GalSimConfigError):
        galsim.config.BuildGSObject(config, 'psf1')[0]
    with assert_raises(galsim.GalSimConfigError):
        galsim.config.BuildGSObject(config, 'psf2')[0]
    with assert_raises(galsim.config.gsobject.SkipThisObject):
        galsim.config.BuildGSObject(config, 'bad1')[0]


if __name__ == "__main__":
    runtests(__file__)
