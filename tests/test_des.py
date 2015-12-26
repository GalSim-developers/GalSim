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
    print 'obj1 = ',obj1

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
    print 'obj2 = ',obj2

    # create an object list
    objlist = [obj1, obj2]

    # save objects to MEDS file
    filename_meds = 'output/test_meds.fits'
    print 'file_name = ',filename_meds
    print 'objlist = ',objlist
    galsim.des.write_meds(objlist, filename_meds, clobber=True)
    print 'wrote MEDS file %s ' % filename_meds

    # test functions in des_meds.py
    print 'reading %s' % filename_meds
    try:
        import meds
    except ImportError:
        print 'Failed to import meds.  Unable to do tests of meds file.'
        # Note that while there are no tests prior to this, the above still checks for 
        # syntax errors in the meds creation software, so it's still worth running as part
        # of the normal unit test runs.
    m = meds.MEDS(filename_meds)

    # get the catalog
    cat = m.get_cat()

    # get number of objects
    n_obj = len(cat)

    # check if the number of objects is correct
    numpy.testing.assert_equal(n_obj,n_obj_test)

    print 'number of objects is %d' % n_obj
    print 'testing if loaded images are the same as original images'
    
    # loop over objects and exposures - test get_cutout
    for iobj in range(n_obj):

        # check ID is correct
        numpy.testing.assert_equal(cat['id'][iobj], iobj+1)

        # get number of cutouts and check if it's right
        n_cut = cat['ncutout'][iobj]
        numpy.testing.assert_equal(n_cut,n_cut_test)

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
            numpy.testing.assert_array_equal(img, objlist[iobj].images[icut].array)
            numpy.testing.assert_array_equal(wth, objlist[iobj].weights[icut].array)
            numpy.testing.assert_array_equal(seg, objlist[iobj].segs[icut].array)
            wcs_orig = objlist[iobj].wcs[icut]
            wcs_array_orig = numpy.array(
                    [ wcs_orig.dudx, wcs_orig.dudy, wcs_orig.dvdx, wcs_orig.dvdy,
                      wcs_orig.origin.x, wcs_orig.origin.y ])
            numpy.testing.assert_array_equal(wcs_array_meds, wcs_array_orig)

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
        numpy.testing.assert_array_equal(true_mosaic_img, img)
        numpy.testing.assert_array_equal(true_mosaic_wth, wth)
        numpy.testing.assert_array_equal(true_mosaic_seg, seg)

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
        'output' : { 'type' : 'des_meds',
                     'nobjects' : nobj,
                     'nstamps_per_object' : n_per_obj,
                     'file_name' : file_name
                   }
    }

    import logging
    logging.basicConfig(format="%(message)s", level=logging.INFO, stream=sys.stdout)
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
    print 'reading %s' % file_name
    try:
        import meds
    except ImportError:
        print 'Failed to import meds.  Unable to do tests of meds file.'
        # Note that while there are no tests prior to this, the above still checks for 
        # syntax errors in the meds creation software, so it's still worth running as part
        # of the normal unit test runs.
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

        numpy.testing.assert_array_equal(ref_im.array, meds_im_array)

        meds_wt_array = m.get_mosaic(iobj, type='weight')
        ref_wt_im = galsim.fits.read(ref_file, hdu=1)
        numpy.testing.assert_array_equal(ref_wt_im.array, meds_wt_array)
        meds_seg_array = m.get_mosaic(iobj, type='seg')
        ref_seg_im = galsim.fits.read(ref_file, hdu=2)
        ref_seg_im = 1 - ref_seg_im  # The seg mag is 1 where badpix == 0
        numpy.testing.assert_array_equal(ref_seg_im.array, meds_seg_array)

    t2 = time.time()
    print 'time for %s = %.2f'%(funcname(),t2-t1)

if __name__ == "__main__":
    test_meds()
    test_meds_config()

