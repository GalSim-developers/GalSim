# Copyright (c) 2012-2021 by the GalSim developers team on GitHub
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
import sys
import os

from ..errors import GalSimValueError, GalSimError, GalSimConfigError
from ..wcs import UniformWCS, AffineTransform
from ..image import Image, ImageI
from ..fits import writeFile
from ..position import PositionD
from ..config import OutputBuilder, ExtraOutputBuilder, BuildImages, GetFinalExtraOutput
from ..config import ParseValue, GetAllParams, GetCurrentValue
from ..config import RegisterOutputType, RegisterExtraOutput

# these image stamp sizes are available in MEDS format
BOX_SIZES = [32,48,64,96,128,192,256]
# while creating the meds file, all the data is stored in memory, and then written to disc once all
# the necessary images have been created.
# You can control the amound of memory used to prevent jamming your system.
MAX_MEMORY = 1e9
# Maximum number of exposures allowed per galaxy (incl. coadd)
MAX_NCUTOUTS = 11
# flags for unavailable data
EMPTY_START_INDEX = 9999
EMPTY_JAC_diag    = 1
EMPTY_JAC_offdiag = 0
EMPTY_SHIFT = 0

class MultiExposureObject(object):
    """
    A class containing exposures for single object, along with other information.

    The MultiExposureObject class represents multiple exposure data for a single object for the
    purpose of writing the images to a MEDS file.

    The `WriteMEDS` function can be used to write a list of MultiExposureObjects to a MEDS file.

    Parameters:
       images:      List of images of the object.
       weight:      List of weight images. [default: None]
       badpix:      List of bad pixel masks. [default: None]
       seg:         List of segmentation maps. [default: None]
       psf:         List of psf images. [default: None]
       wcs:         List of WCS transformations. [default: None]
       id:          Galaxy id. [default: 0]

    Attributes:
        self.images:        List of images of the object.
        self.weight:        List of weight maps.
        self.seg:           List of segmentation maps.
        self.psf:           List of psf images.
        self.wcs:           List of WCS transformations.
        self.n_cutouts:     Number of exposures.
        self.box_size:      Size of each exposure image.
    """

    def __init__(self, images, weight=None, badpix=None, seg=None, psf=None,
                 wcs=None, id=0, cutout_row=None, cutout_col=None):

        # Check that images is valid
        if not isinstance(images,list):
            raise TypeError('images should be a list')
        if len(images) == 0:
            raise GalSimValueError('no cutouts in this object', images)

        # Check that the box sizes are valid
        for i in range(len(images)):
            s = images[i].array.shape
            if s[0] != s[1]:
                raise GalSimValueError('Array shape %s is invalid.  Must be square'%(str(s)),
                                       images[i])
            if s[0] not in BOX_SIZES:
                raise GalSimValueError('Array shape %s is invalid.  Size must be in %s'%(
                                       str(s),str(BOX_SIZES)), images[i])
            if i > 0 and s != images[0].array.shape:
                raise GalSimValueError('Images must all be the same shape', images)

        # The others are optional, but if given, make sure they are ok.
        for lst, name, isim in ( (weight, 'weight', True), (badpix, 'badpix', True),
                                 (seg, 'seg', True), (psf, 'psf', False), (wcs, 'wcs', False) ):
            if lst is not None:
                if not isinstance(lst,list):
                    raise TypeError('%s should be a list'%name)
                if len(lst) != len(images):
                    raise GalSimValueError('%s is the wrong length'%name, lst)
                if isim:
                    for i in range(len(images)):
                        im1 = lst[i]
                        im2 = images[i]
                        if (im1.array.shape != im2.array.shape):
                            raise GalSimValueError(
                                "%s[%d] has the wrong shape."%(name, i), im1)

        # The PSF images don't have to be the same shape as the main images.
        # But make sure all psf images are square and the same shape
        if psf is not None:
            s = psf[i].array.shape
            if s[0] != s[1]:
                raise GalSimValueError(
                    'PSF array shape %s is invalid.  Must be square'%(str(s)), psf[i])
            if s[0] not in BOX_SIZES:
                raise GalSimValueError(
                    'PSF array shape %s is invalid.  Size must be in %s'%(
                        str(s),str(BOX_SIZES)), psf[i])
            if i > 0 and s != psf[0].array.shape:
                raise GalSimValueError('PSF images must all be the same shape', psf[i])

        # Check that wcs are Uniform and convert them to AffineTransforms in case they aren't.
        if wcs is not None:
            for i in range(len(wcs)):
                if not isinstance(wcs[i], UniformWCS):
                    raise GalSimValueError('wcs list should contain UniformWCS objects', wcs)
                elif not isinstance(wcs[i], AffineTransform):
                    wcs[i] = wcs[i].affine()

        self.id = id
        self.images = [ im.view() for im in images ]
        # Convert to 0-based images as preferred by meds.
        for im in self.images:
            # Note: making the list of views above means this won't change the originals.
            im.setOrigin(0,0)
        self.box_size = self.images[0].array.shape[0]
        self.n_cutouts = len(self.images)
        if psf is not None:
            self.psf_box_size = self.images[0].array.shape[0]
        else:
            self.psf_box_size = 0

        # If weight is not provided, create something sensible.
        if weight is not None:
            self.weight = weight
        else:
            self.weight = [Image(self.box_size, self.box_size, init_value=1)]*self.n_cutouts

        # If badpix is provided, combine it into the weight image.
        if badpix is not None:
            for i in range(len(badpix)):
                mask = badpix[i].array != 0
                self.weight[i].array[mask] = 0.

        # If seg is not provided, use all 1's.
        if seg is not None:
            self.seg = seg
        else:
            self.seg = [ImageI(self.box_size, self.box_size, init_value=1)]*self.n_cutouts

        # If wcs is not provided, get it from the images.
        if wcs is not None:
            self.wcs = wcs
        else:
            self.wcs = [ im.wcs.affine(image_pos=im.true_center) for im in self.images ]

        # Normally you would supply cutout_row/cutout_col, since we can't usually
        # assume objects are centered on the stamp. If not supplied, set them to
        # the wcs origin (here that is the center of the stamp).
        if cutout_row is not None:
            self.cutout_row = cutout_row
        else:
            self.cutout_row = [ w.origin.y for w in self.wcs ]
        if cutout_col is not None:
            self.cutout_col = cutout_col
        else:
            self.cutout_col = [ w.origin.x for w in self.wcs ]

        # psf is not required, so leave it as None if not provided.
        self.psf = psf


def WriteMEDS(obj_list, file_name, clobber=True):
    """
    Writes a MEDS file from a list of `MultiExposureObject` instances.

    Parameters:
       obj_list:    List of `MultiExposureObject` instances
       file_name:   Name of meds file to be written
       clobber:     Setting ``clobber=True`` when ``file_name`` is given will silently overwrite
                    existing files. [default True]
    """

    from .._pyfits import pyfits

    # initialise the catalog
    cat = {}
    cat['id'] = []
    cat['box_size'] = []
    cat['ra'] = []
    cat['dec'] = []
    cat['ncutout'] = []
    cat['start_row'] = []
    cat['dudrow'] = []
    cat['dudcol'] = []
    cat['dvdrow'] = []
    cat['dvdcol'] = []
    cat['orig_start_row'] = []
    cat['orig_start_col'] = []
    cat['cutout_row'] = []
    cat['cutout_col'] = []
    cat['psf_box_size'] = []
    cat['psf_start_row'] = []

    # initialise the image vectors
    vec = {}
    vec['image'] = []
    vec['seg'] = []
    vec['weight'] = []
    vec['psf'] = []

    # initialise the image vector index
    n_vec = 0
    psf_n_vec = 0

    # get number of objects
    n_obj = len(obj_list)

    # loop over objects
    for obj in obj_list:

        # initialise the start indices for each image
        start_rows = np.ones(MAX_NCUTOUTS)*EMPTY_START_INDEX
        psf_start_rows = np.ones(MAX_NCUTOUTS)*EMPTY_START_INDEX
        dudrow = np.ones(MAX_NCUTOUTS)*EMPTY_JAC_diag
        dudcol = np.ones(MAX_NCUTOUTS)*EMPTY_JAC_offdiag
        dvdrow = np.ones(MAX_NCUTOUTS)*EMPTY_JAC_offdiag
        dvdcol = np.ones(MAX_NCUTOUTS)*EMPTY_JAC_diag
        cutout_row   = np.ones(MAX_NCUTOUTS)*EMPTY_SHIFT
        cutout_col   = np.ones(MAX_NCUTOUTS)*EMPTY_SHIFT
        # get the number of cutouts (exposures)
        n_cutout = obj.n_cutouts

        # append the catalog for this object
        cat['id'].append(obj.id)
        cat['box_size'].append(obj.box_size)
        # TODO: If the config defines a world position, get the right ra, dec here.
        cat['ra'].append(0.)
        cat['dec'].append(0.)
        cat['ncutout'].append(n_cutout)
        cat['psf_box_size'].append(obj.psf_box_size)

        # loop over cutouts
        for i in range(n_cutout):

            # assign the start row to the end of image vector
            start_rows[i] = n_vec
            psf_start_rows[i] = psf_n_vec
            # update n_vec to point to the end of image vector
            n_vec += len(obj.images[i].array.flatten())
            if obj.psf is not None:
                psf_n_vec += len(obj.psf[i].array.flatten())

            # append the image vectors
            vec['image'].append(obj.images[i].array.flatten())
            vec['seg'].append(obj.seg[i].array.flatten())
            vec['weight'].append(obj.weight[i].array.flatten())
            if obj.psf is not None:
                vec['psf'].append(obj.psf[i].array.flatten())

            # append cutout_row/col
            cutout_row[i] = obj.cutout_row[i]
            cutout_col[i] = obj.cutout_col[i]

            # append the Jacobian
            # col == x
            # row == y
            dudcol[i] = obj.wcs[i].dudx
            dudrow[i] = obj.wcs[i].dudy
            dvdcol[i] = obj.wcs[i].dvdx
            dvdrow[i] = obj.wcs[i].dvdy

            # check if we are running out of memory
            if sys.getsizeof(vec,0) > MAX_MEMORY:  # pragma: no cover
                raise GalSimError(
                    "Running out of memory > %1.0fGB - you can increase the limit by changing "
                    "galsim.des_meds.MAX_MEMORY"%(MAX_MEMORY/1.e9))

        # update the start rows fields in the catalog
        cat['start_row'].append(start_rows)
        cat['psf_start_row'].append(psf_start_rows)

        # add cutout_row/col
        cat['cutout_row'].append(cutout_row)
        cat['cutout_col'].append(cutout_col)

        # add lists of Jacobians
        cat['dudrow'].append(dudrow)
        cat['dudcol'].append(dudcol)
        cat['dvdrow'].append(dvdrow)
        cat['dvdcol'].append(dvdcol)

    # concatenate list to one big vector
    vec['image'] = np.concatenate(vec['image'])
    vec['seg'] = np.concatenate(vec['seg'])
    vec['weight'] = np.concatenate(vec['weight'])
    if obj.psf is not None:
        vec['psf'] = np.concatenate(vec['psf'])

    # get the primary HDU
    primary = pyfits.PrimaryHDU()

    # second hdu is the object_data
    # cf. https://github.com/esheldon/meds/wiki/MEDS-Format
    cols = []
    cols.append( pyfits.Column(name='id',             format='K', array=cat['id']       ) )
    cols.append( pyfits.Column(name='number',         format='K', array=cat['id']       ) )
    cols.append( pyfits.Column(name='ra',             format='D', array=cat['ra']       ) )
    cols.append( pyfits.Column(name='dec',            format='D', array=cat['dec']      ) )
    cols.append( pyfits.Column(name='box_size',       format='K', array=cat['box_size'] ) )
    cols.append( pyfits.Column(name='ncutout',        format='K', array=cat['ncutout']  ) )
    cols.append( pyfits.Column(name='file_id',        format='%dK' % MAX_NCUTOUTS,
                               array=[1]*n_obj) )
    cols.append( pyfits.Column(name='start_row',      format='%dK' % MAX_NCUTOUTS,
                               array=np.array(cat['start_row'])) )
    cols.append( pyfits.Column(name='orig_row',       format='%dD' % MAX_NCUTOUTS,
                               array=[[0]*MAX_NCUTOUTS]*n_obj     ) )
    cols.append( pyfits.Column(name='orig_col',       format='%dD' % MAX_NCUTOUTS,
                               array=[[0]*MAX_NCUTOUTS]*n_obj     ) )
    cols.append( pyfits.Column(name='orig_start_row', format='%dK' % MAX_NCUTOUTS,
                               array=[[0]*MAX_NCUTOUTS]*n_obj     ) )
    cols.append( pyfits.Column(name='orig_start_col', format='%dK' % MAX_NCUTOUTS,
                               array=[[0]*MAX_NCUTOUTS]*n_obj     ) )
    cols.append( pyfits.Column(name='cutout_row',     format='%dD' % MAX_NCUTOUTS,
                               array=np.array(cat['cutout_row'])     ) )
    cols.append( pyfits.Column(name='cutout_col',     format='%dD' % MAX_NCUTOUTS,
                               array=np.array(cat['cutout_col'])     ) )
    cols.append( pyfits.Column(name='dudrow',         format='%dD' % MAX_NCUTOUTS,
                               array=np.array(cat['dudrow'])   ) )
    cols.append( pyfits.Column(name='dudcol',         format='%dD' % MAX_NCUTOUTS,
                               array=np.array(cat['dudcol'])   ) )
    cols.append( pyfits.Column(name='dvdrow',         format='%dD' % MAX_NCUTOUTS,
                               array=np.array(cat['dvdrow'])   ) )
    cols.append( pyfits.Column(name='dvdcol',         format='%dD' % MAX_NCUTOUTS,
                               array=np.array(cat['dvdcol'])   ) )
    cols.append( pyfits.Column(name='psf_box_size',   format='K', array=cat['psf_box_size'] ) )
    cols.append( pyfits.Column(name='psf_start_row',  format='%dK' % MAX_NCUTOUTS,
                               array=np.array(cat['psf_start_row'])) )


    # Depending on the version of pyfits, one of these should work:
    try:
        object_data = pyfits.BinTableHDU.from_columns(cols)
        object_data.name = 'object_data'
    except AttributeError:  # pragma: no cover
        object_data = pyfits.new_table(pyfits.ColDefs(cols))
        object_data.update_ext_name('object_data')

    # third hdu is image_info
    cols = []
    cols.append( pyfits.Column(name='image_path',  format='A256',   array=['generated_by_galsim'] ))
    cols.append( pyfits.Column(name='image_ext',   format='I',      array=[0]                     ))
    cols.append( pyfits.Column(name='weight_path', format='A256',   array=['generated_by_galsim'] ))
    cols.append( pyfits.Column(name='weight_ext',  format='I',      array=[0]                     ))
    cols.append( pyfits.Column(name='seg_path',    format='A256',   array=['generated_by_galsim'] ))
    cols.append( pyfits.Column(name='seg_ext',     format='I',      array=[0]                     ))
    cols.append( pyfits.Column(name='bmask_path',  format='A256',   array=['generated_by_galsim'] ))
    cols.append( pyfits.Column(name='bmask_ext',   format='I',      array=[0]                     ))
    cols.append( pyfits.Column(name='bkg_path',    format='A256',   array=['generated_by_galsim'] ))
    cols.append( pyfits.Column(name='bkg_ext',     format='I',      array=[0]                     ))
    cols.append( pyfits.Column(name='image_id',    format='K',      array=[-1]                    ))
    cols.append( pyfits.Column(name='image_flags', format='K',      array=[-1]                    ))
    cols.append( pyfits.Column(name='magzp',       format='E',      array=[30.]                   ))
    cols.append( pyfits.Column(name='scale',       format='E',      array=[1.]                    ))
    # TODO: Not sure if this is right!
    cols.append( pyfits.Column(name='position_offset', format='D',  array=[0.]                    ))
    try:
        image_info = pyfits.BinTableHDU.from_columns(cols)
        image_info.name = 'image_info'
    except AttributeError:  # pragma: no cover
        image_info = pyfits.new_table(pyfits.ColDefs(cols))
        image_info.update_ext_name('image_info')

    # fourth hdu is metadata
    # default values?
    cols = []
    cols.append( pyfits.Column(name='magzp_ref',     format='E',    array=[30.]                   ))
    cols.append( pyfits.Column(name='DESDATA',       format='A256', array=['generated_by_galsim'] ))
    cols.append( pyfits.Column(name='cat_file',      format='A256', array=['generated_by_galsim'] ))
    cols.append( pyfits.Column(name='coadd_image_id',format='A256', array=['generated_by_galsim'] ))
    cols.append( pyfits.Column(name='coadd_file',    format='A256', array=['generated_by_galsim'] ))
    cols.append( pyfits.Column(name='coadd_hdu',     format='K',    array=[9999]                  ))
    cols.append( pyfits.Column(name='coadd_seg_hdu', format='K',    array=[9999]                  ))
    cols.append( pyfits.Column(name='coadd_srclist', format='A256', array=['generated_by_galsim'] ))
    cols.append( pyfits.Column(name='coadd_wt_hdu',  format='K',    array=[9999]                  ))
    cols.append( pyfits.Column(name='coaddcat_file', format='A256', array=['generated_by_galsim'] ))
    cols.append( pyfits.Column(name='coaddseg_file', format='A256', array=['generated_by_galsim'] ))
    cols.append( pyfits.Column(name='cutout_file',   format='A256', array=['generated_by_galsim'] ))
    cols.append( pyfits.Column(name='max_boxsize',   format='A3',   array=['-1']                  ))
    cols.append( pyfits.Column(name='medsconf',      format='A3',   array=['x']                   ))
    cols.append( pyfits.Column(name='min_boxsize',   format='A2',   array=['-1']                  ))
    cols.append( pyfits.Column(name='se_badpix_hdu', format='K',    array=[9999]                  ))
    cols.append( pyfits.Column(name='se_hdu',        format='K',    array=[9999]                  ))
    cols.append( pyfits.Column(name='se_wt_hdu',     format='K',    array=[9999]                  ))
    cols.append( pyfits.Column(name='seg_hdu',       format='K',    array=[9999]                  ))
    cols.append( pyfits.Column(name='psf_hdu',       format='K',    array=[9999]                  ))
    cols.append( pyfits.Column(name='sky_hdu',       format='K',    array=[9999]                  ))
    cols.append( pyfits.Column(name='fake_coadd_seg',format='K',    array=[9999]                  ))
    try:
        metadata = pyfits.BinTableHDU.from_columns(cols)
        metadata.name = 'metadata'
    except AttributeError:  # pragma: no cover
        metadata = pyfits.new_table(pyfits.ColDefs(cols))
        metadata.update_ext_name('metadata')

    # rest of HDUs are image vectors
    image_cutouts   = pyfits.ImageHDU( vec['image'] , name='image_cutouts')
    weight_cutouts  = pyfits.ImageHDU( vec['weight'], name='weight_cutouts')
    seg_cutouts     = pyfits.ImageHDU( vec['seg']   , name='seg_cutouts')

    hdu_list = [
        primary,
        object_data,
        image_info,
        metadata,
        image_cutouts,
        weight_cutouts,
        seg_cutouts,
    ]

    if obj.psf is not None:
        psf_cutouts     = pyfits.ImageHDU( vec['psf'], name='psf')
        hdu_list.append(psf_cutouts)

    writeFile(file_name, pyfits.HDUList(hdu_list))


# Make the class that will
class MEDSBuilder(OutputBuilder):
    """This class lets you use the `MultiExposureObject` very simply as a special ``output``
    type when using config processing.

    It requires the use of ``galsim.des`` in the config files ``modules`` section.
    """

    def buildImages(self, config, base, file_num, image_num, obj_num, ignore, logger):
        """
        Build a meds file as specified in config.

        Parameters:
           config:      The configuration dict for the output field.
           base:        The base configuration dict.
           file_num:    The current file_num.
           image_num:   The current image_num.
           obj_num:     The current obj_num.
           ignore:      A list of parameters that are allowed to be in config that we can ignore
                        here.
           logger:      If given, a logger object to log progress.

        Returns:
           A list of MultiExposureObjects.

        """
        import time
        t1 = time.time()

        if base.get('image',{}).get('type', 'Single') != 'Single':
            raise GalSimConfigError(
                "MEDS files are not compatible with image type %s."%base['image']['type'])

        req = { 'nobjects' : int , 'nstamps_per_object' : int }
        opt  = {'first_id' : int }
        ignore += [ 'file_name', 'dir', 'nfiles' ]
        params = GetAllParams(config,base,ignore=ignore,req=req, opt=opt)[0]
        first_id = params.get('first_id', obj_num)

        nobjects = params['nobjects']
        nstamps_per_object = params['nstamps_per_object']
        ntot = nobjects * nstamps_per_object

        main_images = BuildImages(ntot, base, image_num=image_num,  obj_num=obj_num, logger=logger)

        # grab list of offsets for cutout_row/cutout_col.
        offsets = GetFinalExtraOutput('meds_get_offset', base, logger)
        # cutout_row/col is the stamp center (**with the center of the first pixel
        # being (0,0)**) + offset
        centers = [0.5*im.array.shape[0]-0.5 for im in main_images]
        cutout_rows = [c+offset.y for c,offset in zip(centers,offsets)]
        cutout_cols = [c+offset.x for c,offset in zip(centers,offsets)]

        weight_images = GetFinalExtraOutput('weight', base, logger)
        if 'badpix' in config:
            badpix_images = GetFinalExtraOutput('badpix', base, logger)
        else:
            badpix_images = None
        psf_images = GetFinalExtraOutput('psf', base, logger)

        obj_list = []
        for i in range(nobjects):
            k1 = i*nstamps_per_object
            k2 = (i+1)*nstamps_per_object
            if badpix_images is not None:
                bpk = badpix_images[k1:k2]
            else:
                bpk = None
            obj = MultiExposureObject(images = main_images[k1:k2],
                                      weight = weight_images[k1:k2],
                                      badpix = bpk,
                                      psf = psf_images[k1:k2],
                                      id = first_id + i,
                                      cutout_row = cutout_rows[k1:k2],
                                      cutout_col = cutout_cols[k1:k2])
            obj_list.append(obj)

        return obj_list

    def writeFile(self, data, file_name, config, base, logger):
        """Write the data to a file.  In this case a MEDS file.

        Parameters:
            data:           The data to write.  Here, this is a list of `MultiExposureObject`.
            file_name:      The file_name to write to.
            config:         The configuration dict for the output field.
            base:           The base configuration dict.
            logger:         If given, a logger object to log progress.
        """
        WriteMEDS(data, file_name)

    def getNImages(self, config, base, file_num, logger=None):
        # This gets called before starting work on the file, so we can use this opportunity
        # to make sure that weight and psf processing are turned on.
        # We just add these as empty dicts, so there is no hdu or file_name parameter, which
        # means they won't actually output anything, but the images will be built, so we can use
        # them in BuildMEDS above.
        if 'weight' not in config:
            config['weight'] = {}
        if 'psf' not in config:
            config['psf'] = {}

        # We use an extra output type to get the offsets of objects in stamps.
        # It doesn't need any parameters.  Just getting its name into the config dict is sufficient.
        if 'meds_get_offset' not in config:
            config['meds_get_offset']={}

        nobjects = ParseValue(config,'nobjects',base,int)[0]
        nstamps_per_object = ParseValue(config,'nstamps_per_object',base,int)[0]

        ntot = nobjects * nstamps_per_object
        return ntot

# This extra output type simply saves the values of the image offsets when an
# object is drawn into the stamp.
class OffsetBuilder(ExtraOutputBuilder):
    """This "extra" output builder saves the stamp offset values for later use.

    It is used as a ``meds_get_offset`` field in the ``output`` section.
    """

    # The function to call at the end of building each stamp
    def processStamp(self, obj_num, config, base, logger):
        offset = base['stamp_offset']
        stamp = base['stamp']
        if 'offset' in stamp:
            offset += GetCurrentValue('offset', base['stamp'], PositionD, base)
        self.scratch[obj_num] = offset

    # The function to call at the end of building each file to finalize the truth catalog
    def finalize(self, config, base, main_data, logger):
        offsets_list = []
        obj_nums = sorted(self.scratch.keys())
        for obj_num in obj_nums:
            offsets_list.append(self.scratch[obj_num])
        return offsets_list

# Register these
RegisterOutputType('MEDS', MEDSBuilder())
RegisterExtraOutput('meds_get_offset', OffsetBuilder())

