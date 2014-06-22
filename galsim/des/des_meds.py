# Copyright (c) 2012-2014 by the GalSim developers team on GitHub
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
"""@file des_meds.py  Module for generating DES Multi-Epoch Data Structures (MEDS) in GalSim.

This module defines the MultiExposureObject class for representing multiple exposure data for a 
single object.  The write_meds() function can be used to write a list of MultiExposureObject
instances to a single MEDS file.

Importing this module also adds these data structures to the config framework, so that MEDS file 
output can subsequently be simulated directly using a config file.
"""

import numpy
import galsim

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

    Initialization
    --------------

    @param images       List of images of the object (GalSim Images).
    @param weights      List of weight maps (GalSim Images). [default: None]
    @param badpix       List of bad pixel masks (GalSim Images). [default: None]
    @param segs         List of segmentation maps (GalSim Images). [default: None]
    @param wcs          List of WCS transformations (GalSim AffineTransforms). [default: None]
    @param id           Galaxy id. [default: 0]

    Attributes
    ----------

    self.images         List of images of the object (GalSim Images).
    self.weights        List of weight maps (GalSim Images).
    self.segs           List of segmentation masks (GalSim Images).
    self.wcs            List of WCS transformations (GalSim AffineTransforms).
    self.n_cutouts      Number of exposures.
    self.box_size       Size of each exposure image.

    Module level variables
    ----------------------

    Images, weights and segs have to be square numpy arrays with size in
    BOX_SIZES = [32,48,64,96,128,196,256].
    Number of exposures for all lists (images,weights,segs,wcs) have to be the same and smaller 
    than MAX_NCUTOUTS (default 11).
    """

    def __init__(self, images, weights=None, badpix=None, segs=None, wcs=None, id=0):

        # assign the ID
        self.id = id

        # check if images is a list
        if not isinstance(images,list):
            raise TypeError('images should be a list')

        # get number of cutouts from image list
        self.images = images
        # get box size from the first image
        self.box_size = self.images[0].array.shape[0]
        self.n_cutouts = len(self.images)

        # see if there are cutouts
        if self.n_cutouts < 1:
            raise ValueError('no cutouts in this object')

        # check if the box size is correct
        if self.box_size not in BOX_SIZES:
            # raise ValueError('box size should be in  [32,48,64,96,128,196,256], is %d' % box_size)
            raise ValueError( 'box size should be in '+str(BOX_SIZES)+', is '+str(self.box_size) )

        # check if weights, segs and wcs were supplied. If not, create sensible values.
        if weights != None:
            self.weights = weights
        else:
            self.weights = [galsim.Image(self.box_size, self.box_size, init_value=1)]*self.n_cutouts

        # check segmaps
        if segs != None:
            self.segs = segs
            # I think eventually, the meds files will have some more sophisticated pixel map
            # where the segmentation info and bad pixel info are separately coded.
            # However, for now, we just set to 0 any bad pixels.
            # (Not that GalSim has any mechanism yet for generating bad pixels, so this is
            # usually a null op, but it's in place for when there is something to do.)
            if badpix != None:
                if len(self.segs) != len(badpix):
                    raise ValueError("segs and badpix are different lengths")
                for i in range(len(self.segs)):
                    if (self.segs[i].array.shape != badpix[i].array.shape):
                        raise ValueError("segs[%d] and badpix[%d] have different shapes."%(i,i))
                    self.segs[i].array[:,:] &= (badpix[i].array == 0)
        elif badpix != None:
            self.segs = badpix
            # Flip the sense of the bits 0 -> 1, other -> 0
            # Again, this might need to become more sophisticated at some point...
            for i in range(len(self.segs)):
                self.segs[i].array[:,:] = (badpix[i].array == 0)
        else:
            self.segs = [galsim.ImageI(self.box_size, self.box_size, init_value=1)]*self.n_cutouts

        # check wcs
        if wcs != None:
            self.wcs = wcs
        else:
            # Get the wcs from the images.  Probably just the pixel scale.
            self.wcs = [ im.wcs.jacobian().setOrigin(im.trueCenter()) for im in self.images ]

         # check if weights,segs,jacks are lists
        if not isinstance(self.weights,list):
            raise TypeError('weights should be a list')
        if not isinstance(self.segs,list):
            raise TypeError('segs should be a list')
        if not isinstance(self.wcs,list):
            raise TypeError('wcs should be a list')


        # loop through the images and check if they are of the same size
        for extname in ('images','weights','segs'):

            # get the class field
            ext = eval('self.' + extname )

            # loop through exposures
            for icutout,cutout in enumerate(ext):

                # get the sizes of array
                nx=cutout.array.shape[0]
                ny=cutout.array.shape[1]

                # x and y size should be the same
                if nx != ny:
                    raise ValueError('%s should be square and is %d x %d' % (extname,nx,ny))

                # check if box size is correct
                if nx != self.box_size:
                    raise ValueError('%s object %d has size %d and should be %d' % 
                            ( extname,icutout,nx,self.box_size ) )

        # see if the number of Jacobians is right
        if len(self.wcs) != self.n_cutouts:
            raise ValueError('number of Jacobians is %d is not equal to number of cutouts %d'%
                    ( len(self.wcs),self.n_cutouts ) )

        # check each Jacobian
        for jac in self.wcs:
            # should ba an AffineTransform instance
            if not isinstance(jac, galsim.AffineTransform):
                raise TypeError('wcs list should contain AffineTransform objects')
            

def write_meds(file_name, obj_list, clobber=True):
    """
    @brief Writes the galaxy, weights, segmaps images to a MEDS file.

    Arguments:
    ----------
    @param file_name:    Name of meds file to be written
    @param obj_list:     List of MultiExposureObjects
    @param clobber       Setting `clobber=True` when `file_name` is given will silently overwrite 
                         existing files. (Default `clobber = True`.)
    """

    import numpy
    import sys
    from galsim import pyfits

    # initialise the catalog
    cat = {}
    cat['ncutout'] = []
    cat['box_size'] = []
    cat['start_row'] = []
    cat['id'] = []
    cat['dudrow'] = []
    cat['dudcol'] = []
    cat['dvdrow'] = []
    cat['dvdcol'] = []
    cat['row0'] = []
    cat['col0'] = []

    # initialise the image vectors
    vec = {}
    vec['image'] = []
    vec['seg'] = []
    vec['weight'] = []

    # initialise the image vector index
    n_vec = 0
    
    # get number of objects
    n_obj = len(obj_list)

    # loop over objects
    for obj in obj_list:

        # initialise the start indices for each image
        start_rows = numpy.ones(MAX_NCUTOUTS)*EMPTY_START_INDEX
        dudrow = numpy.ones(MAX_NCUTOUTS)*EMPTY_JAC_diag 
        dudcol = numpy.ones(MAX_NCUTOUTS)*EMPTY_JAC_offdiag
        dvdrow = numpy.ones(MAX_NCUTOUTS)*EMPTY_JAC_offdiag
        dvdcol = numpy.ones(MAX_NCUTOUTS)*EMPTY_JAC_diag
        row0   = numpy.ones(MAX_NCUTOUTS)*EMPTY_SHIFT
        col0   = numpy.ones(MAX_NCUTOUTS)*EMPTY_SHIFT

        # get the number of cutouts (exposures)
        n_cutout = obj.n_cutouts
        
        # append the catalog for this object
        cat['ncutout'].append(n_cutout)
        cat['box_size'].append(obj.box_size)
        cat['id'].append(obj.id)

        # loop over cutouts
        for i in range(n_cutout):

            # assign the start row to the end of image vector
            start_rows[i] = n_vec
            # update n_vec to point to the end of image vector
            n_vec += len(obj.images[i].array.flatten()) 

            # append the image vectors
            vec['image'].append(obj.images[i].array.flatten())
            vec['seg'].append(obj.segs[i].array.flatten())
            vec['weight'].append(obj.weights[i].array.flatten())


            # append the Jacobian
            dudrow[i] = obj.wcs[i].dudx
            dudcol[i] = obj.wcs[i].dudy
            dvdrow[i] = obj.wcs[i].dvdx
            dvdcol[i] = obj.wcs[i].dvdy
            row0[i]   = obj.wcs[i].origin.x
            col0[i]   = obj.wcs[i].origin.y

            # check if we are running out of memory
            if sys.getsizeof(vec) > MAX_MEMORY:
                raise MemoryError(
                    'Running out of memory > %1.0fGB '%MAX_MEMORY/1.e9 +
                    '- you can increase the limit by changing MAX_MEMORY')

        # update the start rows fields in the catalog
        cat['start_row'].append(start_rows)

        # add lists of Jacobians
        cat['dudrow'].append(dudrow)
        cat['dudcol'].append(dudcol)
        cat['dvdrow'].append(dvdrow)
        cat['dvdcol'].append(dvdcol)
        cat['row0'].append(row0)
        cat['col0'].append(col0)

    # concatenate list to one big vector
    vec['image'] = numpy.concatenate(vec['image'])
    vec['seg'] = numpy.concatenate(vec['seg'])
    vec['weight'] = numpy.concatenate(vec['weight'])

    # get the primary HDU
    primary = pyfits.PrimaryHDU()

    # second hdu is the object_data
    cols = []
    cols.append( pyfits.Column(name='ncutout', format='i4', array=cat['ncutout'] ) )
    cols.append( pyfits.Column(name='id', format='i4', array=cat['id'] ) )
    cols.append( pyfits.Column(name='box_size', format='i4', array=cat['box_size'] ) )
    cols.append( pyfits.Column(name='file_id', format='i4', array=[1]*n_obj) )
    cols.append( pyfits.Column(name='start_row', format='%di4' % MAX_NCUTOUTS,
                               array=numpy.array(cat['start_row'])) )
    cols.append( pyfits.Column(name='orig_row', format='f8', array=[1]*n_obj) )
    cols.append( pyfits.Column(name='orig_col', format='f8', array=[1]*n_obj) )
    cols.append( pyfits.Column(name='orig_start_row', format='i4', array=[1]*n_obj) )
    cols.append( pyfits.Column(name='orig_start_col', format='i4', array=[1]*n_obj) )
    cols.append( pyfits.Column(name='dudrow', format='%df8'% MAX_NCUTOUTS,
                               array=numpy.array(cat['dudrow']) ) )
    cols.append( pyfits.Column(name='dudcol', format='%df8'% MAX_NCUTOUTS,
                               array=numpy.array(cat['dudcol']) ) )
    cols.append( pyfits.Column(name='dvdrow', format='%df8'% MAX_NCUTOUTS,
                               array=numpy.array(cat['dvdrow']) ) )
    cols.append( pyfits.Column(name='dvdcol', format='%df8'% MAX_NCUTOUTS,
                               array=numpy.array(cat['dvdcol']) ) )
    cols.append( pyfits.Column(name='cutout_row', format='%df8'% MAX_NCUTOUTS,
                               array=numpy.array(cat['row0']) ) )
    cols.append( pyfits.Column(name='cutout_col', format='%df8'% MAX_NCUTOUTS,
                               array=numpy.array(cat['col0']) ) )


    object_data = pyfits.new_table(pyfits.ColDefs(cols))
    object_data.update_ext_name('object_data')

    # third hdu is image_info
    cols = []
    cols.append( pyfits.Column(name='image_path', format='A256',   array=['generated_by_galsim'] ) )
    cols.append( pyfits.Column(name='sky_path',   format='A256',   array=['generated_by_galsim'] ) )
    cols.append( pyfits.Column(name='seg_path',   format='A256',   array=['generated_by_galsim'] ) )
    image_info = pyfits.new_table(pyfits.ColDefs(cols))
    image_info.update_ext_name('image_info')

    # fourth hdu is metadata
    cols = []
    cols.append( pyfits.Column(name='cat_file',      format='A256', array=['generated_by_galsim'] ))
    cols.append( pyfits.Column(name='coadd_file',    format='A256', array=['generated_by_galsim'] ))
    cols.append( pyfits.Column(name='coadd_hdu',     format='A1',   array=['x']                   ))
    cols.append( pyfits.Column(name='coadd_seg_hdu', format='A1',   array=['x']                   ))
    cols.append( pyfits.Column(name='coadd_srclist', format='A256', array=['generated_by_galsim'] ))
    cols.append( pyfits.Column(name='coadd_wt_hdu',  format='A1',   array=['x']                   ))
    cols.append( pyfits.Column(name='coaddcat_file', format='A256', array=['generated_by_galsim'] ))
    cols.append( pyfits.Column(name='coaddseg_file', format='A256', array=['generated_by_galsim'] ))
    cols.append( pyfits.Column(name='cutout_file',   format='A256', array=['generated_by_galsim'] ))
    cols.append( pyfits.Column(name='max_boxsize',   format='A3',   array=['x']                   ))
    cols.append( pyfits.Column(name='medsconf',      format='A3',   array=['x']                   ))
    cols.append( pyfits.Column(name='min_boxsize',   format='A2',   array=['x']                   ))
    cols.append( pyfits.Column(name='se_badpix_hdu', format='A1',   array=['x']                   ))
    cols.append( pyfits.Column(name='se_hdu',        format='A1',   array=['x']                   ))
    cols.append( pyfits.Column(name='se_wt_hdu',     format='A1',   array=['x']                   ))
    cols.append( pyfits.Column(name='seg_hdu',       format='A1',   array=['x']                   ))
    cols.append( pyfits.Column(name='sky_hdu',       format='A1',   array=['x']                   ))
    metadata = pyfits.new_table(pyfits.ColDefs(cols))
    metadata.update_ext_name('metadata')

    # rest of HDUs are image vectors
    image_cutouts   = pyfits.ImageHDU( vec['image'] , name='image_cutouts'  )
    weight_cutouts  = pyfits.ImageHDU( vec['weight'], name='weight_cutouts' )
    seg_cutouts     = pyfits.ImageHDU( vec['seg']   , name='seg_cutouts'    )

    # write all
    hdu_list = pyfits.HDUList([
        primary,
        object_data,
        image_info,
        metadata,
        image_cutouts, 
        weight_cutouts,
        seg_cutouts
    ])
    hdu_list.writeto(file_name,clobber=clobber)


# Now add this to the config framework.
import galsim.config

# Make this a valid output type:
galsim.config.process.valid_output_types['des_meds'] = (
    'galsim.des.BuildMEDS',      # Function that builds the objects using config
    'galsim.des.GetNObjForMEDS', # Function that calculates the number of objects
    True,   # Takes nproc argument
    False,  # Takes *_file_name arguments for psf, weight, badpix
    False)  # Takes *_hdu arguments for psf, weight, badpix

def BuildMEDS(file_name, config, nproc=1, logger=None, file_num=0, image_num=0, obj_num=0):
    """
    Build a meds file as specified in config.

    @param file_name         The name of the output file.
    @param config            A configuration dict.
    @param nproc             How many processes to use. [default: 1]
    @param logger            If given, a logger object to log progress. [default: None]
    @param file_num          If given, the current file_num. [default: 0]
    @param image_num         If given, the current image_num. [default: 0]
    @param obj_num           If given, the current obj_num. [default: 0]

    @returns the time taken to build file
    """
    import time
    t1 = time.time()

    config['seq_index'] = file_num
    config['file_num'] = file_num

    ignore = [ 'file_name', 'dir', 'nfiles', 'psf', 'weight', 'badpix', 'nproc' ]
    req = { 'nobjects' : int , 'nstamps_per_object' : int }
    params = galsim.config.GetAllParams(config['output'],'output',config,ignore=ignore,req=req)[0]

    nobjects = params['nobjects']
    nstamps_per_object = params['nstamps_per_object']
    ntot = nobjects * nstamps_per_object

    all_images = galsim.config.BuildImages(
        ntot, config=config, nproc=nproc, logger=logger, obj_num=obj_num,
        make_psf_image=False, make_weight_image=True, make_badpix_image=True)

    main_images = all_images[0]
    weight_images = all_images[2]
    badpix_images = all_images[3]

    obj_list = []
    for i in range(nobjects):
        k1 = i*nstamps_per_object
        k2 = (i+1)*nstamps_per_object
        obj = MultiExposureObject(images = main_images[k1:k2], 
                                  weights = weight_images[k1:k2],
                                  badpix = badpix_images[k1:k2])
        obj_list.append(obj)

    write_meds(file_name, obj_list)

    t2 = time.time()
    return t2-t1

def GetNObjForMEDS(config, file_num, image_num):
    ignore = [ 'file_name', 'dir', 'nfiles', 'psf', 'weight', 'badpix', 'nproc' ]
    req = { 'nobjects' : int , 'nstamps_per_object' : int }
    params = galsim.config.GetAllParams(config['output'],'output',config,ignore=ignore,req=req)[0]
    config['seq_index'] = file_num

    if 'image' in config and 'type' in config['image']:
        image_type = config['image']['type']
        if image_type != 'Single':
            raise AttibuteError("MEDS files are not compatible with image type %s."%image_type)

    nobjects = params['nobjects']
    nstamps_per_object = params['nstamps_per_object']
    ntot = nobjects * nstamps_per_object

    # nobj is a list of nobj per image.
    # The MEDS file is considered to only have a single image, so the list has only 1 element.
    nobj = [ ntot ]
    return nobj

