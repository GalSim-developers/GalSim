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

"""@file des_meds.py  Module for generating DES Multi-Epoch Data Structures (MEDS) in GalSim.

This module defines the `MultiExposureObject` class for representing multiple exposure data for a single object, and the `WCSTransform` class used to store general, locally-linearized WCS information per exposure.  The `write_meds` function
can be used to write a list of `MultiExposureObject` instances to a single MEDS file.

Importing this module also adds these data structures to the config framework, so that MEDS file output can subsequently be simulated directly using a config file.
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
EMPTY_JAC = 999

class WCSTransform(object):
    """
    Very simple class which holds a WCS transformation, including a Jacobian and a shifted centroid.
    The (u,v) coordinate plane is the local tangent plane at the location of the galaxy with 
    North = +v, West = +u. The 'u' coordinace scales as cos(DEC) * dRA.
    Here, 'row' is a row in the image; 'col' is a column in the image. 
    The row/col notation is the same as arrays indexed in C and python, arr[row,col].

    Available fields:
    self.dudrow  -  element 1,1 of the Jacobian matrix
    self.dudcol  -  element 1,2 of the Jacobian matrix
    self.dvdrow  -  element 2,1 of the Jacobian matrix
    self.dvdcol  -  element 2,2 of the Jacobian matrix
    self.row0    -  horizontal position of the centroid as given by SExtractor, in pixel coordinates
    self.col0    -  vertical position of the centroid as given by SExtractor, in pixel coordinates 
    """

    def __init__(self, dudrow, dudcol, dvdrow, dvdcol, row0, col0):
    
        self.dudrow =      dudrow
        self.dudcol =      dudcol
        self.dvdrow =      dvdrow
        self.dvdcol =      dvdcol
        self.row0 =        row0
        self.col0 =        col0

class MultiExposureObject(object):
    """
    A class containing exposures for single object, along with other information.

    Available fields:
        self.images             list of images of the object (GalSim images)
        self.weights            list of weight maps (GalSim images)
        self.segs               list of segmentation masks (GalSim images)
        self.wcstrans           list of WCS transformation (WCSTransform objects)
        self.n_cutouts          number of exposures
        self.box_size           size of each exposure image

    Constructor parameters:
    @param images               list of images of the object (GalSim images)
    @param weights              list of weight maps (GalSim images)
    @param badpix               list of bad pixel masks (GalSim images)
    @param segs                 list of segmentation maps (GalSim images)
    @param wcstrans             list of WCS transformation (WCSTransform objects)
    @param id                   galaxy id

    Images, weights and segs have to be square numpy arrays with size in
    BOX_SIZES = [32,48,64,96,128,196,256].
    Number of exposures for all lists (images,weights,segs,wcstrans) have to be the same and smaller 
    than MAX_NCUTOUTS (default 11).
    """

    def __init__(self, images, weights=None, badpix=None, segs=None, wcstrans=None, id=0):

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

        # check if weights, segs and wcstrans were supplied. If not, create sensible values.
        if weights != None:
            self.weights = weights
        else:
            self.weights = [galsim.ImageF(self.box_size, self.box_size, init_value=1)]*self.n_cutouts

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

        # check wcstrans
        if wcstrans != None:
            self.wcstrans = wcstrans
        else:
            # build jacobians that are just based on the pixel scale, set the centers
            dudrow = 1
            dudcol = 0
            dvdrow = 0
            dvdcol = 1
            # set to the center of the postage stamp
            row0 = float(self.box_size)/2.
            col0 = float(self.box_size)/2.
            self.wcstrans = [ WCSTransform(dudrow * im.scale, dudcol * im.scale,
                        dvdrow * im.scale, dvdcol * im.scale, row0, col0) for im in self.images ]

         # check if weights,segs,jacks are lists
        if not isinstance(self.weights,list):
            raise TypeError('weights should be a list')
        if not isinstance(self.segs,list):
            raise TypeError('segs should be a list')
        if not isinstance(self.wcstrans,list):
            raise TypeError('wcstrans should be a list')


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
        if len(self.wcstrans) != self.n_cutouts:
            raise ValueError('number of Jacobians is %d is not equal to number of cutouts %d'%
                    ( len(self.wcstrans),self.n_cutouts ) )

        # check each Jacobian
        for jac in self.wcstrans:
            # should ba a WCSTransform instance
            if not isinstance(jac, WCSTransform):
                raise TypeError('wcstrans list should contain WCSTransform objects')
            

def write_meds(file_name, obj_list, clobber=True):
    """
    @brief Writes the galaxy, weights, segmaps images to a MEDS file.

    Arguments:
    ----------
    @param file_name:    Name of meds file to be written
    @param obj_list:     List of MultiExposureObjects
    """

    import numpy
    import sys
    import pyfits

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
        dudrow = numpy.ones(MAX_NCUTOUTS)*EMPTY_JAC 
        dudcol = numpy.ones(MAX_NCUTOUTS)*EMPTY_JAC
        dvdrow = numpy.ones(MAX_NCUTOUTS)*EMPTY_JAC
        dvdcol = numpy.ones(MAX_NCUTOUTS)*EMPTY_JAC
        row0   = numpy.ones(MAX_NCUTOUTS)*EMPTY_JAC
        col0   = numpy.ones(MAX_NCUTOUTS)*EMPTY_JAC

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

            # append the image vectors
            # if vector not is already initialised
            if vec['image'] == []:
                vec['image'] = obj.images[i].array.flatten()
                vec['seg'] = obj.segs[i].array.flatten()
                vec['weight'] = obj.weights[i].array.flatten()
            # if vector already exists
            else:
                vec['image'] = numpy.concatenate([vec['image'], obj.images[i].array.flatten()])
                vec['seg'] = numpy.concatenate([vec['seg'], obj.segs[i].array.flatten()])
                vec['weight'] = numpy.concatenate([vec['weight'], obj.weights[i].array.flatten()])

            # append the Jacobian
            dudrow[i] = obj.wcstrans[i].dudrow
            dudcol[i] = obj.wcstrans[i].dudcol
            dvdrow[i] = obj.wcstrans[i].dvdrow
            dvdcol[i] = obj.wcstrans[i].dvdcol
            row0[i]   = obj.wcstrans[i].row0
            col0[i]   = obj.wcstrans[i].col0

            # check if we are running out of memory
            if sys.getsizeof(vec) > MAX_MEMORY:
                raise MemoryError(
                    'Running out of memory > %1.0fGB - you can increase the limit by changing MAX_MEMORY' %
                    MAX_MEMORY/1e9)

            # update n_vec to point to the end of image vector
            n_vec = len(vec['image'])


        # update the start rows fields in the catalog
        cat['start_row'].append(start_rows)

        # add lists of Jacobians
        cat['dudrow'].append(dudrow)
        cat['dudcol'].append(dudcol)
        cat['dvdrow'].append(dvdrow)
        cat['dvdcol'].append(dvdcol)
        cat['row0'].append(row0)
        cat['col0'].append(col0)

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
    cols.append( pyfits.Column(name='dudrow', format='%df8'% MAX_NCUTOUTS, array=cat['dudrow'] ) )
    cols.append( pyfits.Column(name='dudcol', format='%df8'% MAX_NCUTOUTS, array=cat['dudcol'] ) )
    cols.append( pyfits.Column(name='dvdrow', format='%df8'% MAX_NCUTOUTS, array=cat['dvdrow'] ) )
    cols.append( pyfits.Column(name='dvdcol', format='%df8'% MAX_NCUTOUTS, array=cat['dvdcol'] ) )
    cols.append( pyfits.Column(name='cutout_row', format='%df8'% MAX_NCUTOUTS, array=cat['row0'] ) )
    cols.append( pyfits.Column(name='cutout_col', format='%df8'% MAX_NCUTOUTS, array=cat['col0'] ) )


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

def BuildMEDS(file_name, config, nproc=1, logger=None, image_num=0, obj_num=0):
    """
    Build a meds file as specified in config.

    @param file_name         The name of the output file.
    @param config            A configuration dict.
    @param nproc             How many processes to use.
    @param logger            If given, a logger object to log progress.
    @param image_num         If given, the current image_num (default = 0)
    @param obj_num           If given, the current obj_num (default = 0)

    @return time      Time taken to build file
    """
    import time
    t1 = time.time()

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

