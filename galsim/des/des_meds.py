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


import numpy
import galsim

BOX_SIZES = [32,48,64,96,128,196,256]
MAX_MEMORY = 1e9
MAX_NCUTOUTS = 11
EMPTY_START_INDEX = 9999
EMPTY_JAC = 999

galsim_image_types = [galsim.ImageD,galsim.ImageF,galsim.ImageI,galsim.ImageS,
                    galsim.ImageViewD,galsim.ImageViewF,galsim.ImageViewI,
                    galsim.ImageViewS]

class MultiExposureObject(object):
    """
    A class containing exposures for single object, along with other information.

    Available fields:
        self.images             list of images of the object (GalSim images)
        self.weights            list of weight maps (GalSim images)
        self.segs               list of segmentation masks (GalSim images)
        self.jacs               list of Jacobians of transformation 
                                 row,col->ra,dec tangent plane (u,v)
        self.n_cutouts          number of exposures
        self.box_size           size of each exposure image

    Constructor parameters:
    @param images               list of images of the object (numpy arrays)
    @param weights              list of weight maps (numpy arrays)
    @param segs                 list of segmentation masks (numpy arrays)
    @param jacs                 list of Jacobians of transformation 
                                 row,col->ra,dec tangent plane (u,v) (2x2 numpy arrays)

    Images, weights and segs have to be square numpy arrays with size in 
    BOX_SIZES = [32,48,64,96,128,196,256].
    Number of exposures for all lists (images,weights,segs,jacs) have to be the same and smaller 
    than MAX_NCUTOUTS (default 11).
    """

    def __init__(self,images,weights,segs,jacs):

        if not isinstance(images,list):
            raise TypeError('images should be a list')
        if not isinstance(weights,list):
            raise TypeError('weights should be a list')
        if not isinstance(segs,list):
            raise TypeError('segs should be a list')
        if not isinstance(jacs,list):
            raise TypeError('jacs should be a list')

        # get number of cutouts from image list
        self.images = images
        self.weights = weights
        self.segs = segs
        self.jacs = jacs
        # get box size from the first image
        self.box_size = self.images[0].array.shape[0]
        self.n_cutouts = len(self.images)

        # see if there are cutouts
        if self.n_cutouts < 1:
                raise ValueError('no cutouts in this object') 

        # check if the box size is correct
        if self.box_size not in BOX_SIZES:
            raise ValueError('box size should be in  [32,48,64,96,128,196,256], is %d' % box_size)

        # loop through the images and check if they are of the same size
        for extname in ('images','weights','segs'):

            # get the class field
            ext = eval('self.' + extname )

            # loop through exposures
            for icutout,cutout in enumerate(ext):

                # all cutouts should be numpy arrays
                
                if not any([isinstance(cutout,x) for x in galsim_image_types]):
                    raise TypeError('cutout %d in %s is not a GalSim Image' % (icutout,extname))

                # get the sizes of array
                nx=cutout.array.shape[0]
                ny=cutout.array.shape[1]

                # x and y size should be the same
                if nx != ny:
                    raise ValueError('%s should be square and is %d x %d',(extname,nx,ny))

                # check if box size is correct
                if nx != self.box_size:
                    raise ValueError('%s object %d has size %d and should be %d' % (extname,
                                                                    icutout,nx,self.box_size))

        # see if the number of Jacobians is right
        if len(self.jacs) != self.n_cutouts:
            raise ValueError('number of Jacobians is %d is not equal to number of cutouts %d' 
                                        % (len(self.jacs),self.n_cutouts ) )

        # check each Jacobian
        for jac in self.jacs:
            # should ba a numpy array
            if not isinstance(jac,numpy.ndarray):
                raise TypeError('Jacobians should be numpy arrays')
            # should have 2x2 shape
            if jac.shape != (2,2):
                raise ValueError('Jacobians should be 2x2')

def write_meds(filename,objlist,clobber=False):
    """
    Writes the galaxy, weights, segmaps images to a MEDS file.

    Arguments:
    ----------
    @param filename:    Name of meds file to be written
    @param objlist:     List of MultiExposureObjects
    """

    import numpy
    import sys
    import pyfits

    # initialise the catalog
    cat = {}
    cat['ncutout'] = []
    cat['box_size'] = []
    cat['start_row'] = []
    cat['dudrow'] = [] 
    cat['dudcol'] = []
    cat['dvdrow'] = []
    cat['dvdcol'] = []

    # initialise the image vectors
    vec = {}
    vec['image'] = []
    vec['seg'] = []
    vec['weight'] = []

    # initialise the image vector index
    n_vec = 0
    
    # get number of objects
    n_obj = len(objlist)

    # loop over objects
    for obj in objlist:

        # initialise the start indices for each image
        start_rows = numpy.ones(MAX_NCUTOUTS)*EMPTY_START_INDEX
        dudrow = numpy.ones(MAX_NCUTOUTS)*EMPTY_JAC 
        dudcol = numpy.ones(MAX_NCUTOUTS)*EMPTY_JAC
        dvdrow = numpy.ones(MAX_NCUTOUTS)*EMPTY_JAC
        dvdcol = numpy.ones(MAX_NCUTOUTS)*EMPTY_JAC

        # get the number of cutouts (exposures)
        n_cutout = obj.n_cutouts
        
        # append the catalog for this object
        cat['ncutout'].append(n_cutout)
        cat['box_size'].append(obj.box_size)
        
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
                vec['image'] = numpy.concatenate([vec['image'],obj.images[i].array.flatten()])
                vec['seg'] = numpy.concatenate([vec['seg'],obj.segs[i].array.flatten()])
                vec['weight'] = numpy.concatenate([vec['weight'],obj.weights[i].array.flatten()])          

            # append the Jacobian
            dudrow[i] = obj.jacs[i][0][0]  
            dudcol[i] = obj.jacs[i][0][1] 
            dvdrow[i] = obj.jacs[i][1][0] 
            dvdcol[i] = obj.jacs[i][1][1] 

            # check if we are running out of memory
            if sys.getsizeof(vec) > MAX_MEMORY:
                raise MemoryError(
                    'Running out of memory > %1.0fGB - you can increase the limit by changing' % 
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

    # get the primary HDU
    primary = pyfits.PrimaryHDU()

    # second hdu is the object_data
    cols = []
    cols.append( pyfits.Column(name='ncutout', format='i4', array=cat['ncutout'] ) )
    cols.append( pyfits.Column(name='box_size', format='i4', array=cat['box_size'] ) )
    cols.append( pyfits.Column(name='file_id', format='i4', array=[1]*n_obj) ) 
    cols.append( pyfits.Column(name='start_row', format='%di4' % MAX_NCUTOUTS, 
                                                        array=numpy.array(cat['start_row'])) )
    cols.append( pyfits.Column(name='orig_row', format='f8', array=[1]*n_obj) )
    cols.append( pyfits.Column(name='orig_col', format='f8', array=[1]*n_obj) )
    cols.append( pyfits.Column(name='orig_start_row', format='i4', array=[1]*n_obj) ) 
    cols.append( pyfits.Column(name='orig_start_col', format='i4', array=[1]*n_obj) ) 
    cols.append( pyfits.Column(name='cutout_row', format='f8' , array=[1]*n_obj) )
    cols.append( pyfits.Column(name='cutout_col', format='f8' , array=[1]*n_obj) )
    cols.append( pyfits.Column(name='dudrow', format='%df8'% MAX_NCUTOUTS, array=cat['dudrow'] ) )
    cols.append( pyfits.Column(name='dudcol', format='%df8'% MAX_NCUTOUTS, array=cat['dudcol'] ) )
    cols.append( pyfits.Column(name='dvdrow', format='%df8'% MAX_NCUTOUTS, array=cat['dvdrow'] ) )
    cols.append( pyfits.Column(name='dvdcol', format='%df8'% MAX_NCUTOUTS, array=cat['dvdcol'] ) )  
    object_data = pyfits.new_table(pyfits.ColDefs(cols))
    object_data.update_ext_name('object_data')

    # third hdu is image_info
    cols = []
    cols.append( pyfits.Column(name='image_path', format='A119',   array=['generated_by_galsim'] ) )
    cols.append( pyfits.Column(name='sky_path',   format='A124',   array=['generated_by_galsim'] ) )
    cols.append( pyfits.Column(name='seg_path',   format='A123',   array=['generated_by_galsim'] ) ) 
    image_info = pyfits.new_table(pyfits.ColDefs(cols))
    image_info.update_ext_name('image_info')

    # fourth hdu is metadata
    cols = []
    cols.append( pyfits.Column(name='cat_file',      format='A113', array=['generated_by_galsim'] ))  
    cols.append( pyfits.Column(name='coadd_file',    format='A109', array=['generated_by_galsim'] ))  
    cols.append( pyfits.Column(name='coadd_hdu',     format='A1',   array=['x']                   ))  
    cols.append( pyfits.Column(name='coadd_seg_hdu', format='A1',   array=['x']                   ))  
    cols.append( pyfits.Column(name='coadd_srclist', format='A115', array=['generated_by_galsim'] ))  
    cols.append( pyfits.Column(name='coadd_wt_hdu',  format='A1',   array=['x']                   ))  
    cols.append( pyfits.Column(name='coaddcat_file', format='A110', array=['generated_by_galsim'] ))  
    cols.append( pyfits.Column(name='coaddseg_file', format='A113', array=['generated_by_galsim'] ))  
    cols.append( pyfits.Column(name='cutout_file',   format='A108', array=['generated_by_galsim'] ))  
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
    hdu_list = pyfits.HDUList([primary,object_data,image_info,metadata,image_cutouts, 
                                                                weight_cutouts, seg_cutouts])
    hdu_list.writeto(filename,clobber=clobber)





