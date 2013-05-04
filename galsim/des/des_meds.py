"""
Defines the MEDS class to work with MEDS (Multi Epoch Data Structures)

See docs for the MEDS class for more info

    Copyright (C) 2013, Erin Sheldon, BNL

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.
"""

import numpy


BOX_SIZES = [32,48,64,96,128,196,256]
MAX_MEMORY = 1e9
MAX_NCUTOUTS = 11
EMPTY_START_INDEX = 9999
EMPTY_JAC = 999

class MEDS(object):
    """
    Class to work with MEDS (Multi Epoch Data Structures)

    For details of the structure, see
    https://cdcvs.fnal.gov/redmine/projects/deswlwg/wiki/Multi_Epoch_Data_Structure

    One can extract cutouts using get_cutout() and get_mosaic() and
    get_cutout_list()

    One can access all fields from the catalog using [field_name] notation. The
    number of entries is in the .size attribute. Note the actual fields in the
    catalog may change over time.  You can use get_cat() to get the full
    catalog as a recarray.

    The first cutout for an object is always from the coadd.

    public methods
    --------------
    get_cutout(iobj, icutout)
        Get a single cutout for the indicated entry
    get_mosaic(iobj)
        Get a mosaic of all cutouts associated with this coadd object
    get_cutout_list(iobj)
        Get an image list with all cutouts associated with this coadd object
    get_source_path(iobj, icutout)
        Get the source filename associated with the indicated cutout
    get_sky_path(iobj, icutout)
        Get the source sky filename associated with the indicated cutout
    get_source_info(iobj, icutout)
        Get all info about the source images
    get_cat()
        Get the catalog; extension 1
    get_image_info()
        Get the entire image info structure
    get_meta()
        Get all the metadata

    examples
    --------
    from deswl import meds
    m=meds.MEDS(filename)

    # number of coadd objects
    num=m.size

    # number of cutouts for object 35
    m['ncutout'][35]

    # get cutout 3 for object 35
    im=m.get_cutout(35,3)

    # get all the cutouts for object 35 as a single image
    mosaic=m.get_mosaic(35)

    # get all the cutouts for object 35 as a list of images
    im=m.get_cutout_list(35)

    # get a cutout for the weight map
    wt=m.get_cutout(35,3,type='weight')

    # get a cutout for the segmentation map
    seg=m.get_cutout(35,3,type='seg')

    # get the source filename for cutout 3 for object 35
    fname=m.get_source_path(35,3)

    # you can access any of the columns in the
    # catalog (stored as a recarray) directly

    # e.g. get the center in the cutout for use in image processing
    row = m['row_cutout'][35]
    col = m['col_cutout'][35]

    # source filename
    fname = m.get_source_path(35,3)

    # or you can just get the catalog to work with
    cat=m.get_cat()
    info=m.get_image_info()
    meta=m.get_meta()



    Fields in main catalog
    -----------------------

     id                 i4       id from coadd catalog
     ncutout            i4       number of cutouts for this object
     box_size           i4       box size for each cutout
     file_id            i4[NMAX] zero-offset id into the file names in the 
                                 second extension
     start_row          i4[NMAX] zero-offset, points to start of each cutout.
     orig_row           f8[NMAX] zero-offset position in original image
     orig_col           f8[NMAX] zero-offset position in original image
     orig_start_row     i4[NMAX] zero-offset start corner in original image
     orig_start_col     i4[NMAX] zero-offset start corner in original image
     cutout_row         f8[NMAX] zero-offset position in cutout imag
     cutout_col         f8[NMAX] zero-offset position in cutout image
     dudrow             f8[NMAX] jacobian of transformation 
                                 row,col->ra,dec tangent plane (u,v)
     dudcol             f8[NMAX]
     dvdrow             f8[NMAX]
     dvdcol             f8[NMAX]


    requirements
    ------------
    numpy
    fitsio https://github.com/esheldon/fitsio
    """
    def __init__(self, filename):
        import fitsio
        self._filename=filename
        
        self._fits=fitsio.FITS(filename)

        self._cat=self._fits["object_data"][:]
        self._image_info=self._fits["image_info"][:]
        self._meta=self._fits["metadata"][:]

            
    def get_cutout(self, iobj, icutout, type='image'):
        """
        Get a single cutout for the indicated entry

        parameters
        ----------
        iobj:
            Index of the object
        icutout:
            Index of the cutout for this object.
        type: string, optional
            Cutout type. Default is 'image'.  Allowed
            values are 'image','weight'

        returns
        -------
        The cutout image
        """
        self._check_indices(iobj, icutout=icutout)

        box_size=self._cat['box_size'][iobj]
        start_row = self._cat['start_row'][iobj,icutout]
        row_end = start_row + box_size*box_size

        extname=self._get_extension_name(type)

        imflat = self._fits[extname][start_row:row_end]
        im = imflat.reshape(box_size,box_size)
        return im

    def get_mosaic(self, iobj, type='image'):
        """
        Get a mosaic of all cutouts associated with this coadd object

        parameters
        ----------
        iobj:
            Index of the object
        type: string, optional
            Cutout type. Default is 'image'.  Allowed
            values are 'image','weight'

        returns
        -------
        An image holding all cutouts
        """

        self._check_indices(iobj)

        ncutout=self._cat['ncutout'][iobj]
        box_size=self._cat['box_size'][iobj]

        start_row = self._cat['start_row'][iobj,0]
        row_end = start_row + box_size*box_size*ncutout

        extname=self._get_extension_name(type)

        mflat=self._fits[extname][start_row:row_end]
        mosaic=mflat.reshape(ncutout*box_size, box_size)

        return mosaic

    def get_cutout_list(self, iobj, type='image'):
        """
        Get an image list with all cutouts associated with this coadd object

        Note each individual cutout is actually a view into a larger
        mosaic of all images.

        parameters
        ----------
        iobj:
            Index of the object
        type: string, optional
            Cutout type. Default is 'image'.  Allowed
            values are 'image','weight'

        returns
        -------
        A list of images hold all cutouts.
        """

        mosaic=self.get_mosaic(iobj,type=type)
        ncutout=self._cat['ncutout'][iobj]
        box_size=self._cat['box_size'][iobj]
        return self._split_mosaic(mosaic, box_size, ncutout)

    def get_source_info(self, iobj, icutout):
        """
        Get the full source file information for the indicated cutout.

        Includes SE image and sky image

        parameters
        ----------
        iobj: 
            Index of the object
        """
        self._check_indices(iobj, icutout=icutout)
        ifile=self._cat['file_id'][iobj,icutout]
        return self._image_info[ifile]

    def get_source_path(self, iobj, icutout):
        """
        Get the source filename associated with the indicated cutout

        parameters
        ----------
        iobj:
            Index of the object
        icutout:
            Index of the cutout for this object.

        returns
        -------
        The filename
        """

        info=self.get_source_info(iobj, icutout)
        return info['image_path']

    def get_sky_path(self, iobj, icutout):
        """
        Get the source filename associated with the indicated cutout

        parameters
        ----------
        iobj:
            Index of the object
        icutout:
            Index of the cutout for this object.

        returns
        -------
        The filename
        """

        info=self.get_source_info(iobj, icutout)
        return info['sky_path']


    def get_cat(self):
        """
        Get the catalog
        """
        return self._cat

    def get_image_info(self):
        """
        Get all image information
        """
        return self._image_info
    
    def get_meta(self):
        """
        Get all the metadata
        """
        return self._meta

    def _get_extension_name(self, type):
        if type=='image':
            return "image_cutouts"
        elif type=="weight":
            return "weight_cutouts"
        elif type=="seg":
            return "seg_cutouts"
        else:
            raise ValueError("bad cutout type '%s'" % type)

    def _split_mosaic(self, mosaic, box_size, ncutout):
        imlist=[]
        for i in xrange(ncutout):
            r1=i*box_size
            r2=(i+1)*box_size
            imlist.append( mosaic[r1:r2, :] )

        return imlist


    def _check_indices(self, iobj, icutout=None):
        if iobj >= self._cat.size:
            raise ValueError("object index should be within "
                             "[0,%s)" % self._cat.size)

        ncutout=self._cat['ncutout'][iobj]
        if ncutout==0:
            raise ValueError("object %s has no cutouts" % iobj)

        if icutout is not None:
            if icutout >= ncutout:
                raise ValueError("requested cutout index %s for "
                                 "object %s should be in bounds "
                                 "[0,%s)" % (icutout,iobj,ncutout))

    def __repr__(self):
        return repr(self._fits[1])
    def __getitem__(self, item):
        return self._cat[item]

    @property
    def size(self):
        return self._cat.size

class MultiExposureObject(object):
    """
    A class containing exposures for single object, along with other information.

    Available fields:
        self.images             list of images of the object (numpy arrays)
        self.weights            list of weight maps (numpy arrays)
        self.segs               list of segmentation masks (numpy arrays)
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
        self.box_size = self.images[0].shape[0]
        self.n_cutouts = len(self.images)

        if self.n_cutouts < 1:
                raise ValueError('no cutouts in this object') 

        # check if the box size is correct
        if self.box_size not in BOX_SIZES:
            raise ValueError('box size should be in  [32,48,64,96,128,196,256], is %d' % box_size)

        # loop through the images and check if they are of the same size
        for extname in ('images','weights','segs'):

            ext = eval('self.' + extname )

            for icutout,cutout in enumerate(ext):

                if not isinstance(cutout,numpy.ndarray):
                    raise TypeError('cutout %d in %s is not an array' % (icutout,extname))

                nx=cutout.shape[0]
                ny=cutout.shape[1]

                if nx != ny:
                    raise ValueError('%s should be square and is %d x %d',(extname,nx,ny))

                if nx != self.box_size:
                    raise ValueError('%s object %d has size %d and should be %d' % (extname,
                                                                    icutout,nx,self.box_size))

        if len(self.jacs) != self.n_cutouts:
            raise ValueError('number of Jacobians is %d is not equal to number of cutouts %d' 
                                        % (len(self.jacs),self.n_cutouts ) )

        for jac in self.jacs:
            if not isinstance(jac,numpy.ndarray):
                raise TypeError('Jacobians should be numpy arrays')
            if jac.shape != (2,2):
                raise ValueError('Jacobians should be 2x2')


def write(filename,objlist,clobber=False):
    """
    Writes the galaxy, weights, segmaps images to the meds file.

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
                vec['image'] = obj.images[i].flatten()
                vec['seg'] = obj.segs[i].flatten()
                vec['weight'] = obj.weights[i].flatten()
            # if vector already exists
            else:
                vec['image'] = numpy.concatenate([vec['image'],obj.images[i].flatten()])
                vec['seg'] = numpy.concatenate([vec['seg'],obj.segs[i].flatten()])
                vec['weight'] = numpy.concatenate([vec['weight'],obj.weights[i].flatten()])          

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





