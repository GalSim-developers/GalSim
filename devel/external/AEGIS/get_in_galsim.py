# Copyright (c) 2012-2018 by the GalSim developers team on GitHub
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

"""
Write complete catalog into files that can be opened with galsim.RealGalaxy()
and galsim.COSMOSCatalog(). For historical reasons, the input files for the 
galsim modules are:
1) Main catalog file
2) Selection file
3) Fits file
4) Files with galaxy images as hdu (1000 objects per file)
5) Files with psf images as hdu (1000 objects per file)

Separate files are made for each band. 

Requirements: 
Postage stamp images of galaxy and psf in multiple bands, catalog containing
information on each galaxy (in multiple bands). Each tile must have a saved 
file listing the identification number of galaxies with postage stamps.

The script has 3 steps:

1) Assign Number:
Assign individual identification number ('IDENT') to each object in catalog. We 
also want the catalog objects to be randomly shuffled. A new column ('ORDER')
gives the position of the objectt in the final shuffled catalog. Segment (tile)
ID in which that object was detected, number in that 
segment, individual identification number, position in final catalog, number
of the fits file where the postage stamps are saved, hdu number in that fits 
file of the image are saved in index table are saved in index_table.

Save Images:
Postage stamp images of galaxies and psf, in multiple bands, are stored in 
fits files. Each object is saves as the image HDU, whose number is mentioned
in the main catalog. The HDU number od galaxy and psf are same. Each fits file 
has 1000 images.

Save Catalogs: 3 output catalogs (in format required by GalSim modules) are 
produced with the same number of rows. Each catalog has a column 'IDENT' 
which is the unique identification number of each galaxy. Main catalog 
contains basic information of the galaxy, along with location of image 
fits file and noise correlation function. The selection catalog has parameters
that will be used to by galsim.COSMOSCatalog to determine  if the postage 
stamp is good. Fits file contains parametric fit values of each galaxy. 
Since no fits were performed here, fake values are entered. 

The script also saves a complete catalog with all the SExtractor output fields,
as well as those in the 3 final catalogs mentioned above. This catalog is saved
outside out_dir.

Output:
Fits files with galaxy images (in multiple bands), files with psf images (in 
multiple bands), main catalog file, selection file and fits file.

"""
import numpy as np
from astropy.io import fits
from astropy.table import Table,Column, vstack, hstack, join
import os,glob, subprocess

def assign_num(args, all_seg_ids):
    """Assigns individual identification number to each object"""
    seed =122
    np.random.seed(seed)
    print "Assigning number"
    names = ('SEG_ID', 'NUMBER', 'IDENT')
    dtype = ('string', 'int', 'int') 
    index_table = Table(names=names, dtype = dtype)
    ident = 0
    #objects detected are same in all filters. So getting objects in first filter
    #is sufficient
    filt = args.filter_names[0]
    for seg_id in all_seg_ids:
        file_name = args.main_path + seg_id + '/' + filt + '_selected.fits'
        catalog = Table.read(file_name, format='fits')
        idents = range(ident,ident+len(catalog))
        seg_ids = [seg_id]*len(catalog)
        numbers = catalog['NUMBER']       
        temp = Table([seg_ids, numbers,idents],names=names, dtype = dtype)
        index_table = vstack([index_table,temp])
        ident+=len(catalog)
    shuffle_idents = range(len(index_table))
    np.random.shuffle(shuffle_idents)
    index_table= index_table[shuffle_idents]
    order_idents = range(len(index_table))
    file_nums = np.array(order_idents)/1000 + 1
    hdus= np.zeros(len(order_idents))
    names = ('ORDER', 'FILE_NUM', 'HDU')
    dtype = ('int' ,'int', 'int')
    temp = Table([order_idents,file_nums,hdus], names=names, dtype=dtype)
    index_table = hstack([index_table,temp])
    cat_name = args.main_path + 'index_table_' + args.cat_name.replace('filter', '')
    return index_table

def get_images(args, index_table,
               filt, filt_name):
    """Make fits files of galaxy and psf postage stamps"""
    print "Saving images"
    n = np.max(index_table['FILE_NUM'])
    print "Total number of files will be ",n
    for f in range(1,n+1):
        hdu_count = 0
        q, = np.where(index_table['FILE_NUM'] == f)
        im_hdul = fits.HDUList()
        psf_hdul = fits.HDUList()
        for i in q:
            path = args.main_path + '/' + index_table['SEG_ID'][i]+'/postage_stamps/'
            name = filt + '_' + index_table['SEG_ID'][i]+ '_' +str(index_table['NUMBER'][i])+'_gal.fits'
            im_file=  path + name
            name = filt + '_' + index_table['SEG_ID'][i]+ '_' +str(index_table['NUMBER'][i])+'_psf.fits'
            psf_file =  path + name
            im = fits.open(im_file)[0].data
            psf = fits.open(psf_file)[0].data
            im_hdul.append(fits.ImageHDU(im))
            psf_hdul.append(fits.ImageHDU(psf))
            index_table['HDU'][i] = hdu_count
            hdu_count+=1
        path = args.main_path + args.out_dir 
        im_name = args.gal_im_name.replace('umber', str(f))
        im_name = im_name.replace('filter', filt_name)
        im_name = path + im_name
        psf_name = args.psf_im_name.replace('umber', str(f))
        psf_name = psf_name.replace('filter', filt_name)
        psf_name = path + psf_name
        im_hdul.writeto(im_name, clobber=True)
        psf_hdul.writeto(psf_name, clobber=True)
    cat_name = 'index_' + args.cat_name.replace('filter', 'all')
    index_table.sort('ORDER')
    index_table.write(args.main_path + cat_name, format='fits',
                      overwrite=True)
    print 'Saving index catalog at', cat_name
    return index_table
  
def main_table():
    """Columns in main catalog"""
    names = ('IDENT', 'RA', 'DEC', 'MAG', 'BAND', 'WEIGHT', 'GAL_FILENAME')
    names+= ('PSF_FILENAME', 'GAL_HDU', 'PSF_HDU', 'PIXEL_SCALE')
    names+= ('NOISE_MEAN', 'NOISE_VARIANCE', 'NOISE_FILENAME', 'stamp_flux')
    dtype = ('i4', 'f8', 'f8', 'f8', 'S40', 'f8', 'S256')
    dtype+= ('S288', 'i4', 'i4', 'f8')
    dtype+= ('f8', 'f8', 'S208', 'f8')
    table = Table(names=names, dtype=dtype)
    return table
    			
def selection_table():
    """Columns in selection catalog"""
    names = ('IDENT', 'dmag', 'sn_ellip_gauss', 'min_mask_dist_pixels')
    names+= ('average_mask_adjacent_pixel_count', 'peak_image_pixel_count')
    dtype = ('i4', 'f8', 'f8', 'f8', 'f8', 'f8')
    table = Table(names=names, dtype=dtype)
    return table

def fits_table():
    """Columns in parametric fit catalog"""
    names = ('IDENT', 'mag_auto', 'flux_radius', 'zphot','fit_mad_s', 'fit_mad_b')
    names+= ('fit_dvc_btt', 'use_bulgefit', 'viable_sersic', 'flux')
    dtype = ('i4', 'f8', 'f8', 'f8', 'f8','f8')
    dtype+= ('f8', 'i4', 'i4', 'f8')
    table = Table(names=names, dtype=dtype,)
    col = Column( name='sersicfit', shape=(8,), dtype='f8')
    table.add_column(col, index=4)
    col = Column( name='bulgefit', shape=(16,), dtype='f8')
    table.add_column(col, index=5)
    col = Column( name='fit_status', shape=(5,), dtype='i4')
    table.add_column(col, index=6)
    col = Column( name='hlr', shape=(3,), dtype='f8')
    table.add_column(col)
    return table

def get_main_catalog(args, index_table, all_seg_ids):
    """Makes main catalog containing information on all selected galaxies.
    Columns are identical to COSMOS Real Galaxy catalog"""
    print "Creating main catalog" 
    for f, filt in enumerate(args.filter_names):
    	final_table = main_table()
        complete_table=Table()
        for seg_id in all_seg_ids:
            file_name = args.main_path + seg_id + '/' + filt + '_selected.fits'
            seg_cat = Table.read(file_name, format='fits')
            q, = np.where(index_table['SEG_ID'] == seg_id)
            indx_seg = index_table[q]
            temp = join(seg_cat, indx_seg, keys='NUMBER')
            col = Column(temp['HDU'], name='PSF_HDU')
            temp.add_column(col)
            temp.rename_column('MAG_CORR', 'MAG')
            temp.rename_column('HDU', 'GAL_HDU')
            p_scales = np.ones(len(q))*0.03
            weights = np.ones(len(q))
            im = [args.gal_im_name.replace('filter', args.file_filter_name[f])]*len(q)
            im_names = [im[i].replace('umber',str(temp['FILE_NUM'][i])) for i in range(len(im))]
            psf = [args.psf_im_name.replace('filter', args.file_filter_name[f])]*len(q)
            psf_names = [psf[i].replace('umber',str(temp['FILE_NUM'][i])) for i in range(len(psf))]
            noise_names=[args.noise_file_name.replace('filter', args.file_filter_name[f])]*len(q)
            names = ('WEIGHT','GAL_FILENAME', 'PSF_FILENAME',
                     'PIXEL_SCALE', 'NOISE_FILENAME')
            dtype =('f8', 'S256', 'S288', 'f8', 'S208')
            tab = [weights, im_names, psf_names, p_scales, noise_names]
            temp2 = Table(tab, names=names, dtype=dtype)
            temp = hstack([temp,temp2])
            final_table = vstack([final_table,temp], join_type='inner')
            complete_table = vstack([complete_table,temp])
        path = args.main_path + args.out_dir 
        cat_name = args.cat_name.replace('filter', args.file_filter_name[f])
        index_table.sort('ORDER')
        ord_indx = [np.where(i_t==final_table['IDENT'])[0][0] for i_t in index_table['IDENT']]
        final_table[ord_indx].write(path + cat_name, format='fits',
                                                overwrite=True)
        print "Savings fits file at ", path + cat_name
        cat_name = 'complete_' + args.cat_name.replace('filter', args.file_filter_name[f])
        complete_table[ord_indx].write(args.main_path + cat_name, format='fits',
                                                   overwrite=True)
def get_selection_catalog(args, index_table, all_seg_ids):
    """Makes catalog containing information that can be used to select good galaxies.
    Columns are identical to COSMOS Real Galaxy catalog"""
    print "Creating selection catalog"     
    for f, filt in enumerate(args.filter_names):
    	final_table = selection_table()
        for seg_id in all_seg_ids:
            file_name = args.main_path + seg_id + '/' + filt + '_selected.fits'
            seg_cat = Table.read(file_name, format='fits')
            q, = np.where(index_table['SEG_ID'] == seg_id)
            indx_seg = index_table[q]
            temp = join(seg_cat, indx_seg, keys='NUMBER')
            col =Column(np.zeros(len(q)), name='dmag')
            temp.add_column(col)
            final_table = vstack([final_table,temp], join_type='inner')
        path = args.main_path + args.out_dir 
        index_table.sort('ORDER')
        ord_indx = [np.where(i_t==final_table['IDENT'])[0][0] for i_t in index_table['IDENT']]        
        file_name = args.selec_file_name.replace('filter', args.file_filter_name[f])
        final_table[ord_indx].write(path + file_name, format='fits',
                                                overwrite=True)
        print "Savings fits file at ", path + file_name

def get_fits_catalog(args, index_table, all_seg_ids):
    """Makes catalog containing information about parametric fits to the galaxies.
    Columns are identical to COSMOS Real Galaxy catalog"""   
    print "Creating fits catalog" 
    for f, filt in enumerate(args.filter_names):
    	final_table = fits_table()
        for seg_id in all_seg_ids:
            file_name = args.main_path + seg_id + '/' + filt + '_selected.fits'
            seg_cat = Table.read(file_name, format='fits')
            q, = np.where(index_table['SEG_ID'] == seg_id)
            indx_seg = index_table[q]
            temp = join(seg_cat, indx_seg, keys='NUMBER')
            temp.rename_column('MAG_CORR', 'mag_auto')
            temp.rename_column('FLUX_RADIUS', 'flux_radius')            
            col = Column(temp['stamp_flux'], name='flux')
            temp.add_column(col)
            final_table = vstack([final_table,temp], join_type='inner')
        path = args.main_path + args.out_dir 
        index_table.sort('ORDER')
        ord_indx = [np.where(i_t==final_table['IDENT'])[0][0] for i_t in index_table['IDENT']]
        file_name = args.fits_file_name.replace('filter', args.file_filter_name[f])
        print "Savings fits file at ", path + file_name
        final_table[ord_indx].write(path + file_name, format='fits',
                                                overwrite=True)
def make_all(args):
    """Saves Postage stamps and final catalogs in a format that can be read by
     galsim modules"""
    if os.path.isdir(args.main_path + args.out_dir) is False:
            subprocess.call(["mkdir", args.main_path + args.out_dir])
    else:
        for fl in glob.glob(args.main_path + args.out_dir+'*'):
            os.remove(fl)
    all_seg_ids = np.loadtxt(args.seg_list_file, delimiter=" ",dtype='S2')
    index_table = assign_num(args, all_seg_ids)
    for f, filt in enumerate(args.filter_names):
        idx = get_images(args, index_table, filt,
                         args.file_filter_name[f])
    get_main_catalog(args, idx, 
                     all_seg_ids)
    get_selection_catalog(args, idx,
                          all_seg_ids)
    get_fits_catalog(args, idx,
                     all_seg_ids)


