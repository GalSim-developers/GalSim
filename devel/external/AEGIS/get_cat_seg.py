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

"""Program Number: 5
Reduces the full catalog to objects that have postage stamps for each segment.
Then adds a few columns that are required for the main catalog. If a redshift 
and photometric catalog are provided, then objects from main catalog are matched
to the photometric and redshift catalog, and redshift and magnitude values are 
saved. 

Note: If no photometric or redshift catalog, set the input argument for the 
file names to be 'None'. The column names of these two catalogs in the code
need to be changed to their names in the input.

Since parametric fits were not done for the galaxies, the parameter values 
are manually set. If the fits step is performed, read the file below and save that
to the catalog, replacing the fake values set here.
"""

import subprocess
import numpy as np
import os
from astropy.table import Table, Column,hstack
from scipy import spatial

def get_cat_seg(args):
    """For a given segment and filter, identifies the objects for which postage
    stamps are drawn and writes to a new file.
    """
    seg = args.seg_id
    filt = args.filter
    f_str = args.file_filter_name
    cat_name = args.main_path + seg + '/' + filt + '_full.cat'
    new_cat_name = args.main_path + seg + '/' + filt + '_with_pstamp.fits'
    if os.path.isfile(new_cat_name) is True:
                subprocess.call(["rm", new_cat_name])    
    cat = Table.read(cat_name, format= 'ascii.basic')
    obj_list= args.main_path + seg + '/objects_with_p_stamps.txt' 
    objs = np.loadtxt(obj_list, dtype="int")
    temp = cat[objs]

    print " Adding columns for additional catalog information"
    # Columns to add values from photometric and redshift catalog
    col= Column(np.ones(len(temp))*-1,name='zphot',dtype='float',
                description = 'Redshift measured from other catalog')
    temp.add_column(col)
    col= Column(np.ones(len(temp))*-1,name='zphot_err',dtype='float',
                description = 'Error on redshift measured from other catalog')
    temp.add_column(col)
    col= Column(np.ones(len(temp))*99,name='ACS_' + f_str + 'BEST',dtype='float',
               description = 'Magnitude measured by ACS catalog')
    temp.add_column(col)
    col= Column(np.ones(len(temp))*99,name='ACS_' + f_str + 'BESTER',dtype='float',
               description = 'Magnitude error measured by ACS catalog')
    temp.add_column(col)
    col= Column(np.ones(len(temp))*-1,name='ACSTILE',dtype='int',
               description = 'Tile number of object in ACS catalog')
    temp.add_column(col)
    col= Column(np.ones(len(temp))*-1,name='ACSID',dtype='int',
               description = 'ID of object in ACS catalog')
    temp.add_column(col)
    temp.rename_column('ALPHA_J2000', 'RA')
    temp.rename_column('DELTA_J2000', 'DEC')

    # Use KD Tree to match objects in catalogs that are within 1 arcsecond
    c_x = temp['RA']*np.cos(np.radians(temp['DEC']))
    c_y = temp['DEC']
    c_pts = zip(c_x, c_y)
    tolerance = 1/3600.
    # If a photometric catalog is given
    if args.phot_cat_file_name is not 'None':
        phot_cat = Table.read(args.phot_cat_file_name, format="fits")
        #### Set column names as based on the photometric catalog.  
        p_ra = 'ACSRA_'+f_str
        p_dec = 'ACSDEC_'+f_str
        p_x = phot_cat[p_ra]*np.cos(np.radians(phot_cat[p_dec]))
        p_y = phot_cat[p_dec]
        p_tree = spatial.KDTree(zip(p_x, p_y)) 
        # Objects in catalog that are matched to photometric catalog.
        s = p_tree.query(c_pts, distance_upper_bound=tolerance)
        ch_q, = np.where(s[0]!= np.inf)
        c_p = ch_q
        p_c = s[1][ch_q]
        temp['ACS_' + f_str + 'BEST'][c_p] = phot_cat['ACS_' + f_str + 'BEST'][p_c]
        temp['ACS_' + f_str + 'BESTER'][c_p] = phot_cat['ACS_' + f_str + 'BESTER'][p_c]
        temp['ACSID'][c_p] = phot_cat['ACSID'][p_c]
        temp['ACSTILE'][c_p] = phot_cat['ACSTILE'][p_c]
    #If a redshift catalog is given
    if args.phot_cat_file_name is not 'None':
        # Only values with ZQUALITY >3 can be trusted.
        z_cat_temp = Table.read(args.z_cat_file_name, format="fits")
        #### set column names as based on the redshift catalog.     
        z_cat = z_cat_temp[z_cat_temp['ZQUALITY']>3]
        z_x = z_cat['RA']*np.cos(np.radians(z_cat['DEC']))
        z_y = z_cat['DEC']
        z_tree=spatial.KDTree(zip(z_x, z_y))
        # Objects in catalog that are matched to redshift catalog.
        s = z_tree.query(c_pts, distance_upper_bound=tolerance)
        ch_q, = np.where(s[0]!= np.inf)
        c_z = ch_q
        z_c = s[1][ch_q]
        temp['zphot'][c_z] = z_cat['ZBEST'][z_c]
        temp['zphot_err'][c_z] = z_cat['ZERR'][z_c]

    print " Adding required columns for selection catalog"
    # Add columns for selection catalog. 
    col = Column(np.zeros(len(temp)),name='NOISE_MEAN',dtype='float', description = 'Mean of background noise' )
    temp.add_column(col)
    col = Column(np.zeros(len(temp)),name='NOISE_VARIANCE',dtype='float', description = 'Variance of background noise' )
    temp.add_column(col)
    col= Column(np.zeros(len(temp)),name='stamp_flux',dtype='float', description = 'Total flux in the postage stamp' )
    temp.add_column(col)
    col= Column(np.zeros(len(temp)),name='sn_ellip_gauss',dtype='float')
    temp.add_column(col)
    col= Column(np.zeros(len(temp)),name='min_mask_dist_pixels',dtype='float')
    temp.add_column(col)
    col= Column(np.zeros(len(temp)),name='average_mask_adjacent_pixel_count',dtype='float')
    temp.add_column(col)
    col= Column(np.zeros(len(temp)),name='peak_image_pixel_count',dtype='float')
    temp.add_column(col)

    # Add columns for selection file       
    for idx,obj in enumerate(objs):
        path = args.main_path + seg + '/postage_stamps/stamp_stats/'
        stats_file =  path + str(obj) + '_' + filt + '.txt'
        stats = np.loadtxt(stats_file) 
        [b_mean, b_std, flux, min_dist, avg_flux, peak_val, snr] = stats
        temp['NOISE_MEAN'][idx] = b_mean
        temp['NOISE_VARIANCE'][idx] = b_std**2
        temp['stamp_flux'][idx] = flux
        temp['sn_ellip_gauss'][idx] = snr
        temp['min_mask_dist_pixels'][idx] = min_dist
        temp['average_mask_adjacent_pixel_count'][idx] = avg_flux
        temp['peak_image_pixel_count'][idx] = peak_val

    print " Adding columns for parametric fits catalog information"
    # Values manually set.
    # If fits is performed change script below.
    fit_mad_s = np.ones(len(temp))*9999
    fit_mad_b= np.ones(len(temp))*9999
    fit_dvc_btt = np.ones(len(temp))*9999
    use_bulgefit = np.ones(len(temp))*9999
    viable_sersic = np.ones(len(temp))*9999
    fit_status = [[9999.]*5]*len(temp)
    sersicfit = [[9999.]*8]*len(temp)
    bulgefit = [[9999.]*16]*len(temp)
    hlr = [[9999.]*3]*len(temp)
    names = ('fit_mad_s', 'fit_mad_b', 'fit_dvc_btt', 'use_bulgefit')
    names+= ('viable_sersic', 'fit_status', 'sersicfit', 'bulgefit', 'hlr')
    dtype =('f8', 'f8', 'f8','i4')
    dtype+=('i4', 'i4','f8', 'f8', 'f8')
    tab = [fit_mad_s, fit_mad_b, fit_dvc_btt, use_bulgefit]
    tab+= [viable_sersic, fit_status, sersicfit, bulgefit, hlr]
    temp2 = Table(tab, names=names, dtype=dtype)
    temp = hstack([temp,temp2])
    print "Catalog of all galaxies saved at ", new_cat_name
    temp.write(new_cat_name, format='fits') 

if __name__ == '__main__':
    import subprocess
    import numpy as np
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument('--seg_id', default= '0a',
                        help="id of segment to run [Default: '0a']")
    parser.add_argument('--filter', default= 'f814w',
                        help="filter of segment to run [Default: 'f814w']")
    parser.add_argument('--file_filter_name', default ='I' ,
                        help="Name of filter to use  [Default ='I']")
    parser.add_argument('--main_path',
                        default = '/nfs/slac/g/ki/ki19/deuce/AEGIS/AEGIS_catalog_full/')
    parser.add_argument('--seg_file_name', default ='/nfs/slac/g/ki/ki19/deuce/AEGIS/unzip/seg_ids.txt',
                        help="file with all seg id names" )
    parser.add_argument('--phot_cat_file_name', default ='/nfs/slac/g/ki/ki19/deuce/AEGIS/aegis_additional/egsacs_phot_nodup.fits',
                        help="other photometry catalogs" )
    parser.add_argument('--z_cat_file_name', default ='/nfs/slac/g/ki/ki19/deuce/AEGIS/aegis_additional/zcat.deep2.dr4.uniq.fits',
                        help="file with redshift information")
    parser.add_argument('--apply_cuts', default = True ,
                        help="Remove galaxies with imperfect masking during pstamp cleaning step.[Default =True]")
    args = parser.parse_args()
    get_cat_seg(args)

