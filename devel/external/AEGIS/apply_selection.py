"""Program Number: 5
Reduces the full catalog to objects that have postage stamps for each segment.
Then adds a few columns that are required for the main catalog. If a redshift 
and photometric catalog are provided, then objects from main catalog are matched
to the photometric and redshift catlog, and redshift and magnitude values are 
saved. 

Note: If no photomteric or redshift catalog, set the input argument for the 
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

def apply_selection(args):
    """Apply selection cut for each galaxy.
    """
    
    all_seg_ids = np.loadtxt(args.seg_list_file, delimiter=" ",dtype='S2')
    for seg in all_seg_ids:
        cat={}
        temp=[]
        for f, filt in enumerate(args.filter_names):
            cat_name = args.main_path + seg + '/' + filt + '_with_pstamp.fits'
            new_cat_name = args.main_path + seg + '/' + filt + '_selected.fits'
            cat[f] = Table.read(cat_name, format= 'fits')
            #Select only good postage stamps
            cond1 = cat[f]['min_mask_dist_pixels'] > 11
            cond2 = cat[f]['average_mask_adjacent_pixel_count']/ cat[f]['peak_image_pixel_count'] < 0.8
            cond3 = cat[f]['sn_ellip_gauss'] > 0
            good, = np.where((cond1 | cond2) & cond3)
            temp.append(good)
        if args.apply_cuts is True:
            select_gal = reduce(np.intersect1d,temp)
        else:
            # If no selection cut has to be applied
            select_gal = range(len(cat[f]))
        for f, filt in enumerate(args.filter_names):
            new_cat_name = args.main_path + seg + '/' + filt + '_selected.fits'
            new_cat = cat[f][select_gal]
            print "Catalog with pstamps saved at ", new_cat_name
            new_cat.write(new_cat_name, format='fits') 
        select_gal_list= args.main_path + seg + '/gal_in_cat.txt' 
        np.savetxt(select_gal_list, select_gal) 

if __name__ == '__main__':
    import subprocess
    import numpy as np
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument('--filter_names', default= ['f606w','f814w'],
                        help="names of filters [Default: ['f606w','f814w']]")
    parser.add_argument('--main_path',
                        default = '/nfs/slac/g/ki/ki19/deuce/AEGIS/AEGIS_catalog_full/')
    parser.add_argument('--seg_file_name', default ='/nfs/slac/g/ki/ki19/deuce/AEGIS/unzip/seg_ids.txt',
                        help="file with all seg id names" )
    parser.add_argument('--apply_cuts', default = True,
                        help="Remove galaxies with imperfect masking during pstamp cleaning step.[Default =True]")
    args = parser.parse_args()
    apply_selection(args)

