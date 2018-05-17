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

"""Program Number: 6
Applies selection cuts on galaxies in all filters and calls script to
create catalog in a format that can be read by galsim.RealGalaxy().

Note: WEIGHT is still set to 1 for all galaxies. 
"""




import subprocess
import os
import numpy as np
import get_in_galsim
from astropy.table import Table


def apply_selection(args, all_seg_ids):
    """Applies selection cuts to galaxies in each segment.
    galaxy is selected for final catalog if: 
    1) Closest masked pixel is more than 5 pixels away
                  OR
    2) Closest masked pixel is greater than 5 but less than 11 pixels away
                  AND 
       Average flux in 9*9 pixels centered on the closest replaced pixel (before masking) is less than 
       0.2 of the peak flux of central galaxy
                   AND  
    3) HSM was able to measure adaptive moments for the galaxy 
                   OR
        ELLIPTICITY of galaxy as measured by SEXTRACTOR is greater than 0.75  

    Galaxies satisfying these criteria in all filters are selected for the 
    final catalog and saved to *_selected.fits file.

    Some postage stamps that passed the above cuts, but were found to have defects
    upon visual inspection, are removed manually. The SEG ID and NUMBER of these
    galaxies is read from args.manual_bad_stamps and removed from the catalog.
    """
    for seg in all_seg_ids:
        print "Running segment ",seg
        cat={}
        temp=[]
        for f, filt in enumerate(args.filter_names):
            cat_name = args.main_path + seg + '/' + filt + '_with_pstamp.fits'
            cat[f] = Table.read(cat_name, format= 'fits')
            #Remove non elliptical galaxies that failed HSM 
            cond1 = cat[f]['sn_ellip_gauss'] <= 0
            cond2 = cat[f]['ELLIPTICITY'] <= 0.75
            bad, = np.where(cond1 & cond2)
            temp.append(bad)
        remove_gals = reduce(np.union1d,temp)
        print "{0} galaxies failed HSM".format(len(remove_gals))
        temp=[]
        for f, filt in enumerate(args.filter_names):
            cat[f].remove_rows(remove_gals)
            #Remove postage stamps with closest pixel masked during cleaning step is less than 5 pixels or
            #if less than 11 and average flux in adjacent pixels is 20% of brightest pixel.
            cond1 = cat[f]['min_mask_dist_pixels'] < 5
            cond2 = cat[f]['min_mask_dist_pixels'] < 11
            cond3 = cat[f]['average_mask_adjacent_pixel_count']/ cat[f]['peak_image_pixel_count'] > 0.2
            bad, = np.where(cond1 | (cond2 & cond3))
            temp.append(bad)
        remove_gals = reduce(np.union1d,temp)
        print "{0} galaxies failed Cleaning cuts".format(len(remove_gals))
        for f, filt in enumerate(args.filter_names):
            cat[f].remove_rows(remove_gals)
        #If there exists a file to manually remove galaxies
        if os.path.isfile(args.manual_bad_stamps) is True:
            bad_file = np.genfromtxt(args.manual_bad_stamps,dtype='str').T
            bad_gal = bad_file[1][bad_file[0]==seg]
            remove_gals_manual=[]
            for i in bad_gal:
                q, = np.where(cat[f]['NUMBER'] == int(i))
                if len(q) ==0:
                    print "Already removed"
                else:
                    remove_gals_manual.append(q[0])
        else:
            print "No file provided for manual galaxy removal"
            remove_gals =[] 
        print "{0} galaxies removed manually".format(len(remove_gals_manual))
        for f, filt in enumerate(args.filter_names):
            cat[f].remove_rows(remove_gals_manual)
            new_cat_name = args.main_path + seg + '/' + filt + '_selected.fits'
            print "Catalog with pstamps that pass selection cuts saved at ", new_cat_name
            cat[f].write(new_cat_name, format='fits', overwrite=True) 

def main(args):
    all_seg_ids = np.loadtxt(args.seg_list_file, delimiter=" ",dtype='S2')
    if args.apply_cuts is True:
        apply_selection(args, all_seg_ids)
    else:
        # No selection cuts applied. Writes entire input catalog to selected galaxies file
        for seg in all_seg_ids:
            for f, filt in enumerate(args.filter_names):
                cat_name = args.main_path + seg + '/' + filt + '_with_pstamp.fits'
                cat[f] = Table.read(cat_name, format= 'fits')
                new_cat_name = args.main_path + seg + '/' + filt + '_selected.fits'
                cat[f].write(new_cat_name, format='fits', overwrite=True) 
    #Saves catalog in a format that can be read by galsim
    get_in_galsim.make_all(args)

if __name__ == '__main__':
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument('--filter_names', default= ['f606w','f814w'],
                        help="names of filters [Default: ['f606w','f814w']]")
    parser.add_argument('--main_path',
                        default = '/nfs/slac/g/ki/ki19/deuce/AEGIS/AEGIS_catalog_full/')
    parser.add_argument('--out_dir', default = "AEGIS_training_sample/",
                        help="directory containing the final catalog")
    parser.add_argument('--cat_name', default = "AEGIS_galaxy_catalog_filter_25.2.fits",
                        help="Final catalog name")
    parser.add_argument('--gal_im_name', default = "AEGIS_galaxy_images_filter_25.2_number.fits",
                        help="Final name of galaxy images")
    parser.add_argument('--psf_im_name', default = "AEGIS_galaxy_PSF_images_filter_25.2_number.fits",
                        help="Final name of PSF images")
    parser.add_argument('--selec_file_name', default = "AEGIS_galaxy_catalog_filter_25.2_selection.fits",
                        help="Catalog with selection information")
    parser.add_argument('--noise_file_name', default = "acs_filter_unrot_sci_cf.fits",
                        help="File with correlation function of noise")
    parser.add_argument('--file_filter_name', default =['V', 'I'] ,
                        help="Name of filter to use ")
    parser.add_argument('--fits_file_name', default = "AEGIS_galaxy_catalog_filter_25.2_fits.fits",
                        help="Name of Catalog with fit information")
    parser.add_argument('--seg_list_file', default ='/nfs/slac/g/ki/ki19/deuce/AEGIS/unzip/seg_ids.txt',
                        help="file with all seg id names" )
    parser.add_argument('--manual_bad_stamps', default ='manual_bad_stamps.txt',
                        help="file with SEG ID and galaxy NUMBER of visually identified bad postage stamps" )
    parser.add_argument('--apply_cuts', default = True,
                        help="Remove galaxies with imperfect masking during pstamp cleaning step.[Default=True]")
    args = parser.parse_args()
    main(args)
