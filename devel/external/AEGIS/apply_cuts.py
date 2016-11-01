import subprocess
import numpy as np
import get_in_galsim


def apply_selection(args, all_seg_ids):
    """Applies selection cuts to galaxies in each segemnt.
    Selection criteria are: 
    1) Closest masked pixel is more than 11 pixels away
                  OR 
    2) Average flux in 9*9 pixels centered on the closest replaced pixel (before masking) is less than 
       half of the peak flux of central galaxy
                   AND  
    3) SNR in elliptical gaussian filter is positve.
    
    Galaxies statifying these criteria in all filters are selected for the final catalog.
    """
    for seg in all_seg_ids:
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
        for f, filt in enumerate(args.filter_names):
            cat[f].remove_rows(remove_gals)
            #Remove postage stamps with closest pixel masked during cleanin step is less than 5 pixels or
            #if less than 11 and average flux in adjacent pixels is 20% of brightest pixel.
            cond1 = cat[f]['min_mask_dist_pixels'] < 5
            cond2 = cat[f]['min_mask_dist_pixels'] < 11
            cond3 = cat[f]['average_mask_adjacent_pixel_count']/ cat[f]['peak_image_pixel_count'] > 0.2
            bad, = np.where(cond1 | (cond2 & cond3))
            temp.append(bad)
        remove_gals = reduce(np.union1d,temp)
        for f, filt in enumerate(args.filter_names):
            cat[f].remove_rows(remove_gals)
        #If there exists a file to manually remove galaxies
        if os.path.isdir(args.manual_bad_stamps) is True:
            bad_file = np.genfromtxt(args.manual_bad_stamps,dtype='str').T
            bad_gal = bad_file[1][bad_file[0]==seg]
            remove_gals = [np.where(cat[f]['NUMBER'] == i)[0] for i in bad_gal]
        else:
            remove_gals =[] 
        for f, filt in enumerate(args.filter_names):
            cat[f].remove_rows(remove_gals_manual)
            new_cat_name = args.main_path + seg + '/' + filt + '_selected.fits'
            new_cat = cat[f][select_gal]
            if os.path.isfile(new_cat_name) is True:
                subprocess.call(["rm", new_cat_name]) 
            print "Catalog with pstamps saved at ", new_cat_name
            new_cat.write(new_cat_name, format='fits') 

def main(args):
    all_seg_ids = np.loadtxt(args.seg_list_file, delimiter=" ",dtype='S2')
    if args.apply_cuts is True:
        apply_selection(args, all_seg_ids)
        input_cat = args.main_path + seg_id + '/' + filt + '_selected.fits' 
        get_in_galsim(args, inut_cat)
    else:
        # If no selection cut has to be applied
        input_cat = args.main_path + seg_id + '/' + filt + '_with_pstamp.fits' 
        get_in_galsim(args, inut_cat)

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