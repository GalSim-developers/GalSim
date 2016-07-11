# python main2.py --out_path='/nfs/slac/g/ki/ki19/deuce/AEGIS/output_table3/'
# option to pick detection files
# gain as input
#

""" Identify stars in field used to get focus  to calculate PSF.
The code is run on one of the segemnts of the AEGIS 63 fields.Use SExtractor 
simultaneously on F606 and F814 images to identify galaxies and stars with hot 
and cold detection technique.Pick stars to caluclate focus and save for manual
inspection. 
drizzled image files are assumed as file_path/filter_name/file_name
weight image files are assumed as wht_path/filter_name/wht_name

Input of file_name must be file name with the segment id replaced bu the string 'seg_id'.
Input of wht_name must be weight name with the segment id replaced bu the string 'seg_id'.
the two can be the same.

Example an image file EGS_10134_1a_acs_wfc_f606w_30mas_unrot_drz.fits 
in  /nfs/slac/g/ki/ki19/deuce/AEGIS/unzip/f606w with filter f606w and segment id 1a,
the input arguments will be
file_path : /nfs/slac/g/ki/ki19/deuce/AEGIS/unzip
filter_names: ['f606w']
file_name: EGS_10134_seg_id_acs_wfc_filter_30mas_unrot_drz.fits 

The PSF is determined by only by the focus positionfor the tiny tim fields.
Focus is calculated by comparing the brightest stars in a field to the modelled
tiny tim star fields created for different focus postions.
the loction of the already created tiny tim for differnt filters is at:
tt_files_path/filter_name

The TT fields contain fits files for simulated HST PSF for different focus 
position at different postions on the image plane. 
Input of tt_feild_name must be a string listing the file name with the 
focus replaced bu the string 'seg_id'. For eg a file name TinyTim_f-4.fits
with focus position -4 will have the 
tt_file_name: TinyTim_focus.fits
focus=['f-4']

File with list of star postions in feild is stored at 
tt_files_path/filter_name/filter_name_stars.txt





Things to ask Bradley:
GalaxyCatalog: output_params, bright_config_dict, faint_config_dict, 
star_galaxy_weights, spike parameters, manual_mask_file


 """
import subprocess
import galsim
import numpy as np
import run_segment7 as rs

class Main_param:
    """Class containg parameters to pass to run analysis on each segment file."""
    def __init__(self,args):
        self.seg_id = args.seg_id
        self.file_name = args.file_name.replace('seg_id', self.seg_id)
        self.wht_name = args.wht_name.replace('seg_id', self.seg_id) 
        self.det_im_file = args.det_im_file.replace('seg_id', self.seg_id)
        self.det_wht_file = args.det_wht_file.replace('seg_id', self.seg_id)  
        self.filters = args.filter_names
        self.out_path = args.out_path
        self.tt_file_path = args.tt_file_path       
        self.focus = args.focus
        self.sf = args.sf
        self.wht_type = args.wht_type
        self.manual_mask_file = args.manual_mask_file
        ## making weight maps rms
        if args.file_path[-1] != '/':
            self.file_path = args.file_path+'/'
        else:
            self.file_path = args.file_path
        if args.wht_path[-1] != '/':
            self.wht_path = args.wht_path+'/'
        else:
            self.wht_path = args.wht_path
        self.spike_params,self.zero_point_mag= {}, {}
        self.gain, self.star_galaxy_weights   = {}, {}
        self.data_files, self.wht_files = {}, {}
        for i in range(len(self.filters)):
            filter1 = self.filters[i]
            self.data_files[filter1] = self.file_path + filter1 + '/' + self.file_name.replace('filter', filter1)
            self.wht_files[filter1] = self.wht_path + filter1 + '/' + self.wht_name.replace('filter',filter1)
            self.spike_params[filter1] = args.filter_spike_params[i] 
            self.zero_point_mag[filter1] = args.zero_point_mag[i]
            self.star_galaxy_weights[filter1] = args.star_galaxy_weights[i]
            self.gain[filter1] = args.gain[i]
        self.tt_file_name = {}
        for focus in self.focus:
            self.tt_file_name[focus] = args.tt_file_name.replace('focus', 'f'+str(focus))
        



def main2(args):
    print"RUN SEGEMNT4"
    params = Main_param(args)
    rs.run_segment(params)
    

if __name__ == '__main__':
    import subprocess
    import galsim
    import numpy as np
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument('--n_filters', type=int, default=2,
                        help="number of image filters [Default: 2]")
    parser.add_argument('--filter_names', default= ['f814w','f606w'],
                        help="names of filters [Default: ['f606w','f814w']]")
    parser.add_argument('--filter_spike_params', 
                        default= [(0.0367020,77.7674,40.0,2.180), (0.0350087,64.0863,40.0,2.614)],
                        help="Prams of diffraction spikes on filters. These have to in the same order as filter_names [Default: [(0.0350087,64.0863,40.0,2.614), (0.0367020,77.7674,40.0,2.180)]]")
    parser.add_argument('--star_galaxy_weights', 
                        default= [(18.9, 14.955, 0.98), (19.4, 15.508, 0.945)],
                        help="Star galaxy seperation line [Defalt:(x_div, y_div, slope)]")
    parser.add_argument('--zero_point_mag', 
                        default= (25.955, 26.508),
                        help="Zero point magnitides [Default:( 26.508, 25.955)]")
    parser.add_argument('--gain', default=[2260,2100],
                        help="Detector gain in e/ADU[Default:[2260,2100]")
    parser.add_argument('--sf', default=0.316 ,
                        help="Scale factor to correct for correlated noise[Default:0.316 ")
    parser.add_argument('--manual_mask_file', default= 'manual_masks.txt',
                        help="file containing regions that are to be masked [Default:'manual_masks.txt']")
    parser.add_argument('--file_path', default= '/nfs/slac/g/ki/ki19/deuce/AEGIS/unzip/',
                        help="Path of directory containing images[Default:'/nfs/slac/g/ki/ki19/deuce/AEGIS/unzip] ")
    parser.add_argument('--wht_path', default= '/nfs/slac/g/ki/ki19/deuce/AEGIS/unzip',
                        help="Path of directory containing weight files[Default:'/nfs/slac/g/ki/ki19/deuce/AEGIS/unzip] ")
    parser.add_argument('--out_path', default= '/nfs/slac/g/ki/ki19/deuce/AEGIS/output/',
                        help="Path to where you want the output store [Default: /nfs/slac/g/ki/ki19/deuce/AEGIS/output] ")
    parser.add_argument('--file_name', default='EGS_10134_seg_id_acs_wfc_filter_30mas_unrot_drz.fits',
                        help="File name of image with 'seg_id' in place in place of actual segment id [Default:'EGS_10134_seg_id_acs_wfc_f606w_30mas_unrot_drz.fits']")
    parser.add_argument('--wht_name', default='EGS_10134_seg_id_acs_wfc_filter_30mas_unrot_wht.fits',
                        help="Weight file name of image with 'seg_id' in place in place of actual segment id [Default:'EGS_10134_seg_id_acs_wfc_f606w_30mas_unrot_wht.fits']")  
    parser.add_argument('--det_im_file', default='/nfs/slac/g/ki/ki19/deuce/AEGIS/unzip/added/EGS_10134_seg_id_acs_wfc_30mas_unrot_added_drz.fits',
                        help="File name of image used in detction")
    parser.add_argument('--det_wht_file', default='/nfs/slac/g/ki/ki19/deuce/AEGIS/unzip/added/EGS_10134_seg_id_acs_wfc_30mas_unrot_added_rms.fits',
                        help="Weight file name of image of image used in detction")  
    parser.add_argument('--wht_type', default='MAP_WEIGHT',
                        help="Weight file type")
    parser.add_argument('--seg_id', default='1a',
                        help="Segment id to run [Default:1a]")
    parser.add_argument('--tt_file_path', default='/nfs/slac/g/ki/ki19/deuce/AEGIS/tt_starfield/',
                        help="Path of directory contating modelled TT fileds [Default:'/nfs/slac/g/ki/ki19/deuce/AEGIS/tt_starfield/'] ")
    parser.add_argument('--tt_file_name', default= 'TinyTim_focus.fits',
                        help="Name of TT_field file [Default:TinyTim_focus.fits]")
    parser.add_argument('--focus', default= [-10, -9, -8, -7, -6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5],
                        help="List containg focus positions that have TT_fields")
    args = parser.parse_args()

    main2(args)
        
# to do
# make list of all ids
