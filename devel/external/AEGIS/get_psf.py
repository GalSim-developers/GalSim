""" Program Number: 3
Computes the focal length of the telescope for a given image, and uses that to
estimate the PSF. Postage stamps of select galaxies and their PSF are also drawn.(called
in get_pstamps.py)

Requirements: A catalog with sextractor output for the given tile(segment), List
of stars in the segemnt to be used for psf measuremnt, image of region, tt_starfeilds
which contain psf drawn at difernt focus distance, at various points on the chip
(size of tt_starfeild, must be same as input image),segmentation map of the
region. If upon visual inspection, some stars were found to be unsuitable for
psf measuremnt, then their seg id and number must be written in bad_stars.txt

PSF estimation:
The optimal focus and corresponding tt_starfeild in picked in get_pstamps. Here
we compute the focus for a given image while varying number of strs used to 
measure focus. The focus is given by the minimum of the cost function. The focus 
is is computed for different number of stars in the feild. The stars used for 
measuring focus are sorted in decreasing SNR. The mode of focus values for different
number of stars used in focus measurement is set as the focus value for the given
segmenty.   

Cost Function:
The magnitude of diffrence in ellipticities of stars and its corresponing tiny 
tim image. 

Output:
File with focus while varying nuumber of stars. 
 """
import subprocess
import galsim
import numpy as np
import functions as fn
import matplotlib.pyplot as plt
import get_pstamps as gps
from astropy.table import Table, Column
from scipy import stats

class Main_param:
    """Class containg parameters to pass to run analysis on each segment file."""
    def __init__(self,args, bad_stars):
        self.file_name = args.file_name.replace('seg_id', args.seg_id)
        self.seg_id = args.seg_id
        self.filters = args.filter_names
        self.out_path = args.out_path
        self.tt_file_path = args.tt_file_path       
        self.focus = args.focus
        self.bad_stars = bad_stars
        if args.file_path[-1] != '/':
            self.file_path = args.file_path+'/'
        else:
            self.file_path = args.file_path
        self.data_files = {}
        for i in range(len(self.filters)):
            filter1 = self.filters[i]
            self.data_files[filter1] = self.file_path + filter1 + '/' + self.file_name.replace('filter', filter1)
        self.tt_file_name = {}
        for focus in self.focus:
            self.tt_file_name[focus] = args.tt_file_name.replace('focus', 'f'+str(focus)) 

def get_bad_stars(args):
    """Returns list of stars that should be removed from list of stars 
    for PSF estimation"""
    bad_stars ={}
    b_s = np.loadtxt(args.bad_stars_file, dtype='S16')
    s = b_s.shape
    for i in range(0,s[0],2):
        bad_stars[b_s[i][0]]={}
        bad_stars[b_s[i][0]][b_s[i][1]] = b_s[i][2:]
        bad_stars[b_s[i][0]][b_s[i+1][1]] = b_s[i+1][2:]
    return bad_stars

def get_good_stars(params, filter, out_dir):
    """Removes bad strs from from list of stars for PSF estimation"""
    stars = (np.loadtxt(out_dir + filter+'_matched_stars.txt'))[0]
    idx=[]
    for b_s in params.bad_stars[filter]:
        if b_s != 'None':
            q,= np.where(stars == int(b_s))
            idx.append(q[0])     
    return np.delete(range(len(stars)),idx, axis=0)

def get_moments(params, good_stars,
                filter, out_dir):
    """Computes moments of good_stars, and moments for of tt_starfeilds at
    differnt Focus
    """
    print "Computing Moments"
    print filter, params.seg_id
    stars1 = np.loadtxt(out_dir + filter+'_matched_stars.txt').T
    moments = [[],[]]
    hsm_params =galsim.hsm.HSMParams(max_mom2_iter = 1000000000)
    fin_stars=[]
    for num,i in enumerate(good_stars):
        print "Getting moments of star ", int(stars1[i][0])
        x_s = stars1[i][1]
        y_s = stars1[i][2]
        r = stars1[i][3]
        x_t = stars1[i][4]
        y_t = stars1[i][5]
        star_file = params.data_files[filter]
        im_s = fn.get_subImage(x_s, y_s, int(r)*8, star_file,
                            out_dir, None, save_img=False)
        star_result = galsim.hsm.FindAdaptiveMom(im_s, hsmparams=hsm_params, strict=False)
        if star_result.error_message != "":
            print "Moments measurement failed for star{0}".format(i)
            continue
        tt_result = {}
        check = False
        for j, focus in enumerate(params.focus):
            print "Computing moments for focus ", focus
            tt_file = params.tt_file_path + filter+'/'+ params.tt_file_name[focus]
            im_t = fn.get_subImage(x_t, y_t, int(r)*8, tt_file,
                                out_dir, None, save_img=False)   
            result = galsim.hsm.FindAdaptiveMom(im_t, hsmparams=hsm_params, strict=False)
            if result.error_message != "" :
                check  = True
                print "Moments measurement failed for tt star{0} at focus{1}".format(i,focus)
                break      
            tt_result[focus] = result
        if check == False:
            fin_stars.append(i)
            moments[0].append(star_result)
            moments[1].append(tt_result)
    return moments, fin_stars

def calc_cost_fn(params, moments):
    print "Calculating cost function"
    cost_fn = np.zeros([len(params.focus),2])
    for i, focus in enumerate(params.focus):
        for j in range(len(moments[0])):
            e1_star = moments[0][j].observed_shape.getE1()
            e2_star = moments[0][j].observed_shape.getE2()
            e1_tt = moments[1][j][focus].observed_shape.getE1()
            e2_tt = moments[1][j][focus].observed_shape.getE2()
            cost_fn[i][1] += (e1_tt-e1_star)**2 + (e2_tt-e2_star)**2
        cost_fn[i][0] = focus
    return cost_fn


def calc_cost_fn_num(params, moments, num):
    """Compute cost function from moments of star num-all"""
    print "Calculating cost function"
    cost_fn = np.zeros([len(params.focus),2])
    for i, focus in enumerate(params.focus):
        for j in range(num, len(moments[0])):
            e1_star = moments[0][j].observed_shape.getE1()
            e2_star = moments[0][j].observed_shape.getE2()
            e1_tt = moments[1][j][focus].observed_shape.getE1()
            e2_tt = moments[1][j][focus].observed_shape.getE2()
            cost_fn[i][1] += (e1_tt-e1_star)**2 + (e2_tt-e2_star)**2
        cost_fn[i][0] = focus
    return cost_fn

def get_focus_num_stars(params):
    """Computes focus of image, while varying number of stars used to compute 
    Focus. Minimum of 3 strs are always used in measuring focus. The focus
    for diffrent numbers is saved to file. Cost function for focus measurmnts
    with all strs is also saved. stars picked go in decraesing SNR.i.e 3 stars 
    with highest SNR are always included in measurments.  
    """
    out_dir = params.out_path+ '/' + params.seg_id+ '/'
    for filter in params.filters:
        print "Running focus with different star number for filter:", filter
        good_stars = get_good_stars(params, filter, out_dir)
        print "Number of good stars:", len(good_stars)
        # compute moments for all stars 
        moments, final_stars = get_moments(params, good_stars, filter, out_dir)
        focus =  np.zeros([len(final_stars)-3,2])
        # compute focus from 3 - all stars
        for i,num in enumerate(range(3,len(final_stars))):               
            N = len(final_stars) - num - 1
            print "multi num stars ", N
            cost_fn = calc_cost_fn_num(params, moments, N) 
            focus[i][0] = num
            focus[i][1] = cost_fn.T[0][np.argmin(cost_fn.T[1])]
        np.savetxt(out_dir + filter+"_cost_fn.txt", cost_fn)
        np.savetxt(out_dir + filter+"_focus_with_num_stars.txt", focus)
        print focus.T

def get_psf(args):
    """Gets list of stars, if any, to be omitted from PSF estimation"""
    bad_stars = get_bad_stars(args)
    params = Main_param(args, bad_stars[args.seg_id])
    # Computes focus for diffrent number of stars
    get_focus_num_stars(params)
    out_dir = params.out_path+ '/' + params.seg_id+ '/'
    focus={}
    for f,filt in enumerate(params.filters):
        a = np.loadtxt(out_dir+filt+'_focus_with_num_stars.txt')
        focus[filt] = int(stats.mode(a.T[1]).mode[0])
        print "Focus for {0} is {1} ".format(filt, focus[filt])
    # save postage stamps
    print "Getting postage stamps"
    gps.run(params, focus)
            

if __name__ == '__main__':
    import subprocess
    import galsim
    import numpy as np
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument('--seg_id', default='1g',
                        help="Segment id of image to run [Default:1g]")
    parser.add_argument('--filter_names', default= ['f606w','f814w'],
                        help="names of filters [Default: ['f606w','f814w']]")
    parser.add_argument('--bad_stars_file', default= 'bad_stars.txt',
                        help="File containing index of strs that should not be  \
                        used in PSF estimation [Default:'bad_stars.txt'] ")
    parser.add_argument('--file_path', default= '/nfs/slac/g/ki/ki19/deuce/AEGIS/unzip/',
                        help="Path of directory containing input images \
                        [Default:'/nfs/slac/g/ki/ki19/deuce/AEGIS/unzip] ")
    parser.add_argument('--file_name', default='EGS_10134_seg_id_acs_wfc_filter_30mas_unrot_drz.fits',
                        help="File name of measurement image with 'seg_id' & \
                        'filter' in place of image segment id and filter  \
                        [Default:'EGS_10134_seg_id_acs_wfc_f606w_30mas_unrot_drz.fits']")
    parser.add_argument('--out_path', default= '/nfs/slac/g/ki/ki19/deuce/AEGIS/AEGIS_full/',
                        help="Path to where you want the output stored \
                        [Default: /nfs/slac/g/ki/ki19/deuce/AEGIS/AEGIS_full]")
    parser.add_argument('--tt_file_path', 
                        default='/nfs/slac/g/ki/ki19/deuce/AEGIS/tt_starfield/',
                        help="Path of directory contating modelled TT fileds \
                        [Default:'/nfs/slac/g/ki/ki19/deuce/AEGIS/tt_starfield/']")
    parser.add_argument('--tt_file_name', default= 'TinyTim_focus.fits',
                        help="TT_field file name  with 'focus' in place of actual focus \
                        [Default:TinyTim_focus.fits]")
    parser.add_argument('--focus',
                        default= [-10, -9, -8, -7, -6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5],
                        help="List containg focus positions that have TT_fields")
    args = parser.parse_args()
    get_psf(args)

