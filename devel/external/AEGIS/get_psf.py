# modified get_psf.py. 
""" Run this after best stars are picked for all segments and filters.
Manually inspect all strs to be picked for focus estimation
stars saved at output/seg_id/stars
write the id of bad stars in file bad_stars3.txt
Get focus


mdified fom main.py

 """
import subprocess
import galsim
import numpy as np
import functions as fn
import matplotlib.pyplot as plt
import get_postage_stamps6 as gps

class Main_param:
    """Class containg parameters to pass to run analysis on each segment file."""
    def __init__(self,args, bad_stars):
        self.file_name = args.file_name.replace('seg_id', args.seg_id)
        self.wht_name = args.wht_name.replace('seg_id', args.seg_id)
        self.seg_id = args.seg_id
        self.filters = args.filter_names
        self.out_path = args.out_path
        self.tt_file_path = args.tt_file_path       
        self.focus = args.focus
        self.star_galaxy_weights = args.star_galaxy_weights
        self.bad_stars = bad_stars
        if args.file_path[-1] != '/':
            self.file_path = args.file_path+'/'
        else:
            self.file_path = args.file_path
        if args.wht_path[-1] != '/':
            self.wht_path = args.wht_path+'/'
        else:
            self.wht_path = args.wht_path
        self.spike_params = {}
        self.data_files, self.wht_files = {}, {}
        for i in range(len(self.filters)):
            filter1 = self.filters[i]
            self.data_files[filter1] = self.file_path + filter1 + '/' + self.file_name.replace('filter', filter1)
            self.wht_files[filter1] = self.wht_path + filter1 + '/' + self.wht_name.replace('filter',filter1)
            self.spike_params[filter1] = args.filter_spike_params[i] 
        self.tt_file_name = {}
        for focus in self.focus:
            self.tt_file_name[focus] = args.tt_file_name.replace('focus', 'f'+str(focus)) 
        


def get_bad_stars(args):
    bad_stars ={}
    b_s = np.loadtxt(args.bad_stars_file, dtype='S16')
    s = b_s.shape
    #if s[0] != len(args.seg_ids)*len(args.filter_names):
    #    raise AttributeError('Every segment and filter must have 1 entry each in bad_stars')
    for i in range(0,s[0],2):
        bad_stars[b_s[i][0]]={}
        bad_stars[b_s[i][0]][b_s[i][1]] = b_s[i][2:]
        bad_stars[b_s[i][0]][b_s[i+1][1]] = b_s[i+1][2:]
    #print bad_stars
    return bad_stars

def get_good_stars(params, filter, out_dir):
    # Get only index of selected stars
    #print np.loadtxt(out_dir + filter+'_matched_stars.txt')
    stars = (np.loadtxt(out_dir + filter+'_matched_stars.txt'))[0]
    #print stars
    #print params.bad_stars[filter]
    idx=[]
    for b_s in params.bad_stars[filter]:
        if b_s != 'None':
            q,= np.where(stars == int(b_s))
            idx.append(q[0])     
    return np.delete(range(len(stars)),idx, axis=0)


def get_moments(params, good_stars,
                filter, out_dir):
    print "Computing Moments"
    print filter, params.seg_id
    stars1 = np.loadtxt(out_dir + filter+'_matched_stars.txt').T
    moments = [[],[]]
    hsm_params =galsim.hsm.HSMParams(max_mom2_iter = 1000000)
    for num,i in enumerate(good_stars):
        print "Getting moments of star ", int(stars1[i][0])
        x_s = stars1[i][1]
        y_s = stars1[i][2]
        r = stars1[i][3]
        x_t = stars1[i][4]
        y_t = stars1[i][5]
        star_file = params.data_files[filter]
        im_s = fn.get_subImage(x_s, y_s, int(r)*6, star_file,
                            out_dir, None, save_img=False)
        moments[0].append(galsim.hsm.FindAdaptiveMom(im_s, hsmparams=hsm_params))
        moments[1].append({})
        for i, focus in enumerate(params.focus):
            tt_file = params.tt_file_path + filter+'/'+ params.tt_file_name[focus]
            im_t = fn.get_subImage(x_t, y_t, int(r)*6, tt_file,
                                out_dir, None, save_img=False)          
            moments[1][num][focus] = galsim.hsm.FindAdaptiveMom(im_t, hsmparams=hsm_params)
    return moments



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

def plot_star_model(params, focus, good_stars, 
                    out_dir, filter):
    star_dir = out_dir + filter +"good_stars"
    stars1 = np.loadtxt(out_dir + filter+'_matched_stars.txt')
    moments = [[],[]]
    for num,i in enumerate(good_stars):
        x_s = stars1[i][1]
        y_s = stars1[i][2]
        r = stars1[i][3]
        x_t = stars1[i][4]
        y_t = stars1[i][5]
        star_file = params.data_files[filter]
        tt_file = params.tt_file_path + filter+'/'+ params.tt_file_name[focus]
        im_t = fn.get_subImage(x_t, y_t, int(r)*6, tt_file,
                                out_dir, None, save_img=False)  
        im_s = fn.get_subImage(x_s, y_s, int(r)*6, star_file,
                            out_dir, None, save_img=False)
        plt.figure(figsize=[30,20])
        plt.subplot(3,3,1)
        plt.imshow(im_s.array)
        plt.colorbar()
        plt.tile('Star')
        plt.subplot(3,3,1)
        plt.imshow(im_t.array)
        plt.colorbar()
        plt.tile('Model')
        plt.subplot(3,3,1)
        plt.imshow(im_s.array-im_t.array)
        plt.colorbar()
        plt.tile('Star-Model')        
        try:
            plt.savefig(star_dir+ '/'+ str(star_id)+'.png', bbox_inches='tight')
        except:
            subprocess.call(["mkdir", star_dir])
            plt.savefig(star_dir+ '/'+ str(star_id)+'.png', bbox_inches='tight')

def get_focus(params):
    """Return focus value (minimum of cost fn) for each filter """
    out_dir = params.out_path+ '/' + params.seg_id+ '/'
    focus = {}
    for filter in params.filters:
        good_stars = get_good_stars(params, filter, out_dir)
        np.savetxt( out_dir + filter+'_good_stars.txt', good_stars)
        moments = get_moments(params, good_stars, filter, out_dir)
        cost_fn = calc_cost_fn(params, moments)
        focus[filter] =  cost_fn.T[0][np.argmin(cost_fn.T[1])]
        np.savetxt(out_dir + filter +'_cost_fn.txt', cost_fn)
        print " Focus for seg:{0} in filter :{1} is {2}".format(params.seg_id, filter, focus[filter])
    return focus



def plot_focus_num_stars(params):
    out_dir = params.out_path+ '/' + params.seg_id+ '/'
    for filter in params.filters:
        print "Running focus with different star number for filter:", filter
        good_stars = get_good_stars(params, filter, out_dir)
        print "Number of good stars:", len(good_stars) 
        moments = get_moments(params, good_stars, filter, out_dir)
        focus =  np.zeros([len(good_stars)-5,2])
        for i,num in enumerate(range(5,len(good_stars))):               
            N = len(good_stars) - num - 1
            print "multi num stars ", N
            cost_fn = calc_cost_fn_num(params, moments, N) 
            focus[i][0] = num
            focus[i][1] = cost_fn.T[0][np.argmin(cost_fn.T[1])]
        np.savetxt(out_dir + filter+"_cost_fn.txt", cost_fn)
        np.savetxt(out_dir + filter+"_focus_with_num_stars.txt", focus)
        print focus.T
        #print focus.shape
        #print type(focus.T[0][1])
        #print type(focus.T[1][1])
        #plt.figure(figsize=[10,10])
        #plt.scatter(focus.T[0], focus.T[1])
        #plt.xlabel('Number of stars')
        #plt.ylabel('Focus')
        #plt.title('Variation of focus with number of stars used ({0})'.format(filter))
        #plt.savefig(filter+'_focus_num_stars1.png')








def get_psf(args):
    bad_stars = get_bad_stars(args)
    params = Main_param(args, bad_stars[args.seg_id])
    plot_focus_num_stars(params)
    focus = get_focus(params)
    #print "Getting postage stamps"
    gps.run(params)
            

            



            #rs.run_segment(params)



if __name__ == '__main__':
    import subprocess
    import galsim
    import numpy as np
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument('--n_filters', type=int, default=2,
                        help="number of image filters [Default: 2]")
    parser.add_argument('--filter_names', default= ['f606w','f814w'],
                        help="names of filters [Default: ['f606w','f814w']]")
    parser.add_argument('--filter_spike_params', 
                        default= [(0.0350087,64.0863,40.0,2.614), (0.0367020,77.7674,40.0,2.180)],
                        help="Prams of diffraction spikes on filters. These have to in the same order as filter_names [Default: [(0.0350087,64.0863,40.0,2.614), (0.0367020,77.7674,40.0,2.180)]]")
    parser.add_argument('--star_galaxy_weights', 
                        default= (19.0, -9.8, 0.9, -26.9),
                        help="(x_div, y_div, slope, intercept)")
    parser.add_argument('--bad_stars_file', default= '/nfs/slac/g/ki/ki19/deuce/AEGIS/output/bad_stars6.txt',
                        help="Path of file containing bad stars[Default:'/nfs/slac/g/ki/ki19/deuce/AEGIS/unzip] ")
    parser.add_argument('--file_path', default= '/nfs/slac/g/ki/ki19/deuce/AEGIS/unzip/',
                        help="Path of directory containing images[Default:'/nfs/slac/g/ki/ki19/deuce/AEGIS/unzip] ")
    parser.add_argument('--wht_path', default= '/nfs/slac/g/ki/ki19/deuce/AEGIS/unzip',
                        help="Path of directory containing weight files[Default:'/nfs/slac/g/ki/ki19/deuce/AEGIS/unzip] ")
    parser.add_argument('--out_path', default= '/nfs/slac/g/ki/ki19/deuce/AEGIS/output/',
                        help="Path to where you want the output store [Default: /nfs/slac/g/ki/ki19/deuce/AEGIS/output] ")
    parser.add_argument('--file_name', default='EGS_10134_seg_id_acs_wfc_filter_30mas_unrot_drz.fits',
                        help="File name of image with 'seg_id' in place in place of actual segment id [Default:'EGS_10134_seg_id_acs_wfc_f606w_30mas_unrot_drz.fits']")
    parser.add_argument('--wht_name', default='EGS_10134_seg_id_acs_wfc_filter_30mas_unrot_wht.fits',
                        help="Background file name of image with 'seg_id' in place in place of actual segment id [Default:'EGS_10134_seg_id_acs_wfc_f606w_30mas_unrot_wht.fits']")  
    parser.add_argument('--seg_id', default='1a',
                        help="List containing Segment ids to run [Default:'1a']")
    parser.add_argument('--tt_file_path', default='/nfs/slac/g/ki/ki19/deuce/AEGIS/tt_starfield/',
                        help="Path of directory contating modelled TT fileds [Default:'/nfs/slac/g/ki/ki19/deuce/AEGIS/tt_starfield/'] ")
    parser.add_argument('--tt_file_name', default= 'TinyTim_focus.fits',
                        help="Name of TT_field file [Default:TinyTim_focus.fits]")
    parser.add_argument('--focus', default= [-10, -9, -8, -7, -6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5],
                        help="List containg focus positions that have TT_fields")

    parser.add_argument('--run_all', help='Enter yes to run all files')
    args = parser.parse_args()

    get_psf(args)
        
# to do
# make list of all ids
