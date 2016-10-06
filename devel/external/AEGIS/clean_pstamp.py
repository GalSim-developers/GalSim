"""Program Number: 4
Identifies multiple objects in the postage stamp of a galaxy and replaces 
the other object with noise.

Requirements:
postage stamp of galaxy, corresponding segmentation map, noise file

Identification:
The segmentation map corresponding to the object is used to divide the
postage stamp pixels into tho belonging to the main galaxy, other 
objects and background. The pixels belonging to other objects are replaced
by pixels from the noise file. 

Replace pixels:
The other object pixels are replaced with a region of noise map identical
to the region of other object. This is done so as to preserve noise correlations.
The noise pixels values are divided by its standard deviation and multiplied
by the stdev of the background pixels of the postage stamp. 

Stamp Stats:
Some value are saved so that the GalSim COSMOSCatalog class may use to impose
selection criteria on the quality of the postage stamps. 
"""
from astropy.table import Table
import pyfits
import sys
import os

class Main_param:
    """Class containg parameters to pass to run analysis on each galaxy file."""
    def __init__(self,args):
        self.seg_id = args.seg_id
        self.num = args.num
        self.path = args.main_path + '/' + self.seg_id + '/postage_stamps/'
        self.filters = args.filter_names
        self.file_filter_name = args.file_filter_name
        string = args.main_string.replace('segid', self.seg_id) 
        string1 = string.replace('num', self.num)        
        self.gal_files , self.noise_file, self.seg_files = {}, {}, {}
        n = len(self.filters)      
        for i in range(n):
            filter1 = self.filters[i]
            string2 = string1.replace('filter', filter1)
            self.gal_files[filter1] = self.path + string2 + args.image_string
            self.seg_files[filter1] = self.path + string2 + args.seg_string
            string3 = args.noise_file.replace('filter',args.file_filter_name[i])
            self.noise_file[filter1] = args.main_path + '/' + string3

def div_pixels(seg_map, num):
    """Get pixels that belong to image, other objects, background from
    segmentation map
    """
    s = seg_map.shape 
    xs = range(s[0])           
    ys = range(s[1])
    bl=[]
    oth ={}
    oth_segs=[]
    im = []
    check = 0
    for x in xs:
        for y in ys:
            if seg_map[x,y] == 0 :
                #background pixel
                bl.append([x,y])
            elif seg_map[x,y] == int(num)+1 :
                #image pixel
                im.append([x,y])
            else :
                #other object
                if oth.has_key(str(seg_map[x,y])):
                    oth[str(seg_map[x,y])].append([x,y])
                else: 
                    oth[str(seg_map[x,y])] = [[x,y]]
                    oth_segs.append(str(seg_map[x,y]))
    if seg_map[s[0]/2, s[1]/2] != int(num)+1:
        check = 1
    return im, bl, oth, oth_segs, check

def get_stats(arr ,str=None):
    """Returns mean and stdev of arr """
    mean = np.mean(arr)
    t2 = arr - mean
    std = np.std(arr)
    t3 = np.sort(arr)
    if str:
        print 'Measuring', str
    print 'STATS: mean= ',mean, ' stdev=', std
    return mean, std

def get_avg_around_pix(x0, y0,arr):
    """Returns average values of pixels around (x0,y0) in arr"""
    x, y = [x0], [y0]
    if x0>0:
        x.append(x0-1)
    if arr.shape[0]-1>x0:
        x.append(x0+1)
    if y0>0:
        y.append(y0-1)
    if arr.shape[1]-1>y0:
        y.append(y0+1)
    neighb = [arr[i][j] for i in x for j in y]
    avg=sum(neighb)/len(neighb)
    return avg

def get_snr(image_data, b_var):
    """Returns SNr of shape measurement"""
    img = galsim.Image(image_data)
    try:
        res = galsim.hsm.FindAdaptiveMom(img)
        aperture_noise = float(np.sqrt(b_var*2.*np.pi*(res.moments_sigma**2)))
        sn_ellip_gauss = res.moments_amp / aperture_noise
        print 'RES', res.moments_amp, res.moments_sigma
        print 'SNR', sn_ellip_gauss
    except:
        print 'SNR manually set'
        sn_ellip_gauss = -10.
    print 'SNR', sn_ellip_gauss
    return sn_ellip_gauss

def get_min_dist(x0,y0,arr):
    """Returns minimum distance between points in arr and (x0,y0) """
    dist = np.hypot(arr.T[0]-x0, arr.T[1]-y0)
    min_dist = np.min(dist)
    val = np.argmin(dist)
    return min_dist, arr[val]

def get_blank_reg(x_r, y_r, noise_file):
    """Returns reactangular randomly picked region of size x_r * y_r from the 
    noise_file"""
    hdu = pyfits.open(noise_file)
    bl_dat = hdu[0].data
    hdu.close()
    s = bl_dat.shape
    print "x_r", x_r, 's', s
    x0_min = np.random.randint(s[0]-x_r)
    y0_min = np.random.randint(s[1]-y_r)
    x0_max = x0_min + x_r +1
    y0_max = y0_min + y_r +1
    empty = bl_dat[x0_min:x0_max, y0_min:y0_max]
    bl_mean, bl_std = get_stats(bl_dat ,str='Blank region from file')
    return empty, bl_std

def change_others(arr, to_change,
                  noise_file, b_std):
    """Change pixels of other object to background
    @ arr         Postage stamp image of galaxy
    @ to_change   coordinates of pixels that be replaced with noise
    @ noise_file  File with noise pixels
    @ b_std       Std dev of background pixels of pstamp
    """
    xmin, xmax = np.min(to_change.T[0]), np.max(to_change.T[0])
    ymin, ymax = np.min(to_change.T[1]), np.max(to_change.T[1])            
    xr0 = xmax-xmin
    yr0 = ymax-ymin 
    # get noise pixels in a rectangle of size comparable to that which needs replaced  
    bl_dat, bl_std = get_blank_reg(xr0, yr0, noise_file)
    # Change coords of pixels to change to satrt with (0,0)
    bl_change = np.array([to_change.T[0]-xmin, to_change.T[1]-ymin]).T
    # get noise pixel with same coordinates as to change pixels
    bl_pixels = [bl_dat[bl_change[i,0], bl_change[i,1]] for i in range(len(bl_change))]
    bl_dat = bl_dat/bl_std*b_std    
    ### change pixels of oth in arr to blank value
    for p in range(len(to_change)):
        arr[to_change[p][0],to_change[p][1]] = bl_dat[bl_change[p][0],bl_change[p][1]]
    return arr


def clean_pstamp(args):
    """If a postage stamp has object other the central galaxy, the other object
    is replaced by noise. The postage stamp of image must all have a 
    segmentation map. A value other than the id number of the image+1 is 
    replaced in the segmentation map is detected as other object. Other object
    pixels in the image postage stamp is replaced by pixels from a noise file.
    the input noise file should represent noise background expected in the 
    image. The replaced noise pixels are normalized by the standard deviation
    of background pixels in the postage stamp image
    Output: Creates new postage stamp with only the central object, a file
    with info on pixels that were changed and backgroound.
    """
    params = Main_param(args)
    for i, filt in enumerate(params.filters):
        print "Running filter", filt
        if os.path.isdir(params.path + 'stamp_stats') is False:
            subprocess.call(["mkdir", params.path + 'stamp_stats'])
        # open image and seg map
        hdu1 = pyfits.open(params.gal_files[filt])
        hdu2 = pyfits.open(params.seg_files[filt])
        im_dat = hdu1[0].data
        im_hdr = hdu1[0].header
        seg_dat = hdu2[0].data
        hdu1.close()
        hdu2.close()
        shape = im_dat.shape       
        x0, y0= shape[0]/2, shape[1]/2
        # classify pixels as belonging to image, other objects and background
        # using segmentation map
        im, bl, oth, oth_segs, check = div_pixels(seg_dat, params.num)
        # Some bright object is nearby, and it's seg map overlaps with central object
        # manually set output values so it fails selection tests later
        if len(im)==0:
            print "Ignore object"
            peak_val = 0
            min_dist = 0.
            avg_flux = 999.99
            snr = -10.
            info = [0, 0, 0 , min_dist, avg_flux, peak_val, snr]
            np.savetxt(params.path + 'stamp_stats'+'/'+ params.num + '_'+ filt + '.txt', info)
            new_im_name = params.path + filt + '_'+ params.seg_id + '_'+ params.num +'_gal.fits'
            pyfits.writeto(new_im_name, im_dat, im_hdr, clobber=True)
            continue
        # Objects seg map covers entire pstamp, no blank region
        # manually set output values so it fails selection tests later
        if (len(bl)<=1):
            print "Ignore object"
            peak_val = 0
            min_dist = 0.
            avg_flux = 999.99
            snr = -10.
            info = [0, 0, 0 , min_dist, avg_flux, peak_val, snr]
            np.savetxt(params.path + 'stamp_stats'+'/'+ params.num + '_'+ filt + '.txt', info)
            new_im_name = params.path + filt + '_'+ params.seg_id + '_'+ params.num +'_gal.fits'
            pyfits.writeto(new_im_name, im_dat, im_hdr, clobber=True)
            continue
        peak_val = np.max([[im_dat[im[i][0]][im[i][1]]] for i in range(len(im))])
        bck_pixels = [im_dat[bl[i][0], bl[i][1]] for i in range(len(bl))]
        b_mean, b_std = get_stats(np.array(bck_pixels), str='Image Background')
        # No other object present                
        if len(oth_segs)==0 :
            print "No other object"
            print len(bl)
            min_dist = 999.99      
            pix_near_dist = [shape[0]/2, shape[1]/2]
            avg_flux = 0.
            snr = get_snr(im_dat, b_std**2)
            info = [b_mean, b_std, np.sum(im_dat), min_dist, avg_flux, peak_val, snr ]
            print info
            np.savetxt(params.path + 'stamp_stats'+'/'+ params.num + '_'+ filt + '.txt', info)
            new_im_name = params.path + filt + '_'+ params.seg_id + '_'+ params.num +'_gal.fits'
            pyfits.writeto(new_im_name, im_dat, im_hdr, clobber=True)
            continue
        new_im = im_dat.copy()
        min_dists = []
        pix_min_dists = []
        for oth_seg in oth_segs:
            print "Other obejct detected with id ", oth_seg
            print 'MASKING: ', len(oth[oth_seg]) , ' pixels out of ', seg_dat.size 
            print " Noise file used ", params.noise_file
            dist, pix = get_min_dist(x0,y0, np.array(oth[oth_seg]))
            noise_file = params.noise_file[filt]
            new_im = change_others(new_im, np.array(oth[oth_seg]), noise_file, b_std)
            min_dists.append(dist)
            pix_min_dists.append(pix)           
        min_dist = np.min(min_dists)
        pix_near_dist = pix_min_dists[np.argmin(min_dists)]
        avg_flux = get_avg_around_pix(pix_near_dist[0], pix_near_dist[1], im_dat)
        snr = get_snr(new_im, b_std**2)
        info = [b_mean, b_std, np.sum(im_dat), min_dist, avg_flux, peak_val, snr]
        np.savetxt(params.path + 'stamp_stats'+'/'+ params.num + '_'+ filt + '.txt', info)
        new_im_name = params.path + filt + '_'+ params.seg_id + '_'+ params.num +'_gal.fits'
        print 'CREATED NEW POSTAGE STAMP', new_im_name
        pyfits.writeto(new_im_name, new_im, im_hdr, clobber=True)

if __name__ == '__main__':
    import subprocess
    import galsim
    import numpy as np
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument('--seg_id', default='1a',
                        help="Segment id of image to run [Default:1a]")
    parser.add_argument('--num', default='0',type=str,
                        help="Identifier of galaxy to run [Default:0")
    parser.add_argument('--filter_names', default= ['f606w','f814w'],
                        help="names of filters [Default: ['f814w','f606w']]")
    parser.add_argument('--noise_file', type=str, default = 'acs_filter_unrot_sci_noise.fits' ,
                        help="File containing noise in each band, with band name \
                        replaced by'filter'[Default:'acs_filter_unrot_sci_noise.fits']] ")
    parser.add_argument('--file_filter_name', default =['V', 'I'] ,
                        help="Name of filter to use ")
    parser.add_argument('--main_path', 
                        default= '/nfs/slac/g/ki/ki19/deuce/AEGIS/AEGIS_full/',
                        help="Path where image files are stored \
                        [Default:'/nfs/slac/g/ki/ki19/deuce/AEGIS/AEGIS_full/'] ")
    parser.add_argument('--main_string', default='filter_segid_num_',
                        help="String of file name with 'ident','segid','filter' \
                        instead[Default:'ident_segid_filter_']")
    parser.add_argument('--image_string', default='image.fits',
                        help="String of saved galaxy image file [Default:'image.fits']")
    parser.add_argument('--seg_string', default='seg.fits',
                        help="String of saved  segmentation map file[Default:'seg.fits']")
    parser.add_argument('--pixel_scale', default='0.03',
                        help="Pixel scale of galaxy image[Default:'0.03' #arcsec/pixel]")
    args = parser.parse_args()
    clean_pstamp(args)