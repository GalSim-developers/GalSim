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

"""Makes postage stamps of galaxy, PSF and segmentation _comb_seg_map

Criterion for making postage stamps:
In ALL filters: 
* Is NOT a star
* Is NOT in any masked region
* Is NOT detected in other segments
* SNR >0
In ONE filter (The last filter in the input list)
* magnitude<= 25.2

Rectangular postage stamps are made for each selected galaxy. Size of pstamps
determined using eqn 2,3 Hausler 2007. XSIZE & YSIZE are individually 
computed for each band and the highest values are picked as dimensions of the 
postage stamps in both bands.

PSF pstamp size 20*20

PSF:
The focus of image is set to Mode of focus while varying number of stars


Output:
postage stamps of galaxy, PSF and segmentation _comb_seg_map, file with 
NUMBER of objects with postage stamps
"""
import galsim
import os
import numpy as np
import functions as fn
import get_objects as go
import subprocess
import pyfits
from astropy.table import Table, Column
from scipy import stats


def run(params, focus):
    # Remove existing postage stamp images, or creates new folder
    out_dir = params.out_path + params.seg_id+ '/'
    if os.path.isdir(out_dir + 'postage_stamps') is True:
            subprocess.call(["rm", '-r', out_dir + 'postage_stamps'])
            subprocess.call(["mkdir", out_dir + 'postage_stamps'])
    else:
        subprocess.call(["mkdir", out_dir + 'postage_stamps'])
    catalogs = []
    tt_files =[]
    #Open main catalog in all filters
    for filt in params.filters:
        cat_name = out_dir + '/' + filt + "_clean.cat"
        catalog = Table.read(cat_name, format="ascii.basic")
        #make new column to indicate if postamp is created for that object
        col= Column(np.zeros(len(catalog)),name='IS_PSTAMP',dtype='int',
                    description = 'created postage stamp' )
        catalog.add_column(col)
        catalogs.append(catalog)
        tt_file = params.tt_file_path + "/" + filt + "/{}_stars.txt".format(filt)
        tt_files.append(np.loadtxt(tt_file))
    # Get indices of galaxies higher than cut off SNR and not masked. 
    # ALso get their size in differnt filters  
    idx=[[],[],[],[]]    
    for i in range(len(catalogs[0])):
        x0 = catalogs[0]['X_IMAGE'][int(i)]
        y0 = catalogs[0]['Y_IMAGE'][int(i)]
        x_sizes = []
        y_sizes = []
        pos=[]
        # Select objects that satisfy criterion
        for f,filt in enumerate(params.filters):
            cond1 = (catalogs[f]['IS_STAR'][i] == 0)
            cond2 = (catalogs[f]['IN_MASK'][i] == 0)
            cond3 = (catalogs[f]['SNR'][i] >= 0)
            cond4 = (catalogs[f]['MULTI_DET'][i] == 0)
            #Placing magnitude cut on only last filter
            cond5 = (catalogs[-1]['MAG_CORR'][i] <= 25.2)
            if  cond1 and cond2 and cond3 and cond4 and cond5:
                t = (catalogs[f]['THETA_IMAGE'][int(i)])*np.pi/180.
                e = catalogs[f]['ELLIPTICITY'][int(i)]
                A = 2.5*(catalogs[f]['A_IMAGE'][int(i)])*(catalogs[f]['KRON_RADIUS'][int(i)])
                x_size = A*(np.absolute(np.sin(t))+(1-e)*np.absolute(np.cos(t)))
                y_size = A*(np.absolute(np.cos(t))+(1-e)*np.absolute(np.sin(t)))
                x_sizes.append(x_size)
                y_sizes.append(y_size)            
            else:
                break
            # get coordinates of nearest star in tt_starfeild
            tt_pos = fn.get_closest_tt(x0,y0,tt_files[f])
            if tt_pos:
                pos.append(tt_pos)
            else:
                break
            if f == len(params.filters)-1:
                idx[0].append(i)
                idx[1].append(x_sizes)
                idx[2].append(y_sizes)
                idx[3].append(pos)
    obj_ids = np.array(idx[0], dtype=int)
    # save list with NUMBER of all objects with pstamps
    np.savetxt(out_dir+'objects_with_p_stamps.txt', obj_ids, fmt="%i")
    #save catalogs 
    for f,filt in enumerate(params.filters):        
        # column to save focus
        col= Column(np.ones(len(catalog))*focus[filt], name='FOCUS',
                    dtype='int', description = 'Focus of image')
        catalogs[f].add_column(col)
        catalogs[f]['IS_PSTAMP'][obj_ids] = 1
        cat_name = out_dir + '/' + filt + "_full.cat"
        catalogs[f].write(cat_name, format="ascii.basic")
    #Get postage stamp image of the galaxy in all filters. 
    #Postage stamp size is set by the largest filter image  
    for num, i in enumerate(idx[0]):
        print "Saving postage stamp with object id:",i
        gal_images=[]
        psf_images=[]
        info={}
        x0 = catalogs[0]['X_IMAGE'][int(i)]
        y0 = catalogs[0]['Y_IMAGE'][int(i)]
        x_stamp_size = max(idx[1][num])
        y_stamp_size = max(idx[2][num])
        stamp_size =[int(y_stamp_size), int(x_stamp_size)]
        psf_stamp_size=[20,20]
        print "Stamp size of image:", stamp_size
        #import ipdb; ipdb.set_trace()
        gal_header = pyfits.Header()
        psf_header = pyfits.Header()
        temp = go.GalaxyCatalog(None)
        header_params = temp.output_params
        #import ipdb; ipdb.set_trace()
        for f, filt in enumerate(params.filters):
            tt_pos = idx[3][num][f]
            gal_file_name = out_dir + 'postage_stamps/' + filt + '_' + params.seg_id + '_' + str(i)+'_image.fits'
            psf_file_name = out_dir + 'postage_stamps/' + filt + '_' + params.seg_id + '_' + str(i)+'_psf.fits'
            seg_file_name = out_dir + 'postage_stamps/' + filt + '_' + params.seg_id + '_' + str(i)+'_seg.fits'
            gal_name = params.data_files[filt]
            gal_image = fn.get_subImage_pyfits(x0,y0, stamp_size, gal_name,
                                               None, None, save_img=False) 
            psf_name = params.tt_file_path + filt +'/'+ params.tt_file_name[focus[filt]]
            psf_image = fn.get_subImage_pyfits(tt_pos[0],tt_pos[1], psf_stamp_size,
                                               psf_name, None, None, save_img=False)
            seg_name = out_dir + filt +'_comb_seg_map.fits'
            seg_image = fn.get_subImage_pyfits(x0,y0, stamp_size, seg_name,
                                               None, None, save_img=False)
            for header_param in header_params:
                try:
                    gal_header[header_param] = catalogs[f][header_param][i]
                except:
                    gal_header[header_param] = 9999.99
                    
            psf_header['X'] = tt_pos[0]
            psf_header['Y'] = tt_pos[1]
            psf_header['width'] = psf_stamp_size[0]
            psf_header['height'] = psf_stamp_size[1]
            pyfits.writeto(gal_file_name,gal_image,gal_header, clobber=True)
            pyfits.writeto(psf_file_name,psf_image,psf_header, clobber=True)
            pyfits.writeto(seg_file_name,seg_image, clobber=True)
