
## run_segment2.py modified . using astropy table instaed of asciidata

### MU_MAX                   Peak surface brightness above background    
## zero mag added to sextractor              [mag * arcsec**(-2)]
##  weight map converted to rms
####  detction done on 2 filter added image
#import find_objects as find
#import pandas as pd
import asciidata
from astropy.table import Table, Column, vstack
import subprocess
import pyfits
import os
import galsim
import numpy as np
import functions as fn
#import get_focus as focus

#import ipdb; ipdb.set_trace() 
### manual mask cleanup added

def run_segment(params):
    """Find objects in individual segments.
        Detecting objects:
        Run sextractor on file 1. O/p detected objects stroed in .cat file
        Parametrs from sextractor results taht are saved in .cat file is in .param
        Input for sextractor is in .config
        Sextractor is run with bright_config_dict sex parameters on the first 
        filter and detected objects saved in .*_bright.cat
        Sextractor in then run in dual image mode with detection in first filter
        and measurement in other filter images to give their corresponding 
        *_brigt.cat detected images

        Sextractor is run with faint_config_dict sex parameters on the first 
        filter and detected objects saved in .*_faint.cat
        Sextractor in then run in dual image mode with detection in first filter
        and measurement in other filter images to give their corresponding 
        *_faint.cat detected images

        Segmentation map is made for each of the _bright.cat files 
        where regions within 20 pixels of bright detections are masked out 
        for the faint detections.

        Filter faint catalog:  Remove objects in *_faint.cat that are 
        masked in the segmentation map for all filters. O/p is 
        *_filteredfaint.cat

        Merge the bright and filtered faint catologs to give catalog of 
        all objects



        Manual Mask?????????

        """
    #import ipdb; ipdb.set_trace() 
    cat = GalaxyCatalog(params)
    cat.generate_catalog()

    

class GalaxyCatalog:
    
    output_params = ["NUMBER",
    "X_IMAGE",
    "Y_IMAGE",
    "A_IMAGE",
    "B_IMAGE",
    "ALPHA_SKY",
    "DELTA_SKY",
    "XMIN_IMAGE",
    "XMAX_IMAGE",
    "YMIN_IMAGE",
    "YMAX_IMAGE",
    "FLAGS",
    "MU_MAX",
    "MAG_AUTO",
    "CLASS_STAR",
    "FLUX_RADIUS",
    "FLUX_AUTO",
    "FLUXERR_AUTO",
    "KRON_RADIUS",
    "THETA_IMAGE",
    "ELLIPTICITY"]

    bright_config_dict = { 'DETECT_MINAREA' : 140 ,
    'DETECT_THRESH' : 2.2 ,
    'DEBLEND_NTHRESH' : 64 ,
    'DEBLEND_MINCONT' : 0.04 ,
    'CLEAN_PARAM' : 1.0 ,
    'BACK_SIZE' : 400 ,
    'BACK_FILTERSIZE' : 5 ,
    'BACKPHOTO_TYPE' : "LOCAL" ,
    'BACKPHOTO_THICK' : 200,
    'PIXEL_SCALE' : 0.03}

    faint_config_dict = { 'DETECT_MINAREA' : 18 ,
    'DETECT_THRESH' : 1.0 ,
    'DEBLEND_NTHRESH' : 64 ,
    'DEBLEND_MINCONT' : 0.065 ,
    'CLEAN_PARAM' : 1.0 ,
    'BACK_SIZE' : 100 ,
    'BACK_FILTERSIZE' : 3 ,
    'BACKPHOTO_TYPE' : "LOCAL" ,
    'BACKPHOTO_THICK' : 200,
    'PIXEL_SCALE' : 0.03}            
    
    
    def __init__(self,params):
        self.params = params


    def get_sex_op_params(self, out_dir):
        ##### Saves the names of parameters thet sextractor must save
        param_fname = 'sex_out.param'
        try:
            param_file=open(out_dir+ '/'+ param_fname, 'w')
        except:
            subprocess.call(["mkdir", out_dir])
            param_file=open(out_dir+ '/'+ param_fname, 'w')
        print out_dir
        for i in range(len(self.output_params)):
            param_file.write(self.output_params[i])
            param_file.write("\n")
        param_file.close()
        

    def run_sextractor_dual(self, data_files, wht_files,
                            use_dict,out_dir, out_name, filter):
        """ Runs sextractor in dual mode"""
        print "Running dual : ", out_name
        #Create config newfiles[i] and write out to a file
        config_ascii = asciidata.create(2,8+len(use_dict))
        #File-Specific Configurations
        config_ascii[0][0] = 'CATALOG_NAME'
        config_ascii[1][0] = out_dir +'/'+ out_name + ".cat"
        config_ascii[0][1] = 'PARAMETERS_NAME'
        config_ascii[1][1] = out_dir+"/sex_out.param"
        config_ascii[0][2] = 'WEIGHT_TYPE'
        config_ascii[1][2] = self.params.wht_type   #### check if it should ne MAP_RMS
        config_ascii[0][3] = 'WEIGHT_IMAGE'
        config_ascii[1][3] = str(wht_files[0] + ','+ wht_files[1])
        row_counter = 4
        for key, value in use_dict.iteritems():
            config_ascii[0][row_counter] = key
            config_ascii[1][row_counter] = value
            row_counter += 1
        config_ascii[0][row_counter] = 'CHECKIMAGE_NAME'
        config_ascii[1][row_counter] = out_dir+'/'+ out_name + "_seg_map.fits"
        config_ascii[0][row_counter+1] = 'CHECKIMAGE_TYPE'
        config_ascii[1][row_counter+1] = 'SEGMENTATION'
        config_ascii[0][row_counter+2] = 'MAG_ZEROPOINT'
        config_ascii[1][row_counter+2] = self.params.zero_point_mag[filter]
        config_ascii[0][row_counter+3] = 'GAIN'
        config_ascii[1][row_counter+3] = self.params.gain[filter]
        config_fname =  out_dir+ '/'+ out_name + ".config"
        #config_ascii.writeto(config_fname)
        config_ascii.writeto(config_fname)
                
        #Run sextractor and get the catalog
        subprocess.call(["sex", data_files[0],",",data_files[1] , "-c", config_fname])

    def add_bright_faint_column(self, cat_name,tag):
        catalog = Table.read(cat_name, format="ascii.sextractor")
        col= Column(np.ones(len(catalog))*tag,name='IS_BRIGHT',dtype='int', description = 'Detected in hot mode' )
        catalog.add_column(col)
        return catalog

    def make_new_seg(self, seg_map, out_dir, out_name):
        data = pyfits.open(seg_map)[0].data
        new_seg = fn.seg_expand(data, buff=20)
        new_name = out_dir + '/' + out_name+ "_bright_seg_map_new.fits"
        if os.path.isfile(new_name) is True:
                subprocess.call(["rm", new_name])
        pyfits.writeto(new_name,new_seg)

    def filter_cat_with_segmentation_map(self, faint_catalog, seg_map,
                                         out_dir, out_name):
        """Remove objects from the faint catalog that lie in the 
        segemntation map. This is to remove faint objects very close to the
        bright objects
        """
        print "Removing faint objects from segmentationn map for section", self.params.seg_id
        segmentation_file = pyfits.open(seg_map)
        data = segmentation_file[0].data  
        val = [i for i in range(len(faint_catalog)) if (data[int(faint_catalog['Y_IMAGE'][i]),int(faint_catalog['X_IMAGE'][i])] == 0)]
        new_catalog = faint_catalog[val]
        name = out_dir + '/' +out_name+ "_filteredfaint.cat"
        new_catalog.write(name, format="ascii.basic")

    def merge(self, filtered_faint_catalog, 
              bright_catalog, out_name, out_dir):
        """ Merge objects detected in bright and filtered faint catalog """
        #Copy the header to a new file
        print "Merging bright  and faint catalogs for section", self.params.seg_id
        name = out_dir + '/'+ out_name + "_merge.cat"
        faint_cat = Table.read(filtered_faint_catalog, format="ascii.basic" )
        bright_cat = Table.read(bright_catalog, format="ascii.basic" )
        comb_cat = vstack([bright_cat, faint_cat])
        comb_cat.write(name, format="ascii.basic")


    def classification(self, div_params, out_dir):
        """Detected objects are classified as stars or not depending on thier 
        where they lie in the magnitude, peak surface brightness plot """     
        x_max = 25
        print "Performing star galaxy seperation for section", self.params.seg_id
        for filter in self.params.filters:
            x_div = div_params[filter][0]
            y_div = div_params[filter][1]
            slope = div_params[filter][2]
            intercept = y_div - slope*x_div
            out_name = filter
            merged_catalog = out_dir + '/' +out_name+ "_merge.cat"
            catalog = Table.read(merged_catalog, format="ascii.basic")
            snr = np.array(catalog['FLUX_AUTO'])/np.array(catalog['FLUXERR_AUTO'])
            col= Column(snr,name='SNR',description = 'Signal to Noise Ratio')
            catalog.add_column(col)
            # modified snr
            A = catalog['FLUXERR_AUTO']**2 - catalog['FLUX_AUTO']/self.params.gain[filter]
            new_f_err = (A/self.params.sf + catalog['FLUX_AUTO']/self.params.gain[filter])**0.5 
            col= Column(new_f_err,name='NEW_FLUXERR_AUTO',description = 'Modified FLUXERR_AUTO')
            catalog.add_column(col)
            new_snr = np.array(catalog['FLUX_AUTO'])/new_f_err
            col= Column(new_snr,name='NEW_SNR',description = 'Modified Signal to Noise Ratio')
            catalog.add_column(col)
            col= Column(np.zeros(len(catalog)),name='IS_STAR',dtype='int')
            catalog.add_column(col)
            q = fn.is_below_boundary_table(catalog['MAG_AUTO'], catalog['MU_MAX'],
                                           x_div, y_div, slope, intercept, x_max)
            catalog['IS_STAR'][q] = 1
            col= Column(np.zeros(len(catalog)),name='IS_FAKE',dtype='int')
            catalog.add_column(col)
            q = fn.is_below_boundary_table(catalog['MAG_AUTO'], catalog['MU_MAX'],
                                           x_div+1, y_div-2, slope, intercept-2, x_max+1)
            catalog['IS_FAKE'][q] = 1
            catalog.write(out_dir + '/' + out_name + "_class.cat",format="ascii.basic")

    def remove_edge(self, catalog):
        """Remove objects lying at the edge"""
        print "Removing edge objects"
        A = (390.,321.)
        B = (498.,6725.)
        C = (6898.,7287.)
        D = (7002.,806.)
        x_min = catalog['XMIN_IMAGE']
        x_max = catalog['XMAX_IMAGE']
        y_min = catalog['YMIN_IMAGE']
        y_max = catalog['YMAX_IMAGE']
        val = np.zeros(len(catalog))
        val[fn.lies_within_table(x_min, x_max, y_min, y_max, A,B,C,D)] = 1
        col = Column(val, name='IN_BOUNDARY', 
                     description="Inside a masked region", dtype=int)
        catalog.add_column(col)
        #Check this
        return catalog

#########Check#####
    def diffraction_mask_cleanup(self, catalog,
                                 diff_spike_params,  mag_cutoff = 19.0):
        m = diff_spike_params[0] 
        b = diff_spike_params[1]
        w = diff_spike_params[2]*0.5
        theta = diff_spike_params[3]
        x_vertex_sets = []
        y_vertex_sets = []
        print "Identifying diffraction spikes"
        val=np.zeros(len(catalog))

        col = Column(val, name='IN_DIFF_MASK', 
                 description="Close to saturated star", dtype=int)
        catalog.add_column(col)
        cond1 = catalog['MAG_AUTO']  < mag_cutoff
        cond2 = catalog['IS_STAR'] == 1
        q, = np.where( cond1 & cond2)
        ## exit if no saturated stars present
        if len(q)==0:
            print 'No saturated objects found'
            return catalog

        x0 = catalog['X_IMAGE'][q]
        y0 = catalog['Y_IMAGE'][q]
        r = np.mean([catalog['A_IMAGE'][q],catalog['B_IMAGE'][q]])
        flux = catalog['FLUX_AUTO'][q]
        l = m*flux + b

        x_vertices = np.array([x0-w,x0-w,x0+w,x0+w,x0+r,x0+l,x0+l,x0+r,x0+w,x0+w,x0-w,x0-w,x0-r,x0-l,x0-l,x0-r])
        y_vertices = np.array([y0+r,y0+l,y0+l,y0+r,y0+w,y0+w,y0-w,y0-w,y0-r,y0-l,y0-l,y0-r,y0-w,y0-w,y0+w,y0+w])
        (x_vertices, y_vertices) = fn.rotate_table(x_vertices,y_vertices,x0,y0,theta)
        catalog['IN_DIFF_MASK'][q]=1
        print "Identify objects in diffraction spike"

        Xs = np.array([catalog['XMIN_IMAGE'],catalog['XMAX_IMAGE']], dtype=int)
        Ys = np.array([catalog['YMIN_IMAGE'],catalog['YMAX_IMAGE']], dtype=int)
        bottom_pixels = [[(x,Ys[0][i]) for x in range(Xs[0][i],Xs[1][i])]for i in range(len(catalog))]
        top_pixels = [[(x,Ys[1][i]) for x in range(Xs[0][i],Xs[1][i])]for i in range(len(catalog))]
        left_pixels = [[(Xs[0][i],y) for y in range(Ys[0][i],Ys[1][i])]for i in range(len(catalog))]
        right_pixels = [[(Xs[1][i],y) for y in range(Ys[0][i],Ys[1][i])]for i in range(len(catalog))]
        for i in range(len(catalog)):
            pixels = bottom_pixels[i] + left_pixels[i] + top_pixels[i] + right_pixels[i]
            bools = [fn.inpoly(pixel[0],pixel[1],x_vertices.T[j],y_vertices.T[j]) for pixel in pixels for j in range(len(x_vertices.T))]
            if max(bools) == 1:
                catalog['IN_DIFF_MASK'][i]=1
        return catalog

    def manual_mask_cleanup(self, catalog, filt):
        val=np.zeros(len(catalog))
        col = Column(val, name='IN_MANUAL_MASK', 
                     description="In a manual mask region", dtype=int)
        catalog.add_column(col)
        mask_tab = Table.read(self.params.manual_mask_file, format='ascii.basic')
        q, = np.where((self.params.seg_id==mask_tab['SEG']) & (filt==mask_tab['FILTER']))
        if len(q)==0 :
            return catalog
        else:
            for m in q:
                x_vertices = np.array([mask_tab['AX'][m], mask_tab['BX'][m], mask_tab['CX'][m], mask_tab['DX'][m]])
                y_vertices = np.array([mask_tab['AY'][m], mask_tab['BY'][m], mask_tab['CY'][m], mask_tab['DY'][m]])
                Xs = np.array([catalog['XMIN_IMAGE'],catalog['XMAX_IMAGE']], dtype=int)
                Ys = np.array([catalog['YMIN_IMAGE'],catalog['YMAX_IMAGE']], dtype=int)
                bottom_pixels = [[(x,Ys[0][i]) for x in range(Xs[0][i],Xs[1][i])]for i in range(len(catalog))]
                top_pixels = [[(x,Ys[1][i]) for x in range(Xs[0][i],Xs[1][i])]for i in range(len(catalog))]
                left_pixels = [[(Xs[0][i],y) for y in range(Ys[0][i],Ys[1][i])]for i in range(len(catalog))]
                right_pixels = [[(Xs[1][i],y) for y in range(Ys[0][i],Ys[1][i])]for i in range(len(catalog))]
                for i in range(len(catalog)):
                    pixels = bottom_pixels[i] + left_pixels[i] + top_pixels[i] + right_pixels[i]
                    bools = [fn.inpoly(pixel[0],pixel[1],x_vertices,y_vertices) for pixel in pixels]# for j in range(len(x_vertices.T))]
                    if max(bools) == 1:
                        catalog['IN_MANUAL_MASK'][i]=1
            return catalog


        

#########################


    def cleanup_catalog(self, out_dir):
        """Removes objects on boundaries, diffraction spikes, manual mask"""
        for filt in self.params.filters:
            print "Clean up in in filter", filt
            out_name = filt
            class_catalog = out_dir + '/' + out_name + "_class.cat"
            catalog = Table.read(class_catalog, format="ascii.basic")
            catalog = self.remove_edge(catalog)
            diff_spike_params = self.params.spike_params[filt]
            catalog = self.diffraction_mask_cleanup(catalog, diff_spike_params)
            catalog = self.manual_mask_cleanup(catalog, filt)
            catalog = fn.renumber_table(catalog)
            catalog = fn.mask_it_table(catalog)
            catalog.write(out_dir + '/' +out_name+ "_clean.cat",format="ascii.basic")
            #### Make combined seg map
            self.combine_seg_map(filt,  out_dir)
            ## Add manual mask
            ## create final catalog removing maske and boundary object




    def match_to_tt(self, catalog, out_dir,
                    filter, best_stars, dist=200.):
        """Find closest tiny tim PSF image in the tt_starfiled for each of
         the stars picked to find focus stars""" 
        tt_stars = self.params.tt_file_path + "/" + filter + "/{}_stars.txt".format(filter) 
        print 'tt stars', tt_stars
        tt_table = np.loadtxt(tt_stars)        
        x0 = catalog['X_IMAGE'][best_stars]
        y0 = catalog['Y_IMAGE'][best_stars]
        r = catalog['FLUX_RADIUS'][best_stars]
        x = tt_table.T[0]
        y = tt_table.T[1]
        mult = np.ones([len(best_stars),1])
        x1 = x*mult
        y1 = y*mult
        mult = np.ones([len(x),1])
        x01 = x0*mult
        y01 = y0*mult
        d = ((x1-x01.T)**2+(y1-y01.T)**2)**0.5 
        best = np.argmin(d, axis=1)
        q = np.where((abs(x0-x[best]) < dist) & (abs(y0-y[best]) < dist))
        tt_best = best[q]
        matched_stars =np.array([best_stars[q], x0[q],y0[q],r[q], x[tt_best],y[tt_best]])
        file_name = out_dir+'/'+filter+'_matched_stars.txt'
        np.savetxt(file_name, matched_stars )
        return matched_stars.T

    def check_stars(self, best_stars, filter, out_dir):
        """ Check if the best stars are detected as stars in all filters.
        If not then remove them from the list of stars used to get the focus.
        """
        filter_list = list(self.params.filters)
        filter_list.remove(filter)
        for check_filter in filter_list:
            print 'Check strars from {0} in {1}'.format(filter,check_filter)
            cat_name = out_dir + '/' + check_filter + "_clean.cat"
            catalog = Table.read(cat_name, format="ascii.basic")
            select_stars = best_stars
            remove_stars=[]
            for i,idx in enumerate(select_stars):
                if catalog['IS_STAR'][np.int(idx)] == 0:
                    remove_stars.append(i)
            select_stars = np.delete(best_stars, remove_stars, axis=0)
        return select_stars


    def stars_for_focus(self, out_dir):
        """Make postage stamps of stars with the highest SNR"""
        for filter in self.params.filters:
            out_name = filter
            cat_name = out_dir + '/' + out_name + "_clean.cat"
            print "Making postage stamps of stars on filter ", cat_name
            catalog = Table.read(cat_name, format="ascii.basic")
            # get indices of stars with highest SNR
            best_stars = fn.select_good_stars_table(catalog)#,nstars=25)
            select_stars = self.check_stars(best_stars,filter, out_dir)
            matched_stars = self.match_to_tt(catalog, out_dir, filter, select_stars)
            print 'Number of stars selected', len(select_stars)
            num =0
            for i in range(len(matched_stars)):
                x0 =  matched_stars[i][1]
                y0 =  matched_stars[i][2]
                stamp_size =  matched_stars[i][3]*6
                image = self.params.data_files[filter]
                dir_star = out_dir+'/stars/'
                out_name = filter+'_' + str(int(matched_stars[i][0]))
                sub = fn.get_subImage(int(x0), int(y0), int(stamp_size), image,
                                      dir_star, out_name, save_img=True)
                num+=1
    
    def combine_seg_map(self, filt,  out_dir):
        ### combine seg map has value object number+1
        cat_name = out_dir + '/'+ filt +'_clean.cat'
        bright_name = out_dir + '/'+ filt +'_bright_seg_map.fits'
        faint_name = out_dir + '/'+ filt +'_faint_seg_map.fits'
        hdu1 = pyfits.open(bright_name)
        hdu2 = pyfits.open(faint_name)
        br = hdu1[0].data 
        ft = hdu2[0].data 
        hdu2.close()
        hdu1.close()
        cat = Table.read(cat_name, format='ascii.basic')
        # to account for renumbering
        new_seg = br
        q, = np.where(cat['IS_BRIGHT']==0)
        s = ft.shape 
        for i in q:
            for j in range(s[0]):
                pix, = np.where((ft[j,:] == cat['OLD_NUMBER'][i]) & (new_seg[j,:] == 0) )
                new_seg[j][pix] = cat['NUMBER'][i]+1
        new_seg_name = out_dir + '/'+ filt +'_comb_seg_map.fits'
        if os.path.isfile(new_seg_name) is True:
                subprocess.call(["rm", new_seg_name])
        print "Bright faint combined seg map created at", new_seg_name
        pyfits.writeto(new_seg_name, new_seg)


    def generate_catalog(self):
        if os.path.isdir(self.params.out_path) is False:
            subprocess.call(["mkdir", self.params.out_path])
            print "CREATING output folder"
        out_dir = self.params.out_path+ '/' + self.params.seg_id
        print "Printing outdir",out_dir
        self.get_sex_op_params(out_dir)
        ### Detection done on given added image
        for filt in self.params.filters:
            print "measuring filter", filt
            data_files = [self.params.det_im_file, self.params.data_files[filt]]
            wht_files = [self.params.det_wht_file, self.params.wht_files[filt]]
            ###### Create Bright catalog############## 
            print 'sextractor data files', data_files, wht_files
            self.run_sextractor_dual(data_files, wht_files,
                                     self.bright_config_dict, 
                                     out_dir, filt+"_bright", filt)
            ###### Create Faint catalog############## 
            self.run_sextractor_dual(data_files, wht_files,
                                     self.faint_config_dict, 
                                     out_dir, filt+"_faint", filt)
            out_name = filt
            bright_catalog_name = out_dir+'/'+filt + "_bright.cat"
            bright_catalog = self.add_bright_faint_column(bright_catalog_name,1)
            bright_catalog.write(bright_catalog_name, format="ascii.basic")
            faint_catalog_name = out_dir+'/'+filt + "_faint.cat"
            faint_catalog = self.add_bright_faint_column(faint_catalog_name,0)
            seg_map = out_dir+'/'+ out_name + "_bright_seg_map.fits"
            self.make_new_seg(seg_map, out_dir, filt)
            new_seg_map = out_dir+'/'+ filt + "_bright_seg_map_new.fits"
            self.filter_cat_with_segmentation_map(faint_catalog, new_seg_map,
                                                  out_dir, filt)
            filtered_faint_name = out_dir + '/' + filt+ "_filteredfaint.cat"
            # Merge filtered faint catalog and bright catalog
            self.merge(filtered_faint_name, bright_catalog_name, 
                       out_name, out_dir)
        # star-galaxy seperation
        self.classification(self.params.star_galaxy_weights, out_dir)   
        # Mark objects at the boundary and in diffraction spikes 
        self.cleanup_catalog(out_dir)    
        self.stars_for_focus(out_dir)











    


