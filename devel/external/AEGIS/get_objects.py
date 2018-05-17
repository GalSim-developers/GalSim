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

""" Program Number: 1

Program to detect objects in a given image (segment) and make a catalog
using sextractor. Objects are detected with the Hot-Cold method employed in
Rix et al.(2004).

Requirements:
SExtractor Detection:
SExtractor is run in double image mode, with objects detected in det_im_file &
det_wht_file as weight map. These detected objects are then measures in each
band(file_name, wht_name). SExtractor WEIGHT_TYPE is set with wht_type and
det_wht_type for detection and measurement respectively. The config parameters
for Hot-Cold detetction are preset and not input parametrs. The hot(faint)
objects are merged with the cold(bright) catalog that aren't within buffer
region  --buffer.

Cleanup:
Since the noise in the image is not uncorrelated (dut to multidrizzle), the snr
computed from sextractor needs to be modified by parameter scale factr --sf.
The magnitudes computed by SExtractor have to be corrected for dust extinction.
The correction function correct_extinction() in functions.py would have to be
modified when running script on a survey other then AEGIS. The detected objects
are then classified into stars and galaxies depending on their position in the
magnitude Vs peak surface brightness plot. The separation line is set
by star_galaxy_params. Objects that lie on image edge are masked. Region around
saturated stars are masked: masked region set by diff_spike_params. Regions
that were manually observed to have artefacts (eg.ghosts) and are to be masked
are input as manual_mask_file. final catalog is renumbered begining from 0.
Bright and faint seg maps are combined to form a single segmentation map.
The bright seg map objects are expanded by 10 pixels.
Note: Value of object in seg map will be 1 higher than NUMBER in catalog.
Catalog numbers start at 0, while 0 in segmap is no object present.

Stars for PSF estimation:
Select upto 25 stars with the highest SNR that are not masked. If they are
detected as stars in all bands, have an image in tt_starfield within 200 pixels,
and do not have any other objects nearby, they are saved to a list for PSF
estimation. Postage stamps of these stars are also saved for manual inspection.

Output:
Cleaned catalog, combined segmentation map, list of stars for PSF measurement
and postage stamp images of those stars.
"""
import asciidata
import subprocess
import pyfits
import os
import numpy as np
import functions as fn
import clean_pstamp as cp
from astropy.table import Table, Column, vstack


class Main_param:
    """Class containing parameters to pass to run analysis on each segment file.
    it adds path name in front of file name and replace seg_id names and filter
    names.
    """
    def __init__(self, args):
        self.seg_id = args.seg_id
        self.file_name = args.file_name.replace('seg_id', self.seg_id)
        self.wht_name = args.wht_name.replace('seg_id', self.seg_id)
        self.det_im_file = args.det_im_file.replace('seg_id', self.seg_id)
        self.det_wht_file = args.det_wht_file.replace('seg_id', self.seg_id)
        self.filters = args.filter_names
        self.out_path = args.out_path
        self.tt_file_path = args.tt_file_path
        self.sf = args.sf
        self.wht_type = args.wht_type
        self.det_wht_type = args.det_wht_type
        self.manual_mask_file = args.manual_mask_file
        self.buffer = args.buffer
        # making weight maps rms
        if args.file_path[-1] != '/':
            self.file_path = args.file_path + '/'
        else:
            self.file_path = args.file_path
        if args.wht_path[-1] != '/':
            self.wht_path = args.wht_path + '/'
        else:
            self.wht_path = args.wht_path
        self.spike_params, self.zero_point_mag = {}, {}
        self.gain, self.star_galaxy_params = {}, {}
        self.data_files, self.wht_files = {}, {}
        for i in range(len(self.filters)):
            filter1 = self.filters[i]
            self.data_files[filter1] = self.file_path + filter1 + '/' + self.file_name.replace('filter', filter1)
            self.wht_files[filter1] = self.wht_path + filter1 + '/' + self.wht_name.replace('filter', filter1)
            self.spike_params[filter1] = args.diff_spike_params[i]
            self.zero_point_mag[filter1] = args.zero_point_mag[i]
            self.star_galaxy_params[filter1] = args.star_galaxy_params[i]
            self.gain[filter1] = args.gain[i]
        self.det_im_file = self.file_path + self.det_im_file
        self.det_wht_file = self.wht_path + self.det_wht_file


def run_segment(params):
    """Find objects in individual segments.
    Runs SExtractor on a segment and stores the detected objects in .cat file
    SExtractor input settings stored in .config file.
    Parameters to be measured by SExtractor are written to sex.param file.

    Detecting objects, Hot/Cold method :
    1) SExtractor is run in dual image mode, detecting in co-added image and
    measuring in individual filter images to give their corresponding
    *_brigt.cat detected images.
    2) Same step repeated with faint_config_dict SExtractor parameters and
    detected objects saved in .*_faint.cat.
    3) Segmentation map is made for each of the _bright.cat files with regions
    belonging to bright objects expaned by 15 pixels.
    4) Filter faint catalog:  Removes objects in *_faint.cat that are masked in
    the segmentation map for all filters. O/p is *_filteredfaint.cat
    5) Merges bright and filtered faint catalogs to give catalog of all objects.
    """
    cat = GalaxyCatalog(params)
    cat.generate_catalog()


class GalaxyCatalog:
    """Parameters to be measured by SExtractor."""
    output_params = ["NUMBER",
                     "X_IMAGE",
                     "Y_IMAGE",
                     "A_IMAGE",
                     "B_IMAGE",
                     "ALPHA_J2000",
                     "DELTA_J2000",
                     "XMIN_IMAGE",
                     "XMAX_IMAGE",
                     "YMIN_IMAGE",
                     "YMAX_IMAGE",
                     "FLAGS",
                     "MU_MAX",
                     "MAG_AUTO",
                     "MAGERR_AUTO",
                     "MAG_BEST",
                     "MAGERR_BEST",
                     "CLASS_STAR",
                     "FLUX_RADIUS",
                     "FLUX_AUTO",
                     "FLUXERR_AUTO",
                     "KRON_RADIUS",
                     "THETA_IMAGE",
                     "ELLIPTICITY"]
    # SExtractor input config values for bright object detection
    bright_config_dict = {'DETECT_MINAREA': 140,
                          'DETECT_THRESH': 2.2,
                          'DEBLEND_NTHRESH': 64,
                          'DEBLEND_MINCONT': 0.04,
                          'CLEAN_PARAM': 1.0,
                          'BACK_SIZE': 400,
                          'BACK_FILTERSIZE': 5,
                          'BACKPHOTO_TYPE': "LOCAL",
                          'BACKPHOTO_THICK': 200,
                          'PIXEL_SCALE': 0.03}
    # SExtractor input config values for faint object detection
    faint_config_dict = {'DETECT_MINAREA': 18,
                         'DETECT_THRESH': 1.0,
                         'DEBLEND_NTHRESH': 64,
                         'DEBLEND_MINCONT': 0.065,
                         'CLEAN_PARAM': 1.0,
                         'BACK_SIZE': 100,
                         'BACK_FILTERSIZE': 3,
                         'BACKPHOTO_TYPE': "LOCAL",
                         'BACKPHOTO_THICK': 200,
                         'PIXEL_SCALE': 0.03}

    def __init__(self, params):
        self.params = params

    def get_sex_op_params(self, out_dir):
        """Saves the names of parameters thet SExtractor must save"""
        param_fname = 'sex_out.param'
        try:
            param_file = open(out_dir + '/' + param_fname, 'w')
        except:
            subprocess.call(["mkdir", out_dir])
            param_file = open(out_dir + '/' + param_fname, 'w')
        print out_dir
        for i in range(len(self.output_params)):
            param_file.write(self.output_params[i])
            param_file.write("\n")
        param_file.close()

    def run_sextractor_dual(self, data_files, wht_files,
                            use_dict, out_dir, out_name, filt):
        """ Runs sextractor in dual image mode"""
        print "Running dual:", out_name
        # Create config newfiles[i] and write out to a file
        config_ascii = asciidata.create(2, 8 + len(use_dict))
        # File-Specific Configurations
        config_ascii[0][0] = 'CATALOG_NAME'
        config_ascii[1][0] = out_dir + '/' + out_name + ".cat"
        config_ascii[0][1] = 'PARAMETERS_NAME'
        config_ascii[1][1] = out_dir + "/sex_out.param"
        config_ascii[0][2] = 'WEIGHT_TYPE'
        config_ascii[1][2] = str(self.params.wht_type + ',' + self.params.det_wht_type)
        config_ascii[0][3] = 'WEIGHT_IMAGE'
        config_ascii[1][3] = str(wht_files[0] + ',' + wht_files[1])
        row_counter = 4
        for key, value in use_dict.iteritems():
            config_ascii[0][row_counter] = key
            config_ascii[1][row_counter] = value
            row_counter += 1
        config_ascii[0][row_counter] = 'CHECKIMAGE_NAME'
        config_ascii[1][row_counter] = out_dir + '/' + out_name + "_seg_map.fits"
        config_ascii[0][row_counter + 1] = 'CHECKIMAGE_TYPE'
        config_ascii[1][row_counter + 1] = 'SEGMENTATION'
        config_ascii[0][row_counter + 2] = 'MAG_ZEROPOINT'
        config_ascii[1][row_counter + 2] = self.params.zero_point_mag[filt]
        config_ascii[0][row_counter + 3] = 'GAIN'
        config_ascii[1][row_counter + 3] = self.params.gain[filt]
        config_fname = out_dir + '/' + out_name + ".config"
        config_ascii.writeto(config_fname)
        # Run sextractor and get the catalog
        subprocess.call(["sex", data_files[0] + "," + data_files[1], "-c", config_fname])

    def add_bright_faint_column(self, cat_name, tag):
        """Add coloumn 'IS_BRIGHT' in catalog with name cat_name.value of
        column is to tag.
        """
        catalog = Table.read(cat_name, format="ascii.sextractor")
        col = Column(np.ones(len(catalog)) * tag, name='IS_BRIGHT',
                     dtype='int', description='Detected in hot mode')
        catalog.add_column(col)
        return catalog

    def make_new_seg(self, seg_map, out_dir, out_name):
        """Expands bright seg map objects by params.buffer pixels """
        data = pyfits.open(seg_map)[0].data
        new_seg = fn.seg_expand(data, buff=self.params.buffer)
        new_name = out_dir + '/' + out_name + "_bright_seg_map_new.fits"
        if os.path.isfile(new_name) is True:
                subprocess.call(["rm", new_name])
        pyfits.writeto(new_name, new_seg)

    def filter_cat_with_segmentation_map(self, faint_catalog, seg_map,
                                         out_dir, out_name):
        """Removes objects from the faint catalog that lie within the bright
        segmentation map. This is to remove multiple detections of bright
        objects.
        """
        print "Filtering faint objects for section ", self.params.seg_id
        segmentation_file = pyfits.open(seg_map)
        data = segmentation_file[0].data
        fc = faint_catalog
        val = [i for i in range(len(fc)) if (data[int(fc['Y_IMAGE'][i]), int(fc['X_IMAGE'][i])] == 0)]
        new_catalog = faint_catalog[val]
        name = out_dir + '/' + out_name + "_filteredfaint.cat"
        new_catalog.write(name, format="ascii.basic")

    def merge(self, filtered_faint_catalog,
              bright_catalog, filt, out_dir):
        """ Merge objects detected in bright and filtered faint catalog.
        Extinction correction to magnitude is also applied here. """
        print "Merging bright  and faint catalogs for section", self.params.seg_id
        name = out_dir + '/' + filt + "_merge.cat"
        faint_cat = Table.read(filtered_faint_catalog, format="ascii.basic")
        bright_cat = Table.read(bright_catalog, format="ascii.basic")
        comb_cat = vstack([bright_cat, faint_cat])
        # Correct magntiude for extinction
        corr_mag = fn.correct_extinction(comb_cat['MAG_AUTO'], filt)
        col = Column(corr_mag, name='MAG_CORR',
                     description='Extinction corrected MAG_AUTO')
        comb_cat.add_column(col)
        comb_cat.write(name, format="ascii.basic")

    def classification(self, div_params, out_dir):
        """Detected objects are classified as stars or not depending on their
        magnitude and peak surface brightnes."""
        x_max = 25.2
        print "Performing star-galaxy separation for section ", self.params.seg_id
        for filt in self.params.filters:
            x_div = div_params[filt][0]
            y_div = div_params[filt][1]
            slope = div_params[filt][2]
            intercept = y_div - slope * x_div
            out_name = filt
            merged_catalog = out_dir + '/' + out_name + "_merge.cat"
            catalog = Table.read(merged_catalog, format="ascii.basic")
            snr = np.array(catalog['FLUX_AUTO']) / np.array(catalog['FLUXERR_AUTO'])
            col = Column(snr, name='SNR',
                        description='Signal to Noise Ratio')
            catalog.add_column(col)
            # Modified SNR: Correcting of correlations in noise.
            A = catalog['FLUXERR_AUTO']**2 - catalog['FLUX_AUTO'] / self.params.gain[filt]
            new_f_err = (A / self.params.sf + catalog['FLUX_AUTO'] / self.params.gain[filt])**0.5
            col = Column(new_f_err, name='NEW_FLUXERR_AUTO',
                         description='Modified FLUXERR_AUTO')
            catalog.add_column(col)
            new_snr = np.array(catalog['FLUX_AUTO']) / new_f_err
            col = Column(new_snr, name='NEW_SNR',
                         description='Modified Signal to Noise Ratio')
            catalog.add_column(col)
            col = Column(np.zeros(len(catalog)), name='IS_STAR', dtype='int')
            catalog.add_column(col)
            q = fn.is_below_boundary_table(catalog['MAG_CORR'],
                                           catalog['MU_MAX'],
                                           x_div, y_div, slope,
                                           intercept, x_max)
            catalog['IS_STAR'][q] = 1
            col = Column(np.zeros(len(catalog)), name='IS_FAKE', dtype='int')
            catalog.add_column(col)
            q = fn.is_below_boundary_table(catalog['MAG_CORR'],
                                           catalog['MU_MAX'],
                                           x_div + 1, y_div - 2,
                                           slope, intercept - 2,
                                           x_max + 1)
            catalog['IS_FAKE'][q] = 1
            catalog.write(out_dir + '/' + out_name + "_class.cat",
                          format="ascii.basic")

    def remove_edge(self, catalog):
        """Remove objects lying on the image plane edge."""
        print "Removing edge objects"
        A = (390., 321.)
        B = (498., 6725.)
        C = (6898., 7287.)
        D = (7002., 806.)
        x_min = catalog['XMIN_IMAGE']
        x_max = catalog['XMAX_IMAGE']
        y_min = catalog['YMIN_IMAGE']
        y_max = catalog['YMAX_IMAGE']
        val = np.zeros(len(catalog))
        val[fn.lies_within_table(x_min, x_max, y_min, y_max, A, B, C, D)] = 1
        col = Column(val, name='IN_BOUNDARY',
                     description="Inside a masked region", dtype=int)
        catalog.add_column(col)
        return catalog

    def diffraction_mask_cleanup(self, catalog,
                                 diff_spike_params,
                                 mag_cutoff=19.0):
        """Masks objects within diffraction spikes of saturated stars."""
        m = diff_spike_params[0]  # slope
        b = diff_spike_params[1]  # intercept
        w = diff_spike_params[2] * 0.5  # width
        theta = diff_spike_params[3]  # angle
        val = np.zeros(len(catalog))
        col = Column(val, name='IN_DIFF_MASK',
                     description="Close to saturated star", dtype=int)
        catalog.add_column(col)
        print "Identifying saturated stars"
        cond1 = catalog['MAG_CORR'] < mag_cutoff
        cond2 = catalog['IS_STAR'] == 1
        q, = np.where(cond1 & cond2)
        # exit if no saturated stars present
        if len(q) == 0:
            print 'No saturated objects found'
            return catalog
        x0 = catalog['X_IMAGE'][q]
        y0 = catalog['Y_IMAGE'][q]
        r = np.mean([catalog['A_IMAGE'][q], catalog['B_IMAGE'][q]]) * 5
        flux = catalog['FLUX_AUTO'][q]
        l = m * flux + b
        x_vertices = np.array([x0-w,x0-w,x0+w,x0+w,x0+r,x0+l,x0+l,x0+r,x0+w,x0+w,x0-w,x0-w,x0-r,x0-l,x0-l,x0-r])
        y_vertices = np.array([y0+r,y0+l,y0+l,y0+r,y0+w,y0+w,y0-w,y0-w,y0-r,y0-l,y0-l,y0-r,y0-w,y0-w,y0+w,y0+w])
        (x_vertices, y_vertices) = fn.rotate_table(x_vertices, y_vertices, x0, y0, theta)
        catalog['IN_DIFF_MASK'][q] = 1
        print "Identify objects in diffraction spike"
        Xs = np.array([catalog['XMIN_IMAGE'], catalog['XMAX_IMAGE']],
                      dtype=int)
        Ys = np.array([catalog['YMIN_IMAGE'], catalog['YMAX_IMAGE']],
                      dtype=int)
        bottom_pixels = [[(x, Ys[0][i]) for x in range(Xs[0][i],Xs[1][i])]for i in range(len(catalog))]
        top_pixels = [[(x, Ys[1][i]) for x in range(Xs[0][i], Xs[1][i])]for i in range(len(catalog))]
        left_pixels = [[(Xs[0][i], y) for y in range(Ys[0][i], Ys[1][i])]for i in range(len(catalog))]
        right_pixels = [[(Xs[1][i], y) for y in range(Ys[0][i], Ys[1][i])]for i in range(len(catalog))]
        for i in range(len(catalog)):
            pixels = bottom_pixels[i] + left_pixels[i] + top_pixels[i] + right_pixels[i]
            bools = [fn.inpoly(pixel[0], pixel[1], x_vertices.T[j], y_vertices.T[j]) for pixel in pixels for j in range(len(x_vertices.T))]
            if max(bools) == 1:
                catalog['IN_DIFF_MASK'][i] = 1
        return catalog

    def manual_mask_cleanup(self, catalog, filt):
        """Masks regions that were observed to have artefacts upon visual inspection"""
        val = np.zeros(len(catalog))
        col = Column(val, name='IN_MANUAL_MASK',
                     description="In a manual mask region", dtype=int)
        catalog.add_column(col)
        mask_tab = Table.read(self.params.manual_mask_file, format='ascii.basic')
        q, = np.where((self.params.seg_id == mask_tab['SEG']) & (filt == mask_tab['FILTER']))
        if len(q) == 0:
            return catalog
        else:
            for m in q:
                x_vertices = np.array([mask_tab['AX'][m], mask_tab['BX'][m], mask_tab['CX'][m], mask_tab['DX'][m]])
                y_vertices = np.array([mask_tab['AY'][m], mask_tab['BY'][m], mask_tab['CY'][m], mask_tab['DY'][m]])
                Xs = np.array([catalog['XMIN_IMAGE'], catalog['XMAX_IMAGE']],
                              dtype=int)
                Ys = np.array([catalog['YMIN_IMAGE'], catalog['YMAX_IMAGE']],
                              dtype=int)
                bottom_pixels = [[(x, Ys[0][i]) for x in range(Xs[0][i],Xs[1][i])]for i in range(len(catalog))]
                top_pixels = [[(x, Ys[1][i]) for x in range(Xs[0][i], Xs[1][i])]for i in range(len(catalog))]
                left_pixels = [[(Xs[0][i], y) for y in range(Ys[0][i], Ys[1][i])]for i in range(len(catalog))]
                right_pixels = [[(Xs[1][i], y) for y in range(Ys[0][i], Ys[1][i])]for i in range(len(catalog))]
                for i in range(len(catalog)):
                    pixels = bottom_pixels[i] + left_pixels[i] + top_pixels[i] + right_pixels[i]
                    bools = [fn.inpoly(pixel[0], pixel[1], x_vertices, y_vertices) for pixel in pixels]
                    if max(bools) == 1:
                        catalog['IN_MANUAL_MASK'][i] = 1
            return catalog

    def combine_seg_map(self, filt, out_dir):
        """Combines bright and faint segmentation maps. Regions belonging to
        bright objects are expanded by 5 pixels"""
        cat_name = out_dir + '/' + filt + '_clean.cat'
        bright_name = out_dir + '/' + filt + '_bright_seg_map.fits'
        faint_name = out_dir + '/' + filt + '_faint_seg_map.fits'
        hdu1 = pyfits.open(bright_name)
        hdu2 = pyfits.open(faint_name)
        br = hdu1[0].data
        ft = hdu2[0].data
        hdu2.close()
        hdu1.close()
        cat = Table.read(cat_name, format='ascii.basic')
        new_seg = br
        # Expand bright regions by 5 pixels
        q, = np.where(cat['IS_BRIGHT'] == 1)
        for i in q:
            new_seg = fn.seg_expand(new_seg, buff=5, val=int(i) + 1, set_to=int(i) + 1)
            # +1 to account for renumbering
        q, = np.where(cat['IS_BRIGHT'] == 0)
        s = ft.shape
        for i in q:
            for j in range(s[0]):
                pix, = np.where((ft[j, :] == cat['OLD_NUMBER'][i]) & (new_seg[j, :] == 0))
                new_seg[j][pix] = cat['NUMBER'][i] + 1
        new_seg_name = out_dir + '/' + filt + '_comb_seg_map.fits'
        print "Bright faint combined seg map created at", new_seg_name
        pyfits.writeto(new_seg_name, new_seg, clobber=True)
        os.remove(bright_name)
        os.remove(faint_name)

    def cleanup_catalog(self, out_dir):
        """Removes objects on boundaries, diffraction spikes, manual mask"""
        for filt in self.params.filters:
            print "Clean up in in filter", filt
            out_name = filt
            class_catalog = out_dir + '/' + out_name + "_class.cat"
            catalog = Table.read(class_catalog, format="ascii.basic")
            # Remove objects near edges
            catalog = self.remove_edge(catalog)
            # remove objects in diffraction spike
            diff_spike_params = self.params.spike_params[filt]
            # magnitude cutoff of saturated stars is the horizontal line
            # in star-galaxy separation line
            mag_cutoff = self.params.star_galaxy_params[filt][0]
            catalog = self.diffraction_mask_cleanup(catalog, diff_spike_params,
                                                    mag_cutoff=mag_cutoff)
            # remove objects in manual mask
            catalog = self.manual_mask_cleanup(catalog, filt)
            catalog = fn.renumber_table(catalog)
            catalog = fn.mask_it_table(catalog)
            # add columns to catalog that will be useful later
            col = Column(np.zeros(len(catalog)), name='MULTI_DET',
                         dtype='int',
                         description='detected in another segment')
            catalog.add_column(col)
            col = Column(['aa.0000000'] * len(catalog), name='MULTI_DET_OBJ',
                         dtype='S12', description='object id kept')
            catalog.add_column(col)
            col = Column([filt.upper()] * len(catalog), name='BAND',
                         description='measurement filter')
            catalog.add_column(col)
            catalog.write(out_dir + '/' + out_name + "_clean.cat",
                          format="ascii.basic")
            # Make combined seg map
            self.combine_seg_map(filt, out_dir)

    def check_oth_obj(self, x0, y0, r,
                      filt, idx, out_dir):
        """Checks if other objects are close to stars picked for PSF estimation"""
        seg_name = out_dir + '/' + filt +'_comb_seg_map.fits'
        seg_im = fn.get_subImage_pyfits(int(x0), int(y0), [int(r)] * 8, seg_name,
                                        None, None, save_img=False)
        shape = seg_im.shape
        num = seg_im[shape[0] / 2, shape[1] / 2]
        im, bl, oth, oth_segs, check = cp.div_pixels(seg_im, -1)
        if len(oth_segs) > 1:
            print "Multiple objects in star stamp for star {0}, object{1}".format(idx, oth_segs)
            return False
        return True

    def match_to_tt(self, catalog, out_dir,
                    filt, best_stars, dist=200.):
        """Finds closest tiny tim PSF image in the tt_starfiled for each star"""
        tt_stars = self.params.tt_file_path + "/" + filt + "/{}_stars.txt".format(filt)
        print 'tt stars', tt_stars
        tt_table = np.loadtxt(tt_stars)
        # Center and size of stars in catalog.
        x0 = catalog['X_IMAGE'][best_stars]
        y0 = catalog['Y_IMAGE'][best_stars]
        r = catalog['FLUX_RADIUS'][best_stars]
        # Center of tiny tim stars.
        x = tt_table.T[0]
        y = tt_table.T[1]
        mult = np.ones([len(best_stars), 1])
        x1 = x * mult
        y1 = y * mult
        mult = np.ones([len(x), 1])
        x01 = x0 * mult
        y01 = y0 * mult
        d = ((x1 - x01.T)**2 + (y1 - y01.T)**2)**0.5
        best = np.argmin(d, axis=1)
        check_oth = []
        for i, idx in enumerate(best_stars):
            val = self.check_oth_obj(x0[i], y0[i], r[i], filt, idx, out_dir)
            check_oth.append(val)
        q = np.where((abs(x0 - x[best]) < dist) & (abs(y0 - y[best]) < dist) & check_oth)
        tt_best = best[q]
        matched_stars = np.array([best_stars[q], x0[q], y0[q], r[q], x[tt_best], y[tt_best]])
        file_name = out_dir + '/' + filt + '_matched_stars.txt'
        np.savetxt(file_name, matched_stars)
        return matched_stars.T

    def check_stars(self, best_stars, filt, out_dir):
        """Checks if the stars selected for PSF estimation are detected as stars
        in all filters. If not then it removes them from the list."""
        filter_list = list(self.params.filters)
        filter_list.remove(filt)
        for check_filter in filter_list:
            print 'Check strars from {0} in {1}'.format(filt, check_filter)
            cat_name = out_dir + '/' + check_filter + "_clean.cat"
            catalog = Table.read(cat_name, format="ascii.basic")
            select_stars = best_stars
            remove_stars = []
            for i, idx in enumerate(select_stars):
                if catalog['IS_STAR'][np.int(idx)] == 0:
                    remove_stars.append(i)
            select_stars = np.delete(best_stars, remove_stars, axis=0)
        return select_stars

    def stars_for_focus(self, out_dir):
        """Makes postage stamps of stars. Ordered in decreasing highest SNR"""
        for filt in self.params.filters:
            cat_name = out_dir + '/' + filt + "_clean.cat"
            print "Making postage stamps of stars in filter ", cat_name
            catalog = Table.read(cat_name, format="ascii.basic")
            # get indices of stars with highest SNR
            best_stars = fn.select_good_stars_table(catalog)
            select_stars = self.check_stars(best_stars, filt, out_dir)
            matched_stars = self.match_to_tt(catalog, out_dir, filt, select_stars)
            print 'Number of stars selected', len(select_stars)
            num = 0
            for i in range(len(matched_stars)):
                x0 = matched_stars[i][1]
                y0 = matched_stars[i][2]
                stamp_size = matched_stars[i][3] * 8
                image = self.params.data_files[filt]
                dir_star = out_dir + '/stars/'
                out_name = filt + '_' + str(int(matched_stars[i][0]))
                # Save image of selected stars
                sub = fn.get_subImage(int(x0), int(y0), int(stamp_size), image,
                                      dir_star, out_name, save_img=True)
                num += 1

    def generate_catalog(self):
        # create o/p folder if doesn't exist
        if os.path.isdir(self.params.out_path) is False:
            subprocess.call(["mkdir", self.params.out_path])
            print "CREATING output folder"
        out_dir = self.params.out_path + '/' + self.params.seg_id
        print "Printing outdir", out_dir
        # create file taht lists o/p required from SExtractor
        self.get_sex_op_params(out_dir)
        # Detection done on given added image
        for filt in self.params.filters:
            print "Measuring on filter", filt
            data_files = [self.params.det_im_file, self.params.data_files[filt]]
            wht_files = [self.params.det_wht_file, self.params.wht_files[filt]]
            # Create Bright catalog#
            print 'Sextractor input data files', data_files, wht_files
            self.run_sextractor_dual(data_files, wht_files,
                                     self.bright_config_dict,
                                     out_dir, filt + "_bright", filt)
            # Create Faint catalog#
            self.run_sextractor_dual(data_files, wht_files,
                                     self.faint_config_dict,
                                     out_dir, filt + "_faint", filt)
            out_name = filt
            bright_catalog_name = out_dir + '/' + filt + "_bright.cat"
            bright_catalog = self.add_bright_faint_column(bright_catalog_name, 1)
            bright_catalog.write(bright_catalog_name, format="ascii.basic")
            faint_catalog_name = out_dir + '/' + filt + "_faint.cat"
            faint_catalog = self.add_bright_faint_column(faint_catalog_name, 0)
            seg_map = out_dir + '/' + out_name + "_bright_seg_map.fits"
            self.make_new_seg(seg_map, out_dir, filt)
            new_seg_map = out_dir + '/' + filt + "_bright_seg_map_new.fits"
            self.filter_cat_with_segmentation_map(faint_catalog, new_seg_map,
                                                  out_dir, filt)
            filtered_faint_name = out_dir + '/' + filt + "_filteredfaint.cat"
            # Merge filtered faint catalog and bright catalog
            self.merge(filtered_faint_name, bright_catalog_name,
                       out_name, out_dir)
            os.remove(new_seg_map)
        # star-galaxy seperation
        self.classification(self.params.star_galaxy_params, out_dir)
        # Mark objects at the boundary and in diffraction spikes
        self.cleanup_catalog(out_dir)
        # pick stars for PSF estimation
        self.stars_for_focus(out_dir)


if __name__ == '__main__':
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument('--seg_id', default='1a',
                        help="Segment id of image to run [Default:1a]")
    parser.add_argument('--filter_names', default=['f606w', 'f814w'],
                        help="names of filters [Default: ['f606w','f814w']]")
    parser.add_argument('--file_path', default='/nfs/slac/g/ki/ki19/deuce/AEGIS/unzip/',
                        help="Path of directory containing input images \
                        [Default:'/nfs/slac/g/ki/ki19/deuce/AEGIS/unzip] ")
    parser.add_argument('--wht_path', default='/nfs/slac/g/ki/ki19/deuce/AEGIS/unzip',
                        help="Path of directory containing weight files \
                        [Default:'/nfs/slac/g/ki/ki19/deuce/AEGIS/unzip] ")
    parser.add_argument('--out_path', default='/nfs/slac/g/ki/ki19/deuce/AEGIS/AEGIS_catalog_full/',
                        help="Path to where you want the output stored \
                        [Default: /nfs/slac/g/ki/ki19/deuce/AEGIS/AEGIS_catalog_full/]")
    parser.add_argument('--file_name', default='EGS_10134_seg_id_acs_wfc_filter_30mas_unrot_drz.fits',
                        help="File name of measurement image with 'seg_id' & \
                        'filter' in place of image segment id and filter \
                        [Default:'EGS_10134_seg_id_acs_wfc_f606w_30mas_unrot_drz.fits']")
    parser.add_argument('--wht_name', default='EGS_10134_seg_id_acs_wfc_filter_30mas_unrot_rms.fits',
                        help="Name of weight map of measurement image with 'seg_id' \
                        and 'filter' in place of image segment id and filter  \
                        [Default:'EGS_10134_seg_id_acs_wfc_filter_30mas_unrot_rms.fits']")
    parser.add_argument('--det_im_file',
                        default='added/EGS_10134_seg_id_acs_wfc_30mas_unrot_added_drz.fits',
                        help="File name of image to run detection on with '\
                        'seg_id' in place of image segment id. \
                        [Default:'added/EGS_10134_seg_id_acs_wfc_30mas_unrot_added_drz.fits']")
    parser.add_argument('--det_wht_file',
                        default='added/EGS_10134_seg_id_acs_wfc_30mas_unrot_added_rms.fits',
                        help="Weight file name of image of image used in detection with ' \
                        'seg_id' in place of image segment id. \
                        [Default:'added/EGS_10134_seg_id_acs_wfc_30mas_unrot_added_rms.fits']")
    parser.add_argument('--wht_type', default='MAP_RMS',
                        help="SExtractor Weight file type for measurement image. \
                        Default='MAP_RMS'")
    parser.add_argument('--det_wht_type', default='MAP_RMS',
                        help="SExtractor Weight file type for detection image. \
                         Default='MAP_RMS'")
    parser.add_argument('--buffer', default=15,
                        help="Number of pixels used as buffer around bright \
                        objects in Hot-cold detection method.[Default:15(pixels)]")
    parser.add_argument('--diff_spike_params',
                        default=[(0.0441087, 81.0863, 15.0, 2.614),
                                 (0.0448020, 92.7674, 15.0, 2.180)],
                        help="Params of diffraction spikes in each filter. \
                        They must be in the same order as filter_names. \
                        [slope(pixels/ADU),intercept(pixels),width(pixels),angle(degrees)] \
                        [Default: [(0.0350087,64.0863,40.0,2.614),  \
                        (0.0367020,77.7674,40.0,2.180)]]")
    parser.add_argument('--star_galaxy_params',
                        default=[(19.95, 16.17, 0.945), (18.95, 15.11, 0.98)],
                        help="Star galaxy seperation line parametrs \
                        [Default:(x_div, y_div, slope)]")
    parser.add_argument('--zero_point_mag',
                        default=(26.508, 25.955),
                        help="Zero point magnitides for each band \
                        [Default:(26.508, 25.955)]")
    parser.add_argument('--gain', default=[2260, 2100],
                        help="Detector gain in e/ADU[Default:[2260,2100](s)]")
    parser.add_argument('--sf', default=0.316,
                        help="Scale factor to correct for correlated noise \
                        [Default:0.316 ")
    parser.add_argument('--manual_mask_file', default='manual_masks.txt',
                        help="File containing regions that are to be masked\
                        [Default:'manual_masks.txt']")
    parser.add_argument('--tt_file_path',
                        default='/nfs/slac/g/ki/ki19/deuce/AEGIS/tt_starfield/',
                        help="Path of directory containing modelled TT fields \
                        [Default:'/nfs/slac/g/ki/ki19/deuce/AEGIS/tt_starfield/']")
    args = parser.parse_args()
    params = Main_param(args)
    run_segment(params)
