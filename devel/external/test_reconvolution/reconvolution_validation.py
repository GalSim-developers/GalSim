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

"""
@file reconvolution_validation.py
Calculate the accuracy of the reconvolution engine
by comparing the shapes of reconvolved images of elliptical galaxies with
those created directly.
"""

import os
import pdb
import logging
import sys
import numpy
import sys
import math
import pylab
import argparse
import yaml
import galsim
import copy
import datetime  
from galsim import pyfits

HSM_ERROR_VALUE = -99
NO_PSF_VALUE    = -98

def _ErrorResults(ERROR_VALUE,ident):
    """
    @brief Return a results structure with all fields set to ERROR_VALUE, and with id ident
    @param ERROR_VALUE value to fill in result dict fields
    @param ident id of the galaxy
    """

    result = {  'moments_g1' : ERROR_VALUE,
                'moments_g2' : ERROR_VALUE,
                'hsmcorr_g1' : ERROR_VALUE,
                'hsmcorr_g2' : ERROR_VALUE,
                'moments_sigma' : ERROR_VALUE,
                'hsmcorr_sigma' : ERROR_VALUE,
                'ident' : ident }

    return result

def WriteResultsHeader(file_output):
    """
    @brief Writes a header file for results.
    @param file_output  file pointer to be written into
    """
    
    output_header = '# id ' + 'G1_moments G2_moments G1_hsmcorr G2_hsmcorr ' + \
                              'moments_sigma hsmcorr_sigma ' + \
                              '\n'
    file_output.write(output_header) 

def WriteResults(file_output,results):
    """
    #brief Save results to file.
    
    @file_output            file pointer to which results will be written
    @results                dict - result of GetResultsPhoton or GetResultsFFT
    """ 

    output_row_fmt = '%d\t' + '% 2.8e\t'*6 + '\n'

    # loop over the results items
       
    file_output.write(output_row_fmt % (
            results['ident'] ,
            results['moments_g1'] ,
            results['moments_g2'] ,
            results['hsmcorr_g1'] ,
            results['hsmcorr_g2'] ,
            results['moments_sigma'] ,
            results['hsmcorr_sigma'] 
        )) 

def CreateRGC(config):
    """
    @brief Creates a mock real galaxy catalog and saves it to file.
    @param config              main config dict
    """

    # set up the config accordingly
    cosmos_config = copy.deepcopy(config['cosmos_images']);
    cosmos_config['image']['gsparams'] = copy.deepcopy(config['gsparams'])

    if config['args'].debug:      use_logger = logger
    else:                         use_logger = None

    # process the config and create fits file
    try:
        logger.info('creating RGC - running galsim.Process')
        galsim.config.Process(cosmos_config,logger=use_logger)
        logger.info('created RGC images')
    except Exception,e:
        logger.error('failed to build RGC images, message %s' % e)
    # this is a hack - there should be a better way to get this number
    n_gals = len(galsim.config.GetNObjForMultiFits(cosmos_config,0,0))
    logger.info('building real galaxy catalog with %d galaxies' % n_gals)

    # get the file names for the catalog, image and PSF RGC
    filename_gals = os.path.join(config['cosmos_images']['output']['dir'],
        config['cosmos_images']['output']['file_name'])
    filename_psfs = os.path.join(config['cosmos_images']['output']['dir'],
        config['cosmos_images']['output']['psf']['file_name'])
    filename_rgc = os.path.join(
        config['reconvolved_images']['input']['real_catalog']['dir'],
        config['reconvolved_images']['input']['real_catalog']['file_name'])

    # get some additional parameters to put in the catalog
    pixel_scale = config['cosmos_images']['image']['pixel_scale']
    noise_var = config['cosmos_images']['image']['noise']['variance']
    BAND = 'F814W'     # copied from example RGC
    MAG  = 20          # no idea if this is right
    WEIGHT = 10        # ditto

    # get the columns of the catalog
    columns = []
    columns.append( pyfits.Column( name='IDENT'         ,format='J'  ,array=range(0,n_gals)       ))    
    columns.append( pyfits.Column( name='MAG'           ,format='D'  ,array=[MAG] * n_gals        ))
    columns.append( pyfits.Column( name='BAND'          ,format='5A' ,array=[BAND] * n_gals       ))    
    columns.append( pyfits.Column( name='WEIGHT'        ,format='D'  ,array=[WEIGHT] * n_gals     ))    
    columns.append( pyfits.Column( name='GAL_FILENAME'  ,format='23A',array=[filename_gals]*n_gals))            
    columns.append( pyfits.Column( name='PSF_FILENAME'  ,format='27A',array=[filename_psfs]*n_gals))            
    columns.append( pyfits.Column( name='GAL_HDU'       ,format='J'  ,array=range(0,n_gals)       ))    
    columns.append( pyfits.Column( name='PSF_HDU'       ,format='J'  ,array=range(0,n_gals)       ))    
    columns.append( pyfits.Column( name='PIXEL_SCALE'   ,format='D'  ,array=[pixel_scale] * n_gals))        
    columns.append( pyfits.Column( name='NOISE_MEAN'    ,format='D'  ,array=[0] * n_gals          ))
    columns.append( pyfits.Column( name='NOISE_VARIANCE',format='D'  ,array=[noise_var] * n_gals  ))        

    # create table
    hdu_table = pyfits.new_table(columns)
    
    # save all catalogs
    hdu_table.writeto(filename_rgc,clobber=True)
    logger.info('saved real galaxy catalog %s' % filename_rgc)

    # if in debug mode, save some plots
    # if config['args'].debug : SavePreviewRGC(config,filename_rgc)

def SavePreviewRGC(config,filename_rgc,n_gals_preview=10):
    """
    Function for eyeballing the contents of the created mock RGC catalogs.
    Arguments
    ---------
    config              config dict 
    filename_rgc        filename of the newly created real galaxy catalog fits 
    n_gals_preview      how many plots to produce (default=10)
    """

    # open the RGC
    table = pyfits.open(filename_rgc)[1].data

    # get the image and PSF filenames
    fits_gal = table[0]['GAL_FILENAME']
    fits_psf = table[0]['PSF_FILENAME']
    import pylab

    # loop over galaxies and save plots
    for n in range(n_gals_preview):

        img_gal = pyfits.getdata(fits_gal,ext=n)
        img_psf = pyfits.getdata(fits_psf,ext=n)

        pylab.subplot(1,2,1)
        pylab.imshow(img_gal,interpolation='nearest')
        pylab.title('galaxy')
        
        pylab.subplot(1,2,2)
        pylab.imshow(img_psf,interpolation='nearest')
        pylab.title('PSF')
        
        filename_fig = 'fig.previewRGC.%s.%d.png' % (config['args'].filename_config,n)

        pylab.savefig(filename_fig)


def GetReconvImage(config):
    """
    @brief Gets an image of the mock ground observation using a reconvolved method, using 
    an existing real galaxy catalog. Function CreateRGC(config) must be called earlier.
    
    @param config          main config dict read by yaml

    @return Returns a tuple img_gals,img_psfs, which are stripes of postage stamps.
    """

    if config['args'].debug: use_logger = logger
    else: use_logger = None

    # adjust the config for the reconvolved galaxies
    reconv_config = copy.deepcopy(config['reconvolved_images'])
    reconv_config['image']['gsparams'] = copy.deepcopy(config['gsparams'])
    reconv_config['input']['catalog'] = copy.deepcopy(config['cosmos_images']['input']['catalog'])
    reconv_config['gal']['shift'] = copy.deepcopy(config['cosmos_images']['gal']['shift'])

    if config['args'].debug: use_logger = logger
    else: use_logger = None


    # process the input before BuildImage    
    galsim.config.ProcessInput(reconv_config)

    # get number of images
    # n_gals = galsim.config.GetNObjForImage(reconv_config,0)
    n_gals = config['reconvolution_validation_settings']['n_images']
    # import pdb;pdb.set_trace()

    # get the reconvolved galaxies
    img_gals,img_psfs,_,_ = galsim.config.BuildImages(config=reconv_config,make_psf_image=True,
            logger=use_logger,nimages=n_gals,nproc=reconv_config['image']['nproc'])

    return (img_gals,img_psfs)

def GetDirectImage(config):
    """
    @brief Gets an image of the mock ground observation using a direct method, without reconvolution.
    @param  config          main config dict read by yaml
    @return Returns a tuple img_gals,img_psfs, which are stripes of postage stamps
    """

    # adjust the config
    direct_config = copy.deepcopy(config['reconvolved_images'])
    direct_config['image']['gsparams'] = copy.deepcopy(config['gsparams'])
    # switch gals to the original cosmos gals
    direct_config['gal'] = copy.deepcopy(config['cosmos_images']['gal'])  
    direct_config['gal']['flux'] = 1.
    # delete signal to noise - we want the direct images to be of best possible quality
    del direct_config['gal']['signal_to_noise'] 
    direct_config['gal']['shear'] = copy.deepcopy(config['reconvolved_images']['gal']['shear'])  
    direct_config['input'] = copy.deepcopy(config['cosmos_images']['input'])
    
    if config['args'].debug: use_logger = logger
    else: use_logger = None
    # process the input before BuildImage     
    galsim.config.ProcessInput(direct_config)

    # get number of images
    # n_gals = len(galsim.config.GetNObjForMultiFits(direct_config,0,0))
    n_gals = config['reconvolution_validation_settings']['n_images']

    # get the direct galaxies
    img_gals,img_psfs,_,_ = galsim.config.BuildImages(config=direct_config,make_psf_image=True,
        logger=use_logger,nimages=n_gals,nproc=direct_config['image']['nproc'])

    return (img_gals,img_psfs)

def GetShapeMeasurements(image_gal, image_psf, ident=-1):
    """
    @brief measure the image with FindAdaptiveMom and EstimateShear.
    @param image_gal    galsim image of the galaxy
    @param image_psf    galsim image of the PSF
    @param ident        id of the galaxy (default -1)
    @return a dict with fields   moments_g1 ,moments_g2 ,hsmcorr_g1 ,hsmcorr_g2 ,
        moments_sigma ,hsmcorr_sigma ,ident
    """ 

    # which shear estimator to use?
    HSM_SHEAR_EST = "KSB"

    # find adaptive moments  
    try: moments = galsim.hsm.FindAdaptiveMom(image_gal)
    except Exception,e: raise RuntimeError('FindAdaptiveMom error, message: %s' % e)

    # find HSM moments
    if image_psf == None: hsmcorr_phot_e1 =  hsmcorr_phot_e2  = NO_PSF_VALUE 
    else:
        try: 
            hsmcorr   = galsim.hsm.EstimateShear(image_gal,image_psf,strict=True,  
                                                                       shear_est=HSM_SHEAR_EST)
        except Exception,e: raise RuntimeError('EstimateShear error, message: %s' % e)
                
        logger.debug('galaxy %d : adaptive moments G1=% 2.6f\tG2=% 2.6f\tsigma=%2.6f\t hsm \
            corrected moments G1=% 2.6f\tG2=% 2.6f' 
            % ( ident , moments.observed_shape.g1 , moments.observed_shape.g2 , 
                moments.moments_sigma , hsmcorr.corrected_g1,hsmcorr.corrected_g2) )

        # create the output dictionary
        result = {  'moments_g1' : moments.observed_shape.g1,
                    'moments_g2' : moments.observed_shape.g2,
                    'hsmcorr_g1' : hsmcorr.corrected_g1,
                    'hsmcorr_g2' : hsmcorr.corrected_g2,
                    'moments_sigma' :  moments.moments_sigma, 
                    'hsmcorr_sigma' :  hsmcorr.moments_sigma, 
                    'ident' : ident}
        
    return result

def GetPixelDifference(image1,image2,id):
    """
    @brief Returns ratio of maximum pixel difference of two images 
    to the value of the maximum of pixels in the first image.
    Normalises the fluxes to one before comparing.
    
    @param image1      images to compare
    @param image2      images to compare
    
    @return Return a dict with fields: 
    diff        the difference of interest
    ident       id provided earlier

    Not used anymore.
    """

    # get the normalised images
    img1_norm = image1.array/sum(image1.array.flatten())
    img2_norm = image2.array/sum(image2.array.flatten())
    # create a residual image
    diff_image = img1_norm - img2_norm
    # calculate the ratio
    max_diff_over_max_image = abs(diff_image.flatten()).max()/img1_norm.flatten().max()
    logger.debug('max(residual) / max(image1) = %2.4e ' % ( max_diff_over_max_image )  )
    return { 'diff' : max_diff_over_max_image, 'ident' :id }


def RunMeasurement(config,filename_results,mode):
    """
    @brief              Run the comparison of reconvolved and direct imageing.
    @param config       main config dict read by yaml
    @param mode         direct or reconv
    """

    file_results = open(filename_results,'w')
    WriteResultsHeader(file_results)

    # first create the RGC
    if mode == 'reconv':
        try:
            logger.info('creating RGC')
            CreateRGC(config)
        except Exception,e:
            raise ValueError('creating RGC failed, message: %s ' % e)
        image_fun = GetReconvImage
    elif mode == 'direct':
        image_fun = GetDirectImage
    else: raise ValueError('unknown mode %s - should be either reconv or direct' % mode)


    logger.info('building %s image' , mode)
    try:
        (img_gals,img_psfs) = image_fun(config)
        logger.info('finished getting %s image' % mode)
    except Exception,e:
        logger.error('building image failed, message: %s' % e)

    # get number of objects
    nobjects = len(img_gals)

    logger.info('getting shape measurements, saving to file %s' % filename_results)
    # loop over objects
    for i in range(nobjects):

        img_gal = img_gals[i] 
        img_psf = img_psfs[i] 
 
        # get shapes and pixel differences
        try:
            result = GetShapeMeasurements(img_gal, img_psf, i)
        except Exception,e:
            logger.error('failed to get shapes for for galaxy %d. Message:\n %s' % (i,e))
            result = _ErrorResults(HSM_ERROR_VALUE,i)
  
        WriteResults(file_results,result)

    logger.info('done shape measurements, saved to file %s' % filename_results)

    del(img_gals)
    del(img_psfs)

    file_results.close()

def ChangeConfigValue(config,path,value):
    """
    @brief Changes the value of a variable in nested dict config to value.
    The field in the dict-list structure is identified by a list path.
    Example: to change the following field in config dict to value:
    conf['lvl1'][0]['lvl3']
    use the follwing path=['lvl1',0,'lvl2']
    @param    config      an object with dicts and lists
    @param    path        a list of strings and integers, pointing to the field in config that should
                          be changed, see Example above
    @param     Value      new value for this field
    """

    # build a string with the dictionary path
    eval_str = 'config'
    for key in path: 
        # check if path element is a string addressing a dict
        if isinstance(key,str):
            eval_str += '[\'' + key + '\']'
        # check if path element is an integer addressing a list
        elif isinstance(key,int):
            eval_str += '[' + str(key) + ']'
        else: 
            raise ValueError('element in the config path should be either string or int, is %s' 
                % str(type(key)))
    # assign the new value
    try:
        exec(eval_str + '=' + str(value))
        logger.debug('changed %s to %f' % (eval_str,eval(eval_str)))
    except:
        print config
        raise ValueError('wrong path in config : %s' % eval_str)

def RunComparisonForVariedParams(config):
    """
    @brief Runs the comparison of direct and reconv convolution methods, producing results file for each of the 
    varied parameters in the config file, under key 'vary_params'.
    Produces a results file for each parameter and its distinct value.
    The filename of the results file is: 'results.yaml_filename.param_name.param_value_index.cat'
    @param config              the config object, as read by yaml
    """

    # loop over parameters to vary
    for param_name in config['vary_params'].keys():

        # get more info for the parmaeter
        param = config['vary_params'][param_name]
        
        # loop over all values of the parameter, which will be changed
        for iv,value in enumerate(param['values']):

            timestamp  = datetime.datetime.now().strftime("%Y/%m/%d %H:%M:%S")

            # copy the config to the original
            changed_config = copy.deepcopy(config)
            
            # perform the change
            ChangeConfigValue(changed_config,param['path'],value)
            logger.info('%s changed parameter %s to %s' % (timestamp,param_name,str(value)))

            # If the setting change affected reconv image, then rebuild it
            if param['rebuild_reconv'] :
                logger.info('getting reconv results')
                changed_config_reconv = copy.deepcopy(changed_config)
                # Get the results filenames
                filename_results_reconv = 'results.%s.%s.%03d.reconv.cat' % (
                                        config['args'].filename_config, param_name,iv)
                
                # Run and save the measurements
                RunMeasurement(changed_config_reconv,filename_results_reconv,'reconv')             
                logger.info(('saved reconv results for varied parameter %s with value %s\n'
                     + 'filename: %s') % (param_name,str(value),filename_results_reconv) )

            # If the setting change affected direct image, then rebuild it           
            if param['rebuild_direct'] :
                logger.info('getting direct results')
                changed_config_direct = copy.deepcopy(changed_config)
                # Get the results filename
                filename_results_direct = 'results.%s.%s.%03d.direct.cat' % (
                    config['args'].filename_config,param_name,iv)

                # Run the measurement
                RunMeasurement(changed_config_direct,filename_results_direct,'direct')
                logger.info(('saved direct results for varied parameter %s with value %s\n'  
                    + 'filename %s') % ( param_name,str(value),filename_results_direct) )




if __name__ == "__main__":

    description = 'Compare reconvolved and directly created galaxies.'

    # parse arguments
    parser = argparse.ArgumentParser(description=description, add_help=True)
    parser.add_argument('filepath_config', type=str,
                 help='yaml config file, see reconvolution_validation.yaml for example.')
    parser.add_argument('--debug', action="store_true", help='run in debug mode', default=False)
    parser.add_argument('--default_only', action="store_true", 
        help='Run only for default settings, ignore vary_params in config file.\
              --vary_params_only must not be used alongside this option.'
        , default=False)
    parser.add_argument('--vary_params_only', action="store_true", 
        help='Run only for varied settings, do not run the defaults.\
               --default_only must not be used alongside this option.'
        , default=False)
    
    args = parser.parse_args()
    args.filename_config = os.path.basename(args.filepath_config)

    # set up logger
    if args.debug: logger_level = 'logging.DEBUG'
    else:  logger_level = 'logging.INFO'
    logging.basicConfig(format="%(message)s", level=eval(logger_level), stream=sys.stdout)
    logger = logging.getLogger("reconvolution_validation") 

    # sanity check the inputs
    if args.default_only and args.vary_params_only:
        raise('Use either default_only or vary_params_only, or neither.')

    # load the configuration file
    config = yaml.load(open(args.filename_config,'r'))
    config['args'] = args
   
    # set flags what to do
    if args.vary_params_only:
        config['run_default'] = False
        config['run_vary_params'] = True
    elif args.default_only:
        config['run_default'] = True
        config['run_vary_params'] = False
    else:
        config['run_default'] = True
        config['run_vary_params'] = True


    # run only the default settings
    if config['run_default']:
 
        timestamp  = datetime.datetime.now().strftime("%Y/%m/%d %H:%M:%S")

        logger.info('%s running reconv and direct for default settings' % timestamp)
        # Get the results filenames
        
        filename_results_direct = 'results.%s.default.direct.cat' % (config['args'].filename_config)
        filename_results_reconv = 'results.%s.default.reconv.cat' % (config['args'].filename_config)
        
        config_reconv = copy.deepcopy(config)
        RunMeasurement(config_reconv,filename_results_reconv,'reconv')
        config_direct = copy.deepcopy(config)
        RunMeasurement(config_direct,filename_results_direct,'direct')
        
        logger.info(('saved direct and reconv results for default parameter set\n'
             + 'filenames: %s\t%s') % (filename_results_direct,filename_results_reconv))
    
    # run the config including changing of the parameters
    if config['run_vary_params']:
        timestamp  = datetime.datetime.now().strftime("%Y/%m/%d %H:%M:%S")
        logger.info('%s running reconvolution validation for varied parameters' % timestamp)
        RunComparisonForVariedParams(config)

    timestamp  = datetime.datetime.now().strftime("%Y/%m/%d %H:%M:%S")
    logger.info('%s finished' % timestamp)


