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

"""@file photon_vs_fft.py 
Compare images created by FFT and photon shooting rendering methods in GalSim. 
All configuration is specified in a config yaml file.
See photon_vs_fft.yaml for an example.
The script generates the images and compares the pixel intensities as well as images moments.
"""
import logging
import galsim
import sys
import os
import numpy
import argparse
import yaml
import pdb
import math
import copy
import time

HSM_ERROR_VALUE = -99
NO_PSF_VALUE    = -98

def _ErrorResults(ERROR_VALUE,ident):
    """
    @brief get empty results
    @param ERROR_VALUE value to fill in to the result
    @param ident identifier of the galaxy
    """

    result = {  'moments_g1' : ERROR_VALUE,
                        'moments_g2' : ERROR_VALUE,
                        'hsmcorr_g1' : ERROR_VALUE,
                        'hsmcorr_g2' : ERROR_VALUE,
                        'moments_sigma' : ERROR_VALUE,
                        'hsmcorr_sigma' : ERROR_VALUE,
                        'moments_g1err' : ERROR_VALUE,
                        'moments_g2err' : ERROR_VALUE,
                        'hsmcorr_g1err' : ERROR_VALUE,
                        'hsmcorr_g2err' : ERROR_VALUE,
                        'moments_sigmaerr' : ERROR_VALUE,
                        'hsmcorr_sigmaerr' : ERROR_VALUE,
                        'ident' : ident }

    return result

def WriteResultsHeader(file_output):
    """
    @brief Writes a header file for results.
    @param file_output  file pointer to be written into
    """
    
    output_header = '# id ' + 'G1_moments G2_moments G1_hsmcorr G2_hsmcorr ' + \
                              'moments_sigma hsmcorr_sigma ' + \
                              'err_g1obs err_g2obs err_g1hsm err_g2hsm err_sigma err_sigma_hsm' + \
                              '\n'
    file_output.write(output_header) 

def WriteResults(file_output,results):
    """
    @brief Save results to file.
    
    @file_output            file pointer to which results will be written
    @results                dict - result of GetResultsPhoton or GetResultsFFT
    """ 

    output_row_fmt = '%d\t' + '% 2.8e\t'*12 + '\n'

    # loop over the results items
       
    file_output.write(output_row_fmt % (
            results['ident'] ,
            results['moments_g1'] ,
            results['moments_g2'] ,
            results['hsmcorr_g1'] ,
            results['hsmcorr_g2'] ,
            results['moments_sigma'] ,
            results['hsmcorr_sigma'] ,
            results['moments_g1err'] ,
            results['moments_g2err'] ,
            results['hsmcorr_g1err'] ,
            results['hsmcorr_g2err'] ,
            results['moments_sigmaerr'] ,
            results['hsmcorr_sigmaerr'] 
            )) 

def GetShapeMeasurements(image_gal, image_psf, ident=-1):
    """
    @param image_gal    galsim image of the galaxy
    @param image_psf    galsim image of the PSF
    @param ident        id of the galaxy (default -1)
    """

    HSM_SHEAR_EST = "KSB"
    NO_PSF_VALUE = -10
    
    # find adaptive moments  

    try: moments = galsim.hsm.FindAdaptiveMom(image_gal)
    except Exception, emsg : raise RuntimeError('FindAdaptiveMom error: %s' % emsg)
        

    # find HSM moments
    if image_psf == None: hsmcorr_phot_e1 =  hsmcorr_phot_e2  = NO_PSF_VALUE 
    else:
        try: 
            hsmcorr   = galsim.hsm.EstimateShear(image_gal,image_psf,strict=True,  
                                                                       shear_est=HSM_SHEAR_EST)
        except Exception, emsg: raise RuntimeError('EstimateShearHSM error: %s' % emsg)
                
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
                    'moments_g1err' : 0. ,
                    'moments_g2err' : 0. ,
                    'hsmcorr_g1err' : 0. ,
                    'hsmcorr_g2err' : 0. ,
                    'moments_sigmaerr' : 0. ,
                    'hsmcorr_sigmaerr' : 0. ,
                    'ident' : ident}
        
    return result

def RunMeasurementsFFT(config,filename_results): 
    """
    @brief get results for all galaxies using just FFT.

    Arguments
    ---------
    @param config              the yaml config used to create images
    @param filename_results    file to which the results will be written
    """

    # open the file
    file_results = open(filename_results,'w')

    # write header 
    WriteResultsHeader(file_results)

    # First process the input field:
    galsim.config.ProcessInput(config)

    # get number of objects
    if config['ident'] < 0:
        # slightly hacky way to get this number
        nobjects = len(galsim.config.GetNObjForMultiFits(config,0,0))
        obj_num = 0
    else:
        nobjects = 1
        obj_num = config['ident']

    # set the draw method to FFT
    config['image']['draw_method'] = 'fft'

    # modify all 'repeat' keys in config to 1, so that we get single images of galaxies without 
    # repeating them. Config for this test requires to repeat all galaxies with n_trials_per_iter.
    ChangeAllConfigKeys(config,'repeat',1)
    # use logger in galsim.config only in debug mode
    if config['debug']: use_logger = logger
    else: use_logger = None
    # get the images
    img_gals,img_psfs,_,_ = galsim.config.BuildImages( 
        nimages = nobjects , obj_num = obj_num, 
        config=config , make_psf_image=True , logger=use_logger , nproc=config['image']['nproc'])

    if config['save_images']:
        filename_fits_gal = 'img.gal.%s.fits' % filename_results
        filename_fits_psf = 'img.psf.%s.fits' % filename_results
        galsim.fits.writeMulti(img_gals,filename_fits_gal,clobber=True)
        galsim.fits.writeMulti(img_psfs,filename_fits_psf,clobber=True)

    

    # measure the photon and fft images
    for i in range(nobjects):

        if config['ident'] < 0: obj_num = i

        # this bit is still serial, not too good...
        try: 
            result = GetShapeMeasurements(img_gals[i],img_psfs[i],obj_num)
        except Exception,e: 
            logger.error('failed to get GetShapeMeasurements for galaxy %d. Message %s' % (obj_num,e))
            result = _ErrorResults(HSM_ERROR_VALUE,i)

        WriteResults(file_results,result)

    # close the file
    file_results.close()

    logger.info('finished getting FFT results for %d galaxies' % nobjects)


def RunMeasurementsPhotAndFFT(config,filename_results_pht,filename_results_fft): 
    """
    @brief get results for all galaxies using photon shooting adn FFT.

    Arguments
    ---------
    @param   config                   the yaml config used to create images
    @param   filename_results_pht     file to save the photon results
    @param   filename_results_fft     file to save the FFT results
    """

    # Open files
    file_results_fft = open(filename_results_fft,'w')
    file_results_pht = open(filename_results_pht,'w')
    WriteResultsHeader(file_results_fft)
    WriteResultsHeader(file_results_pht)

    # First process the input field:
    galsim.config.ProcessInput(config)

    # get number of objects
    if config['ident'] < 0:
        # slightly hacky way to get this number
        nobjects = len(galsim.config.GetNObjForMultiFits(config,0,0))
        logger.info('found %d galaxies in the config file' % nobjects)
        objects_ids = range(nobjects)
    else:
        nobjects = 1
        objects_ids = [config['ident']]

    # Measure the photon and FFT images
    for i in objects_ids:
       
        try:
            # check if we want to log ouptput from compare_dft_vs_photon_config, only in debug mode
            if config['debug']: use_logger = logger
            else: use_logger = None
            # run compare_dft_vs_photon_config and get the results object
            res = galsim.utilities.compare_dft_vs_photon_config(config, gal_num=i, 
                hsm=True, moments = True, logger=use_logger,
                abs_tol_ellip = float(config['photon_vs_fft_settings']['abs_tol_ellip']),
                abs_tol_size = float(config['photon_vs_fft_settings']['abs_tol_size']),
                n_trials_per_iter = 
                    int(float(config['photon_vs_fft_settings']['n_trials_per_iter'])),
                n_max_iter = int(float(config['photon_vs_fft_settings']['n_max_iter'])),
                n_photons_per_trial =  
                    int(float(config['photon_vs_fft_settings']['n_photons_per_trial']))
                )
            
            results_pht = {  'moments_g1' : res.g1obs_draw - res.delta_g1obs,
                             'moments_g2' : res.g2obs_draw - res.delta_g2obs,
                             'hsmcorr_g1' : res.g1hsm_draw - res.delta_g1hsm,
                             'hsmcorr_g2' : res.g2hsm_draw - res.delta_g2hsm,
                             'moments_sigma' : res.sigma_draw - res.delta_sigma,
                             'hsmcorr_sigma' : res.sighs_draw - res.delta_sighs,
                             'hsmcorr_g1err' : res.err_g1hsm ,
                             'hsmcorr_g2err' : res.err_g2hsm ,
                             'moments_g1err' : res.err_g1obs ,
                             'moments_g2err' : res.err_g2obs ,
                             'moments_sigmaerr' : res.err_sigma,
                             'hsmcorr_sigmaerr' : res.err_sighs,
                             'ident' : i }
            
            results_fft = {  'moments_g1' : res.g1obs_draw ,
                             'moments_g2' : res.g2obs_draw ,
                             'hsmcorr_g1' : res.g1hsm_draw ,
                             'hsmcorr_g2' : res.g2hsm_draw ,
                             'moments_sigma' : res.sigma_draw,
                             'hsmcorr_sigma' : res.sighs_draw,
                             'hsmcorr_g1err' : 0 ,
                             'hsmcorr_g2err' : 0 ,
                             'moments_g1err' : 0 ,
                             'moments_g2err' : 0 ,
                             'moments_sigmaerr' : 0,
                             'hsmcorr_sigmaerr' : 0,
                             'ident' : i }

            logger.info('finished getting photon and FFT measurements from gal %d : time :\
                %s min' % (i,str(res.time/60.)))
        except Exception,e:
            logger.error('failed to get compare_dft_vs_photon_config for galaxy %d. Message:\n %s' 
                % (i,e))
            # if failure, create results with failure flags
            results_fft = _ErrorResults(HSM_ERROR_VALUE,i)
            results_pht = _ErrorResults(HSM_ERROR_VALUE,i)
  
        WriteResults(file_results_pht,results_pht)
        WriteResults(file_results_fft,results_fft)

    # close the files
    file_results_fft.close()
    file_results_pht.close()

    logger.info('finished getting FFT and phot results for %d galaxies' % nobjects)

def ChangeAllConfigKeys(config,key,value):
    """
    @brief recursive function to modify all keys with name 'key' in a dict to value 'value'
    @param key      name of all keys to be modified
    @param value    new value for all keys with name 'key'
    """

    def _stepin(level,key,value):

        if isinstance(level,dict):
            if key in level:
                level[key] = value
            for k in level.keys():
                if isinstance(level[k],dict) or isinstance(level[k],list):
                    _stepin(level[k],key,value)
        elif isinstance(level,list):
            for l in level:
                _stepin(l,key,value)

    _stepin(config,key,value)

def ChangeConfigValue(config,path,value):
    """
    Changes the value of a variable in nested dict config to value.
    The field in the dict-list structure is identified by a list path.
    Example: to change the following field in config dict to value:
    conf['lvl1'][0]['lvl3']
    use the follwing path=['lvl1',0,'lvl2']
    Arguments
    ---------
    @param config       an object with dicts and lists
    @param path         a list of strings and integers, pointing to the field in config that should
                        be changed, see Example above
    @param value        new value for this field
    """

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
    # perform assgnment of the new value
    try:
        exec(eval_str + '=' + str(value))
        logging.debug('changed %s to %f' % (eval_str,eval(eval_str)))
    except:
        print config
        print 'wrong path in config : %s' % eval_str
        raise

def RunComparisonForVariedParams(config):
    """
    Runs the comparison of photon and fft drawing methods, producing results file for each of the 
    varied parameters in the config file, under key 'vary_params'.
    Produces a results file for each parameter and its distinct value.
    The filename of the results file is: 'results.yaml_filename.param_name.param_value_index.cat'
    Arguments
    ---------
    @param config              the config object, as read by yaml
    """

    # Loop over parameters to vary
    for param_name in config['vary_params'].keys():
        
        # Get more info for the parmaeter
        param = config['vary_params'][param_name]
        # Loop over all values of the parameter, which will be changed
        for iv,value in enumerate(param['values']):
            # Copy the config to the original
            changed_config = copy.deepcopy(config)
            # Perform the change
            ChangeConfigValue(changed_config,param['path'],float(value))
            logging.info('changed parameter %s to %s' % (param_name,str(value)))
            # Run the photon vs fft test on the changed configs

            # Get the results filenames
            if config['ident'] < 0:
                filename_results_pht = 'results.%s.%s.%03d.pht.cat' % (
                                    config['filename_config'], param_name,iv)
                filename_results_fft = 'results.%s.%s.%03d.fft.cat' % (
                                    config['filename_config'], param_name,iv)
            else:
                filename_results_pht = 'results.%s.%s.%03d.pht.cat.%03d' % (
                                    config['filename_config'], param_name, iv, config['ident'])
                filename_results_fft = 'results.%s.%s.%03d.fft.cat.%03d' % (
                                    config['filename_config'], param_name, iv, config['ident'])
            

            # If the setting change affected photon image, then rebuild it
            if param['rebuild_pht'] :
                logger.info('getting photon and FFT results')
                changed_config2 = copy.deepcopy(changed_config)

                # Run and save the measurements
                RunMeasurementsPhotAndFFT(changed_config2, 
                    filename_results_pht, filename_results_fft)             
                logging.info(('saved FFT and photon results for varied parameter %s with value %s\n'
                     + 'filenames: %s\t%s') % (param_name,str(value),
                    filename_results_pht,filename_results_fft))

            # just get the FFT results
            else:
                logger.info('getting FFT results only')
                changed_config1 = copy.deepcopy(changed_config)

                # Run the measurement
                RunMeasurementsFFT(changed_config1,filename_results_fft)
                logging.info(('saved FFT results for varied parameter %s with value %s\n'  
                    + 'filename %s') % ( param_name,str(value),filename_results_fft) )
              
if __name__ == "__main__":


    description = \
    'Compare FFT vs Photon shooting. \
    Use the galaxies specified in the corresponding yaml file \
    (see photon_vs_fft.yaml for an example) \
    Example outputs file: \
    results.yaml_filename.param_name.param_index.cat \
    where param_name is the name of the varied parameter \
    (if default set is ran, then will contain word "default" ) \
    param_index - the index of a parameter in the list in the config file.\
    Each row corresponds to a galaxy shape measurement. \
    ' 

    # parse arguments
    parser = argparse.ArgumentParser(description=description, add_help=True)
    parser.add_argument('filename_config', type=str, 
        help='Yaml config file, see photon_vs_fft.yaml for example.')
    parser.add_argument('--default_only', action="store_true", 
        help='Run only for default settings for photons and FFT, ignore vary_params in config file.\
              --vary_params_only must not be used alongside this option.'
        , default=False)
    parser.add_argument('--vary_params_only', action="store_true", 
        help='Run only for varied settings for photons and FFT, do not run the defaults.\
               --default_only must not be used alongside this option.'
        , default=False)
    parser.add_argument('--debug', action="store_true", 
        help='Run with debug verbosity.', default=False)
    parser.add_argument('--save_images', action="store_true", 
        help='save galaxy and PSF images', default=False)
    parser.add_argument(
            '-i', '--ident', type=int, action='store', default=-1, 
            help='id of the galaxy in the catalog to process. If this option is supplied, then \
            only one galaxy will be processed, otherwise all in the config file')
    args = parser.parse_args()

    # set up logger
    if args.debug: logger_level = 'logging.DEBUG'
    else:  logger_level = 'logging.INFO'
    logging.basicConfig(format="%(message)s", level=eval(logger_level), stream=sys.stdout)
    logger = logging.getLogger("photon_vs_fft") 

    # sanity check the inputs
    if args.default_only and args.vary_params_only:
        raise('Use either default_only or vary_params_only, or neither.')

    # load the configuration file
    config = yaml.load(open(args.filename_config,'r'))
    config['debug'] = args.debug
    config['filepath_config'] = args.filename_config
    config['filename_config'] = os.path.basename(config['filepath_config'])
    config['ident'] = args.ident
    config['save_images'] = args.save_images


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

    # decide if run all galaxies or just one specified by command line
    if config['ident'] < 0:
        logger.info('running for all galaxies in the config file')
    else:
        logger.info('running for galaxy %d only' % config['ident'])

    # run only the default settings
    if config['run_default']:
        logger.info('running photon_vs_fft for default settings')
        logger.info('getting photon and FFT results')
        config2 = copy.deepcopy(config)
        # Get the results filenames
        if config['ident'] < 0:
            filename_results_fft = 'results.%s.default.fft.cat' % (config['filename_config'])
            filename_results_pht = 'results.%s.default.pht.cat' % (config['filename_config'])
        else:
            filename_results_fft = 'results.%s.default.fft.cat.%03d' % (
                                                config['filename_config'],config['ident'])
            filename_results_pht = 'results.%s.default.pht.cat.%03d' % (
                                                config['filename_config'],config['ident'])

        # Run and save the measurements
        RunMeasurementsPhotAndFFT(config2, 
            filename_results_pht, filename_results_fft)             
        logging.info(('saved FFT and photon results for default parameter set\n'
             + 'filenames: %s\t%s') % (filename_results_pht,filename_results_fft))
    
    # run the config including changing of the parameters
    if config['run_vary_params']:
        logger.info('running photon_vs_fft for varied parameters')
        RunComparisonForVariedParams(config)

