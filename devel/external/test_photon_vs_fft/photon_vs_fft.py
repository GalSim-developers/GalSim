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
    #brief Save results to file.
    
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

def GetShapeMeasurements(image_gal, image_psf, ident):
    """
    @param image_gal - 
    @param image_psf - 
    @param ident - 
    """

    HSM_SHEAR_EST = "KSB"
    NO_PSF_VALUE = -10

    # find adaptive moments  
    try: moments = galsim.FindAdaptiveMom(image_gal)
    except: raise RuntimeError('FindAdaptiveMom error')
        

    # find HSM moments
    if image_psf == None: hsmcorr_phot_e1 =  hsmcorr_phot_e2  = NO_PSF_VALUE 
    else:
        try: 
            hsmcorr   = galsim.EstimateShearHSM(image_gal,image_psf,strict=True,  
                                                                       shear_est=HSM_SHEAR_EST)
        except: raise RuntimeError('EstimateShearHSM error')
                
        logger.debug('galaxy %d : adaptive moments G1=% 2.6f\tG2=% 2.6f\tsigma=%2.6f\thsm corrected moments G1=% 2.6f\tG2=% 2.6f' 
            % ( ident , moments.observed_shape.g1 , moments.observed_shape.g2 , moments.moments_sigma , hsmcorr.corrected_g1,hsmcorr.corrected_g2) )

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
    @config                 the yaml config used to create images
    @param file_output      opened file to which the results will be written
    """

    # start the timer
    t1 = time.time()

    # open the file
    file_results = open(filename_results,'w')

    # write header 
    WriteResultsHeader(file_results)

    # First process the input field:
    galsim.config.ProcessInput(config)

    # dirty way of getting this number
    nobjects = config['some_variables']['n_gals_in_cat']
    config['image']['draw_method'] = 'fft'

    # modify all 'repeat' keys in config to 1, so that we get single images of galaxies without 
    # repeating them. Config for this test requires to repeat all galaxies with n_trials_per_iter.
    ChangeAllConfigKeys(config,'repeat',1)
    # use logger in galsim.config only in debug mode
    if config['debug']: use_logger = logger
    else: use_logger = None
    # get the images
    try: img_gals,img_psfs,_,_ = galsim.config.BuildImages( nimages = nobjects , config=config , 
        make_psf_image=True , logger=use_logger , nproc=config['image']['nproc'])
    except Exception, e:
        raise RuntimeError('Failed to build FFT image. Message: %s',e)


    # measure the photon and fft images
    for i in range(nobjects):

        # this bit is still serial, not too good...
        try: 
            result = GetShapeMeasurements(img_gals[i],img_psfs[i],i)
        except Exception,e: 
            logger.error('failed to get GetShapeMeasurements for galaxy %d. Message %s' % (i,e))
            result = _ErrorResults(HSM_ERROR_VALUE,i)

        WriteResults(file_results,result)

    logger.info('finished getting FFT results for %d galaxies' % nobjects)


def RunMeasurementsPhotAndFFT(config,filename_results_pht,filename_results_fft): 
    """
    @brief get results for all galaxies using photon shooting adn FFT.

    Arguments
    ---------
    @param   config              the yaml config used to create images
    @param   file_output_pht     opened file to save the photon results
    @param   file_output_fft     opened file to save the FFT results
    """

    # start the timer
    t1 = time.time()

    # Open files
    file_results_fft = open(filename_results_fft,'w')
    file_results_pht = open(filename_results_pht,'w')
    WriteResultsHeader(file_results_fft)
    WriteResultsHeader(file_results_pht)

    # First process the input field:
    galsim.config.ProcessInput(config)

    # dirty way of getting this number
    # nobjects = len(galsim.config.GetNObjForMultiFits(config,0,0))
    nobjects = config['some_variables']['n_gals_in_cat']

    #Mmeasure the photon and FFT images
    for i in range(nobjects):
       
        try:
            # check if we want to log ouptput from compare_dft_vs_photon_config, only in debug mode
            if config['debug']: use_logger = logger
            else: use_logger = None
            # run compare_dft_vs_photon_config and get the results object
            res = galsim.utilities.compare_dft_vs_photon_config(config, gal_num=i, 
                hsm=True, moments = True, logger=use_logger,
                abs_tol_ellip = float(config['compare_dft_vs_photon_config']['abs_tol_ellip']),
                abs_tol_size = float(config['compare_dft_vs_photon_config']['abs_tol_size']),
                n_trials_per_iter = 
                    int(float(config['compare_dft_vs_photon_config']['n_trials_per_iter'])),
                n_photons_per_trial =  
                    int(float(config['compare_dft_vs_photon_config']['n_photons_per_trial'])),
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
            logger.error('failed to get compare_dft_vs_photon_config for galaxy %d. Message:\n %s' % (i,e))
            # if failure, create results with failure flags
            results_fft = _ErrorResults(HSM_ERROR_VALUE,i)
            results_pht = _ErrorResults(HSM_ERROR_VALUE,i)
  
        WriteResults(file_results_pht,results_pht)
        WriteResults(file_results_fft,results_fft)

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
        raise ValueError('wrong path in config : %s' % eval_str)

def RunComparisonForVariedParams(config):
    """
    Runs the comparison of photon and fft drawing methods, producing results file for each of the 
    varied parameters in the config file, under key 'vary_params'.
    Produces a results file for each parameter and it's distinct value.
    The filename of the results file is: 'results.yaml_filename.param_name.param_value_index.cat'
    Arguments
    ---------
    @param config              the config object, as read by yaml
    """

    # Run the default config
    if config['run_default']:
        logging.info('running photon and FFT measurements for default parameter set')
        default_config = config.copy()
        param_name = 'default'
        filename_results_pht = 'results.%s.%s.pht.cat' % (config['filename_config'],
                                                                            param_name)
        filename_results_fft = 'results.%s.%s.fft.cat' % (config['filename_config'],
                                                                            param_name)
        # Run and save the measurements
        RunMeasurementsPhotAndFFT(default_config, 
            filename_results_pht, filename_results_fft)             
        logging.info(('saved FFT and photon results for default parameters\n'
             + 'filenames: %s\t%s') % (filename_results_pht,filename_results_fft))

    # Loop over parameters to vary
    for param_name in config['vary_params'].keys():
        
        # Get more info for the parmaeter
        param = config['vary_params'][param_name]
        # Loop over all values of the parameter, which will be changed
        for iv,value in enumerate(param['values']):
            # Copy the config to the original
            changed_config = config.copy()
            # Perform the change
            ChangeConfigValue(changed_config,param['path'],value)
            logging.info('changed parameter %s to %s' % (param_name,str(value)))
            # Run the photon vs fft test on the changed configs
            
            # If the setting change affected photon image, then rebuild it
            if param['rebuild_photon'] :
                logger.info('getting photon and FFT results')
                changed_config2 = copy.deepcopy(changed_config)
                # Get the results filenames
                filename_results_pht = 'results.%s.%s.%03d.pht.cat' % (config['filename_config'],
                                                                        param_name,iv)
                filename_results_fft = 'results.%s.%s.%03d.fft.cat' % (config['filename_config'],
                                                                        param_name,iv)

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
                # Get the results filename
                filename_results_fft = 'results.%s.%s.%03d.fft.cat' % (
                    config['filename_config'],param_name,iv)

                # Run the measurement
                RunMeasurementsFFT(changed_config1,filename_results_fft)
                logging.info(('saved FFT results for varied parameter %s with value %s\n'  
                    + 'filename %s') % ( param_name,str(value),filename_results_fft) )
              
if __name__ == "__main__":


    description = \
    'Compare FFT vs Photon shooting. \
    Use the galaxies specified in the corresponding yaml file \
    (see photon_vs_fft.yaml for an example)' 

    # parse arguments
    parser = argparse.ArgumentParser(description=description, add_help=True)
    parser.add_argument('filename_config', type=str, 
        help='Yaml config file, see photon_vs_fft.yaml for example.')
    parser.add_argument('--default_only', action="store_true", 
        help='Run only for default settings for photons and FFT, ignore vary_params in config file.'
        , default=False)
    parser.add_argument('--debug', action="store_true", 
        help='Run with debug verbosity.', default=False)
    args = parser.parse_args()

    # set up logger
    if args.debug: logger_level = 'logging.DEBUG'
    else:  logger_level = 'logging.INFO'
    logging.basicConfig(format="%(message)s", level=eval(logger_level), stream=sys.stdout)
    logger = logging.getLogger("photon_vs_fft") 

    # load the configuration file
    config = yaml.load(open(args.filename_config,'r'))
    config['debug'] = args.debug
    config['filename_config'] = args.filename_config

    # run only the default settings
    if args.default_only:
        logger.info('running photon_vs_fft for default settings')
        logger.info('getting photon and FFT results')
        config2 = copy.deepcopy(config)
        # Get the results filenames
        filename_results_pht = 'results.%s.default.pht.cat' % (config['filename_config'])
        filename_results_fft = 'results.%s.default.fft.cat' % (config['filename_config'])
        # Run and save the measurements
        RunMeasurementsPhotAndFFT(config2, 
            filename_results_pht, filename_results_fft)             
        logging.info(('saved FFT and photon results for default parameter set\n'
             + 'filenames: %s\t%s') % (filename_results_pht,filename_results_fft))
    # run the config including changing of the parameters
    else:
        logger.info('running photon_vs_fft for varied parameters')
        RunComparisonForVariedParams(config)

