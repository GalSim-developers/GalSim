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
import pylab
import math
import copy

HSM_ERROR_VALUE = -99
NO_PSF_VALUE    = -98

def SaveResults(filename_output,results_pht,results_fft):
    """
    #brief Save results to file.
    
    @filename_output        file to which results will be written
    @results_pht            dict - result of GetResultsPhoton
    @results_fft            dict - result of GetResultsFFT

    """


    # initialise the output file
    file_output = open(filename_output,'w')
       
    # write the header
    output_row_fmt = '%d\t% 2.6e\t% 2.6e\t% 2.6e\t% 2.6e\t% 2.6e\t' + \
                        '% 2.6e\t% 2.6e\t% 2.6e\t% 2.6e\t% 2.6e\t% 2.6e\t% 2.6e\t% 2.6e\t% 2.6e\t% 2.6e\n'
    output_header = '# id ' +   'G1_moments_fft G2_moments_fft G1_moments_photon G2_moments_photon ' + \
                                'G1_hsmcorr_fft G2_hsmcorr_fft G1_hsmcorr_photon G2_hsmcorr_photon ' + \
                                'moments_fft_sigma moments_photon_sigma ' + \
                                'err_g1obs err_g2obs err_g1hsm err_g2hsm err_sigma ' + \
                                '\n'
    file_output.write(output_header) 

    # loop over the results items
    for (i,v) in enumerate(results_pht):     
        
        file_output.write(output_row_fmt % (
            results_fft[i]['ident'] ,
            results_fft[i]['moments_g1'] ,
            results_fft[i]['moments_g2'] ,
            results_pht[i]['moments_g1'] ,
            results_pht[i]['moments_g2'] ,
            results_fft[i]['hsmcorr_g1'] ,
            results_fft[i]['hsmcorr_g2'] ,
            results_pht[i]['hsmcorr_g1'] ,
            results_pht[i]['hsmcorr_g2'] ,
            results_fft[i]['moments_sigma'] ,
            results_pht[i]['moments_sigma'] ,
            results_pht[i]['moments_g1err'] ,
            results_pht[i]['moments_g2err'] ,
            results_pht[i]['hsmcorr_g1err'] ,
            results_pht[i]['hsmcorr_g2err'] ,
            results_pht[i]['moments_sigmaerr'] 
            ))

    
    logging.info('saved results file %s' % (filename_output))

def GetShapeMeasurements(image_gal, image_psf, ident):
    """
    @image_gal - 
    @image_psf - 
    @ident - 
    """

    HSM_SHEAR_EST = "KSB"
    NO_PSF_VALUE = -10

    # find adaptive moments  
    try: moments = galsim.FindAdaptiveMom(image_gal)
    except: raise RuntimeError('FindAdaptiveMom error')
        
    logger.debug('adaptive moments G1=% 2.6f\tG2=% 2.6f\tsigma=%2.6f' % (
        moments.observed_shape.g1 , moments.observed_shape.g2 , moments.moments_sigma ) )
    
    # find HSM moments
    if image_psf == None: hsmcorr_phot_e1 =  hsmcorr_phot_e2  = NO_PSF_VALUE 
    else:
        try: hsmcorr   = galsim.EstimateShearHSM(image_gal,image_psf,strict=True,
                                                                       shear_est=HSM_SHEAR_EST)
        except: raise RuntimeError('EstimateShearHSM error')
        
    logger.debug('hsm corrected moments     G1=% 2.6f\tG2=% 2.6f' % (
        hsmcorr.corrected_g1,hsmcorr.corrected_g2) )
        
    # create the output dictionary
    result = {  'moments_g1' : moments.observed_shape.g1,
                'moments_g2' : moments.observed_shape.g2,
                'hsmcorr_g1' : hsmcorr.corrected_g1,
                'hsmcorr_g2' : hsmcorr.corrected_g2,
                'moments_sigma' :  moments.moments_sigma, 
                'ident' : ident}
    
    return result

def GetResultsFFT(config): 
    """
    @brief get results for all galaxies using photon shooting.

    Arguments
    ---------
    @config          the yaml config used to create images
    @return          Outputs a list of dictionaries containing the results of comparison.
    """

    # First process the input field:
    galsim.config.ProcessInput(config)

    # dirty way of getting this number
    # nobjects = len(galsim.config.GetNObjForMultiFits(config,0,0))
    # print nobjects
    nobjects = config['some_variables']['n_gals_in_cat']

    config['image']['draw_method'] = 'fft'

    # initialise the results dict
    results_all = []

    # measure the photon and fft images
    for i in range(nobjects):

        obj_num = i * config['some_variables']['n_trials_per_iter']
        img_gals,img_psfs,_,_,_ = galsim.config.BuildSingleImage( obj_num = obj_num , 
            config=config , make_psf_image=True )

        try: 
            result = GetShapeMeasurements(img_gals,img_psfs,i)
        except: 
            logger.error('failed to get GetShapeMeasurements for galaxy %d' % i)
            result = {  'moments_g1' : HSM_ERROR_VALUE,
                        'moments_g2' : HSM_ERROR_VALUE,
                        'hsmcorr_g1' : HSM_ERROR_VALUE,
                        'hsmcorr_g2' : HSM_ERROR_VALUE,
                        'moments_sigma' : HSM_ERROR_VALUE,
                        'ident' : i }

        results_all.append(result)

    return results_all


def GetResultsPhoton(config): 
    """
    @brief get results for all galaxies using photon shooting.

    Arguments
    ---------
    @config          the yaml config used to create images
    @return          Outputs a list of dictionaries containing the results of comparison.
    """

    # First process the input field:
    galsim.config.ProcessInput(config)

    # dirty way of getting this number
    # nobjects = len(galsim.config.GetNObjForMultiFits(config,0,0))
    nobjects = config['some_variables']['n_gals_in_cat']

    # initialise the results dict
    results_all = []

    # measure the photon and fft images
    for i in range(nobjects):

       
        try:
            res = galsim.utilities.compare_dft_vs_photon_config(config, gal_num=i, hsm=True,
                logger=None,
                abs_tol_ellip = float(config['compare_dft_vs_photon_config']['abs_tol_ellip']),
                abs_tol_size = float(config['compare_dft_vs_photon_config']['abs_tol_size']),
                n_trials_per_iter = int(float(config['compare_dft_vs_photon_config']['n_trials_per_iter'])),
                n_photons_per_trial = int(float(config['compare_dft_vs_photon_config']['n_photons_per_trial']))
                )
            result = {  'moments_g1' : res.g1obs_draw - res.delta_g1obs,
                        'moments_g2' : res.g2obs_draw - res.delta_g2obs,
                        'hsmcorr_g1' : res.g1hsm_draw - res.delta_g1hsm,
                        'hsmcorr_g2' : res.g2hsm_draw - res.delta_g2hsm,
                        'moments_sigma' : res.sigma_draw - res.delta_sigma,
                        'hsmcorr_g1err' : res.err_g1hsm ,
                        'hsmcorr_g2err' : res.err_g2hsm ,
                        'moments_g1err' : res.err_g1obs ,
                        'moments_g2err' : res.err_g2obs ,
                        'moments_sigmaerr' : res.err_sigma,
                        'ident' : i }
        except:
            logger.error('failed to get compare_dft_vs_photon_config for galaxy %d' % i)
            result = {  'moments_g1' : HSM_ERROR_VALUE,
                        'moments_g2' : HSM_ERROR_VALUE,
                        'hsmcorr_g1' : HSM_ERROR_VALUE,
                        'hsmcorr_g2' : HSM_ERROR_VALUE,
                        'moments_sigma' : HSM_ERROR_VALUE,
                        'hsmcorr_g1err' : HSM_ERROR_VALUE,
                        'hsmcorr_g2err' : HSM_ERROR_VALUE,
                        'moments_g1err' : HSM_ERROR_VALUE,
                        'moments_g2err' : HSM_ERROR_VALUE,
                        'moments_sigmaerr' : HSM_ERROR_VALUE,
                        'ident' : i }
        
        results_all.append(result)

    return results_all

def ChangeConfigValue(config,path,value):
    """
    Changes the value of a variable in nested dict config to value.
    The field in the dict-list structure is identified by a list path.
    Example: to change the following field in config dict to value:
    conf['lvl1'][0]['lvl3']
    use the follwing path=['lvl1',0,'lvl2']
    Arguments
    ---------
        config      an object with dicts and lists
        path        a list of strings and integers, pointing to the field in config that should
                    be changed, see Example above
        Value       new value for this field
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
    config              the config object, as read by yaml
    """

    # loop over parameters to vary
    for param_name in config['vary_params'].keys():

        # reset the images
        results_pht, results_fft = ( None , None )
        
        # get more info for the parmaeter
        param = config['vary_params'][param_name]
        # loop over all values of the parameter, which will be changed
        for iv,value in enumerate(param['values']):
            # copy the config to the original
            changed_config = config.copy()
            # perform the change
            ChangeConfigValue(changed_config,param['path'],value)
            logging.info('changed parameter %s to %s' % (param_name,str(value)))
            # run the photon vs fft test on the changed configs

            if param['rebuild_fft'] or results_fft == None:
                logger.info('getting FFT results for galaxy %d' % iv)
                changed_config1 = copy.deepcopy(changed_config)
                results_fft = GetResultsFFT(changed_config1)
                
            if param['rebuild_pht'] or results_pht == None:
                logger.info('getting photon results for galaxy %d' % iv)
                changed_config2 = copy.deepcopy(changed_config)
                results_pht = GetResultsPhoton(changed_config2)             

            # get the results filename
            filename_results = 'results.%s.%s.%03d.cat' % (config['filename_config'],param_name,iv)
            # save the results
            SaveResults(filename_results,results_fft=results_fft,results_pht=results_pht)
            logging.info('saved results for varied parameter %s with value %s, filename %s' % (param_name,str(value),filename_results))

if __name__ == "__main__":


    description = 'Compare FFT vs Photon shooting. Use the galaxies specified in the corresponding yaml file (see photon_vs_fft.yaml for an example)'

    # parse arguments
    parser = argparse.ArgumentParser(description=description, add_help=True)
    parser.add_argument('filename_config', type=str, help='yaml config file, see photon_vs_fft.yaml for example.')
    parser.add_argument('--debug', action="store_true", help='run with debug verbosity', default=False)
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


    logger.info('running photon_vs_fft for varied parameters')

    # run the config including changing of the parameters
    RunComparisonForVariedParams(config)

    # run test
    # results = runComparison(config)
    # save the results
    # filename_output = 'results.test.cat'
    # SaveResults(filename_output,results)


