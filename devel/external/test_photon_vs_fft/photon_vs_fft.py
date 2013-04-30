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

HSM_ERROR_VALUE = -99
NO_PSF_VALUE    = -98

def SaveResults(filename_output,results_all_gals):
    """
    Save results to file.
    Arguments
    ---------
    filename_output     - file to which results will be written
    results_all_gals    - list of dictionaries with Photon vs FFT resutls. 
                            See functon testPhotonVsFft for details of the dict.
    """


    # initialise the output file
    file_output = open(filename_output,'w')
       
    # write the header
    output_row_fmt = '%d\t% 2.6f\t% 2.6f\t% 2.6f\t% 2.6f\t% 2.6f\t' + \
                        '% 2.6f\t% 2.6f\t% 2.6f\t% 2.6f\t% 2.6f\t% 2.6f\t% 2.6f\t% 2.6f\n'
    output_header = '# id max_diff_over_max_image ' +  \
                                'E1_moments_fft E2_moments_fft E1_moments_photon E2_moments_photon ' + \
                                'E1_corr_fft E2_hsm_corr_fft E1_corr_photon E2_corr_photon ' + \
                                'moments_fft_sigma moments_photon_sigma corr_fft_sigma corr_photon_sigma\n'
    file_output.write(output_header)

    # loop over the results items
    for res in results_all_gals:

        # write the result item
        file_output.write(output_row_fmt % (res['ident'], res['max_diff_over_max_image'], 
            res['moments_fft_e1'], res['moments_fft_e2'], res['moments_phot_e1'], res['moments_phot_e2'],
            res['hsmcorr_fft_e1'], res['hsmcorr_fft_e2'], res['hsmcorr_phot_e1'], res['hsmcorr_phot_e2'],
            res['moments_fft_sigma'], res['moments_phot_sigma'], res['hsmcorr_fft_sigma'], res['hsmcorr_phot_sigma']
            ))
    
    logging.info('saved results file %s' % (filename_output))


def GetShapeMeasurements(image_gal_phot, image_gal_fft , image_psf, ident):
    """
    For a galaxy images created using photon and fft, and PSF image, measure HSM weighted moments 
    with and without PSF correction. Also measure sizes and maximum pixel differences.
    Arguments
    ---------
    image_gal_phot      image of the galaxy created using photon shooting
    image_gal_fft       image of the galaxy created using fft
    image_psf           image of the PSF
    ident               id of the galaxy, to be saved in the output file
    
    Outputs dictionary containing the results of comparison.
    The dict contains fields:
    max_diff_over_max_image,moments_fft_e1,moments_fft_e2,moments_phot_e1,moments_phot_e2,
    hsm_corr_fft_e1,hsm_corr_fft_e2,hsm_corr_phot_e1,hsm_corr_phot_e2,
    moments_fft_sigma,moments_phot_sigma,hsm_fft_sigma,hsm_phot_sigma,
    image_fft,image_phot,image_psf,ident
    If an error occured in HSM measurement, then a error value is written in the fields of this
    dict.
    """

    # create a residual image
    diff_image = image_gal_phot.array - image_gal_fft.array
    max_diff_over_max_image = abs(diff_image.flatten()).max()/image_gal_fft.array.flatten().max()
    logger.debug('max(residual) / max(image_fft) = %2.4e ' % ( max_diff_over_max_image )  )

    # find adaptive moments
    try:
        moments_phot = galsim.FindAdaptiveMom(image_gal_phot)
        moments_phot_e1 = moments_phot.observed_shape.getE1()
        moments_phot_e2 = moments_phot.observed_shape.getE2()
        moments_phot_sigma = moments_phot.moments_sigma 
    except:
        logging.error('hsm error in moments measurement for phot image of galaxy %s' % str(ident))
        moments_phot_e1 = HSM_ERROR_VALUE
        moments_phot_e2 = HSM_ERROR_VALUE
        moments_phot_sigma = HSM_ERROR_VALUE
    try:
        moments_fft = galsim.FindAdaptiveMom(image_gal_fft)
        moments_fft_e1 = moments_fft.observed_shape.getE1()
        moments_fft_e2 = moments_fft.observed_shape.getE2()
        moments_fft_sigma = moments_fft.moments_sigma 
    except:
        logging.error('hsm error in moments measurement for fft image of galaxy %s' % str(ident))
        moments_fft_e1 = HSM_ERROR_VALUE
        moments_fft_e2 = HSM_ERROR_VALUE
        moments_fft_sigma = HSM_ERROR_VALUE

    logger.debug('adaptive moments phot   E1=% 2.6f\t2=% 2.6f\tsigma=%2.6f' % (moments_phot_e1, moments_phot_e2, moments_phot_sigma)) 
    logger.debug('adaptive moments fft     E1=% 2.6f\t2=% 2.6f\tsigma=%2.6f' % (moments_fft_e1, moments_fft_e2, moments_fft_sigma))

    # find HSM moments
    if image_psf == None:

        hsm_corr_phot_e1 =  hsm_corr_phot_e2  = hsm_corr_fft_e1 = hsm_corr_fft_e2 =\
        hsm_fft_sigma = hsm_phot_sigma = \
        NO_PSF_VALUE 
    
    else:

        try:
            hsmcorr_phot = galsim.EstimateShearHSM(image_gal_phot,image_psf,strict=True)
            hsmcorr_phot_e1 = hsmcorr_phot.corrected_e1
            hsmcorr_phot_e2 = hsmcorr_phot.corrected_e2
            hsmcorr_phot_sigma   = hsmcorr_phot.moments_sigma
        except:
            logger.error('hsm error in hsmcorr measurement of phot image of galaxy %s' % str(ident))
            hsmcorr_phot_e1 = HSM_ERROR_VALUE
            hsmcorr_phot_e2 = HSM_ERROR_VALUE
            hsmcorr_phot_sigma   = HSM_ERROR_VALUE

        try:
            hsmcorr_fft   = galsim.EstimateShearHSM(image_gal_fft,image_psf,strict=True)
            hsmcorr_fft_e1 = hsmcorr_fft.corrected_e1
            hsmcorr_fft_e2 = hsmcorr_fft.corrected_e2
            hsmcorr_fft_sigma   = hsmcorr_fft.moments_sigma
        except:
            logger.error('hsm error in hsmcorr measurement of fft image of galaxy %s' % str(ident))
            hsmcorr_fft_e1 = HSM_ERROR_VALUE
            hsmcorr_fft_e2 = HSM_ERROR_VALUE
            hsmcorr_fft_sigma   = HSM_ERROR_VALUE
        
        logger.debug('hsm corrected moments fft     E1=% 2.6f\tE2=% 2.6f\tsigma=% 2.6f' % (hsmcorr_fft_e1, hsmcorr_fft_e2, hsmcorr_fft_sigma))
        logger.debug('hsm corrected moments phot    E1=% 2.6f\tE2=% 2.6f\tsigma=% 2.6f' % (hsmcorr_phot_e1, hsmcorr_phot_e2, hsmcorr_phot_sigma))
          
    # create the output dictionary
    result={}
    result['max_diff_over_max_image'] = max_diff_over_max_image
    result['moments_fft_e1'] = moments_fft_e1
    result['moments_fft_e2'] = moments_fft_e2
    result['moments_phot_e1'] = moments_phot_e1
    result['moments_phot_e2'] = moments_phot_e2
    result['hsmcorr_fft_e1'] = hsmcorr_fft_e1
    result['hsmcorr_fft_e2'] = hsmcorr_fft_e2
    result['hsmcorr_phot_e1'] = hsmcorr_phot_e1
    result['hsmcorr_phot_e2'] = hsmcorr_phot_e2
    result['moments_fft_sigma'] = moments_fft_sigma
    result['moments_phot_sigma'] = moments_phot_sigma
    result['hsmcorr_fft_sigma'] = hsmcorr_fft_sigma
    result['hsmcorr_phot_sigma'] = hsmcorr_phot_sigma
    result['ident'] = ident

    return result


def RunComparison(config,rebuild_pht,rebuild_fft): 
    """
    Runs the photon vs FFT comparison. Returns a list of dicts, containing the results of moments 
    (and other parameters) measured from the FFT and photon images, for each galaxy.
    
    Arguments
    ---------
    config          the yaml config used to create images

    Outputs a list of dictionaries containing the results of comparison.
    """

    # First process the input field:
    galsim.config.ProcessInput(config)

    # Build the image which has dimensions of
    # x: number of galaxies in the config, y: 2 - one for phot and one for fft

    global img_gals_fft, img_gals_pht, img_psfs_fft
    
    try:

        if rebuild_fft or img_gals_fft==None:
            logger.info('building fft image')
            config['image']['draw_method'] = 'fft'
            img_gals_fft,img_psfs_fft,_,_ = galsim.config.BuildImage(config=config,make_psf_image=True)
        if rebuild_pht or img_gals_pht==None:
            logger.info('building phot image')
            config['image']['draw_method'] = 'phot'
            img_gals_pht,img_psfs_pht,_,_ = galsim.config.BuildImage(config=config,make_psf_image=True)
    except:
        logging.error('building image failed')
        return None

    nobjects = galsim.config.GetNObjForImage(config,0)
    npix = config['image']['stamp_size']

    # initialise the results dict
    results_all = []

    # measure the photon and fft images
    for i in range(nobjects):

        img_fft =  img_gals_fft[galsim.BoundsI(  1 ,   npix, i*npix+1, (i+1)*npix )]
        img_pht =  img_gals_pht[galsim.BoundsI(  1 ,   npix, i*npix+1, (i+1)*npix )]
        img_psf =  img_psfs_fft[galsim.BoundsI(  1 ,   npix, i*npix+1, (i+1)*npix )]

        if config['debug'] == True: SaveImagePlots(config,i,img_fft,img_pht,img_psf)

        # measure and get the resutls
        results = GetShapeMeasurements(img_fft,img_pht,img_psf,i)
        results_all.append(results)

    return results_all

def SaveImagePlots(config,id,img_fft,img_pht,img_psf):

    import pylab
    pylab.figure()

    pylab.subplot(1,4,1)
    pylab.imshow(img_fft.array,interpolation='nearest')
    pylab.colorbar()
    pylab.title('fft image')
    
    pylab.subplot(1,4,2)
    pylab.imshow(img_pht.array,interpolation='nearest')
    pylab.colorbar()
    pylab.title('photon image')
    
    pylab.subplot(1,4,3)
    pylab.imshow(img_fft.array-img_pht.array,interpolation='nearest')
    pylab.colorbar()
    
    pylab.subplot(1,4,4)
    pylab.imshow(img_psf.array,interpolation='nearest')
    pylab.colorbar()
    pylab.title('PSF image')
    
    filename_fig = 'fig.images.%s.%03d.png' % (config['filename_config'],id)
    pylab.savefig(filename_fig)
    logger.debug('saved figure %s' % filename_fig)

    pylab.close()

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
        global img_gals_fft, img_gals_pht, img_psfs_fft
        (img_gals_fft, img_gals_pht, img_psfs_fft) = (None,None,None)

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
            results = RunComparison(changed_config,param['rebuild_pht'],param['rebuild_fft'])
            if results == None:
                logging.error('failed to get results for these settings')
                continue
            # get the results filename
            filename_results = 'results.%s.%s.%03d.cat' % (config['filename_config'],param_name,iv)
            # save the results
            SaveResults(filename_results,results)
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


