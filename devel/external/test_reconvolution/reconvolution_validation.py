import os
import pdb
import pyfits
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

HSM_ERROR_VALUE = -99
NO_PSF_VALUE    = -98

def CreateRGC(config):

    cosmos_config = copy.deepcopy(config['cosmos_images']);
    cosmos_config['image']['gsparams'] = copy.deepcopy(config['gsparams'])
    galsim.config.Process(cosmos_config,logger=None)
    # this is a hack - there should be a better way to get this number
    n_gals = len(galsim.config.GetNObjForMultiFits(cosmos_config,0,0))
    logger.info('building real galaxy catalog with %d galaxies' % n_gals)

    filename_gals = os.path.join(config['cosmos_images']['output']['dir'],
        config['cosmos_images']['output']['file_name'])
    filename_psfs = os.path.join(config['cosmos_images']['output']['dir'],
        config['cosmos_images']['output']['psf']['file_name'])
    pixel_scale = config['cosmos_images']['image']['pixel_scale']
    noise_var = config['cosmos_images']['image']['noise']['variance']
    filename_rgc = os.path.join(
        config['reconvolved_images']['input']['real_catalog']['dir'],
        config['reconvolved_images']['input']['real_catalog']['file_name'])

    BAND = 'F814W'
    MAG  = 20
    WEIGHT = 1

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

    if config['debug'] : PreviewRGC(config,filename_rgc)

def PreviewRGC(config,filename_rgc):
    """
    Function for eyeballing the contents of the created mock RGC catalogs.
    Arguments
    ---------
    config              config dict 
    filename_rgc        filename of the newly created real galaxy catalog fits 
    """

    table = pyfits.open(filename_rgc)[1].data
    print table

    fits_gal = table[0]['GAL_FILENAME']
    fits_psf = table[0]['PSF_FILENAME']
    import pylab

    for n in range(10):

        img_gal = pyfits.getdata(fits_gal,ext=n)
        img_psf = pyfits.getdata(fits_psf,ext=n)

        pylab.subplot(1,2,1)
        pylab.imshow(img_gal,interpolation='nearest')
        
        pylab.subplot(1,2,2)
        pylab.imshow(img_psf,interpolation='nearest')
        
        filename_fig = 'fig.previewRGC.%s.%d.png' % (config['filename_config'],n)

        pylab.savefig(filename_fig)


def GetReconvImage(config):

    reconv_config = copy.deepcopy(config['reconvolved_images'])
    reconv_config['image']['gsparams'] = copy.deepcopy(config['gsparams'])
    reconv_config['input']['catalog'] = copy.deepcopy(config['cosmos_images']['input']['catalog'])
    reconv_config['gal']['shift'] = copy.deepcopy(config['cosmos_images']['gal']['shift'])
    
    galsim.config.ProcessInput(reconv_config)
    # this outputs 4 objects, whereas direct outputs 5, not clear why
    img_gals,img_psfs,_,_ = galsim.config.BuildImage(config=reconv_config,make_psf_image=True)

    return (img_gals,img_psfs)

def GetDirectImage(config):

    direct_config = copy.deepcopy(config['reconvolved_images'])
    direct_config['image']['gsparams'] = copy.deepcopy(config['gsparams'])
    # switch gals to the original cosmos gals
    direct_config['gal'] = copy.deepcopy(config['cosmos_images']['gal'])  
    direct_config['gal']['shear'] = copy.deepcopy(config['reconvolved_images']['gal']['shear'])  
    direct_config['input'] = copy.deepcopy(config['cosmos_images']['input'])
    
    galsim.config.ProcessInput(direct_config)
    # this outputs 5 objects, whereas reconv outputs 4, not clear why
    img_gals,img_psfs,_,_ = galsim.config.BuildImage(config=direct_config,make_psf_image=True)

    return (img_gals,img_psfs)

def GetShapeMeasurements(image_gal, image_psf, ident):
    """
    For a galaxy images created using photon and fft, and PSF image, measure HSM weighted moments 
    with and without PSF correction. Also measure sizes and maximum pixel differences.
    Arguments
    ---------
    image_gal           image of the galaxy
    image_psf           image of the PSF
    ident               id of the galaxy, to be saved in the output file
    
    Outputs dictionary containing the results of comparison.
    The dict contains fields:
    If an error occured in HSM measurement, then a error value is written in the fields of this
    dict.
    """

    # find adaptive moments
    try:
        moments = galsim.FindAdaptiveMom(image_gal)
        moments_e1 = moments.observed_shape.getE1()
        moments_e2 = moments.observed_shape.getE2()
        moments_sigma = moments.moments_sigma 
    except:
        logging.error('hsm error in moments measurement for phot image of galaxy %s' % str(ident))
        moments_e1 = HSM_ERROR_VALUE
        moments_e2 = HSM_ERROR_VALUE
        moments_sigma = HSM_ERROR_VALUE
    
    logger.debug('adaptive moments   E1=% 2.6f\t2=% 2.6f\tsigma=%2.6f' % (moments_e1, moments_e2, moments_sigma)) 
    
    # find HSM moments
    if image_psf == None:

        hsm_corr_e1 =  hsm_corr_e2  = hsm_corr_fft_e1 = hsm_corr_fft_e2 =\
        hsm_fft_sigma = hsm_sigma = \
        NO_PSF_VALUE 
    
    else:

        try:
            hsm = galsim.EstimateShearHSM(image_gal,image_psf,strict=True)
            hsm_corr_e1 = hsm.corrected_e1
            hsm_corr_e2 = hsm.corrected_e2
            hsm_sigma   = hsm.moments_sigma
        except:
            logger.error('hsm error in hsmcorr measurement of phot image of galaxy %s' % str(ident))
            hsm_corr_e1 = HSM_ERROR_VALUE
            hsm_corr_e2 = HSM_ERROR_VALUE
            hsm_sigma   = HSM_ERROR_VALUE

        
        logger.debug('hsm corrected moments    E1=% 2.6f\tE2=% 2.6f\tsigma=% 2.6f' % (hsm_corr_e1, hsm_corr_e2, hsm_sigma))
          
    # create the output dictionary
    result={}
    result['moments_e1'] = moments_e1
    result['moments_e2'] = moments_e2
    result['hsmcorr_e1'] = hsm_corr_e1
    result['hsmcorr_e2'] = hsm_corr_e2
    result['moments_sigma'] = moments_sigma
    result['hsmcorr_sigma'] = hsm_sigma
    result['ident'] = ident

    return result

def GetPixelDifference(image1,image2,id):
    """
    Returns ratio of maximum pixel difference of two images 
    to the value of the maximum of pixels in the first image.
    Arguments
    ---------
    image1
    image2      images to compare
    """
    # create a residual image
    diff_image = image1.array - image2.array
    # calculate the ratio
    max_diff_over_max_image = abs(diff_image.flatten()).max()/image1.array.flatten().max()
    logger.debug('max(residual) / max(image1) = %2.4e ' % ( max_diff_over_max_image )  )
    return { 'diff' : max_diff_over_max_image, 'ident' :id }


def SaveResults(filename_output,results_direct,results_reconv,results_imdiff):
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
                                'E1_moments_direct E2_moments_direct E1_moments_reconv E2_moments_reconv ' + \
                                'E1_corr_direct E2_hsm_corr_direct E1_corr_reconv E2_corr_reconv ' + \
                                'moments_direct_sigma moments_reconv_sigma corr_direct_sigma corr_reconv_sigma\n'
    file_output.write(output_header)

    # loop over the results items
    for i,r in enumerate(results_direct):

        # write the result item
        file_output.write(output_row_fmt % (
            results_direct[i]['ident'], 
            results_imdiff[i]['diff'],
            results_direct[i]['moments_e1'], 
            results_direct[i]['moments_e2'], 
            results_reconv[i]['moments_e1'],  
            results_reconv[i]['moments_e2'],
            results_direct[i]['hsmcorr_e1'], 
            results_direct[i]['hsmcorr_e2'], 
            results_reconv[i]['hsmcorr_e1'], 
            results_reconv[i]['hsmcorr_e2'],
            results_direct[i]['moments_sigma'], 
            results_reconv[i]['moments_sigma'], 
            results_direct[i]['hsmcorr_sigma'], 
            results_reconv[i]['hsmcorr_sigma']
            ))

    file_output.close()
    logging.info('saved results file %s' % (filename_output))

def RunComparison(config,rebuild_reconv,rebuild_direct):

    # try:
    CreateRGC(config)

    # except:
        # raise ValueError('creating RGC failed')

    try:
        if rebuild_reconv or img_gals_reconv==None:
            logger.info('building reconv image')
            (img_reconv,psf_reconv) = GetReconvImage(config)
    except:
        logging.error('building recov image failed')
        return None

    try:
        if rebuild_direct or img_gals_direct==None:
            logger.info('building direct image')
            (img_direct,psf_direct) = GetDirectImage(config)
    except:
        logging.error('building direct image failed')
        return None
  
    # get image size
    npix = config['reconvolved_images']['image']['stamp_size']
    nobjects = galsim.config.GetNObjForImage(config['reconvolved_images'],0)

    # initalise results lists
    results_shape_reconv = []
    results_shape_direct = []
    results_pixel_imdiff = []

    for i in range(nobjects):

        # cut out stamps
        img_gal_reconv =  img_reconv[ galsim.BoundsI(  1 ,   npix, i*npix+1, (i+1)*npix ) ]
        img_gal_direct =  img_direct[ galsim.BoundsI(  1 ,   npix, i*npix+1, (i+1)*npix ) ]
        img_psf_direct =  psf_direct[ galsim.BoundsI(  1 ,   npix, i*npix+1, (i+1)*npix ) ]

        if config['debug']: SaveImagesPlots(config, i, img_gal_reconv, img_gal_direct, img_psf_direct)

        # get shapes
        result_reconv = GetShapeMeasurements(img_gal_reconv, img_psf_direct, i)
        result_direct = GetShapeMeasurements(img_gal_direct, img_psf_direct, i)
        result_imdiff = GetPixelDifference(img_gal_reconv,img_gal_direct, i)

        # append the shape results to list
        results_shape_reconv.append(result_reconv)
        results_shape_direct.append(result_direct)
        results_pixel_imdiff.append(result_imdiff)

    return results_shape_direct,results_shape_reconv,results_pixel_imdiff

def SaveImagesPlots(config,id,img_reconv,img_direct,img_psf):

    fig_xsize,fig_ysize = 30,15

    import pylab
    pylab.figure(figsize=(fig_xsize,fig_ysize))
    pylab.subplot(1,4,1)
    pylab.imshow(img_reconv.array)
    # pylab.colorbar()
    pylab.title('reconvolved image')

    pylab.subplot(1,4,2)
    pylab.imshow(img_direct.array)
    # pylab.colorbar()
    pylab.title('direct image')
    
    pylab.subplot(1,4,3)
    pylab.imshow(img_direct.array - img_reconv.array)
    # pylab.colorbar()
    pylab.title('direct - reconv')

    pylab.subplot(1,4,4)
    pylab.imshow(img_psf.array)
    # pylab.colorbar()
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
    Runs the comparison of direct and reconv convolution methods, producing results file for each of the 
    varied parameters in the config file, under key 'vary_params'.
    Produces a results file for each parameter and it's distinct value.
    The filename of the results file is: 'results.yaml_filename.param_name.param_value_index.cat'
    Arguments
    ---------
    config              the config object, as read by yaml
    filename_config     name of the config file used
    """

    # loop over parameters to vary
    for param_name in config['vary_params'].keys():

        # reset the images
        global img_gals_direct, img_gals_reconv, img_psfs_direct
        (img_gals_direct, img_gals_reconv, img_psfs_direct) = (None,None,None)

        # get more info for the parmaeter
        param = config['vary_params'][param_name]
        # loop over all values of the parameter, which will be changed
        for iv,value in enumerate(param['values']):
            # copy the config to the original
            changed_config = copy.deepcopy(config)
            # perform the change
            ChangeConfigValue(changed_config,param['path'],value)
            logging.info('changed parameter %s to %s' % (param_name,str(value)))
            # run the photon vs fft test on the changed configs
            results_direct,results_reconv,results_pixel_imdiff = RunComparison(
                    changed_config,param['rebuild_reconv'],param['rebuild_direct'])
            # if getting images failed, continue with the loop
            if results_direct == None:  continue
            # get the results filename
            filename_results = 'results.%s.%s.%03d.cat' % (config['filename_config'],param_name,iv)
            # save the results
            SaveResults(filename_results,results_direct,results_reconv,results_pixel_imdiff)
            logging.info('saved results for varied parameter %s with value %s, filename %s' % (param_name,str(value),filename_results))


if __name__ == "__main__":

    description = 'Compare reconvolved and directly created galaxies.'

    # parse arguments
    parser = argparse.ArgumentParser(description=description, add_help=True)
    parser.add_argument('filename_config', type=str,
                 help='yaml config file, see reconvolution_validation.yaml for example.')
    parser.add_argument('--debug', action="store_true", help='run in debug mode', default=False)
    args = parser.parse_args()

    # set up logger
    if args.debug:  logger_level = 'logging.DEBUG'
    else: logger_level = 'logging.INFO'
    logging.basicConfig(format="%(message)s", level=eval(logger_level), stream=sys.stdout)
    logger = logging.getLogger("reconvolution_validation") 

    # load the configuration file
    config = yaml.load(open(args.filename_config,'r'))
    config['filename_config'] = args.filename_config
    config['debug'] = args.debug

    # run comparison for default parameter set and save results
    # results_shape_reconv,results_shape_direct,results_pixel_imdiff = RunComparison(config,True,True)
    # SaveResults('results.test.cat',results_shape_reconv,results_shape_direct,results_pixel_imdiff)

    # run the parameter comparison
    RunComparisonForVariedParams(config)





