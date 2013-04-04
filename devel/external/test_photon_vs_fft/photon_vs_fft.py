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

hsm_error_value = -1
no_psf_value = -1
dirname_figs = 'figures'



def getGSObjectSersicsSample(i):
    """
    Gets all the objects needed to draw images of Sersic galaxy with index i, supplied in the Sersics sample.
    Requiers a global sersic_sample_catalog to be initialised earlier.
    Returns gso containig fields 'gal','psf','pix','galpsf'.
    The PSF used here can be either 'ground' or 'space', 
    is configured in the config['sersics_sample']['psf_type'].
    Parameters of both PSFs are hardcoded here.
    The angle for the galaxy is random.
    The centroid shift is random uniform, according to configuration in config['sersics_sample'].
    PSF ellipticity is random uniform, according to configuration in config['sersics_sample'].
    """

    # init random numbers
    random_seed = 1512413
    rng = galsim.UniformDeviate(random_seed) 

    # ident  n_sersic  half_light_radius [arcsec] |g|
    pix = galsim.Pixel(pixel_scale)

    # get sersic profile parameter
    sersic_index=sersic_sample_catalog[i,1]
    if sersic_index < 0.5 or sersic_index > 4.2:
        logger.warning('skipping galaxy %d in the Sersics catalogue - value %2.2f is out of range [0.5,4.2]' % (i,sersic_index))
        return getGSObjectSersicsSample(i)

    # get rest of parameters
    hlr = sersic_sample_catalog[i,2]
    g = sersic_sample_catalog[i,3]

    # build the galaxy
    profile = galsim.Sersic(n=sersic_index, half_light_radius=hlr)
    beta = 2.*math.pi * rng() * galsim.radians
    profile.applyShear(g=g,beta=beta)
    dx = rng()* config['sersics_sample']['max_random_dx'] * pixel_scale
    dy = rng()* config['sersics_sample']['max_random_dx'] * pixel_scale
    profile.applyShift(dx=dx, dy=dy)

    # build the PSF
    psf_g = rng() * config['sersics_sample']['max_random_psf_g']
    psf_beta = 2.*math.pi * rng() * galsim.radians

    if observe_from == 'space':
        lam=700.
        diam=1.3
        psf = galsim.Airy(lam_over_diam=lam/diam/1.e9*206265.) 
    elif observe_from == 'ground':
        fwhm = 0.65 
        atmos = galsim.Moffat(beta=3,fwhm=fwhm)
        optics = galsim.Airy(lam_over_diam = (700e-9/4)*(180/numpy.pi)*3600)
        psf = galsim.Convolve([atmos,optics],gsparams=gsp)
    else:
        raise ValueError('%s in unknown psf_type. Use \'space\' or \'ground\'' % config['image']['observe_from'])

    psf.applyShear(g=g,beta=psf_beta)

    # build the output dict
    gso = {}
    gso['pix'] = pix
    gso['gal'] = profile
    gso['psf'] = psf
    gso['galpsf'] = galsim.Convolve([gso['gal'],gso['psf']],gsparams=gsp)

    # add the identifier field from the COSMOS catalog
    gso['ident'] = sersic_sample_catalog[i,0]

    return gso


def getGSObjectListGals(i):
    """
    This function generates all GSObjects which are necessary to create an image of an object.
    This involves a galaxy, PSF and pixel kernels.

    Input:
    i - identifier of the object in the config file, under key list_gals 

    Output:
    gso - dictionary with GSObjects gso['gal'], gso['psf'], gso['pix'] 
    """
    
    # initialise the output dict            
    gso = {}

    # get the dictionary information about object i
    obj = config['list_gals'][i]

    # get the galaxy if it is a sersics bulge + disc model          
    logger.debug('creating galaxy with PSF type %s' % (obj['psf']['type']))

    bulge = galsim.Sersic(n=obj['bulge']['sersic_index'], half_light_radius=obj['bulge']['half_light_radius'])
    disc  = galsim.Sersic(n=obj['disc' ]['sersic_index'], half_light_radius=obj['disc']['half_light_radius'])
    
    bulge.applyShear(g1=obj['bulge']['g1'],g2=obj['bulge']['g2'])
    disc.applyShear( g1=obj['disc' ]['g1'],g2=obj['disc' ]['g2'])

    bulge.applyShift(dx=obj['bulge']['dx'],dy=obj['bulge']['dy'])
    disc.applyShift(dx=obj['disc']['dx'],dy=obj['disc']['dy'])

    bulge.setFlux(obj['bulge']['flux'])
    disc.setFlux(obj['disc']['flux'])

    gso['gal'] = bulge + disc
    
    # get pixel kernel
    gso['pix'] = galsim.Pixel(pixel_scale)

    # get the PSF
    if obj['psf']['type'] == 'none':

            gso['psf'] = None
            gso['galpsf'] = gso['gal']

    else:

        if obj['psf']['type'] == 'Moffat':
                
            gso['psf'] = galsim.Moffat(fwhm=obj['psf']['fwhm'],beta=obj['psf']['beta'])

        elif obj['psf']['type'] == 'Airy':

            gso['psf'] = galsim.Airy(lam_over_diam=obj['psf']['lam_over_diam'])

        elif obj['psf']['type'] == 'AtmosphericPSF':

            gso['psf'] = galsim.AtmosphericPSF(fwhm=obj['psf']['fwhm'])
        else:
            logger.error('unknown PSF type, use {Moffat,Airy,AtmosphericPSF,none}')
            sys.exit() 

        gso['psf'].applyShear(g1=obj['psf']['g1'],g2=obj['psf']['g2'])
        gso['psf'].setFlux(1.)
        gso['galpsf'] = galsim.Convolve([gso['gal'],gso['psf']],gsparams=gsp)

        # add the identifier field - set it equal to the index in the list
        gso['ident'] = i


    return gso

def getMeasurements(gso,gso_name='test'):
    """
    Return the measured moments of the galaxies provied in the gso dict. Create the images using Photon and FFT, 
    measure HSM observerd and corrected moments and other statistics.
    
    Arguments
    ---------
        gso                - a dictionary with fields 'gal', 'psf', 'pix'. These are GSObjects which are ready to be drawing
        gso_name = test    - string or number. If plots are being created, then this name will be added to the plot filename.

    Returns a dictionary with the results measured from the Photon and FFT images, or in case of failure, returns None.
    Fields in the output dictionary: 
    max_diff_over_max_image, moments_fft_e1, moments_fft_e2, moments_phot_e1, moments_phot_e2, 
    hsm_corr_fft_e1, hsm_corr_fft_e2, hsm_corr_phot_e1, hsm_corr_phot_e2, 
    moments_fft_sigma, moments_phot_sigma, hsm_fft_sigma, hsm_phot_sigma, 
    image_fft, image_phot, image_psf
    """

    # in case there is no PSF supplied:
    if gso['psf'] != None:
        
        # create the PSF

        if config['psf_draw_method'] == 'fft':
            logger.debug('drawing PSF using draw()')
            image_psf = galsim.ImageF(config['image']['n_pix'],config['image']['n_pix'])
            final_psf = galsim.Convolve([gso['psf'],gso['pix']],gsparams=gsp)
            final_psf.draw(image_psf,dx=pixel_scale)
        elif config['psf_draw_method'] == 'shoot':
            logger.debug('drawing PSF using drawShoot and %2.0e photons' % config['image']['n_photons'])
            image_psf = galsim.ImageF(config['image']['n_pix'],config['image']['n_pix'])
            gso['psf'].drawShoot(image_psf,dx=pixel_scale,n_photons=config['image']['n_photons'])
        else:
            logger.error('%s not a valid PSF drawing method, use \'fft\' or \'shoot\' in the config file' % config['psf_draw_method'])
            sys.exit() 

    # create shoot image
    logger.debug('drawing galaxy using drawShoot and %2.0e photons' % config['image']['n_photons'])
    image_gal_phot = galsim.ImageF(config['image']['n_pix'],config['image']['n_pix'])
    final = gso['galpsf']
    final.setFlux(1.)
    im = final.drawShoot(image_gal_phot,dx=pixel_scale,n_photons=config['image']['n_photons'])
    added_flux = im.added_flux
    logger.debug('drawShoot is ready for galaxy %s, added_flux=%f, scale=%f' % (gso_name,added_flux,pixel_scale) )

    # create fft image
    logger.debug('drawing galaxy using draw (FFT)')
    image_gal_fft = galsim.ImageF(config['image']['n_pix'],config['image']['n_pix'])
    final = galsim.Convolve([gso['galpsf'],gso['pix']],gsparams=gsp)
    final.setFlux(1.)
    final.draw(image_gal_fft,dx=pixel_scale)
    logger.debug('draw using FFT is ready for galaxy %s' % gso_name)
    
    # create a residual image
    diff_image = image_gal_phot.array - image_gal_fft.array
    max_diff_over_max_image = abs(diff_image.flatten()).max()/image_gal_fft.array.flatten().max()
    logger.debug('max(residual) / max(image_fft) = %2.4e ' % ( max_diff_over_max_image )  )

    # save plots if requested
    if args.save_plots:

        # plot the pixel differences
        pylab.figure()
        pylab.clf()
        
        pylab.subplot(131)
        pylab.imshow(image_gal_phot.array,interpolation='nearest')
        pylab.colorbar()
        pylab.title('image_phot')
        
        pylab.subplot(132)
        pylab.imshow(image_gal_fft.array,interpolation='nearest')
        pylab.colorbar()
        pylab.title('image_fft')
        
        pylab.subplot(133)
        pylab.imshow(diff_image,interpolation='nearest')
        pylab.colorbar()
        pylab.title('image_phot - image_fft')

        filename_fig = os.path.join(dirname_figs,'photon_vs_fft_gal%s.png' % str(gso_name))
        pylab.gcf().set_size_inches(20,10)
        pylab.savefig(filename_fig)
        pylab.close()
        logger.info('saved figure %s' % filename_fig)


    # find adaptive moments
    try:
        moments_phot = galsim.FindAdaptiveMom(image_gal_phot)
    except:
        logging.error('hsm error')
        return None

    try:
        moments_fft   = galsim.FindAdaptiveMom(image_gal_fft)
    except:
        logging.error('hsm error')
        return None

    moments_phot_e1 = moments_phot.observed_shape.getE1()
    moments_phot_e2 = moments_phot.observed_shape.getE2()
    moments_phot_sigma = moments_phot.moments_sigma        
    logger.debug('adaptive moments phot   E1=% 2.6f\t2=% 2.6f\tsigma=%2.6f' % (moments_phot_e1, moments_phot_e2, moments_phot_sigma))
   
    moments_fft_e1   = moments_fft.observed_shape.getE1()
    moments_fft_e2   = moments_fft.observed_shape.getE2()
    moments_fft_sigma = moments_fft.moments_sigma
    logger.debug('adaptive moments fft     E1=% 2.6f\t2=% 2.6f\tsigma=%2.6f' % (moments_fft_e1, moments_fft_e2, moments_fft_sigma))


    # find HSM moments
    if gso['psf'] == None:

        hsm_corr_phot_e1 =  hsm_corr_phot_e2  = hsm_corr_fft_e1 = hsm_corr_fft_e2 =\
        hsm_fft_sigma = hsm_phot_sigma = \
        no_psf_value 
    
    else:

        try:
            hsm_phot = galsim.EstimateShearHSM(image_gal_phot,image_psf,strict=True)
        except:
            logger.info('hsm error')
            return None

        try:
            hsm_fft   = galsim.EstimateShearHSM(image_gal_fft,image_psf,strict=True)
        except:
            logger.info('hsm error')
            return None


        # for photon

        # check if HSM has failed
        hsm_corr_phot_e1 = hsm_phot.corrected_e1
        hsm_corr_phot_e2 = hsm_phot.corrected_e2
        hsm_phot_sigma = hsm_phot.moments_sigma

        # for fft

        # check if HSM has failed         
        hsm_corr_fft_e1 = hsm_fft.corrected_e1
        hsm_corr_fft_e2 = hsm_fft.corrected_e2
        hsm_fft_sigma = hsm_fft.moments_sigma

        logger.debug('hsm corrected moments fft     E1=% 2.6f\tE2=% 2.6f\tsigma=% 2.6f' % (hsm_corr_fft_e1, hsm_corr_fft_e2, hsm_fft_sigma))
        logger.debug('hsm corrected moments phot    E1=% 2.6f\tE2=% 2.6f\tsigma=% 2.6f' % (hsm_corr_phot_e1, hsm_corr_phot_e2, hsm_phot_sigma))
          
    # create the output dictionary
    result={}
    result['max_diff_over_max_image'] = max_diff_over_max_image
    result['moments_fft_e1'] = moments_fft_e1
    result['moments_fft_e2'] = moments_fft_e2
    result['moments_phot_e1'] = moments_phot_e1
    result['moments_phot_e2'] = moments_phot_e2
    result['hsm_corr_fft_e1'] = hsm_corr_fft_e1
    result['hsm_corr_fft_e2'] = hsm_corr_fft_e1
    result['hsm_corr_phot_e1'] = hsm_corr_phot_e1
    result['hsm_corr_phot_e2'] = hsm_corr_phot_e1
    result['moments_fft_sigma'] = moments_fft_sigma
    result['moments_phot_sigma'] = moments_phot_sigma
    result['hsm_fft_sigma'] = hsm_fft_sigma
    result['hsm_phot_sigma'] = hsm_phot_sigma
    result['image_fft'] = image_gal_fft
    result['image_phot'] = image_gal_phot
    result['image_psf'] = image_psf

    return result

def testPhotonVsFft():
    """
    For all galaxies specified, produces the images using FFT and photon shooting.
    Saves plots of image residuals and shows differences in the moments measurements, if requested.
    Saves the adaptive moments and HSM results resutls to file if needed.
    If HSM fails, then the results for this case are not reported at all.
    The galaxies can be suplied either from:
    1) list_gals in the config file (see example photon_vs_fft.yaml)
    2) a catalog of single Sersic profiles (see cosmos_sersics_sample_N300.asc) and additional configuration in the config file
    """ 

    if config['image']['n_photons'] < 1e5:
        logger.warning('small number of photons, results may me meaningless')

    # decide if we use list of galaxies in the yaml file or a sersic catalog
    if config['use_galaxies_from'] == 'list_gals':
        n_gals = len(config['list_gals'])
        gso_next = getGSObjectListGals      
    elif config['use_galaxies_from'] == 'sersics_sample':
        global sersic_sample_catalog
        sersic_sample_catalog = numpy.loadtxt(config['sersics_sample']['catalog_filename'])

        sersic_max = 4.2
        sersic_min = 0.5

        select = numpy.logical_and(sersic_sample_catalog[:,1] > sersic_min , sersic_sample_catalog[:,1] < sersic_max)
        sersic_sample_catalog = sersic_sample_catalog[select,:]

        n_gals = sersic_sample_catalog.shape[0]
        gso_next = getGSObjectSersicsSample  
    else:
        raise ValueError('%s in unknown galaxies input form - use list_gals or sersics_sample' % use_galaxies_from )

    results_all = []   

    # loop through the galaxies
    for ig in range(n_gals):

        logger.debug('------------------ galaxy %g ------------------' % ig)

        # get the next galaxy
        gso = gso_next(ig)

        # get results from moments measurements
        results_gso = getMeasurements(gso)


        if results_gso == None:
            delGSO(gso)
            continue     

        # add the identifier 
        results_gso['ident'] = gso['ident']

        # remove gso for memory - if we keep more than 100 then I am getting strange memory errors
        delGSO(gso)
       
        results_all.append(results_gso)

    return results_all


def delGSO(gso):
    """
    Delete all GSObjects in the gso dict, and the dict itself.
    """

    for key in gso.keys():
        del gso[key]

    del gso

def saveResults(filename_output,results_all_gals):
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
       
    output_row_fmt = '%d\t% 2.6f\t% 2.6f\t% 2.6f\t% 2.6f\t% 2.6f\t' + \
                        '%2.6f\t% 2.6f\t% 2.6f\t% 2.6f\t% 2.6f\t% 2.6f\t% 2.6f\t% 2.6f\n'
    output_header = '# id max_diff_over_max_image ' +  \
                                'E1_moments_fft E2_moments_fft E1_moments_photon E2_moments_photon ' + \
                                'E1_hsm_corr_fft E2_hsm_corr_fft E1_hsm_corr_photon E2_hsm_corr_photon ' + \
                                'moments_fft_sigma moments_photon_sigma hsm_fft_sigma hsm_photon_sigma\n'

    file_output.write(output_header)

    # save the output in the file

    for ig,res in enumerate(results_all_gals):
        file_output.write(output_row_fmt % (res['ident'], res['max_diff_over_max_image'], 
                res['moments_fft_e1'], res['moments_fft_e2'], res['moments_phot_e1'],  res['moments_phot_e2'],
                res['hsm_corr_fft_e1'], res['hsm_corr_fft_e2'], res['hsm_corr_phot_e1'], res['hsm_corr_phot_e2'],
                res['moments_fft_sigma'], res['moments_phot_sigma'], res['hsm_fft_sigma'], res['hsm_phot_sigma']
                ))


def testGSParams():
    """
    Perform the Photon vs FFT test for different values of GSParams, as specified in the config file.
    Save results to the output files with the prefix specified in the config file.
    """

    # use global variables to change the GSParams and the observation mode 
    # set up defaults
    global observe_from
    observe_from = 'ground'
    global pixel_scale
    pixel_scale = 0.2
    global gsp
    gsp = galsim.GSParams()

    # loop through the observation modes (ground and space)
    for of,obs in enumerate(config['image']['observe_from']):

        # set the global variable
        observe_from = obs

        # set the global pixel scale for ground and space
        if observe_from == 'ground':
            pixel_scale = 0.2
        elif observe_from == 'space':
            pixel_scale = 0.03
        else: raise ValueError('%s is an invalid name for config[image][observe_from] - use \'space\' or \'ground\'')

        logging.info('using %s based observations' % observe_from)

        # loop through different GSParams
        for vary_gsp in config['gsparams']:

            # get the GSParam name that we want to change
            param_name = vary_gsp['name']

            # loop through different GSParams values
            for nw,new_value in enumerate(vary_gsp['grid']):

                # get the fiducial value for the parameters (just for user information)
                fiducial_value = eval('galsim.GSParams().%s' % param_name)
                logging.info('changing gsparam %s from fiducial %f to %f',param_name,fiducial_value,new_value)

                # set the parameter to new value
                if eval('type(gsp.%s)' % param_name) == int:
                    cmd = 'galsim.GSParams(%s=%d)' % (param_name,new_value)
                else:
                    cmd = 'galsim.GSParams(%s=%e)' % (param_name,new_value)
                gsp = eval(cmd)

                # perform the PhotonVsFFT test
                results_all_gals = testPhotonVsFft()

                # get the filename for the results file
                filename_out = '%s.%s.%s.%d.cat' % (config['filename_output'],observe_from,param_name,nw)

                # save the results
                saveResults(filename_out,results_all_gals)


if __name__ == "__main__":


    description = 'Compare FFT vs Photon shooting. Use the galaxies specified in the corresponding yaml file (see photon_vs_fft.yaml for an example)'

    # parse arguments
    parser = argparse.ArgumentParser(description=description, add_help=True)
    parser.add_argument('filename_config', type=str, help='yaml config file, see photon_vs_fft.yaml for example.')
    parser.add_argument('--save_plots', action="store_true", help='if to generate_images of galaxies and store them in ./images/', default=False)
    global args
    args = parser.parse_args()

    # set up logger
    logging.basicConfig(format="%(message)s", level=logging.INFO, stream=sys.stdout)
    logger = logging.getLogger("photon_vs_fft") 

    # load the configuration file
    filename_config = args.filename_config
    global config
    config = yaml.load(open(filename_config,'r'))


    # run the test
    # results_all_gals = testPhotonVsFft()
    # save the result
    # saveResults(config['filename_output'],results_all_gals)
    # testPhotonVsFft()

    # run the test
    testGSParams()


