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
    logger.info('creating galaxy with PSF type %s' % (obj['psf']['type']))

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
    gso['pix'] = galsim.Pixel(config['image']['scale'])

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
        gso['galpsf'] = galsim.Convolve([gso['gal'],gso['psf']])

    return gso


def testShootVsFft():
    """
    For all galaxies specified, produces the images using FFT and photon shooting.
    Saves plots of image residuals and shows differences in the moments measurements.
    Saves the adaptive moments and HSM results resutls to file if needed.
    If HSM fails or no PSF is used, then all the HSM measurements are set to -1.
    The galaxies can be suplied either from:
    1) list_gals in the config file (see example photon_vs_fft.yaml)
    2) a catalog of single Sersic profiles (see cosmos_sersics_sample_N300.asc) and additional configuration in the config file
    """ 

    if config['image']['n_photons'] < 1e5:
        logger.warning('small number of photons, results may me meaningless')

    # initialise the output file
    file_output = open(config['filename_output'],'w')
       
    output_row_fmt = '%d\t% 2.6f\t% 2.6f\t% 2.6f\t% 2.6f\t% 2.6f\t% 2.6f\t% 2.6f\t% 2.6f\t% 2.6f\t' + \
                        '%2.6f\t% 2.6f\t% 2.6f\t% 2.6f\t% 2.6f\t% 2.6f\t% 2.6f\t% 2.6f\n'
    output_header = '# id max_diff_over_max_image ' +  \
                                'G1_moments_fft G2_moments_fft G1_moments_photon G2_moments_photon ' + \
                                'E1_hsm_obs_fft E2_hsm_obs_fft E1_hsm_obs_photon E2_hsm_obs_photon ' + \
                                'E1_hsm_corr_fft E2_hsm_corr_fft E1_hsm_corr_photon E2_hsm_corr_photon ' + \
                                'moments_fft_sigma moments_shoot_sigma hsm_fft_sigma hsm_shoot_sigma\n'

    file_output.write(output_header)

    # decide if we use list of galaxies in the yaml file or a sersic catalog
    if config['use_galaxies_from'] == 'list_gals':
        n_gals = len(config['list_gals'])
        gso_next = getGSObjectListGals      
    elif config['use_galaxies_from'] == 'sersics_sample':
        global sersic_sample_catalog
        sersic_sample_catalog = numpy.loadtxt(config['sersics_sample']['catalog_filename'])
        n_gals = sersic_sample_catalog.shape[0]
        gso_next = getGSObjectSersicsSample  
    else:
        logger.error('%s in unknown galaxies input form - use list_gals or sersics_sample' % use_galaxies_from )
        sys.exit() 


    # loop through the galaxies
    for ig in range(n_gals):

        # temporarily disable more than 100 Sersics, because the limit imposed in the C code
        # if ig >= 100: break


        logger.info('------------------ galaxy %g ------------------' % ig)

        # get the next galaxy
        gso = gso_next(ig)

        # if somehow we counldn't get the next galaxy
        if gso == None: continue

        if gso['psf'] != None:
            
            # create the PSF

            if config['psf_draw_method'] == 'fft':
                logger.info('drawing PSF using draw()')
                image_psf = galsim.ImageF(config['image']['n_pix'],config['image']['n_pix'])
                final_psf = galsim.Convolve([gso['psf'],gso['pix']])
                final_psf.draw(image_psf,dx=config['image']['scale'])
            elif config['psf_draw_method'] == 'shoot':
                logger.info('drawing PSF using drawShoot and %2.0e photons' % config['image']['n_photons'])
                image_psf = galsim.ImageF(config['image']['n_pix'],config['image']['n_pix'])
                gso['psf'].drawShoot(image_psf,dx=config['image']['scale'],n_photons=config['image']['n_photons'])
            else:
                logger.error('%s not a valid PSF drawing method, use \'fft\' or \'shoot\' in the config file' % config['psf_draw_method'])
                sys.exit() 

        # create shoot image
        logger.info('drawing galaxy using drawShoot and %2.0e photons' % config['image']['n_photons'])
        image_gal_shoot = galsim.ImageF(config['image']['n_pix'],config['image']['n_pix'])
        final = gso['galpsf']
        final.setFlux(1.)
        (im, added_flux) = final.drawShoot(image_gal_shoot,dx=config['image']['scale'],n_photons=config['image']['n_photons'])
        logger.info('drawShoot is ready for galaxy %d, added_flux=%f, scale=%f' % (ig,added_flux,config['image']['scale']) )

        # create fft image
        logger.info('drawing galaxy using draw (FFT)')
        image_gal_fft = galsim.ImageF(config['image']['n_pix'],config['image']['n_pix'])
        final = galsim.Convolve([gso['galpsf'],gso['pix']])
        final.setFlux(1.)
        final.draw(image_gal_fft,dx=config['image']['scale'])
        logger.info('draw using FFT is ready for galaxy %s' % ig)
        
        # create a residual image
        diff_image = image_gal_shoot.array - image_gal_fft.array
        max_diff_over_max_image = abs(diff_image.flatten()).max()/image_gal_fft.array.flatten().max()

        # save plots if requested
        if args.save_plots:

            # plot the pixel differences
            pylab.figure()
            pylab.clf()
            
            pylab.subplot(131)
            pylab.imshow(image_gal_shoot.array,interpolation='nearest')
            pylab.colorbar()
            pylab.title('image_shoot')
            
            pylab.subplot(132)
            pylab.imshow(image_gal_fft.array,interpolation='nearest')
            pylab.colorbar()
            pylab.title('image_fft')
            
            pylab.subplot(133)
            pylab.imshow(diff_image,interpolation='nearest')
            pylab.colorbar()
            pylab.title('image_shoot - image_fft')

            filename_fig = os.path.join(dirname_figs,'photon_vs_fft_gal%d.png' % ig)
            pylab.gcf().set_size_inches(20,10)
            pylab.savefig(filename_fig)
            pylab.close()
            logger.info('saved figure %s' % filename_fig)


        # find adaptive moments
        moments_shoot = galsim.FindAdaptiveMom(image_gal_shoot)
        moments_fft   = galsim.FindAdaptiveMom(image_gal_fft)

        moments_shoot_g1 = moments_shoot.observed_shape.getG1()
        moments_shoot_g2 = moments_shoot.observed_shape.getG2()
        moments_shoot_sigma = moments_shoot.moments_sigma
        moments_fft_g1   = moments_fft.observed_shape.getG1()
        moments_fft_g2   = moments_fft.observed_shape.getG2()
        moments_fft_sigma = moments_fft.moments_sigma
        
        # display resutls

        logger.info('max(residual) / max(image_fft) = %2.4e ' % ( max_diff_over_max_image )  )
        logger.debug('adaptive moments fft     G1=% 2.6f\tG2=% 2.6f\tsigma=%2.6f' % (moments_fft_g1, moments_fft_g2, moments_fft_sigma))
        logger.debug('adaptive moments shoot   G1=% 2.6f\tG2=% 2.6f\tsigma=%2.6f' % (moments_shoot_g1, moments_shoot_g2, moments_shoot_sigma))


        # find HSM moments
        if gso['psf'] == None:

            hsm_obs_shoot_e1 = hsm_obs_shoot_e2 = hsm_obs_fft_e1 = hsm_obs_fft_e2 = \
            hsm_corr_shoot_e1 =  hsm_corr_shoot_e2  = hsm_corr_fft_e1 = hsm_corr_fft_e2 =\
            hsm_fft_sigma = hsm_shoot_sigma = \
            no_psf_value 
        
        else:

            hsm_shoot = galsim.EstimateShearHSM(image_gal_shoot,image_psf,strict=True)
            hsm_fft   = galsim.EstimateShearHSM(image_gal_fft,image_psf,strict=True)

            # check if HSM has failed
            if hsm_shoot.error_message != "":
                logger.debug('hsm_shoot failed with message %s' % hsm_shoot.error_message)
                hsm_obs_shoot_e1 = hsm_obs_shoot_e2  = \
                hsm_corr_shoot_e1 =  hsm_corr_shoot_e2  = hsm_shoot_sigma = \
                hsm_error_value
            else:
                hsm_obs_shoot_e1 = hsm_shoot.observed_shape.getE1()
                hsm_obs_shoot_e2 = hsm_shoot.observed_shape.getE2()
                hsm_corr_shoot_e1 = hsm_shoot.corrected_e1
                hsm_corr_shoot_e2 = hsm_shoot.corrected_e2
                hsm_shoot_sigma = hsm_shoot.moments_sigma

            # check if HSM has failed       
            if hsm_fft.error_message != "":
                logger.debug('hsm_fft failed with message %s' % hsm_fft.error_message)
                hsm_obs_fft_e1 = hsm_obs_fft_e2  = \
                hsm_corr_fft_e1 = hsm_corr_fft_e2  = hsm_fft_sigma = \
                hsm_error_value  
            else:
                hsm_obs_fft_e1 = hsm_fft.observed_shape.getE1()
                hsm_obs_fft_e2 = hsm_fft.observed_shape.getE2()
                hsm_corr_fft_e1 = hsm_fft.corrected_e1
                hsm_corr_fft_e2 = hsm_fft.corrected_e2
                hsm_fft_sigma = hsm_fft.moments_sigma

            logger.debug('hsm observed moments fft      E1=% 2.6f\tE2=% 2.6f' % (hsm_obs_fft_e1, hsm_obs_fft_e2))
            logger.debug('hsm observed moments shoot    E1=% 2.6f\tE2=% 2.6f' % (hsm_obs_shoot_e1, hsm_obs_shoot_e2))

            logger.debug('hsm corrected moments fft     E1=% 2.6f\tE2=% 2.6f' % (hsm_corr_fft_e1, hsm_corr_fft_e2))
            logger.debug('hsm corrected moments shoot   E1=% 2.6f\tE2=% 2.6f' % (hsm_corr_shoot_e1, hsm_corr_shoot_e2))
            
            logger.debug('hsm size sigma fft   % 2.6f' % hsm_fft_sigma)
            logger.debug('hsm size sigma shoot % 2.6f' % hsm_shoot_sigma)

            

        # save the output in the file
        file_output.write(output_row_fmt % (ig, max_diff_over_max_image, 
            moments_fft_g1,  moments_fft_g2, moments_shoot_g1,  moments_shoot_g2,
            hsm_obs_fft_e1, hsm_obs_fft_e2, hsm_obs_shoot_e1, hsm_obs_shoot_e2,
            hsm_corr_fft_e1, hsm_corr_fft_e2, hsm_corr_shoot_e1, hsm_corr_shoot_e2,
            moments_fft_sigma, moments_shoot_sigma, hsm_fft_sigma, hsm_shoot_sigma
            ))

        del_gso(gso)

# delete all GSObjects in the gso dict, and the dict itself
def del_gso(gso):

    for key in gso.keys():
        del gso[key]

    del gso



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
    pix = galsim.Pixel(config['image']['scale'])

    # get sersic profile parameter
    sersic_index=sersic_sample_catalog[i,1]
    if sersic_index < 0.5 or sersic_index > 4.2:
        logger.warning('skipping galaxy %d in the Sersics catalogue - value %2.2f is out of range [0.5,4.2]' % (i,sersic_index))
        return None

    # get rest of parameters
    hlr = sersic_sample_catalog[i,2]
    g = sersic_sample_catalog[i,3]

    # build the galaxy
    profile = galsim.Sersic(n=sersic_index, half_light_radius=hlr)
    beta = 2.*math.pi * rng() * galsim.radians
    profile.applyShear(g=g,beta=beta)
    dx = rng()* config['sersics_sample']['max_random_dx']
    dy = rng()* config['sersics_sample']['max_random_dx']
    profile.applyShift(dx=dx, dy=dy)

    # build the PSF
    psf_g = rng() * config['sersics_sample']['max_random_psf_g']
    psf_beta = 2.*math.pi * rng() * galsim.radians

    if config['sersics_sample']['psf_type'] == 'space':
        lam=700.
        diam=1.3
        psf = galsim.Airy(lam_over_diam=lam/diam/100.)
    elif config['sersics_sample']['psf_type'] == 'ground':
        fwhm = 0.65 # arcsec
        atmos = galsim.Moffat(beta=3,fwhm=fwhm)
        optics = galsim.Airy(lam_over_diam = 0.5 * fwhm )
        psf = galsim.Convolve([atmos,optics])
    else:
        logger.error('%s in unknown psf_type. Use \'space\' or \'ground' % config['sersics_sample']['psf_type'])
        sys.exit()

    psf.applyShear(g=g,beta=psf_beta)

    # build the output dict
    gso = {}
    gso['pix'] = pix
    gso['gal'] = profile
    gso['psf'] = psf
    gso['galpsf'] = galsim.Convolve([gso['gal'],gso['psf']])

    return gso







if __name__ == "__main__":


    description = 'Compare FFT vs photon shooting. Use the galaxies specified in the corresponding yaml file (see photon_vs_fft.yaml for an example)'

    # parse arguments
    parser = argparse.ArgumentParser(description=description, add_help=True)
    parser.add_argument('filename_config', type=str, help='yaml config file, see photon_vs_fft.yaml for example.')
    parser.add_argument('--save_plots', action="store_true", help='if to generate_images of galaxies and store them in ./images/', default=False)
    global args
    args = parser.parse_args()

    # set up logger
    logging.basicConfig(format="%(message)s", level=logging.DEBUG, stream=sys.stdout)
    logger = logging.getLogger("photon_vs_fft") 

    # load the configuration file
    filename_config = args.filename_config
    global config
    config = yaml.load(open(filename_config,'r'))

    # run the test
    testShootVsFft()


