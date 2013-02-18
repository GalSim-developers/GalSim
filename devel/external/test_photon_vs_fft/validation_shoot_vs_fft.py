import logging
import galsim
import sys
import os
import numpy
import argparse
import yaml
import pdb

from matplotlib import pyplot as plt 

def getGSObjects(observed_galaxy):
    
    gso = {}
    
    if observed_galaxy['gal']['type'] == 'sersics':
        
        logger.info('galaxy type is %s' % observed_galaxy['gal']['type'])

        #print observed_galaxy['gal']['params']['rb'], observed_galaxy['gal']['params']['rd']
        gso['bulge'] = galsim.Sersic(observed_galaxy['gal']['profile']['nb'], half_light_radius=observed_galaxy['gal']['profile']['rb'])
        gso['disc'] = galsim.Sersic(observed_galaxy['gal']['profile']['nd'], half_light_radius=observed_galaxy['gal']['profile']['rd'])
        gso['bulge'].applyShear(g=observed_galaxy['gal']['profile']['eb'],beta=observed_galaxy['gal']['profile']['abd']*numpy.pi*galsim.radians)
        gso['disc'].applyShear(g=observed_galaxy['gal']['profile']['ed'],beta=0.*galsim.radians)
        gso['gal'] = observed_galaxy['gal']['profile']['bt'] * gso['bulge'] + (1.-observed_galaxy['gal']['profile']['bt']) * gso['disc']
        gso['psf'] = galsim.Moffat(fwhm=observed_galaxy['psf']['fwhm'],beta=observed_galaxy['psf']['beta'])
        gso['pix'] = galsim.Pixel(config['image']['scale'])
        
    if observed_galaxy['gal']['type'] == 'real':
        
        logger.info('galaxy type is %s' % observed_galaxy['gal']['type'])

        #print observed_galaxy['gal']['params']['rb'], observed_galaxy['gal']['params']['rd']
        
        gso['gal'] = galsim.RealGalaxy(real_galaxy_catalog, index=observed_galaxy['gal']['profile']['id'])
        gso['psf'] = galsim.Moffat(flux=1., fwhm=observed_galaxy['psf']['fwhm'],beta=observed_galaxy['psf']['beta'])
        gso['pix'] = galsim.Pixel(config['image']['scale'])
        
    #TODO: here add operations on galaxies like resizing etc.
    
    if observed_galaxy['gal']['operations']['dilation'] != 0:
    
        logger.info('applying dilation')
        gso['gal'].applyDilation(scale=observed_galaxy['gal']['operations']['dilation'])

    logger.info('setting flux')
    gso['gal'].setFlux(observed_galaxy['gal']['operations']['flux'])
    
    return gso

def testShootVsFfft():
    
    for gal in config['galaxies']:

        gso = getGSObjects(gal)
        logger.info('got GSObjects for galaxy %s' % gal['name'])
	
	# create the PSF
	
	logger.info('drawing PSF using drawShoot and %s photons' % config['image']['n_photons'])

        image_psf_shoot = galsim.ImageF(config['image']['n_pix'],config['image']['n_pix'])
        gso['psf'].drawShoot(image_psf_shoot,dx=config['image']['scale'],n_photons=config['image']['n_photons'])
        logger.info('drawShoot is ready for PSF')
	
	
	# create shoot image
	
	logger.info('drawing galaxy using drawShoot and %s photons' % config['image']['n_photons'])

        image_gal_shoot = galsim.ImageF(config['image']['n_pix'],config['image']['n_pix'])
        final = galsim.Convolve([gso['gal'],gso['psf']])
        final.drawShoot(image_gal_shoot,dx=config['image']['scale'],n_photons=config['image']['n_photons'])
        logger.info('drawShoot is ready for galaxy %s' % gal['name'])

        # create fft image
	
	logger.info('drawing galaxy using draw (FFT)')

        image_gal_fft = galsim.ImageF(config['image']['n_pix'],config['image']['n_pix'])
        final = galsim.Convolve([gso['gal'],gso['psf'],gso['pix']])
        final.draw(image_gal_fft,dx=config['image']['scale'])
        logger.info('draw using FFT is ready for galaxy %s' % gal['name'])
	
	diff_image = image_gal_shoot.array - image_gal_fft.array
	
	hsm_shoot = galsim.psfcorr.EstimateShearHSM(gal_image=image_gal_shoot,PSF_image=image_psf_shoot,strict=False,shear_est='LINEAR')
	hsm_fft   = galsim.psfcorr.EstimateShearHSM(gal_image=image_gal_fft,PSF_image=image_psf_shoot,strict=False,shear_est='LINEAR')
	
	#pdb.set_trace()
	
	hsm_corr_fft_g1= hsm_fft.corrected_g1
	hsm_corr_fft_g2= hsm_fft.corrected_g2
	hsm_corr_shoot_g1= hsm_shoot.corrected_g1
	hsm_corr_shoot_g2= hsm_shoot.corrected_g2
		
	hsm_obs_shoot_g1= hsm_shoot.observed_shape.getG1()
	hsm_obs_shoot_g2= hsm_shoot.observed_shape.getG2()
	hsm_obs_fft_g1= hsm_fft.observed_shape.getG1()
	hsm_obs_fft_g2= hsm_fft.observed_shape.getG2()
	
	hsm_obs_diff_g1 =  hsm_obs_shoot_g1 - hsm_obs_fft_g1 
	hsm_obs_diff_g2 =  hsm_obs_shoot_g2 - hsm_obs_fft_g2
	
	hsm_corr_diff_g1 =  hsm_corr_shoot_g1 - hsm_corr_fft_g1 
	hsm_corr_diff_g2 =  hsm_corr_shoot_g2 - hsm_corr_fft_g2
			
	logger.info('corrected shape fft   gi %2.4e %2.4e' %  (hsm_corr_fft_g1,  hsm_corr_fft_g2))
	logger.info('corrected shape shoot gi %2.4e %2.4e' %  (hsm_corr_shoot_g1,hsm_corr_shoot_g2))
	logger.info('observed shape fft    gi %2.4e %2.4e' %  (hsm_obs_fft_g1,   hsm_obs_fft_g2))
	logger.info('observed shape shoot  gi %2.4e %2.4e' %  (hsm_obs_shoot_g1, hsm_obs_shoot_g2))
			
	logger.info('difference in observed  shape (shoot_gi - fft_gi) %2.4e %2.4e ' % ( hsm_obs_diff_g1, hsm_obs_diff_g2 ) )
	logger.info('difference in corrected shape (shoot_gi - fft_gi) %2.4e %2.4e ' % ( hsm_corr_diff_g1, hsm_corr_diff_g2 )  )
			
	plt.ion()
	plt.clf()
	
	plt.subplot(131)
	plt.imshow(image_gal_shoot.array,interpolation='nearest')
	plt.colorbar()
	plt.title('image_shoot')
	
	plt.subplot(132)
	plt.imshow(image_gal_fft.array,interpolation='nearest')
	plt.colorbar()
	plt.title('image_fft')
	
	plt.subplot(133)
	plt.imshow(diff_image,interpolation='nearest')
	plt.colorbar()
	plt.title('image_shoot - image_fft')
	
	
        


if __name__ == "__main__":


    description = 'Compare FFT vs photon shooting'

    parser = argparse.ArgumentParser(description=description, add_help=True)
    parser.add_argument('filename_config', type=str, help='yaml config file, see validation_shoot_vs_fft.yaml for example.')
    args = parser.parse_args()

    logging.basicConfig(format="%(message)s", level=logging.INFO, stream=sys.stdout)
    logger = logging.getLogger("validation_shoot_vs_fft") 

    filename_config = args.filename_config
    global config
    config = yaml.load(open(filename_config,'r'))
    
    global real_galaxy_catalog
    real_galaxy_catalog = galsim.RealGalaxyCatalog(file_name=config['paths']['filename_real_catalog'], image_dir=config['paths']['directory_real_catalog'])

    testShootVsFfft()



