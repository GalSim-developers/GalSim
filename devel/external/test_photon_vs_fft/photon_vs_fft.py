"""@file validate_shoot_vs_fft.py 
Compare images created by FFT and photon shooting rendering methods in GalSim. 
All configuration is specified in a config yaml file.
See validate_shoot_vs_fft.yaml for an example.
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

filename_output = 'photon_vs_fft_results.txt'

def getGSObjects(obj):
    """
    This function generates all GSObjects which are necessary to create an image of an object.
    This involves a galaxy, PSF and pixel kernels.

    Input:
    obj - dictionary of a single object as defined in the config file

    Output:
    gso - dictionary with GSObjects gso['gal'], gso['psf'], gso['pix'] 
    """
    
    gso = {}

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

    if obj['psf']['type'] == 'none':

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


#
def testShootVsFfft():
    """
    For each of the galaxies in the config['galaxies'], produces the images using FFT and photon shooting.
    Displays plots of image residuals and shows differences in the moments measurements.
    Saves the Adaptive moments resutls to file.
    """

    if config['image']['n_photons'] < 100000:
        logger.warning('small number of photons, results may me meaningless, HSM may crash')

    file_output = open(filename_output,'w')

    # things to output:
    # id of the test object
    # g1_fft
    # g2_fft
    # g1_photon
    # g2_photon
    # max_diff_over_max_image
    

    output_row_fmt = "%d\t% 2.6f\t% 2.6f\t% 2.6f\t% 2.6f\t% 2.6f\n"
    output_header = '# id of the test object g1_fft g2_fft g1_photon g2_photon max_diff_over_max_image\n'
    file_output.write(output_header)

    # loop throught the galaxy catalog
    for ig,obj in enumerate(config['objects']):

        # get the GSObjects 
        gso = getGSObjects(obj)
        
        logger.info('------------------ galaxy %g ------------------' % ig)


        if obj['psf']['type'] != 'none':
            
            # create the PSF
            logger.info('drawing PSF using drawShoot and %2.0e photons' % config['image']['n_photons'])
            image_psf_shoot = galsim.ImageF(config['image']['n_pix'],config['image']['n_pix'])
            gso['psf'].drawShoot(image_psf_shoot,dx=config['image']['scale'],n_photons=config['image']['n_photons'])
            logger.info('drawShoot is ready for PSF of type %s' % obj['psf']['type'])
    
    
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

        image_gal_fft.write('shifted_fft.fits')
        image_gal_shoot.write('shifted_shoot.fits')


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

        filename_fig = 'photon_vs_fft_gal%d.png' % ig
        pylab.gcf().set_size_inches(20,10)
        pylab.savefig(filename_fig)
        pylab.close()
        logger.info('saved figure %s' % filename_fig)

        logger.info('max(residual) / max(image_fft) = %2.4e ' % ( max_diff_over_max_image )  )

        # find adaptive moments
        moments_shoot = galsim.FindAdaptiveMom(image_gal_shoot)
        moments_fft   = galsim.FindAdaptiveMom(image_gal_fft)

        moments_shoot_g1 = moments_shoot.observed_shape.getG1()
        moments_shoot_g2 = moments_shoot.observed_shape.getG2()
        moments_fft_g1   = moments_fft.observed_shape.getG1()
        moments_fft_g2   = moments_fft.observed_shape.getG2()
        moments_diff_g1  = moments_shoot_g1 - moments_fft_g1
        moments_diff_g2  = moments_shoot_g2 - moments_fft_g2


        # display resutls
        logger.info('adaptive moments fft           gi % 2.6f % 2.6f' % (moments_fft_g1,  moments_fft_g2))
        logger.info('adaptive moments shoot         gi % 2.6f % 2.6f' % (moments_shoot_g1,  moments_shoot_g2))
        logger.info('adaptive moments difference    gi % 2.6f % 2.6f' % (moments_diff_g1,  moments_diff_g2))

        file_output.write(output_row_fmt % (ig, moments_fft_g1,  moments_fft_g2, moments_shoot_g1,  moments_shoot_g2, max_diff_over_max_image))

def plotEllipticityBiases():
    """
    This function is a prototype for a plotting function. 
    I am not sure what is the best plots/subplots combination to show what we want to see, so for now 
    let's use this form.
    """

    data = numpy.loadtxt(filename_output)

    n_test_gals = data.shape[0]

    g1_photon=data[:,3]
    g2_photon=data[:,4]
    g1_fft=data[:,1]
    g2_fft=data[:,2]

    de1 = g1_fft-g1_photon
    de2 = g2_fft-g2_photon 
    
    pylab.plot(de1/g1_photon,'x',label='g1')
    pylab.plot(de2/g2_photon,'+',label='g2')
                
    pylab.xlabel('test galaxy #')
    pylab.ylabel('de/e')
    pylab.xlim([-1,n_test_gals])

    pylab.gcf().set_size_inches(10,5)
    pylab.legend()

    filename_fig = 'photon_vs_fft_differences.png';
    pylab.savefig(filename_fig)
    pylab.close()

    logger.info('saved figure %s' % filename_fig)

if __name__ == "__main__":


    description = 'Compare FFT vs photon shooting. Use the galaxies specified in the corresponding yaml file (see validation_shoot_vs_fft.yaml for an example)'

    # parse arguments
    parser = argparse.ArgumentParser(description=description, add_help=True)
    parser.add_argument('filename_config', type=str, help='yaml config file, see validation_shoot_vs_fft.yaml for example.')
    args = parser.parse_args()

    # set up logger
    logging.basicConfig(format="%(message)s", level=logging.INFO, stream=sys.stdout)
    logger = logging.getLogger("validation_shoot_vs_fft") 

    # load the configuration file
    filename_config = args.filename_config
    global config
    config = yaml.load(open(filename_config,'r'))

    # run the test
    testShootVsFfft()

    # save the figure
    plotEllipticityBiases()




