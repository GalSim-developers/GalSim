import numpy
import sys, os
import math
import logging
import galsim as galsim
import galsim.wfirst as wfirst

def main(argv):
    # Where to find and output data
    path, filename = os.path.split(__file__)
    datapath = os.path.abspath(os.path.join(path, "data/"))
    outpath = os.path.abspath(os.path.join(path, "output/"))	

    # In non-script code, use getLogger(__name__) at module scope instead.
    logging.basicConfig(format="%(message)s", level=logging.INFO, stream=sys.stdout)
    logger = logging.getLogger("demo13")

    # initialize (pseudo-)random number generator
    random_seed = 1234567
    rng = galsim.BaseDeviate(random_seed)

    # read in the WFIRST filters
    filters = wfirst.getBandpasses(AB_zeropoint=True);
    logger.debug('Read in filters')
    print 

    # filter has redder red limit
    for filter in filters:
        filters[filter].red_limit = 1197.5 

    # read in SEDs
    SED_names = ['CWW_E_ext', 'CWW_Sbc_ext', 'CWW_Scd_ext', 'CWW_Im_ext']
    SEDs = {}
    mag_norm = 22.0
    for SED_name in SED_names:
        SED_filename = os.path.join(datapath, '{0}.sed'.format(SED_name))
        # Here we create some galsim.SED objects to hold star or galaxy spectra.  The most
        # convenient way to create realistic spectra is to read them in from a two-column ASCII
        # file, where the first column is wavelength and the second column is flux. Wavelengths in
        # the example SED files are in Angstroms, flux in flambda.  The default wavelength type for
        # galsim.SED is nanometers, however, so we need to override by specifying
        # `wave_type = 'Ang'`.
        SED = galsim.SED(SED_filename, wave_type='Ang')
        # The normalization of SEDs affects how many photons are eventually drawn into an image.
        # One way to control this normalization is to specify the flux in a given bandpass
        bandpass = filters['W149']
        print "SED's redlimit = ", SED.red_limit
        bandpass.red_limit = SED.red_limit
        print "Current flux = ", SED.calculateFlux(bandpass=filters['W149'])
        SEDs[SED_name] = SED.withMagnitude(target_magnitude=mag_norm, bandpass=filters['W149'])

    logger.debug('Successfully read in SEDs')

    pixel_scale = wfirst.pixel_scale # 0.11 arcseconds
    exptime = wfirst.exptime # 168.1 seconds

    logger.info('')
    logger.info('Simulating a chromatic bulge+disk galaxy')
    redshift = 0.8

    # make a bulge ...
    mono_bulge = galsim.DeVaucouleurs(half_light_radius=0.5)
    bulge_SED = SEDs['CWW_E_ext'].atRedshift(redshift)
    # The `*` operator can be used as a shortcut for creating a chromatic version of a GSObject:
    bulge = mono_bulge * bulge_SED
    bulge = bulge.shear(g1=0.12, g2=0.07)
    logger.debug('Created bulge component')
    # ... and a disk ...
    mono_disk = galsim.Exponential(half_light_radius=2.0)
    disk_SED = SEDs['CWW_Im_ext'].atRedshift(redshift)
    disk = mono_disk * disk_SED
    disk = disk.shear(g1=0.4, g2=0.2)
    logger.debug('Created disk component')
    # ... and then combine them.
    bdgal = 0.8*bulge+4*disk

    # Note that at this stage, our galaxy is chromatic but our PSF is still achromatic.  
    logger.debug('Created bulge+disk galaxy final profile')

    #sky_level in photons / m^2 / s / arcsec^2
    # hardcoded for now
    sky_level = {'J129':6.509083, 'SNPrism':25.564653, 'F184':3.478826, 'W149':18.577768, 'Y106':6.346696, 'BAO-Grism':6.269559, 'Z087':5.401184, 'H158':6.121490}

    #Read Noise:
    read_noise_rms = 0.0

    # draw profile through WFIRST filters
    gaussian_noise = galsim.GaussianNoise(rng, sigma=0.02/10.)
    for filter_name, filter_ in filters.iteritems():
        
        # Obtaining parameters for Airy PSF
        effective_wavelength = (1e-9)*filters[filter_name].effective_wavelength # now in cm
        effective_diameter = wfirst.diameter*numpy.sqrt(1-wfirst.obscuration**2) 
        lam_over_diam = (1.0*effective_wavelength/wfirst.diameter)*206265.0 # in arcsec

        #Convolve with PSF
        PSF = galsim.Airy(obscuration=wfirst.obscuration, lam_over_diam=lam_over_diam)
        #PSF = galsim.Moffat(fwhm=0.6,beta=2.5)
        bdconv = galsim.Convolve([bdgal, PSF])

    	img = galsim.ImageF(512*2,512*2,scale=pixel_scale) # 64, 64
    	bdconv.drawImage(filter_,image=img)
        #print "A1: ", img.array
        print "F1: ", numpy.sum(img.array), numpy.mean(img.array)

    	print "S =", (numpy.sum(img.array**2)-(numpy.sum(img.array))**2/(64**2))/(64**2)

        #Adding sky level
        sky_level_pix = wfirst.getSkyLevel(filters[filter_name],exp_time=wfirst.exptime)
        img.array[:,:] += sky_level_pix
        print "sky_level_pix = ", sky_level_pix

        #Adding Poisson Noise
        poisson_noise = galsim.PoissonNoise(rng)        
    	#img.addNoise(poisson_noise)
        logger.debug('Created {0}-band image'.format(filter_name))
        out_filename = os.path.join(outpath, 'demo13_{0}.fits'.format(filter_name))
        galsim.fits.write(img,out_filename)
        logger.debug('Wrote {0}-band image to disk'.format(filter_name))

        #print "After adding noise", img.array.min(), img.array.max()
        print "N = ", poisson_noise.getVariance()

    	#Applying a quadratic non-linearity
        beta = -3.57*1e-7
        def NLfunc(x,beta):
            return x + beta*(x**2)
    	#NLfunc = wfirst.NLfunc
    	#img.applyNonlinearity(NLfunc,beta)
    	logger.debug('Applied Nonlinearity to {0}-band image'.format(filter_name))
        out_filename = os.path.join(outpath, 'demo13_NL_{0}.fits'.format(filter_name))
        galsim.fits.write(img,out_filename)
        logger.debug('Wrote {0}-band image with Nonlinearity to disk'.format(filter_name))

    	#Accounting Reciprocity Failure
    	alpha = 0.0065 
        # NOTE: not yet in the module

    	#img.addReciprocityFailure(exp_time=exptime,alpha=alpha)
    	logger.debug('Accounted for Recip. Failure in {0}-band image'.format(filter_name))
        out_filename = os.path.join(outpath, 'demo13_RecipFail_{0}.fits'.format(filter_name))
        galsim.fits.write(img,out_filename)
        logger.debug('Wrote {0}-band image  after accounting for Recip. Failure to disk'.format(filter_name))

        #Adding Read Noise
    	read_noise = galsim.CCDNoise(rng)
        read_noise.setReadNoise(read_noise_rms)
        #img.addNoise(read_noise)
        logger.debug('Added Readnoise for {0}-band image'.format(filter_name))
        out_filename = os.path.join(outpath, 'demo13_ReadNoise_{0}.fits'.format(filter_name))
        galsim.fits.write(img,out_filename)
        logger.debug('Wrote {0}-band image after adding readnoise to disk'.format(filter_name))

    logger.info('You can display the output in ds9 with a command line that looks something like:')
    logger.info('ds9 -rgb -blue -scale limits -0.2 0.8 output/demo13_ReadNoise_J129.fits -green -scale limits'
                +' -0.25 1.0 output/demo13_ReadNoise_W149.fits -red -scale limits -0.25 1.0 output/demo13_ReadNoise_Z087.fits'
                +' -zoom 2 &')

if __name__ == "__main__":
	main(sys.argv)

