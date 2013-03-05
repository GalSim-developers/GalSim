try:
    import galsim
except ImportError:
    path, filename = os.path.split(__file__)
    sys.path.append(os.path.abspath(os.path.join(path, "..")))
    import galsim

import pdb
import pyfits
import logging
import sys
import numpy
import sys
import os
import math
import pylab

# set the path for the example
filename_example_rgc = '/home/kacprzak/code/GalSim/examples/data/real_galaxy_catalog_example.fits'
dirpath_example_rgc = '/home/kacprzak/code/GalSim/examples/data/'
filename_example_gals = '/home/kacprzak/code/GalSim/examples/data/real_galaxy_images.fits'
filename_example_psfs = '/home/kacprzak/code/GalSim/examples/data/real_galaxy_PSF_images.fits'

filename_gals = "val_rgc_gals.fits"
filename_psfs = "val_rgc_psfs.fits"
filename_rgc =  "val_rgc.fits"
dirpath_rgc ='.'

random_seed = 8241572
flux = 1.        				# arbitrary choice, 
final_pixel_scale = 0.5			# should be rather high so that we can do more honest comparison
final_n_pixels    = 100			# should be rather high so that we can do more honest comparison

def getTestShears():
	"""
	This function returns a list of galsim.shear which will be used to test reconvolution.
	This may be rewritten to use a catalog or a config file.
	"""

	# create a list of shears and add example instances
	test_shears = [galsim.Shear(g1=0.02,g2=0.01),galsim.Shear(g1=0.05,g2=-0.05)]

	return test_shears

def getTestReconvolvingPFS():
	"""
	This function returns a list of GSObjects which are PSFs and will be used as reconvolving PSFs.
	This may be rewritten to use a catalog or a config file.
	"""

	# define the list of reconvolving PSFs
	test_psfs = []

	# add two example instances
	psf = galsim.Moffat(fwhm=2.85,beta=3.)
	psf.applyShear(g1=0.02,g2=0.05)
	test_psfs.append(psf)

	psf = galsim.Airy(lam_over_diam=2.)
	psf.applyShear(g1=0.02,g2=-0.05)
	test_psfs.append(psf)
	
	return test_psfs

def getTestGals():
	"""
	Returns a list of dicts, which contain the GSObjects, images and other parameters which will be used to create mock real galaxies.
	This may be rewritten to use a catalog or a config file.
	"""

	# set the number and scale of pixels used for mock real galaxies
	# this is fixed now but may be used later as a parameter of to explore
	img_pix = 300
	img_scale = 0.5

	test_gals = []
	
	tg = {}
	tg['gal'] = galsim.galsim.Exponential(half_light_radius=2.) 	# create a galaxy
	tg['gal'].applyShear(g1=0.3, g2=0.2);							# get some intrinsic ellipticity
	tg['gal'].setFlux(flux)											# set flux for the galaxy
	tg['snr'] = 2e5; 												# set up a snr
	tg['psf'] = galsim.Moffat(fwhm=1,beta=3)						# get a original PSF, which we will later remove
	tg['psf'].applyShear(g1=0.05,g2=0.03)							# set the ellipticity of original PSF
	tg['pix'] = galsim.Pixel(img_scale)
	tg['final'] = galsim.Convolve(tg['gal'],tg['psf'],tg['pix'])
	tg['image_final'] = galsim.ImageD(img_pix,img_pix)				# get an image for original galaxy
	tg['image_psf'] = galsim.ImageD(img_pix,img_pix)				# get an image for original galaxy
	tg['final'].draw(image=tg['image_final'],dx=img_scale)				# get the noise standard deviation based on snr 
	noise_std = numpy.linalg.norm(tg['image_final'].array.flatten())/float(tg['snr'])	# add noise to the galaxy
	tg['image_final'].addNoise(galsim.GaussianDeviate(random_seed+len(test_gals),0.,noise_std))
	tg['psf'].draw(tg['image_psf'],dx=img_scale)					# draw the PSF 
	test_gals.append(tg)											# add this object tg the list

	tg = {}
	tg['gal'] = galsim.galsim.DeVaucouleurs(half_light_radius=3.8)	# create a galaxy
	tg['gal'].applyShear(g1=0.3, g2=0.2);							# get some intrinsic ellipticity
	tg['gal'].setFlux(flux)											# set flux for the galaxy
	tg['snr'] = 2e5; 												# set up a snr
	tg['psf'] = galsim.Moffat(fwhm=1,beta=3)						# get a original PSF, which we will later remove
	tg['psf'].applyShear(g1=0.05,g2=0.03)							# set the ellipticity of original PSF
	tg['pix'] = galsim.Pixel(img_scale)
	tg['final'] = galsim.Convolve(tg['gal'],tg['psf'],tg['pix'])
	tg['image_final'] = galsim.ImageD(img_pix,img_pix)				# get an image for original galaxy
	tg['image_psf'] = galsim.ImageD(img_pix,img_pix)				# get an image for original galaxy
	tg['final'].draw(image=tg['image_final'],dx=img_scale)				# get the noise standard deviation based on snr 
	noise_std = numpy.linalg.norm(tg['image_final'].array.flatten())/float(tg['snr'])	# add noise to the galaxy
	tg['image_final'].addNoise(galsim.GaussianDeviate(random_seed++len(test_gals),0.,noise_std))
	tg['psf'].draw(tg['image_psf'],dx=img_scale)					# draw the PSF 
	test_gals.append(tg)											# add this object tg the list

	return test_gals

def createRGC(test_gals):
	"""
	This function creates and saves a real galaxy catalog, image and psf fits files.
	It uses the list of test_gals created by getTestGals() function.
	"""

	# use the existing example catalog binary table
	n_gals = len(test_gals)
	hdu_table = pyfits.open(filename_example_rgc)
	hdu_table[1].data = hdu_table[1].data[0:n_gals]

	# create lists for new galaxies and psf extensions
	hdu_gals = pyfits.HDUList()
	hdu_psfs = pyfits.HDUList()

	for i,tg in enumerate(test_gals) :

		# save PSF and galaxy into a hdu list
		tg['image_final'].write(fits=hdu_gals)
		tg['image_psf'].write(fits=hdu_psfs)

		# calculate mag and weight, as I don't know how to do it I set it to one for now
		MAG=1.
		WEIGHT=1.

		# fill in the table data
		hdu_table[1].data[i]['IDENT'] = i
		hdu_table[1].data[i]['MAG'] = MAG 								# I am not sure what tg use for this number
		hdu_table[1].data[i]['WEIGHT'] = WEIGHT 						# I am not sure what tg use for this number
		hdu_table[1].data[i]['GAL_FILENAME'] = filename_gals         
		hdu_table[1].data[i]['PSF_FILENAME'] = filename_psfs
		hdu_table[1].data[i]['GAL_HDU'] = i
		hdu_table[1].data[i]['PSF_HDU'] = i
		hdu_table[1].data[i]['PIXEL_SCALE'] = tg['image_final'].getScale()
		# hdu_table[1].data[i]['NOISE_MEAN'] =  0. 						# noise mean is 0 as we subtract the background
		# hdu_table[1].data[i]['NOISE_VARIANCE'] = noise_std			# variance is equal tg sky level

		logger.info('saved image %d' % i)

	# save all catalogs
	hdu_table.writeto(filename_rgc,clobber=True)
	hdu_psfs.writeto(filename_psfs,clobber=True)
	hdu_gals.writeto(filename_gals,clobber=True)

def testValidationRGC(filename_rgc,image_dir):
	"""
	Copied from RealDemo.py.
	Checks if the mock rgc is working as expected.
	"""

	# define some variables etc.
	
	#	this is using the mock RGC
	real_catalog_filename = filename_rgc
	output_dir = '.'
	good_psf_central_fwhm = 0.6 # arcsec; FWHM of smaller Gaussian in the double Gaussian for good 
	                            # seeing
	bad_psf_central_fwhm = 1.3 # arcsec; FWHM of smaller Gaussian in the double Gaussian for bad seeing
	central_psf_amp = 0.8 # relative contribution of inner Gaussian in the double Gaussian PSF
	outer_fwhm_mult = 2.0 # ratio of (outer)/(inner) Gaussian FWHM for double Gaussian PSF
	pixel_scale = 0.2 # arcsec
	g1 = 0.05
	g2 = 0.00
	wmult = 1.0 # oversampling to use in intermediate steps of calculations

	# read in a random galaxy from the training data
	rgc = galsim.RealGalaxyCatalog(real_catalog_filename, image_dir)
	real_galaxy = galsim.RealGalaxy(rgc, random=True)
	print 'Made real galaxy from catalog index ',real_galaxy.index

	# make a target PSF object
	good_psf_inner = galsim.Gaussian(flux=central_psf_amp, fwhm = good_psf_central_fwhm)
	good_psf_outer = galsim.Gaussian(flux=1.0-central_psf_amp,
	                                 fwhm=outer_fwhm_mult*good_psf_central_fwhm)
	good_psf = good_psf_inner + good_psf_outer

	bad_psf_inner = galsim.Gaussian(flux=central_psf_amp, fwhm=bad_psf_central_fwhm)
	bad_psf_outer = galsim.Gaussian(flux=1.0-central_psf_amp, fwhm=outer_fwhm_mult*bad_psf_central_fwhm)
	bad_psf = bad_psf_inner + bad_psf_outer

	pixel = galsim.Pixel(xw=pixel_scale, yw=pixel_scale)
	good_epsf = galsim.Convolve(good_psf, pixel)
	bad_epsf = galsim.Convolve(bad_psf, pixel)

	# simulate some high-quality ground-based data, e.g., Subaru/CFHT with good seeing; with and without
	# shear
	print "Simulating unsheared galaxy in good seeing..."
	sim_image_good_noshear = galsim.simReal(real_galaxy, good_epsf, pixel_scale, rand_rotate=False)
	print "Simulating sheared galaxy in good seeing..."
	sim_image_good_shear = galsim.simReal(real_galaxy, good_epsf, pixel_scale, g1=g1, g2=g2,
	                                      rand_rotate=False)

	# simulate some poor-quality ground-based data, e.g., a bad night for SDSS; with and without shear
	print "Simulating unsheared galaxy in bad seeing..."
	sim_image_bad_noshear = galsim.simReal(real_galaxy, bad_epsf, pixel_scale, rand_rotate=False)
	print "Simulating sheared galaxy in bad seeing..."
	sim_image_bad_shear = galsim.simReal(real_galaxy, bad_epsf, pixel_scale, g1=g1, g2=g2,
	                                     rand_rotate=False)

	# write to files: original galaxy, original PSF, 2 target PSFs, 4 simulated images
	# note: will differ each time it is run, because we chose a random image
	print "Drawing images and writing to files!"

	N = real_galaxy.original_image.getGoodImageSize(real_galaxy.pixel_scale, wmult)
	orig_gal_img = galsim.ImageF(N, N)
	orig_gal_img.setScale(real_galaxy.pixel_scale)
	real_galaxy.original_image.draw(orig_gal_img.view())
	orig_gal_img.write(os.path.join(output_dir, 'demoreal.orig_gal.fits'), clobber=True)

	N = real_galaxy.original_PSF.getGoodImageSize(real_galaxy.pixel_scale, wmult)
	orig_psf_img = galsim.ImageF(N, N)
	orig_psf_img.setScale(real_galaxy.pixel_scale)
	real_galaxy.original_PSF.draw(orig_psf_img.view())
	orig_psf_img.write(os.path.join(output_dir, 'demoreal.orig_PSF.fits'), clobber=True)

	good_epsf_img = good_epsf.draw(dx=pixel_scale)
	good_epsf_img.write(os.path.join(output_dir, 'demoreal.good_target_PSF.fits'), clobber=True)

	bad_epsf_img = bad_epsf.draw(dx=pixel_scale)
	bad_epsf_img.write(os.path.join(output_dir, 'demoreal.bad_target_PSF.fits'), clobber=True)

	sim_image_good_noshear.write(os.path.join(output_dir, 'demoreal.good_simulated_image.noshear.fits'),
	                             clobber=True)
	sim_image_good_shear.write(os.path.join(output_dir, 'demoreal.good_simulated_image.shear.fits'),
	                           clobber=True)
	sim_image_bad_noshear.write(os.path.join(output_dir, 'demoreal.bad_simulated_image.noshear.fits'),
	                            clobber=True)
	sim_image_bad_shear.write(os.path.join(output_dir, 'demoreal.bad_simulated_image.shear.fits'),
	                          clobber=True)

def getComparisonImages(test_gals,filename_rgc,image_dir):
	"""
	Uses the created RGC and truth information from test_gals to perform shearing and reconvolution
	of images. The shears and reconvolving PSFs are created in getTestShears() and getTestReconvolvingPFS().
	Outputs a list of dicts, which contain all the images needed to perform comparisons.
	"""

	# initialise the list
	comparison_images = []

	# load the mock real galaxy catalog
	rgc = galsim.RealGalaxyCatalog(filename_rgc, image_dir)

	# get the shears and reconvolving PSFs
	list_shears = getTestShears()
	list_psfs = getTestReconvolvingPFS()

	# get the pixel kernel
	pix = galsim.Pixel(final_pixel_scale)

	# currently the loop is over all elements of cartesian product of test shears, 
	# test reconvolving PSFs and mock real galaxies. This may be too much, so maybe it will be changed
	# to some fiducial value + deviations.

	# loop over shears
	for s,shear in enumerate(list_shears):

		# loop over reconvolving PSFs
		for p,psf in enumerate(list_psfs):

			# loop over mock real galaxies
			for g in range(rgc.nobjects):

				# get the mock real galaxy, apply shear and reconvolve
				gal_real = galsim.RealGalaxy(rgc, index=g)
				gal_real.applyShear(shear)
				gal_reconv = galsim.Convolve([gal_real,psf,pix])
				
				# create the equivalent to the above without using reconvolution
				gal_true   = test_gals[g]['gal']
				gal_true.applyShear(shear)
				gal_conv = galsim.Convolve([gal_true,psf,pix])

				# initialise the images
				image_reconv = galsim.ImageD(final_n_pixels,final_n_pixels)
				image_conv   = galsim.ImageD(final_n_pixels,final_n_pixels)
				image_reconv_psf = galsim.ImageD(final_n_pixels,final_n_pixels)

				# set the flux so that the visual comparison is easier
				gal_reconv.setFlux(1.)
				gal_conv.setFlux(1.)

				# draw the images
				gal_reconv.draw(image_reconv,dx=final_pixel_scale)
				gal_conv.draw(image_conv,dx=final_pixel_scale)
				psf.draw(image_reconv_psf,dx=final_pixel_scale)
				
				# get the comparison dicts
				comparison_images.append( {
					'img_reconv': image_reconv, 
					'img_conv' : image_conv, 
					'img_reconv_psf': image_reconv_psf,
					'img_orig': test_gals[g]['image_final'], 
					'img_orig_psf' : test_gals[g]['image_psf']
					} )

				print 'Made real galaxy from catalog index #%d, psf #%d, shear #%d'  % (g,p,s)

	return comparison_images

def plotPixelDifferences(comparison_images):
	"""
	This function plots and saves png files of the images of considered galaxies.
	"""

	# for all comparison images
	for i,images in enumerate(comparison_images):

		pylab.figure(i)
		pylab.clf()

		pylab.subplot(2,3,1)
		pylab.imshow(images['img_orig'].array,interpolation='nearest')
		pylab.colorbar();
		pylab.xlabel('real galaxy convolved with real PSF')

		pylab.subplot(2,3,4)
		pylab.imshow(images['img_orig_psf'].array,interpolation='nearest')
		pylab.colorbar()
		pylab.xlabel('real PSF')

		pylab.subplot(2,3,5)
		pylab.imshow(images['img_reconv_psf'].array,interpolation='nearest')
		pylab.colorbar()
		pylab.xlabel('reconvolving PSF')

		pylab.subplot(2,3,3)
		pylab.imshow(images['img_conv'].array,interpolation='nearest')
		pylab.colorbar()
		pylab.xlabel('galaxy image with reconvolving PSF - true')

		pylab.subplot(2,3,2)
		pylab.imshow(images['img_reconv'].array,interpolation='nearest')
		pylab.colorbar()
		pylab.xlabel('galaxy image with reconvolving PSF - using reconvolution')

		pylab.subplot(2,3,6)
		pylab.imshow(images['img_reconv'].array - images['img_conv'].array ,interpolation='nearest')
		pylab.colorbar()
		pylab.xlabel('reconvolution - true')

		pylab.gcf().set_size_inches(20,10)
		pylab.savefig('comparison_%d.png' % i)

		print 'saved comparison_%d.png' % i


if __name__ == "__main__":

	logging.basicConfig(format="%(message)s", level=logging.INFO, stream=sys.stdout)
	logger = logging.getLogger("demo1") 

	# get the objects that we will want to save in mock real galaxy catalogs
	test_gals = getTestGals()

	createRGC(test_gals)
	# testValidationRGC(filename_rgc,'.')
	comparison_images = getComparisonImages(test_gals,filename_rgc,'.')

	plotPixelDifferences(comparison_images)


