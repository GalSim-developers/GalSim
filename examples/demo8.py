"""
Some example scripts to make multi-object images using the GalSim library.
"""

import sys
import os
import subprocess
import math
import numpy
import logging
import time
import galsim

# Use multiple processes to draw image in parallel
def main(argv):
    """
    Make an image containing 10 x 10 postage stamps.
    The galaxies are bulge + disk with parameters drawn from random variates
    Each galaxy is drawn using photon shooting.
    """
    from multiprocessing import Process, Queue, current_process, cpu_count

    logging.basicConfig(format="%(message)s", level=logging.INFO, stream=sys.stdout)
    logger = logging.getLogger("demo8")

    # Define some parameters we'll use below.

    single_file_name = os.path.join('output','bpd_single.fits')
    multi_file_name = os.path.join('output','bpd_multi.fits')

    random_seed = 1512413
    sky_level = 1.e4        # ADU / arcsec^2
    pixel_scale = 0.28      # arcsec
    nx_pixels = 64          # size of each postage stamp in pixels
    ny_pixels = 64
    nx_stamps = 10          # number of postage stamps in each direction
    ny_stamps = 10

    gal_flux_min = 1.e4     # Range for galaxy flux
    gal_flux_max = 1.e5  
    bulge_hlr_min = 0.3     # Range for bulge's half-light radius (arcsec)
    bulge_hlr_max = 0.9
    bulge_e_min = 0.        # Range for bulge's ellipticity
    bulge_e_max = 0.3
    bulge_frac_min = 0.1    # Range for bulge fraction
    bulge_frac_max = 0.5
    disk_hlr_min = 0.5      # Range for disk's half-light radius (arcsec)
    disk_hlr_max = 1.5
    disk_e_min = 0.2        # Range for disk's ellipticity
    disk_e_max = 0.8

    psf_fwhm = 0.65         # arcsec

    logger.info('Starting demo script 8')

    def draw_stamp(seed):
        """A function that draws a single postage stamp using a given seed for the 
           random number generator.
           Returns the image and the total time taken.
        """
        t1 = time.time()

        # Initialize the random number generator we will be using.
        rng = galsim.UniformDeviate(seed)

        # Make the pixel:
        pix = galsim.Pixel(xw = pixel_scale)

        # Make the PSF profile:
        psf = galsim.Moffat(fwhm = psf_fwhm, beta = 2.4)

        # Make the galaxy profile:
        f = rng() * (bulge_frac_max-bulge_frac_min) + bulge_frac_min
        #print 'flux = ',f
        hlr = rng() * (bulge_hlr_max-bulge_hlr_min) + bulge_hlr_min
        #print 'hlr = ',hlr
        beta_ellip = rng() * 2*math.pi * galsim.radians
        #print 'beta_ellip = ',beta_ellip
        ellip = rng() * (bulge_e_max-bulge_e_min) + bulge_e_min
        #print 'ellip = ',ellip

        bulge = galsim.Sersic(n=3.6, half_light_radius=hlr, flux=f)
        bulge.applyShear(e=ellip, beta=beta_ellip)

        hlr = rng() * (disk_hlr_max-disk_hlr_min) + disk_hlr_min
        #print 'hlr = ',hlr
        beta_ellip = rng() * 2*math.pi * galsim.radians
        #print 'beta_ellip = ',beta_ellip
        ellip = rng() * (disk_e_max-disk_e_min) + disk_e_min
        #print 'ellip = ',ellip

        disk = galsim.Sersic(n=1.5, half_light_radius=hlr)
        disk.applyShear(e=ellip, beta=beta_ellip)
        disk.setFlux(1.-f)

        gal = galsim.Add([bulge,disk])

        flux = rng() * (gal_flux_max-gal_flux_min) + gal_flux_min
        gal.setFlux(flux)

        # Build the final object by convolving the galaxy and PSF 
        # Not including the pixel -- since we are using drawShoot
        final_nopix = galsim.Convolve([psf, gal])

        # Define the stamp image
        stamp = galsim.ImageF(nx_pixels, ny_pixels)
        stamp.setScale(pixel_scale)

        # Photon shooting automatically convolves by the pixel, so we've made sure not
        # to include it in the profile!
        sky_level_pixel = sky_level * pixel_scale**2
        final_nopix.drawShoot(stamp, noise=sky_level_pixel/100, uniform_deviate=rng)

        # For photon shooting, galaxy already has poisson noise, so we want to make 
        # sure not to add that noise again!  Thus, we just add sky noise, which 
        # is Poisson with the mean = sky_level_pixel
        stamp.addNoise(galsim.PoissonDeviate(rng, mean=sky_level_pixel))

        t2 = time.time()
        return stamp, t2-t1

    def worker(input, output):
        """input is a queue with (seed, info) tuples:
               seed is the argement to pass to draw_stamp
               info is passed along to the output queue.
           output is a queue storing (result, info, proc) tuples:
               result is the returned tuple from draw_stamp: (image, time).
               info is passed through from the input queue.
               proc is the process name.
        """
        for (seed, info) in iter(input.get, 'STOP'):
            result = draw_stamp(seed)
            output.put( (result, info, current_process().name) )
    
    ntot = nx_stamps * ny_stamps

    # Take seeds to be sequential after the given seed value. 
    # This way different galaxies are deterministic, but uncorrelated.
    seeds = [ random_seed + k for k in range(ntot) ]

    # First draw the image using just a single process:
    t1 = time.time()
    image_single = galsim.ImageF(nx_stamps * nx_pixels , ny_stamps * ny_pixels)
    image_single.setScale(pixel_scale)

    k = 0
    for ix in range(nx_stamps):
        for iy in range(ny_stamps):
            bounds = galsim.BoundsI(ix*nx_pixels+1 , (ix+1)*nx_pixels, 
                                    iy*ny_pixels+1 , (iy+1)*ny_pixels)
            im, t = draw_stamp(seeds[k])
            image_single[bounds] = im
            proc = current_process().name
            logger.info('%s: Time for stamp (%d,%d) was %f',proc,ix,iy,t)
            k = k+1
    t2 = time.time()
    
    # Now do the same thing, but use multiple processes
    image_multi = galsim.ImageF(nx_stamps * nx_pixels , ny_stamps * ny_pixels)
    image_multi.setScale(pixel_scale)

    nproc = 4
    logger.info('Using ncpu = %d processes',nproc)

    # Set up the task list
    task_queue = Queue()
    k = 0
    for ix in range(nx_stamps):
        for iy in range(ny_stamps):
            task_queue.put( (seeds[k], [ix,iy]) ) 
            k = k+1
    
    # Run the tasks
    # Each Process command starts up a parallel process that will keep checking the queue 
    # for a new task. If there is one there, it grabs it and does it. If not, it waits 
    # until there is one to grab. When it finds a 'STOP', it shuts down. 
    done_queue = Queue()
    for k in range(nproc):
        Process(target=worker, args=(task_queue, done_queue)).start()

    # In the meanwhile, the main process keeps going.  We pull each image off of the 
    # done_queue and put it in the appropriate place on the main image.  
    # This loop is happening while the other processes are still working on their tasks.
    # You'll see that these logging statements get print out as the stamp images are still 
    # being drawn.  
    for i in range(ntot):
        result, info, proc = done_queue.get()
        ix = info[0]
        iy = info[1]
        bounds = galsim.BoundsI(ix*nx_pixels+1 , (ix+1)*nx_pixels, 
                                iy*ny_pixels+1 , (iy+1)*ny_pixels)
        im = result[0]
        image_multi[bounds] = im
        t = result[1]
        logger.info('%s: Time for stamp (%d,%d) was %f',proc,ix,iy,t)

    t3 = time.time()

    # Stop the processes
    # The 'STOP's could have been put on the task list before starting the processes, or you
    # can wait.  In some cases it can be useful to clear out the done_queue (as we just did)
    # and then add on some more tasks.  We don't need that here, but it's perfectly fine to do.
    # Once you are done with the processes, putting nproc 'STOP's will stop them all.
    # This is important, because the program will keep running as long as there are running
    # processes, even if the main process gets to the end.  So you do want to make sure to 
    # add those 'STOP's at some point!
    for k in range(nproc):
        task_queue.put('STOP')

    logger.info('Total time taken using a single process = %f',t2-t1)
    logger.info('Total time taken using %d processes = %f',nproc,t3-t2)

    # Now write the images to disk.
    image_single.write(single_file_name, clobber=True)
    image_multi.write(multi_file_name, clobber=True)
    logger.info('Wrote images to %r and %r',single_file_name, multi_file_name)

    numpy.testing.assert_array_equal(image_single.array, image_multi.array,
                                     err_msg="Images are not equal")
    logger.info('Images created using single and multiple processes are exactly equal.')
    logger.info('')

    print


if __name__ == "__main__":
    main(sys.argv)
