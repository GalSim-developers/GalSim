"""
Demo #9

The ninth script in our tutorial about using GalSim in python scripts: examples/demo*.py.
(This file is designed to be viewed in a window 100 characters wide.)

This script simulates cluster lensing or galaxy-galaxy lensing.  The graviational shear 
applied to each galaxy is calculated for an NFW halo mass profile.  We simulate observations 
of galaxies around 20 different clusters -- 5 each of 4 different masses.  Each cluster
has its own file, organized into 4 directories (one for each mass).  In order to have a total
of only 100 galaxies (for time considerations), each image only has 5 background galaxies.
For more realistic investigations, you would of course want to use more galaxies per image,
but for the purposes of the demo script, this suffices.

New features introduced in this demo:

- Use shears from an NFW Halo model
- Make multiple output files
- Write a weight image as the second HDU in each file
- Use multiple processes to construct each file in parallel.
"""

import sys
import os
import subprocess
import math
import numpy
import logging
import time
import galsim

def main(argv):
    """
    Make 4 directories, each with 5 files, each of which has 5 galaxies.
    
    Also, each directory corresponds to a different mass halo.
    The files in each direction are just different noise realizations and galaxy locations.

    The images also all have a second HDU with a weight image.

    And we build the multiple files in parallel.
    """
    from multiprocessing import Process, Queue, current_process, cpu_count

    logging.basicConfig(format="%(message)s", level=logging.INFO, stream=sys.stdout)
    logger = logging.getLogger("demo8")

    # Define some parameters we'll use below.

    mass_list = [ 1.e15, 7.e14, 4.e14, 2.e14 ]  # mass in Msun/h
    nfiles = 5 # number of files per item in mass list
    nobj = 5  # number of objects to draw for each file

    image_size = 256
    stamp_size = 64
    pixel_scale = 0.43
    sky_level = 1.e3

    psf_fwhm = 0.9

    gal_eta_rms = 0.2
    gal_n_min = 0.5
    gal_n_max = 4.
    gal_hlr_min = 0.8
    gal_hlr_max = 2.8
    gal_flux_min = 1.e5
    gal_flux_max = 1.e6

    random_seed = 8383721

    logger.info('Starting demo script 9')

    def build_file(seed, file_name, mass):
        """A function that does all the work to build a single file.
           Returns the total time taken.
        """
        t1 = time.time()

        full_image = galsim.ImageF(image_size, image_size)
        full_image.setScale(pixel_scale)
        full_image.setCenter(0,0)

        for k in range(nobj):

            # Initialize the random number generator we will be using for this object:
            rng = galsim.UniformDeviate(seed+k)
            gd = galsim.GaussianDeviate(rng, sigma = gal_eta_rms)

            # Determine where this object is going to go:
            x = (rng()-0.5) * (image_size-stamp_size)
            y = (rng()-0.5) * (image_size-stamp_size)
            print 'x,y = ',x,y

            # Define the stamp image
            xmin = int(math.floor(x-stamp_size/2 + 0.5))
            ymin = int(math.floor(y-stamp_size/2 + 0.5))
            bounds = galsim.BoundsI(xmin, xmin+stamp_size-1, ymin, ymin+stamp_size-1)
            print 'bounds = ',bounds
            print 'full_image bounds = ',full_image.bounds
            bounds = bounds & full_image.bounds
            print 'bounds => ',bounds
            stamp = full_image[bounds]
            stamp.setScale(pixel_scale)

            # Make the pixel:
            pix = galsim.Pixel(pixel_scale)

            # Make the PSF profile:
            psf = galsim.Kolmogorov(fwhm = psf_fwhm)

            # Determine the random values for the galaxy:
            flux = rng() * (gal_flux_max-gal_flux_min) + gal_flux_min
            print 'flux = ',flux
            n = rng() * (gal_n_max-gal_n_min) + gal_n_min
            print 'n = ',n
            hlr = rng() * (gal_hlr_max-gal_hlr_min) + gal_hlr_min
            print 'hlr = ',hlr
            eta1 = gd()  # Unlike g or e, large values of eta are valid, so no need to cutoff.
            eta2 = gd()
            print 'eta = ',eta1,eta2

            # Make the galaxy profile with these values:
            gal = galsim.Sersic(n=n, half_light_radius=hlr, flux=flux)
            gal.applyShear(eta1=eta1, eta2=eta2)

            # Build the final object
            final = galsim.Convolve([psf, pix, gal])

            # Draw the stamp image
            final.draw(stamp)
            print 'draw stamp ',k
        print 'Done drawing stamps'

        # Add Poisson noise to the full image
        sky_level_pixel = sky_level * pixel_scale**2
        full_image += sky_level_pixel
        full_image.addNoise(galsim.CCDNoise(rng))
        full_image -= sky_level_pixel
        print 'Added noise'

        # Write the file to disk:
        full_image.write(file_name)
        print 'Write image to ',file_name

        t2 = time.time()
        return t2-t1

    def worker(input, output):
        """input is a queue with (args, info) tuples:
               args are the arguements to pass to build_file
               info is passed along to the output queue.
           output is a queue storing (result, info, proc) tuples:
               result is the return value of from build_file
               info is passed through from the input queue.
               proc is the process name.
        """
        for (args, info) in iter(input.get, 'STOP'):
            result = build_file(*args)
            output.put( (result, info, current_process().name) )
    
    ntot = nfiles * len(mass_list)

    # Take seeds to be sequential after the given seed value. 
    # But need to step by the number of galaxies in each file.
    seeds = [ random_seed + k*nobj for k in range(ntot) ]

    # First draw the image using just a single process:
    t1 = time.time()

    seed = random_seed
    for i in range(len(mass_list)):
        mass = mass_list[i]
        dir_name = "mass%d"%i
        dir = os.path.join('output',dir_name)
        if not os.path.isdir(dir): os.mkdir(dir)
        for j in range(nfiles):
            file_name = "cluster%d.fits"%j
            full_name = os.path.join(dir,file_name)
            t = build_file(seed, full_name, mass)
            # Need to step by the number of galaxies in each file.
            seed += nobj
            proc = current_process().name
            logger.info('%s: Time for file %s was %f',proc,full_name,t)

    t2 = time.time()
    
    # Now do the same thing, but use multiple processes

    nproc = 4
    logger.info('Using ncpu = %d processes',nproc)

    # Set up the task list
    task_queue = Queue()
    seed = random_seed
    for i in range(len(mass_list)):
        mass = mass_list[i]
        dir_name = "mass%d"%i
        dir = os.path.join('output',dir_name)
        for j in range(nfiles):
            file_name = "cluster%02d.fits"%j
            full_name = os.path.join(dir,file_name)
            task_queue.put( ( (seed, full_name, mass), full_name ) )
            # Need to step by the number of galaxies in each file.
            seed += nobj

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
        file_name = info
        t = result
        logger.info('%s: Time for file %s was %f',proc,file_name,t)

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

    t3 = time.time()

    logger.info('Total time taken using a single process = %f',t2-t1)
    logger.info('Total time taken using %d processes = %f',nproc,t3-t2)

    for i in range(len(mass_list)):
        mass = mass_list[i]
        dir_name = "mass%d"%i
        dir = os.path.join('output',dir_name)
        for j in range(nfiles):
            file_name1 = "cluster%d.fits"%j
            file_name2 = "cluster%02d.fits"%j
            full_name1 = os.path.join(dir,file_name1)
            full_name2 = os.path.join(dir,file_name2)
            print 'full_name1 = ',full_name1
            print 'full_name2 = ',full_name2
            im1 = galsim.fits.read(full_name1)
            im2 = galsim.fits.read(full_name2)
            print 'im1 = ',im1
            print 'im2 = ',im2
            numpy.testing.assert_array_equal(
                    im1.array, im2.array,
                    err_msg="Files %s and %s are not equal"%(full_name1,full_name2))
    logger.info('Files created using single and multiple processes are exactly equal.')
    logger.info('')

    print


if __name__ == "__main__":
    main(sys.argv)
