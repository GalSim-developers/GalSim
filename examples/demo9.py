# Copyright 2012, 2013 The GalSim developers:
# https://github.com/GalSim-developers
#
# This file is part of GalSim: The modular galaxy image simulation toolkit.
#
# GalSim is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# GalSim is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with GalSim.  If not, see <http://www.gnu.org/licenses/>
#
"""
Demo #9

The ninth script in our tutorial about using GalSim in python scripts: examples/demo*.py.
(This file is designed to be viewed in a window 100 characters wide.)

This script simulates cluster lensing or galaxy-galaxy lensing.  The gravitational shear
applied to each galaxy is calculated for an NFW halo mass profile.  We simulate observations 
of galaxies around 20 different clusters -- 5 each of 4 different masses.  Each cluster
has its own file, organized into 4 directories (one for each mass).  For each cluster, we
draw 20 lensed galaxies at random positions of the image.

New features introduced in this demo:

- im = galsim.ImageS(xsize, ysize)
- pos = galsim.PositionD(x, y)
- nfw = galsim.NFWHalo(mass, conc, z, omega_m, omega_lam)
- g1,g2 = nfw.getShear(pos, z)
- mag = nfw.getMagnification(pos, z)
- pos = bounds.trueCenter()

- Make multiple output files.
- Place galaxies at random positions on a larger image.
- Write a bad pixel mask and a weight image as the second and third HDUs in each file.
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
    Make 4 directories, each with 5 files, each of which has 20 galaxies.
    
    Also, each directory corresponds to a different mass halo.
    The files in each direction are just different noise realizations and galaxy locations.

    The images also all have a second HDU with a weight image.

    And we build the multiple files in parallel.
    """
    from multiprocessing import Process, Queue, current_process, cpu_count

    logging.basicConfig(format="%(message)s", level=logging.INFO, stream=sys.stdout)
    logger = logging.getLogger("demo9")

    # Define some parameters we'll use below.

    mass_list = [ 1.e15, 7.e14, 4.e14, 2.e14 ]  # mass in Msun/h
    nfiles = 5 # number of files per item in mass list
    nobj = 20  # number of objects to draw for each file

    image_size = 512       # pixels
    pixel_scale = 0.20     # arcsec / pixel
    sky_level = 1.e6       # ADU / arcsec^2

    psf_fwhm = 0.5         # arcsec

    gal_eta_rms = 0.4      # eta is defined as ln(a/b)
    gal_hlr_min = 0.4      # arcsec
    gal_hlr_max = 1.2      # arcsec
    gal_flux_min = 1.e4    # ADU
    gal_flux_max = 1.e6    # ADU

    nfw_conc = 4           # concentration parameter = virial radius / NFW scale radius
    nfw_z_halo = 0.3       # redshift of the halo
    nfw_z_source = 0.6     # redshift of the lensed sources
    omega_m = 0.3          # Omega matter for the background cosmology.
    omega_lam = 0.7        # Omega lambda for the background cosmology.

    random_seed = 8383721

    logger.info('Starting demo script 9')

    def build_file(seed, file_name, mass):
        """A function that does all the work to build a single file.
           Returns the total time taken.
        """
        t1 = time.time()

        full_image = galsim.ImageF(image_size, image_size)
        full_image.setScale(pixel_scale)

        # The weight image will hold the inverse variance for each pixel.
        weight_image = galsim.ImageF(image_size, image_size)
        weight_image.setScale(pixel_scale)

        # It is common for astrometric images to also have a bad pixel mask.  We don't have any
        # defect simulation currently, so our bad pixel masks are currently all zeros. 
        # But someday, we plan to add defect functionality to GalSim, at which point, we'll
        # be able to mark those defects on a bad pixel mask.
        # Note: the S in ImageS means to use "short int" for the data type.
        # This is a typical choice for a bad pixel image.
        badpix_image = galsim.ImageS(image_size, image_size)
        badpix_image.setScale(pixel_scale)

        # Setup the NFWHalo stuff:
        nfw = galsim.NFWHalo(mass=mass, conc=nfw_conc, redshift=nfw_z_halo,
                             omega_m=omega_m, omega_lam=omega_lam)
        # Note: the last two are optional.  If they are omitted, then (omega_m=0.3, omega_lam=0.7) 
        # are actually the defaults.  If you only specify one of them, the other is set so that 
        # the total is 1.  But you can define both values so that the total is not 1 if you want.
        # Radiation is assumed to be zero and dark energy equation of state w = -1.
        # If you want to include either radiation or more complicated dark energy models,
        # you can define your own cosmology class that defines the functions a(z), E(a), and 
        # Da(z_source, z_lens).  Then you can pass this to NFWHalo as a `cosmo` parameter.

        # The "true" center of the image is allowed to be halfway between two pixels, as is the 
        # case for even-sized images.  full_image.bounds.center() is an integer position,
        # which would be 1/2 pixel up and to the right of the true center in this case.
        im_center = full_image.bounds.trueCenter()

        for k in range(nobj):

            # Initialize the random number generator we will be using for this object:
            rng = galsim.UniformDeviate(seed+k)

            # Determine where this object is going to go.
            # We choose points randomly within a donut centered at the center of the main image
            # in order to avoid placing galaxies too close to the halo center where the lensing 
            # is not weak.  We use an inner radius of 10 arcsec and an outer radius of 50 arcsec,
            # which takes us essentially to the edge of the image.
            radius = 50
            inner_radius = 10
            max_rsq = radius**2
            min_rsq = inner_radius**2
            while True:  # (This is essentially a do..while loop.)
                x = (2.*rng()-1) * radius
                y = (2.*rng()-1) * radius
                rsq = x**2 + y**2
                if rsq >= min_rsq and rsq <= max_rsq: break
            pos = galsim.PositionD(x,y)

            # We also need the position in pixels to determine where to place the postage
            # stamp on the full image.
            image_pos = pos / pixel_scale + im_center

            # For even-sized postage stamps, the nominal center (returned by stamp.bounds.center())
            # cannot be at the true center (returned by stamp.bounds.trueCenter()) of the postage 
            # stamp, since the nominal center values have to be integers.  Thus, the nominal center
            # is 1/2 pixel up and to the right of the true center.
            # If we used odd-sized postage stamps, we wouldn't need to do this.
            x_nominal = image_pos.x + 0.5
            y_nominal = image_pos.y + 0.5

            # Get the integer values of these which will be the actual nominal center of the 
            # postage stamp image.
            ix_nominal = int(math.floor(x_nominal+0.5))
            iy_nominal = int(math.floor(y_nominal+0.5))

            # The remainder will be accounted for in a shift
            dx = x_nominal - ix_nominal
            dy = y_nominal - iy_nominal

            # Make the pixel:
            pix = galsim.Pixel(pixel_scale)

            # Make the PSF profile:
            psf = galsim.Kolmogorov(fwhm = psf_fwhm)

            # Determine the random values for the galaxy:
            flux = rng() * (gal_flux_max-gal_flux_min) + gal_flux_min
            hlr = rng() * (gal_hlr_max-gal_hlr_min) + gal_hlr_min
            gd = galsim.GaussianDeviate(rng, sigma = gal_eta_rms)
            eta1 = gd()  # Unlike g or e, large values of eta are valid, so no need to cutoff.
            eta2 = gd()

            # Make the galaxy profile with these values:
            gal = galsim.Exponential(half_light_radius=hlr, flux=flux)
            gal.applyShear(eta1=eta1, eta2=eta2)

            # Now apply the appropriate lensing effects for this position from 
            # the NFW halo mass.
            try:
                g1,g2 = nfw.getShear( pos , nfw_z_source )
                shear = galsim.Shear(g1=g1,g2=g2)
            except:
                # This shouldn't happen, since we exclude the inner 10 arcsec, but it's a 
                # good idea to use the try/except block here anyway.
                import warnings        
                warnings.warn("Warning: NFWHalo shear is invalid -- probably strong lensing!  " +
                              "Using shear = 0.")
                shear = galsim.Shear(g1=0,g2=0)

            mu = nfw.getMagnification( pos , nfw_z_source )
            if mu < 0:
                import warnings
                warnings.warn("Warning: mu < 0 means strong lensing!  Using mu=25.")
                mu = 25
            elif mu > 25:
                import warnings
                warnings.warn("Warning: mu > 25 means strong lensing!  Using mu=25.")
                mu = 25

            gal.applyMagnification(mu)
            gal.applyShear(shear)

            # Build the final object
            final = galsim.Convolve([psf, pix, gal])

            # Account for the non-integral portion of the position
            final.applyShift(dx*pixel_scale,dy*pixel_scale)

            # Draw the stamp image
            stamp = final.draw(dx=pixel_scale)

            # Recenter the stamp at the desired position:
            stamp.setCenter(ix_nominal,iy_nominal)

            # Find overlapping bounds
            bounds = stamp.bounds & full_image.bounds
            full_image[bounds] += stamp[bounds]


        # Add Poisson noise to the full image
        sky_level_pixel = sky_level * pixel_scale**2

        # Going to the next seed isn't really required, but it matches the behavior of the 
        # config parser, so doing this will result in identical output files.
        # If you didn't care about that, you could instead construct this as a continuation
        # of the last RNG from the above loop
        rng = galsim.BaseDeviate(seed+nobj)
        full_image.addNoise(galsim.PoissonNoise(rng,sky_level=sky_level_pixel))

        # For the weight image, we only want the noise from the sky.  (If we were including
        # read_noise, we'd want that as well.)  Including the Poisson noise from the objects
        # as well tends to bias fits that use this as a weight, since the model becomes
        # magnitude-dependent.
        # The variance is just sky_level_pixel.  And we want the inverse of this.
        weight_image.fill(1./sky_level_pixel)

        # Write the file to disk:
        galsim.fits.writeMulti([full_image, badpix_image, weight_image], file_name)

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
    
    t1 = time.time()

    ntot = nfiles * len(mass_list)

    try:
        from multiprocessing import cpu_count
        ncpu = cpu_count()
        if ncpu > ntot:
            nproc = ntot
        else:
            nproc = ncpu
        logger.info("ncpu = %d.  Using %d processes",ncpu,nproc)
    except:
        nproc = 2
        logger.info("Unable to determine ncpu.  Using %d processes",nproc)

    # Make output directory if not already present.
    if not os.path.isdir('output'):
        os.mkdir('output')

    # Set up the task list
    task_queue = Queue()
    seed = random_seed
    for i in range(len(mass_list)):
        mass = mass_list[i]
        dir_name = "nfw%d"%(i+1)
        dir = os.path.join('output',dir_name)
        if not os.path.isdir(dir): os.mkdir(dir)
        for j in range(nfiles):
            file_name = "cluster%04d.fits"%j
            full_name = os.path.join(dir,file_name)
            # We put on the task queue the args to the buld_file function and
            # some extra info to pass through to the output queue.
            # Our extra info is just the file name that we use to write out which file finished.
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

    t2 = time.time()

    logger.info('Total time taken using %d processes = %f',nproc,t2-t1)


if __name__ == "__main__":
    main(sys.argv)
