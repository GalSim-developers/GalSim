# Copyright (c) 2012-2018 by the GalSim developers team on GitHub
# https://github.com/GalSim-developers
#
# This file is part of GalSim: The modular galaxy image simulation toolkit.
# https://github.com/GalSim-developers/GalSim
#
# GalSim is free software: redistribution and use in source and binary forms,
# with or without modification, are permitted provided that the following
# conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions, and the disclaimer given in the accompanying LICENSE
#    file.
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions, and the disclaimer given in the documentation
#    and/or other materials provided with the distribution.
#
"""
Demo #9

The ninth script in our tutorial about using GalSim in python scripts: examples/demo*.py.
(This file is designed to be viewed in a window 100 characters wide.)

This script simulates cluster lensing or galaxy-galaxy lensing.  The gravitational shear
applied to each galaxy is calculated for an NFW halo mass profile.  We simulate observations
of galaxies around 20 different clusters -- 5 each of 4 different masses.  Each cluster
has its own file, organized into 4 directories (one for each mass).  For each cluster, we
draw 20 lensed galaxies located at random positions in the image.  The PSF is appropriate for a
space-like simulation.  (Some of the numbers used are the values for HST.)  And we apply
a cubic telescope distortion for the WCS.  Finally, we also output a truth catalog for each
output image that could be used for testing the accuracy of shape or flux measurements.

New features introduced in this demo:

- psf = OpticalPSF(lam, diam, ..., trefoil1, trefoil2, nstruts, strut_thick, strut_angle)
- im = galsim.ImageS(xsize, ysize, wcs)
- pos = galsim.PositionD(x, y)
- nfw = galsim.NFWHalo(mass, conc, z, omega_m, omega_lam)
- g1,g2 = nfw.getShear(pos, z)
- mag = nfw.getMagnification(pos, z)
- dist = galsim.DistDeviate(rng, function, x_min, x_max)
- pos = image.true_center
- wcs = galsim.UVFunction(ufunc, vfunc, xfunc, yfunc, origin)
- wcs.toWorld(profile, image_pos)
- wcs.makeSkyImage(image, sky_level)
- image_pos = wcs.toImage(pos)
- image.invertSelf()
- truth_cat = galsim.OutputCatalog(names, types)
- bounds.isDefined()

- Make multiple output files.
- Place galaxies at random positions on a larger image.
- Write a bad pixel mask and a weight image as the second and third HDUs in each file.
- Use multiple processes to construct each file in parallel.
"""

import sys
import os
import math
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

    mass_list = [ 7.e14, 4.e14, 2.e14, 1.e14 ]  # mass in Msun/h
    nfiles = 5             # number of files per item in mass list

    image_size = 512       # pixels
    sky_level = 1.e2       # ADU / arcsec^2

    psf_D = 2.4            # meters
    psf_lam = 900.0        # nanometers; note that OpticalPSF will automatically convert units to
                           # get lam/diam in units of arcsec, unless told otherwise.  In this case,
                           # that is (900e-9m / 2.4m) * 206265 arcsec/rad = 0.077 arcsec.
    psf_obsc = 0.125       # (0.3m / 2.4m) = 0.125
    psf_nstruts = 4
    psf_strut_thick = 0.07
    psf_strut_angle = 15 * galsim.degrees

    psf_defocus = 0.04     # The aberrations are all taken to be quite modest here.
    psf_astig1 = 0.03      # (I don't actually know what are appropriate for HST...)
    psf_astig2 = -0.01
    psf_coma1 = 0.02
    psf_coma2 = 0.04
    psf_trefoil1 = -0.02
    psf_trefoil2 = 0.04

    gal_r_min = 0.05       # arcsec
    gal_r_max = 0.20       # arcsec
    gal_h_over_r_min = 0.1 #
    gal_h_over_r_max = 0.2 #
    gal_flux_min = 1.e4    # ADU
    gal_flux_max = 1.e6    # ADU

    field_g1 = 0.03        # The field shear is some cosmic shear applied to the whole field,
    field_g2 = 0.01        # taken to be behind the foreground NFW halo.
    nfw_conc = 4           # concentration parameter = virial radius / NFW scale radius
    nfw_z_halo = 0.3       # redshift of the halo
    nfw_z_source = 0.6     # redshift of the lensed sources
    omega_m = 0.3          # Omega matter for the background cosmology.
    omega_lam = 0.7        # Omega lambda for the background cosmology.

    field_shear = galsim.Shear(g1=field_g1, g2=field_g2)

    random_seed = 8383721

    logger.info('Starting demo script 9')

    def build_file(seed, file_name, mass, nobj, rng, truth_file_name, halo_id, first_obj_id):
        """A function that does all the work to build a single file.
           Returns the total time taken.
        """
        t1 = time.time()

        # Build the image onto which we will draw the galaxies.
        full_image = galsim.ImageF(image_size, image_size)

        # The "true" center of the image is allowed to be halfway between two pixels, as is the
        # case for even-sized images.  full_image.center is an integer position,
        # which would be 1/2 pixel up and to the right of the true center in this case.
        im_center = full_image.true_center

        # For the WCS, this time we use UVFunction, which lets you define arbitrary u(x,y)
        # and v(x,y) functions.  We use a simple cubic radial function to create a
        # pincushion distortion.  This is a typical kind of telescope distortion, although
        # we exaggerate the magnitude of the effect to make it more apparent.
        # The pixel size in the center of the image is 0.05, but near the corners (r=362),
        # the pixel size is approximately 0.075, which is much more distortion than is
        # normally present in typical telescopes.  But it makes the effect of the variable
        # pixel area obvious when you look at the weight image in the output files.
        ufunc1 = lambda x,y : 0.05 * x * (1. + 2.e-6 * (x**2 + y**2))
        vfunc1 = lambda x,y : 0.05 * y * (1. + 2.e-6 * (x**2 + y**2))

        # It's not required to provide the inverse functions.  However, if we don't, then
        # you will only be able to do toWorld operations, not the inverse toImage.
        # The inverse function does not have to be exact either.  For example, you could provide
        # a function that does some kind of iterative solution to whatever accuracy you care
        # about.  But in this case, we can do the exact inverse.
        #
        # Let w = sqrt(u**2 + v**2) and r = sqrt(x**2 + y**2).  Then the solutions are:
        # x = (u/w) r and y = (u/w) r, and we use Cardano's method to solve for r given w:
        # See http://en.wikipedia.org/wiki/Cubic_function#Cardano.27s_method
        #
        # w = 0.05 r + 2.e-6 * 0.05 * r**3
        # r = 100 * ( ( 5 sqrt(w**2 + 5.e3/27) + 5 w )**(1./3.) -
        #           - ( 5 sqrt(w**2 + 5.e3/27) - 5 w )**(1./3.) )

        def xfunc1(u,v):
            import math
            wsq = u*u + v*v
            if wsq == 0.:
                return 0.
            else:
                w = math.sqrt(wsq)
                temp = 5. * math.sqrt(wsq + 5.e3/27)
                r = 100. * ( (temp + 5*w)**(1./3.) - (temp - 5*w)**(1./3) )
                return u * r/w

        def yfunc1(u,v):
            import math
            wsq = u*u + v*v
            if wsq == 0.:
                return 0.
            else:
                w = math.sqrt(wsq)
                temp = 5. * math.sqrt(wsq + 5.e3/27)
                r = 100. * ( (temp + 5*w)**(1./3.) - (temp - 5*w)**(1./3) )
                return v * r/w

        # You could pass the above functions to UVFunction, and normally we would do that.
        # The only down side to doing so is that the specification of the WCS in the FITS
        # file is rather ugly.  GalSim is able to turn the python byte code into strings,
        # but they are basically a really ugly mess of random-looking characters.  GalSim
        # will be able to read it back in, but human readers will have no idea what WCS
        # function was used.  To see what they look like, uncomment this line and comment
        # out the later wcs line.
        #wcs = galsim.UVFunction(ufunc1, vfunc1, xfunc1, yfunc1, origin=im_center)

        # If you provide the functions as strings, then those strings will be preserved
        # in the FITS header in a form that is more legible to human readers.
        # It also has the extra benefit of matching the output from demo9.yaml, which we
        # always try to do.  The config file has no choice but to specify the functions
        # as strings.

        ufunc = '0.05 * x * (1. + 2.e-6 * (x**2 + y**2))'
        vfunc = '0.05 * y * (1. + 2.e-6 * (x**2 + y**2))'
        xfunc = ('( lambda w: ( 0 if w==0 else ' +
                 '100.*u/w*(( 5*(w**2 + 5.e3/27.)**0.5 + 5*w )**(1./3.) - ' +
                           '( 5*(w**2 + 5.e3/27.)**0.5 - 5*w )**(1./3.))))( (u**2+v**2)**0.5 )')
        yfunc = ('( lambda w: ( 0 if w==0 else ' +
                 '100.*v/w*(( 5*(w**2 + 5.e3/27.)**0.5 + 5*w )**(1./3.) - ' +
                           '( 5*(w**2 + 5.e3/27.)**0.5 - 5*w )**(1./3.))))( (u**2+v**2)**0.5 )')

        # The origin parameter defines where on the image should be considered (x,y) = (0,0)
        # in the WCS functions.
        wcs = galsim.UVFunction(ufunc, vfunc, xfunc, yfunc, origin=im_center)

        # Assign this wcs to full_image
        full_image.wcs = wcs

        # The weight image will hold the inverse variance for each pixel.
        # We can set the wcs directly on construction with the wcs parameter.
        weight_image = galsim.ImageF(image_size, image_size, wcs=wcs)

        # It is common for astrometric images to also have a bad pixel mask.  We don't have any
        # defect simulation currently, so our bad pixel masks are currently all zeros.
        # But someday, we plan to add defect functionality to GalSim, at which point, we'll
        # be able to mark those defects on a bad pixel mask.
        # Note: the S in ImageS means to use "short int" for the data type.
        # This is a typical choice for a bad pixel image.
        badpix_image = galsim.ImageS(image_size, image_size, wcs=wcs)

        # We also draw a PSF image at the location of every galaxy.  This isn't normally done,
        # and since some of the PSFs overlap, it's not necessarily so useful to have this kind
        # of image.  But in this case, it's fun to look at the psf image, especially with
        # something like log scaling in ds9 to see how crazy an aberrated OpticalPSF with
        # struts can look when there is no atmospheric component to blur it out.
        psf_image = galsim.ImageF(image_size, image_size, wcs=wcs)

        # We will also write some truth information to an output catalog.
        # In real simulations, it is often useful to have a catalog of the truth values
        # to compare to measurements either directly or as cuts on the galaxy sample to
        # find where systematic errors are largest.
        # For now, we just make an empty OutputCatalog object with the names and types of the
        # columns.
        names = [ 'object_id', 'halo_id',
                  'flux', 'radius', 'h_over_r', 'inclination.rad', 'theta.rad',
                  'mu', 'redshift', 'shear.g1', 'shear.g2',
                  'pos.x', 'pos.y', 'image_pos.x', 'image_pos.y',
                  'halo_mass', 'halo_conc', 'halo_redshift' ]
        types = [ int, int,
                  float, float, float, float, float,
                  float, float, float, float,
                  float, float, float, float,
                  float, float, float ]
        truth_cat = galsim.OutputCatalog(names, types)

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

        # Make the PSF profile outside the loop to minimize the (significant) OpticalPSF
        # construction overhead.
        psf = galsim.OpticalPSF(
            lam=psf_lam, diam=psf_D, obscuration=psf_obsc,
            nstruts=psf_nstruts, strut_thick=psf_strut_thick, strut_angle=psf_strut_angle,
            defocus=psf_defocus, astig1=psf_astig1, astig2=psf_astig2,
            coma1=psf_coma1, coma2=psf_coma2, trefoil1=psf_trefoil1, trefoil2=psf_trefoil2)

        for k in range(nobj):

            # Initialize the random number generator we will be using for this object:
            ud = galsim.UniformDeviate(seed+k+1)

            # Determine where this object is going to go.
            # We choose points randomly within a donut centered at the center of the main image
            # in order to avoid placing galaxies too close to the halo center where the lensing
            # is not weak.  We use an inner radius of 3 arcsec and an outer radius of 21 arcsec,
            # which is large enough to cover all the way to the corners, although we'll need
            # to watch out for galaxies that are fully off the edge of the image.
            radius = 21
            inner_radius = 3
            max_rsq = radius**2
            min_rsq = inner_radius**2
            while True:  # (This is essentially a do..while loop.)
                x = (2.*ud()-1) * radius
                y = (2.*ud()-1) * radius
                rsq = x**2 + y**2
                if rsq >= min_rsq and rsq <= max_rsq: break
            pos = galsim.PositionD(x,y)

            # We also need the position in pixels to determine where to place the postage
            # stamp on the full image.
            image_pos = wcs.toImage(pos)

            # For even-sized postage stamps, the nominal center (available as stamp.center)
            # cannot be at the true center (available as stamp.true_center) of the postage stamp,
            # since the nominal center values have to be integers.  Thus, the nominal center is
            # 1/2 pixel up and to the right of the true center.
            # If we used odd-sized postage stamps, we wouldn't need to do this.
            x_nominal = image_pos.x + 0.5
            y_nominal = image_pos.y + 0.5

            # Get the integer values of these which will be the actual nominal center of the
            # postage stamp image.
            ix_nominal = int(math.floor(x_nominal+0.5))
            iy_nominal = int(math.floor(y_nominal+0.5))

            # The remainder will be accounted for in an offset when we draw.
            dx = x_nominal - ix_nominal
            dy = y_nominal - iy_nominal
            offset = galsim.PositionD(dx,dy)

            # Draw the flux from a power law distribution: N(f) ~ f^-1.5
            # For this, we use the class DistDeviate which can draw deviates from an arbitrary
            # probability distribution.  This distribution can be defined either as a functional
            # form as we do here, or as tabulated lists of x and p values, from which the
            # function is interpolated.
            flux_dist = galsim.DistDeviate(ud, function = lambda x:x**-1.5,
                                           x_min = gal_flux_min,
                                           x_max = gal_flux_max)
            flux = flux_dist()

            # We introduce here another surface brightness profile, called InclinedExponential.
            # It represents a typical 3D galaxy disk profile inclined at an arbitrary angle
            # relative to face on.
            #
            #     inclination =  0 degrees corresponds to a face-on disk, which is equivalent to
            #                             the regular Exponential profile.
            #     inclination = 90 degrees corresponds to an edge-on disk.
            #
            # A random orientation corresponds to the inclination angle taking the probability
            # distribution:
            #
            #     P(inc) = 0.5 sin(inc)
            #
            # so we again use a DistDeviate to generate these values.
            inc_dist = galsim.DistDeviate(ud, function = lambda x: 0.5 * math.sin(x),
                                          x_min=0, x_max=math.pi)
            inclination = inc_dist() * galsim.radians

            # The parameters scale_radius and scale_height give the scale distances in the
            # 3D distribution:
            #
            #     I(R,z) = I_0 / (2 scale_height) * sech^2(z/scale_height) * exp(-r/scale_radius)
            #
            # These values can be given separately if desired.  However, it is often easier to
            # give the ratio scale_h_over_r as an independent value, since the radius and height
            # values are correlated, while h/r is approximately independent of h or r.
            h_over_r = ud() * (gal_h_over_r_max-gal_h_over_r_min) + gal_h_over_r_min

            radius = ud() * (gal_r_max-gal_r_min) + gal_r_min

            # The inclination is around the x-axis, so we want to rotate the galaxy by a
            # random angle.
            theta = ud() * math.pi * 2. * galsim.radians

            # Make the galaxy profile with these values:
            gal = galsim.InclinedExponential(scale_radius=radius, scale_h_over_r=h_over_r,
                                             inclination=inclination, flux=flux)
            gal = gal.rotate(theta)

            # Now apply the appropriate lensing effects for this position from
            # the NFW halo mass.
            try:
                g1,g2 = nfw.getShear( pos , nfw_z_source )
                nfw_shear = galsim.Shear(g1=g1,g2=g2)
            except:
                # This shouldn't happen, since we exclude the inner 10 arcsec, but it's a
                # good idea to use the try/except block here anyway.
                import warnings
                warnings.warn("Warning: NFWHalo shear is invalid -- probably strong lensing!  " +
                              "Using shear = 0.")
                nfw_shear = galsim.Shear(g1=0,g2=0)

            nfw_mu = nfw.getMagnification( pos , nfw_z_source )
            if nfw_mu < 0:
                import warnings
                warnings.warn("Warning: mu < 0 means strong lensing!  Using mu=25.")
                nfw_mu = 25
            elif nfw_mu > 25:
                import warnings
                warnings.warn("Warning: mu > 25 means strong lensing!  Using mu=25.")
                nfw_mu = 25

            # Calculate the total shear to apply
            # Since shear addition is not commutative, it is worth pointing out that
            # the order is in the sense that the second shear is applied first, and then
            # the first shear.  i.e. The field shear is taken to be behind the cluster.
            # Kind of a cosmic shear contribution between the source and the cluster.
            # However, this is not quite the same thing as doing:
            #     gal.shear(field_shear).shear(nfw_shear)
            # since the shear addition ignores the rotation that would occur when doing the
            # above lines.  This is normally ok, because the rotation is not observable, but
            # it is worth keeping in mind.
            total_shear = nfw_shear + field_shear

            # Apply the magnification and shear to the galaxy
            gal = gal.magnify(nfw_mu)
            gal = gal.shear(total_shear)

            # Build the final object
            final = galsim.Convolve([gal, psf])

            # Draw the stamp image
            # To draw the image at a position other than the center of the image, you can
            # use the offset parameter, which applies an offset in pixels relative to the
            # center of the image.
            # We also need to provide the local wcs at the current position.
            local_wcs = wcs.local(image_pos)
            stamp = final.drawImage(wcs=local_wcs, offset=offset)

            # Recenter the stamp at the desired position:
            stamp.setCenter(ix_nominal,iy_nominal)

            # Find overlapping bounds
            bounds = stamp.bounds & full_image.bounds
            # If there is no overlap, then the intersection comes out as not defined, which we
            # can check with bounds.isDefined().
            if not bounds.isDefined():
                logger.info("object %d is fully off the edge of the image.  Skipping this one.", k)
                continue
            full_image[bounds] += stamp[bounds]

            # Also draw the PSF
            psf_stamp = galsim.ImageF(stamp.bounds) # Use same bounds as galaxy stamp
            psf.drawImage(psf_stamp, wcs=local_wcs, offset=offset)
            psf_image[bounds] += psf_stamp[bounds]

            # Add the truth information for this object to the truth catalog
            row = ( (first_obj_id + k), halo_id,
                    flux, radius, h_over_r, inclination.rad, theta.rad,
                    nfw_mu, nfw_z_source, total_shear.g1, total_shear.g2,
                    pos.x, pos.y, image_pos.x, image_pos.y,
                    mass, nfw_conc, nfw_z_halo )
            truth_cat.addRow(row)

        # Add Poisson noise to the full image
        # Note: The normal calculation of Poission noise isn't quite correct right now.
        # The pixel area is variable, which means the amount of sky flux that enters each
        # pixel is also variable.  The wcs classes have a function `makeSkyImage` which
        # will fill an image with the correct amount of sky flux given the sky level
        # in units of ADU/arcsec^2.  We use the weight image as our work space for this.
        wcs.makeSkyImage(weight_image, sky_level)

        # Add this to the current full_image (temporarily).
        full_image += weight_image

        # Add Poisson noise, given the current full_image.
        # The config parser uses a different random number generator for file-level and
        # image-level values than for the individual objects.  This makes it easier to
        # parallelize the calculation if desired.  In fact, this is why we've been adding 1
        # to each seed value all along.  The seeds for the objects take the values
        # random_seed+1 .. random_seed+nobj.  The seed for the image is just random_seed,
        # which we built already (below) when we calculated how many objects need to
        # be in each file.  Use the same rng again here, since this is also at image scope.
        full_image.addNoise(galsim.PoissonNoise(rng))

        # Subtract the sky back off.
        full_image -= weight_image

        # The weight image is nominally the inverse variance of the pixel noise.  However, it is
        # common to exclude the Poisson noise from the objects themselves and only include the
        # noise from the sky photons.  The variance of the noise is just the sky level, which is
        # what is currently in the weight_image.  (If we wanted to include the variance from the
        # objects too, then we could use the full_image before we added the PoissonNoise to it.)
        # So all we need to do now is to invert the values in weight_image.
        weight_image.invertSelf()

        # Write the file to disk:
        galsim.fits.writeMulti([full_image, badpix_image, weight_image, psf_image], file_name)

        # And write the truth catalog file
        truth_cat.write(truth_file_name)

        t2 = time.time()
        return t2-t1

    def worker(input, output):
        """input is a queue with (args, info) tuples:
               args are the arguments to pass to build_file
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
    halo_id = 0
    first_obj_id = 0
    for i in range(len(mass_list)):
        mass = mass_list[i]
        dir_name = "nfw%d"%(i+1)
        dir = os.path.join('output',dir_name)
        if not os.path.isdir(dir): os.mkdir(dir)
        for j in range(nfiles):
            file_name = "cluster%04d.fits"%j
            file_name = os.path.join(dir,file_name)
            truth_file_name = "truth%04d.dat"%j
            truth_file_name = os.path.join(dir,truth_file_name)

            # Each image has a different number of objects.
            # We use a random number from 15 to 30.
            ud = galsim.UniformDeviate(seed)
            min = 15
            max = 30
            nobj = int(math.floor(ud() * (max-min+1))) + min
            logger.info('Number of objects for %s = %d',file_name,nobj)

            # We put on the task queue the args to the buld_file function and
            # some extra info to pass through to the output queue.
            # Our extra info is just the file name that we use to write out which file finished.
            args = (seed, file_name, mass, nobj, ud, truth_file_name, halo_id, first_obj_id)
            task_queue.put( (args, file_name) )
            # Need to step by the number of galaxies in each file.
            seed += nobj
            halo_id += 1
            first_obj_id += nobj

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
