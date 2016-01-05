# Copyright (c) 2012-2015 by the GalSim developers team on GitHub
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

# This file defines a few custom types related to making stamps of blended galaxies.

import galsim
import os
import math
import numpy

#
# Define the Blend stamp type
#

def SetupBlend(config, xsize, ysize, ignore, logger):
    """Do the appropriate setup for a Blend stamp.
    """
    # Parse the necessary parameters and make sure they are the right type.
    req = { 'n_neighbors' : int, 'min_sep' : float, 'max_sep' : float }
    galsim.config.CheckAllParams(config['stamp'], req=req, ignore=ignore)

    # Now farm off to the regular stamp setup function the rest of the work of parsing
    # the size and position of the stamp.
    ignore = ignore + ['n_neighbors', 'min_sep', 'max_sep']
    return galsim.config.SetupBasic(config, xsize, ysize, ignore, logger)


def ProfileBlend(config, psf, gsparams, logger):
    """
    Build a list of galaxy profiles, each convolved with the psf, to use for the blend image.
    """
    # Build the neighbors first, so the final "current_val" for gal is the main galaxy.
    n_neighbors = galsim.config.ParseValue(config['stamp'], 'n_neighbors', config, int)[0]
    min_sep = galsim.config.ParseValue(config['stamp'], 'min_sep', config, float)[0]
    max_sep = galsim.config.ParseValue(config['stamp'], 'max_sep', config, float)[0]

    neighbor_gals = []
    for i in range(n_neighbors):
        gal = galsim.config.BuildGSObject(config, 'gal', gsparams=gsparams, logger=logger)[0]
        neighbor_gals.append(gal)
        # Remove the current_val stuff from config['gal'] so we don't get the same galaxy
        # each time.
        galsim.config.RemoveCurrent(config['gal'], keep_safe=True)

    # GalSim has a RandomCircle position type that we can use to generate random positions
    # in an annulus.  Of course, we could do this be hand here, but this is easier.
    neighbor_pos = []
    for i in range(n_neighbors):
        config_pos = { 'pos' : { 'type' : 'RandomCircle',
                                 'radius' : max_sep,
                                 'inner_radius' : min_sep }
                     }
        pos = galsim.config.ParseValue(config_pos, 'pos', config, galsim.PositionD)[0]
        neighbor_pos.append(pos)

    main_gal = galsim.config.BuildGSObject(config, 'gal', gsparams=gsparams, logger=logger)[0]

    # Save these in the config dict, so BlendSet can use them to do further processing.
    config['blend_neighbor_gals'] = neighbor_gals
    config['blend_neighbor_pos'] = neighbor_pos
    config['blend_main_gal'] = main_gal

    profiles = [ main_gal ] + [ gal.shift(pos) for gal, pos in zip(neighbor_gals, neighbor_pos) ]
    if psf:
        profiles = [ galsim.Convolve(gal, psf) for gal in profiles ]

    return profiles

def DrawBlend(profiles, image, method, offset, config):
    """
    Draw the profiles onto the stamp.
    """
    n_neighbors = len(profiles)-1

    # Draw the central galaxy using the basic draw function.
    galsim.config.DrawBasic(profiles[0], image, method, offset, config)

    # We'll want a copy of just the neighbors for the deblend image.
    # Otherwise we could have just drawn these on the main image with add_to_image = True
    neighbor_image = image.copy()
    neighbor_image.setZero()

    # Draw all the neighbor stamps
    for p in profiles[1:]:
        galsim.config.DrawBasic(p, neighbor_image, method, offset, config, add_to_image=True)

    # Save this for the deblend processing
    config['blend_neighbor_image'] = neighbor_image

    image += neighbor_image

    return image

def WhitenBlend(profiles, image, config):
    """
    Whiten the noise on the stamp according to the existing noise in all the profiles.
    """
    total = galsim.Add(profiles)
    return galsim.config.WhitenBasic(total, image, config)


galsim.config.RegisterStampType('Blend',
                                setup_func = SetupBlend,
                                prof_func = ProfileBlend,
                                draw_func = DrawBlend,
                                whiten_func = WhitenBlend,
)


#
# Define the BlendSet stamp type
# This currently doesn't work with image.nproc != 1.
#

def ProfileBlendSet(config, psf, gsparams, logger):
    """
    Build a list of galaxy profiles, each convolved with the psf, to use for the blend image.
    """
    if ('blend_profiles' in config and
        config['obj_num'] < config['blend_first'] + len(config['blend_profiles'])):
        # Then the full image is already drawn.  Return None to indicate this to
        # the downstream functions.
        return None
    else:
        # Run the regular ProfileBlend function to build the profiles.
        profiles = ProfileBlend(config, psf, gsparams, logger)
        # Save this for next time
        config['blend_profiles'] = profiles
        # And mark this as the first object in the set
        config['blend_first'] = config['obj_num']
        return profiles


def DrawBlendSet(profiles, image, method, offset, config):
    """
    Draw the profiles onto the stamp.
    """
    if 'stamp_xsize' not in config or 'stamp_ysize' not in config:
        raise RuntimeError("stamp size must be given for stamp type=BlendSet")

    nx = config['stamp_xsize']
    ny = config['stamp_ysize']
    wcs = config['wcs']

    if profiles is None:
        # Then we've already drawn the full image.
        full_images = config['blend_full_images']
    else:
        # We need to draw an image large enough that each of the cutouts will be contained within.
        bounds = galsim.BoundsI(galsim.PositionI(0,0))
        for pos in config['blend_neighbor_pos']:
            image_pos = wcs.toImage(pos)
            # Convert to nearest integer position
            image_pos = galsim.PositionI( int(image_pos.x+0.5), int(image_pos.y+0.5) )
            bounds += image_pos
        bounds = bounds.withBorder(max(nx,ny)//2 + 1)

        full_images = []
        for prof in profiles:
            im = galsim.ImageF(bounds=bounds, wcs=wcs)
            im = galsim.config.DrawBasic(prof, im, method, offset-im.trueCenter(), config)
            full_images.append(im)

        # Save it in the config dict for next time.
        config['blend_full_images'] = full_images

    # Figure out what bounds to use for the cutouts.
    k = config['obj_num'] - config['blend_first']
    if k == 0:
        center_pos = galsim.PositionI(0,0)
    else:
        center_pos = config['blend_neighbor_pos'][k-1]
    center_image_pos = wcs.toImage(center_pos)
    xmin = int(center_image_pos.x) - nx//2 + 1
    ymin = int(center_image_pos.y) - ny//2 + 1
    bounds = galsim.BoundsI(xmin, xmin+nx-1, ymin, ymin+ny-1)
    # Save this for the noise cutout
    config['blend_bounds'] = bounds

    # Add up the cutouts from the profile images
    image.setZero()
    image.wcs = wcs
    for full_im in full_images:
        assert full_im.bounds.includes(bounds)
        image += full_im[bounds]

    # And also build the neighbor image for the deblend image
    neighbor_image = image.copy()
    neighbor_image -= full_images[k][bounds]
    config['blend_neighbor_image'] = neighbor_image

    return image

def WhitenBlendSet(profiles, image, config):
    """
    Whiten the noise on the stamp according to the existing noise in all the profiles.
    """
    k = config['obj_num'] - config['blend_first']
    # Only whiten the images once.
    if k == 0:
        current_var = 0
        for prof, full_im in zip(config['blend_profiles'], config['blend_full_images']):
            current_var += galsim.config.WhitenBasic(prof, full_im, config)
        config['blend_current_var'] = current_var
        if current_var != 0:
            # Then we whitened the noise somewhere.  Rebuild the stamp
            image.setZero()
            bounds = config['blend_bounds']
            for full_im in config['blend_full_images']:
                assert full_im.bounds.includes(bounds)
                image += full_im[bounds]
    else:
        current_var = config['blend_current_var']
    return current_var


def NoiseBlendSet(config, image, skip, current_var, logger):
    """Add the sky and noise"""
    # We want the noise realization to be the same for all galaxies in the set,
    # so we only generate the noise the first time and save it, pulling out the right cutout
    # for the subsequent stamps in this set.

    k = config['obj_num'] - config['blend_first']
    if k == 0:
        # If we are on the first galaxy, draw the noise.
        full_noise_image = config['blend_full_images'][0].copy()
        full_noise_image.setZero()
        galsim.config.NoiseBasic(config, full_noise_image, skip, current_var, logger)
        config['blend_full_noise_image'] = full_noise_image
    else:
        full_noise_image = config['blend_full_noise_image']

    bounds = config['blend_bounds']
    image += full_noise_image[bounds]
    return image


# Use the regular Blend functions for Setup and Whiten.
galsim.config.RegisterStampType('BlendSet',
                                setup_func = SetupBlend,
                                prof_func = ProfileBlendSet,
                                draw_func = DrawBlendSet,
                                whiten_func = WhitenBlendSet,
                                noise_func = NoiseBlendSet)


#
# Define the deblend extra output type.
# This is a custom extra output field to write a second file that is identical
# except for not including the flux of the neigbors.  We do this by grabbing the
# pre-noise postage stamp of the neighbor and then subtracting this off from the
# final postage stamp that includes all the noise.  So the Poisson noise of the
# neighbors is still there as is the noise from whitening.  Just not the actual
# neighbor flux.  The represents the result of a perfect deblender.
#

def DeblendStamp(images, scratch, config, base, obj_num, logger=None):
    """Save the stamps of just the neighbor fluxes.  We'll subtract them from the full image
    at the end.
    """
    im = base['blend_neighbor_image']
    # When we made neighbor image, we didn't have the right bounds yet, so set it now the
    # same way the main stamp's bounds are set.
    if im is None:
        pass
    elif base['stamp_center'] is not None:
        im.setCenter(base['stamp_center'])
    else:
        im.setOrigin(base['image_origin'])
    # Save it in the scratch dict using this obj_num as the key.
    scratch[obj_num] = im

def DeblendImage(images, scratch, config, base, obj_nums, logger=None):
    """Copy the full final image over and then subtract off the neighbor-only fluxes.
    """
    # Start with a copy of the regular final image.
    image = base['current_image'].copy()
    for obj_num in obj_nums:
        # Subtract off the images we made of the (noise-free) neighbors
        neighbor_image = scratch[obj_num]
        b = neighbor_image.bounds & image.getBounds()
        if b.isDefined():
            image[b] -= neighbor_image[b]
    # Save this in the list of images to write out.
    k = base['image_num'] - base['start_image_num']
    images[k] = image


galsim.config.RegisterExtraOutput('deblend',
                                  stamp_func = DeblendStamp,
                                  image_func = DeblendImage,
                                  write_func = galsim.fits.writeMulti)

#
# When used with MEDS files, we modify it slightly so that it can write out as a meds file,
# rather than a simple Fits file.
#

def DeblendMedsFinalize(images, scratch, data, config, base, logger):
    """Convert from the list of images we've been making into a list of MultiExposureObjects
    we can use to write the MEDS file.
    """
    # cf. code to make obj_list in BuildMEDS function in galsim/des/des_meds.py
    obj_list = []
    k1 = 0
    # Copy over everything but the images. Replace the data images with the deblend images.
    for obj in data:
        k2 = k1 + obj.n_cutouts
        new_images = images[k1:k2]
        weight = obj.weight
        psf = obj.psf
        new_obj = galsim.des.MultiExposureObject(images=new_images, weight=weight, psf=psf)
        obj_list.append(new_obj)
        k1 = k2
    return obj_list


galsim.config.RegisterExtraOutput('deblend_meds',
                                  stamp_func = DeblendStamp,
                                  image_func = DeblendImage,
                                  final_func = DeblendMedsFinalize,
                                  write_func = galsim.des.WriteMEDS)


