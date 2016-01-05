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

# This file defines a few custom types for use by blend.yaml.

import galsim
import os
import math

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

    # BlendSet will set these to mean tha twe don't want to rebuild fresh galaxies, so if this
    # is present, just take the galaxies from here.
    if 'use_neighbor_gals' in config:
        neighbor_gals = config['use_neighbor_gals']
    else:
        neighbor_gals = []
        for i in range(n_neighbors):
            gal = galsim.config.BuildGSObject(config, 'gal', gsparams=gsparams, logger=logger)[0]
            neighbor_gals.append(gal)
            # Remove the current_val stuff from config['gal'] so we don't get the same galaxy
            # each time.
            galsim.config.RemoveCurrent(config['gal'], keep_safe=True)
        # Save this for BlendSet to use for the next blend if necessary.
        config['neighbor_gals'] = neighbor_gals

    if 'use_neighbor_pos' in config:
        neighbor_pos = config['use_neighbor_pos']
    else:
        # GalSim has a RandomCircle position type that we can use to generate random positions
        # in an annulus.
        neighbor_pos = []
        for i in range(n_neighbors):
            config_pos = { 'pos' : { 'type': 'RandomCircle',
                                     'radius': max_sep,
                                     'inner_radius' : min_sep } 
                         }
            pos = galsim.config.ParseValue(config_pos, 'pos', config, galsim.PositionD)[0]
            neighbor_pos.append(pos)
        config['neighbor_pos'] = neighbor_pos

    # Again, BlendSet might have set this up already.
    if 'use_main_gal' in config:
        main_gal = config['use_main_gal']
    else:
        main_gal = galsim.config.BuildGSObject(config, 'gal', gsparams=gsparams, logger=logger)[0]
        config['main_gal'] = main_gal

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
    print config['obj_num'], profiles[0]
    galsim.config.DrawBasic(profiles[0], image, method, offset, config)

    # We'll want a copy of just the neighbors for the noise-only blend image.
    # Otherwise we could have just drawn these on the main image with add_to_image = True
    neighbor_image = image.copy()
    neighbor_image.setZero()

    # Draw all the neighbor stamps
    for p in profiles[1:]:
        galsim.config.DrawBasic(p, neighbor_image, method, offset, config)

    # Save this for the noise-only processing
    config['neighbor_image'] = neighbor_image

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
# Define the blend_noiseonly extra output type.
#

def BlendNoiseOnlyStamp(images, scratch, config, base, obj_num, logger=None):
    """Save the stamps of just the neighbor fluxes.  We'll subtract them from the full image
    at the end.
    """
    im = base['neighbor_image']
    # When we made neighbor image, we didn't have the right bounds yet, so set it now the
    # same way the main stamp's bounds are set.
    if im is None:
        pass
    elif base['stamp_center'] is not None:
        im.setCenter(base['stamp_center'])
    else:
        im.setOrigin(config['image_origin'])
    # Save it in the scratch dict using this obj_num as the key.
    scratch[obj_num] = im

def BlendNoiseOnlyImage(images, scratch, config, base, obj_nums, logger=None):
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


galsim.config.RegisterExtraOutput('blend_noiseonly',
                                  stamp_func = BlendNoiseOnlyStamp,
                                  image_func = BlendNoiseOnlyImage,
                                  write_func = galsim.fits.writeMulti)


