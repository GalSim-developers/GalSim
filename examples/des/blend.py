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

# This file defines a few custom types related to making stamps of blended galaxies.

import galsim
import galsim.des
import os
import math
import numpy

#
# Define the Blend stamp type
#

def BuildBlendProfiles(self, config, base, psf, gsparams, logger):
    """
    Build a list of galaxy profiles, each convolved with the psf, to use for the blend image.

    This is used by both BlendBuilder and BlendSetBuilder.
    """
    # Build the neighbors first, so the final "current" value for gal is the main galaxy.
    n_neighbors = galsim.config.ParseValue(config, 'n_neighbors', base, int)[0]
    min_sep = galsim.config.ParseValue(config, 'min_sep', base, float)[0]
    max_sep = galsim.config.ParseValue(config, 'max_sep', base, float)[0]

    self.neighbor_gals = []
    for i in range(n_neighbors):
        gal = galsim.config.BuildGSObject(base, 'gal', gsparams=gsparams, logger=logger)[0]
        self.neighbor_gals.append(gal)
        # Remove the current stuff from base['gal'] so we don't get the same galaxy
        # each time.
        galsim.config.RemoveCurrent(base['gal'], keep_safe=True)

    # GalSim has a RandomCircle position type that we can use to generate random positions
    # in an annulus.  Of course, we could do this be hand here, but this is easier.
    self.neighbor_pos = []
    for i in range(n_neighbors):
        config_pos = { 'pos' : { 'type' : 'RandomCircle',
                                 'radius' : max_sep,
                                 'inner_radius' : min_sep }
                     }
        pos = galsim.config.ParseValue(config_pos, 'pos', base, galsim.PositionD)[0]
        self.neighbor_pos.append(pos)

    self.main_gal = galsim.config.BuildGSObject(base, 'gal', gsparams=gsparams,
                                                logger=logger)[0]

    profiles = [ self.main_gal ]
    profiles += [ gal.shift(pos) for gal, pos in zip(self.neighbor_gals, self.neighbor_pos) ]
    if psf:
        profiles = [ galsim.Convolve(gal, psf) for gal in profiles ]

    return profiles


class BlendBuilder(galsim.config.StampBuilder):

    def setup(self, config, base, xsize, ysize, ignore, logger):
        """Do the appropriate setup for a Blend stamp.
        """
        self.first = None  # Mark that we don't have anything stored yet.

        # Parse the necessary parameters and make sure they are the right type.
        req = { 'n_neighbors' : int, 'min_sep' : float, 'max_sep' : float }
        galsim.config.CheckAllParams(config, req=req, ignore=ignore)

        # Now farm off to the regular stamp setup function the rest of the work of parsing
        # the size and position of the stamp.
        ignore = ignore + ['n_neighbors', 'min_sep', 'max_sep']
        return super(self.__class__,self).setup(config, base, xsize, ysize, ignore, logger)

    def buildProfile(self, config, base, psf, gsparams, logger):
        return BuildBlendProfiles(self, config, base, psf, gsparams, logger)

    def draw(self, profiles, image, method, offset, config, base, logger):
        """
        Draw the profiles onto the stamp.
        """
        n_neighbors = len(profiles)-1

        # Draw the central galaxy using the basic draw function.
        image = galsim.config.DrawBasic(profiles[0], image, method, offset, config, base, logger)

        # We'll want a copy of just the neighbors for the deblend image.
        # Otherwise we could have just drawn these on the main image with add_to_image = True
        self.neighbor_image = image.copy()
        self.neighbor_image.setZero()

        # Draw all the neighbor stamps
        for p in profiles[1:]:
            galsim.config.DrawBasic(p, self.neighbor_image, method, offset, config, base, logger,
                                    add_to_image=True)

        # Save this in base for the deblend output
        base['blend_neighbor_image'] = self.neighbor_image

        image += self.neighbor_image

        return image

    def whiten(self, profiles, image, config, base, logger):
        """
        Whiten the noise on the stamp according to the existing noise in all the profiles.
        """
        total = galsim.Add(profiles)
        return super(self.__class__,self).whiten(total, image, config, base, logger)


galsim.config.RegisterStampType('Blend', BlendBuilder())


#
# Define the BlendSet stamp type
# This currently doesn't work with image.nproc != 1.
#

class BlendSetBuilder(galsim.config.StampBuilder):

    # This is the same as the setup function for Blend, so there is a bit of duplicated code
    # here, but it was simpler to have BlendSetBuilder derive directly from StampBuilder
    # so the super calls go directly to that.
    def setup(self, config, base, xsize, ysize, ignore, logger):
        """Do the appropriate setup for a Blend stamp.
        """
        # Make sure that we start over on the start of a new file.
        if base['obj_num'] == base['start_obj_num']:
            self.first = None

        # Parse the necessary parameters and make sure they are the right type.
        req = { 'n_neighbors' : int, 'min_sep' : float, 'max_sep' : float }
        galsim.config.CheckAllParams(config, req=req, ignore=ignore)

        # Now farm off to the regular stamp setup function the rest of the work of parsing
        # the size and position of the stamp.
        ignore = ignore + ['n_neighbors', 'min_sep', 'max_sep']
        return super(self.__class__, self).setup(config, base, xsize, ysize, ignore, logger)

    def buildProfile(self, config, base, psf, gsparams, logger):
        """
        Build a list of galaxy profiles, each convolved with the psf, to use for the blend image.
        """
        if (self.first is not None and base['obj_num'] < self.first + len(self.profiles)):
            # Then the full image is already drawn.  Return None to indicate this to
            # the downstream functions.
            return None
        else:
            # Run the above BuildBlendProfiles function to build the profiles.
            self.profiles = BuildBlendProfiles(self, config, base, psf, gsparams, logger)
            # And mark this as the first object in the set
            self.first = base['obj_num']
            return self.profiles

    def draw(self, profiles, image, method, offset, config, base, logger):
        """
        Draw the profiles onto the stamp.
        """
        if 'stamp_xsize' not in base or 'stamp_ysize' not in base:
            raise RuntimeError("stamp size must be given for stamp type=BlendSet")

        nx = base['stamp_xsize']
        ny = base['stamp_ysize']
        wcs = base['wcs']

        if profiles is not None:
            # Then we haven't drawn the full image yet.
            # We need to draw an image large enough to contain each of the cutouts
            bounds = galsim.BoundsI(galsim.PositionI(0,0))
            for pos in self.neighbor_pos:
                image_pos = wcs.toImage(pos)
                # Convert to nearest integer position
                image_pos = galsim.PositionI( int(image_pos.x+0.5), int(image_pos.y+0.5) )
                bounds += image_pos
            bounds = bounds.withBorder(max(nx,ny)//2 + 1)

            self.full_images = []
            for prof in profiles:
                im = galsim.ImageF(bounds=bounds, wcs=wcs)
                galsim.config.DrawBasic(prof, im, method, offset-im.true_center, config, base,
                                        logger)
                self.full_images.append(im)

        # Figure out what bounds to use for the cutouts.
        k = base['obj_num'] - self.first
        if k == 0:
            center_pos = galsim.PositionI(0,0)
        else:
            center_pos = self.neighbor_pos[k-1]
        center_image_pos = wcs.toImage(center_pos)
        xmin = int(center_image_pos.x) - nx//2 + 1
        ymin = int(center_image_pos.y) - ny//2 + 1
        self.bounds = galsim.BoundsI(xmin, xmin+nx-1, ymin, ymin+ny-1)

        # Add up the cutouts from the profile images
        image.setZero()
        image.wcs = wcs
        for full_im in self.full_images:
            assert full_im.bounds.includes(self.bounds)
            image += full_im[self.bounds]

        # And also build the neighbor image for the deblend image
        self.neighbor_image = image.copy()
        self.neighbor_image -= self.full_images[k][self.bounds]

        # Save this in base for the deblend output
        base['blend_neighbor_image'] = self.neighbor_image

        return image

    def whiten(self, profiles, image, config, base, logger):
        """
        Whiten the noise on the stamp according to the existing noise in all the profiles.
        """
        k = base['obj_num'] - self.first
        # Only whiten the images once.
        if k == 0:
            self.current_var = 0
            for prof, full_im in zip(self.profiles, self.full_images):
                self.current_var += super(self.__class__,self).whiten(
                        prof, full_im, config, base, logger)
            if self.current_var != 0:
                # Then we whitened the noise somewhere.  Rebuild the stamp
                image.setZero()
                for full_im in self.full_images:
                    assert full_im.bounds.includes(self.bounds)
                    image += full_im[self.bounds]
        return self.current_var


    def addNoise(self, config, base, image, skip, current_var, logger):
        """Add the sky and noise"""
        # We want the noise realization to be the same for all galaxies in the set,
        # so we only generate the noise the first time and save it, pulling out the right cutout
        # for the subsequent stamps in this set.

        k = base['obj_num'] - self.first
        if k == 0:
            # If we are on the first galaxy, draw the noise.
            self.full_noise_image = self.full_images[0].copy()
            self.full_noise_image.setZero()
            self.full_noise_image, self.current_var = super(self.__class__,self).addNoise(
                    config, base, self.full_noise_image, skip, current_var, logger)

        image += self.full_noise_image[self.bounds]
        return image, self.current_var


# Use the regular Blend functions for Setup and Whiten.
galsim.config.RegisterStampType('BlendSet', BlendSetBuilder())


#
# Define the deblend extra output type.
# This is a custom extra output field to write a second file that is identical
# except for not including the flux of the neigbors.  We do this by grabbing the
# pre-noise postage stamp of the neighbor and then subtracting this off from the
# final postage stamp that includes all the noise.  So the Poisson noise of the
# neighbors is still there as is the noise from whitening.  Just not the actual
# neighbor flux.  The represents the result of a perfect deblender.
#

class DeblendBuilder(galsim.config.ExtraOutputBuilder):

    def processStamp(self, obj_num, config, base, logger):
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
        self.scratch[obj_num] = im

    def processImage(self, index, obj_nums, config, base, logger):
        """Copy the full final image over and then subtract off the neighbor-only fluxes.
        """
        # Start with a copy of the regular final image.
        image = base['current_image'].copy()
        for obj_num in obj_nums:
            # Subtract off the images we made of the (noise-free) neighbors
            neighbor_image = self.scratch[obj_num]
            b = neighbor_image.bounds & image.bounds
            if b.isDefined():
                image[b] -= neighbor_image[b]
        # Save this in the list of images to write out.
        self.data[index] = image


galsim.config.RegisterExtraOutput('deblend', DeblendBuilder())

#
# When used with MEDS files, we modify it slightly so that it can write out as a meds file,
# rather than a simple Fits file.
#

class DeblendMedsBuilder(DeblendBuilder):

    def finalize(self, config, base, main_data, logger):
        """Convert from the list of images we've been making into a list of MultiExposureObjects
        we can use to write the MEDS file.
        """
        # cf. code to make obj_list in BuildMEDS function in galsim/des/des_meds.py
        obj_list = []
        k1 = 0
        # Copy over everything but the images. Replace the data images with the deblend images.
        for obj in main_data:
            k2 = k1 + obj.n_cutouts
            new_images = self.data[k1:k2]
            weight = obj.weight
            psf = obj.psf
            new_obj = galsim.des.MultiExposureObject(images=new_images, weight=weight, psf=psf)
            obj_list.append(new_obj)
            k1 = k2
        return obj_list

    def writeFile(self, file_name, config, base, logger):
        galsim.des.WriteMEDS(self.final_data, file_name)


galsim.config.RegisterExtraOutput('deblend_meds', DeblendMedsBuilder())


