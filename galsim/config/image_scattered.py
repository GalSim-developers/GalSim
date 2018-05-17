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

import galsim
import logging

# This file adds image type Scattered, which places individual stamps at arbitrary
# locations on a larger image.

from .image import ImageBuilder
class ScatteredImageBuilder(ImageBuilder):

    def setup(self, config, base, image_num, obj_num, ignore, logger):
        """Do the initialization and setup for building the image.

        This figures out the size that the image will be, but doesn't actually build it yet.

        @param config       The configuration dict for the image field.
        @param base         The base configuration dict.
        @param image_num    The current image number.
        @param obj_num      The first object number in the image.
        @param ignore       A list of parameters that are allowed to be in config that we can
                            ignore here. i.e. it won't be an error if these parameters are present.
        @param logger       If given, a logger object to log progress.

        @returns xsize, ysize
        """
        logger.debug('image %d: Building Scattered: image, obj = %d,%d',
                     image_num,image_num,obj_num)

        self.nobjects = self.getNObj(config, base, image_num)
        logger.debug('image %d: nobj = %d',image_num,self.nobjects)

        # These are allowed for Scattered, but we don't use them here.
        extra_ignore = [ 'image_pos', 'world_pos', 'stamp_size', 'stamp_xsize', 'stamp_ysize',
                         'nobjects' ]
        opt = { 'size' : int , 'xsize' : int , 'ysize' : int }
        params = galsim.config.GetAllParams(config, base, opt=opt, ignore=ignore+extra_ignore)[0]

        size = params.get('size',0)
        full_xsize = params.get('xsize',size)
        full_ysize = params.get('ysize',size)

        if (full_xsize <= 0) or (full_ysize <= 0):
            raise galsim.GalSimConfigError(
                "Both image.xsize and image.ysize need to be defined and > 0.")

        # If image_force_xsize and image_force_ysize were set in config, make sure it matches.
        if ( ('image_force_xsize' in base and full_xsize != base['image_force_xsize']) or
             ('image_force_ysize' in base and full_ysize != base['image_force_ysize']) ):
            raise galsim.GalSimConfigError(
                "Unable to reconcile required image xsize and ysize with provided "
                "xsize=%d, ysize=%d, "%(full_xsize,full_ysize))

        return full_xsize, full_ysize


    def buildImage(self, config, base, image_num, obj_num, logger):
        """Build an Image containing multiple objects placed at arbitrary locations.

        @param config       The configuration dict for the image field.
        @param base         The base configuration dict.
        @param image_num    The current image number.
        @param obj_num      The first object number in the image.
        @param logger       If given, a logger object to log progress.

        @returns the final image and the current noise variance in the image as a tuple
        """
        full_xsize = base['image_xsize']
        full_ysize = base['image_ysize']
        wcs = base['wcs']

        full_image = galsim.ImageF(full_xsize, full_ysize)
        full_image.setOrigin(base['image_origin'])
        full_image.wcs = wcs
        full_image.setZero()
        base['current_image'] = full_image

        if 'image_pos' in config and 'world_pos' in config:
            raise galsim.GalSimConfigValueError(
                "Both image_pos and world_pos specified for Scattered image.",
                (config['image_pos'], config['world_pos']))

        if 'image_pos' not in config and 'world_pos' not in config:
            xmin = base['image_origin'].x
            xmax = xmin + full_xsize-1
            ymin = base['image_origin'].y
            ymax = ymin + full_ysize-1
            config['image_pos'] = {
                'type' : 'XY' ,
                'x' : { 'type' : 'Random' , 'min' : xmin , 'max' : xmax },
                'y' : { 'type' : 'Random' , 'min' : ymin , 'max' : ymax }
            }

        stamps, current_vars = galsim.config.BuildStamps(
                self.nobjects, base, logger=logger, obj_num=obj_num, do_noise=False)

        base['index_key'] = 'image_num'

        for k in range(self.nobjects):
            # This is our signal that the object was skipped.
            if stamps[k] is None: continue
            bounds = stamps[k].bounds & full_image.bounds
            logger.debug('image %d: full bounds = %s',image_num,str(full_image.bounds))
            logger.debug('image %d: stamp %d bounds = %s',image_num,k,str(stamps[k].bounds))
            logger.debug('image %d: Overlap = %s',image_num,str(bounds))
            if bounds.isDefined():
                full_image[bounds] += stamps[k][bounds]
            else:
                logger.info(
                    "Object centered at (%d,%d) is entirely off the main image, "
                    "whose bounds are (%d,%d,%d,%d)."%(
                        stamps[k].center.x, stamps[k].center.y,
                        full_image.bounds.xmin, full_image.bounds.xmax,
                        full_image.bounds.ymin, full_image.bounds.ymax))

        # Bring the image so far up to a flat noise variance
        current_var = galsim.config.FlattenNoiseVariance(
                base, full_image, stamps, current_vars, logger)

        return full_image, current_var

    def makeTasks(self, config, base, jobs, logger):
        """Turn a list of jobs into a list of tasks.

        Here we just have one job per task.

        @param config       The configuration dict for the image field.
        @param base         The base configuration dict.
        @param jobs         A list of jobs to split up into tasks.  Each job in the list is a
                            dict of parameters that includes 'image_num' and 'obj_num'.
        @param logger       If given, a logger object to log progress.

        @returns a list of tasks
        """
        return [ [ (job, k) ] for k, job in enumerate(jobs) ]

    def addNoise(self, image, config, base, image_num, obj_num, current_var, logger):
        """Add the final noise to a Scattered image

        @param image        The image onto which to add the noise.
        @param config       The configuration dict for the image field.
        @param base         The base configuration dict.
        @param image_num    The current image number.
        @param obj_num      The first object number in the image.
        @param current_var  The current noise variance in each postage stamps.
        @param logger       If given, a logger object to log progress.
        """
        base['current_noise_image'] = base['current_image']
        galsim.config.AddSky(base,image)
        galsim.config.AddNoise(base,image,current_var,logger)


    def getNObj(self, config, base, image_num):
        """Get the number of objects that will be built for this image.

        @param config       The configuration dict for the image field.
        @param base         The base configuration dict.
        @param image_num    The current image number.

        @returns the number of objects
        """
        orig_index_key = base.get('index_key',None)
        base['index_key'] = 'image_num'
        base['image_num'] = image_num

        # Allow nobjects to be automatic based on input catalog
        if 'nobjects' not in config:
            nobj = galsim.config.ProcessInputNObjects(base)
            if nobj is None:
                raise galsim.GalSimConfigError(
                    "Attribute nobjects is required for image.type = Scattered")
        else:
            nobj = galsim.config.ParseValue(config,'nobjects',base,int)[0]
        base['index_key'] = orig_index_key
        return nobj

# Register this as a valid image type
from .image import RegisterImageType
RegisterImageType('Scattered', ScatteredImageBuilder())


