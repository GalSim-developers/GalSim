# Copyright (c) 2012-2022 by the GalSim developers team on GitHub
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

class MixedSceneBuilder(galsim.config.StampBuilder):

    def setup(self, config, base, xsize, ysize, ignore, logger):
        if 'objects' not in config:
            raise AttributeError('objets field is required for MixedScene stamp type')
        objects = config['objects']

        # Propagate any stamp rng_num or index_key into the various object fields:
        objects.pop('rng_num', None)  # Also remove them from here if necessary.
        objects.pop('index_key', None)
        if not config.get('_propagated_rng_index', False):
            config['_propagated_rng_index'] = True
            rng_num = config.get('rng_num', None)
            index_key = config.get('index_key', None)
            for key in objects.keys():
                galsim.config.PropagateIndexKeyRNGNum(base[key], index_key, rng_num)

        rng = galsim.config.GetRNG(config, base)
        ud = galsim.UniformDeviate(rng)
        p = ud()  # A random number between 0 and 1.

        # If the user is careful, this will be 1, but if not, renormalize for them.
        norm = float(sum(objects.values()))

        if 'obj_type' in config:
            obj_type = galsim.config.ParseValue(config, 'obj_type', base, str)[0]
            obj_type_index = list(objects.keys()).index(obj_type)
        else:
            # Figure out which object field to use
            obj_type = None  # So we can check that it was set to something.
            obj_type_index = 0
            for key, value in objects.items():
                p1 = value / norm
                if p < p1:
                    # Use this object
                    obj_type = key
                    break
                else:
                    p -= p1
                    obj_type_index += 1
            if obj_type is None:
                # This shouldn't happen, but maybe possible from rounding errors.  Use the last one.
                obj_type = list(objects.keys())[-1]
                obj_type_index -= 1
                logger.error("Error in MixedScene.  Didn't pick an object to use.  Using %s",obj_type)

        # Save this in the dict so it can be used by e.g. the truth catalog or to do something
        # different depending on which kind of object we have.
        base['current_obj_type'] = obj_type
        base['current_obj_type_index'] = obj_type_index
        base['current_obj'] = None

        # Add objects field to the ignore list
        # Also ignore magnify and shear, which we allow here for convenience to act on whichever
        # object ends up being chosen.
        ignore = ignore + ['objects', 'magnify', 'shear', 'obj_type']

        # Now go on and do the rest of the normal setup.
        return super(MixedSceneBuilder, self).setup(config,base,xsize,ysize,ignore,logger)

    def buildProfile(self, config, base, psf, gsparams, logger):
        obj_type = base['current_obj_type']
        logger.info('obj %d: Drawing %s', base['obj_num'], obj_type)

        # Make the appropriate object using the obj_type field
        obj = galsim.config.BuildGSObject(base, obj_type, gsparams=gsparams, logger=logger)[0]
        # Also save this in case useful for some calculation.
        base['current_obj'] = obj

        # Only shear and magnify are allowed, but this general TransformObject function will
        # work to implement those.
        obj, safe = galsim.config.TransformObject(obj, config, base, logger)

        if psf:
            if obj:
                return galsim.Convolve(obj,psf)
            else:
                return psf
        else:
            if obj:
                return obj
            else:
                return None

galsim.config.stamp.RegisterStampType('MixedScene', MixedSceneBuilder())
