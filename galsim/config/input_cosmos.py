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

import logging

from .input import InputLoader, GetInputObj, RegisterInputType
from .gsobject import RegisterObjectType
from .util import GetRNG
from .value import GetAllParams, SetDefaultIndex, RegisterValueType
from ..errors import GalSimConfigError
from ..gsparams import GSParams
from ..galaxy_sample import GalaxySample, COSMOSCatalog

# This file adds input types cosmos_catalog and sample_galaxy and gsobject types
# COSMOSGalaxy and SampleGalaxy.

# The SampleGalaxy doesn't need anything special other than registration as a valid input type.
# However, we do make a custom Loader so that we can add a logger line with some information about
# the number of objects in the catalog that passed the initial cuts and other basic catalog info.
class SampleLoader(InputLoader):
    def __init__(self, cls, input_field):
        self.cls_name = cls.__name__
        self.input_field = input_field
        super().__init__(cls)

    def setupImage(self, cosmos_cat, config, base, logger):
        if logger:
            # Only report as a warning the first time.  After that, use info.
            first = not base.get('_SampleLoader_reported_as_warning',False)
            base['_SampleLoader_reported_as_warning'] = True
            if first:
                log_level = logging.WARNING
            else:
                log_level = logging.INFO
            # It should be required that base['input']['cosmos_catalog'] exists, but
            # just in case someone calls this in a weird way, use get() with defaults.
            cc = base.get('input',{}).get(self.input_field,{})
            if isinstance(cc,list): cc = cc[0]
            out_str = ''
            if 'sample' in cc:
                out_str += '\n  sample = %s'%cc['sample']
            if 'dir' in cc:
                out_str += '\n  dir = %s'%cc['dir']
            if 'file_name' in cc:
                out_str += '\n  file_name = %s'%cc['file_name']
            if out_str != '':
                logger.log(log_level, 'Using user-specified %s: %s',self.cls_name, out_str)
            logger.info("file %d: Sample catalog has %d total objects; %d passed initial cuts.",
                        base['file_num'], cosmos_cat.getNTot(), cosmos_cat.nobjects)
            if base.get('gal',{}).get('gal_type',None) == 'parametric':
                logger.log(log_level,"Using parametric galaxies.")
            else:
                logger.log(log_level,"Using real galaxies.")

RegisterInputType('cosmos_catalog', SampleLoader(COSMOSCatalog, 'cosmos_catalog'))
RegisterInputType('galaxy_sample', SampleLoader(GalaxySample, 'galaxy_sample'))

# The gsobject types coupled to these are COSMOSGalaxy and SampleGalaxy respectively.

def _BuildCOSMOSGalaxy(config, base, ignore, gsparams, logger):
    """Build a COSMOS galaxy using the cosmos_catalog input item.
    """
    sample_cat = GetInputObj('cosmos_catalog', config, base, 'COSMOSGalaxy')
    return _FinishBuildSampleGalaxy(config, base, ignore, gsparams, logger,
                                    sample_cat, 'COSMOSGalaxy')

def _BuildSampleGalaxy(config, base, ignore, gsparams, logger):
    """Build a sample galaxy using the galaxy_sample input item.
    """
    sample_cat = GetInputObj('galaxy_sample', config, base, 'SampleGalaxy')
    return _FinishBuildSampleGalaxy(config, base, ignore, gsparams, logger,
                                    sample_cat, 'SampleGalaxy')

def _FinishBuildSampleGalaxy(config, base, ignore, gsparams, logger, sample_cat, cls_name):
    ignore = ignore + ['num']

    # Special: if galaxies are selected based on index, and index is Sequence or Random, and max
    # isn't set, set it to nobjects-1.
    if 'index' in config:
        SetDefaultIndex(config, sample_cat.nobjects)

    opt = { "index" : int,
            "gal_type" : str,
            "noise_pad_size" : float,
            "deep" : bool,
            "sersic_prec": float,
            "chromatic": bool,
            "area": float,
            "exptime": float,
    }

    kwargs, safe = GetAllParams(config, base, opt=opt, ignore=ignore)
    if gsparams: kwargs['gsparams'] = GSParams(**gsparams)

    rng = GetRNG(config, base, logger, cls_name)

    if 'index' not in kwargs:
        kwargs['index'], n_rng_calls = sample_cat.selectRandomIndex(1, rng=rng, _n_rng_calls=True)
        safe = False

        # Make sure this process gives consistent results regardless of the number of processes
        # being used.
        if not isinstance(sample_cat, GalaxySample) and rng is not None:
            # Then sample_cat is really a proxy, which means the rng was pickled, so we need to
            # discard the same number of random calls from the one in the config dict.
            rng.discard(int(n_rng_calls))

    kwargs['rng'] = rng

    # NB. Even though index is officially optional, it will always be present, either because it was
    #     set by a call to selectRandomIndex, explicitly by the user, or due to the call to
    #     SetDefaultIndex.
    index = kwargs['index']
    if index >= sample_cat.nobjects:
        raise GalSimConfigError(
            "index=%s has gone past the number of entries in the %s"%(index, cls_name))

    logger.debug('obj %d: %s kwargs = %s',base.get('obj_num',0), cls_name, kwargs)

    kwargs['self'] = sample_cat

    # Use a staticmethod of GalaxySample to avoid pickling the result of makeGalaxy()
    # The RealGalaxy in particular has a large serialization, so it is more efficient to
    # make it in this process, which is what happens here.
    gal = GalaxySample._makeGalaxy(**kwargs)

    return gal, safe

# Register this as a valid gsobject type
RegisterObjectType('COSMOSGalaxy', _BuildCOSMOSGalaxy, input_type='cosmos_catalog')
RegisterObjectType('SampleGalaxy', _BuildSampleGalaxy, input_type='galaxy_sample')

# Finally, also provide accessor to values in the parametric catalog

def _GetCOSMOSValue(config, base, value_type):
    cosmos_cat = GetInputObj('cosmos_catalog', config, base, 'COSMOSValue')
    req = { 'key': str, 'index': int }
    kwargs, safe = GetAllParams(config, base, req=req)
    return cosmos_cat.getValue(**kwargs)

def _GetSampleValue(config, base, value_type):
    sample_cat = GetInputObj('galaxy_sample', config, base, 'SampleValue')
    req = { 'key': str, 'index': int }
    kwargs, safe = GetAllParams(config, base, req=req)
    return sample_cat.getValue(**kwargs)

RegisterValueType('COSMOSValue', _GetCOSMOSValue, [float], input_type='cosmos_catalog')
RegisterValueType('SampleValue', _GetSampleValue, [float], input_type='galaxy_sample')
