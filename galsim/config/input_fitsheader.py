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
import galsim

def _GenerateFromFitsHeader(param, param_name, base, value_type):
    """@brief Return a value read from a FITS header
    """
    if 'fits_header' not in base:
        raise ValueError("No fits header available for %s.type = FitsHeader"%param_name)

    req = { 'key' : str }
    opt = { 'num' : int }
    kwargs, safe = galsim.config.GetAllParams(param, param_name, base, req=req, opt=opt)
    key = kwargs['key']

    num = kwargs.get('num',0)
    if num < 0:
        raise ValueError("Invalid num < 0 supplied for FitsHeader: num = %d"%num)
    if num >= len(base['fits_header']):
        raise ValueError("Invalid num supplied for FitsHeader (too large): num = %d"%num)
    header = base['fits_header'][num]

    if key not in header.keys():
        raise ValueError("key %s not found in the FITS header in %s"%(key,kwargs['file_name']))

    val = header.get(key)
    #print base['file_num'],'Header: key = %s, val = %s'%(key,val)
    return val, safe


