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

def _type_by_letter(key):
    if len(key) < 2:
        raise AttributeError("Invalid user-defined variable %r"%key)
    if key[0] == 'f':
        return float
    elif key[0] == 'i':
        return int
    elif key[0] == 'b':
        return bool
    elif key[0] == 's':
        return str
    elif key[0] == 'a':
        return galsim.Angle
    elif key[0] == 'p':
        return galsim.PositionD
    elif key[0] == 'g':
        return galsim.Shear
    else:
        raise AttributeError("Invalid Eval variable: %s (starts with an invalid letter)"%key)

def _GenerateFromEval(config, base, value_type):
    """@brief Evaluate a string as the provided type
    """
    req = { 'str' : str }
    opt = {}
    ignore = galsim.config.value.standard_ignore
    for key in config.keys():
        if key not in (ignore + req.keys()):
            opt[key] = _type_by_letter(key)
    #print 'opt = ',opt
    #print 'base has ',base.keys()
            
    params, safe = galsim.config.GetAllParams(config, base, req=req, opt=opt, ignore=ignore)
    #print 'params = ',params
    string = params['str']
    #print 'string = ',string

    # Bring the user-defined variables into scope.
    for key in opt.keys():
        exec(key[1:] + ' = params[key]')
        #print key[1:],'=',eval(key[1:])

    # Also bring in any top level eval_variables
    if 'eval_variables' in base:
        #print 'found eval_variables = ',base['eval_variables']
        if not isinstance(base['eval_variables'],dict):
            raise AttributeError("eval_variables must be a dict")
        opt = {}
        for key in base['eval_variables'].keys():
            if key not in ignore:
                opt[key] = _type_by_letter(key)
        #print 'opt = ',opt
        params, safe1 = galsim.config.GetAllParams(base['eval_variables'], base, opt=opt,
                                     ignore=ignore)
        #print 'params = ',params
        safe = safe and safe1
        for key in opt.keys():
            exec(key[1:] + ' = params[key]')
            #print key[1:],'=',eval(key[1:])

    # Also, we allow the use of math functions
    import math
    import numpy
    import os

    # Try evaluating the string as is.
    try:
        val = value_type(eval(string))
        #print base['obj_num'],'Simple Eval(%s) = %s'%(string,val)
        return val, safe
    except:
        pass

    # Then try bringing in the allowed variables to see if that works:
    if 'image_pos' in base:
        image_pos = base['image_pos']
    if 'world_pos' in base:
        world_pos = base['world_pos']
    if 'image_center' in base:
        image_center = base['image_center']
    if 'image_origin' in base:
        image_origin = base['image_origin']
    if 'image_xsize' in base:
        image_xsize = base['image_xsize']
    if 'image_ysize' in base:
        image_ysize = base['image_ysize']
    if 'stamp_xsize' in base:
        stamp_xsize = base['stamp_xsize']
    if 'stamp_ysize' in base:
        stamp_ysize = base['stamp_ysize']
    if 'pixel_scale' in base:
        pixel_sclae = base['pixel_scale']
    if 'rng' in base:
        rng = base['rng']
    if 'file_num' in base:
        file_num = base.get('file_num',0)
    if 'image_num' in base:
        image_num = base.get('image_num',0)
    if 'obj_num' in base:
        obj_num = base['obj_num']
    if 'start_obj_num' in base:
        start_obj_num = base.get('start_obj_num',0)
    for key in galsim.config.valid_input_types.keys():
        if key in base:
            exec(key + ' = base[key]')
    try:
        val = value_type(eval(string))
        #print base['obj_num'],'Eval(%s) needed extra variables: val = %s'%(string,val)
        return val, False
    except:
        raise ValueError("Unable to evaluate string %r as a %s"%(string,value_type))

