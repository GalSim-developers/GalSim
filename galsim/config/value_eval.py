# Copyright (c) 2012-2016 by the GalSim developers team on GitHub
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
from __future__ import print_function

import galsim

# This file handles the parsing for the special Eval type.

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
    elif key[0] == 'x':
        return None
    else:
        raise AttributeError("Invalid Eval variable: %s (starts with an invalid letter)"%key)

def _GenerateFromEval(config, base, value_type):
    """@brief Evaluate a string as the provided type
    """
    # We allow the following modules to be used in the eval string:
    import math
    import numpy
    import numpy as np  # Both np.* and numpy.* are allowed.
    import os

    # These will be the variables to use for evaluating the eval statement.
    # Start with the current locals and globals, and add extra items to them.
    ldict = locals().copy()
    gdict = globals().copy()

    #print('Start Eval')
    req = { 'str' : str }
    opt = {}
    ignore = galsim.config.standard_ignore  # in value.py
    for key in config.keys():
        if key not in (ignore + list(req)):
            opt[key] = _type_by_letter(key)
    #print('opt = ',opt)
    #print('base has ',base.keys())
    #print('config = ',config)

    if isinstance(config['str'], str):
        # The ParseValue function can get confused if the first character is an @, but the 
        # whole string isn't a Current item.  e.g. @image.pixel_scale * @image.stamp_size.
        # So if config['str'] is a string, just get it.  Otherwise, try parsing the dict.
        string = config['str']
        params, safe = galsim.config.GetAllParams(config, base, opt=opt, ignore=ignore+['str'])
    else:
        params, safe = galsim.config.GetAllParams(config, base, req=req, opt=opt, ignore=ignore)
        #print('params = ',params)
        string = params['str']
    #print('string = ',string)

    # Parse any "Current" items indicated with an @ sign.
    if '@' in string:
        import re
        # Find @items using regex.  They can include alphanumeric chars plus '.'.
        keys = re.findall(r'@[\w\.]*', string)
        #print('@keys = ',keys)
        # Remove duplicates
        keys = np.unique(keys).tolist()
        #print('unique @keys = ',keys)
        for key0 in keys:
            key = key0[1:] # Remove the @ sign.
            value = galsim.config.GetCurrentValue(key, base)
            # Give a probably unique name to this value
            key_name = "temp_variable_" + key.replace('.','_')
            #print('key_name = ',key_name)
            #print('value = ',value)
            # Replaces all occurrences of key0 with the key_name.
            string = string.replace(key0,key_name)
            # Finally, bring the key's variable name into scope.
            ldict[key_name] = value

    # Bring the user-defined variables into scope.
    #print('Loading keys in ',opt)
    for key in opt:
        #print('key = ',key)
        ldict[key[1:]] = params[key]
        #print(key[1:],'=',eval(key[1:],gdict,ldict))

    # Also bring in any top level eval_variables that might be relevant.
    if 'eval_variables' in base:
        #print('found eval_variables = ',base['eval_variables'])
        if not isinstance(base['eval_variables'],dict):
            raise AttributeError("eval_variables must be a dict")
        opt = {}
        ignore = []
        for key in base['eval_variables']:
            # Only add variables that appear in the string.
            if key[1:] in string:
                opt[key] = _type_by_letter(key)
            else:
                ignore.append(key)
        #print('opt = ',opt)
        params, safe1 = galsim.config.GetAllParams(base['eval_variables'],
                                                   base, opt=opt, ignore=ignore)
        #print('params = ',params)
        safe = safe and safe1
        for key in opt:
            #print('key = ',key)
            ldict[key[1:]] = params[key]
            #print(key[1:],'=',eval(key[1:],gdict,ldict))

    # Try evaluating the string as is.
    try:
        val = eval(string, gdict, ldict)
        if value_type is not None:
            val = value_type(val)
        #print(base['obj_num'],'Simple Eval(%s) = %s'%(string,val))
        return val, safe
    except KeyboardInterrupt:
        raise
    except:
        pass

    # Then try bringing in the allowed variables to see if that works:
    base_variables = [ 'image_pos', 'world_pos', 'image_center', 'image_origin', 'image_bounds',
                       'image_xsize', 'image_ysize', 'stamp_xsize', 'stamp_ysize', 'pixel_scale',
                       'wcs', 'rng', 'file_num', 'image_num', 'obj_num', 'start_obj_num', ]
    for key in base_variables:
        if key in base:
            ldict[key] = base[key]
    try:
        val = eval(string, gdict, ldict)
        #print(base['obj_num'],'Eval(%s) needed extra variables: val = %s'%(string,val))
        if value_type is not None:
            val = value_type(val)
        return val, False
    except KeyboardInterrupt:
        raise
    except Exception as e:
        raise ValueError("Unable to evaluate string %r as a %s\n"%(string,value_type) + str(e))


# Register this as a valid value type
from .value import RegisterValueType
RegisterValueType('Eval', _GenerateFromEval, 
                  [ float, int, bool, str, galsim.Angle, galsim.Shear, galsim.PositionD, None ])
