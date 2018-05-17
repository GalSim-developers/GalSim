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
from __future__ import print_function

import galsim
import numpy as np
import re

# This file handles the parsing for the special Eval type.

def _type_by_letter(key):
    if len(key) < 2:
        raise galsim.GalSimConfigError("Invalid user-defined variable %r"%key)
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
    elif key[0] == 'c':
        return galsim.CelestialCoord
    elif key[0] == 'g':
        return galsim.Shear
    elif key[0] == 'x':
        return None
    else:
        raise galsim.GalSimConfigError(
            "Invalid Eval variable: %s (starts with an invalid letter)"%key)

eval_base_variables = [ 'image_pos', 'world_pos', 'image_center', 'image_origin', 'image_bounds',
                        'image_xsize', 'image_ysize', 'stamp_xsize', 'stamp_ysize', 'pixel_scale',
                        'wcs', 'rng', 'file_num', 'image_num', 'obj_num', 'start_obj_num',
                        'world_center', ]

from .value import standard_ignore
eval_ignore = ['str','_fn'] + standard_ignore

def _isWordInString(w, s):
    # Return if a given word is in the given string.
    # Note, this specifically looks for the whole word. e.g. if w = 'yes' and s = 'eyestrain',
    # then `w in s` returns True, but `_isWordInString(w,s)` returns False.
    # cf. https://stackoverflow.com/questions/5319922/python-check-if-word-is-in-a-string
    return re.search(r'\b({0})\b'.format(w),s) is not None

def _GenerateFromEval(config, base, value_type):
    """@brief Evaluate a string as the provided type
    """
    #print('Start Eval')
    #print('config = ',galsim.config.CleanConfig(config))
    if '_value' in config:
        return config['_value']
    elif '_fn' in config:
        #print('Using saved function')
        fn = config['_fn']
    else:
        # If the function is not already compiled, then this is the first time through, so do
        # a full parsing of all the possibilities.

        from future.utils import exec_
        # These will be the variables to use for evaluating the eval statement.
        # Start with the current locals and globals, and add extra items to them.
        if 'eval_gdict' not in base:
            gdict = globals().copy()
            # We allow the following modules to be used in the eval string:
            exec_('import math', gdict)
            exec_('import numpy', gdict)
            exec_('import numpy as np', gdict)
            exec_('import os', gdict)
            base['eval_gdict'] = gdict
        else:
            gdict = base['eval_gdict']

        if 'str' not in config:
            raise galsim.GalSimConfigError(
                "Attribute str is required for type = %s"%(config['type']))
        string = config['str']

        # Turn any "Current" items indicated with an @ sign into regular variables.
        if '@' in string:
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
                config['x' + key_name] = { 'type' : 'Current', 'key' : key }

        # The parameters to the function are the keys in the config dict minus their initial char.
        params = [ key[1:] for key in config.keys() if key not in eval_ignore ]

        # Also bring in any top level eval_variables that might be relevant.
        if 'eval_variables' in base:
            #print('found eval_variables = ',galsim.config.CleanConfig(base['eval_variables']))
            if not isinstance(base['eval_variables'],dict):
                raise galsim.GalSimConfigError("eval_variables must be a dict")
            for key in base['eval_variables']:
                # Only add variables that appear in the string.
                if _isWordInString(key[1:],string) and key[1:] not in params:
                    config[key] = { 'type' : 'Current',
                                    'key' : 'eval_variables.' + key }
                    params.append(key[1:])

        # Also check for the allowed base variables:
        for key in eval_base_variables:
            if key in base and _isWordInString(key,string) and key not in params:
                config['x' + key] = { 'type' : 'Current', 'key' : key }
                params.append(key)
        #print('params = ',params)
        #print('config = ',config)

        # Now compile the string into a lambda function, which will be faster for subsequent
        # passes into this builder.
        try:
            if len(params) == 0:
                value = eval(string, gdict)
                config['_value'] = value
                return value
            else:
                fn_str = 'lambda %s: %s'%(','.join(params), string)
                #print('fn_str = ',fn_str)
                fn = eval(fn_str, gdict)
                config['_fn'] = fn
        except KeyboardInterrupt:
            raise
        except Exception as e:
            raise galsim.GalSimConfigError(
                "Unable to evaluate string %r as a %s\n%r"%(string, value_type, e))
 
    # Always need to evaluate any parameters to pass to the function
    opt = {}
    for key in config.keys():
        if key not in eval_ignore:
            opt[key] = _type_by_letter(key)
    #print('opt = ',opt)
    params, safe = galsim.config.GetAllParams(config, base, opt=opt, ignore=eval_ignore)
    #print('params = ',params)

    # Strip off the first character of the keys
    params = { key[1:] : value for key, value in params.items() }
    #print('params => ',params)

    # Evaluate the compiled function
    try:
        val = fn(**params)
        #print('val = ',val)
        return val, safe
    except KeyboardInterrupt:
        raise
    except Exception as e:
        raise galsim.GalSimConfigError(
            "Unable to evaluate string %r as a %s\n%r"%(config['str'],value_type, e))


# Register this as a valid value type
from .value import RegisterValueType
RegisterValueType('Eval', _GenerateFromEval,
                  [ float, int, bool, str, galsim.Angle, galsim.Shear, galsim.PositionD,
                    galsim.CelestialCoord, None ])
