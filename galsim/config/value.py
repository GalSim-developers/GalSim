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

from past.builtins import basestring
import sys
import galsim

# This file handles the parsing of values given in the config dict.  It includes the basic
# parsing functionality along with generators for most of the simple value types.
# Additional value types are defined in value_random.py, value_eval.py, input.py,
# input_powerspectrum.py, input_nfw.py, and input_fitsheader.

# This module-level dict will store all the registered value types.
# See the RegisterValueType function at the end of this file.
# The keys are the (string) names of the value types, and the values are a tuple of the
# function to call to generate the value and a list of the types (float, int, str, etc.)
# that the value type is able to generate.
valid_value_types = {}


# Standard keys to ignore while parsing values:
standard_ignore = [
    'type', 'current', 'index_key', 'repeat', 'rng_num', '_gen_fn', '_get',
    '#' # When we read in json files, there represent comments
]

def ParseValue(config, key, base, value_type):
    """@brief Read or generate a parameter value from config.

    @returns the tuple (value, safe).
    """
    param = config[key]
    #print('ParseValue for key = ',key,', value_type = ',str(value_type))
    #print('param = ',param)
    #print('nums = ',base.get('file_num',0), base.get('image_num',0), base.get('obj_num',0))

    if isinstance(param, dict):

        type_name = param.get('type',None)
        #print('type = ',type_name)
        #print(param['type'], value_type)

        # Check what index key we want to use for this value.
        index, index_key = galsim.config.GetIndex(param, base, is_sequence=(type_name=='Sequence'))
        #print('index, index_key = ',index,index_key)

        if '_gen_fn' in param:
            generate_func = param['_gen_fn']

            if 'current' in param:
                cval, csafe, cvalue_type, cindex, cindex_key = param['current']
                if 'repeat' in param:
                    repeat = galsim.config.ParseValue(param, 'repeat', base, int)[0]
                    use_current = (cindex//repeat == index//repeat)
                else:
                    use_current = (cindex == index)
                if use_current:
                    if (value_type is not None and cvalue_type is not None and
                            cvalue_type != value_type):
                        raise galsim.GalSimConfigError(
                            "Attempt to parse %s multiple times with different value types: "
                            "%s and %s"%(key, value_type, cvalue_type))
                    #print(index,'Using current value of ',key,' = ',param['current'][0])
                    return cval, csafe
        else:
            # Only need to check this the first time.
            if 'type' not in param:
                raise galsim.GalSimConfigError(
                    "%s.type attribute required when providing a dict."%(key))

            # Check if the value_type is valid.
            # (See valid_value_types defined at the top of the file.)
            if type_name not in valid_value_types:
                raise galsim.GalSimConfigValueError("Unrecognized %s.type"%(key), type_name,
                                                    valid_value_types)

            # Get the generating function and the list of valid types for it.
            generate_func, valid_types = valid_value_types[type_name]

            if value_type not in valid_types:
                raise galsim.GalSimConfigValueError(
                    "Invalid value_type specified for parameter %s with type=%s."%(key, type_name),
                    value_type, valid_types)

            param['_gen_fn'] = generate_func

        #print('generate_func = ',generate_func)
        val_safe = generate_func(param, base, value_type)
        #print('returned val, safe = ',val_safe)
        if isinstance(val_safe, tuple):
            val, safe = val_safe
        else:  # pragma: no cover
            # If a user-defined type forgot to return safe, just assume safe = False
            # It's an easy mistake to make and the TypeError that gets emitted isn't
            # terribly informative about what the error is.
            val = val_safe
            safe = False

        # Make sure we really got the right type back.  (Just in case...)
        if value_type is not None and not isinstance(val,value_type) and val is not None:
            val = value_type(val)

        # Save the current value for possible use by the Current type
        param['current'] = (val, safe, value_type, index, index_key)
        #print(key,' = ',val)

        return val, safe

    else: # Not a dict

        # Check for some special markup on string items and convert them to normal dicts.
        if isinstance(param, basestring):
            if param[0] == '$':
                config[key] = { 'type': 'Eval', 'str': str(param[1:]) }
                return ParseValue(config, key, base, value_type)
            if param[0] == '@':
                config[key] = { 'type': 'Current', 'key': str(param[1:]) }
                return ParseValue(config, key, base, value_type)

        # See if it's already the right kind of object, in which case we can just return it.
        if value_type is None or isinstance(param, value_type):
            #print(key,' = ',param)
            return param, True

        # Convert lists to dicts with type=List
        if isinstance(param, list) and value_type is not list:
            config[key] = { 'type': 'List', 'items': param }
            return ParseValue(config, key, base, value_type)

        # The rest of these are special processing options for specific value_types:
        if value_type is galsim.Angle:
            # Angle is a special case.  Angles are specified with a final string to
            # declare what unit to use.
            val = _GetAngleValue(param)
        elif value_type is bool:
            # For bool, we allow a few special string conversions
            val = _GetBoolValue(param)
        elif value_type is galsim.PositionD:
            # For PositionD, we allow a string of x,y
            val = _GetPositionValue(param)
        elif value_type is None or param is None:
            # If no value_type is given, just return whatever we have in the dict and hope
            # for the best.
            val = param
        else:
            # If none of the above worked, just try a normal value_type initialization.
            # This makes sure strings are converted to float (or other type) if necessary.
            # In particular things like 1.e6 aren't converted to float automatically
            # by the yaml reader. (Although I think this is a bug.)
            try:
                val = value_type(param)
            except (ValueError, TypeError):
                raise galsim.GalSimConfigError("Could not parse %s as a %s"%(param, value_type))
        #print(key,' = ',val)

        # Save the converted type for next time so it will hit the first if statement here
        # instead of recalculating the value.
        config[key] = val
        return val, True


def GetCurrentValue(key, config, value_type=None, base=None):
    """@brief Get the current value of another config item given the key name.

    @param key          The (extended) key value in the dict to get the current value of.
    @param config       The config dict from which to get the key.
    @param value_type   The value_type expected.  [default: None, which means it won't check
                        that the value is the right type.]
    @param base         The base config dict.  [default: None, which means use base=config]

    @returns the current value
    """
    #print('GetCurrent %s.  value_type = %s'%(key,value_type))
    if base is None:
        base = config

    if '.' in key:
        config, key = galsim.config.ParseExtendedKey(config, key)

    val, safe = EvaluateCurrentValue(key, config, base, value_type)
    return val

def EvaluateCurrentValue(key, config, base, value_type=None):
    """Helper function to evaluate the current value at config[key] where key is no longer
    an extended key, and config is the local dict where it is relevant.

    @param key          The key value in the dict to get the current value of.
    @param config       The config dict from which to get the key.
    @param base         The base config dict.
    @param value_type   The value_type expected.  [default: None, which means it won't check
                        that the value is the right type.]
    """
    if not isinstance(config[key], dict):
        if value_type is not None or (isinstance(config[key],str) and config[key][0] in ('@','$')):
            # This will work fine to evaluate the current value, but will also
            # compute it if necessary
            #print('Not dict. Parse value normally')
            return ParseValue(config, key, base, value_type)
        else:
            # If we are not given the value_type, and it's not a dict, then the
            # item is probably just some value already.
            # (Unless it is a base item, in which case, it is not safe.)
            #print('Not dict, no value_type.  Assume %s is ok.'%d[k])
            return config[key], (config != base)
    else:
        if value_type is None and 'current' in config[key]:
            # If there is already a current val, use it.
            #print('Dict with current.  Use it: ',d[k]['current'][0])
            return config[key]['current'][:2]
        else:
            # Otherwise, parse the value for this key
            #print('Parse value normally')
            return ParseValue(config, key, base, value_type)

def SetDefaultIndex(config, num):
    """
    When the number of items in a list is known, we allow the user to omit some of
    the parameters of a Sequence or Random and set them automatically based on the
    size of the list, catalog, etc.
    """
    # We use a default item (set to True) to indicate that the value of nitems, last, or max
    # has been set here, rather than by the user.  This way if the number of items in the
    # catalog changes from one file to the next, it will be update correctly to the new
    # number of catalog entries.

    if 'index' not in config:
        config['index'] = {
            'type' : 'Sequence',
            'nitems' : num,
            'default' : num
        }
    elif isinstance(config['index'],dict) and 'type' in config['index']:
        index = config['index']

        if index.get('default',-1) == num: return
        if '_get' in index: del index['_get']

        type_name = index['type']
        if type_name == 'Sequence' and 'nitems' in index and 'default' in index:
            index['nitems'] = num
            index['default'] = num
        elif (type_name == 'Sequence'
              and 'nitems' not in index
              and index.get('step',1) > 0
              and ('last' not in index or 'default' in index) ):
            index['last'] = num-1
            index['default'] = num
        elif ( type_name == 'Sequence'
               and 'nitems' not in index
               and index.get('step',1) < 0
               and ('last' not in index or 'default' in index) ):
            index['last'] = 0
            index['default'] = num
        elif ( type_name == 'Random'
               and ('min' not in index or 'default' in index)
               and ('max' not in index or 'default' in index) ):
            index['min'] = 0
            index['max'] = num-1
            index['default'] = num
    if 'index_key' in config:
        config['index']['index_key'] = config['index_key']


def CheckAllParams(config, req={}, opt={}, single=[], ignore=[]):
    """@brief Check that the parameters for a particular item are all valid

    @returns a dict, get, with get[key] = value_type for all keys to get.
    """
    if '_get' in config: return config['_get']

    get = {}
    valid_keys = list(req) + list(opt)
    # Check required items:
    for (key, value_type) in req.items():
        if key in config:
            get[key] = value_type
        else:
            raise galsim.GalSimConfigError(
                "Attribute %s is required for type = %s"%(key,config.get('type',None)))

    # Check optional items:
    for (key, value_type) in opt.items():
        if key in config:
            get[key] = value_type

    # Check items for which exacly 1 should be defined:
    for s in single:
        valid_keys += list(s)
        count = 0
        for (key, value_type) in s.items():
            if key in config:
                count += 1
                if count > 1:
                    raise galsim.GalSimConfigError(
                        "Only one of the attributes %s is allowed for type = %s"%(
                            s.keys(),config.get('type',None)))
                get[key] = value_type
        if count == 0:
            raise galsim.GalSimConfigError(
                "One of the attributes %s is required for type = %s"%(
                    s.keys(),config.get('type',None)))

    # Check that there aren't any extra keys in config aside from a few we expect:
    valid_keys += ignore
    valid_keys += standard_ignore
    for key in config:
        # Generators are allowed to use item names that start with _, which we ignore here.
        if key not in valid_keys and not key.startswith('_'):
            raise galsim.GalSimConfigError("Unexpected attribute %s found"%(key))

    config['_get'] = get
    return get


def GetAllParams(config, base, req={}, opt={}, single=[], ignore=[]):
    """@brief Check and get all the parameters for a particular item

    @returns the tuple (kwargs, safe).
    """
    get = CheckAllParams(config,req,opt,single,ignore)
    kwargs = {}
    safe = True
    for (key, value_type) in sorted(get.items()):
        val, safe1 = ParseValue(config, key, base, value_type)
        safe = safe and safe1
        kwargs[key] = val
    return kwargs, safe




#
# Now the functions for directly converting an item in the config dict into a value.
# The ones that need a special function are: Angle, PositionD, and bool.
#

def _GetAngleValue(param):
    """ @brief Convert a string consisting of a value and an angle unit into an Angle.
    """
    try :
        value, unit = param.rsplit(None,1)
        value = float(value)
        unit = galsim.AngleUnit.from_name(unit)
        return galsim.Angle(value, unit)
    except (ValueError, TypeError, AttributeError) as e:
        raise galsim.GalSimConfigError("Unable to parse %s as an Angle. Caught %s"%(param,e))


def _GetPositionValue(param):
    """ @brief Convert a tuple or a string that looks like "a,b" into a galsim.PositionD.
    """
    try:
        x = float(param[0])
        y = float(param[1])
    except (ValueError, TypeError):
        try:
            x, y = param.split(',')
            x = float(x.strip())
            y = float(y.strip())
        except (ValueError, TypeError, AttributeError) as e:
            raise galsim.GalSimConfigError(
                "Unable to parse %s as a PositionD. Caught %s"%(param,e))
    return galsim.PositionD(x,y)


def _GetBoolValue(param):
    """ @brief Convert a string to a bool
    """
    if isinstance(param,str):
        if param.strip().upper() in ('TRUE', 'YES'):
            return True
        elif param.strip().upper() in ('FALSE', 'NO'):
            return False
        else:
            try:
                val = bool(int(param))
                return val
            except (ValueError, TypeError, AttributeError) as e:
                raise galsim.GalSimConfigError(
                    "Unable to parse %s as a bool. Caught %s"%(param,e))
    else:
        # This always works.
        # Everything in Python is convertible to bool.
        return bool(param)


#
# Now all the GenerateFrom functions:
#

def _GenerateFromG1G2(config, base, value_type):
    """@brief Return a Shear constructed from given (g1, g2)
    """
    req = { 'g1' : float, 'g2' : float }
    kwargs, safe = GetAllParams(config, base, req=req)
    #print(base['obj_num'],'Generate from G1G2: kwargs = ',kwargs)
    return galsim.Shear(**kwargs), safe

def _GenerateFromE1E2(config, base, value_type):
    """@brief Return a Shear constructed from given (e1, e2)
    """
    req = { 'e1' : float, 'e2' : float }
    kwargs, safe = GetAllParams(config, base, req=req)
    #print(base['obj_num'],'Generate from E1E2: kwargs = ',kwargs)
    return galsim.Shear(**kwargs), safe

def _GenerateFromEta1Eta2(config, base, value_type):
    """@brief Return a Shear constructed from given (eta1, eta2)
    """
    req = { 'eta1' : float, 'eta2' : float }
    kwargs, safe = GetAllParams(config, base, req=req)
    #print(base['obj_num'],'Generate from Eta1Eta2: kwargs = ',kwargs)
    return galsim.Shear(**kwargs), safe

def _GenerateFromGBeta(config, base, value_type):
    """@brief Return a Shear constructed from given (g, beta)
    """
    req = { 'g' : float, 'beta' : galsim.Angle }
    kwargs, safe = GetAllParams(config, base, req=req)
    #print(base['obj_num'],'Generate from GBeta: kwargs = ',kwargs)
    return galsim.Shear(**kwargs), safe

def _GenerateFromEBeta(config, base, value_type):
    """@brief Return a Shear constructed from given (e, beta)
    """
    req = { 'e' : float, 'beta' : galsim.Angle }
    kwargs, safe = GetAllParams(config, base, req=req)
    #print(base['obj_num'],'Generate from EBeta: kwargs = ',kwargs)
    return galsim.Shear(**kwargs), safe

def _GenerateFromEtaBeta(config, base, value_type):
    """@brief Return a Shear constructed from given (eta, beta)
    """
    req = { 'eta' : float, 'beta' : galsim.Angle }
    kwargs, safe = GetAllParams(config, base, req=req)
    #print(base['obj_num'],'Generate from EtaBeta: kwargs = ',kwargs)
    return galsim.Shear(**kwargs), safe

def _GenerateFromQBeta(config, base, value_type):
    """@brief Return a Shear constructed from given (q, beta)
    """
    req = { 'q' : float, 'beta' : galsim.Angle }
    kwargs, safe = GetAllParams(config, base, req=req)
    #print(base['obj_num'],'Generate from QBeta: kwargs = ',kwargs)
    return galsim.Shear(**kwargs), safe

def _GenerateFromXY(config, base, value_type):
    """@brief Return a PositionD constructed from given (x,y)
    """
    req = { 'x' : float, 'y' : float }
    kwargs, safe = GetAllParams(config, base, req=req)
    #print(base['obj_num'],'Generate from XY: kwargs = ',kwargs)
    return galsim.PositionD(**kwargs), safe

def _GenerateFromRTheta(config, base, value_type):
    """@brief Return a PositionD constructed from given (r,theta)
    """
    req = { 'r' : float, 'theta' : galsim.Angle }
    kwargs, safe = GetAllParams(config, base, req=req)
    r = kwargs['r']
    theta = kwargs['theta']
    import math
    #print(base['obj_num'],'Generate from RTheta: kwargs = ',kwargs)
    return galsim.PositionD(r*theta.cos(), r*theta.sin()), safe

def _GenerateFromRADec(config, base, value_type):
    """@brief Return a CelestialCoord constructed from given (ra,dec)
    """
    req = { 'ra' : galsim.Angle, 'dec' : galsim.Angle }
    kwargs, safe = GetAllParams(config, base, req=req)
    #print(base['obj_num'],'Generate from RADec: kwargs = ',kwargs)
    return galsim.CelestialCoord(**kwargs), safe

def _GenerateFromRad(config, base, value_type):
    """@brief Return an Angle constructed from given theta in radians
    """
    req = { 'theta' : float }
    kwargs, safe = GetAllParams(config, base, req=req)
    #print(base['obj_num'],'Generate from Rad: kwargs = ',kwargs)
    return kwargs['theta'] * galsim.radians, safe

def _GenerateFromDeg(config, base, value_type):
    """@brief Return an Angle constructed from given theta in degrees
    """
    req = { 'theta' : float }
    kwargs, safe = GetAllParams(config, base, req=req)
    #print(base['obj_num'],'Generate from Deg: kwargs = ',kwargs)
    return kwargs['theta'] * galsim.degrees, safe

def _GenerateFromSequence(config, base, value_type):
    """@brief Return next in a sequence of integers
    """
    ignore = [ 'default' ]
    opt = { 'first' : value_type, 'last' : value_type, 'step' : value_type,
            'repeat' : int, 'nitems' : int, 'index_key' : str }
    kwargs, safe = GetAllParams(config, base, opt=opt, ignore=ignore)

    step = kwargs.get('step',1)
    first = kwargs.get('first',0)
    repeat = kwargs.get('repeat',1)
    last = kwargs.get('last',None)
    nitems = kwargs.get('nitems',None)

    if repeat <= 0:
        raise galsim.GalSimConfigValueError(
            "Invalid repeat for type = Sequence (must be > 0)", repeat)
    if last is not None and nitems is not None:
        raise galsim.GalSimConfigError(
            "At most one of the attributes last and nitems is allowed for type = Sequence")

    index, index_key = galsim.config.GetIndex(kwargs, base, is_sequence=True)
    #print('in GenFromSequence: index = ',index,index_key)

    if value_type is bool:
        # Then there are only really two valid sequences: Either 010101... or 101010...
        # Aside from the repeat value of course.
        if first:
            first = 1
            step = -1
            nitems = 2
        else:
            first = 0
            step = 1
            nitems = 2

    elif value_type is float:
        if last is not None:
            nitems = int( (last-first)/step + 0.5 ) + 1
    else:
        if last is not None:
            nitems = (last - first)//step + 1
    #print('nitems = ',nitems)
    #print('repeat = ',repeat)

    index = index // repeat
    #print('index => ',index)

    if nitems is not None and nitems > 0:
        index = index % nitems
        #print('index => ',index)

    value = first + index*step
    #print(base[index_key],'Sequence index = %s + %d*%s = %s'%(first,index,step,value))
    return value, False


def _GenerateFromNumberedFile(config, base, value_type):
    """@brief Return a file_name using a root, a number, and an extension
    """
    if 'num' not in config:
        config['num'] = { 'type' : 'Sequence' }
    req = { 'root' : str , 'num' : int }
    opt = { 'ext' : str , 'digits' : int }
    kwargs, safe = GetAllParams(config, base, req=req, opt=opt)

    template = kwargs['root']
    if 'digits' in kwargs:
        template += '%%0%dd'%kwargs['digits']
    else:
        template += '%d'
    if 'ext' in kwargs:
        template += kwargs['ext']
    s = eval("'%s'%%%d"%(template,kwargs['num']))
    #print(base['obj_num'],'NumberedFile = ',s)
    return s, safe

def _GenerateFromFormattedStr(config, base, value_type):
    """@brief Create a string from a format string
    """
    req = { 'format' : str, 'items' : list }
    # Ignore items for now, we'll deal with it differently.
    params, safe = GetAllParams(config, base, req=req)
    format = params['format']
    items = params['items']

    # Figure out what types we are expecting for the list elements:
    tokens = format.split('%')
    val_types = []
    skip = False
    for token in tokens[1:]:  # skip first one.
        # It we have set skip, then skip this one.
        if skip:
            skip = False
            continue
        # If token == '', then this is a %% in the original string.  Skip this and the next token.
        if len(token) == 0:
            skip = True
            continue
        token = token.lstrip('0123456789lLh') # ignore field size, and long/short specification
        if len(token) == 0:
            raise galsim.GalSimConfigError("Unable to parse %r as a valid format string"%format)
        if token[0].lower() in 'diouxX':
            val_types.append(int)
        elif token[0].lower() in 'eEfFgG':
            val_types.append(float)
        elif token[0].lower() in 'rs':
            val_types.append(str)
        else:
            raise galsim.GalSimConfigError("Unable to parse %r as a valid format string"%format)

    if len(val_types) != len(items):
        raise galsim.GalSimConfigError(
            "Number of items for FormatStr (%d) does not match number expected from "
            "format string (%d)"%(len(items), len(val_types)))
    vals = []
    for index in range(len(items)):
        val, safe1 = ParseValue(items, index, base, val_types[index])
        safe = safe and safe1
        vals.append(val)

    final_str = format%tuple(vals)
    #print(base['obj_num'],'FormattedStr = ',final_str)
    return final_str, safe


def _GenerateFromList(config, base, value_type):
    """@brief Return next item from a provided list
    """
    req = { 'items' : list }
    opt = { 'index' : int }
    # Only Check, not Get.  We need to handle items a bit differently, since it's a list.
    CheckAllParams(config, req=req, opt=opt)
    items = config['items']
    if not isinstance(items,list):
        raise galsim.GalSimConfigError("items entry for type=List is not a list.")

    # Setup the indexing sequence if it hasn't been specified using the length of items.
    SetDefaultIndex(config, len(items))
    index, safe = ParseValue(config, 'index', base, int)

    if index < 0 or index >= len(items):
        raise galsim.GalSimConfigError("index %d out of bounds for type=List"%index)
    val, safe1 = ParseValue(items, index, base, value_type)
    safe = safe and safe1
    #print(base['obj_num'],'List index = %d, val = %s'%(index,val))
    return val, safe

def _GenerateFromSum(config, base, value_type):
    """@brief Return next item from a provided list
    """
    req = { 'items' : list }
    # Only Check, not Get.  We need to handle items a bit differently, since it's a list.
    CheckAllParams(config, req=req)
    items = config['items']
    if not isinstance(items,list):
        raise galsim.GalSimConfigError("items entry for type=List is not a list.")

    sum, safe = ParseValue(items, 0, base, value_type)

    for k in range(1,len(items)):
        val, safe1 = ParseValue(items, k, base, value_type)
        sum += val
        safe = safe and safe1

    return sum, safe

def _GenerateFromCurrent(config, base, value_type):
    """@brief Get the current value of another config item.
    """
    if '_kd' in config:
        k, d = config['_kd']
    else:
        req = { 'key' : str }
        params, safe = GetAllParams(config, base, req=req, ignore=['_kd'])
        key = params['key']
        #print('GetCurrent %s.  value_type = %s'%(key,value_type))

        d, k = galsim.config.ParseExtendedKey(base, key)
        config['_kd'] = k,d

    try:
        return EvaluateCurrentValue(k, d, base, value_type)
    except (TypeError, ValueError) as e:
        raise galsim.GalSimConfigError("%s\nError generating Current value with key = %s"%(e,k))


def RegisterValueType(type_name, gen_func, valid_types, input_type=None):
    """Register a value type for use by the config apparatus.

    A few notes about the signature of the generating function:

    1. The config parameter is the dict for the current value to be generated.  So it should
       be the case that config['type'] == type_name.
    2. The base parameter is the original config dict being processed.
    3. The value_type parameter is the intended type of the generated value.  It should
       be one of the values that you specify as valid in valid_types.
    4. The return value of gen_func should be a tuple consisting of the value and a boolean,
       safe, which indicates whether the generated value is safe to use again rather than
       regenerate for subsequent objects.  This will be used upstream to determine if
       objects constructed using this value are safe to keep or if they have to be rebuilt.

    The allowed types to include in valid_types are: float, int, bool, str, galsim.Angle,
    galsim.Shear, galsim.PositionD.  In addition, including None in this list means that
    it is valid to use this type if you don't necessarily know what type you are expecting.
    This happens when building a truth catalog where each item should already be generated
    and the current value and type stored, so currently the only two types that allow
    None as a valid type are Current and Eval.

    @param type_name        The name of the 'type' specification in the config dict.
    @param gen_func         A function to generate a value from the config information.
                            The call signature is
                                value, safe = Generate(config, base, value_type)
    @param valid_types      A list of types for which this type name is valid.
    @param input_type       If the generator utilises an input object, give the key name of the
                            input type here.  (If it uses more than one, this may be a list.)
                            [default: None]
    """
    valid_value_types[type_name] = (gen_func, tuple(valid_types))
    if input_type is not None:
        from .input import RegisterInputConnectedType
        if isinstance(input_type, list): # pragma: no cover
            for key in input_type:
                RegisterInputConnectedType(key, type_name)
        else:
            RegisterInputConnectedType(input_type, type_name)


RegisterValueType('List', _GenerateFromList,
              [ float, int, bool, str, galsim.Angle, galsim.Shear, galsim.PositionD,
                galsim.CelestialCoord ])
RegisterValueType('Current', _GenerateFromCurrent,
                 [ float, int, bool, str, galsim.Angle, galsim.Shear, galsim.PositionD,
                   galsim.CelestialCoord, None ])
RegisterValueType('Sum', _GenerateFromSum,
             [ float, int, galsim.Angle, galsim.Shear, galsim.PositionD ])
RegisterValueType('Sequence', _GenerateFromSequence, [ float, int, bool ])
RegisterValueType('NumberedFile', _GenerateFromNumberedFile, [ str ])
RegisterValueType('FormattedStr', _GenerateFromFormattedStr, [ str ])
RegisterValueType('Rad', _GenerateFromRad, [ galsim.Angle ])
RegisterValueType('Radians', _GenerateFromRad, [ galsim.Angle ])
RegisterValueType('Deg', _GenerateFromDeg, [ galsim.Angle ])
RegisterValueType('Degrees', _GenerateFromDeg, [ galsim.Angle ])
RegisterValueType('E1E2', _GenerateFromE1E2, [ galsim.Shear ])
RegisterValueType('EBeta', _GenerateFromEBeta, [ galsim.Shear ])
RegisterValueType('G1G2', _GenerateFromG1G2, [ galsim.Shear ])
RegisterValueType('GBeta', _GenerateFromGBeta, [ galsim.Shear ])
RegisterValueType('Eta1Eta2', _GenerateFromEta1Eta2, [ galsim.Shear ])
RegisterValueType('EtaBeta', _GenerateFromEtaBeta, [ galsim.Shear ])
RegisterValueType('QBeta', _GenerateFromQBeta, [ galsim.Shear ])
RegisterValueType('XY', _GenerateFromXY, [ galsim.PositionD ])
RegisterValueType('RTheta', _GenerateFromRTheta, [ galsim.PositionD ])
RegisterValueType('RADec', _GenerateFromRADec, [ galsim.CelestialCoord ])
