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

from .value_eval import _GenerateFromEval
from .value_random import _GenerateFromRandom, _GenerateFromRandomGaussian, _GenerateFromRandomPoisson, _GenerateFromRandomBinomial, _GenerateFromRandomWeibull, _GenerateFromRandomGamma, _GenerateFromRandomChi2, _GenerateFromRandomDistribution, _GenerateFromRandomCircle
from .input import _GenerateFromCatalog, _GenerateFromDict
from .input_fitsheader import _GenerateFromFitsHeader
from .input_powerspectrum import _GenerateFromPowerSpectrumShear, _GenerateFromPowerSpectrumMagnification
from .input_nfw import _GenerateFromNFWHaloShear, _GenerateFromNFWHaloMagnification

valid_value_types = {
    # The values are tuples with:
    # - the build function to call
    # - a list of types for which the type is valid
    'List' : ('_GenerateFromList', 
              [ float, int, bool, str, galsim.Angle, galsim.Shear, galsim.PositionD ]),
    'Eval' : ('_GenerateFromEval', 
              [ float, int, bool, str, galsim.Angle, galsim.Shear, galsim.PositionD ]),
    'Current' : ('_GenerateFromCurrent', 
                 [ float, int, bool, str, galsim.Angle, galsim.Shear, galsim.PositionD ]),
    'Sum' : ('_GenerateFromSum', 
             [ float, int, galsim.Angle, galsim.Shear, galsim.PositionD ]),
    'Catalog' : ('_GenerateFromCatalog', [ float, int, bool, str ]),
    'Dict' : ('_GenerateFromDict', [ float, int, bool, str ]),
    'FitsHeader' : ('_GenerateFromFitsHeader', [ float, int, bool, str ]),
    'Sequence' : ('_GenerateFromSequence', [ float, int, bool ]),
    'Random' : ('_GenerateFromRandom', [ float, int, bool, galsim.Angle ]),
    'RandomGaussian' : ('_GenerateFromRandomGaussian', [ float ]),
    'RandomPoisson' : ('_GenerateFromRandomPoisson', [ float, int ]),
    'RandomBinomial' : ('_GenerateFromRandomBinomial', [ float, int, bool ]),
    'RandomWeibull' : ('_GenerateFromRandomWeibull', [ float ]),
    'RandomGamma' : ('_GenerateFromRandomGamma', [ float ]),
    'RandomChi2' : ('_GenerateFromRandomChi2', [ float ]),
    'RandomDistribution' : ('_GenerateFromRandomDistribution', [ float ]),
    'RandomCircle' : ('_GenerateFromRandomCircle', [ galsim.PositionD ]),
    'NumberedFile' : ('_GenerateFromNumberedFile', [ str ]),
    'FormattedStr' : ('_GenerateFromFormattedStr', [ str ]),
    'Rad' : ('_GenerateFromRad', [ galsim.Angle ]),
    'Radians' : ('_GenerateFromRad', [ galsim.Angle ]),
    'Deg' : ('_GenerateFromDeg', [ galsim.Angle ]),
    'Degrees' : ('_GenerateFromDeg', [ galsim.Angle ]),
    'E1E2' : ('_GenerateFromE1E2', [ galsim.Shear ]),
    'EBeta' : ('_GenerateFromEBeta', [ galsim.Shear ]),
    'G1G2' : ('_GenerateFromG1G2', [ galsim.Shear ]),
    'GBeta' : ('_GenerateFromGBeta', [ galsim.Shear ]),
    'Eta1Eta2' : ('_GenerateFromEta1Eta2', [ galsim.Shear ]),
    'EtaBeta' : ('_GenerateFromEtaBeta', [ galsim.Shear ]),
    'QBeta' : ('_GenerateFromQBeta', [ galsim.Shear ]),
    'XY' : ('_GenerateFromXY', [ galsim.PositionD ]),
    'RTheta' : ('_GenerateFromRTheta', [ galsim.PositionD ]),
    'NFWHaloShear' : ('_GenerateFromNFWHaloShear', [ galsim.Shear ]),
    'NFWHaloMagnification' : ('_GenerateFromNFWHaloMagnification', [ float ]),
    'PowerSpectrumShear' : ('_GenerateFromPowerSpectrumShear', [ galsim.Shear ]),
    'PowerSpectrumMagnification' : ('_GenerateFromPowerSpectrumMagnification', [ float ]),
}
 
# Standard keys to ignore while parsing values:
standard_ignore = [ 
    'type',
    'current_val', 'current_safe', 'current_value_type',
    'current_obj_num', 'current_image_num', 'current_file_num',
    '#' # When we read in json files, there represent comments
]

def ParseValue(config, key, base, value_type):
    """@brief Read or generate a parameter value from config.

    @returns the tuple (value, safe).
    """
    param = config[key]
    #print 'ParseValue for param_name = ',key,', value_type = ',str(value_type)
    #print 'param = ',param
    #print 'nums = ',base.get('file_num',0), base.get('image_num',0), base.get('obj_num',0)

    # First see if we can assign by param by a direct constant value
    if isinstance(param, value_type):
        #print key,' = ',param
        return param, True
    elif not isinstance(param, dict):
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
        else:
            # Make sure strings are converted to float (or other type) if necessary.
            # In particular things like 1.e6 aren't converted to float automatically
            # by the yaml reader. (Although I think this is a bug.)
            val = value_type(param)
        # Save the converted type for next time.
        config[key] = val
        #print key,' = ',val
        return val, True
    elif 'type' not in param:
        raise AttributeError(
            "%s.type attribute required in config for non-constant parameter %s."%(key,key))
    elif ( 'current_val' in param 
           and param['current_obj_num'] == base.get('obj_num',0)
           and param['current_image_num'] == base.get('image_num',0)
           and param['current_file_num'] == base.get('file_num',0) ):
        if param['current_value_type'] != value_type:
            raise ValueError(
                "Attempt to parse %s multiple times with different value types"%key)
        #print base['obj_num'],'Using current value of ',key,' = ',param['current_val']
        return param['current_val'], param['current_safe']
    else:
        # Otherwise, we need to generate the value according to its type
        # (See valid_value_types defined at the top of the file.)

        type_name = param['type']
        #print 'type = ',type_name
        #print param['type'], value_type

        # First check if the value_type is valid.
        if type_name not in valid_value_types:
            raise AttributeError(
                "Unrecognized type = %s specified for parameter %s"%(type_name,key))
            
        if value_type not in valid_value_types[type_name][1]:
            raise AttributeError(
                "Invalid value_type = %s specified for parameter %s with type = %s."%(
                    value_type, key, type_name))

        generate_func = eval(valid_value_types[type_name][0])
        #print 'generate_func = ',generate_func
        val, safe = generate_func(param, base, value_type)
        #print 'returned val, safe = ',val,safe

        # Make sure we really got the right type back.  (Just in case...)
        if not isinstance(val,value_type):
            val = value_type(val)

        # Save the current value for possible use by the Current type
        param['current_val'] = val
        param['current_safe'] = safe
        param['current_value_type'] = value_type
        param['current_obj_num'] = base.get('obj_num',0)
        param['current_image_num'] = base.get('image_num',0)
        param['current_file_num'] = base.get('file_num',0)
        #print key,' = ',val
        return val, safe


def GetCurrentValue(config, key):
    """@brief Return the current value of a parameter (either stored or a simple value)
    """
    param = config[key]
    if isinstance(param, dict):
        return param['current_val']
    else: 
        return param

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
            'default' : True,
        }
    elif ( isinstance(config['index'],dict) 
           and 'type' in config['index'] ):
        index = config['index']
        type_name = index['type']
        if ( type_name == 'Sequence' 
             and 'nitems' in index 
             and 'default' in index ):
            index['nitems'] = num
            index['default'] = True
        elif ( type_name == 'Sequence' 
               and 'nitems' not in index
               and ('step' not in index or (isinstance(index['step'],int) and index['step'] > 0) )
               and ('last' not in index or 'default' in index) ):
            index['last'] = num-1
            index['default'] = True
        elif ( type_name == 'Sequence'
               and 'nitems' not in index
               and ('step' in index and (isinstance(index['step'],int) and index['step'] < 0) ) ):
            # Normally, the value of default doesn't matter.  Its presence is sufficient
            # to indicate True.  However, here we have three options.  
            # 1) first and last are both set by default
            # 2) first (only) is set by default
            # 3) last (only) is set by default
            # So set default to the option we are using, so we update with the correct method.
            if ( ('first' not in index and 'last' not in index)
                 or ('default' in index and index['default'] == 1) ):
                index['first'] = num-1
                index['last'] = 0
                index['default'] = 1
            elif ( 'first' not in index 
                   or ('default' in index and index['default'] == 2) ):
                index['first'] = num-1
                index['default'] = 2
            elif ( 'last' not in index 
                   or ('default' in index and index['default'] == 3) ):
                index['last'] = 0
                index['default'] = 3
        elif ( type_name == 'Random'
               and ('min' not in index or 'default' in index)
               and ('max' not in index or 'default' in index) ):
            index['min'] = 0
            index['max'] = num-1
            index['default'] = True


def CheckAllParams(config, req={}, opt={}, single=[], ignore=[]):
    """@brief Check that the parameters for a particular item are all valid
    
    @returns a dict, get, with get[key] = value_type for all keys to get.
    """
    get = {}
    valid_keys = req.keys() + opt.keys()
    # Check required items:
    for (key, value_type) in req.items():
        if key in config:
            get[key] = value_type
        else:
            raise AttributeError(
                "Attribute %s is required for %s"%(key,config['type']))

    # Check optional items:
    for (key, value_type) in opt.items():
        if key in config:
            get[key] = value_type

    # Check items for which exacly 1 should be defined:
    for s in single: 
        if not s: # If no items in list, don't require one of them to be present.
            break
        valid_keys += s.keys()
        count = 0
        for (key, value_type) in s.items():
            if key in config:
                count += 1
                if count > 1:
                    raise AttributeError(
                        "Only one of the attributes %s is allowed for %s"%(s.keys(),config['type']))
                get[key] = value_type
        if count == 0:
            raise AttributeError(
                "One of the attributes %s is required for %s"%(s.keys(),config['type']))

    # Check that there aren't any extra keys in param aside from a few we expect:
    valid_keys += ignore
    valid_keys += standard_ignore
    for key in config.keys():
        # Generators are allowed to use item names that start with _, which we ignore here.
        if key not in valid_keys and not key.startswith('_'):
            raise AttributeError("Unexpected attribute %s found"%key)

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
    # Just in case there are unicode strings.   python 2.6 doesn't like them in kwargs.
    kwargs = dict([(k.encode('utf-8'), v) for k,v in kwargs.iteritems()])
    return kwargs, safe



def _GetAngleValue(param):
    """ @brief Convert a string consisting of a value and an angle unit into an Angle.
    """
    try :
        value, unit = param.rsplit(None,1)
        value = float(value)
        unit = galsim.angle.get_angle_unit(unit)
        return galsim.Angle(value, unit)
    except Exception as e:
        raise AttributeError("Unable to parse %s as an Angle."%param)


def _GetPositionValue(param):
    """ @brief Convert a tuple or a string that looks like "a,b" into a galsim.PositionD.
    """
    try:
        x = float(param[0])
        y = float(param[1])
    except:
        try:
            x, y = param.split(',')
            x = float(x.strip())
            y = float(y.strip())
        except:
            raise AttributeError("Unable to parse %s as a PositionD."%param)
    return galsim.PositionD(x,y)


def _GetBoolValue(param):
    """ @brief Convert a string to a bool
    """
    if isinstance(param,str):
        if param.strip().upper() in [ 'TRUE', 'YES' ]:
            return True
        elif param.strip().upper() in [ 'FALSE', 'NO' ]:
            return False
        else:
            try:
                val = bool(int(param))
                return val
            except:
                raise AttributeError("Unable to parse %s as a bool."%param)
    else:
        try:
            val = bool(param)
            return val
        except:
            raise AttributeError("Unable to parse %s as a bool."%param)



#
# Now all the GenerateFrom functions:
#

def _GenerateFromG1G2(config, base, value_type):
    """@brief Return a Shear constructed from given (g1, g2)
    """
    req = { 'g1' : float, 'g2' : float }
    kwargs, safe = GetAllParams(config, base, req=req)
    #print base['obj_num'],'Generate from G1G2: kwargs = ',kwargs
    return galsim.Shear(**kwargs), safe

def _GenerateFromE1E2(config, base, value_type):
    """@brief Return a Shear constructed from given (e1, e2)
    """
    req = { 'e1' : float, 'e2' : float }
    kwargs, safe = GetAllParams(config, base, req=req)
    #print base['obj_num'],'Generate from E1E2: kwargs = ',kwargs
    return galsim.Shear(**kwargs), safe

def _GenerateFromEta1Eta2(config, base, value_type):
    """@brief Return a Shear constructed from given (eta1, eta2)
    """
    req = { 'eta1' : float, 'eta2' : float }
    kwargs, safe = GetAllParams(config, base, req=req)
    #print base['obj_num'],'Generate from Eta1Eta2: kwargs = ',kwargs
    return galsim.Shear(**kwargs), safe

def _GenerateFromGBeta(config, base, value_type):
    """@brief Return a Shear constructed from given (g, beta)
    """
    req = { 'g' : float, 'beta' : galsim.Angle }
    kwargs, safe = GetAllParams(config, base, req=req)
    #print base['obj_num'],'Generate from GBeta: kwargs = ',kwargs
    return galsim.Shear(**kwargs), safe

def _GenerateFromEBeta(config, base, value_type):
    """@brief Return a Shear constructed from given (e, beta)
    """
    req = { 'e' : float, 'beta' : galsim.Angle }
    kwargs, safe = GetAllParams(config, base, req=req)
    #print base['obj_num'],'Generate from EBeta: kwargs = ',kwargs
    return galsim.Shear(**kwargs), safe

def _GenerateFromEtaBeta(config, base, value_type):
    """@brief Return a Shear constructed from given (eta, beta)
    """
    req = { 'eta' : float, 'beta' : galsim.Angle }
    kwargs, safe = GetAllParams(config, base, req=req)
    #print base['obj_num'],'Generate from EtaBeta: kwargs = ',kwargs
    return galsim.Shear(**kwargs), safe

def _GenerateFromQBeta(config, base, value_type):
    """@brief Return a Shear constructed from given (q, beta)
    """
    req = { 'q' : float, 'beta' : galsim.Angle }
    kwargs, safe = GetAllParams(config, base, req=req)
    #print base['obj_num'],'Generate from QBeta: kwargs = ',kwargs
    return galsim.Shear(**kwargs), safe

def _GenerateFromXY(config, base, value_type):
    """@brief Return a PositionD constructed from given (x,y)
    """
    req = { 'x' : float, 'y' : float }
    kwargs, safe = GetAllParams(config, base, req=req)
    #print base['obj_num'],'Generate from XY: kwargs = ',kwargs
    return galsim.PositionD(**kwargs), safe

def _GenerateFromRTheta(config, base, value_type):
    """@brief Return a PositionD constructed from given (r,theta)
    """
    req = { 'r' : float, 'theta' : galsim.Angle }
    kwargs, safe = GetAllParams(config, base, req=req)
    r = kwargs['r']
    theta = kwargs['theta']
    import math
    #print base['obj_num'],'Generate from RTheta: kwargs = ',kwargs
    return galsim.PositionD(r*math.cos(theta.rad()), r*math.sin(theta.rad())), safe

def _GenerateFromRad(config, base, value_type):
    """@brief Return an Angle constructed from given theta in radians
    """
    req = { 'theta' : float }
    kwargs, safe = GetAllParams(config, base, req=req)
    #print base['obj_num'],'Generate from Rad: kwargs = ',kwargs
    return kwargs['theta'] * galsim.radians, safe

def _GenerateFromDeg(config, base, value_type):
    """@brief Return an Angle constructed from given theta in degrees
    """
    req = { 'theta' : float }
    kwargs, safe = GetAllParams(config, base, req=req)
    #print base['obj_num'],'Generate from Deg: kwargs = ',kwargs
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
    index_key = kwargs.get('index_key',base.get('index_key','obj_num'))
    if repeat <= 0:
        raise ValueError(
            "Invalid repeat=%d (must be > 0) for Sequence"%repeat)
    if last is not None and nitems is not None:
        raise AttributeError(
            "At most one of the attributes last and nitems is allowed for Sequence")
    if index_key not in [ 'obj_num_in_file', 'obj_num', 'image_num', 'file_num' ]:
        raise AttributeError(
            "Invalid index=%s for Sequence."%index_key)

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
            nitems = (last - first)/step + 1

    if index_key == 'obj_num_in_file':
        index = base['obj_num'] - base.get('start_obj_num',0)
    else:
        index = base[index_key]
    index = index / repeat

    if nitems is not None and nitems > 0:
        index = index % nitems

    value = first + index*step
    #print base[index_key],'Sequence index = %s + %d*%s = %s'%(first,index,step,value)
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
    #print base['obj_num'],'NumberedFile = ',s
    return s, safe

def _GenerateFromFormattedStr(config, base, value_type):
    """@brief Create a string from a format string
    """
    req = { 'format' : str }
    # Ignore items for now, we'll deal with it differently.
    ignore = [ 'items' ]
    params, safe = GetAllParams(config, base, req=req, ignore=ignore)
    format = params['format']

    # Check that items is present and is a list.
    if 'items' not in config:
        raise AttributeError("Attribute items is required for type = FormattedStr")
    items = config['items']
    if not isinstance(items,list):
        raise AttributeError("items entry is not a list.")

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
            raise ValueError("Unable to parse '%s' as a valid format string"%format)
        if token[0].lower() in 'diouxX':
            val_types.append(int)
        elif token[0].lower() in 'eEfFgG':
            val_types.append(float)
        elif token[0].lower() in 'rs':
            val_types.append(str)
        else:
            raise ValueError("Unable to parse '%s' as a valid format string"%format)

    if len(val_types) != len(items):
        raise ValueError(
            "Number of items for FormatStr (%d) does not match number expected from "%len(items)+
            "format string (%d)"%len(val_types))
    vals = []
    for index in range(len(items)):
        val, safe1 = ParseValue(items, index, base, val_types[index])
        safe = safe and safe1
        vals.append(val)

    final_str = format%tuple(vals)
    #print base['obj_num'],'FormattedStr = ',final_str
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
        raise AttributeError("items entry is not a list.")

    # Setup the indexing sequence if it hasn't been specified using the length of items.
    SetDefaultIndex(config, len(items))
    index, safe = ParseValue(config, 'index', base, int)

    if index < 0 or index >= len(items):
        raise AttributeError("index %d out of bound"%index)
    val, safe1 = ParseValue(items, index, base, value_type)
    safe = safe and safe1
    #print base['obj_num'],'List index = %d, val = %s'%(index,val)
    return val, safe
 
def _GenerateFromSum(config, base, value_type):
    """@brief Return next item from a provided list
    """
    req = { 'items' : list }
    # Only Check, not Get.  We need to handle items a bit differently, since it's a list.
    CheckAllParams(config, req=req)
    items = config['items']
    if not isinstance(items,list):
        raise AttributeError("items entry is not a list.")

    sum, safe = ParseValue(items, 0, base, value_type)

    for k in range(1,len(items)):
        val, safe1 = ParseValue(items, k, base, value_type)
        sum += val
        safe = safe and safe1
        
    return sum, safe
 

def _GenerateFromCurrent(config, base, value_type):
    """@brief Get the current value of another config item.
    """
    req = { 'key' : str }
    params, safe = GetAllParams(config, base, req=req)

    key = params['key']

    # This next bit is basically identical to the code for Dict.get(key) in catalog.py.
    # Make a list of keys
    chain = key.split('.')
    d = base

    # We may need to make one adjustment.  If the first item in the key is 'input', then
    # the key is probably wrong relative to the current config dict.  We make each input
    # item a list, so the user can have more than one input dict for example.  But if 
    # they aren't using that, we don't want them to have to know about it if they try to 
    # take soemthing from there for a Current item.  
    # So we change, e.g., 
    #     input.fits_header.file_name 
    # --> input.fits_header.0.file_name
    if chain[0] == 'input' and len(chain) > 2:
        try:
            k = int(chain[2])
        except:
            chain.insert(2,0)

    while len(chain):
        k = chain.pop(0)

        # Try to convert to an integer:
        try: k = int(k)
        except ValueError: pass

        if chain: 
            # If there are more keys, just set d to the next in the chanin.
            d = d[k]
        else: 
            # Otherwise, parse the value for this key
            val,safe = ParseValue(d, k, base, value_type)
            #print base['obj_num'],'Current key = %s, value = %s'%(key,val)
            return val,safe

    raise ValueError("Invalid key = %s given for Current"%key)


