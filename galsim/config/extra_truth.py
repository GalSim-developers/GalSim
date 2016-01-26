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

# The truth extra output type builds an OutputCatalog with truth information about each of the
# objects being built by the configuration processing.  It stores the appropriate row information
# in scratch space for each stamp and then adds them in order at the end of the file processing.
# This means that the stamps can be built out of order by the multiprocessing and still show
# up in the correct order in the output catalog.

# Note that the order of the column names in the output catalog is taken from 
# config['output']['truth']['columns'].keys().  So if config is a regular dict, the order
# of the keys is semi-arbitrary.  However, if config is an OrderedDict, the keys come out
# in the order specified.  The standard galsim executable reads the config file into an 
# OrderedDict for precisely this reason.

# The function that returns the kwargs for constructing the OutputCatalog
def GetTruthKwargs(config, base, logger=None):
    if logger and not hasattr(config, '__reversed__'):
        # If config doesn't have a __reversed__ attribute, then it's not an OrderedDict.
        # Probably it's just a regular dict.  So warn the user that the columns are in 
        # arbitrary order.
        # (This was the simplest difference I could find between dict and OrderedDict that
        #  seemed relevant.)
        logger.warn('The config dict is not an OrderedDict.  The columns in the output truth '+
                    'catalog will be in arbitrary order.')
    columns = config['columns']
    truth_names = columns.keys()
    return { 'names' : truth_names }

# The function to call at the end of building each stamp
def ProcessTruthStamp(truth_cat, scratch, config, base, obj_num, logger=None):
    cols = config['columns']
    row = []
    types = []
    for name in truth_cat.getNames():
        key = cols[name]
        if isinstance(key, dict):
            # Then the "key" is actually something to be parsed in the normal way.
            # Caveat: We don't know the value_type here, so we give None.  This allows
            # only a limited subset of the parsing.  Usually enough for truth items, but
            # not fully featured.
            value = galsim.config.ParseValue(cols,name,base,None)[0]
        elif not isinstance(key,basestring):
            # The item can just be a constant value.
            value = key
        elif key[0] == '$':
            # This can also be handled by ParseValue
            value = galsim.config.ParseValue(cols,name,base,None)[0]
        else:
            value = galsim.config.GetCurrentValue(key, base)
        row.append(value)
        types.append(type(value))
    if truth_cat.getNObjects() == 0:
        truth_cat.setTypes(types)
    elif truth_cat.getTypes() != types:
        if logger:
            logger.error("Type mismatch found when building truth catalog at object %d",
                base['obj_num'])
            logger.error("Types for current object = %s",repr(types))
            logger.error("Expecting types = %s",repr(truth_cat.getTypes()))
        raise RuntimeError("Type mismatch found when building truth catalog.")
    scratch[obj_num] = row

# The function to call at the end of building each file to finalize the truth catalog
def FinalizeTruth(truth_cat, scratch, config, base, logger=None):
    # Add all the rows in order to the OutputCatalog
    obj_nums = sorted(scratch.keys())
    for obj_num in obj_nums:
        row = scratch[obj_num]
        truth_cat.addRow(row)
    return truth_cat

# Older versions of pyfits can't pickle HDUs, so this is a reimplementation of the
# OutputCatalog.writeFitsHdu function that can be run through a proxy OutputCatalog.
def BuildTruthHDU(truth_cat):
    import numpy
    from galsim._pyfits import pyfits
    data = truth_cat.makeData()
    cols = []
    for name in data.dtype.names:
        dt = data.dtype[name]
        if dt.kind in numpy.typecodes['AllInteger']:
            cols.append(pyfits.Column(name=name, format='J', array=data[name]))
        elif dt.kind in numpy.typecodes['AllFloat']:
            cols.append(pyfits.Column(name=name, format='D', array=data[name]))
        else:
            cols.append(pyfits.Column(name=name, format='%dA'%dt.itemsize, array=data[name]))
    cols = pyfits.ColDefs(cols)
    try:
        tbhdu = pyfits.BinTableHDU.from_columns(cols)
    except:
        tbhdu = pyfits.new_table(cols)
    return tbhdu

# Register this as a valid extra output
from .extra import RegisterExtraOutput
RegisterExtraOutput('truth',
                    init_func = galsim.OutputCatalog,
                    kwargs_func = GetTruthKwargs,
                    stamp_func = ProcessTruthStamp, 
                    final_func = FinalizeTruth,
                    write_func = galsim.OutputCatalog.write,
                    hdu_func = BuildTruthHDU)
