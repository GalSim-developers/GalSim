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

from past.builtins import basestring
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

from .extra import ExtraOutputBuilder
class TruthBuilder(ExtraOutputBuilder):
    """Build an output truth catalog with user-defined columns, typically taken from
    current values of various quantities for each constructed object.
    """
    def initialize(self, data, scratch, config, base, logger):
        # Call the base class initialize first.
        super(TruthBuilder,self).initialize(data,scratch,config,base,logger)

        # Warn if the config dict isn't an OrderedDict.
        if logger and not hasattr(config, '__reversed__') and not hasattr(self,'warned'): # pragma: no cover
            # If config doesn't have a __reversed__ attribute, then it's not an OrderedDict.
            # Probably it's just a regular dict.  So warn the user that the columns are in
            # arbitrary order.
            # (This was the simplest difference I could find between dict and OrderedDict that
            #  seemed relevant.)
            logger.warning('The config dict is not an OrderedDict.  The columns in the output '
                           'truth catalog will be in arbitrary order.')
            self.warned = True

    # The function to call at the end of building each stamp
    def processStamp(self, obj_num, config, base, logger):
        cols = config['columns']
        row = []
        types = []
        for name in cols:
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
            elif key[0] == '@':
                # Pop off an initial @ if there is one.
                value = galsim.config.GetCurrentValue(str(key[1:]), base)
            else:
                # str(key) handles the possibility of unicode.  In particular, this happens with
                # JSON files.
                value = galsim.config.GetCurrentValue(str(key), base)
            row.append(value)
            types.append(type(value))
        if 'types' not in self.scratch:
            self.scratch['types'] = types
        elif self.scratch['types'] != types: # pragma: no cover
            logger.error("Type mismatch found when building truth catalog at object %d",
                         base['obj_num'])
            logger.error("Types for current object = %s",repr(types))
            logger.error("Expecting types = %s",repr(self.scratch['types']))
            raise galsim.GalSimConfigError("Type mismatch found when building truth catalog.")
        self.scratch[obj_num] = row

    # The function to call at the end of building each file to finalize the truth catalog
    def finalize(self, config, base, main_data, logger):
        # Make the OutputCatalog
        cols = config['columns']
        # Note: Provide a default here, because if all items were skipped it would otherwise
        # lead to a KeyError.
        types = self.scratch.pop('types', [float] * len(cols))
        self.cat = galsim.OutputCatalog(names=cols.keys(), types=types)

        # Add all the rows in order to the OutputCatalog
        # Note: types was popped above, so only the obj_num keys are left.
        obj_nums = sorted(self.scratch.keys())
        for obj_num in obj_nums:
            row = self.scratch[obj_num]
            self.cat.addRow(row)
        return self.cat

    # Write the catalog to a file
    def writeFile(self, file_name, config, base, logger):
        self.cat.write(file_name)

    # Create an HDU of the FITS binary table.
    def writeHdu(self, config, base, logger):
        return self.cat.writeFitsHdu()

# Register this as a valid extra output
from .extra import RegisterExtraOutput
RegisterExtraOutput('truth', TruthBuilder())
