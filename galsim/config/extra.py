# Copyright (c) 2012-2021 by the GalSim developers team on GitHub
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

import os
import logging
import inspect

from .util import LoggerWrapper, SetDefaultExt, RetryIO, SafeManager
from .value import ParseValue
from ..utilities import ensure_dir
from ..errors import GalSimConfigValueError, GalSimConfigError
from ..fits import writeMulti

# This file handles the processing of extra output items in addition to the primary output file
# in config['output']. The ones that are defined natively in GalSim are psf, weight, badpix,
# and truth.  See extra_*.py for the specific functions for each of these.

# This module-level dict will store all the registered "extra" output types.
# See the RegisterExtraOutput function at the end of this file.
# The keys will be the (string) names of the extra output types, and the values will be
# builder classes that will perform the different processing functions.
valid_extra_outputs = {}

def SetupExtraOutput(config, logger=None):
    """
    Set up the extra output items as necessary, including building Managers for the work
    space so they can work safely in multi-processing mode.  Each builder will be placed in
    config['extra_builder'][key] where key is the key in galsim.config.valid_extra_outputs.

    Parameters:
        config:     The configuration dict.
        logger:     If given, a logger object to log progress. [default: None]
    """
    logger = LoggerWrapper(logger)
    output = config['output']
    file_num = config.get('file_num',0)

    # We'll iterate through this list of keys a few times
    all_keys = [ k for k in valid_extra_outputs.keys() if k in output ]

    # We don't need the manager stuff if we (a) are already in a multiprocessing Process, or
    # (b) config.image.nproc == 1.
    use_manager = (
            'current_nproc' not in config and
            'image' in config and 'nproc' in config['image'] and
            ParseValue(config['image'], 'nproc', config, int)[0] != 1 )

    if use_manager and 'output_manager' not in config:
        from multiprocessing.managers import ListProxy, DictProxy
        class OutputManager(SafeManager): pass

        # We'll use a list and a dict as work space to do the extra output processing.
        OutputManager.register('dict', dict, DictProxy)
        OutputManager.register('list', list, ListProxy)
        # Start up the output_manager
        config['output_manager'] = OutputManager()
        config['output_manager'].start()

    if 'extra_builder' not in config:
        config['extra_builder'] = {}

    # Keep track of any skipped obj_nums, since usually need to treat them differently.
    # Note: it would be slightly nicer to use a set here, but there isn't a pre-defined
    # multiprocessing.managers.SetProxy type, so we just use a dict like a set by giving
    # each item the value None.
    if '_skipped_obj_nums' in config:
        config['_skipped_obj_nums'].clear()
    elif use_manager:
        config['_skipped_obj_nums'] = config['output_manager'].dict()
    else:
        config['_skipped_obj_nums'] = dict()

    for key in all_keys:
        logger.debug('file %d: Setup output item %s',file_num,key)

        # Make the work space structures
        if use_manager:
            data = config['output_manager'].list()
            scratch = config['output_manager'].dict()
        else:
            data = list()
            scratch = dict()

        # Make the data list the right length now to avoid issues with multiple
        # processes trying to append at the same time.
        nimages = config.get('nimages', 1)
        for k in range(nimages):
            data.append(None)

        # Create the builder, giving it the data and scratch objects as work space.
        field = config['output'][key]
        builder = valid_extra_outputs[key]
        builder.initialize(data, scratch, field, config, logger)
        # And store it in the config dict
        config['extra_builder'][key] = builder

        logger.debug('file %d: Setup output %s object',file_num,key)


def SetupExtraOutputsForImage(config, logger=None):
    """Perform any necessary setup for the extra output items at the start of a new image.

    Parameters:
        config:     The configuration dict.
        logger:     If given, a logger object to log progress. [default: None]
    """
    if 'output' in config:
        if 'extra_builder' not in config:
            SetupExtraOutput(config, logger)
        for key, builder in config['extra_builder'].items():
            field = config['output'][key]
            builder.setupImage(field, config, logger)

def ProcessExtraOutputsForStamp(config, skip, logger=None):
    """Run the appropriate processing code for any extra output items that need to do something
    at the end of building each object.

    This gets called after all the object flux is added to the stamp, but before the sky level
    and noise are added.

    Parameters:
        config:     The configuration dict.
        skip:       Was the drawing of this object skipped?
        logger:     If given, a logger object to log progress. [default: None]
    """
    if 'output' in config:
        obj_num = config['obj_num']
        for key, builder in config.get('extra_builder',{}).items():
            field = config['output'][key]
            if skip:
                config['_skipped_obj_nums'][obj_num] = None
                builder.processSkippedStamp(obj_num, field, config, logger)
            else:
                builder.processStamp(obj_num, field, config, logger)


def ProcessExtraOutputsForImage(config, logger=None):
    """Run the appropriate processing code for any extra output items that need to do something
    at the end of building each image

    Parameters:
        config:     The configuration dict.
        logger:     If given, a logger object to log progress. [default: None]
    """
    if 'output' in config:
        obj_nums = None
        for key, builder in config.get('extra_builder',{}).items():
            image_num = config.get('image_num',0)
            start_image_num = config.get('start_image_num',0)
            if obj_nums is None:
                # Figure out which obj_nums were used for this image.
                file_num = config.get('file_num',0)
                start_obj_num = config.get('start_obj_num',0)
                nobj = config.get('nobj', [1])
                k = image_num - start_image_num
                for i in range(k):
                    start_obj_num += nobj[i]
                obj_nums = range(start_obj_num, start_obj_num+nobj[k])
                # Omit skipped obj_nums
                skipped = config['_skipped_obj_nums']
                obj_nums = [ n for n in obj_nums if n not in skipped ]
            field = config['output'][key]
            index = image_num - start_image_num
            builder.processImage(index, obj_nums, field, config, logger)


def WriteExtraOutputs(config, main_data, logger=None):
    """Write the extra output objects to files.

    This gets run at the end of the functions for building the regular output files.

    Parameters:
        config:     The configuration dict.
        main_data:  The main file data in case it is needed.
        logger:     If given, a logger object to log progress. [default: None]
    """
    logger = LoggerWrapper(logger)
    output = config['output']
    if 'retry_io' in output:
        ntries = ParseValue(config['output'],'retry_io',config,int)[0]
        # This is how many retries.  Do at least 1, so ntries is 1 more than this.
        ntries = ntries + 1
    else:
        ntries = 1

    if 'dir' in output:
        default_dir = ParseValue(output,'dir',config,str)[0]
    else:
        default_dir = None

    if 'noclobber' in output:
        noclobber = ParseValue(output,'noclobber',config,bool)[0]
    else:
        noclobber = False

    if 'extra_last_file' not in config:
        config['extra_last_file'] = {}

    for key, builder in config['extra_builder'].items():
        field = output[key]
        if 'file_name' in field:
            SetDefaultExt(field, '.fits')
            file_name = ParseValue(field,'file_name',config,str)[0]
        else:  # pragma: no cover  This is covered, but codecov thinks it isn't.
            # If no file_name, then probably writing to hdu
            continue
        if 'dir' in field:
            dir = ParseValue(field,'dir',config,str)[0]
        else:
            dir = default_dir

        if dir is not None:
            file_name = os.path.join(dir,file_name)

        ensure_dir(file_name)

        if noclobber and os.path.isfile(file_name):
            logger.warning('Not writing %s file %d = %s because output.noclobber = True '
                           'and file exists',key,config['file_num'],file_name)
            continue

        if config['extra_last_file'].get(key, None) == file_name:
            # If we already wrote this file, skip it this time around.
            # (Typically this is applicable for psf, where we may only want 1 psf file.)
            logger.info('Not writing %s file %d = %s because already written',
                        key,config['file_num'],file_name)
            continue

        # Do any final processing that needs to happen.
        builder.ensureFinalized(field, config, main_data, logger)

        # Call the write function, possibly multiple times to account for IO failures.
        write_func = builder.writeFile
        args = (file_name,field,config,logger)
        RetryIO(write_func, args, ntries, file_name, logger)
        config['extra_last_file'][key] = file_name
        logger.debug('file %d: Wrote %s to %r',config['file_num'],key,file_name)


def AddExtraOutputHDUs(config, main_data, logger=None):
    """Write the extra output objects to either HDUS or images as appropriate and add them
    to the existing data.

    This gets run at the end of the functions for building the regular output files.

    Note: the extra items must have hdu numbers ranging continuously (in any order) starting
    at len(data).  Typically first = 1, since the main image is the primary HDU, numbered 0.

    Parameters:
        config:     The configuration dict.
        main_data:  The main file data as a list of images.  Usually just [image] where
                    image is the primary image to be written to the output file.
        logger:     If given, a logger object to log progress. [default: None]

    Returns:
        data with additional hdus added
    """
    output = config['output']
    hdus = {}
    for key, builder in config['extra_builder'].items():
        field = output[key]
        if 'hdu' in field:
            hdu = ParseValue(field,'hdu',config,int)[0]
        else:  # pragma: no cover  This is covered, but codecov thinks it isn't.
            # If no hdu, then probably writing to file
            continue
        if hdu <= 0 or hdu in hdus:
            raise GalSimConfigValueError("hdu is invalid or a duplicate.",hdu)

        # Do any final processing that needs to happen.
        builder.ensureFinalized(field, config, main_data, logger)

        # Build the HDU for this output object.
        hdus[hdu] = builder.writeHdu(field,config,logger)

    first = len(main_data)
    for h in range(first,len(hdus)+first):
        if h not in hdus:
            raise GalSimConfigError("Cannot skip hdus.  No output found for hdu %d"%h)
    # Turn hdus into a list (in order)
    hdulist = [ hdus[k] for k in range(first,len(hdus)+first) ]
    return main_data + hdulist

def CheckNoExtraOutputHDUs(config, output_type, logger=None):
    """Check that none of the extra output objects want to add to the HDU list.

    Raises an exception if one of them has an hdu field.

    Parameters:
        config:         The configuration dict.
        output_type:    A string to use in the error message to indicate which output type
                        had a problem.
        logger:         If given, a logger object to log progress. [default: None]
    """
    logger = LoggerWrapper(logger)
    output = config['output']
    for key in config['extra_builder'].keys():
        field = output[key]
        if 'hdu' in field:
            hdu = ParseValue(field,'hdu',config,int)[0]
            logger.error("Extra output %s requesting to write to hdu %d", key, hdu)
            raise GalSimConfigError(
                "Output type %s cannot add extra images as HDUs"%output_type)


def GetFinalExtraOutput(key, config, main_data=[], logger=None):
    """Get the finalized output object for the given extra output key

    Parameters:
        key:        The name of the output field in config['output']
        config:     The configuration dict.
        main_data:  The main file data in case it is needed.  [default: []]
        logger:     If given, a logger object to log progress. [default: None]

    Returns:
        the final data to be output.
    """
    field = config['output'][key]
    return config['extra_builder'][key].ensureFinalized(field, config, main_data, logger)

class ExtraOutputBuilder(object):
    """A base class for building some kind of extra output object along with the main output.

    The base class doesn't do anything, but it defines the function signatures that a derived
    class can override to perform specific processing at any of several steps in the processing.

    The builder gets initialized with a list and and dict to use as work space.
    The typical work flow is to save something in scratch[obj_num] for each object built, and then
    process them all at the end of each image into data[k].  Then finalize may do something
    additional at the end of the processing to prepare the data to be written.

    It's worth remembering that the objects could potentially be processed in a random order if
    multiprocessing is being used.  The above work flow will thus work regardless of the order
    that the stamps and/or images are processed.

    Also, because of how objects are duplicated across processes during multiprocessing, you
    should not count on attributes you set in the builder object during the stamp or image
    processing stages to be present in the later finalize or write stages.  You should write
    any information you want to persist into the scratch or data objects, which are set up
    to handle the multiprocessing communication properly.
    """
    def initialize(self, data, scratch, config, base, logger):
        """Do any initial setup for this builder at the start of a new output file.

        The base class implementation saves two work space items into self.data and self.scratch
        that can be used to safely communicate across multiple processes.

        Parameters:
            data:       An empty list of length nimages to use as work space.
            scratch:    An empty dict that can be used as work space.
            config:     The configuration field for this output object.
            base:       The base configuration dict.
            logger:     If given, a logger object to log progress. [default: None]
        """
        self.data = data
        self.scratch = scratch
        self.final_data = None

    def setupImage(self, config, base, logger):
        """Perform any necessary setup at the start of an image.

        This function will be called at the start of each image to allow for any setup that
        needs to happen at this point in the processing.

        Parameters:
            config:     The configuration field for this output object.
            base:       The base configuration dict.
            logger:     If given, a logger object to log progress. [default: None]
        """
        pass

    def processStamp(self, obj_num, config, base, logger):
        """Perform any necessary processing at the end of each stamp construction.

        This function will be called after each stamp is built, but before the noise is added,
        so the existing stamp image has the true surface brightness profile (unless photon shooting
        was used, in which case there will necessarily be noise from that process).

        Remember, these stamps may be processed out of order.  Saving data to the scratch dict
        is safe, even if multiprocessing is being used.

        Parameters:
           obj_num:    The object number
           config:     The configuration field for this output object.
           base:       The base configuration dict.
           logger:     If given, a logger object to log progress. [default: None]
        """
        pass  # pragma: no cover  (all our ExtraBuilders override this function.)

    def processSkippedStamp(self, obj_num, config, base, logger):
        """Perform any necessary processing for stamps that were skipped in the normal processing.

        This function will be called for stamps that are not built because they were skipped
        for some reason.  Normally, you would not want to do anything for the extra outputs in
        these cases, but in case some module needs to do something in these cases as well, this
        method can be overridden.

        Parameters:
            obj_num:    The object number
            config:     The configuration field for this output object.
            base:       The base configuration dict.
            logger:     If given, a logger object to log progress. [default: None]
        """
        pass

    def processImage(self, index, obj_nums, config, base, logger):
        """Perform any necessary processing at the end of each image construction.

        This function will be called after each full image is built.

        Remember, these images may be processed out of order.  But if using the default
        constructor, the data list is already set to be the correct size, so it is safe to
        access self.data[k], where k = base['image_num'] - base['start_image_num'] is the
        appropriate index to use for this image.

        Parameters:
            index:      The index in self.data to use for this image.  This isn't the image_num
                        (which can be accessed at base['image_num'] if needed), but rather
                        an index that starts at 0 for the first image being worked on and
                        goes up to nimages-1.
            obj_nums:   The object numbers that were used for this image.
            config:     The configuration field for this output object.
            base:       The base configuration dict.
            logger:     If given, a logger object to log progress. [default: None]
        """
        pass

    def ensureFinalized(self, config, base, main_data, logger):
        """A helper function in the base class to make sure finalize only gets called once by the
        different possible locations that might need it to have been called.

        Parameters:
            config:     The configuration field for this output object.
            base:       The base configuration dict.
            main_data:  The main file data in case it is needed.
            logger:     If given, a logger object to log progress. [default: None]

        Returns:
            the final version of the object.
        """
        if self.final_data is None:
            self.final_data = self.finalize(config, base, main_data, logger)
        return self.final_data

    def finalize(self, config, base, main_data, logger):
        """Perform any final processing at the end of all the image processing.

        This function will be called after all images have been built.

        It returns some sort of final version of the object.  In the base class, it just returns
        self.data, but depending on the meaning of the output object, something else might be
        more appropriate.

        Parameters:
           config:     The configuration field for this output object.
           base:       The base configuration dict.
           main_data:  The main file data in case it is needed.
           logger:     If given, a logger object to log progress. [default: None]

        Returns:
           The final version of the object.
        """
        return self.data

    def writeFile(self, file_name, config, base, logger):
        """Write this output object to a file.

        The base class implementation is appropriate for the cas that the result of finalize
        is a list of images to be written to a FITS file.

        Parameters:
            file_name:  The file to write to.
            config:     The configuration field for this output object.
            base:       The base configuration dict.
            logger:     If given, a logger object to log progress. [default: None]
        """
        writeMulti(self.final_data, file_name)

    def writeHdu(self, config, base, logger):
        """Write the data to a FITS HDU with the data for this output object.

        The base class implementation is appropriate for the cas that the result of finalize
        is a list of images of length 1 to be written to a FITS file.

        Parameters:
            config:     The configuration field for this output object.
            base:       The base configuration dict.
            logger:     If given, a logger object to log progress. [default: None]

        Returns:
            an HDU with the output data.
        """
        if len(self.data) != 1:  # pragma: no cover  (Not sure if this is possible.)
            raise GalSimConfigError(
                    "%d %s images were created. Expecting 1."%(n,self._extra_output_key))
        return self.data[0]


def RegisterExtraOutput(key, builder):
    """Register an extra output field for use by the config apparatus.

    The builder parameter should be a subclass of galsim.config.ExtraOutputBuilder.
    See that class for the functions that should be defined and their signatures.
    Not all functions need to be overridden.  If nothing needs to be done at a particular place
    in the processing, you can leave the base class function, which doesn't do anything.

    Parameters:
        key:        The name of the output field in config['output']
        builder:    A builder object to use for building the extra output object.
                    It should be an instance of a subclass of ExtraOutputBuilder.
    """
    builder._extra_output_key = key
    valid_extra_outputs[key] = builder

# Nothing is registered here.  The appropriate items are registered in extra_*.py.
