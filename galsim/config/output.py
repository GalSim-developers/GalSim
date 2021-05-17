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

from .util import LoggerWrapper, UpdateNProc, CopyConfig, MultiProcess, SetupConfigRNG
from .util import RetryIO, SetDefaultExt
from .input import ProcessInput
from .extra import valid_extra_outputs, SetupExtraOutput, WriteExtraOutputs
from .extra import AddExtraOutputHDUs, CheckNoExtraOutputHDUs
from .value import ParseValue, CheckAllParams
from .image import BuildImage, GetNObjForImage
from ..errors import GalSimConfigError, GalSimConfigValueError
from ..utilities import ensure_dir
from ..fits import writeMulti

# This file handles building the output files according to the specifications in config['output'].
# This file includes the basic functionality, but it calls out to helper functions for the
# different types of output files.  It includes the implementation of the default output type,
# 'Fits'.  See output_multifits.py for 'MultiFits' and output_datacube.py for 'DataCube'.

# This module-level dict will store all the registered output types.
# See the RegisterOutputType function at the end of this file.
# The keys are the (string) names of the output types, and the values will be builder objects
# that will perform the different stages of processing to construct and write the output file(s).
valid_output_types = {}

def BuildFiles(nfiles, config, file_num=0, logger=None, except_abort=False):
    """
    Build a number of output files as specified in config.

    Parameters:
        nfiles:         The number of files to build.
        config:         A configuration dict.
        file_num:       If given, the first file_num. [default: 0]
        logger:         If given, a logger object to log progress. [default: None]
        except_abort:   Whether to abort processing when a file raises an exception (True)
                        or just report errors and continue on (False). [default: False]

    Returns:
        the final config dict that was used.
    """
    logger = LoggerWrapper(logger)
    import time
    t1 = time.time()

    # The next line relies on getting errors when the rng is undefined.  However, the default
    # rng is None, which is a valid thing to construct a Deviate object from.  So for now,
    # set the rng to object() to make sure we get errors where we are expecting to.
    config['rng'] = object()

    # Process the input field for the first file.  Often there are "safe" input items
    # that won't need to be reprocessed each time.  So do them here once and keep them
    # in the config for all file_nums.  This is more important if nproc != 1.
    ProcessInput(config, logger=logger, safe_only=True)

    jobs = []  # Will be a list of the kwargs to use for each job
    info = []  # Will be a list of (file_num, file_name) correspongind to each jobs.

    # Count from 0 to make sure image_num, etc. get counted right.  We'll start actually
    # building the files at first_file_num.
    first_file_num = file_num
    file_num = 0
    image_num = 0
    obj_num = 0

    # Figure out how many processes we will use for building the files.
    if 'output' not in config: config['output'] = {}
    output = config['output']
    if nfiles > 1 and 'nproc' in output:
        nproc = ParseValue(output, 'nproc', config, int)[0]
        # Update this in case the config value is -1
        nproc = UpdateNProc(nproc, nfiles, config, logger)
        # We'll want a pristine version later to give to the workers.
    else:
        nproc = 1
    orig_config = CopyConfig(config)

    if nfiles == 0:
        logger.error("No files were made, since nfiles == 0.")
        return orig_config

    for k in range(nfiles + first_file_num):
        SetupConfigFileNum(config, file_num, image_num, obj_num, logger)

        builder = valid_output_types[output['type']]
        builder.setup(output, config, file_num, logger)

        # Process the input fields that might be relevant at file scope:
        ProcessInput(config, logger=logger, file_scope_only=True)

        # Get the number of objects in each image for this file.
        nobj = GetNObjForFile(config,file_num,image_num,logger=logger)

        # The kwargs to pass to BuildFile
        kwargs = {
            'file_num' : file_num,
            'image_num' : image_num,
            'obj_num' : obj_num
        }

        if file_num >= first_file_num:
            # Get the file_name here, in case it needs to create directories, which is not
            # safe to do with multiple processes. (At least not without extra code in the
            # getFilename function...)
            file_name = builder.getFilename(output, config, logger)
            jobs.append(kwargs)
            info.append( (file_num, file_name) )

        # nobj is a list of nobj for each image in that file.
        # So len(nobj) = nimages and sum(nobj) is the total number of objects
        # This gets the values of image_num and obj_num ready for the next loop.
        file_num += 1
        image_num += len(nobj)
        obj_num += sum(nobj)

    def done_func(logger, proc, k, result, t2):
        file_num, file_name = info[k]
        file_name2, t = result  # This is the t for which 0 means the file was skipped.
        if file_name2 != file_name:  # pragma: no cover  (I think this should never happen.)
            raise GalSimConfigError("Files seem to be out of sync. %s != %s", file_name, file_name2)
        if t != 0 and logger:
            if proc is None: s0 = ''
            else: s0 = '%s: '%proc
            logger.warning(s0 + 'File %d = %s: time = %f sec', file_num, file_name, t)

    def except_func(logger, proc, k, e, tr):
        file_num, file_name = info[k]
        if proc is None: s0 = ''
        else: s0 = '%s: '%proc
        logger.error(s0 + 'Exception caught for file %d = %s', file_num, file_name)
        if except_abort:
            logger.debug('%s',tr)
            logger.error('File %s not written.',file_name)
        else:
            logger.warning('%s',tr)
            logger.error('File %s not written! Continuing on...',file_name)

    # Convert to the tasks structure we need for MultiProcess
    # Each task is a list of (job, k) tuples.  In this case, we only have one job per task.
    tasks = [ [ (job, k) ] for (k, job) in enumerate(jobs) ]

    results = MultiProcess(nproc, orig_config, BuildFile, tasks, 'file',
                           logger, done_func = done_func,
                           except_func = except_func,
                           except_abort = except_abort)
    t2 = time.time()

    if len(results) == 0:
        nfiles_written = 0
    else:
        fnames, times = zip(*results)
        nfiles_written = sum([ t!=0 for t in times])

    if nfiles_written == 0:
        logger.error('No files were written.  All were either skipped or had errors.')
    else:
        if nfiles_written > 1 and nproc != 1:
            logger.warning('Total time for %d files with %d processes = %f sec',
                           nfiles_written,nproc,t2-t1)
        logger.warning('Done building files')

    #Return the config used for the run - this may be useful since one can
    #save information here in e.g. custom output types
    return orig_config

output_ignore = [ 'nproc', 'skip', 'noclobber', 'retry_io' ]

def BuildFile(config, file_num=0, image_num=0, obj_num=0, logger=None):
    """
    Build an output file as specified in config.

    Parameters:
        config:         A configuration dict.
        file_num:       If given, the current file_num. [default: 0]
        image_num:      If given, the current image_num. [default: 0]
        obj_num:        If given, the current obj_num. [default: 0]
        logger:         If given, a logger object to log progress. [default: None]

    Returns:
        (file_name, t), a tuple of the file name and the time taken to build file
        Note: t==0 indicates that this file was skipped.
    """
    logger = LoggerWrapper(logger)
    import time
    t1 = time.time()

    SetupConfigFileNum(config, file_num, image_num, obj_num, logger)
    output = config['output']
    output_type = output['type']
    builder = valid_output_types[output_type]
    builder.setup(output, config, file_num, logger)

    # Put these values in the config dict so we won't have to run them again later if
    # we need them.  e.g. ExtraOuput processing uses these.
    nobj = GetNObjForFile(config,file_num,image_num,logger=logger)
    nimages = len(nobj)
    config['nimages'] = nimages
    config['nobj'] = nobj
    logger.debug('file %d: BuildFile with type=%s to build %d images, starting with %d',
                 file_num,output_type,nimages,image_num)

    # Make sure the inputs and extra outputs are set up properly.
    ProcessInput(config, logger=logger)
    SetupExtraOutput(config, logger=logger)

    # Get the file name
    file_name = builder.getFilename(output, config, logger)

    # Check if we ought to skip this file
    if 'skip' in output and ParseValue(output, 'skip', config, bool)[0]:
        logger.warning('Skipping file %d = %s because output.skip = True',file_num,file_name)
        return file_name, 0  # Note: time=0 is the indicator that a file was skipped.
    if ('noclobber' in output
        and ParseValue(output, 'noclobber', config, bool)[0]
        and os.path.isfile(file_name)):
        logger.warning('Skipping file %d = %s because output.noclobber = True'
                       ' and file exists',file_num,file_name)
        return file_name, 0

    if logger.isEnabledFor(logging.DEBUG):
        logger.debug('file %d: file_name = %s',file_num,file_name)
    else:
        logger.warning('Start file %d = %s', file_num, file_name)

    ignore = output_ignore + list(valid_extra_outputs)
    data = builder.buildImages(output, config, file_num, image_num, obj_num, ignore, logger)

    # If any images came back as None, then remove them, since they cannot be written.
    data = [ im for im in data if im is not None ]

    if len(data) == 0:
        logger.warning('Skipping file %d = %s because all images were None',file_num,file_name)
        return file_name, 0

    # Go back to file_num as the default index_key.
    config['index_key'] = 'file_num'

    data = builder.addExtraOutputHDUs(config, data, logger)

    if 'retry_io' in output:
        ntries = ParseValue(output,'retry_io',config,int)[0]
        # This is how many _re_-tries.  Do at least 1, so ntries is 1 more than this.
        ntries = ntries + 1
    else:
        ntries = 1

    args = (data, file_name, output, config, logger)
    RetryIO(builder.writeFile, args, ntries, file_name, logger)
    logger.debug('file %d: Wrote %s to file %r',file_num,output_type,file_name)

    builder.writeExtraOutputs(config, data, logger)

    t2 = time.time()

    return file_name, t2-t1

def GetNFiles(config, logger=None):
    """
    Get the number of files that will be made, based on the information in the config dict.

    Parameters:
        config:     The configuration dict.
        logger:     If given, a logger object to log progress. [default: None]

    Returns:
        the number of files
    """
    output = config.get('output',{})
    output_type = output.get('type','Fits')
    if output_type not in valid_output_types:
        raise GalSimConfigValueError("Invalid output.type.", output_type,
                                     list(valid_output_types.keys()))
    return valid_output_types[output_type].getNFiles(output, config, logger=logger)


def GetNImagesForFile(config, file_num, logger=None):
    """
    Get the number of images that will be made for the file number file_num, based on the
    information in the config dict.

    Parameters:
        config:     The configuration dict.
        file_num:   The current file number.
        logger:     If given, a logger object to log progress. [default: None]

    Returns:
        the number of images
    """
    output = config.get('output',{})
    output_type = output.get('type','Fits')
    if output_type not in valid_output_types:
        raise GalSimConfigValueError("Invalid output.type.", output_type,
                                     list(valid_output_types.keys()))
    return valid_output_types[output_type].getNImages(output, config, file_num, logger=logger)


def GetNObjForFile(config, file_num, image_num, logger=None):
    """
    Get the number of objects that will be made for each image built as part of the file file_num,
    which starts at image number image_num, based on the information in the config dict.

    Parameters:
        config:     The configuration dict.
        file_num:   The current file number.
        image_num:  The current image number.
        logger:     If given, a logger object to log progress. [default: None]

    Returns:
        a list of the number of objects in each image [ nobj0, nobj1, nobj2, ... ]
    """
    output = config.get('output',{})
    output_type = output.get('type','Fits')
    if output_type not in valid_output_types:
        raise GalSimConfigValueError("Invalid output.type.", output_type,
                                     list(valid_output_types.keys()))
    return valid_output_types[output_type].getNObjPerImage(output, config, file_num, image_num,
                                                           logger=logger)


def SetupConfigFileNum(config, file_num, image_num, obj_num, logger=None):
    """Do the basic setup of the config dict at the file processing level.

    Includes:
    - Set config['file_num'] = file_num
    - Set config['image_num'] = image_num
    - Set config['obj_num'] = obj_num
    - Set config['index_key'] = 'file_num'
    - Set config['start_image_num'] = image_num
    - Set config['start_obj_num'] = obj_num
    - Make sure config['output'] exists
    - Set default config['output']['type'] to 'Fits' if not specified
    - Check that the specified output type is valid.

    Parameters:
        config:     A configuration dict.
        file_num:   The current file_num. (If file_num=None, then don't set file_num or
                    start_obj_num items in the config dict.)
        image_num:  The current image_num.
        obj_num:    The current obj_num.
        logger:     If given, a logger object to log progress. [default: None]
    """
    logger = LoggerWrapper(logger)
    config['file_num'] = file_num
    config['start_obj_num'] = obj_num
    config['start_image_num'] = image_num
    config['image_num'] = image_num
    config['obj_num'] = obj_num
    config['index_key'] = 'file_num'

    if 'output' not in config:
        config['output'] = {}
    if 'type' not in config['output']:
        config['output']['type'] = 'Fits'

    # Check that the type is valid
    output_type = config['output']['type']
    if output_type not in valid_output_types:
        raise GalSimConfigValueError("Invalid output.type.", output_type,
                                     list(valid_output_types.keys()))


class OutputBuilder(object):
    """A base class for building and writing the output objects.

    The base class defines the call signatures of the methods that any derived class should follow.
    It also includes the implementation of the default output type: Fits.
    """

    # A class attribute that sub-classes may override.
    default_ext = '.fits'

    def setup(self, config, base, file_num, logger):
        """Do any necessary setup at the start of processing a file.

        The base class just calls SetupConfigRNG, but this provides a hook for sub-classes to
        do more things before any processing gets started on this file.

        Parameters:
            config:     The configuration dict for the output type.
            base:       The base configuration dict.
            file_num:   The current file_num.
            logger:     If given, a logger object to log progress.
        """
        seed = SetupConfigRNG(base, logger=logger)
        logger.debug('file %d: seed = %d',file_num,seed)

    def getFilename(self, config, base, logger):
        """Get the file_name for the current file being worked on.

        Note that the base class defines a default extension = '.fits'.
        This can be overridden by subclasses by changing the default_ext property.

        Parameters:
            config:     The configuration dict for the output type.
            base:       The base configuration dict.
            logger:     If given, a logger object to log progress.

        Returns:
            the filename to build.
        """
        if 'file_name' in config:
            SetDefaultExt(config['file_name'], self.default_ext)
            file_name = ParseValue(config, 'file_name', base, str)[0]
        elif 'root' in base and self.default_ext is not None:
            # If a file_name isn't specified, we use the name of the config file + '.fits'
            file_name = base['root'] + self.default_ext
        else:
            raise GalSimConfigError(
                "No file_name specified and unable to generate it automatically.")

        # Prepend a dir to the beginning of the filename if requested.
        if 'dir' in config:
            dir = ParseValue(config, 'dir', base, str)[0]
            file_name = os.path.join(dir,file_name)

        ensure_dir(file_name)

        return file_name

    def buildImages(self, config, base, file_num, image_num, obj_num, ignore, logger):
        """Build the images for output.

        In the base class, this function just calls BuildImage to build the single image to
        put in the output file.  So the returned list only has one item.

        Parameters:
            config:     The configuration dict for the output field.
            base:       The base configuration dict.
            file_num:   The current file_num.
            image_num:  The current image_num.
            obj_num:    The current obj_num.
            ignore:     A list of parameters that are allowed to be in config that we can
                        ignore here.  i.e. it won't be an error if they are present.
            logger:     If given, a logger object to log progress.

        Returns:
            a list of the images built
        """
        # There are no extra parameters to get, so just check that there are no invalid parameters
        # in the config dict.
        ignore += [ 'file_name', 'dir', 'nfiles' ]
        CheckAllParams(config, ignore=ignore)

        image = BuildImage(base, image_num, obj_num, logger=logger)
        return [ image ]

    def getNFiles(self, config, base, logger=None):
        """Returns the number of files to be built.

        In the base class, this is just output.nfiles.

        Parameters:
            config:     The configuration dict for the output field.
            base:       The base configuration dict.
            logger:     If given, a logger object to log progress.

        Returns:
            the number of files to build.
        """
        if 'nfiles' in config:
            return ParseValue(config, 'nfiles', base, int)[0]
        else:
            return 1

    def getNImages(self, config, base, file_num, logger=None):
        """Returns the number of images to be built for a given ``file_num``.

        In the base class, we only build a single image, so it returns 1.

        Parameters:
            config:     The configuration dict for the output field.
            base:       The base configuration dict.
            file_num:   The current file number.
            logger:     If given, a logger object to log progress.

        Returns:
           the number of images to build.
        """
        return 1

    def getNObjPerImage(self, config, base, file_num, image_num, logger=None):
        """
        Get the number of objects that will be made for each image built as part of the file
        file_num, which starts at image number image_num, based on the information in the config
        dict.

        Parameters:
            config:         The configuration dict.
            base:           The base configuration dict.
            file_num:       The current file number.
            image_num:      The current image number (the first one for this file).
            logger:         If given, a logger object to log progress.

        Returns:
            a list of the number of objects in each image [ nobj0, nobj1, nobj2, ... ]
        """
        nimages = self.getNImages(config, base, file_num, logger=logger)
        nobj = [ GetNObjForImage(base, image_num+j, logger=logger) for j in range(nimages) ]
        base['image_num'] = image_num  # Make sure this is set back to current image num.
        return nobj

    def canAddHdus(self):
        """Returns whether it is permissible to add extra HDUs to the end of the data list.

        In the base class, this returns True.
        """
        return True

    def addExtraOutputHDUs(self, config, data, logger):
        """If appropriate, add any extra output items that go into HDUs to the data list.

        Parameters:
            config:     The configuration dict for the output field.
            data:       The data to write.  Usually a list of images.
            logger:     If given, a logger object to log progress.

        Returns:
            data (possibly updated with additional items)
        """
        if self.canAddHdus():
            data = AddExtraOutputHDUs(config, data, logger)
        else:
            CheckNoExtraOutputHDUs(config, config['output']['type'], logger)
        return data

    def writeFile(self, data, file_name, config, base, logger):
        """Write the data to a file.

        Parameters:
            data:           The data to write.  Usually a list of images returned by
                            buildImages, but possibly with extra HDUs tacked onto the end
                            from the extra output items.
            file_name:      The file_name to write to.
            config:         The configuration dict for the output field.
            base:           The base configuration dict.
            logger:         If given, a logger object to log progress.
        """
        writeMulti(data,file_name)

    def writeExtraOutputs(self, config, data, logger):
        """If appropriate, write any extra output items that write their own files.

        Parameters:
            config:     The configuration dict for the output field.
            data:       The data to write.  Usually a list of images.
            logger:     If given, a logger object to log progress.
        """
        WriteExtraOutputs(config, data, logger)


def RegisterOutputType(output_type, builder):
    """Register an output type for use by the config apparatus.

    Parameters:
        output_type:    The name of the type in config['output']
        builder:        A builder object to use for building and writing the output file.
                        It should be an instance of OutputBuilder or a subclass thereof.
    """
    # Make a concrete instance of the builder.
    valid_output_types[output_type] = builder

# The base class is also the builder for type = Fits.
RegisterOutputType('Fits', OutputBuilder())

