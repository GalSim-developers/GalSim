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

import os
import galsim
import logging

# This file handles building the output files according to the specifications in config['output'].
# This file includes the basic functionality, but it calls out to helper functions for the
# different types of output files.  It includes the implementation of the default output type,
# 'Fits'.  See output_multifits.py for 'MultiFits' and output_datacube.py for 'DataCube'.

# This module-level dict will store all the registered output types.
# See the RegisterOutputType function at the end of this file.
# The keys are the (string) names of the output types, and the values will be builder objects
# will perform the different stages of processing to construct and write the output file(s).
valid_output_types = {}


def BuildFiles(nfiles, config, file_num=0, logger=None):
    """
    Build a number of output files as specified in config.

    @param nfiles           The number of files to build.
    @param config           A configuration dict.
    @param file_num         If given, the first file_num. [default: 0]
    @param logger           If given, a logger object to log progress. [default: None]
    """
    import time
    t1 = time.time()

    # Process the input field for the first file.  Often there are "safe" input items
    # that won't need to be reprocessed each time.  So do them here once and keep them
    # in the config for all file_nums.  This is more important if nproc != 1.
    galsim.config.ProcessInput(config, file_num=0, logger=logger, safe_only=True)

    # We'll want a pristine version later to give to the workers.
    orig_config = galsim.config.CopyConfig(config)

    jobs = []

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
        nproc = galsim.config.ParseValue(output, 'nproc', config, int)[0]
        # Update this in case the config value is -1
        nproc = galsim.config.UpdateNProc(nproc, nfiles, config, logger)
    else:
        nproc = 1

    for k in range(nfiles + first_file_num):
        SetupConfigFileNum(config, file_num, image_num, obj_num)
        seed = galsim.config.SetupConfigRNG(config)

        # Get the number of objects in each image for this file.
        nobj = GetNObjForFile(config,file_num,image_num)

        # Process the input fields that might be relevant at file scope:
        galsim.config.ProcessInput(config, file_num=file_num, logger=logger, file_scope_only=True)

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
            output_type = output['type']
            file_name = valid_output_types[output_type].getFilename(output, config, logger)
            jobs.append( (kwargs, (file_num, file_name)) )

        # nobj is a list of nobj for each image in that file.
        # So len(nobj) = nimages and sum(nobj) is the total number of objects
        # This gets the values of image_num and obj_num ready for the next loop.
        file_num += 1
        image_num += len(nobj)
        obj_num += sum(nobj)

    def done_func(logger, proc, info, result, t2):
        file_num, file_name = info
        file_name2, t = result  # This is the t for which 0 means the file was skipped.
        if file_name2 != file_name:
            raise RuntimeError("Files seem to be out of sync. %s != %s",file_name, file_name2)
        if t != 0 and logger:
            if proc is None: s0 = ''
            else: s0 = '%s: '%proc
            logger.warn(s0 + 'File %d = %s: time = %f sec', file_num, file_name, t)

    def except_func(logger, proc, e, tr, info):
        if logger:
            file_num, file_name = info
            if proc is None: s0 = ''
            else: s0 = '%s: '%proc
            logger.error(s0 + 'Exception caught for file %d = %s', file_num, file_name)
            logger.error('%s',tr)
            logger.error('File %s not written! Continuing on...',file_name)

    results = galsim.config.MultiProcess(nproc, orig_config, BuildFile, jobs, 'file',
                                         logger, done_func = done_func,
                                         except_func = except_func,
                                         except_abort = False)
    t2 = time.time()

    if not results:
        nfiles_written = 0
    else:
        fnames, times = zip(*results)
        nfiles_written = sum([ t!=0 for t in times])

    if nfiles_written == 0:
        if logger:
            logger.error('No files were written.  All were either skipped or had errors.')
    else:
        if logger:
            if nfiles_written > 1 and nproc != 1:
                logger.warn('Total time for %d files with %d processes = %f sec',
                            nfiles_written,nproc,t2-t1)
            logger.warn('Done building files')


output_ignore = [ 'file_name', 'dir', 'nfiles', 'nproc', 'skip', 'noclobber', 'retry_io' ]

def BuildFile(config, file_num=0, image_num=0, obj_num=0, logger=None):
    """
    Build an output file as specified in config.

    @param config           A configuration dict.
    @param file_num         If given, the current file_num. [default: 0]
    @param image_num        If given, the current image_num. [default: 0]
    @param obj_num          If given, the current obj_num. [default: 0]
    @param logger           If given, a logger object to log progress. [default: None]

    @returns a tuple of the file name and the time taken to build file: (file_name, t)
    Note: t==0 indicates that this file was skipped.
    """
    import time
    t1 = time.time()

    SetupConfigFileNum(config,file_num,image_num,obj_num)
    seed = galsim.config.SetupConfigRNG(config)
    if logger:
        logger.debug('file %d: seed = %d',file_num,seed)

    # Put these values in the config dict so we won't have to run them again later if
    # we need them.  e.g. ExtraOuput processing uses these.
    nobj = GetNObjForFile(config,file_num,image_num)
    nimages = len(nobj)
    config['nimages'] = nimages
    config['nobj'] = nobj

    output = config['output']
    output_type = output['type']

    if logger:
        logger.debug('file %d: Build File with type=%s to build %d images, starting with %d',
                      file_num,output_type,nimages,image_num)

    # Make sure the inputs and extra outputs are set up properly.
    galsim.config.ProcessInput(config, file_num=file_num, logger=logger)
    galsim.config.SetupExtraOutput(config, file_num=file_num, logger=logger)

    builder = valid_output_types[output_type]

    # Get the file name
    file_name = builder.getFilename(output, config, logger)

    # Check if we ought to skip this file
    if 'skip' in output and galsim.config.ParseValue(output, 'skip', config, bool)[0]:
        if logger:
            logger.warn('Skipping file %d = %s because output.skip = True',file_num,file_name)
        t2 = time.time()
        return file_name, 0
    if ('noclobber' in output
        and galsim.config.ParseValue(output, 'noclobber', config, bool)[0]
        and os.path.isfile(file_name)):
        if logger:
            logger.warn('Skipping file %d = %s because output.noclobber = True' +
                        ' and file exists',file_num,file_name)
        t2 = time.time()
        return file_name, 0

    if logger:
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug('file %d: file_name = %s',file_num,file_name)
        else:
            logger.warn('Start file %d = %s', file_num, file_name)

    ignore = output_ignore + galsim.config.valid_extra_outputs.keys()
    data = builder.buildImages(output, config, file_num, image_num, obj_num, ignore, logger)

    if builder.canAddHdus():
        data = data + galsim.config.BuildExtraOutputHDUs(config,logger,len(data))

    if 'retry_io' in output:
        ntries = galsim.config.ParseValue(output,'retry_io',config,int)[0]
        # This is how many _re_-tries.  Do at least 1, so ntries is 1 more than this.
        ntries = ntries + 1
    else:
        ntries = 1

    args = (data, file_name)
    RetryIO(builder.writeFile, args, ntries, file_name, logger)
    if logger:
        logger.debug('file %d: Wrote %s to file %r',file_num,output_type,file_name)

    galsim.config.WriteExtraOutputs(config,logger)
    t2 = time.time()

    return file_name, t2-t1

def GetNImagesForFile(config, file_num):
    """
    Get the number of images that will be made for the file number file_num, based on the
    information in the config dict.

    @param config           The configuration dict.
    @param file_num         The current file number.

    @returns the number of images
    """
    output = config['output']
    if 'type' in config['output']:
        output_type = output['type']
    else:
        output_type = 'Fits'

    return valid_output_types[output_type].getNImages(output, config, file_num)


def GetNObjForFile(config, file_num, image_num):
    """
    Get the number of objects that will be made for each image built as part of the file file_num,
    which starts at image number image_num, based on the information in the config dict.

    @param config           The configuration dict.
    @param file_num         The current file number.
    @param image_num        The current image number.

    @returns a list of the number of objects in each image [ nobj0, nobj1, nobj2, ... ]
    """
    nimages = GetNImagesForFile(config, file_num)

    try :
        nobj = [ galsim.config.GetNObjForImage(config, image_num+j) for j in range(nimages) ]
    except ValueError : # (This may be raised if something needs the input stuff)
        galsim.config.ProcessInput(config, file_num=file_num)
        nobj = [ galsim.config.GetNObjForImage(config, image_num+j) for j in range(nimages) ]
    return nobj


def SetupConfigFileNum(config, file_num, image_num, obj_num):
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

    @param config           A configuration dict.
    @param file_num         The current file_num. (If file_num=None, then don't set file_num or
                            start_obj_num items in the config dict.)
    @param image_num        The current image_num.
    @param obj_num          The current obj_num.
    """
    if file_num is None:
        if 'file_num' not in config: config['file_num'] = 0
        if 'start_obj_num' not in config: config['start_obj_num'] = obj_num
        if 'start_image_num' not in config: config['start_image_num'] = image_num
    else:
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
        raise AttributeError("Invalid output.type=%s."%output_type)


def SetDefaultExt(config, default_ext):
    """Set a default ext in a config 'file_name' field if appropriate.

    @param config           The configuration dict for the item that might need to be given
                            a default 'ext' value.
    @param default_ext      The default extension to set in the config dict if one is not set.
    """
    if default_ext is not None:
        if ( isinstance(config,dict) and 'type' in config and
            config['type'] == 'NumberedFile' and 'ext' not in config ):
            config['ext'] = default_ext


# A helper function to retry io commands
def RetryIO(func, args, ntries, file_name, logger):
    for itry in range(ntries):
        try:
            ret = func(*args)
        except IOError as e:
            if itry == ntries-1:
                # Then this was the last try.  Just re-raise the exception.
                raise
            else:
                if logger:
                    logger.warn('File %s: Caught IOError: %s',file_name,str(e))
                    logger.warn('This is try %d/%d, so sleep for %d sec and try again.',
                                itry+1,ntries,itry+1)
                import time
                time.sleep(itry+1)
                continue
        else:
            break
    return ret


class OutputBuilder(object):
    """A base class for building and writing the output objects.

    The base class defines the call signatures of the methods that any derived class should follow.
    It also includes the implementation of the default output type: Fits.
    """

    # A class attribute that sub-classes may override.
    default_ext = '.fits'

    def getFilename(self, config, base, logger):
        """Get the file_name for the current file being worked on.

        Note that the base class defines a default extension = '.fits'.
        This can be overridden by subclasses by changing the default_ext property.

        @param config           The configuration dict for the output type.
        @param base             The base configuration dict.
        @param image_num        The current image_num.
        @param obj_num          The current obj_num.
        @param ignore           A list of parameters that are allowed to be in config['output']
                                that we can ignore here.  i.e. it won't be an error if these
                                parameters are present.
        @param logger           If given, a logger object to log progress.

        @returns the filename to build.
        """
        if 'file_name' in config:
            SetDefaultExt(config['file_name'], self.default_ext)
            file_name = galsim.config.ParseValue(config, 'file_name', base, str)[0]
        elif 'root' in base and self.default_ext is not None:
            # If a file_name isn't specified, we use the name of the config file + '.fits'
            file_name = base['root'] + self.default_ext
        else:
            raise AttributeError("No file_name specified and unable to generate it automatically.")

        # Prepend a dir to the beginning of the filename if requested.
        if 'dir' in config:
            dir = galsim.config.ParseValue(config, 'dir', base, str)[0]
            if dir and not os.path.isdir(dir): os.makedirs(dir)
            file_name = os.path.join(dir,file_name)

        return file_name

    def buildImages(self, config, base, file_num, image_num, obj_num, ignore, logger):
        """Build the images for output.

        In the base class, this function just calls BuildImage to build the single image to
        put in the output file.  So the returned list only has one item.

        @param config           The configuration dict for the output field.
        @param base             The base configuration dict.
        @param file_num         The current file_num.
        @param image_num        The current image_num.
        @param obj_num          The current obj_num.
        @param ignore           A list of parameters that are allowed to be in config that we can
                                ignore here.  i.e. it won't be an error if they are present.
        @param logger           If given, a logger object to log progress.

        @returns a list of the images built
        """
        # There are no extra parameters to get, so just check that there are no invalid parameters
        # in the config dict.
        galsim.config.CheckAllParams(config, ignore=ignore)

        image = galsim.config.BuildImage(base, image_num, obj_num, logger=logger)
        return [ image ]

    def getNImages(self, config, base, file_num):
        """Returns the number of images to be built.

        In the base class, we only build a single image, so it returns 1.

        @param config           The configuration dict for the output field.
        @param base             The base configuration dict.
        @param file_num         The current file number.

        @returns the number of images to build.
        """
        return 1

    def writeFile(self, data, file_name):
        """Write the data to a file.

        @param data             The data to write.  Usually a list of images returned by
                                buildImages, but possibly with extra HDUs tacked onto the end
                                from the extra output items.
        @param file_name        The file_name to write to.
        """
        galsim.fits.writeMulti(data,file_name)

    def canAddHdus(self):
        """Returns whether it is permissible to add extra HDUs to the end of the data list.

        In the base class, this returns True.
        """
        return True


def RegisterOutputType(output_type, builder):
    """Register an output type for use by the config apparatus.

    @param output_type      The name of the type in config['output']
    @param builder          A builder object to use for building and writing the output file.
                            It should be an instance of OutputBuilder or a subclass thereof.
    """
    # Make a concrete instance of the builder.
    valid_output_types[output_type] = builder

# The base class is also the builder for type = Fits.
RegisterOutputType('Fits', OutputBuilder())

