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


def BuildFiles(nfiles, config, file_num=0, image_num=0, obj_num=0, nproc=1, logger=None):
    """
    Build a number of output files as specified in config.
    
    @param nfiles           The number of files to build.
    @param config           A configuration dict.
    @param file_num         If given, the first file_num. [default: 0]
    @param image_num        If given, the first image_num. [default: 0]
    @param obj_num          If given, the first obj_num. [default: 0]
    @param nproc            How many processes to use for building the images. [default: 1]
    @param logger           If given, a logger object to log progress. [default: None]
    """
 
    # nproc_image will be passed to the build function to be acted on at that level.
    nproc_image = 1
    if nproc > nfiles:
        if nfiles == 1:
            nproc_image = nproc 
            nproc = 1
        else:
            if logger and logger.isEnabledFor(logging.DEBUG):
                logger.debug("There are only %d files.  Reducing nproc to %d."%(nfiles,nfiles))
            nproc = nfiles

    def worker(input, output, config, logger):
        proc = current_process().name
        for job in iter(input.get, 'STOP'):
            try:
                (kwargs, file_num, file_name) = job
                galsim.config.RemoveCurrent(config, keep_safe=True)
                galsim.config.ProcessInput(config, file_num=file_num, logger=logger)
                galsim.config.SetupExtraOutput(config, file_num=file_num, logger=logger)
                kwargs['config'] = config
                kwargs['logger'] = logger
                result = BuildFile(**kwargs)
                output.put( (result, file_num, file_name, proc) )
            except Exception as e:
                import traceback
                tr = traceback.format_exc()
                if logger and logger.isEnabledFor(logging.WARN):
                    logger.warn('%s: Caught exception %s\n%s',proc,str(e),tr)
                output.put( (e, file_num, file_name, tr) )
        if logger and logger.isEnabledFor(logging.DEBUG):
            logger.debug('%s: Received STOP',proc)

    # Set up the multi-process task_queue if we're going to need it.
    if nproc > 1:
        # NB: See the function BuildStamps for more verbose comments about how
        # the multiprocessing stuff works.
        from multiprocessing import Process, Queue, current_process
        task_queue = Queue()

    # Now start working on the files.
    image_num = 0
    obj_num = 0
    config['file_num'] = 0
    config['image_num'] = 0
    config['obj_num'] = 0

    # Process the input field for the first file.  Often there are "safe" input items
    # that won't need to be reprocessed each time.  So do them here once and keep them
    # in the config for all file_nums.  This is more important if nproc != 1.
    galsim.config.ProcessInput(config, file_num=0, logger=logger, safe_only=True)

    # We'll want a pristine version later to give to the workers.
    orig_config = galsim.config.CopyConfig(config)
    for file_num in range(nfiles):
        if logger and logger.isEnabledFor(logging.DEBUG):
            logger.debug('file_num, image_num, obj_num = %d,%d,%d',file_num,image_num,obj_num)
        SetupConfigFileNum(config,file_num,image_num,obj_num)
        seed = galsim.config.SetupConfigRNG(config)
        if logger and logger.isEnabledFor(logging.DEBUG):
            logger.debug('file %d: seed = %d',file_num,seed)

        # Process the input fields that might be relevant at file scope:
        galsim.config.ProcessInput(config, file_num=file_num, logger=logger, file_scope_only=True)

        # Assign some of the kwargs we know now:
        kwargs = {
            'nproc' : nproc_image,
            'file_num' : file_num,
            'image_num' : image_num,
            'obj_num' : obj_num
        }

        output = config['output']
        # This also updates nimages or nobjects as needed if they are being automatically
        # set from an input catalog.
        nobj = GetNObjForFile(config,file_num,image_num)
        if logger and logger.isEnabledFor(logging.DEBUG):
            logger.debug('file %d: nobj = %s',file_num,str(nobj))

        # nobj is a list of nobj for each image in that file.
        # So len(nobj) = nimages and sum(nobj) is the total number of objects
        # This gets the values of image_num and obj_num ready for the next loop.
        image_num += len(nobj)
        obj_num += sum(nobj)

        # Get the file_name here, in case it needs to create directories, which is not 
        # safe to do with multiple processes. (At least not without extra code in the 
        # GetFilename function...)
        output_type = output.get('type', 'Fits')
        default_ext = valid_output_types[output_type]['ext']
        file_name = GetFilename(output, config, default_ext)

        # This is where we actually build the file.
        # If we're doing multiprocessing, we send this information off to the task_queue.
        # Otherwise, we just call BuildFile
        if nproc > 1:
            import copy
            # Make new copies of kwargs so we can update them without
            # clobbering the versions for other tasks on the queue.
            kwargs1 = copy.copy(kwargs)
            task_queue.put( (kwargs1, file_num, file_name) )
        else:
            try:
                config1 = galsim.config.CopyConfig(orig_config)
                galsim.config.RemoveCurrent(config1, keep_safe=True)
                galsim.config.ProcessInput(config1, file_num=file_num, logger=logger)
                galsim.config.SetupExtraOutput(config1, file_num=file_num, logger=logger)
                kwargs['config'] = config1
                kwargs['logger'] = logger 
                file_name, t = BuildFile(**kwargs)
                if logger and logger.isEnabledFor(logging.WARN):
                    logger.warn('File %d = %s: time = %f sec', file_num, file_name, t)
            except Exception as e:
                import traceback
                tr = traceback.format_exc()
                if logger:
                    logger.error('Exception caught for file %d = %s', file_num, file_name)
                    logger.error('%s',tr)
                    logger.error('%s',e)
                    try:
                        logger.error('File %s not written! Continuing on...',file_name)
                    except:
                        pass

    # If we're doing multiprocessing, here is the machinery to run through the task_queue
    # and process the results.
    if nproc > 1:
        if logger and logger.isEnabledFor(logging.WARN):
            logger.warn("Using %d processes for file processing",nproc)
        import time
        t1 = time.time()
        # Run the tasks
        done_queue = Queue()
        p_list = []
        config1 = galsim.config.CopyConfig(orig_config)
        config1['current_nproc'] = nproc
        logger_proxy = galsim.config.GetLoggerProxy(logger)
        for j in range(nproc):
            p = Process(target=worker, args=(task_queue, done_queue, config1, logger_proxy),
                        name='Process-%d'%(j+1))
            p.start()
            p_list.append(p)

        # Log the results.
        nfiles_written = 0  # Don't count skipped files.
        for k in range(nfiles):
            result, file_num, file_name, proc = done_queue.get()
            if isinstance(result,Exception):
                # result is really the exception, e
                # proc is really the traceback
                if logger:
                    logger.error('Exception caught for file %d = %s', file_num, file_name)
                    # And proc is really the traceback.
                    logger.error('%s',proc)
                    logger.error('%s',result)
                    logger.error('File %s not written! Continuing on...',file_name)
            else:
                file_name, t = result
                if t != 0 and logger and logger.isEnabledFor(logging.WARN):
                    logger.warn('%s: File %d = %s: time = %f sec', proc, file_num, file_name, t)
                    nfiles_written += 1

        # Stop the processes
        for j in range(nproc):
            task_queue.put('STOP')
        for j in range(nproc):
            p_list[j].join()
        task_queue.close()
        t2 = time.time()
        if logger and logger.isEnabledFor(logging.WARN):
            logger.warn('Total time for %d files with %d processes = %f sec', 
                        nfiles_written,nproc,t2-t1)

    if logger and logger.isEnabledFor(logging.WARN):
        logger.warn('Done building files')

output_ignore = [ 'file_name', 'dir', 'nfiles', 'nproc', 'skip', 'noclobber', 'retry_io' ]

def BuildFile(config, file_num=0, image_num=0, obj_num=0, nproc=1, logger=None):
    """
    Build an output file as specified in config.
    
    @param config           A configuration dict.
    @param file_num         If given, the current file_num. [default: 0]
    @param image_num        If given, the current image_num. [default: 0]
    @param obj_num          If given, the current obj_num. [default: 0]
    @param nproc            How many processes to use for building the images. [default: 1]
    @param logger           If given, a logger object to log progress. [default: None]

    @returns a tuple of the file name and the time taken to build file: (file_name, t)
    Note: t==0 indicates that this file was skipped.
    """
    import time
    t1 = time.time()

    if 'output' not in config:
        config['output'] = {}
    output = config['output']
    output_type = output.get('type','Fits')

    if output_type not in valid_output_types:
        raise AttributeError("Invalid output.type=%s."%output_type)
    if logger and logger.isEnabledFor(logging.DEBUG):
        logger.debug('file %d: Build File with type=%s for file, image, obj = %d,%d,%d',
                      file_num,output_type,file_num,image_num,obj_num)

    SetupConfigFileNum(config,file_num,image_num,obj_num)
    seed = galsim.config.SetupConfigRNG(config)
    if logger and logger.isEnabledFor(logging.DEBUG):
        logger.debug('file %d: seed = %d',file_num,seed)

    # Get the file name
    default_ext = valid_output_types[output_type]['ext']
    file_name = GetFilename(output, config, default_ext)

    # Check if we ought to skip this file
    if 'skip' in output and galsim.config.ParseValue(output, 'skip', config, bool)[0]:
        if logger and logger.isEnabledFor(logging.WARN):
            logger.warn('Skipping file %d = %s because output.skip = True',file_num,file_name)
        t2 = time.time()
        return file_name, 0
    if ('noclobber' in output
        and galsim.config.ParseValue(output, 'noclobber', config, bool)[0]
        and os.path.isfile(file_name)):
        if logger and logger.isEnabledFor(logging.WARN):
            logger.warn('Skipping file %d = %s because output.noclobber = True' +
                        ' and file exists',file_num,file_name)
        t2 = time.time()
        return file_name, 0

    if logger: 
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug('file %d: file_name = %s',file_num,file_name)
        elif logger.isEnabledFor(logging.WARN):
            logger.warn('Start file %d = %s', file_num, file_name)

    build_func = valid_output_types[output_type]['build']
    ignore = output_ignore + galsim.config.valid_extra_outputs.keys()
    data = build_func(config, file_num, image_num, obj_num, nproc, ignore, logger)

    can_add_hdus = valid_output_types[output_type]['hdus']
    if can_add_hdus:
        data = data + galsim.config.BuildExtraOutputHDUs(config,logger,len(data))

    if 'retry_io' in output:
        ntries = galsim.config.ParseValue(output,'retry_io',config,int)[0]
        # This is how many _re_-tries.  Do at least 1, so ntries is 1 more than this.
        ntries = ntries + 1
    else:
        ntries = 1

    write_func = valid_output_types[output_type]['write']
    args = (data, file_name)
    RetryIO(write_func, args, ntries, file_name, logger)
    if logger and logger.isEnabledFor(logging.DEBUG):
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
    if 'output' in config and 'type' in config['output']:
        output_type = config['output']['type']
    else:
        output_type = 'Fits'

    # Check that the type is valid
    if output_type not in valid_output_types:
        raise AttributeError("Invalid output.type=%s."%output_type)

    nim_func = valid_output_types[output_type]['nim']
    return nim_func(config, file_num)
 

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

    config['index_key'] = 'file_num'
    config['file_num'] = file_num
    config['image_num'] = image_num
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
    - Set config['start_obj_num'] = obj_num

    @param config           A configuration dict.
    @param file_num         The current file_num. (If file_num=None, then don't set file_num or
                            start_obj_num items in the config dict.)
    @param image_num        The current image_num.
    @param obj_num          The current obj_num.
    """
    if file_num is None:
        if 'file_num' not in config: config['file_num'] = 0
        if 'start_obj_num' not in config: config['start_obj_num'] = obj_num
    else:
        config['file_num'] = file_num
        config['start_obj_num'] = obj_num
    config['image_num'] = image_num
    config['obj_num'] = obj_num
    config['index_key'] = 'file_num'
    if 'output' not in config: config['output'] = {}


def SetDefaultExt(config, ext):
    """Set a default ext if appropriate"""
    if ext is not None:
        if ( isinstance(config,dict) and 'type' in config and 
            config['type'] == 'NumberedFile' and 'ext' not in config ):
            config['ext'] = ext


def GetFilename(config, base, default_ext=None):
    """Get the file_name for the current file being worked on.
    """
    if 'file_name' in config:
        SetDefaultExt(config['file_name'],default_ext)
        file_name = galsim.config.ParseValue(config, 'file_name', base, str)[0]
    elif 'root' in config and default_ext is not None:
        # If a file_name isn't specified, we use the name of the config file + '.fits'
        file_name = config['root'] + default_ext
    else:
        raise AttributeError("No file_name specified and unable to generate it automatically.")

    # Prepend a dir to the beginning of the filename if requested.
    if 'dir' in config:
        dir = galsim.config.ParseValue(config, 'dir', base, str)[0]
        if dir and not os.path.isdir(dir): os.makedirs(dir)
        file_name = os.path.join(dir,file_name)

    return file_name


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
                if logger and logger.isEnabledFor(logging.WARN):
                    logger.warn('File %s: Caught IOError: %s',file_name,str(e))
                    logger.warn('This is try %d/%d, so sleep for %d sec and try again.',
                                itry+1,ntries,itry+1)
                import time
                time.sleep(itry+1)
                continue
        else:
            break
    return ret


def BuildFits(config, file_num, image_num, obj_num, nproc, ignore, logger):
    """
    Build a regular fits file as specified in config.
    
    @param config           A configuration dict.
    @param file_num         The current file_num.
    @param image_num        The current image_num.
    @param obj_num          The current obj_num.
    @param nproc            How many processes to use. (ignored in this case)
    @param ignore           A list of parameters that are allowed to be in config['output']
                            that we can ignore here.  i.e. it won't be an error if these
                            parameters are present.
    @param logger           If given, a logger object to log progress.

    @returns the image in a list with one item: [ image ]
    """
    # There are no extra parameters to get, so just check that there are no invalid parameters
    # in the config dict.
    galsim.config.CheckAllParams(config['output'], ignore=ignore)

    image = galsim.config.BuildImage(config, logger=logger, image_num=image_num, obj_num=obj_num)
    return [ image ]

def GetNImagesFits(config, file_num):
    """
    Get the number of images for a Fits file type.  i.e. 1.

    @param config           The configuration dict.
    @param file_num         The current file number.

    @returns 1
    """
    return 1

valid_output_types = {}

def RegisterOutputType(output_type, build_func, write_func, nimages_func,
                       extra_hdus=False, default_ext='.fits'):
    """Register an output type for use by the config apparatus.

    @param output_type      The name of the type in config['output']
    @param build_func       The function to call for building the necessary data.
                            The call signature is:
                                data = Build(config, file_num, image_num, obj_num, nproc,
                                             ignore, logger)
    @param write_func       The function to use for writing the data to a file.
                            The call signature is:
                                Write(data, file_name)
    @param nimages_func     A function that returns the number of images that will be built.
                            The call signature is 
                                nimages = GetNImages(config, file_num)
    @param extra_hdus       Whether extra hdus can be added to the end of the data for the extra
                            output items. [default: False]
    @param default_ext      The default file extension if none is given. [default: '.fits']
    """
    valid_output_types[output_type] = {
        'build' : build_func,
        'write' : write_func,
        'nim' : nimages_func,
        'hdus' : extra_hdus,
        'ext' : default_ext
    }

RegisterOutputType('Fits', BuildFits, galsim.fits.writeMulti, GetNImagesFits, extra_hdus=True)

