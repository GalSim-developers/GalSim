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

from .output_datacube import *
from .output_multifits import *

valid_output_types = { 
    # The values are tuples with:
    # - the build function to call
    # - a function that merely counts the number of objects that will be built by the function
    # - whether the Builder takes nproc.
    # - whether the Builder takes psf_file_name, weight_file_name, and badpix_file_name.
    # - whether the Builder takes psf_hdu, weight_hdu, and badpix_hdu.
    # See the des module for examples of how to extend this from a module.
    'Fits' : ('BuildFits', 'GetNImagesFits', False, True, True),
    'MultiFits' : ('BuildMultiFits', 'GetNImagesMultiFits', True, True, False),
    'DataCube' : ('BuildDataCube', 'GetNImagesDataCube', True, True, False),
}


def BuildFiles(nfiles, config, file_num=0, nproc=1, logger=None):
    """
    Build a number of output files as specified in config.
    
    @param nfiles           The number of files to build.
    @param config           A configuration dict.
    @param file_num         If given, the first file_num. [default: 0]
    @param nproc            How many processes to use for building the images. [default: 1]
    @param logger           If given, a logger object to log progress. [default: None]
    """
    import time
    t1 = time.time()

    # If only 1 file, then pass the nproc option on to the BuildImages function.
    if nfiles == 1:
        nproc_image = nproc
        nproc = 1
    else:
        nproc_image = 1
        nproc = galsim.config.UpdateNProc(nproc, nfiles, config, logger)

    output = config['output']
    if not isinstance(output, dict):
        raise AttributeError("config.output is not a dict.")

    # Get the output type.  Default = Fits
    if 'type' not in output:
        output['type'] = 'Fits' 
    type = output['type']

    # Check that the type is valid
    if type not in valid_output_types:
        raise AttributeError("Invalid output.type=%s."%type)

    # build_func is the function we'll call to build each file.
    build_func = eval(valid_output_types[type][0])

    # can_do_multiple says whether the function can in principal do multiple files
    can_do_multiple = valid_output_types[type][2]

    # extra_file_name says whether the function takes psf_file_name, etc.
    extra_file_name = valid_output_types[type][3]

    # extra_hdu says whether the function takes psf_hdu, etc.
    extra_hdu = valid_output_types[type][4]
    if logger and logger.isEnabledFor(logging.DEBUG):
        logger.debug('type = %s',type)
        logger.debug('extra_file_name = %s',extra_file_name)
        logger.debug('extra_hdu = %d',extra_hdu)

    extra_keys = [ 'psf', 'weight', 'badpix' ]
    last_file_name = {}
    for key in extra_keys:
        last_file_name[key] = None

    def worker(input, output):
        proc = current_process().name
        for job in iter(input.get, 'STOP'):
            try:
                (kwargs, file_num, file_name, logger) = job
                if logger and logger.isEnabledFor(logging.DEBUG):
                    logger.debug('%s: Received job to do file %d, %s',proc,file_num,file_name)
                galsim.config.ProcessInput(kwargs['config'], file_num=file_num, logger=logger)
                if logger and logger.isEnabledFor(logging.DEBUG):
                    logger.debug('%s: After ProcessInput for file %d',proc,file_num)
                kwargs['logger'] = logger
                t = build_func(**kwargs)
                if logger and logger.isEnabledFor(logging.DEBUG):
                    logger.debug('%s: After %s for file %d',proc,build_func,file_num)
                output.put( (t, file_num, file_name, proc) )
            except Exception as e:
                import traceback
                tr = traceback.format_exc()
                if logger and logger.isEnabledFor(logging.WARN):
                    logger.warn('%s: Caught exception %s\n%s',proc,str(e),tr)
                output.put( (e, file_num, file_name, tr) )

    # Set up the multi-process task_queue if we're going to need it.
    if nproc > 1:
        # NB: See the function BuildStamps for more verbose comments about how
        # the multiprocessing stuff works.
        from multiprocessing import Process, Queue, current_process
        task_queue = Queue()

    # Get a proxy of the logger to send to the task_queue.
    logger_proxy = galsim.config.GetLoggerProxy(logger)

    # Process the input field for the first file.  Often there are "safe" input items
    # that won't need to be reprocessed each time.  So do them here once and keep them
    # in the config for all file_nums.  This is more important if nproc != 1.
    galsim.config.ProcessInput(config, file_num=0, logger=logger, safe_only=True)

    # We'll want a pristine version later to give to the workers.
    orig_config = galsim.config.CopyConfig(config)

    first_file_num = file_num
    image_num = 0
    obj_num = 0

    nfiles_use = nfiles
    for file_num in range(first_file_num, first_file_num+nfiles):
        config['index_key'] = 'file_num'
        config['file_num'] = file_num
        config['image_num'] = image_num
        config['start_obj_num'] = obj_num
        config['obj_num'] = obj_num

        # Process the input fields that might be relevant at file scope:
        galsim.config.ProcessInput(config, file_num=file_num, logger=logger_proxy,
                                   file_scope_only=True)

        # The kwargs to pass to BuildFile
        kwargs = {
            'file_num' : file_num,
            'image_num' : image_num,
            'obj_num' : obj_num
        }

        if can_do_multiple:
            kwargs['nproc'] = nproc_image

        nobj = GetNObjForFile(config,file_num,image_num)
        if logger and logger.isEnabledFor(logging.DEBUG):
            logger.debug('file %d: nobj = %s',file_num,str(nobj))

        file_name = GetFilename(output, config, '.fits')
        kwargs['file_name'] = file_name

        # nobj is a list of nobj for each image in that file.
        # So len(nobj) = nimages and sum(nobj) is the total number of objects
        # This gets the values of image_num and obj_num ready for the next loop.
        image_num += len(nobj)
        obj_num += sum(nobj)

        # Check if we ought to skip this file
        if ('skip' in output 
                and galsim.config.ParseValue(output, 'skip', config, bool)[0]):
            if logger and logger.isEnabledFor(logging.WARN):
                logger.warn('Skipping file %d = %s because output.skip = True',file_num,file_name)
            nfiles_use -= 1
            continue
        if ('noclobber' in output 
                and galsim.config.ParseValue(output, 'noclobber', config, bool)[0]
                and os.path.isfile(file_name)):
            if logger and logger.isEnabledFor(logging.WARN):
                logger.warn('Skipping file %d = %s because output.noclobber = True' +
                            ' and file exists',file_num,file_name)
            nfiles_use -= 1
            continue

        if 'dir' in output:
            default_dir = galsim.config.ParseValue(output,'dir',config,str)[0]
        else:
            default_dir = None

        # Check if we need to build extra images for write out as well
        for extra_key in [ key for key in extra_keys if key in output ]:
            if logger and logger.isEnabledFor(logging.DEBUG):
                logger.debug('extra_key = %s',extra_key)
            output_extra = output[extra_key]

            output_extra['type'] = 'default'
            req = {}
            single = []
            opt = {}
            ignore = []
            if extra_file_name and extra_hdu:
                single += [ { 'file_name' : str, 'hdu' : int } ]
                opt['dir'] = str
            elif extra_file_name:
                req['file_name'] = str
                opt['dir'] = str
            elif extra_hdu:
                req['hdu'] = int

            if extra_key == 'psf': 
                ignore += ['draw_method', 'signal_to_noise']
            if extra_key == 'weight': 
                ignore += ['include_obj_var']
            if 'file_name' in output_extra:
                SetDefaultExt(output_extra['file_name'],'.fits')
            params, safe = galsim.config.GetAllParams(output_extra,config,
                                                      req=req, opt=opt, single=single,
                                                      ignore=ignore)

            if 'file_name' in params:
                f = params['file_name']
                if 'dir' in params:
                    dir = params['dir']
                    if dir and not os.path.isdir(dir): os.makedirs(dir)
                # else use default dir from above.
                if default_dir:
                    f = os.path.join(default_dir,f)
                # If we already wrote this file, skip it this time around.
                # (Typically this is applicable for psf, where we may only want 1 psf file.)
                if last_file_name[key] == f:
                    if logger and logger.isEnabledFor(logging.WARN):
                        logger.warn('skipping %s, since already written',f)
                    continue
                kwargs[ extra_key+'_file_name' ] = f
                last_file_name[key] = f
            elif 'hdu' in params:
                kwargs[ extra_key+'_hdu' ] = params['hdu']

        # This is where we actually build the file.
        # If we're doing multiprocessing, we send this information off to the task_queue.
        # Otherwise, we just call build_func.
        if nproc > 1:
            import copy
            # Make new copies of config and kwargs so we can update them without
            # clobbering the versions for other tasks on the queue.
            kwargs1 = copy.copy(kwargs)
            # Clear out unsafe proxy objects, since there seems to be a bug in the manager
            # package where this can cause strange KeyError exceptions in the incref function.
            # It seems to be related to having multiple identical proxy objects that then
            # get deleted.  e.g. if the first N files use one dict, then the next N use another,
            # and so forth.  I don't really get it, but clearing them out here seems to 
            # fix the problem.
            galsim.config.ProcessInput(config, file_num=file_num, logger=logger_proxy, safe_only=True)
            kwargs1['config'] = galsim.config.CopyConfig(config)
            task_queue.put( (kwargs1, file_num, file_name, logger_proxy) )
        else:
            try:
                if logger and logger.isEnabledFor(logging.WARN):
                    logger.warn('Start file %d = %s', file_num, file_name)
                galsim.config.ProcessInput(config, file_num=file_num, logger=logger_proxy)
                if logger and logger.isEnabledFor(logging.DEBUG):
                    logger.debug('file %d: After ProcessInput',file_num)
                kwargs['config'] = config
                kwargs['logger'] = logger 
                t = build_func(**kwargs)
                if logger and logger.isEnabledFor(logging.WARN):
                    logger.warn('File %d = %s, time = %f sec', file_num, file_name, t)
            except Exception as e:
                import traceback
                tr = traceback.format_exc()
                if logger:
                    logger.error('Exception caught for file %d = %s', file_num, file_name)
                    logger.error('%s',tr)
                    logger.error('%s',e)
                    logger.error('File %s not written! Continuing on...',file_name)

    # If we're doing multiprocessing, here is the machinery to run through the task_queue
    # and process the results.
    if nproc > 1:
        if logger and logger.isEnabledFor(logging.WARN):
            logger.warn("Using %d processes",nproc)
        import time
        t1 = time.time()
        # Run the tasks
        done_queue = Queue()
        p_list = []
        for j in range(nproc):
            p = Process(target=worker, args=(task_queue, done_queue), name='Process-%d'%(j+1))
            p.start()
            p_list.append(p)

        # Log the results.
        if logger and logger.isEnabledFor(logging.DEBUG):
            logger.debug('nfiles_use = %d',nfiles_use)
        for k in range(nfiles_use):
            t, file_num, file_name, proc = done_queue.get()
            if isinstance(t,Exception):
                # t is really the exception, e
                # proc is really the traceback
                if logger:
                    logger.error('Exception caught for file %d = %s', file_num, file_name)
                    logger.error('%s',proc)
                    logger.error('%s',t)
                    logger.error('File %s not written! Continuing on...',file_name)
            else:
                if logger and logger.isEnabledFor(logging.WARN):
                    logger.warn('%s: File %d = %s: time = %f sec', proc, file_num, file_name, t)

        # Stop the processes
        for j in range(nproc):
            task_queue.put('STOP')
        for j in range(nproc):
            p_list[j].join()
        task_queue.close()
        t2 = time.time()
        if logger and logger.isEnabledFor(logging.WARN):
            logger.warn('Total time for %d files with %d processes = %f sec', 
                        nfiles_use,nproc,t2-t1)

    if logger and logger.isEnabledFor(logging.WARN):
        logger.warn('Done building files')


output_ignore = [ 'file_name', 'dir', 'nfiles', 'nproc', 'skip', 'noclobber', 'retry_io' ]


def BuildFits(file_name, config, logger=None, 
              file_num=0, image_num=0, obj_num=0, 
              psf_file_name=None, psf_hdu=None,
              weight_file_name=None, weight_hdu=None,
              badpix_file_name=None, badpix_hdu=None):
    """
    Build a regular fits file as specified in config.
    
    @param file_name        The name of the output file.
    @param config           A configuration dict.
    @param logger           If given, a logger object to log progress. [default: None]
    @param file_num         If given, the current file_num. [default: 0]
    @param image_num        If given, the current image_num. [default: 0]
    @param obj_num          If given, the current obj_num. [default: 0]
    @param psf_file_name    If given, write a psf image to this file. [default: None]
    @param psf_hdu          If given, write a psf image to this hdu in file_name. [default: None]
    @param weight_file_name If given, write a weight image to this file. [default: None]
    @param weight_hdu       If given, write a weight image to this hdu in file_name.  [default: 
                            None]
    @param badpix_file_name If given, write a badpix image to this file. [default: None]
    @param badpix_hdu       If given, write a badpix image to this hdu in file_name. [default:
                            None]

    @returns the time taken to build file.
    """
    import time
    t1 = time.time()

    SetupConfigFileNum(config,file_num,image_num,obj_num)
    seed = galsim.config.SetupConfigRNG(config)
    if logger and logger.isEnabledFor(logging.DEBUG):
        logger.debug('file %d: seed = %d',file_num,seed)

    # The GetNImagesForFile function performs some basic setup, so in the interest of avoiding
    # code duplication, call it here, even though we don't really need the output.  (Although,
    # we do got ahead and use it in the debug logging, since we have it.)
    nimages = GetNImagesForFile(config, file_num)

    output = config['output']
    output_type = output['type']

    if logger and logger.isEnabledFor(logging.DEBUG):
        logger.debug('file %d: BuildFits for %s: file, image, obj = %d,%d,%d',
                      file_num,output_type,nimages,image_num)

    # hdus is a dict with hdus[i] = the item in all_images to put in the i-th hdu.
    hdus = {}
    # The primary hdu is always the main image.
    hdus[0] = 0

    if psf_file_name or psf_hdu:
        make_psf_image = True
        if psf_hdu: 
            if psf_hdu <= 0 or psf_hdu in hdus.keys():
                raise ValueError("psf_hdu = %d is invalid or a duplicate."%pdf_hdu)
            hdus[psf_hdu] = 1
    else:
        make_psf_image = False

    if weight_file_name or weight_hdu:
        make_weight_image = True
        if weight_hdu: 
            if weight_hdu <= 0 or weight_hdu in hdus.keys():
                raise ValueError("weight_hdu = %d is invalid or a duplicate."&weight_hdu)
            hdus[weight_hdu] = 2
    else:
        make_weight_image = False

    if badpix_file_name or badpix_hdu:
        make_badpix_image = True
        if badpix_hdu: 
            if badpix_hdu <= 0 or badpix_hdu in hdus.keys():
                raise ValueError("badpix_hdu = %d is invalid or a duplicate."&badpix_hdu)
            hdus[badpix_hdu] = 3
    else:
        make_badpix_image = False

    for h in range(len(hdus.keys())):
        if h not in hdus.keys():
            raise ValueError("Image for hdu %d not found.  Cannot skip hdus."%h)

    all_images = galsim.config.BuildImage(
            config=config, logger=logger, image_num=image_num, obj_num=obj_num,
            make_psf_image=make_psf_image,
            make_weight_image=make_weight_image,
            make_badpix_image=make_badpix_image)
    # returns a tuple ( main_image, psf_image, weight_image, badpix_image )

    hdulist = []
    for h in range(len(hdus.keys())):
        assert h in hdus.keys()  # Checked for this above.
        hdulist.append(all_images[hdus[h]])
    # We can use hdulist in writeMulti even if the main image is the only one in the list.

    if 'output' in config and 'retry_io' in config['output']:
        ntries = galsim.config.ParseValue(config['output'],'retry_io',config,int)[0]
        # This is how many _re_-tries.  Do at least 1, so ntries is 1 more than this.
        ntries = ntries + 1
    else:
        ntries = 1

    RetryIO(galsim.fits.writeMulti, (hdulist, file_name), ntries, file_name, logger)
    if logger and logger.isEnabledFor(logging.DEBUG):
        if len(hdus.keys()) == 1:
            logger.debug('file %d: Wrote image to fits file %r',
                         config['file_num'],file_name)
        else:
            logger.debug('file %d: Wrote image (with extra hdus) to multi-extension fits file %r',
                         config['file_num'],file_name)

    if psf_file_name:
        RetryIO(galsim.fits.write, (all_images[1], psf_file_name),
                ntries, psf_file_name, logger)
        if logger and logger.isEnabledFor(logging.DEBUG):
            logger.debug('file %d: Wrote psf image to fits file %r',
                         config['file_num'],psf_file_name)

    if weight_file_name:
        RetryIO(galsim.fits.write, (all_images[2], weight_file_name),
                  ntries, weight_file_name, logger)
        if logger and logger.isEnabledFor(logging.DEBUG):
            logger.debug('file %d: Wrote weight image to fits file %r',
                         config['file_num'],weight_file_name)

    if badpix_file_name:
        RetryIO(galsim.fits.write, (all_images[3], badpix_file_name),
                  ntries, badpix_file_name, logger)
        if logger and logger.isEnabledFor(logging.DEBUG):
            logger.debug('file %d: Wrote badpix image to fits file %r',
                         config['file_num'],badpix_file_name)

    t2 = time.time()
    return t2-t1


def GetNImagesForFile(config, file_num):
    """
    Get the number of images that will be made for the file number file_num, based on the
    information in the config dict.
    @param config           The configuration dict.
    @param file_num         The current file number.
    @returns the number of images
    """
    if 'output' not in config:
        config['output'] = {}
    if 'type' not in config['output']:
        config['output']['type'] = 'Fits'
    output_type = config['output']['type']

    # Check that the type is valid
    if output_type not in valid_output_types:
        raise AttributeError("Invalid output.type=%s."%output_type)

    # These might be required for nimages
    config['index_key'] = 'file_num'
    config['file_num'] = file_num
    seed = galsim.config.SetupConfigRNG(config)

    nim_func = eval(valid_output_types[output_type][1])
 
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
    elif 'root' in base and default_ext is not None:
        # If a file_name isn't specified, we use the name of the config file + '.fits'
        file_name = base['root'] + default_ext
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

def GetNImagesFits(config, file_num):
    """
    Get the number of images for a Fits file type.  i.e. 1.
    @param config           The configuration dict.
    @param file_num         The current file number.
    @returns 1
    """
    return 1

