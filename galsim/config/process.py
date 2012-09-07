
import sys
import os
import subprocess
import logging
import time
import copy

import galsim


def Process(config, logger=None):
    """
    Do all processing of the provided configuration dict
    """

    # If we don't have a root specified yet, we generate it from the current script.
    if 'root' not in config:
        import inspect
        script_name = os.path.basename(
            inspect.getfile(inspect.currentframe())) # script filename (usually with path)
        # Strip off a final suffix if present.
        config['root'] = os.path.splitext(script_name)[0]

    ProcessInput(config, logger)
    ProcessOutput(config, logger)


def ProcessInput(config, logger=None):
    """
    Process the input field, reading in any specified input files.
    These files are saved in the top level of config.

    config['catalog'] = the catalog specified by config.input.catalog
    config['real_catalog'] = the catalog specified by config.input.real_catalog
    """

    # Process the input field (read any necessary input files)
    if 'input' in config:
        input = config['input']
        if not isinstance(input, dict):
            raise AttributeError("config.input is not a dict.")

        # Read the input catalogs if provided
        cat_type = { 'catalog' : 'InputCatalog' , 
                     'real_catalog' : 'RealGalaxyCatalog' }
        for key in [ k for k in cat_type.keys() if k in input ]:
            catalog = input[key]
            catalog['type'] = cat_type[key]
            input_cat = galsim.config.gsobject._BuildSimple(catalog, key, config, {})[0]
            if logger:
                logger.info('Read %d objects from %s',input_cat.nobjects,key)
            # Store input_cat in the config for use by BuildGSObject function.
            config[key] = input_cat

        # Check that there are no other attributes specified.
        galsim.config.CheckAllParams(input, 'input', ignore=cat_type.keys())


def ProcessOutput(config, logger=None):
    """
    Process the output field, building and writing all the specified image files.
    """

    # Make config['output'] exist if it doesn't yet.
    if 'output' not in config:
        config['output'] = {}
    output = config['output']
    if not isinstance(output, dict):
        raise AttributeError("config.output is not a dict.")

    type = output.get('type','Fits') # Default is Fits

    # If (1) type is MultiFits or DataCube,
    #    (2) the image type is Single, and
    #    (3) there is an input catalog,
    # then we don't require the nimages to be set in advance.
    # We let nimages default to input_cat.nobjects
    if ( ( type == 'MultiFits' or type == 'DataCube' ) and
         'nimages' not in output and
         ( 'image' not in config or 'type' not in config['image'] or 
           config['image']['type'] == 'Single' ) and
         'catalog' in config ):
        output['nimages'] = config['catalog'].nobjects

    # Also, if (1) type is Fits,
    #          (2) the image type is Scattered, and
    #          (3) there is an input catalog
    # we do the same thing for image['nobjects']
    if ( type == 'Fits' and 
         'image' in config and 'type' in config['image'] and
         config['image']['type'] == 'Scattered' and
         'nobjects' not in config['image'] and
         'catalog' in config ):
        config['image']['nobjects'] = config['catalog'].nobjects

    ignore = [ 'file_name', 'dir', 'nfiles', 'psf', 'weight', 'badpix', 'nproc' ]
    if type == 'Fits':
        build_func = BuildFits
        galsim.config.CheckAllParams(output, 'output', ignore=ignore)

    elif type == 'MultiFits':
        build_func = BuildMultiFits
        req = { 'nimages' : int }
        params = galsim.config.CheckAllParams(output, 'output', req=req, ignore=ignore)

    elif type == 'DataCube':
        build_func = BuildDataCube
        req = { 'nimages' : int }
        params = galsim.config.CheckAllParams(output, 'output', req=req, ignore=ignore)

    else:
        raise AttributeError("Invalid output.type=%s."%type)
 
    # The kwargs to pass to build_func
    # We'll be building this up as we go...
    kwargs = {}

    if 'nfiles' in output:
        nfiles = galsim.config.ParseValue(output, 'nfiles', config, int)[0]
    else:
        nfiles = 1 

    if 'nproc' in output:
        nproc = galsim.config.ParseValue(output, 'nproc', config, int)[0]
    else:
        nproc = 1 

    if nproc > nfiles:
        if nfiles == 1 and (type == 'MultiFits' or type == 'DataCube'):
            kwargs['nproc'] = nproc 
        else:
            import warnings
            warnings.warn(
                "Trying to use more processes than files: output.nproc=%d, "%nproc +
                "output.nfiles=%d"%nfiles)
            nproc = nfiles
    if nproc <= 0:
        # Try to figure out a good number of processes to use
        try:
            from multiprocessing import cpu_count
            ncpu = cpu_count()
            if nfiles == 1 and (type == 'MultiFits' or type == 'DataCube'):
                kwargs['nproc'] = ncpu 
                if logger:
                    logger.info("ncpu = %d.",ncpu)
            else:
                if ncpu > nfiles:
                    nproc = ncpu
                else:
                    nproc = nfiles
                if logger:
                    logger.info("ncpu = %d.  Using %d processes",ncpu,nproc)
        except:
            raise AttributeError(
                "config.nprof <= 0, but unable to determine number of cpus.")
    
    if nproc > 1:
        # NB: See the function BuildStamps for more verbose comments about how
        # the multiprocessing stuff works.
        from multiprocessing import Process, Queue, current_process

        def worker(input, output):
            """
            input is a queue with (args, info) tuples:
                kwargs are the arguments to pass to build_func
                info is passed along to the output queue.
            output is a queue storing (result, info, proc) tuples:
                result is the returned value from build_func (file_name, time).
                info is passed through from the input queue.
                proc is the process name.
            """
            for (kwargs, info) in iter(input.get, 'STOP'):
                result = build_func(**kwargs)
                output.put( (result, info, current_process().name) )

        # Set up the task list
        task_queue = Queue()

    for k in range(nfiles):
        # Get the file_name
        if 'file_name' in output:
            file_name = galsim.config.ParseValue(output, 'file_name', config, str)[0]
        elif 'root' in config:
            # If a file_name isn't specified, we use the name of the config file + '.fits'
            file_name = config['root'] + '.fits'
        else:
            raise AttributeError(
                "No output.file_name specified and unable to generate it automatically.")
        
        # Prepend a dir to the beginning of the filename if requested.
        if 'dir' in output:
            dir = galsim.config.ParseValue(output, 'dir', config, str)[0]
            if not os.path.isdir(dir):
                os.mkdir(output['dir'])
            file_name = os.path.join(dir,file_name)

        kwargs['file_name'] = file_name
        kwargs['config'] = config

        if type == 'MultiFits' or type == 'DataCube':
            nimages = galsim.config.ParseValue(output, 'nimages', config, int)[0]
            kwargs['nimages'] = nimages 

        for extra in [ k for k in [ 'psf' , 'weight', 'badpix' ] if k in output ]:
            extra_file_name = None
            output_extra = output[extra]
            if 'file_name' in output_extra:
                extra_file_name = output_extra['file_name']
                if 'dir' in output:
                    extra_file_name = os.path.join(output['dir'],extra_file_name)
                kwargs[ extra+'_file_name' ] = extra_file_name
            elif type == 'MultiFits':
                raise AttributeError(
                    "Only the file_name version of %s output is possible for "%extra+
                    "output type == MultiFits.")
            else:
                raise NotImplementedError(
                    "Only the file_name version of %s output is currently implemented."%extra)
    
        if nproc > 1:
            new_kwargs = {}
            new_kwargs.update(kwargs)
            task_queue.put( new_kwargs, file_name )
        else:
            # Apparently the logger isn't picklable, so can't send that for nproc > 1
            kwargs['logger'] = logger 
            t = build_func(**kwargs)
            if logger:
                logger.info('Built file %s: total time = %f sec', file_name, t)

    if nproc > 1:
        # Run the tasks
        done_queue = Queue()
        for j in range(nproc):
            Process(target=worker, args=(task_queue, done_queue)).start()

        # Log the results.
        for i in range(nimages):
            t, file_name, proc = done_queue.get()
            if logger:
                logger.info('%s: File %s: total time = %f sec', proc, file_name, t)

        # Stop the processes
        for j in range(nproc):
            task_queue.put('STOP')

    if logger:
        logger.info('Done building files')


def BuildFits(file_name, config, logger=None,
              psf_file_name=None, psf_hdu=None,
              weight_file_name=None, weight_hdu=None,
              badpix_file_name=None, badpix_hdu=None):
    """
    Build a regular fits file as specified in config.
    
    @param file_name  The name of the output file.
    @param config     A configuration dict.
    @param base       The base configuration dict.  Normally config = base['output']
    @param logger     If given, a logger object to log progress.
    @param psf_file_name     If given, write a psf image to this file
    @param psf_hdu           If given, write a psf image to this hdu in file_name
    @param weight_file_name  If given, write a weight image to this file
    @param weight_hdu        If given, write a weight image to this hdu in file_name
    @param badpix_file_name  If given, write a badpix image to this file
    @param badpix_hdu        If given, write a badpix image to this hdu in file_name

    @return time      Time taken to build filek
    """
    t1 = time.time()

    if psf_file_name:
        make_psf_image = True
    elif psf_hdu:
        raise NotImplementedError("Sorry, psf hdu output is not currently implemented.")
    else:
        make_psf_image = False

    if weight_file_name or weight_hdu:
        raise NotImplementedError("Sorry, weight image output is not currently implemented.")
    if badpix_file_name or badpix_hdu:
        raise NotImplementedError("Sorry, badpix image output is not currently implemented.")

    all_images = BuildImage(
            config=config, logger=logger, 
            make_psf_image=make_psf_image,
            make_weight_image=False, 
            make_badpix_image=False)
    # returns a tuple ( main_image, psf_image, weight_image, badpix_image )

    all_images[0].write(file_name, clobber=True)
    if logger:
        logger.info('Wrote image to fits file %r',file_name)

    if psf_file_name:
        all_images[1].write(psf_file_name, clobber=True)
        if logger:
            logger.info('Wrote psf image to fits file %r',psf_file_name)

    t2 = time.time()
    return t2-t1


def BuildMultiFits(file_name, nimages, config, nproc=1, logger=None,
                   psf_file_name=None, weight_file_name=None, badpix_file_name=None):
    """
    Build a regular fits file as specified in config.
    
    @param file_name  The name of the output file.
    @param nimages    The number of images (and hence hdus in the output file)
    @param config     A configuration dict.
    @param nproc      How many processes to use.
    @param logger     If given, a logger object to log progress.
    @param psf_file_name     If given, write a psf image to this file
    @param weight_file_name  If given, write a weight image to this file
    @param badpix_file_name  If given, write a badpix image to this file

    @return time      Time taken to build filek
    """
    t1 = time.time()

    if psf_file_name:
        make_psf_image = True
    else:
        make_psf_image = False
    if weight_file_name:
        raise NotImplementedError("Sorry, weight image output is not currently implemented.")
    if badpix_file_name:
        raise NotImplementedError("Sorry, badpix image output is not currently implemented.")
    if nproc > 1:
        import warnings
        warnings.warn("Sorry, multiple processes not currently implemented for BuildMultiFits.")
        

    main_images = []
    psf_images = []
    weight_images = []
    badpix_images = []
    for k in range(nimages):
        t2 = time.time()
        all_images = BuildImage(
                config=config, logger=logger,
                make_psf_image=make_psf_image, 
                make_weight_image=False,
                make_badpix_image=False)
        # returns a tuple ( main_image, psf_image, weight_image, badpix_image )
        t3 = time.time()
        main_images += [ all_images[0] ]
        psf_images += [ all_images[1] ]
        weight_images += [ all_images[2] ]
        badpix_images += [ all_images[3] ]

        if logger:
            # Note: numpy shape is y,x
            ys, xs = all_images[0].array.shape
            logger.info('Image %d: size = %d x %d, time = %f sec', k, xs, ys, t3-t2)


    galsim.fits.writeMulti(main_images, file_name, clobber=True)
    if logger:
        logger.info('Wrote images to multi-extension fits file %r',file_name)

    if psf_file_name:
        galsim.fits.writeMulti(psf_images, psf_file_name, clobber=True)
        if logger:
            logger.info('Wrote psf images to multi-extension fits file %r',psf_file_name)

    t4 = time.time()
    return t4-t1


def BuildDataCube(file_name, nimages, config, nproc=1, logger=None,
                  psf_file_name=None, psf_hdu=None,
                  weight_file_name=None, weight_hdu=None,
                  badpix_file_name=None, badpix_hdu=None):
    """
    Build a regular fits file as specified in config.
    
    @param file_name  The name of the output file.
    @param nimages    The number of images in the data cube
    @param config     A configuration dict.
    @param nproc      How many processes to use.
    @param logger     If given, a logger object to log progress.
    @param psf_file_name     If given, write a psf image to this file
    @param psf_hdu           If given, write a psf image to this hdu in file_name
    @param weight_file_name  If given, write a weight image to this file
    @param weight_hdu        If given, write a weight image to this hdu in file_name
    @param badpix_file_name  If given, write a badpix image to this file
    @param badpix_hdu        If given, write a badpix image to this hdu in file_name

    @return time      Time taken to build file
    """
    t1 = time.time()

    if psf_file_name:
        make_psf_image = True
    elif psf_hdu:
        raise NotImplementedError("Sorry, psf hdu output is not currently implemented.")
    else:
        make_psf_image = False
    if weight_file_name or weight_hdu:
        raise NotImplementedError("Sorry, weight image output is not currently implemented.")
    if badpix_file_name or badpix_hdu:
        raise NotImplementedError("Sorry, badpix image output is not currently implemented.")
    if nproc > 1:
        import warnings
        warnings.warn("Sorry, multiple processe not currently implemented for BuildMultiFits.")

    # All images need to be the same size for a data cube.
    # Enforce this by buliding the first image outside the below loop and setting
    # config['image_xsize'] and config['image_ysize'] to be the size of the first image.
    t2 = time.time()
    all_images = BuildImage(
            config=config, logger=logger,
            make_psf_image=make_psf_image, 
            make_weight_image=False,
            make_badpix_image=False)
    t3 = time.time()
    if logger:
        # Note: numpy shape is y,x
        ys, xs = all_images[0].array.shape
        logger.info('Image 0: size = %d x %d, time = %f sec', xs, ys, t3-t2)

    # Note: numpy shape is y,x
    image_ysize, image_xsize = all_images[0].array.shape
    config['image_xsize'] = image_xsize
    config['image_ysize'] = image_ysize

    main_images = [ all_images[0] ]
    psf_images = [ all_images[1] ]
    weight_images = [ all_images[2] ]
    badpix_images = [ all_images[3] ]

    for k in range(1,nimages):
        t4 = time.time()
        all_images = BuildImage(
                config=config, logger=logger,
                make_psf_image=make_psf_image, 
                make_weight_image=False,
                make_badpix_image=False)
        t5 = time.time()
        main_images += [ all_images[0] ]
        psf_images += [ all_images[1] ]
        weight_images += [ all_images[2] ]
        badpix_images += [ all_images[3] ]
        if logger:
            # Note: numpy shape is y,x
            ys, xs = all_images[0].array.shape
            logger.info('Image %d: size = %d x %d, time = %f sec', k, xs, ys, t5-t4)

    galsim.fits.writeCube(main_images, file_name, clobber=True)
    if logger:
        logger.info('Wrote image to fits data cube %r',file_name)

    if psf_file_name:
        galsim.fits.writeCube(psf_images, psf_file_name, clobber=True)
        if logger:
            logger.info('Wrote psf images to fits data cube %r',psf_file_name)

    t6 = time.time()
    return t6-t1


def BuildImage(config, logger=None,
               make_psf_image=False, make_weight_image=False, make_badpix_image=False):
    """
    Build an image according the information in config.

    @param config     A configuration dict.
    @param logger     If given, a logger object to log progress.
    @param make_psf_image      Whether to make psf_image
    @param make_weight_image   Whether to make weight_image
    @param make_badpix_image   Whether to make badpix_image

    @return (image, psf_image, weight_image, badpix_image)  

    Note: All 4 images are always returned in the return tuple,
          but the latter 3 might be None depending on the parameters make_*_image.
    """

    # Make config['image'] exist if it doesn't yet.
    if 'image' not in config:
        config['image'] = {}
    image = config['image']
    if not isinstance(image, dict):
        raise AttributeError("config.image is not a dict.")

    # Normally, random_seed is just a number, which really means to use that number
    # for the first item and go up sequentially from there for each object.
    # However, we allow for random_seed to be a gettable parameter, so for the 
    # normal case, we just convert it into a Sequence.
    if 'random_seed' in image and not isinstance(image['random_seed'],dict):
        first_seed = galsim.config.ParseValue(image, 'random_seed', config, int)[0]
        image['random_seed'] = { 'type' : 'Sequence' , 'first' : first_seed }

    if 'draw_method' not in image:
        image['draw_method'] = 'fft'

    type = image.get('type','Single') # Default is Single

    valid_types = [ 'Single', 'Tiled', 'Scattered' ]
    if type not in valid_types:
        raise AttributeError("Invalue image.type=%s."%type)

    build_func = eval('Build' + type + 'Image')
    return build_func(
            config=config, logger=logger,
            make_psf_image=make_psf_image, 
            make_weight_image=False,
            make_badpix_image=False)


def BuildSingleImage(config, logger=None,
                     make_psf_image=False, make_weight_image=False, make_badpix_image=False):
    """
    Build an image consisting of a single stamp

    @param config     A configuration dict.
    @param logger     If given, a logger object to log progress.
    @param make_psf_image      Whether to make psf_image
    @param make_weight_image   Whether to make weight_image
    @param make_badpix_image   Whether to make badpix_image

    @return (image, psf_image, weight_image, badpix_image)  

    Note: All 4 images are always returned in the return tuple,
          but the latter 3 might be None depending on the parameters make_*_image.    
    """
    ignore = [ 'draw_method', 'noise', 'wcs', 'nproc' ]
    opt = { 'random_seed' : int , 'size' : int , 'xsize' : int , 'ysize' : int ,
            'pixel_scale' : float }
    params = galsim.config.GetAllParams(
        config['image'], 'image', config, opt=opt, ignore=ignore)[0]

    # If image_xsize and image_ysize were set in config, this overrides the read-in params.
    if 'image_xsize' in config and 'image_ysize' in config:
        xsize = config['image_xsize']
        ysize = config['image_ysize']
    else:
        size = params.get('size',0)
        xsize = params.get('xsize',size)
        ysize = params.get('ysize',size)

    if (xsize == 0) != (ysize == 0):
        raise AttributeError(
            "Both (or neither) of image.xsize and image.ysize need to be defined  and != 0.")

    if 'pix' not in config:
        pixel_scale = params.get('pixel_scale',1.0)
        config['pix'] = { 'type' : 'Pixel' , 'xw' : pixel_scale }

    if 'random_seed' in params:
        seed = params['random_seed']
    else:
        seed = None
    return BuildSingleStamp(
            seed=seed, config=config, xsize=xsize, ysize=ysize, 
            do_noise=True, logger=logger,
            make_psf_image=make_psf_image, 
            make_weight_image=make_weight_image,
            make_badpix_image=make_badpix_image)


def BuildTiledImage(config, logger=None,
                    make_psf_image=False, make_weight_image=False, make_badpix_image=False):
    """
    Build an image consisting of a tiled array of postage stamps

    @param config     A configuration dict.
    @param logger     If given, a logger object to log progress.
    @param make_psf_image      Whether to make psf_image
    @param make_weight_image   Whether to make weight_image
    @param make_badpix_image   Whether to make badpix_image

    @return (image, psf_image, weight_image, badpix_image)  

    Note: All 4 images are always returned in the return tuple,
          but the latter 3 might be None depending on the parameters make_*_image.    
    """
    ignore = [ 'random_seed', 'draw_method', 'noise', 'wcs', 'nproc' ]
    req = { 'nx_tiles' : int , 'ny_tiles' : int }
    opt = { 'stamp_size' : int , 'stamp_xsize' : int , 'stamp_ysize' : int ,
            'border' : int , 'xborder' : int , 'yborder' : int ,
            'pixel_scale' : float , 'nproc' : int }
    params = galsim.config.GetAllParams(
        config['image'], 'image', config, req=req, opt=opt, ignore=ignore)[0]

    nx_tiles = params['nx_tiles']
    ny_tiles = params['ny_tiles']
    nstamps = nx_tiles * ny_tiles

    stamp_size = params.get('stamp_size',0)
    stamp_xsize = params.get('stamp_xsize',stamp_size)
    stamp_ysize = params.get('stamp_ysize',stamp_size)

    if (stamp_xsize == 0) != (stamp_ysize == 0):
        raise AttributeError(
            "Both (or neither) of image.stamp_xsize and image.stamp_ysize need to be "+
            "defined and != 0.")

    border = params.get("border",0)
    xborder = params.get("xborder",border)
    yborder = params.get("yborder",border)

    do_noise = xborder >= 0 and yborder >= 0
    # TODO: Note: if one of these is < 0 and the other is > 0, then
    #       this will add noise to the border region.  Not exactly the 
    #       design, but I didn't bother to do the bookkeeping right to 
    #       make the borders pure 0 in that case.

    # If image_xsize and image_ysize were set in config, this overrides the read-in params.
    if 'image_xsize' in config and 'image_ysize' in config:
        stamp_xsize = (config['image_xsize']+xborder) / nx_tiles - xborder
        stamp_ysize = (config['image_ysize']+yborder) / ny_tiles - yborder
        full_xsize = (stamp_xsize + xborder) * nx_tiles - xborder
        full_ysize = (stamp_ysize + yborder) * ny_tiles - yborder
        if ( full_xsize != config['image_xsize'] or full_ysize != config['image_ysize'] ):
            raise ValueError(
                "Unable to reconcile saved image_xsize and image_ysize with current "+
                "nx_tiles=%d, ny_tiles=%d, "%(nx_tiles,ny_tiles) +
                "xborder=%d, yborder=%d\n"%(xborder,yborder) +
                "Calculated full_size = (%d,%d) "%(full_xsize,full_ysize)+
                "!= required (%d,%d)."%(config['image_xsize'],config['image_ysize']))

    if stamp_xsize == 0:
        if 'random_seed' in config['image']:
            seed = galsim.config.ParseValue(config['image'],'random_seed',config,int)[0]
        else:
            seed = None
        first_images = BuildSingleStamp(
            seed=seed, config=config, xsize=stamp_xsize, ysize=stamp_ysize, 
            do_noise=do_noise, logger=logger,
            make_psf_image=make_psf_image, 
            make_weight_image=make_weight_image,
            make_badpix_image=make_badpix_image)
        # Note: numpy shape is y,x
        stamp_ysize, stamp_xsize = first_images[0].array.shape
        images = [ first_images[0] ]
        psf_images = [ first_images[1] ]
        weight_images = [ first_images[2] ]
        badpix_images = [ first_images[3] ]
        nstamps -= 1
    else:
        images = []
        psf_images = []
        weight_images = []
        badpix_images = []

    full_xsize = (stamp_xsize + xborder) * nx_tiles - xborder
    full_ysize = (stamp_ysize + yborder) * ny_tiles - yborder

    pixel_scale = params.get('pixel_scale',1.0)
    if 'pix' not in config:
        config['pix'] = { 'type' : 'Pixel' , 'xw' : pixel_scale }

    nproc = params.get('nproc',1)

    full_image = galsim.ImageF(full_xsize,full_ysize)
    full_image.setZero()
    full_image.setScale(pixel_scale)

    if make_psf_image:
        full_psf_image = galsim.ImageF(full_xsize,full_ysize)
        full_psf_image.setZero()
        full_psf_image.setScale(pixel_scale)
    else:
        full_psf_image = None

    if make_weight_image:
        full_weight_image = galsim.ImageF(full_xsize,full_ysize)
        full_weight_image.setZero()
        full_weight_image.setScale(pixel_scale)
    else:
        full_weight_image = None

    if make_badpix_image:
        full_badpix_image = galsim.ImageF(full_xsize,full_ysize)
        full_badpix_image.setZero()
        full_badpix_image.setScale(pixel_scale)
    else:
        full_badpix_image = None

    stamp_images = BuildStamps(
            nstamps=nstamps, config=config, xsize=stamp_xsize, ysize=stamp_ysize,
            nproc=nproc, do_noise=do_noise, logger=logger,
            make_psf_image=make_psf_image,
            make_weight_image=make_weight_image,
            make_badpix_image=make_badpix_image)

    images += stamp_images[0]
    psf_images += stamp_images[1]
    weight_images += stamp_images[2]
    badpix_images += stamp_images[3]

    k = 0
    for ix in range(nx_tiles):
        for iy in range(ny_tiles):
            if k < len(images):
                xmin = ix * (stamp_xsize + xborder) + 1
                xmax = xmin + stamp_xsize-1
                ymin = iy * (stamp_ysize + yborder) + 1
                ymax = ymin + stamp_ysize-1
                b = galsim.BoundsI(xmin,xmax,ymin,ymax)
                full_image[b] += images[k]
                if make_psf_image:
                    full_psf_image[b] += psf_images[k]
                if make_weight_image:
                    full_weight_image[b] += weight_images[k]
                if make_badpix_image:
                    full_badpix_image[b] += badpix_images[k]
                k = k+1

    if not do_noise and 'noise' in config['image']:
        # If we didn't apply noise in each stamp, then we need to apply it now.

        # Use the current rng stored in config
        rng = config['rng']

        draw_method = galsim.config.GetCurrentValue(config['image'],'draw_method')
        if draw_method == 'fft':
            AddNoiseFFT(full_image,config['image']['noise'],rng)
        elif draw_method == 'phot':
            AddNoisePhot(full_image,config['image']['noise'],rng)
        else:
            raise AttributeError("Unknown draw_method %s."%draw_method)

    return full_image, full_psf_image, full_weight_image, full_badpix_image


def BuildStamps(nstamps, config, xsize, ysize, nproc=1, do_noise=True, logger=None,
                make_psf_image=False, make_weight_image=False, make_badpix_image=False):
    """
    Build a number of postage stamp images as specified by the config dict.

    @param nstamps    How many stamps to build
    @param config     A configuration dict.
    @param nproc      How many processes to use.
    @param do_noise   Whether to add noise to the image (according to config['noise'])
    @param logger     If given, a logger object to log progress.
    @param make_psf_image      Whether to make psf_image
    @param make_weight_image   Whether to make weight_image
    @param make_badpix_image   Whether to make badpix_image

    @return (images, psf_images, weight_images, badpix_images)  (All in tuple are lists)
    """
    def worker(input, output):
        """
        input is a queue with (args, info) tuples:
            kwargs are the arguments to pass to BuildSingleStamp
            info is passed along to the output queue.
        output is a queue storing (result, info, proc) tuples:
            result is the returned tuple from BuildSingleStamp: 
                (image, psf_image, weight_image, badpix_image, time).
            info is passed through from the input queue.
            proc is the process name.
        """
        for (kwargs, info) in iter(input.get, 'STOP'):
            result = BuildSingleStamp(**kwargs)
            output.put( (result, info, current_process().name) )
    
    # The kwargs to pass to build_func.
    # We'll be adding to this below...
    kwargs = { 'config' : config,
               'xsize' : xsize, 'ysize' : ysize, 
               'do_noise' : do_noise,
               'make_psf_image' : make_psf_image,
               'make_weight_image' : make_weight_image,
               'make_badpix_image' : make_badpix_image }

    if nproc == 1:

        images = []
        psf_images = []
        weight_images = []
        badpix_images = []

        for k in range(nstamps):
            if 'random_seed' in config['image']:
                seed = galsim.config.ParseValue(config['image'],'random_seed',config,int)[0]
            else:
                seed = None
            kwargs['seed'] = seed

            result = BuildSingleStamp(**kwargs)
            images += [ result[0] ]
            psf_images += [ result[1] ]
            weight_images += [ result[2] ]
            badpix_images += [ result[3] ]
            if logger:
                # Note: numpy shape is y,x
                ys, xs = result[0].array.shape
                t = result[4]
                logger.info('Stamp %d: size = %d x %d, time = %f sec', k, xs, ys, t)

    else: # nproc > 1

        from multiprocessing import Process, Queue, current_process

        # Don't save any 'current_val' results in the config, so we don't waste time sending
        # pickled versions of things back and forth.
        # TODO: This means things like gal.resolution won't work.  Once we are able to 
        # pickle GSObjects, we should send the constructed object, rather than config.
        config['no_save'] = True

        # Initialize the images list to have the correct size.
        # This is important here, since we'll be getting back images in a random order,
        # and we need them to go in the right places (in order to have deterministic
        # output files).  So we initialize the list to be the right size.
        images = [ None for i in range(nstamps) ]
        psf_images = [ None for i in range(nstamps) ]
        weight_images = [ None for i in range(nstamps) ]
        badpix_images = [ None for i in range(nstamps) ]

        # Set up the task list
        task_queue = Queue()
        for k in range(nstamps):
            # Note: we currently pull out the seed from config, since that is always
            # going to get clobbered by the multi-processing, since it involves a state
            # variable.  However, there may be other items in config that have state 
            # variables as well.  So the long term solution will be to construct the 
            # full profile in the main processor, and then send that to each of the 
            # parallel processors to draw the image.  But that will require our GSObjects
            # to be picklable, which they aren't currently.
            if 'random_seed' in config['image']:
                seed = galsim.config.ParseValue(config['image'],'random_seed',config,int)[0]
            else:
                seed = None
            # Need to make a new copy of kwargs, otherwise python's shallow copying 
            # means that each task ends up getting the same kwargs object, each with the 
            # same seed value.  Not what we want.
            new_kwargs = {}
            new_kwargs.update(kwargs)
            new_kwargs['seed'] = seed
            #print 'k = ',k,' seed = ',seed
            # Apparently the logger isn't picklable, so can't send that as an arg.
            task_queue.put( ( new_kwargs, k) )

        # Run the tasks
        # Each Process command starts up a parallel process that will keep checking the queue 
        # for a new task. If there is one there, it grabs it and does it. If not, it waits 
        # until there is one to grab. When it finds a 'STOP', it shuts down. 
        done_queue = Queue()
        for j in range(nproc):
            Process(target=worker, args=(task_queue, done_queue)).start()

        # In the meanwhile, the main process keeps going.  We pull each image off of the 
        # done_queue and put it in the appropriate place on the main image.  
        # This loop is happening while the other processes are still working on their tasks.
        # You'll see that these logging statements get print out as the stamp images are still 
        # being drawn.  
        for i in range(nstamps):
            result, k, proc = done_queue.get()
            images[k] = result[0]
            psf_images[k] = result[1]
            weight_images[k] = result[2]
            badpix_images[k] = result[3]
            if logger:
                # Note: numpy shape is y,x
                ys, xs = result[0].array.shape
                t = result[4]
                logger.info('%s: Stamp %d: size = %d x %d, time = %f sec', 
                            proc, k, xs, ys, t)

        # Stop the processes
        # The 'STOP's could have been put on the task list before starting the processes, or you
        # can wait.  In some cases it can be useful to clear out the done_queue (as we just did)
        # and then add on some more tasks.  We don't need that here, but it's perfectly fine to do.
        # Once you are done with the processes, putting nproc 'STOP's will stop them all.
        # This is important, because the program will keep running as long as there are running
        # processes, even if the main process gets to the end.  So you do want to make sure to 
        # add those 'STOP's at some point!
        for j in range(nproc):
            task_queue.put('STOP')

    if logger:
        logger.info('Done making images')

    return images, psf_images, weight_images, badpix_images
 

def BuildSingleStamp(seed, config, xsize, ysize, do_noise=True, logger=None,
                     make_psf_image=False, make_weight_image=False, make_badpix_image=False):
    """
    Build a single image using the given seed and config file

    @param seed       The random number seed to use for this stamp.  0 means use time.
    @param config     A configuration dict.
    @param xsize      The xsize of the image to build
    @param ysize      The ysize of the image to build
    @param do_noise   Whether to add noise to the image (according to config['noise'])
    @param logger     If given, a logger object to log progress.
    @param make_psf_image      Whether to make psf_image
    @param make_weight_image   Whether to make weight_image
    @param make_badpix_image   Whether to make badpix_image

    @return image, psf_image, weight_image, badpix_image, time
    """
    t1 = time.time()

    # Initialize the random number generator we will be using.
    #print 'seed = ',seed
    if seed:
        rng = galsim.UniformDeviate(seed)
    else:
        rng = galsim.UniformDeviate()
    # Store the rng in the config for use by BuildGSObject function.
    config['rng'] = rng
    if 'gd' in config:
        del config['gd']  # In case it was set.

    psf = BuildPSF(config,logger)
    t2 = time.time()

    pix = BuildPix(config,logger)
    t3 = time.time()

    gal = BuildGal(config,logger)
    t4 = time.time()
    #print 'seed, gal.flux = ',seed,gal.getFlux()

    # Check that we have at least gal or psf.
    if not (gal or psf):
        raise AttributeError("At least one of gal or psf must be specified in config.")

    draw_method = galsim.config.ParseValue(config['image'],'draw_method',config,str)[0]
    #print 'draw = ',draw_method
    if draw_method == 'fft':
        im = DrawStampFFT(psf,pix,gal,config,xsize,ysize)
        if do_noise and 'noise' in config['image']:
            AddNoiseFFT(im,config['image']['noise'],rng)
    elif draw_method == 'phot':
        im = DrawStampPhot(psf,gal,config,xsize,ysize,rng)
        if do_noise and 'noise' in config['image']:
            AddNoisePhot(im,config['image']['noise'],rng)
    else:
        raise AttributeError("Unknown draw_method %s."%draw_method)
    t5 = time.time()

    if make_psf_image:
        # Note: numpy shape is y,x
        ysize, xsize = im.array.shape
        psf_im = DrawPSFStamp(psf,pix,config,xsize,ysize)
    else:
        psf_im = None

    t6 = time.time()

    #if logger:
        #logger.info('   Times: %f, %f, %f, %f, %f', t2-t1, t3-t2, t4-t3, t5-t4, t6-t5)
    return im, psf_im, None, None, t6-t1


def BuildPSF(config, logger=None):
    """
    Parse the field config['psf'] returning the built psf object.
    """
 
    if 'psf' in config:
        if not isinstance(config['psf'], dict):
            raise AttributeError("config.psf is not a dict.")
        psf = galsim.config.BuildGSObject(config, 'psf')[0]
    else:
        psf = None

    return psf

def BuildPix(config, logger=None):
    """
    Parse the field config['pix'] returning the built pix object.
    """
 
    if 'pix' in config: 
        if not isinstance(config['pix'], dict):
            raise AttributeError("config.pix is not a dict.")
        pix = galsim.config.BuildGSObject(config, 'pix')[0]
    else:
        pix = None

    return pix


def BuildGal(config, logger=None):
    """
    Parse the field config['gal'] returning the built gal object.
    """
 
    if 'gal' in config:
        # If we are specifying the size according to a resolution, then we 
        # need to get the PSF's half_light_radius.
        if not isinstance(config['gal'], dict):
            raise AttributeError("config.gal is not a dict.")
        if 'resolution' in config['gal']:
            if 'psf' not in config:
                raise AttributeError(
                    "Cannot use gal.resolution if no psf is set.")
            if 'saved_re' not in config['psf']:
                raise AttributeError(
                    'Cannot use gal.resolution with psf.type = %s'%config['psf']['type'])
            psf_re = config['psf']['saved_re']
            resolution = galsim.config.ParseValue(config['gal'], 'resolution', config, float)[0]
            gal_re = resolution * psf_re
            config['gal']['half_light_radius'] = gal_re

        gal = galsim.config.BuildGSObject(config, 'gal')[0]
    else:
        gal = None
    return gal



def DrawStampFFT(psf, pix, gal, config, xsize, ysize):
    """
    Draw an image using the given psf, pix and gal profiles (which may be None)
    using the FFT method for doing the convolution.

    @return the resulting image.
    """
    if 'image' in config and 'wcs' in config['image']:
        wcs_shear = CalculateWCSShear(config['image']['wcs'])
    else:
        wcs_shear = None

    if wcs_shear:
        nopix_list = [ prof for prof in (psf,gal) if prof is not None ]
        nopix = galsim.Convolve(nopix_list)
        nopix.applyShear(wcs_shear)
        if pix:
            final = galsim.Convolve([nopix, pix])
        else:
            final = nopix
        config['wcs_shear'] = wcs_shear
    else:
        fft_list = [ prof for prof in (psf,pix,gal) if prof is not None ]
        final = galsim.Convolve(fft_list)

    if 'image' in config and 'pixel_scale' in config['image']:
        pixel_scale = galsim.config.ParseValue(config['image'], 'pixel_scale', config, float)[0]
    else:
        pixel_scale = 1.0

    if not xsize:
        im = final.draw(dx=pixel_scale)
    else:
        im = galsim.ImageF(xsize, ysize)
        im.setScale(pixel_scale)
        #print 'pixel_scale = ',pixel_scale
        final.draw(im, dx=pixel_scale)

    if 'gal' in config and 'signal_to_noise' in config['gal']:
        import math
        import numpy
        if 'flux' in config['gal']:
            raise AttributeError(
                'Only one of signal_to_noise or flux may be specified for gal')

        if 'image' in config and 'noise' in config['image']:
            noise_var = CalculateNoiseVar(config['image']['noise'], pixel_scale)
        else:
            raise AttributeError(
                "Need to specify noise level when using gal.signal_to_noise")
        sn_target = galsim.config.ParseValue(config['gal'], 'signal_to_noise', config, float)[0]
            
        # Now determine what flux we need to get our desired S/N
        # There are lots of definitions of S/N, but here is the one used by Great08
        # We use a weighted integral of the flux:
        # S = sum W(x,y) I(x,y) / sum W(x,y)
        # N^2 = Var(S) = sum W(x,y)^2 Var(I(x,y)) / (sum W(x,y))^2
        # Now we assume that Var(I(x,y)) is dominated by the sky noise, so
        # Var(I(x,y)) = var
        # We also assume that we are using a matched filter for W, so W(x,y) = I(x,y).
        # Then a few things cancel and we find that
        # S/N = sqrt( sum I(x,y)^2 / var )

        sn_meas = math.sqrt( numpy.sum(im.array**2) / noise_var )
        # Now we rescale the flux to get our desired S/N
        flux = sn_target / sn_meas
        #print 'noise_var = ',noise_var
        #print 'sn_meas = ',sn_meas
        #print 'flux = ',flux
        im *= flux
    return im

def AddNoiseFFT(im, noise, rng):
    """
    Add noise to an image according to the noise specifications in the noise dict
    appropriate for an image that has been drawn using the fft method.
    """
    type = noise.get('type','CCDNoise')
    pixel_scale = im.getScale()
    if type == 'Gaussian':
        single = [ { 'sigma' : float , 'variance' : float } ]
        params = galsim.config.GetAllParams(noise, 'noise', noise, single=single)[0]

        if 'sigma' in params:
            sigma = params['sigma']
        else:
            import math
            sigma = math.sqrt(params['variance'])
        im.addNoise(galsim.GaussianDeviate(rng,sigma=sigma))
        #if logger:
            #logger.info('   Added Gaussian noise with sigma = %f',sigma)
    elif type == 'CCDNoise':
        req = { 'sky_level' : float }
        opt = { 'gain' : float , 'read_noise' : float }
        params = galsim.config.GetAllParams(noise, 'noise', noise, req=req, opt=opt)[0]
        sky_level = params['sky_level']
        gain = params.get('gain',1.0)
        read_noise = params.get('read_noise',0.0)

        sky_level_pixel = sky_level * pixel_scale**2
        im += sky_level_pixel
        #print 'before CCDNoise: rng() = ',rng()
        im.addNoise(galsim.CCDNoise(rng, gain=gain, read_noise=read_noise))
        #print 'after CCDNoise: rng() = ',rng()
        im -= sky_level_pixel
        #if logger:
            #logger.info('   Added CCD noise with sky_level = %f, ' +
                        #'gain = %f, read_noise = %f',sky_level,gain,read_noise)
    else:
        raise AttributeError("Invalid type %s for noise"%type)


def DrawStampPhot(psf, gal, config, xsize, ysize, rng):
    """
    Add noise to an image according to the noise specifications in the noise dict
    appropriate for an image that has been drawn using the phot method.
    """

    phot_list = [ prof for prof in (psf,gal) if prof is not None ]
    final = galsim.Convolve(phot_list)

    if 'image' in config and 'wcs' in config['image']:
        wcs_shear = CalculateWCSShear(config['image']['wcs'])
    else:
        wcs_shear = None

    if wcs_shear:
        final.applyShear(wcs_shear)
        config['wcs_shear'] = wcs_shear
                    
    if 'signal_to_noise' in config['gal']:
        raise NotImplementedError(
            "gal.signal_to_noise not implemented for draw_method = phot")

    if 'image' in config and 'pixel_scale' in config['image']:
        pixel_scale = galsim.config.ParseValue(config['image'], 'pixel_scale', config, float)[0]
    else:
        pixel_scale = 1.0

    if 'image' in config and 'max_extra_noise' in config['image']:
        max_extra_noise = galsim.config.ParseValue(
                config['image'], 'max_extra_noise', config, float)[0]
    else:
        max_extra_noise = 0.01

    if max_extra_noise < 0.:
        raise ValueError("image.max_extra_noise cannot be negative")

    if max_extra_noise > 0.:
        if 'image' in config and 'noise' in config['image']:
            noise_var = CalculateNoiseVar(config['image']['noise'], pixel_scale)
        else:
            raise AttributeError(
                "Need to specify noise level when using draw_method = phot")
        if noise_var < 0.:
            raise ValueError("noise_var calculated to be < 0.")
        max_extra_noise *= noise_var

    if not xsize:
        # TODO: Change this once issue #82 is done.
        raise AttributeError(
            "image size must be specified when doing photon shooting.")
    else:
        im = galsim.ImageF(xsize, ysize)
        im.setScale(pixel_scale)
        #print 'noise_var = ',noise_var
        #print 'im.scale = ',im.scale
        #print 'im.bounds = ',im.bounds
        #print 'before drawShoot: rng() = ',rng()
        final.drawShoot(im, noise=max_extra_noise, uniform_deviate=rng)
        #print 'after drawShoot: rng() = ',rng()

    return im
    
def AddNoisePhot(im, noise, rng):
    """
    Add noise to an image according to the noise specifications in the noise dict
    appropriate for an image that has been drawn using the phot method.
    """
    type = noise.get('type','CCDNoise')
    pixel_scale = im.getScale()
    if type == 'Gaussian':
        single = [ { 'sigma' : float , 'variance' : float } ]
        params = galsim.config.GetAllParams(noise, 'noise', noise, single=single)[0]

        if 'sigma' in params:
            sigma = params['sigma']
        else:
            sigma = math.sqrt(params['variance'])
        im.addNoise(galsim.GaussianDeviate(rng,sigma=sigma))
        #if logger:
            #logger.info('   Added Gaussian noise with sigma = %f',sigma)
    elif type == 'CCDNoise':
        req = { 'sky_level' : float }
        opt = { 'gain' : float , 'read_noise' : float }
        params = galsim.config.GetAllParams(noise, 'noise', noise, req=req, opt=opt)[0]
        sky_level = params['sky_level']
        gain = params.get('gain',1.0)
        read_noise = params.get('read_noise',0.0)

        sky_level_pixel = params['sky_level'] * pixel_scale**2
        # For photon shooting, galaxy already has poisson noise, so we want 
        # to make sure not to add that again!
        im *= gain
        im.addNoise(galsim.PoissonDeviate(rng, mean=sky_level_pixel*gain))
        im /= gain
        im.addNoise(galsim.GaussianDeviate(rng, sigma=read_noise))
        #if logger:
            #logger.info('   Added CCD noise with sky_level = %f, ' +
                        #'gain = %f, read_noise = %f',sky_level,gain,read_noise)
    else:
        raise AttributeError("Invalid type %s for noise",type)


def DrawPSFStamp(psf, pix, config, xsize, ysize):
    """
    Draw an image using the given psf and pix profiles.

    @return the resulting image.
    """

    if not psf:
        raise AttributeError("DrawPSFStamp requires psf to be provided.")

    if 'wcs_shear' in config:
        wcs_shear = config['wcs_shear']
    else:
        wcs_shear = None

    if wcs_shear:
        psf = psf.createSheared(wcs_shear)

    psf_list = [ prof for prof in (psf,pix) if prof is not None ]

    final_psf = galsim.Convolve(psf_list)


    if 'image' in config and 'pixel_scale' in config['image']:
        pixel_scale = galsim.config.ParseValue(config['image'], 'pixel_scale', config, float)[0]
    else:
        pixel_scale = 1.0

    # Special: if the galaxy was shifted, then also shift the psf 
    if 'shift' in config['gal']:
        gal_shift = galsim.config.GetCurrentValue(config['gal'],'shift')
        final_psf.applyShift(gal_shift.dx, gal_shift.dy)

    if xsize:
        psf_im = galsim.ImageF(xsize,ysize)
        psf_im.setScale(pixel_scale)
        final_psf.draw(psf_im, dx=pixel_scale)
    else:
        psf_im = final_psf.draw(dx=pixel_scale)

    return psf_im
           
def CalculateWCSShear(wcs):
    """
    Calculate the WCS shear from the WCS specified in the wcs dict.
    TODO: Should add in more WCS types than just a simple shear
          E.g. a full CD matrix and (eventually) things like TAN and TNX.
    """
    if not isinstance(wcs, dict):
        raise AttributeError("image.wcs is not a dict.")

    type = wcs.get('type','Shear')

    if type == 'Shear':
        req = { 'shear' : galsim.Shear }
        params = galsim.config.GetAllParams(wcs, 'wcs', wcs, req=req)[0]
        return params['shear']
    else:
        raise AttributeError("Invalid type %s for wcs",type)

def CalculateNoiseVar(noise, pixel_scale):
    """
    Calculate the noise variance from the noise specified in the noise dict.
    """
    if not isinstance(noise, dict):
        raise AttributeError("image.noise is not a dict.")

    type = noise.get('type','CCDNoise')

    if type == 'Gaussian':
        single = [ { 'sigma' : float , 'variance' : float } ]
        params = galsim.config.GetAllParams(noise, 'noise', noise, single=single)[0]
        if 'sigma' in params:
            sigma = params['sigma']
            var = sigma * sigma
        else:
            var = params['variance']
    elif type == 'CCDNoise':
        req = { 'sky_level' : float }
        opt = { 'gain' : float , 'read_noise' : float }
        params = galsim.config.GetAllParams(noise, 'noise', noise, req=req, opt=opt)[0]
        sky_level = params['sky_level']
        gain = params.get('gain',1.0)
        read_noise = params.get('read_noise',0.0)
        var = params['sky_level'] * pixel_scale**2
        var /= gain
        var += read_noise * read_noise
    else:
        raise AttributeError("Invalid type %s for noise",type)

    return var


