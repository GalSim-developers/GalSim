
import os
import time
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

    config['catalog'] = the catalog specified by config.input.catalog, if provided
    config['real_catalog'] = the catalog specified by config.input.real_catalog, if provided
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

    if 'type' not in output:
        output['type'] = 'Fits'  # Default is Fits
    type = output['type']

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
                "output.nfiles=%d.  Reducing nproc to %d."%(nfiles,nfiles))
            nproc = nfiles
    if nproc <= 0:
        # Try to figure out a good number of processes to use
        try:
            from multiprocessing import cpu_count
            ncpu = cpu_count()
            if nfiles == 1 and (type == 'MultiFits' or type == 'DataCube'):
                kwargs['nproc'] = ncpu # Use this value in BuildImage rather than here.
                nproc = 1
                if logger:
                    logger.info("ncpu = %d.",ncpu)
            else:
                if ncpu > nfiles:
                    nproc = nfiles
                else:
                    nproc = ncpu
                if logger:
                    logger.info("ncpu = %d.  Using %d processes",ncpu,nproc)
        except:
            warnings.warn(
                "config.output.nproc <= 0, but unable to determine number of cpus.")
            nproc = 1
            if logger:
                logger.info("Unable to determine ncpu.  Using %d processes",nproc)
    
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
            if not os.path.isdir(dir): os.mkdir(dir)
            file_name = os.path.join(dir,file_name)
        else:
            dir = None

        kwargs['file_name'] = file_name
        kwargs['config'] = config

        if type == 'MultiFits' or type == 'DataCube':
            nimages = galsim.config.ParseValue(output, 'nimages', config, int)[0]
            kwargs['nimages'] = nimages 

        for extra in [ k for k in [ 'psf' , 'weight', 'badpix' ] if k in output ]:
            extra_file_name = None
            output_extra = output[extra]
            if 'file_name' in output_extra:
                extra_file_name = galsim.config.ParseValue(output[extra],'file_name',config,str)[0]
                if 'dir' in output:
                    extra_file_name = os.path.join(dir,extra_file_name)
                kwargs[ extra+'_file_name' ] = extra_file_name
            elif type != 'Fits':
                raise AttributeError(
                    "Only the file_name version of %s output is possible for "%extra+
                    "output type == %s."%type)
            elif 'hdu' not in output_extra:
                raise AttributeError("Must specify either file_name or hdu for %s output."%extra)
            else:
                extra_hdu = galsim.config.ParseValue(output[extra],'hdu',config,int)[0]
                kwargs[ extra+'_hdu' ] = extra_hdu
    
        if nproc > 1:
            new_kwargs = {}
            new_kwargs.update(kwargs)
            task_queue.put( (new_kwargs, file_name) )
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
        for k in range(nfiles):
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
            config=config, logger=logger, 
            make_psf_image=make_psf_image,
            make_weight_image=make_weight_image,
            make_badpix_image=make_badpix_image)
    # returns a tuple ( main_image, psf_image, weight_image, badpix_image )

    hdulist = []
    for h in range(len(hdus.keys())):
        assert h in hdus.keys()  # Checked for this above.
        hdulist.append(all_images[hdus[h]])
        print 'Add allimages[%d] to hdulist'%hdus[h]

    # This next line is ok even if the main image is the only one in the list.
    galsim.fits.writeMulti(hdulist, file_name)
    if logger:
        if len(hdus.keys()) == 1:
            logger.info('Wrote image to fits file %r',file_name)
        else:
            logger.info('Wrote image (with extra hdus) to multi-extension fits file %r',file_name)

    if psf_file_name:
        all_images[1].write(psf_file_name)
        if logger:
            logger.info('Wrote psf image to fits file %r',psf_file_name)

    if weight_file_name:
        all_images[2].write(weight_file_name)
        if logger:
            logger.info('Wrote weight image to fits file %r',weight_file_name)

    if badpix_file_name:
        all_images[3].write(badpix_file_name)
        if logger:
            logger.info('Wrote badpix image to fits file %r',badpix_file_name)

    t2 = time.time()
    return t2-t1


def BuildMultiFits(file_name, nimages, config, nproc=1, logger=None,
                   psf_file_name=None, weight_file_name=None, badpix_file_name=None):
    """
    Build a multi-extension fits file as specified in config.
    
    @param file_name  The name of the output file.
    @param nimages    The number of images (and hence hdus in the output file)
    @param config     A configuration dict.
    @param nproc      How many processes to use.
    @param logger     If given, a logger object to log progress.
    @param psf_file_name     If given, write a psf image to this file
    @param weight_file_name  If given, write a weight image to this file
    @param badpix_file_name  If given, write a badpix image to this file

    @return time      Time taken to build file
    """
    t1 = time.time()

    if psf_file_name:
        make_psf_image = True
    else:
        make_psf_image = False

    if weight_file_name:
        make_weight_image = True
    else:
        make_weight_image = False

    if badpix_file_name:
        make_badpix_image = True
    else:
        make_badpix_image = False

    if nproc > 1:
        import warnings
        warnings.warn("Sorry, multiple processes not currently implemented for BuildMultiFits.")

    main_images = []
    psf_images = []
    weight_images = []
    badpix_images = []
    for k in range(nimages):
        t2 = time.time()
        all_images = galsim.config.BuildImage(
                config=config, logger=logger,
                make_psf_image=make_psf_image, 
                make_weight_image=make_weight_image,
                make_badpix_image=make_badpix_image)
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


    galsim.fits.writeMulti(main_images, file_name)
    if logger:
        logger.info('Wrote images to multi-extension fits file %r',file_name)

    if psf_file_name:
        galsim.fits.writeMulti(psf_images, psf_file_name)
        if logger:
            logger.info('Wrote psf images to multi-extension fits file %r',psf_file_name)

    if weight_file_name:
        galsim.fits.writeMulti(weight_images, weight_file_name)
        if logger:
            logger.info('Wrote weight images to multi-extension fits file %r',weight_file_name)

    if badpix_file_name:
        galsim.fits.writeMulti(badpix_images, badpix_file_name)
        if logger:
            logger.info('Wrote badpix images to multi-extension fits file %r',badpix_file_name)


    t4 = time.time()
    return t4-t1


def BuildDataCube(file_name, nimages, config, nproc=1, logger=None,
                  psf_file_name=None, weight_file_name=None, badpix_file_name=None):
    """
    Build a multi-image fits data cube as specified in config.
    
    @param file_name  The name of the output file.
    @param nimages    The number of images in the data cube
    @param config     A configuration dict.
    @param nproc      How many processes to use.
    @param logger     If given, a logger object to log progress.
    @param psf_file_name     If given, write a psf image to this file
    @param weight_file_name  If given, write a weight image to this file
    @param badpix_file_name  If given, write a badpix image to this file

    @return time      Time taken to build file
    """
    t1 = time.time()

    if psf_file_name:
        make_psf_image = True
    else:
        make_psf_image = False

    if weight_file_name:
        make_weight_image = True
    else:
        make_weight_image = False

    if badpix_file_name:
        make_badpix_image = True
    else:
        make_badpix_image = False

    if make_badpix_image:
        raise NotImplementedError("Sorry, badpix image output is not currently implemented.")

    if nproc > 1:
        import warnings
        warnings.warn("Sorry, multiple processe not currently implemented for BuildMultiFits.")

    # All images need to be the same size for a data cube.
    # Enforce this by buliding the first image outside the below loop and setting
    # config['image_xsize'] and config['image_ysize'] to be the size of the first image.
    t2 = time.time()
    all_images = galsim.config.BuildImage(
            config=config, logger=logger,
            make_psf_image=make_psf_image, 
            make_weight_image=make_weight_image,
            make_badpix_image=make_badpix_image)
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
        all_images = galsim.config.BuildImage(
                config=config, logger=logger,
                make_psf_image=make_psf_image, 
                make_weight_image=make_weight_image,
                make_badpix_image=make_badpix_image)
        t5 = time.time()
        main_images += [ all_images[0] ]
        psf_images += [ all_images[1] ]
        weight_images += [ all_images[2] ]
        badpix_images += [ all_images[3] ]
        if logger:
            # Note: numpy shape is y,x
            ys, xs = all_images[0].array.shape
            logger.info('Image %d: size = %d x %d, time = %f sec', k, xs, ys, t5-t4)

    galsim.fits.writeCube(main_images, file_name)
    if logger:
        logger.info('Wrote image to fits data cube %r',file_name)

    if psf_file_name:
        galsim.fits.writeCube(psf_images, psf_file_name)
        if logger:
            logger.info('Wrote psf images to fits data cube %r',psf_file_name)

    if weight_file_name:
        galsim.fits.writeCube(weight_images, weight_file_name)
        if logger:
            logger.info('Wrote weight images to fits data cube %r',weight_file_name)

    if badpix_file_name:
        galsim.fits.writeCube(badpix_images, badpix_file_name)
        if logger:
            logger.info('Wrote badpix images to fits data cube %r',badpix_file_name)

    t6 = time.time()
    return t6-t1


