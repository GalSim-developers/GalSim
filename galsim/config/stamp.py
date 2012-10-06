
import time
import galsim


def BuildStamps(nobjects, config, xsize, ysize, 
                obj_num=0, nproc=1, sky_level=None, do_noise=True, logger=None,
                make_psf_image=False, make_weight_image=False, make_badpix_image=False):
    """
    Build a number of postage stamp images as specified by the config dict.

    @param nobjects            How many postage stamps to build.
    @param config              A configuration dict.
    @param xsize               The size of a single stamp in the x direction.
    @param ysize               The size of a single stamp in the y direction.
    @param obj_num             If given, the current obj_num (default = 0)
    @param nproc               How many processes to use.
    @param sky_level           The background sky level to add to the image.
    @param do_noise            Whether to add noise to the image (according to config['noise']).
    @param logger              If given, a logger object to log progress.
    @param make_psf_image      Whether to make psf_image.
    @param make_weight_image   Whether to make weight_image.
    @param make_badpix_image   Whether to make badpix_image.

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
        import copy
        for (kwargs, config, obj_num, nobj, info) in iter(input.get, 'STOP'):
            #print 'In worker.'
            #print 'got obj_num = ',obj_num
            #print 'nobj = ',nobj
            #print 'info = ',info
            results = []
            # Make new copies of config and kwargs so we can update them without
            # clobbering the versions for other tasks on the queue.
            # (The config modifications come in BuildSingleStamp.)
            config = copy.copy(config)
            kwargs = copy.copy(kwargs)
            for i in range(nobj):
                kwargs['config'] = config
                kwargs['obj_num'] = obj_num + i
                results.append(BuildSingleStamp(**kwargs))
            output.put( (results, info, current_process().name) )
    
    # The kwargs to pass to build_func.
    # We'll be adding to this below...
    kwargs = {
        'xsize' : xsize, 'ysize' : ysize, 
        'sky_level' : sky_level,
        'do_noise' : do_noise,
        'make_psf_image' : make_psf_image,
        'make_weight_image' : make_weight_image,
        'make_badpix_image' : make_badpix_image
    }
    # Apparently the logger isn't picklable, so can't send that as an arg.

    if nproc > nobjects:
        import warnings
        warnings.warn(
            "Trying to use more processes than objects: image.nproc=%d, "%nproc +
            "nobjects=%d.  Reducing nproc to %d."%(nobjects,nobjects))
        nproc = nobjects

    if nproc <= 0:
        # Try to figure out a good number of processes to use
        try:
            from multiprocessing import cpu_count
            ncpu = cpu_count()
            if ncpu > nobjects:
                nproc = nobjects
            else:
                nproc = ncpu
            if logger:
                logger.info("ncpu = %d.  Using %d processes",ncpu,nproc)
        except:
            raise AttributeError(
                "config.image.nproc <= 0, but unable to determine number of cpus.")
    
    if nproc > 1:
        from multiprocessing import Process, Queue, current_process

        # Initialize the images list to have the correct size.
        # This is important here, since we'll be getting back images in a random order,
        # and we need them to go in the right places (in order to have deterministic
        # output files).  So we initialize the list to be the right size.
        images = [ None for i in range(nobjects) ]
        psf_images = [ None for i in range(nobjects) ]
        weight_images = [ None for i in range(nobjects) ]
        badpix_images = [ None for i in range(nobjects) ]

        # Number of objects to do in each task:
        # At most nobjects / nproc.
        # At least 1 normally, but number in Ring if doing a Ring test
        # Shoot for gemoetric mean of these two.
        max_nobj = nobjects / nproc
        min_nobj = 1
        #print 'gal' in config
        if ( 'gal' in config and isinstance(config['gal'],dict) and 'type' in config['gal'] and
             config['gal']['type'] == 'Ring' and 'num' in config['gal'] ):
            min_nobj = galsim.config.ParseValue(config['gal'], 'num', config, int)[0]
            #print 'Found ring: num = ',min_nobj
        if max_nobj < min_nobj: 
            nobj_per_task = min_nobj
        else:
            import math
            # This formula keeps nobj a multiple of min_nobj, so Rings are intact.
            nobj_per_task = min_nobj * int(math.sqrt(float(max_nobj) / float(min_nobj)))
        #print 'nobj_per_task = ',nobj_per_task

        # Set up the task list
        task_queue = Queue()
        for k in range(0,nobjects,nobj_per_task):
            # Send kwargs, config, obj_num, nobj, k
            if k + nobj_per_task > nobjects:
                task_queue.put( ( kwargs, config, obj_num+k, nobjects-k, k ) )
            else:
                task_queue.put( ( kwargs, config, obj_num+k, nobj_per_task, k ) )
            #print 'put task ',obj_num+k,nobj_per_task,k

        # Run the tasks
        # Each Process command starts up a parallel process that will keep checking the queue 
        # for a new task. If there is one there, it grabs it and does it. If not, it waits 
        # until there is one to grab. When it finds a 'STOP', it shuts down. 
        done_queue = Queue()
        for j in range(nproc):
            Process(target=worker, args=(task_queue, done_queue)).start()

        # In the meanwhile, the main process keeps going.  We pull each set of images off of the 
        # done_queue and put them in the appropriate place in the lists.
        # This loop is happening while the other processes are still working on their tasks.
        # You'll see that these logging statements get print out as the stamp images are still 
        # being drawn.  
        for i in range(0,nobjects,nobj_per_task):
            results, k, proc = done_queue.get()
            for result in results:
                images[k] = result[0]
                psf_images[k] = result[1]
                weight_images[k] = result[2]
                badpix_images[k] = result[3]
                if logger:
                    # Note: numpy shape is y,x
                    ys, xs = result[0].array.shape
                    t = result[4]
                    logger.info('%s: Stamp %d: size = %d x %d, time = %f sec', proc, k, xs, ys, t)
                k += 1

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

    else : # nproc == 1

        images = []
        psf_images = []
        weight_images = []
        badpix_images = []

        for k in range(nobjects):
            kwargs['obj_num'] = obj_num+k
            kwargs['config'] = config
            kwargs['obj_num'] = obj_num+k
            kwargs['logger'] = logger
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


    if logger:
        logger.info('Done making images')

    return images, psf_images, weight_images, badpix_images
 

def BuildSingleStamp(config, xsize, ysize,
                     obj_num=0, sky_level=None, do_noise=True, logger=None,
                     make_psf_image=False, make_weight_image=False, make_badpix_image=False):
    """
    Build a single image using the given config file

    @param config              A configuration dict.
    @param xsize               The xsize of the image to build.
    @param ysize               The ysize of the image to build.
    @param obj_num             If given, the current obj_num (default = 0)
    @param sky_level           The background sky level to add to the image.
    @param do_noise            Whether to add noise to the image (according to config['noise']).
    @param logger              If given, a logger object to log progress.
    @param make_psf_image      Whether to make psf_image.
    @param make_weight_image   Whether to make weight_image.
    @param make_badpix_image   Whether to make badpix_image.

    @return image, psf_image, weight_image, badpix_image, time
    """
    t1 = time.time()

    config['seq_index'] = obj_num 
    # Initialize the random number generator we will be using.
    if 'random_seed' in config['image']:
        seed = galsim.config.ParseValue(config['image'],'random_seed',config,int)[0]
        #print 'Using seed = ',seed
        rng = galsim.UniformDeviate(seed)
    else:
        rng = galsim.UniformDeviate()
    # Store the rng in the config for use by BuildGSObject function.
    config['rng'] = rng
    if 'gd' in config:
        del config['gd']  # In case it was set.

    # Determine where this object is going to go:
    if 'center' in config['image']:
        import math
        center = galsim.config.ParseValue(config['image'],'center',config,galsim.PositionD)[0]
        #print 'center x,y = ',center.x,center.y
        icenter = galsim.PositionI(
            int(math.floor(center.x+0.5)),
            int(math.floor(center.y+0.5)) )
        final_shift = galsim.PositionD(center.x-icenter.x , center.y-icenter.y)
        # Calculate and save the position relative to the image center
        config['pos'] = (center - config['image_cen']) * config['pixel_scale']
        #print 'center = ',center
        #print 'image_cen = ',config['image_cen']
        #print 'pos = ',center - config['image_cen']
        #print 'pos => ',config['pos']
    else:
        center = None
        icenter = None
        final_shift = None

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

    #print 'config[image] = ',config['image']
    draw_method = galsim.config.ParseValue(config['image'],'draw_method',config,str)[0]
    #print 'draw = ',draw_method
    if draw_method == 'fft':
        im = DrawStampFFT(psf,pix,gal,config,xsize,ysize,sky_level,final_shift)
        if icenter:
            im.setCenter(icenter.x, icenter.y)
        if make_weight_image:
            weight_im = galsim.ImageF(im.bounds)
            #print 'make weight_im from im.bounds = ',im.bounds
            #print 'weight_im.bounds = ',weight_im.bounds
            weight_im.setScale(im.scale)
            weight_im.setZero()
        else:
            weight_im = None
        if do_noise:
            if 'noise' in config['image']:
                AddNoiseFFT(im,weight_im,config['image']['noise'],config,rng,sky_level)
            elif sky_level:
                pixel_scale = im.getScale()
                im += sky_level * pixel_scale**2

    elif draw_method == 'phot':
        im = DrawStampPhot(psf,gal,config,xsize,ysize,rng,sky_level,final_shift)
        if icenter:
            im.setCenter(icenter.x, icenter.y)
        if make_weight_image:
            weight_im = galsim.ImageF(im.bounds)
            weight_im.setScale(im.scale)
            weight_im.setZero()
        else:
            weight_im = None
        if do_noise:
            if 'noise' in config['image']:
                AddNoisePhot(im,weight_im,config['image']['noise'],config,rng,sky_level)
            elif sky_level:
                pixel_scale = im.getScale()
                im += sky_level * pixel_scale**2

    else:
        raise AttributeError("Unknown draw_method %s."%draw_method)

    if make_badpix_image:
        badpix_im = galsim.ImageS(im.bounds)
        badpix_im.setScale(im.scale)
        badpix_im.setZero()
    else:
        badpix_im = None

    t5 = time.time()

    if make_psf_image:
        psf_im = DrawPSFStamp(psf,pix,config,im.bounds,final_shift)
    else:
        psf_im = None

    t6 = time.time()

    if logger:
        logger.debug('   Times: %f, %f, %f, %f, %f', t2-t1, t3-t2, t4-t3, t5-t4, t6-t5)
    return im, psf_im, weight_im, badpix_im, t6-t1


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
        if not isinstance(config['gal'], dict):
            raise AttributeError("config.gal is not a dict.")
        gal = galsim.config.BuildGSObject(config, 'gal')[0]
    else:
        gal = None
    return gal



def DrawStampFFT(psf, pix, gal, config, xsize, ysize, sky_level, final_shift):
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

    if final_shift:
        final.applyShift(final_shift.x*pixel_scale, final_shift.y*pixel_scale)
        #print 'shift by ',final_shift.x,final_shift.y
        #print 'which in arcsec is ',final_shift.x*pixel_scale,final_shift.y*pixel_scale

    #print 'final.flux = ',final.getFlux()
    if xsize:
        im = galsim.ImageF(xsize, ysize)
    else:
        im = None

    im = final.draw(image=im, dx=pixel_scale)

    if 'gal' in config and 'signal_to_noise' in config['gal']:
        import math
        import numpy
        if 'flux' in config['gal']:
            raise AttributeError(
                'Only one of signal_to_noise or flux may be specified for gal')

        if 'image' in config and 'noise' in config['image']:
            noise_var = CalculateNoiseVar(config['image']['noise'], pixel_scale, sky_level)
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

def AddNoiseFFT(im, weight_im, noise, base, rng, sky_level):
    """
    Add noise to an image according to the noise specifications in the noise dict
    appropriate for an image that has been drawn using the FFT method.
    """
    if not isinstance(noise, dict):
        raise AttributeError("image.noise is not a dict.")

    if 'type' not in noise:
        noise['type'] = 'CCDNoise'  # Default is CCDNoise
    type = noise['type']
    pixel_scale = im.getScale()

    # First add the sky noise, if provided
    if sky_level:
        im += sky_level * pixel_scale**2

    # Check if a weight image should include the object variance.
    if weight_im:
        include_obj_var = False
        if ('output' in base and 'weight' in base['output'] and 
            'include_obj_var' in base['output']['weight']):
            include_obj_var = galsim.config.ParseValue(
                base['output']['weight'], 'include_obj_var', base, bool)[0]

    # Then add the correct kind of noise
    if type == 'Gaussian':
        single = [ { 'sigma' : float , 'variance' : float } ]
        params = galsim.config.GetAllParams(noise, 'noise', noise, single=single)[0]

        if 'sigma' in params:
            sigma = params['sigma']
        else:
            import math
            sigma = math.sqrt(params['variance'])
        im.addNoise(galsim.GaussianDeviate(rng,sigma=sigma))

        if weight_im:
            weight_im += sigma*sigma
        #if logger:
            #logger.debug('   Added Gaussian noise with sigma = %f',sigma)

    elif type == 'CCDNoise':
        req = {}
        opt = { 'gain' : float , 'read_noise' : float }
        if sky_level:
            # The noise sky_level is only required here if the image doesn't have any.
            opt['sky_level'] = float
        else:
            req['sky_level'] = float
        params = galsim.config.GetAllParams(noise, 'noise', noise, req=req, opt=opt)[0]
        gain = params.get('gain',1.0)
        read_noise = params.get('read_noise',0.0)

        if 'sky_level' in params:
            sky_level_pixel = params['sky_level'] * pixel_scale**2
            im += sky_level_pixel

        if weight_im:
            import math
            if include_obj_var:
                # The image right now has the variance in each pixel.  So before going on with the 
                # noise, copy these over to the weight image and invert.
                weight_im.copyFrom(im)
                if gain != 1.0:
                    weight_im /= math.sqrt(gain)
                if read_noise != 0.0:
                    weight_im += read_noise*read_noise
            else:
                # Otherwise, just add the sky and read_noise:
                weight_im += sky_level_pixel / math.sqrt(gain) + read_noise*read_noise

        #print 'before CCDNoise: rng() = ',rng()
        im.addNoise(galsim.CCDNoise(rng, gain=gain, read_noise=read_noise))
        #print 'after CCDNoise: rng() = ',rng()
        if 'sky_level' in params:
            im -= sky_level_pixel
        #if logger:
            #logger.debug('   Added CCD noise with sky_level = %f, ' +
                         #'gain = %f, read_noise = %f',sky_level,gain,read_noise)
    else:
        raise AttributeError("Invalid type %s for noise"%type)


def DrawStampPhot(psf, gal, config, xsize, ysize, rng, sky_level, final_shift):
    """
    Draw an image using the given psf and gal profiles (which may be None)
    using the photon shooting method for doing the convolution.

    @return the resulting image.
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

    if final_shift:
        final.applyShift(final_shift.x, final_shift.y)

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
            noise_var = CalculateNoiseVar(config['image']['noise'], pixel_scale, sky_level)
        else:
            raise AttributeError(
                "Need to specify noise level when using draw_method = phot")
        if noise_var < 0.:
            raise ValueError("noise_var calculated to be < 0.")
        max_extra_noise *= noise_var

    if xsize:
        im = galsim.ImageF(xsize, ysize)
    else:
        im = None

    im = final.drawShoot(image=im, dx=pixel_scale, max_extra_noise=max_extra_noise, rng=rng)[0]

    return im
    
def AddNoisePhot(im, weight_im, noise, base, rng, sky_level):
    """
    Add noise to an image according to the noise specifications in the noise dict
    appropriate for an image that has been drawn using the photon-shooting method.
    """
    if not isinstance(noise, dict):
        raise AttributeError("image.noise is not a dict.")

    if 'type' not in noise:
        noise['type'] = 'CCDNoise'  # Default is CCDNoise
    type = noise['type']
    pixel_scale = im.getScale()

    # First add the sky noise, if provided
    if sky_level:
        im += sky_level * pixel_scale**2

    if type == 'Gaussian':
        single = [ { 'sigma' : float , 'variance' : float } ]
        params = galsim.config.GetAllParams(noise, 'noise', noise, single=single)[0]

        if 'sigma' in params:
            sigma = params['sigma']
        else:
            import math
            sigma = math.sqrt(params['variance'])
        im.addNoise(galsim.GaussianDeviate(rng,sigma=sigma))

        if weight_im:
            weight_im += sigma*sigma
        #if logger:
            #logger.debug('   Added Gaussian noise with sigma = %f',sigma)

    elif type == 'CCDNoise':
        req = {}
        opt = { 'gain' : float , 'read_noise' : float }
        if sky_level:
            opt['sky_level'] = float
        else:
            req['sky_level'] = float
            sky_level = 0. # Switch from None to 0.
        params = galsim.config.GetAllParams(noise, 'noise', noise, req=req, opt=opt)[0]
        if 'sky_level' in params:
            sky_level += params['sky_level']
        gain = params.get('gain',1.0)
        read_noise = params.get('read_noise',0.0)

        # We don't have an exact value for the variance in each pixel, but the drawn image
        # before adding the Poisson noise is our best guess for the variance from the 
        # object's flux, so just use that for starters.
        if weight_im and include_obj_var:
            weight_im.copyFrom(im)

        # For photon shooting, galaxy already has Poisson noise, so we want 
        # to make sure not to add that again!
        if sky_level != 0.:
            sky_level_pixel = sky_level * pixel_scale**2
            if gain != 1.0: im *= gain
            im.addNoise(galsim.PoissonDeviate(rng, mean=sky_level_pixel*gain))
            if gain != 1.0: im /= gain
        if read_noise != 0.:
            im.addNoise(galsim.GaussianDeviate(rng, sigma=read_noise))

        # Add in these effects to the weight image:
        if weight_im:
            import math
            if sky_level != 0.0 or read_noise != 0.0:
                weight_im += sky_level_pixel / math.sqrt(gain) + read_noise * read_noise
        #if logger:
            #logger.debug('   Added CCD noise with sky_level = %f, ' +
                         #'gain = %f, read_noise = %f',sky_level,gain,read_noise)

    else:
        raise AttributeError("Invalid type %s for noise",type)


def DrawPSFStamp(psf, pix, config, bounds, final_shift):
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

    if ('output' in config and 
        'psf' in config['output'] and 
        'real_space' in config['output']['psf'] ):
        real_space = galsim.config.ParseValue(config['output']['psf'],'real_space',config,bool)[0]
    else:
        real_space = None
        
    final_psf = galsim.Convolve(psf_list, real_space=real_space)

    if 'image' in config and 'pixel_scale' in config['image']:
        pixel_scale = galsim.config.ParseValue(config['image'], 'pixel_scale', config, float)[0]
    else:
        pixel_scale = 1.0

    # Special: if the galaxy was shifted, then also shift the psf 
    if 'shift' in config['gal']:
        gal_shift = galsim.config.GetCurrentValue(config['gal'],'shift')
        final_psf.applyShift(gal_shift.x, gal_shift.y)

    # Also apply any "final" shift to the psf.
    if final_shift:
        final_psf.applyShift(final_shift.x, final_shift.y)

    psf_im = galsim.ImageF(bounds)
    psf_im.setScale(pixel_scale)
    final_psf.draw(psf_im, dx=pixel_scale)

    return psf_im
           
def CalculateWCSShear(wcs):
    """
    Calculate the WCS shear from the WCS specified in the wcs dict.
    TODO: Should add in more WCS types than just a simple shear
          E.g. a full CD matrix and (eventually) things like TAN and TNX.
    """
    if not isinstance(wcs, dict):
        raise AttributeError("image.wcs is not a dict.")

    if 'type' not in wcs:
        wcs['type'] = 'Shear'  # Default is Shear
    type = wcs['type']

    if type == 'Shear':
        req = { 'shear' : galsim.Shear }
        params = galsim.config.GetAllParams(wcs, 'wcs', wcs, req=req)[0]
        return params['shear']
    else:
        raise AttributeError("Invalid type %s for wcs",type)

def CalculateNoiseVar(noise, pixel_scale, sky_level):
    """
    Calculate the noise variance from the noise specified in the noise dict.
    """
    if not isinstance(noise, dict):
        raise AttributeError("image.noise is not a dict.")

    if 'type' not in noise:
        noise['type'] = 'CCDNoise'  # Default is CCDNoise
    type = noise['type']

    if type == 'Gaussian':
        single = [ { 'sigma' : float , 'variance' : float } ]
        params = galsim.config.GetAllParams(noise, 'noise', noise, single=single)[0]
        if 'sigma' in params:
            sigma = params['sigma']
            var = sigma * sigma
        else:
            var = params['variance']
    elif type == 'CCDNoise':
        req = {}
        opt = { 'gain' : float , 'read_noise' : float }
        if sky_level:
            opt['sky_level'] = float
        else:
            req['sky_level'] = float
        params = galsim.config.GetAllParams(noise, 'noise', noise, req=req, opt=opt)[0]
        if 'sky_level' in params:
            sky_level = params['sky_level']
        gain = params.get('gain',1.0)
        read_noise = params.get('read_noise',0.0)
        var = params['sky_level'] * pixel_scale**2
        var /= gain
        var += read_noise * read_noise
    else:
        raise AttributeError("Invalid type %s for noise",type)

    return var


