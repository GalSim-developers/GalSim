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

valid_output_types = { 
    # The values are tuples with:
    # - the build function to call
    # - a function that merely counts the number of objects that will be built by the function
    # - whether the Builder takes nproc.
    # - whether the Builder takes psf_file_name, weight_file_name, and badpix_file_name.
    # - whether the Builder takes psf_hdu, weight_hdu, and badpix_hdu.
    # See the des module for examples of how to extend this from a module.
    'Fits' : ('BuildFits', 'GetNObjForFits', False, True, True),
    'MultiFits' : ('BuildMultiFits', 'GetNObjForMultiFits', True, True, False),
    'DataCube' : ('BuildDataCube', 'GetNObjForDataCube', True, True, False),
}


# A helper function to retry io commands
def _retry_io(func, args, ntries, file_name, logger):
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

    config['index_key'] = 'file_num'
    config['file_num'] = file_num
    config['image_num'] = image_num
    config['start_obj_num'] = obj_num
    config['obj_num'] = obj_num
    if logger and logger.isEnabledFor(logging.DEBUG):
        logger.debug('file %d: BuildFits for %s: file, image, obj = %d,%d,%d',
                      config['file_num'],file_name,file_num,image_num,obj_num)

    if ( 'image' in config 
         and 'random_seed' in config['image'] 
         and not isinstance(config['image']['random_seed'],dict) ):
        first = galsim.config.ParseValue(config['image'], 'random_seed', config, int)[0]
        config['image']['random_seed'] = { 'type' : 'Sequence', 'first' : first }

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

    _retry_io(galsim.fits.writeMulti, (hdulist, file_name), ntries, file_name, logger)
    if logger and logger.isEnabledFor(logging.DEBUG):
        if len(hdus.keys()) == 1:
            logger.debug('file %d: Wrote image to fits file %r',
                         config['file_num'],file_name)
        else:
            logger.debug('file %d: Wrote image (with extra hdus) to multi-extension fits file %r',
                         config['file_num'],file_name)

    if psf_file_name:
        _retry_io(galsim.fits.write, (all_images[1], psf_file_name),
                  ntries, psf_file_name, logger)
        if logger and logger.isEnabledFor(logging.DEBUG):
            logger.debug('file %d: Wrote psf image to fits file %r',
                         config['file_num'],psf_file_name)

    if weight_file_name:
        _retry_io(galsim.fits.write, (all_images[2], weight_file_name),
                  ntries, weight_file_name, logger)
        if logger and logger.isEnabledFor(logging.DEBUG):
            logger.debug('file %d: Wrote weight image to fits file %r',
                         config['file_num'],weight_file_name)

    if badpix_file_name:
        _retry_io(galsim.fits.write, (all_images[3], badpix_file_name),
                  ntries, badpix_file_name, logger)
        if logger and logger.isEnabledFor(logging.DEBUG):
            logger.debug('file %d: Wrote badpix image to fits file %r',
                         config['file_num'],badpix_file_name)

    t2 = time.time()
    return t2-t1


def GetNObjForFits(config, file_num, image_num):
    ignore = [ 'file_name', 'dir', 'nfiles', 'psf', 'weight', 'badpix', 'nproc',
               'skip', 'noclobber', 'retry_io' ]
    galsim.config.CheckAllParams(config['output'], 'output', ignore=ignore)
    try : 
        nobj = [ galsim.config.GetNObjForImage(config, image_num) ]
    except ValueError : # (This may be raised if something needs the input stuff)
        galsim.config.ProcessInput(config, file_num=file_num)
        nobj = [ galsim.config.GetNObjForImage(config, image_num) ]
    return nobj

