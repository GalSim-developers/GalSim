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

# This file does the equivalent of check_yaml in the examples directory.
# We don't name it test_*.py, since it's not really unit tests per se, but it
# does test the config code.  Therefore, we exclude this file from being
# run in scons tests, but we let it run in Travis to accurately report the
# coverage of the config files, which otherwise come out rather low.

from __future__ import print_function
import os
import sys
import logging
import shutil

from galsim_test_helpers import *

try:
    import galsim
except ImportError:
    path, filename = os.path.split(__file__)
    sys.path.append(os.path.abspath(os.path.join(path, "..")))
    import galsim

def remove_dir(dir_name):
    """Remove a directory in ../examples
    """
    test_dir = os.path.dirname(__file__)
    full_dir_name = os.path.join(test_dir, '../examples', dir_name)
    if os.path.exists(full_dir_name):
        shutil.rmtree(full_dir_name)

def in_examples(f):
    """A decorator that lets the code run as though it were run in the examples directory.
    """
    import functools
    @functools.wraps(f)
    def f2(*args, **kwargs):
        original_dir = os.getcwd()
        try:
            os.chdir('../examples')
            new_dir = os.getcwd()
            if new_dir not in sys.path:
                sys.path.append(new_dir)
            return f(*args, **kwargs)
        finally:
            os.chdir(original_dir)
    return f2

def check_same(f1, f2):
    import check_diff
    same = check_diff.same(f1,f2)
    if not same:
        check_diff.report(f1,f2)
    return same

logging.basicConfig(format="%(message)s", stream=sys.stdout)

@timer
@in_examples
def test_demo1():
    """Check that demo1 runs properly.
    """
    import demo1
    print('Running demo1.py')
    demo1.main([])
    logger = logging.getLogger('galsim')
    logger.setLevel(logging.WARNING)
    config = galsim.config.ReadConfig('demo1.yaml', logger=logger)[0]
    print('Running demo1.yaml')
    galsim.config.Process(config, logger=logger, except_abort=True)
    # There is no assert at the end of this one, since they are not expected to be identical
    # due to the lack of a specified seed.  This just checks for syntax errors.

@timer
@in_examples
def test_demo2():
    """Check that demo2 makes the same image using demo2.py and demo2.yaml.
    """
    import demo2
    print('Running demo2.py')
    demo2.main([])
    logger = logging.getLogger('galsim')
    logger.setLevel(logging.WARNING)
    config = galsim.config.ReadConfig('demo2.yaml', file_type='yaml', logger=logger)[0]
    print('Running demo2.yaml')
    galsim.config.Process(config, logger=logger, except_abort=True)
    assert check_same('output/demo2.fits', 'output_yaml/demo2.fits')

@timer
@in_examples
def test_demo3():
    """Check that demo3 makes the same image using demo3.py and demo3.yaml.
    """
    import demo3
    print('Running demo3.py')
    demo3.main([])
    logger = logging.getLogger('galsim')
    logger.setLevel(logging.WARNING)
    config = galsim.config.ReadConfig('demo3.yaml', logger=logger)[0]
    print('Running demo3.yaml')
    galsim.config.Process(config, logger=logger, except_abort=True)
    assert check_same('output/demo3.fits', 'output_yaml/demo3.fits')
    assert check_same('output/demo3_epsf.fits', 'output_yaml/demo3_epsf.fits')

@timer
@in_examples
def test_demo4():
    """Check that demo4 makes the same image using demo4.py and demo4.yaml.
    """
    import demo4
    print('Running demo4.py')
    demo4.main([])
    logger = logging.getLogger('galsim')
    logger.setLevel(logging.WARNING)
    config = galsim.config.ReadConfig('demo4.yaml', logger=logger)[0]
    print('Running demo4.yaml')
    galsim.config.Process(config, logger=logger, except_abort=True)
    assert check_same('output/multi.fits', 'output_yaml/multi.fits')

@timer
@in_examples
def test_demo5():
    """Check that demo5 makes the same image using demo5.py and demo5.yaml.
    """
    import demo5
    print('Running demo5.py')
    demo5.main([])
    logger = logging.getLogger('galsim')
    logger.setLevel(logging.WARNING)
    config = galsim.config.ReadConfig('demo5.yaml', logger=logger)[0]
    print('Running demo5.yaml')
    galsim.config.Process(config, logger=logger, except_abort=True)
    assert check_same('output/g08_psf.fits', 'output_yaml/g08_psf.fits')
    assert check_same('output/g08_gal.fits', 'output_yaml/g08_gal.fits')

@timer
@in_examples
def test_demo6():
    """Check that demo6 makes the same image using demo6.py and demo6.yaml.
    """
    import demo6
    print('Running demo6.py')
    demo6.main([])
    logger = logging.getLogger('galsim')
    logger.setLevel(logging.WARNING)
    configs = galsim.config.ReadConfig('demo6.yaml', logger=logger)
    print('Running demo6.yaml pass #1')
    galsim.config.Process(configs[0], logger=logger, except_abort=True)
    print('Running demo6.yaml pass #2')
    galsim.config.Process(configs[1], logger=logger, except_abort=True)
    assert check_same('output/psf_real.fits', 'output_yaml/psf_real.fits')
    assert check_same('output/cube_real.fits', 'output_yaml/cube_real.fits')

@timer
@in_examples
def test_demo7():
    """Check that demo7 makes the same image using demo7.py and demo7.yaml.
    """
    import demo7
    import gzip
    import shutil
    print('Running demo7.py')
    demo7.main([])
    logger = logging.getLogger('galsim')
    logger.setLevel(logging.WARNING)
    config = galsim.config.ReadConfig('demo7.yaml', logger=logger)[0]
    print('Running demo7.yaml')
    galsim.config.Process(config, logger=logger, except_abort=True)
    # gzip class in python 2.6 doesn't implement context correctly.  So do that one manually,
    # even though with gzip.open(...) as f_in would work fine on 2.7+
    f_in = gzip.open('output/cube_phot.fits.gz', 'rb')
    with open('output/cube_phot.fits', 'wb') as f_out:
        shutil.copyfileobj(f_in, f_out)
    f_in.close()
    f_in = gzip.open('output_yaml/cube_phot.fits.gz', 'rb')
    with open('output_yaml/cube_phot.fits', 'wb') as f_out:
        shutil.copyfileobj(f_in, f_out)
    f_in.close()
    assert check_same('output/cube_phot.fits', 'output_yaml/cube_phot.fits')

@timer
@in_examples
def test_demo8():
    """Check that demo8 makes the same image using demo8.py and demo8.yaml.
    """
    import demo8
    print('Running demo8.py')
    demo8.main([])
    logger = logging.getLogger('galsim')
    logger.setLevel(logging.WARNING)
    configs = galsim.config.ReadConfig('demo8.yaml', logger=logger)
    print('Running demo8.yaml pass #1')
    galsim.config.Process(configs[0], logger=logger, except_abort=True)
    print('Running demo8.yaml pass #2')
    galsim.config.Process(configs[1], logger=logger, except_abort=True)
    assert check_same('output/bpd_single.fits', 'output_yaml/bpd_single.fits')
    assert check_same('output/bpd_multi.fits', 'output_yaml/bpd_multi.fits')

@timer
@in_examples
def test_demo9():
    """Check that demo9 makes the same image using demo9.py and demo9.yaml.
    """
    # For this one, we'll use the json file instead, partly just to get coverage of the
    # JSON parsing functionality, but also because the changes to the base config
    # are pretty minor, so we can effect them with the new_params option.
    import demo9
    print('Running demo9.py')
    demo9.main([])
    logger = logging.getLogger('galsim')
    logger.setLevel(logging.WARNING)
    config = galsim.config.ReadConfig('json/demo9.json', logger=logger)[0]
    print('Running demo9.json')
    new_params = { 'output.skip' : { 'type' : 'List', 'items' : [0,0,0,0,0,1] } }
    galsim.config.Process(config, logger=logger, new_params=new_params, njobs=3, job=1,
                          except_abort=True)
    galsim.config.Process(config, logger=logger, new_params=new_params, njobs=3, job=2,
                          except_abort=True)
    galsim.config.Process(config, logger=logger, new_params=new_params, njobs=3, job=3,
                          except_abort=True)
    new_params = { 'output.noclobber' : True }
    galsim.config.Process(config, logger=logger, new_params=new_params, except_abort=True)
    for dir_num in range(1,5):
        for file_num in range(5):
            file_name = 'nfw%d/cluster%04d.fits'%(dir_num, file_num)
            truth_name = 'nfw%d/truth%04d.dat'%(dir_num, file_num)
            assert check_same('output/'+file_name , 'output_json/'+file_name)
            assert check_same('output/'+truth_name , 'output_json/'+truth_name)

@timer
@in_examples
def test_demo10():
    """Check that demo10 makes the same image using demo10.py and demo10.yaml.
    """
    import demo10
    print('Running demo10.py')
    demo10.main([])
    logger = logging.getLogger('galsim')
    logger.setLevel(logging.WARNING)
    config = galsim.config.ReadConfig('demo10.yaml', logger=logger)[0]
    print('Running demo10.yaml')
    galsim.config.Process(config, logger=logger, except_abort=True)
    assert check_same('output/power_spectrum.fits', 'output_yaml/power_spectrum.fits')

@timer
@in_examples
def test_demo11():
    """Check that demo11 makes the same image using demo11.py and demo11.yaml.
    """
    import demo11
    print('Running demo11.py')
    demo11.main([])
    logger = logging.getLogger('galsim')
    logger.setLevel(logging.WARNING)
    config = galsim.config.ReadConfig('demo11.yaml', logger=logger)[0]
    print('Running demo11.yaml')
    galsim.config.Process(config, logger=logger, except_abort=True)
    assert check_same('output/tabulated_power_spectrum.fits.fz',
                      'output_yaml/tabulated_power_spectrum.fits.fz')

@timer
@in_examples
def test_demo12():
    """Check that demo12 runs properly.
    """
    import demo12
    print('Running demo12.py')
    demo12.main([])
    # There is no demo12.yaml yet, so all this does is check for syntax errors in demo12.py.

@timer
@in_examples
def test_demo13():
    """Check that demo13 runs properly.
    """
    import demo13
    print('Running demo13.py')
    demo13.main(['nuse=3','ntot=100','filters=YJH'])
    # There is no demo13.yaml yet, so all this does is check for syntax errors in demo13.py.

@timer
@in_examples
def test_des():
    """Check that draw_psf makes the same image using draw_psf.py and draw_psf.yaml.

    Also run a few of the config files in the des directory to make sure they at least
    run to completion without errors.
    """
    original_dir = os.getcwd()
    try:
        os.chdir('des')
        new_dir = os.getcwd()
        if new_dir not in sys.path:
            sys.path.append(new_dir)

        import draw_psf
        print('Running draw_psf.py')
        draw_psf.main(['last=1'])
        logger = logging.getLogger('galsim')
        logger.setLevel(logging.WARNING)

        print('Running draw_psf.yaml')
        configs = galsim.config.ReadConfig('draw_psf.yaml', logger=logger)
        for config in configs:
            config['output']['nfiles'] = 1
            galsim.config.Process(config, logger=logger, except_abort=True)
        assert check_same('output/DECam_00154912_01_psfex_image.fits',
                          'output_yaml/DECam_00154912_01_psfex_image.fits')
        assert check_same('output/DECam_00154912_01_fitpsf_image.fits',
                          'output_yaml/DECam_00154912_01_fitpsf_image.fits')

        config = galsim.config.ReadConfig('meds.yaml', logger=logger)[0]
        config['output']['nfiles'] = 1
        config['output']['nobjects'] = 100
        config['gal']['items'][0]['gal_type'] = 'parametric'
        config['input']['cosmos_catalog']['file_name'] = '../data/real_galaxy_catalog_23.5_example.fits'
        del config['input']['cosmos_catalog']['sample']
        config['input']['des_wcs']['bad_ccds'] = list(range(2,63))  # All but CCD 1
        galsim.config.Process(config, logger=logger, except_abort=True)

        input_cosmos = config['input']['cosmos_catalog'] # Save example COSMOS catalog spec.
        config = galsim.config.ReadConfig('blend.yaml', logger=logger)[0]
        galsim.config.Process(config, logger=logger, except_abort=True)

        config = galsim.config.ReadConfig('blendset.yaml', logger=logger)[0]
        config['input']['cosmos_catalog'] = input_cosmos
        config['input']['des_psfex']['file_name']['num'] = 1
        galsim.config.Process(config, logger=logger, except_abort=True)

    finally:
        os.chdir(original_dir)

@timer
@in_examples
def test_great3():
    """Check that the great3 config files run properly.
    """
    original_dir = os.getcwd()
    try:
        os.chdir('great3')
        new_dir = os.getcwd()
        if new_dir not in sys.path:
            sys.path.append(new_dir)
        logger = logging.getLogger('galsim')
        logger.setLevel(logging.WARNING)
        # Some changes to speed up the run, since we mostly just want to check that all the
        # template and reject features work properly, which doesn't require many galaxies.
        p1 = { 'output.nfiles' : 1, 'output.noclobber' : False,
               'image.nx_tiles' : 1, 'image.ny_tiles' : 20 }
        p2 = p1.copy()
        p2['input.cosmos_catalog.file_name'] = '../data/real_galaxy_catalog_23.5_example.fits'
        p2['input.cosmos_catalog.sample']  = ''
        for file_name in ['cgc.yaml', 'cgc_psf.yaml',
                          'rgc.yaml', 'rgc_psf.yaml']:
            configs = galsim.config.ReadConfig(file_name, logger=logger)
            print('Running ',file_name)
            if 'psf' in file_name:
                new_params = p1
            else:
                new_params = p2
            for config in configs:
                galsim.config.Process(config, logger=logger, new_params=new_params,
                                      except_abort=True)
    finally:
        os.chdir(original_dir)

@timer
@in_examples
def test_psf_wf_movie():
    # Mock a command-line arguments object so we can run in the current process
    class Args(object):
        seed = 1
        r0_500 = 0.2
        nlayers = 3
        time_step = 0.03
        exptime = 0.3
        screen_size = 51.2
        screen_scale = 0.1
        max_speed = 20.0
        x = 0.0
        y = 0.0
        lam = 700.0
        diam = 4.0
        obscuration = 0.0
        nstruts = 0
        strut_thick = 0.05
        strut_angle = 0.0
        psf_nx = 512
        psf_scale = 0.005
        accumulate = False
        pad_factor = 1.0
        oversampling = 1.0
        psf_vmax = 0.0003
        wf_vmax = 50.0
        outfile = "output/test_psf_wf_movie.mp4"
    import psf_wf_movie
    try:
        psf_wf_movie.make_movie(Args)
        # Just checks that this runs, not the value of the output.
    except OSError as e:
        print(e)
        print('skipping test of psf_wf_movie.make_movie')

@timer
@in_examples
def test_fft_vs_geom_movie():
    # Mock a command-line arguments object so we can run in the current process
    class Args(object):
        seed = 1
        n = 10
        jmax = 15
        ell = 4.0
        sigma = 0.05
        r0_500 = 0.2
        nlayers = 3
        time_step = 0.025
        screen_size = 51.2
        screen_scale = 0.1
        max_speed = 20.0
        lam = 700.0
        diam = 4.0
        obscuration = 0.0
        nstruts = 0
        strut_thick = 0.05
        strut_angle = 0.0
        nx = 256
        size = 3.0
        accumulate = False
        pad_factor = 1.0
        oversampling = 1.0
        geom_oversampling = 1.0
        geom_nphot = 100000
        vmax = 1.e-3
        out = "output/test_fft_vs_geom_"
        do_fft=1
        do_geom=1
        make_movie=1
        make_plots=1
    import fft_vs_geom_movie
    try:
        fft_vs_geom_movie.make_movie(Args)
        # Just checks that this runs, not the value of the outputs.
    except OSError as e:
        print(e)
        print('skipping test of fft_vs_geom_movie.make_movie')


if __name__ == "__main__":
    remove_dir('output')
    remove_dir('output_yaml')
    remove_dir('output_json')
    test_demo1()
    test_demo2()
    test_demo3()
    test_demo4()
    test_demo5()
    test_demo6()
    test_demo7()
    test_demo8()
    test_demo9()
    test_demo10()
    test_demo11()
    test_demo12()
    test_demo13()
    test_des()
    test_great3()
    test_psf_wf_movie()
    test_fft_vs_geom_movie()
