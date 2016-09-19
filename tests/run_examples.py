# Copyright (c) 2012-2016 by the GalSim developers team on GitHub
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
import numpy as np
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

@timer
@in_examples
def test_demo2():
    """Check that demo2 makes the same image using demo2.py and demo2.yaml.
    """
    import demo2
    import check_diff
    print('Running demo2.py')
    demo2.main([])
    logging.basicConfig(format="%(message)s", level=logging.INFO, stream=sys.stdout)
    logger = logging.getLogger('galsim')
    config = galsim.config.ReadConfig('demo2.yaml', logger=logger)[0]
    print('Running demo2.yaml')
    galsim.config.Process(config, logger=logger)
    assert check_diff.same('output/demo2.fits', 'output_yaml/demo2.fits')


@timer
@in_examples
def test_demo3():
    """Check that demo3 makes the same image using demo3.py and demo3.yaml.
    """
    import demo3
    import check_diff
    print('Running demo3.py')
    demo3.main([])
    logging.basicConfig(format="%(message)s", level=logging.INFO, stream=sys.stdout)
    logger = logging.getLogger('galsim')
    config = galsim.config.ReadConfig('demo3.yaml', logger=logger)[0]
    print('Running demo3.yaml')
    galsim.config.Process(config, logger=logger)
    assert check_diff.same('output/demo3.fits', 'output_yaml/demo3.fits')
    assert check_diff.same('output/demo3_epsf.fits', 'output_yaml/demo3_epsf.fits')

@timer
@in_examples
def test_demo4():
    """Check that demo4 makes the same image using demo4.py and demo4.yaml.
    """
    import demo4
    import check_diff
    print('Running demo4.py')
    demo4.main([])
    logging.basicConfig(format="%(message)s", level=logging.INFO, stream=sys.stdout)
    logger = logging.getLogger('galsim')
    config = galsim.config.ReadConfig('demo4.yaml', logger=logger)[0]
    print('Running demo4.yaml')
    galsim.config.Process(config, logger=logger)
    assert check_diff.same('output/multi.fits', 'output_yaml/multi.fits')

@timer
@in_examples
def test_demo5():
    """Check that demo5 makes the same image using demo5.py and demo5.yaml.
    """
    import demo5
    import check_diff
    print('Running demo5.py')
    demo5.main([])
    logging.basicConfig(format="%(message)s", level=logging.INFO, stream=sys.stdout)
    logger = logging.getLogger('galsim')
    config = galsim.config.ReadConfig('demo5.yaml', logger=logger)[0]
    print('Running demo5.yaml')
    galsim.config.Process(config, logger=logger)
    assert check_diff.same('output/g08_psf.fits', 'output_yaml/g08_psf.fits')
    assert check_diff.same('output/g08_gal.fits', 'output_yaml/g08_gal.fits')

@timer
@in_examples
def test_demo6():
    """Check that demo6 makes the same image using demo6.py and demo6.yaml.
    """
    import demo6
    import check_diff
    print('Running demo6.py')
    demo6.main([])
    logging.basicConfig(format="%(message)s", level=logging.INFO, stream=sys.stdout)
    logger = logging.getLogger('galsim')
    configs = galsim.config.ReadConfig('demo6.yaml', logger=logger)
    print('Running demo6.yaml pass #1')
    galsim.config.Process(configs[0], logger=logger)
    print('Running demo6.yaml pass #2')
    galsim.config.Process(configs[1], logger=logger)
    assert check_diff.same('output/psf_real.fits', 'output_yaml/psf_real.fits')
    assert check_diff.same('output/cube_real.fits', 'output_yaml/cube_real.fits')

@timer
@in_examples
def test_demo7():
    """Check that demo7 makes the same image using demo7.py and demo7.yaml.
    """
    import demo7
    import check_diff
    import gzip
    import shutil
    print('Running demo7.py')
    demo7.main([])
    logging.basicConfig(format="%(message)s", level=logging.INFO, stream=sys.stdout)
    logger = logging.getLogger('galsim')
    config = galsim.config.ReadConfig('demo7.yaml', logger=logger)[0]
    print('Running demo7.yaml')
    galsim.config.Process(config, logger=logger)
    with gzip.open('output/cube_phot.fits.gz', 'rb') as f_in, \
         open('output/cube_phot.fits', 'wb') as f_out:
        shutil.copyfileobj(f_in, f_out)
    with gzip.open('output_yaml/cube_phot.fits.gz', 'rb') as f_in, \
         open('output_yaml/cube_phot.fits', 'wb') as f_out:
        shutil.copyfileobj(f_in, f_out)
    assert check_diff.same('output/cube_phot.fits', 'output_yaml/cube_phot.fits')

@timer
@in_examples
def test_demo8():
    """Check that demo8 makes the same image using demo8.py and demo8.yaml.
    """
    import demo8
    import check_diff
    print('Running demo8.py')
    demo8.main([])
    logging.basicConfig(format="%(message)s", level=logging.INFO, stream=sys.stdout)
    logger = logging.getLogger('galsim')
    configs = galsim.config.ReadConfig('demo8.yaml', logger=logger)
    print('Running demo8.yaml pass #1')
    galsim.config.Process(configs[0], logger=logger)
    print('Running demo8.yaml pass #2')
    galsim.config.Process(configs[1], logger=logger)
    assert check_diff.same('output/bpd_single.fits', 'output_yaml/bpd_single.fits')
    assert check_diff.same('output/bpd_multi.fits', 'output_yaml/bpd_multi.fits')

@timer
@in_examples
def test_demo9():
    """Check that demo9 makes the same image using demo9.py and demo9.yaml.
    """
    # For this one, we'll use the json file instead, partly just to get coverage of the
    # JSON parsing functionality, but also because the changes to the base config
    # are pretty minor, so we can effect them with the new_params option.
    import demo9
    import check_diff
    print('Running demo9.py')
    demo9.main([])
    logging.basicConfig(format="%(message)s", level=logging.INFO, stream=sys.stdout)
    logger = logging.getLogger('galsim')
    config = galsim.config.ReadConfig('json/demo9.json', logger=logger)[0]
    print('Running demo9.json')
    new_params = { 'output.skip' : { 'type' : 'List', 'items' : [0,0,0,0,0,1] } }
    galsim.config.Process(config, logger=logger, new_params=new_params, njobs=3, job=1)
    galsim.config.Process(config, logger=logger, new_params=new_params, njobs=3, job=2)
    galsim.config.Process(config, logger=logger, new_params=new_params, njobs=3, job=3)
    new_params = { 'output.noclobber' : True }
    galsim.config.Process(config, logger=logger, new_params=new_params)
    for dir_num in range(1,5):
        for file_num in range(5):
            file_name = 'nfw%d/cluster%04d.fits'%(dir_num, file_num)
            truth_name = 'nfw%d/truth%04d.dat'%(dir_num, file_num)
            assert check_diff.same('output/'+file_name , 'output_json/'+file_name)
            assert check_diff.same('output/'+truth_name , 'output_json/'+truth_name)

@timer
@in_examples
def test_demo10():
    """Check that demo10 makes the same image using demo10.py and demo10.yaml.
    """
    import demo10
    import check_diff
    print('Running demo10.py')
    demo10.main([])
    logging.basicConfig(format="%(message)s", level=logging.INFO, stream=sys.stdout)
    logger = logging.getLogger('galsim')
    config = galsim.config.ReadConfig('demo10.yaml', logger=logger)[0]
    print('Running demo10.yaml')
    galsim.config.Process(config, logger=logger)
    assert check_diff.same('output/power_spectrum.fits', 'output_yaml/power_spectrum.fits')

@timer
@in_examples
def test_demo11():
    """Check that demo11 makes the same image using demo11.py and demo11.yaml.
    """
    import demo11
    import check_diff
    print('Running demo11.py')
    demo11.main([])
    logging.basicConfig(format="%(message)s", level=logging.INFO, stream=sys.stdout)
    logger = logging.getLogger('galsim')
    config = galsim.config.ReadConfig('demo11.yaml', logger=logger)[0]
    print('Running demo11.yaml')
    galsim.config.Process(config, logger=logger)
    assert check_diff.same('output/tabulated_power_spectrum.fits.fz',
                           'output_yaml/tabulated_power_spectrum.fits.fz')


if __name__ == "__main__":
    shutil.rmtree('../examples/output')
    shutil.rmtree('../examples/output_yaml')
    shutil.rmtree('../examples/output_json')
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
