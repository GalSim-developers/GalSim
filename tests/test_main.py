# Copyright (c) 2012-2023 by the GalSim developers team on GitHub
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
import sys
import logging
from unittest import mock
import galsim
import galsim.main  # Not imported automatically
import galsim.__main__
from galsim_test_helpers import *

# This file tests the galsim executable.
# Note: Most of the real functional tests are elsewhere.  E.g. test_config*
#       This really just tests the bits related specifically to the functions in galsim/main.py.

@timer
def test_args():
    """Test the argument parsing of galsim executable.
    """
    # no args or -h prints description and exits
    print('This should print the description...')  # Can only check by inspection, not assert
    with assert_raises(SystemExit):
        args = galsim.main.parse_args([])
    print('This should print the description...')
    with assert_raises(SystemExit):
        args = galsim.main.parse_args(['-h'])

    # --version prints version and exits
    print('This should print the version...')
    with assert_raises(SystemExit):
        args = galsim.main.parse_args(['--version'])

    # Any other options, but missing config file, also prints description and exits
    print('This should print the description...')
    with assert_raises(SystemExit):
        args = galsim.main.parse_args(['-v','3','-n','5'])

    # Normal operation requires a config_file parameter
    config_file = 'test.yaml'
    args = galsim.main.parse_args([config_file])
    assert args.config_file == config_file
    assert args.verbosity == 1
    assert args.log_file is None
    assert args.log_format == '%(message)s'
    assert args.file_type is None
    assert args.module is None
    assert args.profile is False
    assert args.njobs == 1
    assert args.job == 1
    assert args.except_abort is False
    assert args.version is False
    assert args.variables == []

    # Check setting each of those to be something else
    args = galsim.main.parse_args([config_file, '-v', '3'])
    assert args.config_file == config_file
    assert args.verbosity == 3

    args = galsim.main.parse_args([config_file, '-l', 'test.log'])
    assert args.config_file == config_file
    assert args.log_file == 'test.log'

    args = galsim.main.parse_args([config_file, '--log_format', '%(asctime)s - %(name)s - %(levelname)s - %(message)s'])
    assert args.config_file == config_file
    assert args.log_format == '%(asctime)s - %(name)s - %(levelname)s - %(message)s'

    args = galsim.main.parse_args([config_file, '-f', 'yaml'])
    assert args.config_file == config_file
    assert args.file_type == 'yaml'
    args = galsim.main.parse_args([config_file, '-f', 'json'])
    assert args.config_file == config_file
    assert args.file_type == 'json'

    args = galsim.main.parse_args([config_file, '-m', 'des'])
    assert args.config_file == config_file
    assert args.module == ['des']
    args = galsim.main.parse_args([config_file, '-m', 'des', '-m', 'galsim_extra'])
    assert args.config_file == config_file
    assert args.module == ['des', 'galsim_extra']

    args = galsim.main.parse_args([config_file, '-p'])
    assert args.config_file == config_file
    assert args.profile is True

    args = galsim.main.parse_args([config_file, '-n', '3'])
    assert args.config_file == config_file
    assert args.njobs == 3

    args = galsim.main.parse_args([config_file, '-n', '3', '-j', '3'])
    assert args.config_file == config_file
    assert args.njobs == 3
    assert args.job == 3
    args = galsim.main.parse_args([config_file, '-n', '3', '-j', '1'])
    assert args.config_file == config_file
    assert args.njobs == 3
    assert args.job == 1

    args = galsim.main.parse_args([config_file, '-x'])
    assert args.config_file == config_file
    assert args.except_abort is True

    # --version with a config file doesn't exit, but still prints the version.
    print('This should print the version...')
    args = galsim.main.parse_args([config_file, '--version'])
    assert args.config_file == config_file

    # Additional arguments are accumulated in args.variables
    args = galsim.main.parse_args([config_file, 'output.nfiles=1', 'output.dir="."'])
    assert args.config_file == config_file
    assert args.variables == ['output.nfiles=1', 'output.dir="."']

    # Some invalid parameters
    with assert_raises(ValueError):
        galsim.main.parse_args([config_file, '-n', '0'])
    with assert_raises(ValueError):
        galsim.main.parse_args([config_file, '-n', '-1'])
    with assert_raises(ValueError):
        galsim.main.parse_args([config_file, '-n', '3', '-j', '4'])
    with assert_raises(ValueError):
        galsim.main.parse_args([config_file, '-n', '3', '-j', '0'])
    with assert_raises(ValueError):
        galsim.main.parse_args([config_file, '-j', '3'])

    # The ones handled by ArgumentParser raise SystemExit and print to stderr
    # To avoid ugly text output during pytest runs, redirect stderr to stdout for a moment.
    sys_stderr = sys.stderr
    sys.stderr = sys.stdout
    with assert_raises(SystemExit):
        galsim.main.parse_args([config_file, '-f', 'invalid'])
    with assert_raises(SystemExit):
        galsim.main.parse_args([config_file, '-v', '-1'])
    with assert_raises(SystemExit):
        galsim.main.parse_args([config_file, '-v', '4'])
    sys.stderr = sys_stderr

# Need to call these before each time make_logger is repeated.  Else duplicate handles.
def remove_handler():
    logger = logging.getLogger('galsim')
    for handler in logger.handlers:
        logger.removeHandler(handler)

@timer
def test_logger():
    args = galsim.main.parse_args(['test.yaml'])

    remove_handler()
    logger = galsim.main.make_logger(args)
    assert logger.getEffectiveLevel() == logging.WARNING
    print('handlers = ',logger.handlers)
    print('This should print...')
    logger.warning("Test warning")
    print('This should not print...')
    logger.info("Test info")
    logger.debug("Test debug")
    print('Done')

    args.verbosity = 3
    args.log_file = 'output/test_logger.log'
    remove_handler()
    logger = galsim.main.make_logger(args)
    print('handlers = ',logger.handlers)
    assert logger.getEffectiveLevel() == logging.DEBUG
    logger.warning("Test warning")
    logger.info("Test info")
    logger.debug("Test debug")

    with open(args.log_file) as f:
        assert f.readline().strip() == "Test warning"
        assert f.readline().strip() == "Test info"
        assert f.readline().strip() == "Test debug"

    args.verbosity = 0
    remove_handler()
    logger = galsim.main.make_logger(args)
    print('handlers = ',logger.handlers)
    assert logger.getEffectiveLevel() == logging.ERROR
    logger.warning("Test warning")
    logger.info("Test info")
    logger.debug("Test debug")

    with open(args.log_file) as f:
        assert f.readlines() == []

    args.verbosity = 3
    args.log_format = '%(levelname)s - %(message)s'
    remove_handler()
    logger = galsim.main.make_logger(args)
    print('handlers = ',logger.handlers)
    assert logger.getEffectiveLevel() == logging.DEBUG
    logger.warning("Test warning")
    logger.info("Test info")
    logger.debug("Test debug")

    with open(args.log_file) as f:
        assert f.readline().strip() == "WARNING - Test warning"
        assert f.readline().strip() == "INFO - Test info"
        assert f.readline().strip() == "DEBUG - Test debug"

@timer
def test_parse_variables():
    logger = logging.getLogger('test_main')
    logger.setLevel(logging.ERROR)

    # Empty list -> empty dict
    new_params = galsim.main.parse_variables([], logger)
    assert new_params == {}

    vars = ["output.nfiles=1", "output.dir='.'"]
    new_params = galsim.main.parse_variables(vars, logger)
    assert new_params['output.nfiles'] == 1
    assert new_params['output.dir'] == '.'

    # Lists or dicts will be parsed here
    vars = ["psf={'type':'Gaussian','sigma':0.4}", "output.skip=[0,0,0,0,0,1]"]
    new_params = galsim.main.parse_variables(vars, logger)
    assert new_params['psf'] == {'type' : 'Gaussian', 'sigma' : 0.4}
    assert new_params['output.skip'] == [0,0,0,0,0,1]

    # Things that don't parse properly are just returned verbatim as a string
    # (Presumably they would give an appropriate error later when they are used if the
    # string is not a valid value for whatever this is.)
    vars = ["psf={'type':'Gaussian' : 'sigma':0.4}"]
    new_params = galsim.main.parse_variables(vars, logger)
    assert new_params['psf'] == "{'type':'Gaussian' : 'sigma':0.4}"

    # Missing = is an error
    vars = ["output.nfiles","1"]
    assert_raises(galsim.GalSimError, galsim.main.parse_variables, vars, logger)
    vars = ["output.nfiles-1"]
    assert_raises(galsim.GalSimError, galsim.main.parse_variables, vars, logger)

    # Should work correctly if yaml isn't available.
    # Although it doesn't always parse quite as nicely. E.g. requires ", not ' for string quotes.
    with mock.patch.dict(sys.modules, {'yaml':None}):
        vars = ['psf={"type":"Gaussian","sigma":0.4}', 'output.skip=[0,0,0,0,0,1]']
        new_params = galsim.main.parse_variables(vars, logger)
        assert new_params['psf'] == {'type' : 'Gaussian', 'sigma' : 0.4}
        assert new_params['output.skip'] == [0,0,0,0,0,1]

@timer
def test_modules():
    config = {}
    galsim.main.add_modules(config, None)
    assert config == {}

    galsim.main.add_modules(config, [])
    assert config == {}

    galsim.main.add_modules(config, ['des'])
    assert config == {'modules' : ['des']}

    galsim.main.add_modules(config, ['galsim_extra'])
    assert config == {'modules' : ['des', 'galsim_extra']}

    galsim.main.add_modules(config, ['a','b','c'])
    assert config == {'modules' : ['des', 'galsim_extra', 'a', 'b', 'c']}

@timer
def test_process():
    # Test running an extremely simple config, just testing a few features of how
    # the process_config() function works.
    config = {
        'gal' : {
            'type' : 'Gaussian',
            'sigma' : 2,
        },
        'output' : {
            'dir' : 'output',
            'file_name' : 'test_main.fits',
        }
    }
    file_name = os.path.join('output', 'test_main.fits')
    if os.path.exists(file_name):
        os.remove(file_name)
    args = galsim.main.parse_args(['test.yaml','-v','1'])
    remove_handler()
    logger = galsim.main.make_logger(args)

    galsim.main.process_config([config], args, logger)
    assert os.path.exists(file_name)
    assert config['root'] == 'test'  # This is set automatically
    args.profile = True
    print('Should print profile:')
    galsim.main.process_config([config], args, logger)
    assert config['profile'] is True
    print('Done')

    remove_handler()
    os.remove(file_name)
    config_file = os.path.join('input','test.yaml')
    galsim.main.main([config_file,'-v','1'])
    assert os.path.exists(file_name)

    with mock.patch('sys.argv', ['galsim', config_file, '-v', '1']):
        remove_handler()
        os.remove(file_name)
        galsim.main.run_main()
        assert os.path.exists(file_name)

        remove_handler()
        os.remove(file_name)
        galsim.__main__.run_main()
        assert os.path.exists(file_name)

if __name__ == "__main__":
    runtests(__file__)
