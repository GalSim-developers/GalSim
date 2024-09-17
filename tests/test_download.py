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
import shutil
from unittest import mock
import galsim
import galsim.download_cosmos  # Not imported automatically
from galsim_test_helpers import *

# This file tests the galsim_download_cosmos executable.
# We use a lock of mocking to do these tests.  We mock user input, urlopen, and more.
# The directory fake_cosmos has a tarball for the 23.5 sample and a directory for 25.2.
# So we use these separately at times when we need one or the other to exist.

@timer
def test_args():
    """Test the argument parsing
    """
    # -h prints description and exits
    print('This should print the description...')  # Can only check by inspection, not assert
    with assert_raises(SystemExit):
        args = galsim.download_cosmos.parse_args(['-h'])

    # Check defaults:
    args = galsim.download_cosmos.parse_args([])
    assert args.verbosity == 2
    assert args.force is False
    assert args.quiet is False
    assert args.unpack is False
    assert args.save is False
    assert args.dir is None
    assert args.sample == '25.2'
    assert args.nolink is False

    # Check setting each of those to be something else
    args = galsim.download_cosmos.parse_args(['-v', '3'])
    assert args.verbosity == 3
    args = galsim.download_cosmos.parse_args(['-v', '0'])
    assert args.verbosity == 0

    args = galsim.download_cosmos.parse_args(['-f'])
    assert args.force is True

    args = galsim.download_cosmos.parse_args(['-q'])
    assert args.quiet is True

    args = galsim.download_cosmos.parse_args(['-u'])
    assert args.unpack is True

    args = galsim.download_cosmos.parse_args(['--save'])
    assert args.save is True

    args = galsim.download_cosmos.parse_args(['-d','~/share'])
    assert args.dir == '~/share'

    args = galsim.download_cosmos.parse_args(['-s','23.5'])
    assert args.sample == '23.5'
    args = galsim.download_cosmos.parse_args(['-s','25.2'])
    assert args.sample == '25.2'

    args = galsim.download_cosmos.parse_args(['--nolink'])
    assert args.nolink is True

    # Some invalid parameters
    # To avoid ugly text output during pytest runs, redirect stderr to stdout for a moment.
    sys_stderr = sys.stderr
    sys.stderr = sys.stdout
    with assert_raises(SystemExit):
        galsim.download_cosmos.parse_args(['-s', '25.9'])
    with assert_raises(SystemExit):
        galsim.download_cosmos.parse_args(['-v', '-1'])
    with assert_raises(SystemExit):
        galsim.download_cosmos.parse_args(['-v', '4'])
    sys.stderr = sys_stderr

# global for the bleh and delay functions
count = 0

@timer
def test_query():
    """Test the query_yes_no function

    Need to mock the input function for this
    """
    from galsim.download_cosmos import query_yes_no

    def bleh():
        global count
        count += 1
        return 'y' if count % 5 == 0 else 'bleh'

    def delay_y():
        global count
        count += 1
        return 'y' if count % 5 == 0 else ''

    def delay_n():
        global count
        count += 1
        return 'n' if count % 5 == 0 else ''

    with mock.patch('galsim.download_cosmos.get_input', return_value='y'):
        assert query_yes_no('', 'yes') == 'yes'
    with mock.patch('galsim.download_cosmos.get_input', return_value='yes'):
        assert query_yes_no('', 'yes') == 'yes'
    with mock.patch('galsim.download_cosmos.get_input', return_value='n'):
        assert query_yes_no('', 'yes') == 'no'
    with mock.patch('galsim.download_cosmos.get_input', return_value='no'):
        assert query_yes_no('', 'yes') == 'no'
    with mock.patch('galsim.download_cosmos.get_input', return_value=''):
        assert query_yes_no('', 'yes') == 'yes'
    with mock.patch('galsim.download_cosmos.get_input', bleh):
        assert query_yes_no('', 'yes') == 'yes'

    with mock.patch('galsim.download_cosmos.get_input', return_value='y'):
        assert query_yes_no('', 'no') == 'yes'
    with mock.patch('galsim.download_cosmos.get_input', return_value='yes'):
        assert query_yes_no('', 'no') == 'yes'
    with mock.patch('galsim.download_cosmos.get_input', return_value='n'):
        assert query_yes_no('', 'no') == 'no'
    with mock.patch('galsim.download_cosmos.get_input', return_value='no'):
        assert query_yes_no('', 'no') == 'no'
    with mock.patch('galsim.download_cosmos.get_input', return_value=''):
        assert query_yes_no('', 'no') == 'no'
    with mock.patch('galsim.download_cosmos.get_input', bleh):
        assert query_yes_no('', 'yes') == 'yes'

    with mock.patch('galsim.download_cosmos.get_input', return_value='y'):
        assert query_yes_no('', None) == 'yes'
    with mock.patch('galsim.download_cosmos.get_input', return_value='yes'):
        assert query_yes_no('', None) == 'yes'
    with mock.patch('galsim.download_cosmos.get_input', return_value='n'):
        assert query_yes_no('', None) == 'no'
    with mock.patch('galsim.download_cosmos.get_input', return_value='no'):
        assert query_yes_no('', None) == 'no'
    with mock.patch('galsim.download_cosmos.get_input', delay_n):
        assert query_yes_no('', None) == 'no'
    with mock.patch('galsim.download_cosmos.get_input', delay_y):
        assert query_yes_no('', None) == 'yes'
    with mock.patch('galsim.download_cosmos.get_input', bleh):
        assert query_yes_no('', None) == 'yes'

    with assert_raises(ValueError):
        query_yes_no('', 'invalid')

# Need to call these before each time make_logger is repeated.  Else duplicate handles.
def remove_handler():
    logger = logging.getLogger('galsim')
    for handler in logger.handlers:
        logger.removeHandler(handler)

@timer
def test_names():
    """Test the get_names function
    """
    from galsim.download_cosmos import get_names

    args = galsim.download_cosmos.parse_args([])
    remove_handler()
    logger = galsim.download_cosmos.make_logger(args)
    url, target, target_dir, link_dir, unpack_dir, do_link = get_names(args, logger)
    assert url == 'https://zenodo.org/record/3242143/files/COSMOS_25.2_training_sample.tar.gz'
    assert target_dir == galsim.meta_data.share_dir
    assert do_link is False
    assert target == os.path.join(target_dir, 'COSMOS_25.2_training_sample.tar.gz')
    assert unpack_dir == os.path.join(target_dir, 'COSMOS_25.2_training_sample')
    assert link_dir == os.path.join(galsim.meta_data.share_dir, 'COSMOS_25.2_training_sample')

    args = galsim.download_cosmos.parse_args(['-d','~/share','-s','23.5'])
    url, target, target_dir, link_dir, unpack_dir, do_link = get_names(args, logger)
    assert url == 'https://zenodo.org/record/3242143/files/COSMOS_23.5_training_sample.tar.gz'
    assert target_dir == os.path.expanduser('~/share')
    assert do_link is True
    assert target == os.path.join(target_dir, 'COSMOS_23.5_training_sample.tar.gz')
    assert unpack_dir == os.path.join(target_dir, 'COSMOS_23.5_training_sample')
    assert link_dir == os.path.join(galsim.meta_data.share_dir, 'COSMOS_23.5_training_sample')

    args = galsim.download_cosmos.parse_args(['-d','share','--nolink'])
    url, target, target_dir, link_dir, unpack_dir, do_link = get_names(args, logger)
    assert url == 'https://zenodo.org/record/3242143/files/COSMOS_25.2_training_sample.tar.gz'
    assert target_dir == 'share'
    assert do_link is False
    assert target == os.path.join(target_dir, 'COSMOS_25.2_training_sample.tar.gz')
    assert unpack_dir == os.path.join(target_dir, 'COSMOS_25.2_training_sample')
    assert link_dir == os.path.join(galsim.meta_data.share_dir, 'COSMOS_25.2_training_sample')


class fake_urlopen:
    err = None

    # We don't want to actually check things on the internet.  So this class fakes
    # up the real urlopen function.
    def __init__(self, url):
        self.n = 10
        pass
    def info(self):
        return {
            "Server": "nginx/1.16.1",
            "Content-Type": "application/octet-stream",
            "Content-Length": "728",
            "Connection": "close",
            "Content-MD5": "e05cfe60c037c645d61ac70545cc2a99",
            "Content-Security-Policy": "default-src 'none';",
            "X-Content-Type-Options": "nosniff",
            "X-Download-Options": "noopen",
            "X-Permitted-Cross-Domain-Policies": "none",
            "X-Frame-Options": "sameorigin",
            "X-XSS-Protection": "1; mode=block",
            "Content-Disposition": "attachment; filename=COSMOS_25.2_training_sample.tar.gz",
            "ETag": "\"md5:e05cfe60c037c645d61ac70545cc2a99\"",
            "Last-Modified": "Sun, 31 May 2020 02:19:18 GMT",
            "Date": "Thu, 11 Jun 2020 16:06:07 GMT",
            "Accept-Ranges": "none",
            "X-RateLimit-Limit": "60",
            "X-RateLimit-Remaining": "59",
            "X-RateLimit-Reset": "1591891628",
            "Retry-After": "60",
            "Strict-Transport-Security": "max-age=0",
            "Referrer-Policy": "strict-origin-when-cross-origin",
            "Set-Cookie": "session=3765c5a1d5211e53_5ee2566f",
            "X-Session-ID": "3765c5a1d5211e53_5ee2566f",
            "X-Request-ID": "2b720f14bdd71a29031a5cb415b391f8"
        }
    def read(self, block_sz):
        if self.err:
            raise OSError(self.err)
        if self.n:
            self.n -= 1
            return b'x' * 80  # ignore block_sz
        else:
            return b''

@timer
def test_check():
    """Test the get_meta and check_existing functions

    The latter of these is really the most interesting, and has the most bits
    of anything in the script that are worth checking with unit tests.
    """
    from galsim.download_cosmos import get_names, get_meta, check_existing

    args = galsim.download_cosmos.parse_args(['-d','fake_cosmos','-q','-v','3'])
    remove_handler()
    logger = galsim.download_cosmos.make_logger(args)
    url, target, target_dir, link_dir, unpack_dir, do_link = get_names(args, logger)

    # Check get_meta
    with mock.patch('galsim.download_cosmos.urlopen', fake_urlopen):
        meta = get_meta(url, args, logger)
        assert meta['Content-Length'] == "728"
        assert meta['Content-MD5'] == "e05cfe60c037c645d61ac70545cc2a99"

    # File already exists and is current.
    do_download = check_existing(target, unpack_dir, meta, args, logger)
    assert do_download is False

    # Some changes imply it's obsolete
    meta['Server'] =  "nginx/1.23.1"
    meta['X-Content-Type-Options'] = "sniff"
    meta['Last-Modified'] = "Tue, 12 Mar 2019 08:12:12 GMT"
    meta['Date'] = "Sun, 14 Jun 2020 20:00:00 GMT"
    meta['X-RateLimit-Remaining'] = "31"
    meta['Retry-After'] = "120"
    meta['Set-Cookie'] =  "session=2b720f14bdd71a29031a5cb415b391f8"
    do_download = check_existing(target, unpack_dir, meta, args, logger)
    assert do_download is False

    # Force download anyway
    args.quiet = False
    args.force = True
    do_download = check_existing(target, unpack_dir, meta, args, logger)
    assert do_download is True

    # Ask whether to re-download
    args.force = False
    with mock.patch('galsim.download_cosmos.get_input', return_value='y'):
        do_download = check_existing(target, unpack_dir, meta, args, logger)
    assert do_download is True
    with mock.patch('galsim.download_cosmos.get_input', return_value='n'):
        do_download = check_existing(target, unpack_dir, meta, args, logger)
    assert do_download is False

    # Meta data is obsolete
    meta1 = meta.copy()
    meta1['Content-Length'] = "9999"
    meta1['Content-MD5'] = "f05cfe60c037c645d61ac70545cc2a99"
    args.quiet = True
    do_download = check_existing(target, unpack_dir, meta1, args, logger)
    assert do_download is True

    # If they change the name of the checksum key, we consider it obsolete.
    meta2 = meta1.copy()
    meta2['Content-New-MD5'] = "e05cfe60c037c645d61ac70545cc2a99"
    del meta2['Content-MD5']
    del meta2['ETag']
    do_download = check_existing(target, unpack_dir, meta2, args, logger)
    assert do_download is True

    args.quiet = False
    args.force = True
    do_download = check_existing(target, unpack_dir, meta1, args, logger)
    assert do_download is True

    args.force = False
    with mock.patch('galsim.download_cosmos.get_input', return_value='y'):
        do_download = check_existing(target, unpack_dir, meta1, args, logger)
    assert do_download is True
    with mock.patch('galsim.download_cosmos.get_input', return_value='n'):
        do_download = check_existing(target, unpack_dir, meta1, args, logger)
    assert do_download is False

    # Meta data is missing
    args.quiet = True
    do_download = check_existing(target, 'output', meta, args, logger)
    assert do_download is True

    # Tarball is present, but wrong size
    args = galsim.download_cosmos.parse_args(['-d','fake_cosmos','-s','23.5','-q'])
    url, target, target_dir, link_dir, unpack_dir, do_link = get_names(args, logger)
    do_download = check_existing(target, 'output', meta1, args, logger)
    assert do_download is True

    args.quiet = False
    args.force = True
    do_download = check_existing(target, 'output', meta1, args, logger)
    assert do_download is True

    args.force = False
    with mock.patch('galsim.download_cosmos.get_input', return_value='y'):
        do_download = check_existing(target, unpack_dir, meta1, args, logger)
    assert do_download is True
    with mock.patch('galsim.download_cosmos.get_input', return_value='n'):
        do_download = check_existing(target, unpack_dir, meta1, args, logger)
    assert do_download is False

    # Tarball is present, and correct size
    args.quiet = True
    do_download = check_existing(target, 'output', meta, args, logger)
    assert do_download is False

    args.quiet = False
    args.force = True
    do_download = check_existing(target, 'output', meta, args, logger)
    assert do_download is True

    args.force = False
    with mock.patch('galsim.download_cosmos.get_input', return_value='y'):
        do_download = check_existing(target, unpack_dir, meta, args, logger)
    assert do_download is True
    with mock.patch('galsim.download_cosmos.get_input', return_value='n'):
        do_download = check_existing(target, unpack_dir, meta, args, logger)
    assert do_download is False

    # Tarball and unpack_dir both missing
    args = galsim.download_cosmos.parse_args(['-d','input'])
    url, target, target_dir, link_dir, unpack_dir, do_link = get_names(args, logger)
    do_download = check_existing(target, unpack_dir, meta, args, logger)
    assert do_download is True


@timer
def test_download():
    """Test the download function

    This one is a little silly.  It's almost completely mocked.  But we can at least check
    that there are no bugs that would raise an exception of some sort.
    """
    from galsim.download_cosmos import get_names, get_meta, download

    args = galsim.download_cosmos.parse_args(['-d','output','-q'])
    remove_handler()
    logger = galsim.download_cosmos.make_logger(args)
    url, target, target_dir, link_dir, unpack_dir, do_link = get_names(args, logger)

    with mock.patch('galsim.download_cosmos.urlopen', fake_urlopen):
        meta = get_meta(url, args, logger)

        print('Start download with verbosity = 2')
        download(True, url, target, meta, args, logger)

        print('Start download with verbosity = 1')
        args.verbosity = 1
        download(True, url, target, meta, args, logger)

        print('Start download with verbosity = 3')
        args.verbosity = 3
        download(True, url, target, meta, args, logger)

        print('Start download with verbosity = 0')
        args.verbosity = 0
        download(True, url, target, meta, args, logger)

        print("Don't download")
        download(False, url, target, meta, args, logger)

        fake_urlopen.err = 'Permission denied'
        with CaptureLog() as cl:
            assert_raises(OSError, download, True, url, target, meta, args, cl.logger)
        assert "Rerun using sudo" in cl.output

        fake_urlopen.err = 'Disk quota exceeded'
        with CaptureLog() as cl:
            assert_raises(OSError, download, True, url, target, meta, args, cl.logger)
        assert "You might need to download this in an alternate location" in cl.output

        fake_urlopen.err = 'gack'
        assert_raises(OSError, download, True, url, target, meta, args, logger)
    fake_urlopen.err = None

@timer
def test_unpack():
    """Test the check_unpack and unpack functions
    """
    from galsim.download_cosmos import get_names, check_unpack, unpack

    # If we downloaded the file, then we usually want to unpack
    args = galsim.download_cosmos.parse_args(['-d','fake_cosmos','-s','23.5','-q'])
    remove_handler()
    logger = galsim.download_cosmos.make_logger(args)
    url, target, target_dir, link_dir, unpack_dir, do_link = get_names(args, logger)
    meta = fake_urlopen(url).info()

    print('unpack_dir = ',unpack_dir)
    if os.path.exists(unpack_dir):
        shutil.rmtree(unpack_dir)

    # Regular case, downloaded file and not unpacked yet
    do_unpack = check_unpack(True, unpack_dir, target, args)
    assert do_unpack is True

    # If we didn't download, but tarball exists, still unpack
    do_unpack = check_unpack(False, unpack_dir, target, args)
    assert do_unpack is True

    # Now unpack it
    print('unpack with verbose = 2:')
    unpack(True, target, target_dir, unpack_dir, meta, args, logger)

    shutil.rmtree(unpack_dir)
    print('unpack with verbose = 3:')
    args.verbosity = 3
    unpack(True, target, target_dir, unpack_dir, meta, args, logger)

    shutil.rmtree(unpack_dir)
    print('unpack with verbose = 1:')
    args.verbosity = 1
    unpack(True, target, target_dir, unpack_dir, meta, args, logger)

    print("Don't unpack")
    unpack(False, target, target_dir, unpack_dir, meta, args, logger)

    # If it is already unpacked, probably don't unpack it
    do_unpack = check_unpack(False, unpack_dir, target, args)
    assert do_unpack is False

    # Unless we expressly say to on the command line
    args.unpack = True
    do_unpack = check_unpack(False, unpack_dir, target, args)
    assert do_unpack is True

    # Or if we downloaded a new tarball, we will unpack it.
    args.unpack = False
    do_unpack = check_unpack(True, unpack_dir, target, args)
    assert do_unpack is True

    # Or if not quiet, it will ask whether to re-unpack the tarball
    args.quiet = False
    with mock.patch('galsim.download_cosmos.get_input', return_value='y'):
        do_unpack = check_unpack(False, unpack_dir, target, args)
    assert do_unpack is True
    with mock.patch('galsim.download_cosmos.get_input', return_value='n'):
        do_unpack = check_unpack(False, unpack_dir, target, args)
    assert do_unpack is False

    # Finally, if the tarball doesn't exist, then we can't unpack it.
    target2 = target.replace('23.5','25.2')
    do_unpack = check_unpack(True, unpack_dir, target2, args)
    assert do_unpack is False

@timer
def test_remove():
    """Test the check_remove and remove_tarball function
    """
    from galsim.download_cosmos import get_names, check_remove, remove_tarball

    args = galsim.download_cosmos.parse_args(['-d','fake_cosmos','-s','23.5','-q'])
    remove_handler()
    logger = galsim.download_cosmos.make_logger(args)
    url, target, target_dir, link_dir, unpack_dir, do_link = get_names(args, logger)

    # Normally, we remove the tarball if we unpacked it.
    do_remove = check_remove(True, target, args)
    assert do_remove is True

    # Or if we say to save it, then save it
    args.save = True
    do_remove = check_remove(True, target, args)
    assert do_remove is False

    # If we didn't unpack it, probably don't delete it
    args.save = False
    do_remove = check_remove(False, target, args)
    assert do_remove is False

    # But ask if not quiet to be sure
    args.quiet = False
    with mock.patch('galsim.download_cosmos.get_input', return_value='y'):
        do_remove = check_remove(False, target, args)
    assert do_remove is True
    with mock.patch('galsim.download_cosmos.get_input', return_value='n'):
        do_remove = check_remove(False, target, args)
    assert do_remove is False

    target1 = target + '.tar'
    with open(target1,'w') as f:
        f.write('blah')
    assert os.path.isfile(target1)
    remove_tarball(False, target1, logger)
    assert os.path.isfile(target1)
    remove_tarball(True, target1, logger)
    assert not os.path.isfile(target1)


@timer
def test_link():
    """Test the link_cosmos function
    """
    from galsim.download_cosmos import get_names, make_link

    args = galsim.download_cosmos.parse_args(['-d','fake_cosmos','-q'])
    remove_handler()
    logger = galsim.download_cosmos.make_logger(args)
    url, target, target_dir, link_dir, unpack_dir, do_link = get_names(args, logger)
    link_dir = os.path.join('output', 'COSMOS_25.2_training_sample')

    # If link doesn't exist yet, make it.
    if os.path.lexists(link_dir):
        os.unlink(link_dir)
    assert not os.path.lexists(link_dir)
    make_link(True, unpack_dir, link_dir, args, logger)
    assert os.path.exists(link_dir)
    assert os.path.islink(link_dir)

    # If link already exists, remove and remake
    make_link(True, unpack_dir, link_dir, args, logger)
    assert os.path.exists(link_dir)
    assert os.path.islink(link_dir)

    # If do_link is False, don't make it
    os.unlink(link_dir)
    make_link(False, unpack_dir, link_dir, args, logger)
    assert not os.path.exists(link_dir)

    # If link exists, but is a directory, don't remove it
    os.mkdir(link_dir)
    assert os.path.isdir(link_dir)
    make_link(True, unpack_dir, link_dir, args, logger)
    assert os.path.exists(link_dir)
    assert os.path.isdir(link_dir)
    assert not os.path.islink(link_dir)

    # Unless force
    args.force = True
    make_link(True, unpack_dir, link_dir, args, logger)
    assert os.path.exists(link_dir)
    assert os.path.islink(link_dir)

    # Or ask
    args.force = False
    args.quiet = False
    os.unlink(link_dir)
    os.mkdir(link_dir)
    with mock.patch('galsim.download_cosmos.get_input', return_value='n'):
        make_link(True, unpack_dir, link_dir, args, logger)
    assert os.path.isdir(link_dir)
    assert not os.path.islink(link_dir)
    with mock.patch('galsim.download_cosmos.get_input', return_value='y'):
        make_link(True, unpack_dir, link_dir, args, logger)
    assert os.path.islink(link_dir)

    # If it's a broken link, remove and relink
    os.unlink(link_dir)
    os.symlink('invalid', link_dir)
    assert os.path.lexists(link_dir)
    assert os.path.islink(link_dir)
    assert not os.path.exists(link_dir)
    make_link(True, unpack_dir, link_dir, args, logger)
    assert os.path.exists(link_dir)
    assert os.path.islink(link_dir)

    # If it's a file, probably remove and relink
    os.unlink(link_dir)
    with open(link_dir,'w') as f: f.write('blah')
    args.quiet = True
    assert os.path.exists(link_dir)
    assert not os.path.islink(link_dir)
    make_link(True, unpack_dir, link_dir, args, logger)
    assert os.path.exists(link_dir)
    assert os.path.islink(link_dir)

    os.unlink(link_dir)
    with open(link_dir,'w') as f: f.write('blah')
    args.quiet = False
    args.force = True
    assert os.path.exists(link_dir)
    assert not os.path.islink(link_dir)
    make_link(True, unpack_dir, link_dir, args, logger)
    assert os.path.exists(link_dir)
    assert os.path.islink(link_dir)

    # But ask if appropriate
    os.unlink(link_dir)
    with open(link_dir,'w') as f: f.write('blah')
    args.force = False
    assert os.path.exists(link_dir)
    assert not os.path.islink(link_dir)
    with mock.patch('galsim.download_cosmos.get_input', return_value='n'):
        make_link(True, unpack_dir, link_dir, args, logger)
    assert os.path.exists(link_dir)
    assert not os.path.islink(link_dir)
    with mock.patch('galsim.download_cosmos.get_input', return_value='y'):
        make_link(True, unpack_dir, link_dir, args, logger)
    assert os.path.exists(link_dir)
    assert os.path.islink(link_dir)

@timer
def test_full():
    """Test the full script
    """
    link_dir1 = os.path.join('output', 'COSMOS_23.5_training_sample')
    link_dir2 = os.path.join('output', 'COSMOS_25.2_training_sample')
    if os.path.lexists(link_dir1):
        os.unlink(link_dir1)
    if os.path.lexists(link_dir2):
        os.unlink(link_dir2)

    with mock.patch('galsim.download_cosmos.share_dir', 'output'), \
         mock.patch('galsim.download_cosmos.urlopen', fake_urlopen):

        remove_handler()
        assert not os.path.islink(link_dir1)
        galsim.download_cosmos.main(['-d','fake_cosmos','-q','-s','23.5','--save'])
        assert os.path.islink(link_dir1)

        remove_handler()
        assert not os.path.islink(link_dir2)
        with mock.patch('sys.argv', ['galsim_download_cosmos', '-d', 'fake_cosmos', '-q']):
            galsim.download_cosmos.run_main()
        assert os.path.islink(link_dir2)


if __name__ == "__main__":
    runtests(__file__)
