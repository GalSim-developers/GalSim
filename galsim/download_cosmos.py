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
"""
A program to download the COSMOS RealGalaxy catalog for use with GalSim.
"""

from __future__ import print_function
from builtins import input
import os, sys, tarfile, subprocess, shutil, json
try:
    from urllib2 import urlopen
except:
    from urllib.request import urlopen

from .utilities import ensure_dir

script_name = 'galsim_download_cosmos'

def parse_args():
    """Handle the command line arguments using either argparse (if available) or optparse.
    """

    # Another potential option we might want to add is to download the smaller training sample
    # rather than the full 4 GB file.  Right now, this just downloads the larger catalog.

    # Short description strings common to both parsing mechanisms
    description = "This program will download the COSMOS RealGalaxy catalog and images\n"
    description += "and place them in the GalSim share directory so they can be used as\n "
    description += "the default files for the RealGalaxyCatalog class.\n"
    description += "See https://github.com/GalSim-developers/GalSim/wiki/RealGalaxy%20Data\n"
    description += "for more details about the files being downloaded."
    epilog = "Note: The unpacked files total almost 6 GB in size!\n"

    try:
        import argparse

        # Build the parser and add arguments
        parser = argparse.ArgumentParser(description=description, epilog=epilog, add_help=True)
        parser.add_argument(
            '-v', '--verbosity', type=int, action='store', default=2, choices=(0, 1, 2, 3),
            help='Integer verbosity level: min=0, max=3 [default=2]')
        parser.add_argument(
            '-f', '--force', action='store_const', default=False, const=True,
            help='Force overwriting the current file if one exists')
        parser.add_argument(
            '-q', '--quiet', action='store_const', default=False, const=True,
            help="Don't ask about re-downloading an existing file. (implied by verbosity=0)")
        parser.add_argument(
            '-u', '--unpack', action='store_const', default=False, const=True,
            help='Re-unpack the tar file if not downloading')
        parser.add_argument(
            '--save', action='store_const', default=False, const=True,
            help="Save the tarball after unpacking.")
        parser.add_argument(
            '-d', '--dir', action='store', default=None,
            help="Install into an alternate directory and link from the share/galsim directory")
        parser.add_argument(
            '-s', '--sample', action='store', default='25.2', choices=('23.5', '25.2'),
            help="Flux limit for sample to download; either 23.5 or 25.2")
        parser.add_argument(
            '--nolink', action='store_const', default=False, const=True,
            help="Don't link to the alternate directory from share/galsim")
        args = parser.parse_args()

    except ImportError:
        # Use optparse instead
        import optparse

        # Usage string not automatically generated for optparse, so generate it
        usage = "usage: %s [-h] [-v {0,1,2,3}] [-f] [-q] [-u] [-s] [-d] [--nolink]"%script_name
        # Build the parser
        parser = optparse.OptionParser(usage=usage, description=description, epilog=epilog)
        # optparse only allows string choices, so take verbosity as a string and make it int later
        parser.add_option(
            '-v', '--verbosity', type="choice", action='store', choices=('0', '1', '2', '3'),
            default='2', help='Integer verbosity level: min=0, max=3 [default=2]')
        parser.add_option(
            '-f', '--force', action='store_const', default=False, const=True,
            help='Force overwriting the current file if one exists')
        parser.add_option(
            '-q', '--quiet', action='store_const', default=False, const=True,
            help="Don't ask about re-downloading an existing file. (implied by verbosity=0)")
        parser.add_option(
            '-u', '--unpack', action='store_const', default=False, const=True,
            help='Re-unpack the tar file if not downloading')
        parser.add_option(
            '--save', action='store_const', default=False, const=True,
            help="Save the tarball after unpacking.")
        parser.add_option(
            '-d', '--dir', action='store', default=None,
            help="Install into an alternate directory and link from the share/galsim directory")
        parser.add_option(
            '-s', '--sample', type="choice", action='store', choices=('23.5', '25.2'),
            default='25.2', help="Flux limit for sample to download; either 23.5 or 25.2")
        parser.add_option(
            '--nolink', action='store_const', default=False, const=True,
            help="Don't link to the alternate directory from share/galsim")
        (args, posargs) = parser.parse_args()

        # Remembering to convert to an integer type
        args.verbosity = int(args.verbosity)

    if args.verbosity == 0:
        args.quiet = True

    # Return the args
    return args

# Based on recipe 577058: http://code.activestate.com/recipes/577058/
def query_yes_no(question, default="yes"):
    """Ask a yes/no question via input() and return their answer.

    "question" is a string that is presented to the user.
    "default" is the presumed answer if the user just hits <Enter>.
        It must be "yes" (the default), "no" or None (meaning
        an answer is required of the user).

    The "answer" return value is one of "yes" or "no".
    """
    valid = {"yes":"yes",   "y":"yes",  "ye":"yes",
             "no":"no",     "n":"no"}
    if default == None:
        prompt = " [y/n] "
    elif default == "yes":
        prompt = " [Y/n] "
    elif default == "no":
        prompt = " [y/N] "
    else:
        raise ValueError("invalid default answer: '%s'" % default)

    while 1:
        sys.stdout.write(question + prompt)
        choice = input().lower()
        if default is not None and choice == '':
            return default
        elif choice in valid.keys():
            return valid[choice]
        else:
            sys.stdout.write("Please respond with 'yes' or 'no' "\
                             "(or 'y' or 'n').\n")

def download(url, target, unpack_dir, args, logger):
    logger.warning('Downloading from url:\n  %s',url)
    logger.warning('Target location is:\n  %s',target)
    logger.info('')

    # See how large the file to be downloaded is.
    u = urlopen(url)
    meta = u.info()
    logger.debug("Meta information about url:\n%s",str(meta))
    file_size = int(meta.get("Content-Length"))
    file_name = os.path.basename(url)
    logger.info("Size of %s: %d MBytes" , file_name, file_size/1024**2)

    # Make sure the directory we want to put this file exists.
    ensure_dir(target)

    # Check if the file already exists and if it is the right size
    do_download = True
    if os.path.isfile(target):
        logger.info("")
        existing_file_size = os.path.getsize(target)
        if args.force:
            logger.info("Target file already exists.  Size = %d MBytes.  Forced re-download.",
                        existing_file_size/1024**2)
        elif file_size == existing_file_size:
            if args.quiet:
                logger.info("Target file already exists.  Not re-downloading.")
                do_download = False
            else:
                q = "Target file already exists.  Overwrite?"
                yn = query_yes_no(q, default='no')
                if yn == 'no':
                    do_download = False
        else:
            logger.warning("Target file already exists, but it seems to be either incomplete, "
                           "corrupt, or obsolete")
            if args.quiet:
                logger.info("Size of existing file = %d MBytes.  Re-downloading.",
                            existing_file_size/1024**2)
            else:
                q = "Size of existing file = %d MBytes.  Re-download?"%(existing_file_size/1024**2)
                yn = query_yes_no(q, default='yes')
                if yn == 'no':
                    do_download = False
    elif unpack_dir is not None and os.path.isdir(unpack_dir):
        logger.info("")

        # Check that this is the current version.
        meta_file = os.path.join(unpack_dir, 'meta.json')
        logger.debug('meta_file = %s',meta_file)
        if os.path.isfile(meta_file):
            logger.debug('meta_file exists')
            with open(meta_file) as fp:
                saved_meta_dict = json.load(fp)
                # Get rid of the unicode
                saved_meta_dict = dict([ (str(k),str(v)) for k,v in saved_meta_dict.items()])
            logger.debug("current meta information is %s",saved_meta_dict)
            meta_dict = dict(meta)
            logger.debug("url's meta information is %s",meta_dict)
            obsolete = False
            for k in meta_dict:
                if k.lower() == 'date':
                    continue  # This one isn't expected to match.
                elif k not in saved_meta_dict:
                    logger.debug("key %s is missing in saved meta information",k)
                    obsolete = True
                elif meta_dict[k] != saved_meta_dict[k]:
                    logger.debug("key %s differs: %s != %s",k,meta_dict[k],saved_meta_dict[k])
                    obsolete = True
                else:
                    logger.debug("key %s matches",k)
        else:
            logger.debug('meta_file does not exist')
            obsolete = True

        if obsolete:
            if args.quiet or args.force:
                logger.warning("The version currently on disk is obsolete.  "
                               "Downloading new version.")
            else:
                q = "The version currently on disk is obsolete.  Download new version?"
                yn = query_yes_no(q, default='yes')
                if yn == 'no':
                    do_download = False
        elif args.force:
            logger.info("Target file has already been downloaded and unpacked.  "
                        "Forced re-download.")
        elif args.quiet:
            logger.info("Target file has already been downloaded and unpacked.  "
                        "Not re-downloading.")
            do_download = False
            args.save = True  # Don't delete it!
        else:
            q = "Target file has already been downloaded and unpacked.  Re-download?"
            yn = query_yes_no(q, default='no')
            if yn == 'no':
                do_download = False
                args.save = True

    # The next bit is based on one of the answers here: (by PabloG)
    # http://stackoverflow.com/questions/22676/how-do-i-download-a-file-over-http-using-python
    # The progress feature in that answer is important here, since downloading such a large file
    # will take a while.
    if do_download:
        logger.info("")
        try:
            with open(target, 'wb') as f:
                file_size_dl = 0
                block_sz = 32 * 1024
                next_dot = file_size/100.  # For verbosity==1, the next size for writing a dot.
                while True:
                    buffer = u.read(block_sz)
                    if not buffer:
                        break

                    file_size_dl += len(buffer)
                    f.write(buffer)

                    # Status bar
                    if args.verbosity >= 2:
                        status = r"Downloading: %5d / %d MBytes  [%3.2f%%]" % (
                            file_size_dl/1024**2, file_size/1024**2, file_size_dl*100./file_size)
                        status = status + '\b'*len(status)
                        sys.stdout.write(status)
                        sys.stdout.flush()
                    elif args.verbosity >= 1 and file_size_dl > next_dot:
                        sys.stdout.write('.')
                        sys.stdout.flush()
                        next_dot += file_size/100.
            logger.info("Download complete.")
        except (IOError, OSError) as e:
            # Try to give a reasonable suggestion for some common IOErrors.
            logger.error("\n\nOSError: %s",str(e))
            if 'Permission denied' in str(e):
                logger.error("Rerun using sudo %s",script_name)
                logger.error("If this is not possible, you can download to an alternate location:")
                logger.error("    %s -d dir_name --nolink\n",script_name)
            elif 'Disk quota' in str(e) or 'No space' in str(e):
                logger.error("You might need to download this in an alternate location and link:")
                logger.error("    %s -d dir_name\n",script_name)
            raise

    return do_download, target, meta

def unpack(target, target_dir, unpack_dir, meta, args, logger):
    logger.info("Unpacking the tarball...")
    with tarfile.open(target) as tar:
        if args.verbosity >= 3:
            tar.list(verbose=True)
        elif args.verbosity >= 2:
            tar.list(verbose=False)
        tar.extractall(target_dir)

    # Write the meta information to a file, meta.json to mark what version this all is.
    meta_file = os.path.join(unpack_dir, 'meta.json')
    with open(meta_file,'w') as fp:
        json.dump(dict(meta), fp)

    logger.info("Extracted contents of tar file.")

def unzip(target, args, logger):
    logger.info("Unzipping file")
    subprocess.call(["gunzip", target])
    logger.info("Done")

def link_target(unpack_dir, link_dir, args, logger):
    logger.debug("Linking to %s from %s", unpack_dir, link_dir)
    if os.path.exists(link_dir):
        if os.path.islink(link_dir):
            # If it exists and is a link, we just remove it and relink without any fanfare.
            logger.debug("Removing existing link")
            os.remove(link_dir)
        else:
            # If it is not a link, we need to figure out what to do with it.
            if os.path.isdir(link_dir):
                # If it's a directory, probably want to keep it.
                logger.warning("%s already exists and is a directory.",link_dir)
                if args.force:
                    logger.warning("Removing the existing files to make the link.")
                elif args.quiet:
                    logger.warning("Link cannot be made.  (Use -f to force removal of existing dir.)")
                    return
                else:
                    q = "Remove the existing files to make the link?"
                    yn = query_yes_no(q, default='no')
                    if yn == 'no':
                        return
                shutil.rmtree(link_dir)
            else:
                # If it's not a directory, it's probably corrupt, so the default is to remove it.
                logger.warning("%s already exists, but strangely isn't a directory.",link_dir)
                if args.force or args.quiet:
                    logger.warning("Removing the existing file.")
                else:
                    q = "Remove the existing file?"
                    yn = query_yes_no(q, default='yes')
                    if yn == 'no':
                        return
                os.path.remove(link_dir)
    os.symlink(unpack_dir, link_dir)
    logger.info("Made link to %s from %s", unpack_dir, link_dir)

def main():
    from ._version import __version__ as version
    from .meta_data import share_dir

    args = parse_args()

    # Parse the integer verbosity level from the command line args into a logging_level string
    import logging
    logging_levels = { 0: logging.CRITICAL,
                       1: logging.WARNING,
                       2: logging.INFO,
                       3: logging.DEBUG }
    logging_level = logging_levels[args.verbosity]

    # Setup logging to go to sys.stdout or (if requested) to an output file
    logging.basicConfig(format="%(message)s", level=logging_level, stream=sys.stdout)
    logger = logging.getLogger('galsim')

    # Give diagnostic about GalSim version
    logger.debug("GalSim version: %s",version)
    logger.debug("This download script is: %s",__file__)
    logger.info("Type %s -h to see command line options.\n",script_name)

    # Some definitions:
    # share_dir is the base galsim share directory, e.g. /usr/local/share/galsim/
    # target_dir is where we will put the downloaded file, usually == share_dir.
    # unpack_dir is the directory that the tarball will unpack into.
    # url is the url from which we will download the tarball.
    # file_name is the name of the file to download, taken from the url.
    # target is the full path of the downloaded tarball

    if args.dir is not None:
        target_dir = args.dir
        link = not args.nolink
    else:
        target_dir = share_dir
        link = False

    url = "http://great3.jb.man.ac.uk/leaderboard/data/public/COSMOS_%s_training_sample.tar.gz"%(
            args.sample)
    file_name = os.path.basename(url)
    target = os.path.join(target_dir, file_name)
    unpack_dir = target[:-len('.tar.gz')]

    # Download the tarball
    new_download, target, meta = download(url, target, unpack_dir, args, logger)

    # Usually we unpack if we downloaded the tarball or if specified by the command line option.
    do_unpack = new_download or args.unpack

    # If the unpack dir is missing, then need to unpack
    if not os.path.exists(unpack_dir):
        do_unpack = True

    # But of course if there is no tarball, we can't unpack it
    if not os.path.isfile(target):
        do_unpack = False

    # If we have a downloaded tar file, ask if it should be re-unpacked.
    if not do_unpack and not args.quiet and os.path.isfile(target):
        logger.info("")
        q = "Tar file is already unpacked.  Re-unpack?"
        yn = query_yes_no(q, default='no')
        if yn == 'yes':
            do_unpack=True

    # Unpack the tarball
    if do_unpack:
        unpack(target, target_dir, unpack_dir, meta, args, logger)

    # Usually, we remove the tarball if we unpacked it and command line doesn't specify to save it.
    do_remove = do_unpack and not args.save

    # But if we didn't unpack it, and they didn't say to save it, ask if we should remove it.
    if os.path.isfile(target) and not do_remove and not args.save and not args.quiet:
        logger.info("")
        q = "Remove the tarball?"
        yn = query_yes_no(q, default='no')
        if yn == 'yes':
            do_remove = True

    # Remove the tarball
    if do_remove:
        logger.info("Removing the tarball to save space")
        os.remove(target)

    # If we are downloading to an alternate directory, we (usually) link to it from share/galsim
    if link:
        # Get the directory where this would normally have been unpacked.
        link_dir = os.path.join(share_dir, file_name)[:-len('.tar.gz')]
        link_target(unpack_dir, link_dir, args, logger)
