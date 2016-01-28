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
"""
A program to download the COSMOS RealGalaxy catalog for use with GalSim.
"""

import os, sys, urllib2, tarfile, subprocess

# Since this will be installed in the same directory as our galsim executable,
# we need to do the same trick about changing the path so it imports the real
# galsim module, not that executable.
temp = sys.path[0]
sys.path = sys.path[1:]
import galsim
sys.path = [temp] + sys.path

script_name = os.path.basename(__file__)


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
            '-s', '--save', action='store_const', default=False, const=True,
            help="Save the tarball after unpacking.")
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
            '-s', '--save', action='store_const', default=False, const=True,
            help="Save the tarball after unpacking.")
        (args, posargs) = parser.parse_args()

        # Remembering to convert to an integer type
        args.verbosity = int(args.verbosity)

    if args.verbosity == 0:
        args.quiet = True

    # Return the args
    return args

# Based on recipe 577058: http://code.activestate.com/recipes/577058/
def query_yes_no(question, default="yes"):
    """Ask a yes/no question via raw_input() and return their answer.

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
        choice = raw_input().lower()
        if default is not None and choice == '':
            return default
        elif choice in valid.keys():
            return valid[choice]
        else:
            sys.stdout.write("Please respond with 'yes' or 'no' "\
                             "(or 'y' or 'n').\n")

def download(url, target, unpack_dir, args, logger):

    logger.info('Downloading from url:\n  %s',url)
    logger.info('Target location is:\n  %s',target)

    # See how large the file to be downloaded is.
    u = urllib2.urlopen(url)
    meta = u.info()
    logger.debug("\nMeta information about url:\n%s",str(meta))
    file_size = int(meta.getheaders("Content-Length")[0]) / 1024**2
    file_name = os.path.basename(url)
    logger.info("\nSize of %s: %d MBytes" , file_name, file_size)

    # Check if the file already exists and if it is the right size
    do_download = True
    if os.path.isfile(target):
        logger.info("")
        existing_file_size = os.path.getsize(target) / 1024**2
        if args.force:
            logger.info("Target file already exists.  Size = %d MBytes.  Forced re-download.",
                        existing_file_size)
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
            logger.warn("Target file already exists, but it seems to be incomplete.")
            if args.quiet:
                logger.warn("Size of existing file = %d MBytes.  Re-downloading.",
                            existing_file_size)
            else:
                q = "Size of existing file = %d MBytes.  Re-download?"%(existing_file_size)
                yn = query_yes_no(q, default='yes')
                if yn == 'no':
                    do_download = False
    elif unpack_dir is not None and os.path.isdir(unpack_dir):
        logger.info("")
        if args.force:
            logger.info("Target file has already been downloaded and unpacked.  "+
                        "Forced re-download.")
        else:
            if args.quiet:
                logger.info("Target file has already been downloaded and unpacked.  "+
                            "Not re-downloading.")
                do_download = False
                args.save = True  # Don't try to re-delete it!
            else:
                q = "Target file has already been downloaded and unpacked.  Re-download?"
                yn = query_yes_no(q, default='no')
                if yn == 'no':
                    do_download = False
                    args.save = True

    # The next bit is based on one of the answers here: (by PabloG)
    # http://stackoverflow.com/questions/22676/how-do-i-download-a-file-over-http-using-python
    # The progress bar feature in that answer is important here, since this will take a while,
    # since the file is so big.
    if do_download:
        logger.info("")
        with open(target, 'wb') as f:
            file_size_dl = 0
            block_sz = 32 * 1024
            while True:
                buffer = u.read(block_sz)
                if not buffer:
                    break

                file_size_dl += len(buffer) / 1024
                f.write(buffer)

                # Status bar
                if args.verbosity >= 2:
                    fsdl = file_size_dl / 1024
                    status = r"Downloading: %5d / %d MBytes  [%3.2f%%]" % (
                        fsdl, file_size, fsdl * 100. / file_size)
                    status = status + chr(8)*(len(status)+1)
                    print status,
                    sys.stdout.flush()
        logger.info("Download complete.")

    return do_download, target

def unpack(new_download, target, share_dir, args, logger):
    if new_download or args.unpack:
        logger.info("Unpacking the tarball...")
        with tarfile.open(target) as tar:
            if args.verbosity >= 3:
                tar.list(verbose=True)
            elif args.verbosity >= 2:
                tar.list(verbose=False)
            tar.extractall(share_dir)
        logger.info("Extracted contents of tar file.")

    if not args.save:
        logger.info("Removing the tarball to save space")
        os.remove(target)

def unzip(target, args, logger):
    logger.info("Unzipping file")
    subprocess.call(["gunzip", target])
    logger.info("Done")


def main():
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
    logger.debug("GalSim version: %s",galsim.__version__)
    logger.debug("This download script is: %s",__file__)
    logger.info("Type %s -h to see command line options.",script_name)

    url = "http://great3.jb.man.ac.uk/leaderboard/data/public/COSMOS_23.5_training_sample.tar.gz"
    share_dir = galsim.meta_data.share_dir

    file_name = os.path.basename(url)
    target = os.path.join(share_dir, file_name)
    unpack_dir = target[:-len('.tar.gz')]

    url2 = "http://great3.jb.man.ac.uk/leaderboard/data/public/real_galaxy_galsim_selection.fits.gz"
    file_name2 = os.path.basename(url2)
    target2 = os.path.join(unpack_dir, file_name2)
    target2_nogz = target2[:-len('.gz')]

    if os.path.exists(unpack_dir) and not os.path.exists(target2_nogz):
        logger.info("Detected old version without %s",file_name2)
        logger.info("Only downloading %s",file_name2)

        download(url2, target2, None, args, logger)
        unzip(target2, args, logger)

    else:
        new_download, target = download(url, target, unpack_dir, args, logger)
        unpack(new_download, target, share_dir, args, logger)

if __name__ == "__main__":
    main()
