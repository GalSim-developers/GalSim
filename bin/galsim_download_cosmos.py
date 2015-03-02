# Copyright (c) 2012-2014 by the GalSim developers team on GitHub
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

import os, sys, urllib2

# Since this will be installed in the same directory as our galsim executable,
# we need to do the same trick about changing the path so it imports the real
# galsim module, not that executable.
temp = sys.path[0]
sys.path = sys.path[1:]
import galsim
sys.path = [temp] + sys.path


def parse_args():
    """Handle the command line arguments using either argparse (if available) or optparse.
    """

    # Another potential option we might want to add is to download the smaller training sample
    # rather than the full 4 GB file.  Right now, this just downloads the larger catalog.

    # Short description strings common to both parsing mechanisms
    version_str = "GalSim Version %s"%galsim.version
    description = "galsim_download_cosmos will download the COSMOS RealGalaxy catalog "
    description += "and place it in the GalSim share directory so it can be used as "
    description += "the default file_name for the RealGalaxyCatalog class. "
    description += "See https://github.com/GalSim-developers/GalSim/wikiealGalaxy%20Data "
    description += "for details about where this program is downloading from."
    
    try:
        import argparse
        
        # Build the parser and add arguments
        parser = argparse.ArgumentParser(description=description, add_help=True)
        parser.add_argument(
            '-v', '--verbosity', type=int, action='store', default=2, choices=(0, 1, 2, 3),
            help='integer verbosity level: min=0, max=3 [default=2]')
        parser.add_argument(
            '-f', '--force', action='store_const', default=False, const=True,
            help='force overwriting the current file if one exists')
        parser.add_argument(
            '-q', '--quiet', action='store_const', default=False, const=True,
            help="don't ask about re-downloading an existing file.")
        parser.add_argument(
            '--version', action='store_const', default=False, const=True,
            help='show the version of GalSim')
        args = parser.parse_args()

        if args.version:
            print version_str

    except ImportError:
        # Use optparse instead
        import optparse

        # Usage string not automatically generated for optparse, so generate it
        usage = """usage: galsim_download_cosmos [-h] [-v {0,1,2,3}] [-l LOG_FILE] [--version] """
        # Build the parser
        parser = optparse.OptionParser(usage=usage, description=description)
        # optparse only allows string choices, so take verbosity as a string and make it int later
        parser.add_option(
            '-v', '--verbosity', type="choice", action='store', choices=('0', '1', '2', '3'),
            default='2', help='integer verbosity level: min=0, max=3 [default=2]')
        parser.add_option(
            '-f', '--force', action='store_const', default=False, const=True,
            help='force overwriting the current file if one exists')
        parser.add_argument(
            '-q', '--quiet', action='store_const', default=False, const=True,
            help="don't ask about re-downloading an existing file.")
        parser.add_option(
            '--version', action='store_const', default=False, const=True,
            help='show the version of GalSim')

        # Remembering to convert to an integer type
        args.verbosity = int(args.verbosity) 

        if args.version:
            print version_str

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
    
    url = "http://great3.jb.man.ac.uk/leaderboard/data/public/COSMOS_23.5_training_sample.tar.gz"
    file_name = os.path.basename(url)
    share_dir = galsim.meta_data.share_dir
    target = os.path.join(share_dir, file_name)

    logger.info('Downloading from url:\n  %s',url)
    logger.info('Target location is %s',target)

    # See how large the file to be downloaded is.
    u = urllib2.urlopen(url)
    meta = u.info()
    file_size = int(meta.getheaders("Content-Length")[0]) / 1024**2
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
                logger.info("Target file already exists.")
                do_download = False
            else:
                yn = query_yes_no("Target file already exists.  Overwrite?", default='no')
                if yn == 'no':
                    do_download = False
        else:
            print "Target file already exists, but it seems to be corrupt."
            print "Size of existing file = %d MBytes   "%(existing_file_size),
            if args.quiet:
                logger.info("Re-downloading.")
            else:
                yn = query_yes_no("Re-download?", default='yes')
                if yn == 'no':
                    do_download = False

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


if __name__ == "__main__":
    main()
