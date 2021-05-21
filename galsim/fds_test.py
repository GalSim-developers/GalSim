# Copyright (c) 2012-2021 by the GalSim developers team on GitHub
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
I found these routines helpful in tracking down some errors about too many files being open.
I've fixed the errors, but I'll leave this module here in case it is useful for someone else
down the road.  I lifted the code from the following StackOverflow answers:
http://stackoverflow.com/questions/4814970/subprocess-check-output-doesnt-seem-to-exist-python-2-6-5

To use it do the following::

    >>> import galsim.fds_test as fds
    >>> ...
    >>> try:
    >>>     [... Code that might raise OSError ...]
    >>> except OSError as e:
    >>>     print 'Caught ',e
    >>>     fds.printOpenFiles()
    >>>     raise

Of course, you can also do fds.printOpenFiles() elsewhere too for information.

You can also keep track of the number of open files and pipes with::

    >>> print 'files, pipes = ',fds.openFiles()
"""

from __future__ import print_function
import builtins

openfiles = set()
oldfile = builtins.file
class newfile(oldfile):
    def __init__(self, *args):
        self.x = args[0]
        print("### OPENING %s ###" % str(self.x))
        oldfile.__init__(self, *args)
        openfiles.add(self)

    def close(self):
        print("### CLOSING %s ###" % str(self.x))
        oldfile.close(self)
        openfiles.remove(self)

oldopen = builtins.open
def newopen(*args):
    return newfile(*args)
builtins.file = newfile
builtins.open = newopen

def getOpenFiles(do_print=False):
    """Return the number of open files and pipes for current process

    .. warning: will only work on a UNIX-like OS.
    """
    import subprocess
    import os

    pid = os.getpid()
    # check_output is py2.7 only:
    #procs = subprocess.check_output( [ "lsof", '-w', '-Ff', "-p", str( pid ) ] )
    p = subprocess.Popen(['lsof', '-w', '-Ff', '-p', str(pid)],
                         stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    procs = p.communicate()[0]
    p.stdout.close()
    p.stderr.close()

    procs = [s for s in procs.split( '\n' ) if s and s[ 0 ] == 'f' and s[1: ].isdigit()]
    if do_print:
        print('procs = ',procs)
        print('nprocs = ',len(procs))
    return len(openfiles), len(procs) - len(openfiles)

def printOpenFiles():
    print("### %d OPEN FILES: [%s]" % (len(openfiles), ", ".join(f.x for f in openfiles)))
    nopen = getOpenFiles(do_print=True)
    print("files, pipes = %d, %d"%nopen)

