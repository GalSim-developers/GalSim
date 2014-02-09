# Copyright 2012-2014 The GalSim developers:
# https://github.com/GalSim-developers
#
# This file is part of GalSim: The modular galaxy image simulation toolkit.
#
# GalSim is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# GalSim is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with GalSim.  If not, see <http://www.gnu.org/licenses/>
#
"""@file fds_test.py
I found these routines helpful in tracking down some errors about too many files being open.
I've fixed the errors, but I'll leave this module here in case it is useful for someone else
down the road.  I lifted the code from the following StackOverflow answers:
http://stackoverflow.com/questions/4814970/subprocess-check-output-doesnt-seem-to-exist-python-2-6-5

To use it do the following:

     import galsim.fds_test as fds

     ...

     try:
        ... Code that might raise OSError ...
     except OSError as e:
        print 'Caught ',e
        fds.printOpenFiles()
        raise

Of course, you can also do fds.printOpenFiles() elsewhere too for information.

You can also keep track of the number of open files and pipes with:

    print 'files, pipes = ',fds.openFiles()
"""


import __builtin__
openfiles = set()
oldfile = __builtin__.file
class newfile(oldfile):
    def __init__(self, *args):
        self.x = args[0]
        print "### OPENING %s ###" % str(self.x)            
        oldfile.__init__(self, *args)
        openfiles.add(self)

    def close(self):
        print "### CLOSING %s ###" % str(self.x)
        oldfile.close(self)
        openfiles.remove(self)

oldopen = __builtin__.open
def newopen(*args):
    return newfile(*args)
__builtin__.file = newfile
__builtin__.open = newopen

def getOpenFiles(do_print=False):
    '''
    return the number of open files and pipes for current process

    .. warning: will only work on UNIX-like os-es.
    '''
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

    procs = filter( 
            lambda s: s and s[ 0 ] == 'f' and s[1: ].isdigit(),
            procs.split( '\n' ) )
    if do_print:
        print 'procs = ',procs
        print 'nprocs = ',len(procs)
    return len(openfiles), len(procs) - len(openfiles)

def printOpenFiles():
    print "### %d OPEN FILES: [%s]" % (len(openfiles), ", ".join(f.x for f in openfiles))
    nopen = getOpenFiles(do_print=True)
    print "files, pipes = %d, %d"%nopen

