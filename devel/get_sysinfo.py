#!/usr/bin/env python

import sys
import subprocess

print "Native byteorder = "+str(sys.byteorder)+" endian"
subprocess.check_call(['../bin/sizeof_SIFD'])

