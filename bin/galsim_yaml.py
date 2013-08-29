
# For backwards compatibility.
# `galsim_yaml` is equivalent to `galsim -f yaml`, although in most cases,
# the `-f yaml` part is unnecessary.

import sys
import subprocess
print 'Note: galsim_yaml has been deprecated.  Use galsim instead.'
print 'Running galsim -f yaml',' '.join(sys.argv[1:])
print
subprocess.call( ['galsim','-f','yaml'] + sys.argv[1:] )
