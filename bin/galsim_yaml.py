
# For backwards compatibility.
# `galsim_yaml` is equivalent to `galsim -f yaml`, although in most cases,
# the `-f yaml` part is unnecessary.

import sys
import subprocess
subprocess.call( ['galsim','-f','yaml'] + sys.argv[1:] )
