
# For backwards compatibility.
# `galsim_json` is equivalent to `galsim -f json`, although if the config
# file has an extension that starts with `.j`, the `-f json` part is
# unnecessary.

import sys
import subprocess
print 'Note: galsim_json has been deprecated.  Use galsim instead.'
print 'Running galsim -f json',' '.join(sys.argv[1:])
print
subprocess.call( ['galsim','-f','json'] + sys.argv[1:] )
