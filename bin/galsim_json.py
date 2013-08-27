
# For backwards compatibility.
# `galsim_json` is equivalent to `galsim -f json`, although if the config
# file has an extension that starts with `.j`, the `-f json` part is
# unnecessary.

import sys
import subprocess
subprocess.call( ['galsim','-f','json'] + sys.argv[1:] )
