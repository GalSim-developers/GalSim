
# Just a quick little script that runs the version of python that the user
# chose with the SCons commands.  SCons will add the appropriate shebang
# at the top and make this an executable called installed_python.
# We use this to run the demo scripts with the correct version of python.

import sys
import subprocess
subprocess.call( [sys.executable] + sys.argv[1:] )
