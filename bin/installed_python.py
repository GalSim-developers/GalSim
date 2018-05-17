# Copyright (c) 2012-2018 by the GalSim developers team on GitHub
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


# Just a quick little script that runs the version of python that the user
# chose with the SCons commands.  SCons will add the appropriate shebang
# at the top and make this an executable called installed_python.
# We use this to run the demo scripts with the correct version of python.

import sys
import subprocess
subprocess.call( [sys.executable] + sys.argv[1:] )
