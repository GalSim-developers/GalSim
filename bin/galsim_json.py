# Copyright (c) 2012-2017 by the GalSim developers team on GitHub
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

# For backwards compatibility.
# `galsim_json` is equivalent to `galsim -f json`, although if the config
# file has an extension that starts with `.j`, the `-f json` part is
# unnecessary.

from __future__ import print_function

import sys
import subprocess
print('Note: galsim_json has been deprecated.  Use galsim instead.')
print('Running galsim -f json',' '.join(sys.argv[1:]))
print()
subprocess.call( ['galsim','-f','json'] + sys.argv[1:] )
