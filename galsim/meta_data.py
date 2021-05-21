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

# Note: This version of meta_data.py is correct for the setup.py installations, but not for
# SCons installations, which install into PREFIX/share/galsim using whatever PREFIX is specified.
# So `scons isntall` will overwrite this file with the correct values.

import os

if 'GALSIM_SHARE_DIR' in os.environ: # pragma: no cover  (Only tested in a subprocess)
    share_dir = os.environ['GALSIM_SHARE_DIR']
else:
    galsim_dir = os.path.split(os.path.realpath(__file__))[0]
    install_dir = os.path.split(galsim_dir)[0]
    share_dir = os.path.join(install_dir, 'galsim', 'share')
