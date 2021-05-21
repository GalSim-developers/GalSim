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

from ._galsim import j0, j1, jv, kv, yv, iv, j0_root, jv_root

# Alias the "n" names, which don't get any advantage from being implemented differently,
# so we only have the generic nu implementation.  But to match scipy.special, we also
# allow the user to write jn, kn, or yn.  No in, since that's a reserved word, of course.
jn = jv
kn = kv
yn = yv
