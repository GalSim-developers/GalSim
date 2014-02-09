# Copyright 2012-2014 The GalSim developers:
# https://github.com/GalSim-developers
#
# This file is part of GalSim: The modular galaxy image simulation toolkit.
#
# GalSim is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# GalSim is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with GalSim.  If not, see <http://www.gnu.org/licenses/>
#


# For backwards compatibility.
# `galsim_yaml` is equivalent to `galsim -f yaml`, although in most cases,
# the `-f yaml` part is unnecessary.

import sys
import subprocess
print 'Note: galsim_yaml has been deprecated.  Use galsim instead.'
print 'Running galsim -f yaml',' '.join(sys.argv[1:])
print
subprocess.call( ['galsim','-f','yaml'] + sys.argv[1:] )
