# Copyright 2012, 2013 The GalSim developers:
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
import numpy as np
import os
import sys

from galsim_test_helpers import *

try:
    import galsim
except ImportError:
    path, filename = os.path.split(__file__)
    sys.path.append(os.path.abspath(os.path.join(path, "..")))
    import galsim
import galsim.catalog

def test_single_row():
    """Test that we can read catalogs with just one row (#394)
    """
    import time
    t1 = time.time()
    filename = "test394.txt"
    with open(filename, 'w') as f:
        f.write("3 4 5\n")
    cat = galsim.catalog.Catalog(filename, file_type='ascii')
    np.testing.assert_array_equal(
        cat.data, np.array([["3","4","5"]]),
        err_msg="galsim.catalog.Catalog.__init__ failed to read 1-row file")
    os.remove(filename)
    t2 = time.time()
    print 'time for %s = %.2f'%(funcname(),t2-t1)

if __name__ == "__main__":
    test_single_row()
