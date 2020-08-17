# Copyright (c) 2012-2019 by the GalSim developers team on GitHub
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

from __future__ import print_function
import numpy as np

import galsim
from galsim_test_helpers import timer


@timer
def test_shear_position():
    # zero position
    shear = galsim.Shear(g1=-0.03, g2=0.7)
    pos = galsim.PositionD(x=0, y=0)
    sheared_pos = pos.shear(shear)
    np.testing.assert_almost_equal(
        sheared_pos.x, 0,
        err_msg="x coord is wrong for shear with zero pos"
    )
    np.testing.assert_almost_equal(
        sheared_pos.y, 0,
        err_msg="y coord is wrong for shear with zero pos"
    )

    # zero shear
    shear = galsim.Shear(g1=0, g2=0)
    pos = galsim.PositionD(x=0.1, y=-0.5)
    sheared_pos = pos.shear(shear)
    np.testing.assert_almost_equal(
        sheared_pos.x, pos.x,
        err_msg="x coord is wrong for zero shear with pos"
    )
    np.testing.assert_almost_equal(
        sheared_pos.y, pos.y,
        err_msg="y coord is wrong for zero shear with pos"
    )

    # full thing
    shear = galsim.Shear(g1=-0.1, g2=0.1)
    pos = galsim.PositionD(x=0.1, y=-0.5)
    sheared_pos = pos.shear(shear)
    mat = shear.getMatrix()
    np.testing.assert_almost_equal(
        sheared_pos.x,
        pos.x * mat[0, 0] + pos.y * mat[0, 1],
        err_msg="x coord is wrong for shear with pos"
    )
    np.testing.assert_almost_equal(
        sheared_pos.y,
        pos.x * mat[1, 0] + pos.y * mat[1, 1],
        err_msg="y coord is wrong for shear with pos"
    )


if __name__ == "__main__":
    test_shear_position()
