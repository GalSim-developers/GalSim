# Copyright (c) 2012-2023 by the GalSim developers team on GitHub
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

import numpy as np

import galsim
from galsim_test_helpers import *


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


@timer
def test_shear_position_image_integration_pixelwcs():
    wcs = galsim.PixelScale(0.3)
    obj1 = galsim.Gaussian(sigma=3)
    obj2 = galsim.Gaussian(sigma=2)
    pos2 = galsim.PositionD(3, 5)
    sum = obj1 + obj2.shift(pos2)
    shear = galsim.Shear(g1=0.1, g2=0.18)
    im1 = galsim.Image(50, 50, wcs=wcs)
    sum.shear(shear).drawImage(im1, center=im1.center)

    # Equivalent to shear each object separately and drawing at the sheared position.
    im2 = galsim.Image(50, 50, wcs=wcs)
    obj1.shear(shear).drawImage(im2, center=im2.center)
    obj2.shear(shear).drawImage(
        im2,
        add_to_image=True,
        center=im2.center + wcs.toImage(pos2.shear(shear)),
    )

    print("err:", np.max(np.abs(im1.array - im2.array)))
    assert np.allclose(im1.array, im2.array, rtol=0, atol=5e-8)


@timer
def test_shear_position_image_integration_offsetwcs():
    wcs = galsim.OffsetShearWCS(
        0.3,
        galsim.Shear(g1=-0.2, g2=0.4),
        origin=galsim.PositionD(x=-1.1, y=2.7),
        world_origin=galsim.PositionD(x=0.6, y=-2.5),
    )

    obj1 = galsim.Gaussian(sigma=3)
    obj2 = galsim.Gaussian(sigma=2)
    pos2 = galsim.PositionD(3, 5)
    pos1 = galsim.PositionD(0, 0)
    sum_obj = obj1 + obj2.shift(pos2)
    shear = galsim.Shear(g1=0.1, g2=0.18)
    im1 = galsim.Image(50, 50, wcs=wcs)
    sum_obj.shear(shear).drawImage(im1, center=wcs.toImage(pos1))

    # Equivalent to shear each object separately and drawing at the sheared position.
    im2 = galsim.Image(50, 50, wcs=wcs)
    obj1.shear(shear).drawImage(im2, center=wcs.toImage(pos1))
    obj2.shear(shear).drawImage(
        im2,
        add_to_image=True,
        center=wcs.toImage(pos2.shear(shear)),
    )

    print("err:", np.max(np.abs(im1.array - im2.array)))
    assert np.allclose(im1.array, im2.array, rtol=0, atol=2e-7), (
        np.max(np.abs(im1.array - im2.array))
    )


if __name__ == "__main__":
    runtests(__file__)
