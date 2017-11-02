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

import galsim
import numpy
from galsim.deprecated import depr

# Don't forget to also remove the MetaImage stuff from image.py when we get rid of this
# round of deprecations.  (It doesn't really work to insert a metaclass after the class
# is already made, so I couldn't put the MetaImage stuff here.)

def ImageViewS(*args, **kwargs):
    """A deprecated alias for galsim.Image(..., dtype=numpy.int16)
    """
    depr('ImageView[type] or ImageViewS', 1.1, 'ImageS')
    kwargs['dtype'] = numpy.int16
    return galsim.Image(*args, **kwargs)

def ImageViewI(*args, **kwargs):
    """A deprecated alias for galsim.Image(..., dtype=numpy.int32)
    """
    depr('ImageView[type] or ImageViewI', 1.1, 'ImageI')
    kwargs['dtype'] = numpy.int32
    return galsim.Image(*args, **kwargs)

def ImageViewF(*args, **kwargs):
    """A deprecated alias for galsim.Image(..., dtype=numpy.float32)
    """
    depr('ImageView[type] or ImageViewF', 1.1, 'ImageF')
    kwargs['dtype'] = numpy.float32
    return galsim.Image(*args, **kwargs)

def ImageViewD(*args, **kwargs):
    """A deprecated alias for galsim.Image(..., dtype=numpy.float64)
    """
    depr('ImageView[type] or ImageViewD', 1.1, 'ImageD')
    kwargs['dtype'] = numpy.float64
    return galsim.Image(*args, **kwargs)

def ConstImageViewS(*args, **kwargs):
    """A deprecated alias for galsim.Image(..., dtype=numpy.int16, make_const=True)
    """
    depr('ConstImageView[type] or ConstImageViewS', 1.1, 'ImageS(..., make_const=True)')
    kwargs['dtype'] = numpy.int16
    kwargs['make_const'] = True
    return galsim.Image(*args, **kwargs)

def ConstImageViewI(*args, **kwargs):
    """A deprecated alias for galsim.Image(..., dtype=numpy.int32, make_const=True)
    """
    depr('ConstImageView[type] or ConstImageViewI', 1.1, 'ImageI(..., make_const=True)')
    kwargs['dtype'] = numpy.int32
    kwargs['make_const'] = True
    return galsim.Image(*args, **kwargs)

def ConstImageViewF(*args, **kwargs):
    """A deprecated alias for galsim.Image(..., dtype=numpy.float32, make_const=True)
    """
    depr('ConstImageView[type] or ConstImageViewF', 1.1, 'ImageF(..., make_const=True)')
    kwargs['dtype'] = numpy.float32
    kwargs['make_const'] = True
    return galsim.Image(*args, **kwargs)

def ConstImageViewD(*args, **kwargs):
    """A deprecated alias for galsim.Image(..., dtype=numpy.float64, make_const=True)
    """
    depr('ConstImageView[type] or ConstImageViewD', 1.1, 'ImageD(..., make_const=True)')
    kwargs['dtype'] = numpy.float64
    kwargs['make_const'] = True
    return galsim.Image(*args, **kwargs)

galsim.ImageViewS = ImageViewS
galsim.ImageViewI = ImageViewI
galsim.ImageViewF = ImageViewF
galsim.ImageViewD = ImageViewD
galsim.ConstImageViewS = ConstImageViewS
galsim.ConstImageViewI = ConstImageViewI
galsim.ConstImageViewF = ConstImageViewF
galsim.ConstImageViewD = ConstImageViewD

galsim.ImageView = {
    numpy.int16 : ImageViewS,
    numpy.int32 : ImageViewI,
    numpy.float32 : ImageViewF,
    numpy.float64 : ImageViewD
}

galsim.ConstImageView = {
    numpy.int16 : ConstImageViewS,
    numpy.int32 : ConstImageViewI,
    numpy.float32 : ConstImageViewF,
    numpy.float64 : ConstImageViewD
}

def at(self, x, y):
    """This method is a synonym for im(x,y).  It is a bit faster than im(x,y), since GalSim
    does not have to parse the different options available for __call__.  (i.e. im(x,y) or
    im(pos) or im(x=x,y=y))
    """
    depr('image.at(x,y)', 1.5, 'image.getValue(x,y)')
    return self.getValue(x,y)

galsim.Image.at = at

def Image_getXMin(self):
    depr('image.getXMin()', 1.5, 'image.xmin')
    return self.xmin

def Image_getXMax(self):
    depr('image.getXMax()', 1.5, 'image.xmax')
    return self.xmax

def Image_getYMin(self):
    depr('image.getYMin()', 1.5, 'image.ymin')
    return self.ymin

def Image_getYMax(self):
    depr('image.getYMax()', 1.5, 'image.ymax')
    return self.ymax

def Image_getBounds(self):
    depr('image.getBounds()', 1.5, 'image.bounds')
    return self.bounds

galsim.Image.getXMin = Image_getXMin
galsim.Image.getXMax = Image_getXMax
galsim.Image.getYMin = Image_getYMin
galsim.Image.getYMax = Image_getYMax
galsim.Image.getBounds = Image_getBounds
