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

from __future__ import print_function
import galsim

obj = galsim.Gaussian(flux=3.539e6, sigma=0.1)

rng = galsim.BaseDeviate(5678)
silicon = galsim.SiliconSensor(rng=rng)

im = obj.drawImage(nx=17, ny=17, scale=0.3, dtype=float, method='phot', sensor=silicon)
im.setCenter(0,0)
im.write('test.fits')

print('im min = ',im.array.min())
print('im max = ',im.array.max())
print('im(0,0) = ',im(0,0))
print('+- 1 along column: ',im(0,1),im(0,-1))
print('+- 1 along row:    ',im(1,0),im(-1,0))

area_image = silicon.calculate_pixel_areas(im)
area_image.write('area.fits')
print('area min = ',area_image.array.min())
print('area max = ',area_image.array.max())
print('area(0,0) = ',area_image(0,0))

print('+- 1 along column: ',area_image(0,1),area_image(0,-1))
print('+- 1 along row:    ',area_image(1,0),area_image(-1,0))

