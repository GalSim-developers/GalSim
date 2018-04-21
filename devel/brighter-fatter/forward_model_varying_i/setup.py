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

# setup.py
# Run with python setup.py build_ext --inplace

from distutils.core import setup
from distutils.extension import Extension
import os

include_dirs = ["/System/Library/Frameworks/Python.framework/Versions/2.7/include/python2.7/","/System/Library/Frameworks/Python.framework/Versions/2.7/Extras/lib/python/numpy/core/include","."]

#import numpy
#include_dirs = [numpy.get_include(), '.'] # This works but get deprecated warning

library_dirs=['/usr/local/lib']
libraries=[]

files = ["forward.cpp", "forward_convert.cpp"]

os.environ['CC'] = 'g++'

setup(name="forward",    
      ext_modules=[
                    Extension("forward",files,
                    library_dirs=library_dirs,
                    libraries=libraries,
                    include_dirs=include_dirs,
                    depends=[]),
                    ]
     )
