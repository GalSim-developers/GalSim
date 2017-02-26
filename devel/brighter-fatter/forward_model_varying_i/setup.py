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
