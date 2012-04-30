"""
A few adjustments to the Image classes at the python layer
"""
from . import _galsim

def Image_setitem(self, key, value)
    self.subImage(key).copyFrom(value)

def Image_getitem(self, key)
    return self.subImage(key)

# inject these as methods of Image classes
for Class in _galsim.Image.itervalues():
    Class.__setitem__ = Image_setitem
    Class.__getitem__ = Image_getitem

for Class in _galsim.ImageView.itervalues():
    Class.__setitem__ = Image_setitem
    Class.__getitem__ = Image_getitem

for Class in _galsim.ConstImageView.itervalues():
    Class.__getitem__ = Image_getitem

del Class    # cleanup public namespace
