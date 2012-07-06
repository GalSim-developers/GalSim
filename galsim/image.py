"""
A few adjustments to the Image classes at the python layer
"""
from . import _galsim

def Image_setitem(self, key, value):
    self.subImage(key).copyFrom(value)

def Image_getitem(self, key):
    return self.subImage(key)

def Image_add(self, other):
    ret = self.copy()
    ret += other
    return ret

def Image_iadd(self, other):
    try:
        self.array[:,:] += other.array
    except AttributeError:
        self.array[:,:] += other
    return self

def Image_sub(self, other):
    ret = self.copy()
    ret -= other
    return ret

def Image_isub(self, other):
    try:
        self.array[:,:] -= other.array
    except AttributeError:
        self.array[:,:] -= other
    return self

def Image_mul(self, other):
    ret = self.copy()
    ret *= other
    return ret

def Image_imul(self, other):
    try:
        self.array[:,:] *= other.array
    except AttributeError:
        self.array[:,:] *= other
    return self

def Image_div(self, other):
    ret = self.copy()
    ret /= other
    return ret

def Image_idiv(self, other):
    try:
        self.array[:,:] /= other.array
    except AttributeError:
        self.array[:,:] /= other
    return self

def Image_copy(self):
    # self can be an Image or an ImageView, but the return type needs to be an Image.
    # So use the array.dtype.type attribute to get the type of the underlying data,
    # which in turn can be used to index our Image dictionary:
    return _galsim.Image[self.array.dtype.type](self)

# Some function to enable pickling of images
def ImageView_getinitargs(self):
    return self.array, self.xMin, self.yMin

# An image is really pickled as an ImageView
def Image_getstate(self):
    return self.array, self.xMin, self.yMin

def Image_setstate(self, args):
    type = self.array.dtype.type
    self.__class__ = _galsim.ImageView[type]
    self.__init__(*args)

# inject these as methods of Image classes
for Class in _galsim.Image.itervalues():
    Class.__setitem__ = Image_setitem
    Class.__getitem__ = Image_getitem
    Class.__add__ = Image_add
    Class.__radd__ = Image_add
    Class.__iadd__ = Image_iadd
    Class.__sub__ = Image_sub
    Class.__isub__ = Image_isub
    Class.__mul__ = Image_mul
    Class.__rmul__ = Image_mul
    Class.__imul__ = Image_imul
    Class.__div__ = Image_div
    Class.__truediv__ = Image_div
    Class.__idiv__ = Image_idiv
    Class.__itruediv__ = Image_idiv
    Class.copy = Image_copy
    Class.__getstate_manages_dict__ = 1
    Class.__getstate__ = Image_getstate
    Class.__setstate__ = Image_setstate

for Class in _galsim.ImageView.itervalues():
    Class.__setitem__ = Image_setitem
    Class.__getitem__ = Image_getitem
    Class.__add__ = Image_add
    Class.__radd__ = Image_add
    Class.__iadd__ = Image_iadd
    Class.__sub__ = Image_sub
    Class.__isub__ = Image_isub
    Class.__mul__ = Image_mul
    Class.__rmul__ = Image_mul
    Class.__imul__ = Image_imul
    Class.__div__ = Image_div
    Class.__truediv__ = Image_div
    Class.__idiv__ = Image_idiv
    Class.__itruediv__ = Image_idiv
    Class.copy = Image_copy
    Class.__getinitargs__ = ImageView_getinitargs

for Class in _galsim.ConstImageView.itervalues():
    Class.__getitem__ = Image_getitem
    Class.__add__ = Image_add
    Class.__radd__ = Image_add
    Class.__sub__ = Image_sub
    Class.__mul__ = Image_mul
    Class.__rmul__ = Image_mul
    Class.__div__ = Image_div
    Class.__truediv__ = Image_div
    Class.copy = Image_copy
    Class.__getinitargs__ = ImageView_getinitargs

del Class    # cleanup public namespace
