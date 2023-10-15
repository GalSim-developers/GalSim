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

# This file is logically part of utilities.py, but to avoid circular imports, we
# put a few things in here so files can import _utilties without triggering an ImportError.

import os
import functools
import weakref

from .errors import GalSimError, GalSimValueError

# Python 2/3 compatible definition of basestring without past.builtins
# (Based on this SO answer: https://stackoverflow.com/a/33699705/1332281)
basestring = ("".__class__, u"".__class__, b"".__class__)

class lazy_property:
    """
    This decorator will act similarly to @property, but will be efficient for multiple access
    to values that require some significant calculation.

    It works by replacing the attribute with the computed value, so after the first access,
    the property (an attribute of the class) is superseded by the new attribute of the instance.

    Note that is should only be used for non-mutable data, since the calculation will not be
    repeated if anything about the instance changes.

    Usage::

        @lazy_property
        def slow_function_to_be_used_as_a_property(self):
            x =  ...  # Some slow calculation.
            return x

    Base on an answer from http://stackoverflow.com/a/6849299
    """
    def __init__(self, fget):
        self.fget = fget
        self.func_name = fget.__name__

    def __get__(self, obj, cls):
        if obj is None:
            return self
        value = self.fget(obj)
        setattr(obj, self.func_name, value)
        return value

class doc_inherit:
    '''
    This decorator will grab a doc string from a base class version of a method.
    Useful if the subclass doesn't change anything about the method API, but just has
    a specialized implementation.  This lets the documentation live only in one place.

    Usage::

        class Base:
            def some_method(self):
                """A nice description of the functionality
                """
                pass

        class Sub(Base):

            @doc_inherit
            def some_method(self):
                # Don't bother with any doc string here.
                pass

    Based on the Docstring Inheritance Decorator at:

    https://github.com/ActiveState/code/wiki/Python_index_1

    Although I (MJ) modified it slightly, since the original recipe there had a bug that made it
    not work properly with 2 levels of sub-classing (e.g. Pixel <- Box <- GSObject).
    '''
    def __init__(self, mthd):
        self.mthd = mthd
        self.name = mthd.__name__

    def __get__(self, obj, cls):
        for parent in cls.__bases__: # pragma: no branch
            parfunc = getattr(parent, self.name, None)
            if parfunc and getattr(parfunc, '__doc__', None): # pragma: no branch
                break

        if obj:
            return self.get_with_inst(obj, cls, parfunc)
        else:
            return self.get_no_inst(cls, parfunc)

    def get_with_inst(self, obj, cls, parfunc):
        @functools.wraps(self.mthd, assigned=('__name__','__module__'))
        def f(*args, **kwargs):
            return self.mthd(obj, *args, **kwargs)
        return self.use_parent_doc(f, parfunc)

    def get_no_inst(self, cls, parfunc):
        @functools.wraps(self.mthd, assigned=('__name__','__module__'))
        def f(*args, **kwargs): # pragma: no cover (without inst, this is not normally called.)
            return self.mthd(*args, **kwargs)
        return self.use_parent_doc(f, parfunc)

    def use_parent_doc(self, func, source):
        if source is None: # pragma: no cover
            raise NameError("Can't find '%s' in parents"%self.name)
        func.__doc__ = source.__doc__
        return func


def isinteger(value):
    """Check if a value is an integer type (including np.int64, long, etc.)

    Specifically, it checks whether value == int(value).

    Parameter:
        value:      The value to be checked whether it is an integer

    Returns:
        True if the value is an integer type, False otherwise.
    """
    try:
        return value == int(value)
    except TypeError:
        return False

def ensure_dir(target):
    """
    Make sure the directory for the target location exists, watching for a race condition

    In particular check if the OS reported that the directory already exists when running
    makedirs, which can happen if another process creates it before this one can

    Parameter:
        target:     The file name for which to ensure that all necessary directories exist.
    """
    _ERR_FILE_EXISTS=17
    dir = os.path.dirname(target)
    if dir == '': return

    exists = os.path.exists(dir)
    if not exists:
        try:
            os.makedirs(dir)
        except OSError as err:  # pragma: no cover
            # check if the file now exists, which can happen if some other
            # process created the directory between the os.path.exists call
            # above and the time of the makedirs attempt.  This is OK
            if err.errno != _ERR_FILE_EXISTS:
                raise err

    elif exists and not os.path.isdir(dir):
        raise OSError("tried to make directory '%s' "
                      "but a non-directory file of that "
                      "name already exists" % dir)


class LRU_Cache:
    """Simplified Least Recently Used Cache.

    Mostly stolen from http://code.activestate.com/recipes/577970-simplified-lru-cache/,
    but added a method for dynamic resizing.  The least recently used cached item is
    overwritten on a cache miss.

    Parameters:
        user_function:  A python function to cache.
        maxsize:        Maximum number of inputs to cache.  [Default: 1024]

    Example::

        >>> def slow_function(*args) # A slow-to-evaluate python function
        >>>    ...
        >>>
        >>> v1 = slow_function(*k1)  # Calling function is slow
        >>> v1 = slow_function(*k1)  # Calling again with same args is still slow
        >>> cache = galsim.utilities.LRU_Cache(slow_function)
        >>> v1 = cache(*k1)  # Returns slow_function(*k1), slowly the first time
        >>> v1 = cache(*k1)  # Returns slow_function(*k1) again, but fast this time.
    """
    def __init__(self, user_function, maxsize=1024):
        # Link layout:     [PREV, NEXT, KEY, RESULT]
        self.root = [None, None, None, None]
        self.user_function = user_function
        self.cache = {}
        self.maxsize = maxsize
        self.clear()

    def clear(self):
        self.root = [None, None, None, None]
        root = self.root
        cache = self.cache
        cache.clear()
        maxsize = self.maxsize
        last = root
        for i in range(maxsize):
            key = object()
            cache[key] = last[1] = last = [last, root, key, None]
        root[0] = last

    def __call__(self, *key):
        cache = self.cache
        root = self.root
        link = cache.get(key)
        if link is not None:
            # Cache hit: move link to last position
            link_prev, link_next, _, result = link
            link_prev[1] = link_next
            link_next[0] = link_prev
            last = root[0]
            last[1] = root[0] = link
            link[0] = last
            link[1] = root
            return result
        # Cache miss: evaluate and insert new key/value at root, then increment root
        #             so that just-evaluated value is in last position.
        result = self.user_function(*key)
        root = self.root  # re-establish root in case user_function modified it due to recursion
        root[2] = key
        root[3] = result
        oldroot = root
        root = self.root = root[1]
        root[2], oldkey = None, root[2]
        root[3], oldvalue = None, root[3]
        del cache[oldkey]
        cache[key] = oldroot
        return result

    def resize(self, maxsize):
        """Resize the cache.

        Increasing the size of the cache is non-destructive, i.e., previously cached inputs remain
        in the cache.  Decreasing the size of the cache will necessarily remove items from the
        cache if the cache is already filled.  Items are removed in least recently used order.

        Parameters:
            maxsize:    The new maximum number of inputs to cache.
        """
        oldsize = self.maxsize
        if maxsize == oldsize:
            return
        else:
            root = self.root
            cache = self.cache
            if maxsize <= 0:
                raise GalSimValueError("Invalid maxsize", maxsize)
            if maxsize < oldsize:
                for i in range(oldsize - maxsize):
                    # Delete root.next
                    current_next_link = root[1]
                    new_next_link = root[1] = root[1][1]
                    new_next_link[0] = root
                    del cache[current_next_link[2]]
            else: #  maxsize > oldsize:
                for i in range(maxsize - oldsize):
                    # Insert between root and root.next
                    key = object()
                    cache[key] = link = [root, root[1], key, None]
                    root[1][0] = link
                    root[1] = link
        self.maxsize = maxsize


def math_eval(str, other_modules=()):
    """Evaluate a string that may include numpy, np, or math commands.

    Parameters:
        str:            The string to evaluate
        other_modules.  Other modules in addition to numpy, np, math to import as well.
                        Should be given as a list of strings.  [default: None]

    Returns:
        Whatever the string evaluates to.
    """
    gdict = globals().copy()
    exec('import galsim', gdict)
    exec('import numpy', gdict)
    exec('import numpy as np', gdict)

    exec('import math', gdict)
    exec('import coord', gdict)
    for m in other_modules:  # pragma: no cover  (We don't use this.)
        exec('import ' + m, gdict)

    # A few other things that show up in reprs, so useful to import here.
    exec('from numpy import array, uint16, uint32, int16, int32, float32, float64, complex64, complex128, ndarray',
         gdict)
    exec('from astropy.units import Unit', gdict)

    return eval(str, gdict)

class WeakMethod:
    """Wrap a method in a weakref.

    This is useful if you want to specialize a function if certain conditions hold.
    You can check those conditions and return one of several possible implementations as
    a `lazy_property`.

    Using just a normal ``weakref`` doesn't work, but this class will work.

    From http://code.activestate.com/recipes/81253-weakmethod/
    """
    def __init__(self, f):
        self.f = f.__func__
        self.c = weakref.ref(f.__self__)
    def __call__(self, *args):
        try:
            # If the reference is dead, self.c() will be None, so this will raise an
            # AttributeError: 'NoneType' object has no attribute ...
            # Hopefully the method itself won't raise an AttributeError for anything else.
            return self.f(self.c(), *args)
        except AttributeError:  # pragma: no cover
            raise GalSimError('Method called on dead object')
