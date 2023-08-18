# Copyright (c) 2012-2022 by the GalSim developers team on GitHub
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
import logging
import copy
from contextlib import contextmanager
import warnings

from .random import BaseDeviate
from .utilities import printoptions


# This file has some utilities that we use in the tests, but which might be useful
# for other code bases who want to use them as well.

def do_pickle(obj1, func = lambda x : x, irreprable=False):
    """Check that the object is picklable.  Also that it has basic == and != functionality.
    """
    from numbers import Integral, Real, Complex
    import pickle
    import copy
    # In case the repr uses these:
    import galsim
    import coord
    from numpy import array, uint16, uint32, int16, int32, float32, float64, complex64, complex128, ndarray
    from astropy.units import Unit
    import astropy.io.fits
    from distutils.version import LooseVersion
    from collections.abc import Hashable

    print('Try pickling ',str(obj1))

    obj2 = pickle.loads(pickle.dumps(obj1))
    assert obj2 is not obj1
    f1 = func(obj1)
    f2 = func(obj2)
    if not (f1 == f2):  # pragma: no cover
        print('obj1 = ',repr(obj1))
        print('obj2 = ',repr(obj2))
        print('func(obj1) = ',repr(f1))
        print('func(obj2) = ',repr(f2))
    assert f1 == f2

    # Check that == works properly if the other thing isn't the same type.
    assert f1 != object()
    assert object() != f1

    # Test the hash values are equal for two equivalent objects.
    if isinstance(obj1, Hashable):
        if not(hash(obj1) == hash(obj2)): # pragma: no cover
            print('hash = ',hash(obj1),hash(obj2))
        assert hash(obj1) == hash(obj2)

    obj3 = copy.copy(obj1)
    assert obj3 is not obj1
    random = hasattr(obj1, 'rng') or isinstance(obj1, BaseDeviate) or 'rng' in repr(obj1)
    if not random:  # Things with an rng attribute won't be identical on copy.
        f3 = func(obj3)
        assert f3 == f1
    elif isinstance(obj1, BaseDeviate):
        f1 = func(obj1)  # But BaseDeviates will be ok.  Just need to remake f1.
        f3 = func(obj3)
        assert f3 == f1

    obj4 = copy.deepcopy(obj1)
    assert obj4 is not obj1
    f4 = func(obj4)
    if random: f1 = func(obj1)
    if not (f4 == f1):  # pragma: no cover
        print('func(obj1) = ',repr(f1))
        print('func(obj4) = ',repr(f4))
    assert f4 == f1  # But everything should be identical with deepcopy.

    # Also test that the repr is an accurate representation of the object.
    # The gold standard is that eval(repr(obj)) == obj.  So check that here as well.
    # A few objects we don't expect to work this way in GalSim; when testing these, we set the
    # `irreprable` kwarg to true.  Also, we skip anything with random deviates since these don't
    # respect the eval/repr roundtrip.

    if not random and not irreprable:
        # A further complication is that the default numpy print options do not lead to sufficient
        # precision for the eval string to exactly reproduce the original object, and start
        # truncating the output for relatively small size arrays.  So we temporarily bump up the
        # precision and truncation threshold for testing.
        with printoptions(precision=20, threshold=np.inf):
            obj5 = eval(repr(obj1))
        f5 = func(obj5)
        if not (f5 == f1):  # pragma: no cover
            print('obj1 = ',repr(obj1))
            print('obj5 = ',repr(obj5))
            print('f1 = ',f1)
            print('f5 = ',f5)
        assert f5 == f1, "func(obj1) = %r\nfunc(obj5) = %r"%(f1, f5)
    else:
        # Even if we're not actually doing the test, still make the repr to check for syntax errors.
        repr(obj1)

    # Historical note:
    # We used to have a test here where we perturbed the construction arguments to make sure
    # that objects that should be different really are different.
    # However, that used `__getinitargs__`, which we've moved away from using for pickle, so
    # none of our classes get checked this way anymore.  So we removed this section.
    # This means that this inequality test has to be done manually via all_obj_diff.
    # See releases v2.3 or earlier for the old way we did this.


def all_obj_diff(objs, check_hash=True):
    """ Helper function that verifies that each element in `objs` is unique and, if hashable,
    produces a unique hash."""

    from collections.abc import Hashable
    from collections import Counter
    # Check that all objects are unique.
    # Would like to use `assert len(objs) == len(set(objs))` here, but this requires that the
    # elements of objs are hashable (and that they have unique hashes!, which is what we're trying
    # to test!.  So instead, we just loop over all combinations.
    for i, obji in enumerate(objs):
        assert obji == obji
        assert not (obji != obji)
        # Could probably start the next loop at `i+1`, but we start at 0 for completeness
        # (and to verify a != b implies b != a)
        for j, objj in enumerate(objs):
            if i == j:
                continue
            assert obji != objj, ("Found equivalent objects {0} == {1} at indices {2} and {3}"
                                  .format(obji, objj, i, j))

    if not check_hash:
        return
    # Now check that all hashes are unique (if the items are hashable).
    if not isinstance(objs[0], Hashable):
        return
    hashes = [hash(obj) for obj in objs]
    if not (len(hashes) == len(set(hashes))):  # pramga: no cover
        for k, v in Counter(hashes).items():
            if v <= 1:
                continue
            print("Found multiple equivalent object hashes:")
            for i, obj in enumerate(objs):
                if hash(obj) == k:
                    print(i, repr(obj))
    assert len(hashes) == len(set(hashes))


def timer(f):
    import functools

    @functools.wraps(f)
    def f2(*args, **kwargs):
        import time
        t0 = time.time()
        result = f(*args, **kwargs)
        t1 = time.time()
        fname = repr(f).split()[1]
        print('time for %s = %.2f' % (fname, t1-t0))
        return result
    return f2


class CaptureLog:
    """A context manager that saves logging output into a string that is accessible for
    checking in unit tests.

    After exiting the context, the attribute `output` will have the logging output.

    Sample usage:

            >>> with CaptureLog() as cl:
            ...     cl.logger.info('Do some stuff')
            >>> assert cl.output == 'Do some stuff'

    """
    def __init__(self, level=3):
        from io import StringIO

        logging_levels = { 0: logging.CRITICAL,
                           1: logging.WARNING,
                           2: logging.INFO,
                           3: logging.DEBUG }
        self.logger = logging.getLogger('CaptureLog')
        self.logger.setLevel(logging_levels[level])
        self.stream = StringIO()
        self.handler = logging.StreamHandler(self.stream)
        self.logger.addHandler(self.handler)

    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        self.handler.flush()
        self.output = self.stream.getvalue().strip()
        self.handler.close()


# We used to roll our own versions of these, but numpy.testing has good ones now.
from numpy.testing import assert_raises
from numpy.testing import assert_warns

# Context to make it easier to profile bits of the code
class profile:
    def __init__(self, sortby='tottime', nlines=30):
        self.sortby = sortby
        self.nlines = nlines

    def __enter__(self):
        import cProfile, pstats
        self.pr = cProfile.Profile()
        self.pr.enable()
        return self

    def __exit__(self, type, value, traceback):
        import pstats
        self.pr.disable()
        ps = pstats.Stats(self.pr).sort_stats(self.sortby)
        ps.print_stats(self.nlines)
