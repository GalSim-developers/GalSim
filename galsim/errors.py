# Copyright (c) 2012-2021 by the GalSim developers team on GitHub
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

# Define the class hierarchy for errors and warnings emitted by GalSim that aren't
# obviously one of the standard python errors.

import warnings
from contextlib import contextmanager

# Note to developers about which exception to throw.
#
# Aside from the below classes, which should be preferred for most errors, we also
# throw the following in some cases.
#
# TypeError:            Use this for errors that in a more strongly typed language would probably
#                       be a compiler error. For instance, it is used for the following errors:
#                       - a parameter with the wrong type
#                       - the wrong number of unnamed args when processing `*args` by hand.
#                       - missing or invalid kwargs when processing `**kwargs` by hand.
#
# OSError:              Use this for errors related to I/O, disk access, etc.
#                       Note: In Python 2, there was a distinction between IOError and OSError, but
#                       there was never much difference in reality, and in Python 3, they made both
#                       OSError. We should just use OSError for all such kinds of errors.
#
# NotImplementedError:  Use this for code that is not implemented by design and which will never
#                       be implemented. E.g. GSObject and Position use this for their __init__
#                       implementations, since it is invalid to instantiate the base class.
#                       Use GalSimNotImplementedError for features that might someday be
#                       implemented.
#
# AttributeError:       Use this only for an attempt to access an attribute that an object does not
#                       have. Like TypeError, this should be reserved for things that a more
#                       strongly typed language would catch at compile time. We don't currently
#                       raise this anywhere in GalSim.
#
# RuntimeError:         Don't use this. Use GalSimError (or a subclass) for any run-time errors.
#
# ValueError:           Don't use this. Use one of the below exceptions that derive from ValueError.
#
# KeyError:             Don't use this. Use GalSimKeyError instead
#
# IndexError:           Don't use this. Use GalSimIndexError instead.
#
# std::runtime_error:   Use this for errors in the C++ layer, and use the convert_cpp_errors()
#                       context to convert these errors into GalSimErrors. E.g. GSFitsWCS._invert_pv
#                       uses this for non-convergence, which gets converted into GalSimError in
#                       the Python layer.
#                       When possible, it is preferable to guard against any such events by making
#                       appropriate checks in the Python layer before dropping down into C++.
#                       E.g. Image checks for anything that might cause the C++ Image class to
#                       throw an exception and raises some kind of GalSim exception first.
#                       Nonetheless, it is good practice to use the `with convert_cpp_errors()`
#                       context for all calls to the C++ layer, just in case.
#
# GalSim-specific error classes:
# ------------------------------
#
# GalSimError:          Use this for what would normally be a RuntimeError. Usually some exceptional
#                       occurrence in otherwise correct code. E.g. an algorithm not converging or
#                       a singular matrix encountered. It can also be used when the program does
#                       things out of order; e.g. PowerSpectrum raises this when getShear and the
#                       like are called before `buildGrid`. This is also the catch-all exception
#                       to use when none of the other GalSim exceptions are appropriate.
#
# GalSimValueError:     Use this when a user provides an invalid value for a parameter.
#                       Note: it has an optional argument to give a list of allowed values when
#                       that is appropriate.
#
# GalSimKeyError        Use this for accessing a dict-like object with an invalid key. E.g.
#                       FitsHeader and Catalog raise this for accessing invalid columns.
#
# GalSimIndexError      Use this for the equivalent of accessing a list-like object with an
#                       invalid index. E.g. RealGalaxyCatalog and Catalog raise this for accessing
#                       invalid rows.
#
# GalSimRangeError:     Use this when a user provides an value outside of some allowed range.
#                       You should also give the min/max values of the allowed range. The max
#                       is optional, because it's not uncommon for there to be no upper limit.
#                       If only the upper limit is relevant and not the lower limit, you may
#                       use min=None to indicate this.
#
# GalSimBoundsError:    Use this when a position is outside its allowed bounds. It's basically
#                       the same as GalSimRangeError, but in two dimensions.
#
# GalSimUndefinedBoundsError:   Use this when the user tries to perform an operation on an
#                               Image with undefined bounds (and which requires the bounds to be
#                               defined).
#
# GalSimImmutableError: Use this when the user tries to modify an immutable Image in some way.
#
# GalSimIncompatibleValuesError:    Use this when two or more parameters are invalid when used
#                                   in combination. E.g. providing more than one size parameter
#                                   to Moffat, Sersic, Gaussian, etc. The conflicting values
#                                   should be given as extra keywords to the constructor, which
#                                   are mentioned in the error message.
#                                   Note: if one of the conflicting values is self (e.g. adding two
#                                   SEDs with different redshifts), then don't name the kwarg self.
#                                   Instead use something like `self_sed=self`.
#
# GalSimSEDError:       Use this when an SED is required to be either spectral or dimensionless,
#                       and the other kind of SED is provided.
#
# GalSimHSMError:       Use this for errors from the HSM algorithm.  They are emitted in C++, but
#                       we use `with convert_cpp_errors(GalSimHSMError):` to convert them.
#
# GalSimFFTSizeError:   Use this when a requested FFT would exceed the relevant maximum_fft_size
#                       for the object, so the recommendation is raise this parameter if that
#                       is possible.
#
# GalSimConfigError:    Use this for errors processing a config dict.
#
# GalSimConfigValueError:   Use this when a config dict has a value that is invalid. Basically,
#                           whenever you would normally use GalSimValueError when processing
#                           a config dict, you should use this instead.
#
# GalSimNotImplementedError:  Use this for features that we have not yet implemented, but which may
#                             be implemented someday. So it's not a necessarily invalid usage, just
#                             something that doesn't work currently.

class GalSimError(RuntimeError):
    """The base class for GalSim-specific run-time errors.
    """
    # Minimal version of these to make GalSimError reprable and picklable.
    def __repr__(self): return 'galsim.GalSimError(%r)'%(str(self))
    def __eq__(self, other): return self is other or repr(self) == repr(other)
    def __hash__(self): return hash(repr(self))


class GalSimValueError(GalSimError, ValueError):
    """A GalSim-specific exception class indicating that some user-input value is invalid.

    Attributes:
        value:          The invalid value
        allowed_values: A list of allowed values if appropriate (may be None)
    """
    def __init__(self, message, value, allowed_values=None):
        self.message = message
        self.value = value
        self.allowed_values = allowed_values

        message += " Value {0!s}".format(value)
        if allowed_values:
            message += " not in {0!s}".format(allowed_values)
        super(GalSimValueError, self).__init__(message)

    def __repr__(self):
        return 'galsim.GalSimValueError(%r,%r,%r)'%(self.message, self.value, self.allowed_values)
    def __reduce__(self):  # Need to override this whenever constructor take extra params
        return GalSimValueError, (self.message, self.value, self.allowed_values)


class GalSimKeyError(GalSimError, KeyError):
    """A GalSim-specific exception class indicating an attempt to access a dict-like object
    with an invalid key.

    Attributes:
        key:        The invalid key
    """
    def __init__(self, message, key):
        self.message = message
        self.key = key
        super(GalSimKeyError, self).__init__(message, key)  # Need to pass key or pickle fails.

    def __str__(self):
        return self.message + " Key {0!s}".format(self.key)

    def __repr__(self):
        return 'galsim.GalSimKeyError(%r,%r)'%(self.message, self.key)


class GalSimIndexError(GalSimError, IndexError):
    """A GalSim-specific exception class indicating an attempt to access a list-like object
    with an invalid index.

    Attributes:
        index:      The invalid index
    """
    def __init__(self, message, index):
        self.message = message
        self.index = index
        super(GalSimIndexError, self).__init__(message, index)

    def __str__(self):
        return self.message + " Index {0!s}".format(self.index)

    def __repr__(self):
        return 'galsim.GalSimIndexError(%r,%r)'%(self.message, self.index)


class GalSimRangeError(GalSimError, ValueError):
    """A GalSim-specific exception class indicating that some user-input value is
    outside of the allowed range of values.

    Attributes:
        value:      The invalid value
        min:        The minimum allowed value (may be None)
        max:        The maximum allowed value (may be None)
    """
    def __init__(self, message, value, min, max=None):
        self.message = message
        self.value = value
        self.min = min
        self.max = max

        message += " Value {0!s} not in range [{1!s}, {2!s}]".format(value, min, max)
        super(GalSimRangeError, self).__init__(message)

    def __repr__(self):
        return 'galsim.GalSimRangeError(%r,%r,%r,%r)'%(self.message, self.value, self.min, self.max)
    def __reduce__(self):
        return GalSimRangeError, (self.message, self.value, self.min, self.max)


class GalSimBoundsError(GalSimError, ValueError):
    """A GalSim-specific exception class indicating that some user-input position is
    outside of the allowed bounds.

    Attributes:
        pos:        The invalid position
        bounds:     The bounds in which it was expected to fall
    """
    def __init__(self, message, pos, bounds):
        self.message = message
        self.pos = pos
        self.bounds = bounds

        message += " {0!s} not in {1!s}".format(pos, bounds)
        super(GalSimBoundsError, self).__init__(message)

    def __repr__(self):
        return 'galsim.GalSimBoundsError(%r,%r,%r)'%(self.message, self.pos, self.bounds)
    def __reduce__(self):
        return GalSimBoundsError, (self.message, self.pos, self.bounds)


class GalSimUndefinedBoundsError(GalSimError):
    """A GalSim-specific exception class indicating an attempt to access the extent of
    a `Bounds` instance that has not yet been defined.
    """
    def __repr__(self):
        return 'galsim.GalSimUndefinedBoundsError(%r)'%(str(self))


class GalSimImmutableError(GalSimError):
    """A GalSim-specific exception class indicating an attempt to modify an immutable image.

    Attributes:
        image:      The image that the user attempted to modify
    """
    def __init__(self, message, image):
        self.message = message
        self.image = image

        message += " Image: {0!s}".format(image)
        super(GalSimImmutableError, self).__init__(message)

    def __repr__(self):
        return 'galsim.GalSimImmutableError(%r,%r)'%(self.message, self.image)
    def __reduce__(self):
        return GalSimImmutableError, (self.message, self.image)


class GalSimIncompatibleValuesError(GalSimError, ValueError, TypeError):
    """A GalSim-specific exception class indicating that 2 or more user-input values are
    incompatible as given.

    Attributes:
        values:     A dict of {name : value} giving the values that in combination are invalid.
    """
    def __init__(self, message, values={}, **kwargs):
        self.message = message
        self.values = dict(values, **kwargs)

        message += " Values {0!s}".format(self.values)
        super(GalSimIncompatibleValuesError, self).__init__(message)

    # Note: the repr of values can rearrange the items, but the dicts should compare equal.
    def __eq__(self, other):
        return (self is other or
                (isinstance(other, GalSimIncompatibleValuesError) and
                 self.message == other.message and
                 self.values == other.values))
    def __repr__(self):
        return 'galsim.GalSimIncompatibleValuesError(%r,%r)'%(self.message, self.values)
    def __reduce__(self):
        return GalSimIncompatibleValuesError, (self.message, self.values)


class GalSimSEDError(GalSimError, TypeError):
    """A GalSim-specific exception class indicating an attempt to do something invalid for the
    kind of `SED` that is present.  Typically involving a dimensionless `SED` where a spectral
    `SED` is required (or vice versa).

    Attributes:
        sed:        The invalid `SED`
    """
    def __init__(self, message, sed):
        self.message = message
        self.sed = sed

        message += " SED: {0!s}".format(sed)
        super(GalSimSEDError, self).__init__(message)

    def __repr__(self):
        return 'galsim.GalSimSEDError(%r,%r)'%(self.message, self.sed)
    def __reduce__(self):
        return GalSimSEDError, (self.message, self.sed)


class GalSimHSMError(GalSimError):
    """A GalSim-specific exception class indicating some kind of failure of the HSM algorithms
    """
    def __repr__(self):
        return 'galsim.GalSimHSMError(%r)'%(str(self))


class GalSimFFTSizeError(GalSimError):
    """A GalSim-specific exception class indicating that a requested FFT exceeds the relevant
    maximum_fft_size.

    Attributes:
        size:       The size that was deemed too large
        mem:        The estimated memory that would be required (in GB) for the FFT.
    """
    def __init__(self, message, size):
        self.message = message
        self.size = size
        self.mem = size * size * 24. / 1024**3
        message += "\nThe required FFT size would be {0} x {0}, which requires ".format(size)
        message += "{0:.2f} GB of memory.\n".format(self.mem)
        message += "If you can handle the large FFT, you may update gsparams.maximum_fft_size."
        super(GalSimFFTSizeError, self).__init__(message)

    def __repr__(self):
        return 'galsim.GalSimFFTSizeError(%r,%r)'%(self.message, self.size)
    def __reduce__(self):
        return GalSimFFTSizeError, (self.message, self.size)


class GalSimConfigError(GalSimError, ValueError):
    """A GalSim-specific exception class indicating some kind of failure processing a
    configuration file.
    """
    def __repr__(self):
        return 'galsim.GalSimConfigError(%r)'%(str(self))


class GalSimConfigValueError(GalSimValueError, GalSimConfigError):
    """A GalSim-specific exception class indicating that a config entry has an invalid value.

    Attributes:
        value:          The invalid value
        allowed_values: A list of allowed values if appropriate (may be None)
    """
    def __repr__(self):
        return 'galsim.GalSimConfigValueError(%r,%r,%r)'%(
            self.message, self.value, self.allowed_values)
    def __reduce__(self):
        return GalSimConfigValueError, (self.message, self.value, self.allowed_values)


class GalSimNotImplementedError(GalSimError, NotImplementedError):
    """A GalSim-specific exception class indicating that the feature being attempted is not
    currently implemented.

    If this is a feature you feel you need, please open an issue about it at

        https://github.com/GalSim-developers/GalSim/issues

    Even better, feel free to offer to contribute code to implement the feature.
    """
    def __repr__(self):
        return 'galsim.GalSimNotImplementedError(%r)'%(str(self))


# Note: Can use galsim_warn to raise warnings with this warning class.
class GalSimWarning(UserWarning):
    """The base class for GalSim-emitted warnings.
    """
    def __repr__(self): return 'galsim.GalSimWarning(%r)'%(str(self))
    def __eq__(self, other): return self is other or repr(self) == repr(other)
    def __hash__(self): return hash(repr(self))


# Note: By default python ignores DeprecationWarnings.  Apparently they are really
#       for python system deprecations.  GalSim deprecations are thus only subclassed from
#       GalSimWarning, not DeprecationWarning.
class GalSimDeprecationWarning(GalSimWarning):
    """A GalSim-specific warning class used for deprecation warnings.
    """
    def __repr__(self): return 'galsim.GalSimDeprecationWarning(%r)'%(str(self))

@contextmanager
def convert_cpp_errors(error_type=GalSimError):
    try:
        yield
    except RuntimeError as err:
        raise error_type(str(err))

def galsim_warn(message):
    """A helper function for emitting a GalSimWarning with the given message
    """
    warnings.warn(message, GalSimWarning)
