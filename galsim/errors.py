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

# Define the class hierarchy for errors and warnings emitted by GalSim that aren't
# obviously one of the standard python errors.

from builtins import super

# Note to developers about which exception to throw.
#
# Aside from the below classes, which should be preferred for most errors, we also
# throw the following in some cases.
#
# TypeError:    Use this for errors that in a more strongly typed language would probably
#               be a compiler error.  For instance, it is used for the following errors:
#                - a parameter has the wrong type
#                - the wrong number of unnamed args when processing `*args` by hand.
#                - missing or invalid kwargs when processing `**kwargs` by hand.
#
# OSError:      Use this for errors related to I/O, disk access, etc.  Note: In Python 2,
#               there was a distinction between IOError and OSError, but there was never much
#               difference in reality, and in Python 3, they made everything OSError.
#               We should just use OSError for all such kinds of errors.
#
# KeyError:     Use this for the equivalent of accessing a dict-like object with an invalid key.
#               E.g. FitsHeader and Catalog raise this for accessing invalid columns.
#
# IndexError:   Use this for the equivalent of accessing a list-like object with an invalid index.
#               E.g. RealGalaxyCatalog and Catalog raise this for accessing invalid rows.
#
# NotImplementedError:  Use this for features that we have not implemented.  Even if there is
#                       no future intent to do so.  E.g. GSObject defines uses this for a number
#                       of methods that are invalid for non-x-analytic profiles where the
#                       functionality is not implemented (and never will be) in some of the
#                       derived classes.
#                       Also, use it for calls that are invalid in a base class perhaps, but are
#                       valid for derived classes.  E.g. GSObject and Position use this for their
#                       __init__ implementations.
#
# AttributeError:   Use this only for an attempt to access an attribute that an object does not
#                   have.  We don't currently raise this anywhere in GalSim.
#
# RuntimeError: Don't use this.  Use GalSimError (or a subclass) for any run-time errors.
#
# ValueError:   Don't use this.  Use one of the below exceptions that derive from ValueError.
#
# std::runtime_error:   Use this for errors in the C++ layer, and put a try/except guard around
#                       the C++ call in the Python layer to convert to a GalSimError.  E.g.
#                       GSFitsWCS._invert_pv uses this for non-convergence, but we convert to
#                       a GalSimError in Python.
#                       When possible, try to guard against any such events by making appropriate
#                       checks in the Python layer before dropping down into C++.  E.g. Image
#                       checks for anything that might cause the C++ Image class to throw an
#                       exception and raises some kind of GalSim exception first.


class GalSimError(RuntimeError):
    """The base class for GalSim-specific run-time errors.
    """
    # Minimal version of these to make GalSimError reprable and picklable.
    def __repr__(self): return 'galsim.GalSimError(%r)'%(str(self))
    def __eq__(self, other): return repr(self) == repr(other)
    def __hash__(self): return hash(repr(self))


class GalSimValueError(GalSimError, ValueError):
    """A GalSim-specific exception class indicating that some user-input value is invalid.

    Attrubutes:

        value = the invalid value
        allowed_values = a list of allowed values if appropriate (may be None)
    """
    def __init__(self, message, value, allowed_values=None):
        self.message = message
        self.value = value
        self.allowed_values = allowed_values

        message += " Value {0!s}".format(value)
        if allowed_values:
            message += " not in {0!s}".format(allowed_values)
        super().__init__(message)

    def __repr__(self):
        return 'galsim.GalSimValueError(%r,%r,%r)'%(self.message, self.value, self.allowed_values)
    def __reduce__(self):  # Need to override this whenever constructor take extra params
        return GalSimValueError, (self.message, self.value, self.allowed_values)


class GalSimKeyError(GalSimError, KeyError):
    """A GalSim-specific exception class indicating an attempt to access a dict-like object
    with an invalid key.
    """
    def __repr__(self):
        return 'galsim.GalSimKeyError(%r)'%(str(self))


class GalSimIndexError(GalSimError, IndexError):
    """A GalSim-specific execption class indicating an attempt to access a list-like object
    with an invalid index.
    """
    def __repr__(self):
        return 'galsim.GalSimIndexError(%r)'%(str(self))


class GalSimRangeError(GalSimError, ValueError):
    """A GalSim-specific exception class indicating that some user-input value is
    outside of the allowed range of values.

    Attrubutes:

        value = the invalid value
        min = the minimum allowed value (may be None)
        max = the maximum allowed value (may be None)
    """
    def __init__(self, message, value, min, max=None):
        self.message = message
        self.value = value
        self.min = min
        self.max = max

        message += " Value {0!s} not in range [{1!s}, {2!s}]".format(value, min, max)
        super().__init__(message)

    def __repr__(self):
        return 'galsim.GalSimRangeError(%r,%r,%r,%r)'%(self.message, self.value, self.min, self.max)
    def __reduce__(self):
        return GalSimRangeError, (self.message, self.value, self.min, self.max)


class GalSimBoundsError(GalSimError, ValueError):
    """A GalSim-specific exception class indicating that some user-input position is
    outside of the allowed bounds.

    Attrubutes:

        pos = the invalid position
        bounds = the bounds in which it was expected to fall
    """
    def __init__(self, message, pos, bounds):
        self.message = message
        self.pos = pos
        self.bounds = bounds

        message += " {0!s} not in {1!s}".format(pos, bounds)
        super().__init__(message)

    def __repr__(self):
        return 'galsim.GalSimBoundsError(%r,%r,%r)'%(self.message, self.pos, self.bounds)
    def __reduce__(self):
        return GalSimBoundsError, (self.message, self.pos, self.bounds)


class GalSimUndefinedBoundsError(GalSimError):
    """A GalSim-specific exception class indicating an attempt to access the range of bounds
    that have not yet been defined.
    """
    def __repr__(self):
        return 'galsim.GalSimUndefinedBoundsError(%r)'%(str(self))


class GalSimImmutableError(GalSimError):
    """A GalSim-specific exception class indicating an attempt to modify an immutable image.

    Attrubutes:

        image = the image that the user attempted to modify
    """
    def __init__(self, message, image):
        self.message = message
        self.image = image

        message += " Image: {0!s}".format(image)
        super().__init__(message)

    def __repr__(self):
        return 'galsim.GalSimImmutableError(%r,%r)'%(self.message, self.image)
    def __reduce__(self):
        return GalSimImmutableError, (self.message, self.image)


class GalSimIncompatibleValuesError(GalSimError, ValueError, TypeError):
    """A GalSim-specific exception class indicating that 2 or more user-input values are
    incompatible as given.

    Attrubutes:

        values = a dict of {name : value} giving the values that in combination are invalid.
    """
    def __init__(self, message, values={}, **kwargs):
        self.message = message
        self.values = dict(values, **kwargs)

        message += " Values {0!s}".format(self.values)
        super().__init__(message)

    def __repr__(self):
        return 'galsim.GalSimIncompatibleValuesError(%r,%r)'%(self.message, self.values)
    def __reduce__(self):
        return GalSimIncompatibleValuesError, (self.message, self.values)


class GalSimSEDError(GalSimError, TypeError):
    """A GalSim-specific exception class indicating an attempt to do something invalid for the
    kind of SED that is present.  Typically involving a dimensionless SED where a spectral SED
    is required (or vice versa).

    Attrubutes:

        sed = the invalid SED
    """
    def __init__(self, message, sed):
        self.message = message
        self.sed = sed

        message += " SED: {0!s}".format(sed)
        super().__init__(message)

    def __repr__(self):
        return 'galsim.GalSimSEDError(%r,%r)'%(self.message, self.sed)
    def __reduce__(self):
        return GalSimSEDError, (self.message, self.sed)


class GalSimHSMError(GalSimError):
    """A GalSim-specific exception class indicating some kind of failure of the HSM algorithms
    """
    def __repr__(self):
        return 'galsim.GalSimHSMError(%r)'%(str(self))


class GalSimConfigError(GalSimError, ValueError):
    """A GalSim-specific exception class indicating some kind of failure processing a
    configuration file.
    """
    def __repr__(self):
        return 'galsim.GalSimConfigError(%r)'%(str(self))


class GalSimConfigValueError(GalSimValueError, GalSimConfigError):
    """A GalSim-specific exception class indicating that a config entry has an invalid value.

    Attrubutes:

        value = the invalid value
        allowed_values = a list of allowed values if appropriate (may be None)
    """
    def __repr__(self):
        return 'galsim.GalSimConfigValueError(%r,%r,%r)'%(
            self.message, self.value, self.allowed_values)
    def __reduce__(self):
        return GalSimConfigValueError, (self.message, self.value, self.allowed_values)


class GalSimWarning(UserWarning):
    """The base class for GalSim-emitted warnings.
    """
    def __repr__(self): return 'galsim.GalSimWarning(%r)'%(str(self))
    def __eq__(self, other): return repr(self) == repr(other)
    def __hash__(self): return hash(repr(self))


# Note: By default python ignores DeprecationWarnings.  Apparently they are really
#       for python system deprecations.  GalSim deprecations are thus only subclassed from
#       GalSimWarning, not DeprecationWarning.
class GalSimDeprecationWarning(GalSimWarning):
    """A GalSim-specific warning class used for deprecation warnings.
    """
    def __repr__(self): return 'galsim.GalSimDeprecationWarning(%r)'%(str(self))
