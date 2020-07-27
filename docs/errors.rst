
Errors and Warnings
###################

GalSim uses some custom Exception and Warning classes when it finds some exceptional
occurrence:

`GalSimError`
            This is the base class for all other GalSim exceptions.  So you can catch this
            exception if you want to catch just the exceptions raised by GalSim.  It is
            roughly analogous to a ``RuntimeError``.

`GalSimValueError`
            This indicates that you provided an invalid value for an argument to a function.
            It includes attributes that tell you about what value you provided and sometimes
            about the allowed values.

`GalSimKeyError`
            This indicates that you tried to access some dict-like object (e.g. `FitsHeader`
            or `Catalog`) with an invalid key.

`GalSimIndexError`
            This indicates that you tried to access some list-like object (e.g. `RealGalaxyCatalog`)
            with an invalid index.

`GalSimRangeError`
            This indicates that you provided a value that is outside of the allowed range.  It
            includes attributes indicating what value you provided and what the allowed range is.

`GalSimBoundsError`
            This indicates that you used a `Position` outside of the allowed `Bounds`.  It is
            basically a `GalSimRangeError`, but in two dimensions.  It includes attributes that
            tell you the `Position` and the allowed `Bounds`.

`GalSimUndefinedBoundsError`
            This indicates that you are trying to use an undefined `Bounds` instance in a context
            where it must be defined.

`GalSimImmutableError`
            This indicates that you tried to change an immutable `Image` in some way.

`GalSimIncompatibleValuesError`
            This indicates that two or more values that you provided to some function are not
            compatible with each other.  It includes attributes telling you the two values that
            are incompatible.

`GalSimSEDError`
            This indicates that you tried to use an SED in a context where it is required to be
            either spectral or dimensionless, and you provided the other kind.

`GalSimHSMError`
            This indicates that the HSM algorithm raised some kind of exception.

`GalSimFFTSizeError`
            This indicates that something you did requires a very large FFT, in particular one
            that is larger than the relevant ``gsparams.maximum_fft_size`` parameter.  It includes
            attributes that tell you both the size that was required and how much memory it would
            have used, so you can decide whether you want to adjust some parameters of your
            simulation or to adjust the object's `GSParams` options.

`GalSimConfigError`
            This indicates that there was some kind of failure processing a configuration file.

`GalSimConfigValueError`
            This indicates that some parameter in your configuration file is an invalid value.

`GalSimNotImplementedError`
            This indicates that you tried to do something that is not implemented currently.

`GalSimWarning`
            This indicates that you did something that is not necessarily an error, but we think
            it is likely that you didn't do something right.

`GalSimDeprecationWarning`
            This indicates that you are using functionality that is currently deprecated.
            Your code will generally continue to work until the next major upgrade, but you are
            encouraged to update your code to the new syntax.

.. autoclass:: galsim.GalSimError

.. autoclass:: galsim.GalSimValueError
    :show-inheritance:

.. autoclass:: galsim.GalSimKeyError
    :show-inheritance:

.. autoclass:: galsim.GalSimIndexError
    :show-inheritance:

.. autoclass:: galsim.GalSimRangeError
    :show-inheritance:

.. autoclass:: galsim.GalSimBoundsError
    :show-inheritance:

.. autoclass:: galsim.GalSimUndefinedBoundsError
    :show-inheritance:

.. autoclass:: galsim.GalSimImmutableError
    :show-inheritance:

.. autoclass:: galsim.GalSimIncompatibleValuesError
    :show-inheritance:

.. autoclass:: galsim.GalSimSEDError
    :show-inheritance:

.. autoclass:: galsim.GalSimHSMError
    :show-inheritance:

.. autoclass:: galsim.GalSimFFTSizeError
    :show-inheritance:

.. autoclass:: galsim.GalSimConfigError
    :show-inheritance:

.. autoclass:: galsim.GalSimConfigValueError
    :show-inheritance:

.. autoclass:: galsim.GalSimNotImplementedError
    :show-inheritance:

.. autoclass:: galsim.GalSimWarning
    :show-inheritance:

.. autoclass:: galsim.GalSimDeprecationWarning
    :show-inheritance:
