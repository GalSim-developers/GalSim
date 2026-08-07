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
"""Single-probe shim around the astropy modules GalSim imports at top level.

astropy.units cannot initialise inside frozen binaries (Nuitka onefile,
PyInstaller-style bundles): its PLY parser builds grammar tables through
``sys._getframe()`` locals introspection that compiled frames do not provide
(see astropy/astropy#15069). GalSim only needs astropy.units for
spectral/chromatic features; the geometry, zernike and photon-shooting paths
never touch it.

This module probes astropy ONCE per process. In any normal source install the
real modules are re-exported unchanged, so behaviour is identical to importing
astropy directly. When astropy.units cannot initialise, inert fallbacks are
exported instead: imports succeed, config parameter schemas keep working
(upper-case attribute names resolve to a placeholder type that is safe in
``isinstance`` checks), and spectral features fail only if actually exercised.
"""

__all__ = ["units", "constants", "wcs", "Quantity", "Unit",
           "HAVE_ASTROPY_UNITS"]


class _FallbackType:
    """Placeholder for unit/quantity classes in isinstance() checks."""

    def __init__(self, *args, **kwargs):
        pass


class _FallbackValue:
    """Inert placeholder surviving attribute access, calls and arithmetic."""
    __slots__ = ()

    def __call__(self, *args, **kwargs):
        return self

    def __getattr__(self, name):
        # Keep introspection honest: probing __file__/__wrapped__/etc. must
        # raise, or tools that scan objects (inspect, copy, pickle) misbehave.
        if name.startswith("__") and name.endswith("__"):
            raise AttributeError(name)
        return _FallbackType if name[:1].isupper() else self

    def __mul__(self, other):
        return self
    __rmul__ = __truediv__ = __rtruediv__ = __pow__ = __mul__
    __add__ = __radd__ = __sub__ = __rsub__ = __mul__

    def __float__(self):
        return 1.0

    def __repr__(self):
        return "<galsim astropy fallback>"


_FALLBACK_VALUE = _FallbackValue()


class _FallbackModule:
    """Module stand-in: upper-case attrs -> placeholder type, else values."""

    def __init__(self, name):
        self._name = name

    def __getattr__(self, name):
        if name.startswith("__") and name.endswith("__"):
            raise AttributeError(name)
        return _FallbackType if name[:1].isupper() else _FALLBACK_VALUE

    def __repr__(self):
        return "<galsim fallback for %s>" % self._name


try:
    from astropy import units, constants
    HAVE_ASTROPY_UNITS = True
except Exception:  # pragma: no cover - frozen builds without working units
    units = _FallbackModule("astropy.units")
    constants = _FallbackModule("astropy.constants")
    HAVE_ASTROPY_UNITS = False

if HAVE_ASTROPY_UNITS:
    try:
        import astropy.wcs as wcs
    except Exception:  # pragma: no cover - degrade wcs independently
        wcs = _FallbackModule("astropy.wcs")
else:
    wcs = _FallbackModule("astropy.wcs")

# Names galsim.config imports directly.
Quantity = units.Quantity
Unit = units.Unit
