# Copyright (c) 2012-2017 by the GalSim developers team on GitHub
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

from future.utils import iteritems
import galsim
from galsim.deprecated import depr
from galsim import ChromaticObject

def ChromaticObject_draw(self, *args, **kwargs):
    """A deprecated synonym for obj.drawImage(method='no_pixel')
    """
    depr('draw', 1.1, "drawImage(..., method='no_pixel'",
            'Note: drawImage has different args than draw did.  '+
            'Read the docs for the method keywords carefully.')
    normalization = kwargs.pop('normalization','f')
    if normalization in ['flux','f']:
        return self.drawImage(*args, method='no_pixel', **kwargs)
    else:
        return self.drawImage(*args, method='sb', **kwargs)

def ChromaticObject_copy(self):
    """Returns a copy of an object.  This preserves the original type of the object."""
    import copy

    depr('copy', 1.5, "", "ChromaticObjects are immutable, so there's no need for copy.")
    cls = self.__class__
    ret = cls.__new__(cls)
    for k, v in iteritems(self.__dict__):
        if k == 'objlist':
            # explicitly copy all individual items of objlist, not just the list itself
            ret.__dict__[k] = [o.copy() for o in v]
        else:
            ret.__dict__[k] = copy.copy(v)
    return ret


galsim.ChromaticObject.draw = ChromaticObject_draw
galsim.ChromaticObject.copy = ChromaticObject_copy

class Chromatic(ChromaticObject):
    """Construct chromatic versions of galsim GSObjects.

    This class was deprecated in GalSim v1.5.  Please see the ChromaticObject docstring for
    information on instantiating ChromaticObjects.

    Initialization
    --------------

    @param gsobj    A GSObject instance to be chromaticized.
    @param SED      An SED object.
    """
    def __init__(self, gsobj, SED):
        depr("Chromatic", 1.5, '',
             "Construct products of GSObjects and SEDs through multiplication: "
             "`chrom_obj = gsobj * sed.`")

        flux = gsobj.getFlux()
        self.SED = SED * flux
        self.obj = gsobj / flux
        self.wave_list = SED.wave_list
        # Chromaticized GSObjects are separable into spatial (x,y) and spectral (lambda) factors.
        self.separable = True
        self.interpolated = False
        self.deinterpolated = self

    def __eq__(self, other):
        return (isinstance(other, galsim.Chromatic) and
                self.obj == other.obj and
                self.SED == other.SED)

    def __hash__(self):
        return hash(("galsim.Chromatic", self.obj, self.SED))

    def __repr__(self):
        return 'galsim.Chromatic(%r,%r)'%(self.obj, self.SED)

    def __str__(self):
        return 'galsim.Chromatic(%s,%s)'%(self.obj, self.SED)

    def evaluateAtWavelength(self, wave):
        """Evaluate this chromatic object at a particular wavelength.

        @param wave  Wavelength in nanometers.

        @returns the monochromatic object at the given wavelength.
        """
        return self.SED(wave) * self.obj

galsim.chromatic.Chromatic = Chromatic
galsim.Chromatic = Chromatic
