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

import galsim
from galsim.deprecated import depr

def Bandpass_rdiv(self, other):
    depr('__rdiv__', 1.3, "Bandpass(throughput=lambda wave:other/bandpass(wave))",
            "We removed this function because we don't know of any clear use case. "+
            "If you have one, please open an issue, and we can add this function back.")
    blue_limit = self.blue_limit
    red_limit = self.red_limit
    wave_list = self.wave_list

    if isinstance(other, galsim.Bandpass):
        if len(other.wave_list) > 0:
            wave_list = np.union1d(wave_list, other.wave_list)
        blue_limit = max([self.blue_limit, other.blue_limit])
        red_limit = min([self.red_limit, other.red_limit])
        wave_list = wave_list[(wave_list >= blue_limit) & (wave_list <= red_limit)]

    if hasattr(other, '__call__'):
        tp = lambda w: other(w) / self.func(w)
    else:
        tp = lambda w: other / self.func(w)

    return galsim.Bandpass(tp, 'nm', blue_limit, red_limit,
                           _wave_list=wave_list)

def Bandpass_copy(self):
    depr('copy', 1.5, "", "Bandpasses are immutable, so there's no need for copy.")
    import copy
    return copy.deepcopy(self)


galsim.Bandpass.__rdiv__ = Bandpass_rdiv
galsim.Bandpass.__rtruediv__ = Bandpass_rdiv
galsim.Bandpass.copy = Bandpass_copy
