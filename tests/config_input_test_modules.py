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


import galsim.config
import galsim.config.value_eval
import numpy as np

galsim.config.eval_base_variables += ["input_size_0", "input_size_arr"]


def _ret_size(size=-1):
    return size


class InputSizeLoader(galsim.config.InputLoader):
    def getKwargs(self, config, base, logger):
        req = {"size": int}
        kwargs, safe = galsim.config.GetAllParams(config, base, req=req)
        if self.takes_logger:
            kwargs["logger"] = logger
        return kwargs, True

    def initialize(self, input_objs, num, base, logger):
        if num == 0:
            base['input_size_0'] = input_objs[0]
        if all(iobj is not None for iobj in input_objs):
            base['input_size_arr'] = np.array(input_objs, dtype=int)


galsim.config.RegisterInputType(
    "input_size_module",
    InputSizeLoader(_ret_size, file_scope=True),
)
