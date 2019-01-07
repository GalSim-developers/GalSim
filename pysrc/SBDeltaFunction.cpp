/* -*- c++ -*-
 * Copyright (c) 2012-2018 by the GalSim developers team on GitHub
 * https://github.com/GalSim-developers
 *
 * This file is part of GalSim: The modular galaxy image simulation toolkit.
 * https://github.com/GalSim-developers/GalSim
 *
 * GalSim is free software: redistribution and use in source and binary forms,
 * with or without modification, are permitted provided that the following
 * conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice, this
 *    list of conditions, and the disclaimer given in the accompanying LICENSE
 *    file.
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 *    this list of conditions, and the disclaimer given in the documentation
 *    and/or other materials provided with the distribution.
 */

#include "PyBind11Helper.h"
#include "SBDeltaFunction.h"

namespace galsim {

    void pyExportSBDeltaFunction(PY_MODULE& _galsim)
    {
        py::class_<SBDeltaFunction, BP_BASES(SBProfile)>(GALSIM_COMMA "SBDeltaFunction" BP_NOINIT)
            .def(py::init<double, GSParams>());
    }

} // namespace galsim
