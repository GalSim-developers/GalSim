/* -*- c++ -*-
 * Copyright (c) 2012-2023 by the GalSim developers team on GitHub
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
#include "SBConvolve.h"

namespace galsim {

    static SBConvolve* construct(
        const std::list<SBProfile>& plist, bool real_space, GSParams gsparams)
    {
        return new SBConvolve(plist, real_space, gsparams);
    }

    void pyExportSBConvolve(py::module& _galsim)
    {
        py::class_<SBConvolve, SBProfile>(_galsim, "SBConvolve")
            .def(py::init(&construct));
        py::class_<SBAutoConvolve, SBProfile>(_galsim, "SBAutoConvolve")
            .def(py::init<const SBProfile&, bool, GSParams>());
        py::class_<SBAutoCorrelate, SBProfile>(_galsim, "SBAutoCorrelate")
            .def(py::init<const SBProfile&, bool, GSParams>());
    }

} // namespace galsim
