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
#include "SBProfile.h"
#include "SBTransform.h"

namespace galsim {

    template <typename T, typename W>
    static void WrapTemplates(W& wrapper)
    {
        wrapper.def("draw", (void (SBProfile::*)(ImageView<T>, double) const)&SBProfile::draw);
        wrapper.def("drawK", (void (SBProfile::*)(ImageView<std::complex<T> >, double) const)
                    &SBProfile::drawK);
    }

    void pyExportSBProfile(PY_MODULE& _galsim)
    {
        py::class_<GSParams>(GALSIM_COMMA "GSParams" BP_NOINIT)
            .def(py::init<
                 int, int, double, double, double, double, double, double, double, double,
                 double, double, double>());

        py::class_<SBProfile> pySBProfile(GALSIM_COMMA "SBProfile" BP_NOINIT);
        pySBProfile
            .def("xValue", &SBProfile::xValue)
            .def("kValue", &SBProfile::kValue)
            .def("maxK", &SBProfile::maxK)
            .def("stepK", &SBProfile::stepK)
            .def("centroid", &SBProfile::centroid)
            .def("getFlux", &SBProfile::getFlux)
            .def("getPositiveFlux", &SBProfile::getPositiveFlux)
            .def("getNegativeFlux", &SBProfile::getNegativeFlux)
            .def("maxSB", &SBProfile::maxSB)
            .def("shoot", &SBProfile::shoot);
        WrapTemplates<float>(pySBProfile);
        WrapTemplates<double>(pySBProfile);
    }

} // namespace galsim
