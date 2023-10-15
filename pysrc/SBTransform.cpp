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
#include "SBTransform.h"

namespace galsim {

    static SBTransform* MakeSBT(const SBProfile& sbin, size_t ijac,
                                double cenx, double ceny, double ampScaling,
                                const GSParams& gsparams)
    {
        const double* jac = reinterpret_cast<const double*>(ijac);
        return new SBTransform(sbin, jac, Position<double>(cenx,ceny), ampScaling, gsparams);
    }

    template <typename T>
    void _ApplyKImagePhases(ImageView<std::complex<T> > image, double imscale, size_t ijac,
                            double cenx, double ceny, double fluxScaling)
    {
        const double* jac = reinterpret_cast<const double*>(ijac);
        ApplyKImagePhases(image, imscale, jac, cenx, ceny, fluxScaling);
    }

    template <typename T>
    static void WrapTemplates(py::module& _galsim)
    {
        typedef void (*phase_func)(ImageView<std::complex<T> >,
                                   double, size_t, double, double, double);
        _galsim.def("ApplyKImagePhases", phase_func(&_ApplyKImagePhases));
    }

    void pyExportSBTransform(py::module& _galsim)
    {
        py::class_<SBTransform, SBProfile>(_galsim, "SBTransform")
            .def(py::init(&MakeSBT));

        WrapTemplates<float>(_galsim);
        WrapTemplates<double>(_galsim);
    }


} // namespace galsim
