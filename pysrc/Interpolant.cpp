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
#include <cstdlib>
#include "Interpolant.h"

namespace galsim {

    static void XvalMany(const Interpolant& interp, size_t ivals, int N)
    {
        double* vals = reinterpret_cast<double*>(ivals);
        interp.xvalMany(vals, N);
    }

    static void UvalMany(const Interpolant& interp, size_t ivals, int N)
    {
        double* vals = reinterpret_cast<double*>(ivals);
        interp.uvalMany(vals, N);
    }

    void pyExportInterpolant(py::module& _galsim)
    {
        py::class_<Interpolant >(_galsim, "Interpolant")
            .def("xval", &Interpolant::xval)
            .def("uval", &Interpolant::uval)
            .def("xvalMany", &XvalMany)
            .def("uvalMany", &UvalMany)
            .def("getPositiveFlux", &Interpolant::getPositiveFlux)
            .def("getNegativeFlux", &Interpolant::getNegativeFlux)
            .def("urange", &Interpolant::urange);

        py::class_<Delta, Interpolant>(_galsim, "Delta")
            .def(py::init<GSParams>());

        py::class_<Nearest, Interpolant>(_galsim, "Nearest")
            .def(py::init<GSParams>());

        py::class_<SincInterpolant, Interpolant>(_galsim, "SincInterpolant")
            .def(py::init<GSParams>());

        py::class_<Lanczos, Interpolant>(_galsim, "Lanczos")
            .def(py::init<int,bool,GSParams>());

        py::class_<Linear, Interpolant>(_galsim, "Linear")
            .def(py::init<GSParams>());

        py::class_<Cubic, Interpolant>(_galsim, "Cubic")
            .def(py::init<GSParams>());

        py::class_<Quintic, Interpolant>(_galsim, "Quintic")
            .def(py::init<GSParams>());
    }

} // namespace galsim
