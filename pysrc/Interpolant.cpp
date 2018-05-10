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
#include <cstdlib>
#include "Interpolant.h"

namespace galsim {

    void pyExportInterpolant(PY_MODULE& _galsim)
    {
        py::class_<Interpolant BP_NONCOPYABLE>(GALSIM_COMMA "Interpolant" BP_NOINIT);

        py::class_<Delta, BP_BASES(Interpolant)>(GALSIM_COMMA "Delta" BP_NOINIT)
            .def(py::init<double,GSParams>());

        py::class_<Nearest, BP_BASES(Interpolant)>(GALSIM_COMMA "Nearest" BP_NOINIT)
            .def(py::init<double,GSParams>());

        py::class_<SincInterpolant, BP_BASES(Interpolant)>(GALSIM_COMMA "SincInterpolant" BP_NOINIT)
            .def(py::init<double,GSParams>());

        py::class_<Lanczos, BP_BASES(Interpolant)>(GALSIM_COMMA "Lanczos" BP_NOINIT)
            .def(py::init<int,bool,double,GSParams>())
            .def("urange", &Lanczos::urange);

        py::class_<Linear, BP_BASES(Interpolant)>(GALSIM_COMMA "Linear" BP_NOINIT)
            .def(py::init<double,GSParams>());

        py::class_<Cubic, BP_BASES(Interpolant)>(GALSIM_COMMA "Cubic" BP_NOINIT)
            .def(py::init<double,GSParams>());

        py::class_<Quintic, BP_BASES(Interpolant)>(GALSIM_COMMA "Quintic" BP_NOINIT)
            .def(py::init<double,GSParams>());
    }

} // namespace galsim
