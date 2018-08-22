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
#include "TableOld.h"

namespace galsim {

    static Table2DOld* MakeTable2DOld(size_t ix, size_t iy, size_t ivals, int Nx, int Ny,
                                const char* interp_c)
    {
        const double* x = reinterpret_cast<const double*>(ix);
        const double* y = reinterpret_cast<const double*>(iy);
        const double* vals = reinterpret_cast<const double*>(ivals);
        std::string interp(interp_c);

        Table2DOld::interpolant i = Table2DOld::linear;
        if (interp == "floor") i = Table2DOld::floor;
        else if (interp == "ceil") i = Table2DOld::ceil;
        else if (interp == "nearest") i = Table2DOld::nearest;

        return new Table2DOld(x, y, vals, Nx, Ny, i);
    }

    static void InterpMany2D(const Table2DOld& Table2DOld, size_t ix, size_t iy, size_t ivals, int N)
    {
        const double* x = reinterpret_cast<const double*>(ix);
        const double* y = reinterpret_cast<const double*>(iy);
        double* vals = reinterpret_cast<double*>(ivals);
        Table2DOld.interpMany(x, y, vals, N);
    }

    static void Gradient(const Table2DOld& Table2DOld, double x, double y, size_t igrad)
    {
        double* grad = reinterpret_cast<double*>(igrad);
        Table2DOld.gradient(x, y, grad[0], grad[1]);
    }

    static void GradientMany(const Table2DOld& Table2DOld,
                             size_t ix, size_t iy, size_t idfdx, size_t idfdy, int N)
    {
        const double* x = reinterpret_cast<const double*>(ix);
        const double* y = reinterpret_cast<const double*>(iy);
        double* dfdx = reinterpret_cast<double*>(idfdx);
        double* dfdy = reinterpret_cast<double*>(idfdy);
        Table2DOld.gradientMany(x, y, dfdx, dfdy, N);
    }

    void pyExportTableOld(PY_MODULE& _galsim)
    {
        py::class_<Table2DOld>(GALSIM_COMMA "_LookupTable2DOld" BP_NOINIT)
            .def(PY_INIT(&MakeTable2DOld))
            .def("interp", &Table2DOld::lookup)
            .def("interpMany", &InterpMany2D)
            .def("gradient", &Gradient)
            .def("gradientMany", &GradientMany);
    }

} // namespace galsim
