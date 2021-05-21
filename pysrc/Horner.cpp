/* -*- c++ -*-
 * Copyright (c) 2012-2021 by the GalSim developers team on GitHub
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
#include "math/Horner.h"

namespace galsim {
namespace math {

    static void _Horner(size_t ix, int nx, size_t icoef, int nc, size_t iresult)
    {
        double* x = reinterpret_cast<double*>(ix);
        double* coef = reinterpret_cast<double*>(icoef);
        double* result = reinterpret_cast<double*>(iresult);
        Horner(x, nx, coef, nc, result);
    }

    static void _Horner2D(size_t ix, size_t iy, int nx, size_t icoef, int ncx, int ncy,
                          size_t iresult, size_t itemp)
    {
        double* x = reinterpret_cast<double*>(ix);
        double* y = reinterpret_cast<double*>(iy);
        double* coef = reinterpret_cast<double*>(icoef);
        double* result = reinterpret_cast<double*>(iresult);
        double* temp = reinterpret_cast<double*>(itemp);
        Horner2D(x, y, nx, coef, ncx, ncy, result, temp);
    }

    void pyExportHorner(PY_MODULE& _galsim)
    {
        GALSIM_DOT def("Horner", &_Horner);
        GALSIM_DOT def("Horner2D", &_Horner2D);
    }

} // namespace math
} // namespace galsim

