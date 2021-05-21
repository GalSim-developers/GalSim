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
#include "WCS.h"

namespace galsim {

    void CallApplyCD(int n, size_t x_data, size_t y_data, size_t cd_data)
    {
        double* x = reinterpret_cast<double*>(x_data);
        double* y = reinterpret_cast<double*>(y_data);
        const double* cd = reinterpret_cast<const double*>(cd_data);
        ApplyCD(n, x, y, cd);
    }

    void CallInvertAB(int n, int nab, size_t u_data, size_t v_data, size_t ab_data,
                      size_t x_data, size_t y_data, bool doiter,
                      int nabp, size_t abp_data)
    {
        const double* u = reinterpret_cast<const double*>(u_data);
        const double* v = reinterpret_cast<const double*>(v_data);
        const double* ab = reinterpret_cast<const double*>(ab_data);
        const double* abp = reinterpret_cast<const double*>(abp_data);
        double* x = reinterpret_cast<double*>(x_data);
        double* y = reinterpret_cast<double*>(y_data);
        InvertAB(n, nab, u, v, ab, x, y, doiter, nabp, abp);
    }

    void pyExportWCS(PY_MODULE& _galsim)
    {
        GALSIM_DOT def("ApplyCD", &CallApplyCD);
        GALSIM_DOT def("InvertAB", &CallInvertAB);
    }

} // namespace galsim
