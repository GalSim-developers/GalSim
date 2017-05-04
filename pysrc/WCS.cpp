/* -*- c++ -*-
 * Copyright (c) 2012-2017 by the GalSim developers team on GitHub
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

#include "galsim/IgnoreWarnings.h"

#define BOOST_NO_CXX11_SMART_PTR
#include "boost/python.hpp"
#include "WCS.h"

namespace bp = boost::python;

namespace galsim {

    void CallApplyCD(int n, size_t x_data, size_t y_data, size_t cd_data)
    {
        double* xar = reinterpret_cast<double*>(x_data);
        double* yar = reinterpret_cast<double*>(y_data);
        const double* cdar = reinterpret_cast<const double*>(cd_data);
        ApplyCD(n, xar, yar, cdar);
    };

    void CallApplyPV(int n, int m, size_t u_data, size_t v_data, size_t pv_data)
    {
        double* uar = reinterpret_cast<double*>(u_data);
        double* var = reinterpret_cast<double*>(v_data);
        const double* pvar = reinterpret_cast<const double*>(pv_data);
        ApplyPV(n, m, uar, var, pvar);
    };

    bp::tuple CallInvertPV(double u, double v, size_t pv_data)
    {
        const double* pvar = reinterpret_cast<const double*>(pv_data);
        InvertPV(u, v, pvar);
        return bp::make_tuple(u,v);
    };

    bp::tuple CallInvertAB(int m, double x, double y, size_t ab_data, size_t abp_data)
    {
        const double* abar = reinterpret_cast<const double*>(ab_data);
        const double* abpar = reinterpret_cast<const double*>(abp_data);
        InvertAB(m, x, y, abar, abpar);
        return bp::make_tuple(x,y);
    };

    void pyExportWCS() {
        bp::def("ApplyPV", &CallApplyPV);
        bp::def("ApplyCD", &CallApplyCD);
        bp::def("InvertPV", &CallInvertPV);
        bp::def("InvertAB", &CallInvertAB);
    }

} // namespace galsim

