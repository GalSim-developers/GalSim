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
#include "integ/Int.h"
#include "math/Hankel.h"
#include <iostream>

namespace galsim {
namespace integ {

#if defined(__GNUC__) && __GNUC__ >= 6
// Workaround for a bug in some versions of gcc 6-8.
// https://gcc.gnu.org/bugzilla/show_bug.cgi?id=80947
#pragma GCC visibility push(hidden)
#endif
    // A C++ function object that just calls a python function.
    class PyFunc :
        public std::unary_function<double, double>
    {
    public:
        PyFunc(const py::function& func) : _func(func) {}
        double operator()(double x) const
        { return PY_CAST<double>(_func(x)); }
    private:
        const py::function& _func;
    };
#if defined(__GNUC__) && __GNUC__ >= 6
#pragma GCC visibility pop
#endif

    // Integrate a python function using int1d.
    py::tuple PyInt1d(const py::function& func, double min, double max,
                      double rel_err=DEFRELERR, double abs_err=DEFABSERR)
    {
        PyFunc pyfunc(func);
        try {
            double res = int1d(pyfunc, min, max, rel_err, abs_err);
            return py::make_tuple(true, res);
        } catch (IntFailure& e) {
            return py::make_tuple(false, e.what());
        }
    }

    // Integrate a python function using int1d.
    void PyHankel(const py::function& func, size_t ik, size_t ires, int N,
                  double nu, double rmax,
                  double rel_err=DEFRELERR, double abs_err=DEFABSERR)
    {
        const double* k = reinterpret_cast<const double*>(ik);
        double* res = reinterpret_cast<double*>(ires);
        PyFunc pyfunc(func);
        if (rmax == 0.) {
            for (int i=0; i<N; ++i) {
                res[i] = math::hankel_inf(pyfunc, k[i], nu, rel_err, abs_err);
            }
        } else {
            for (int i=0; i<N; ++i) {
                res[i] = math::hankel_trunc(pyfunc, k[i], nu, rmax, rel_err, abs_err);
            }
        }
    }

    void pyExportInteg(PY_MODULE& _galsim)
    {
        GALSIM_DOT def("PyInt1d", &PyInt1d);
        GALSIM_DOT def("PyHankel", &PyHankel);
    }

} // namespace integ
} // namespace galsim

