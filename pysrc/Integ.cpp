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

#include "PyBind11Helper.h"
#include "integ/Int.h"
#include <iostream>

namespace galsim {
namespace integ {

    // A C++ function object that just calls a python function.
    class PyFunc :
        public std::unary_function<double, double>
    {
    public:
        PyFunc(const bp::object& func) : _func(func) {}
        double operator()(double x) const
        { return CAST<double>(_func(x)); }
    private:
        const bp::object& _func;
    };

    // Integrate a python function using int1d.
    bp::tuple PyInt1d(const bp::object& func, double min, double max,
                      double rel_err=DEFRELERR, double abs_err=DEFABSERR)
    {
        PyFunc pyfunc(func);
        try {
            double res = int1d(pyfunc, min, max, rel_err, abs_err);
            return bp::make_tuple(true, res);
        } catch (IntFailure& e) {
            return bp::make_tuple(false, e.what());
        }
    }

    void pyExportInteg(PB11_MODULE& _galsim)
    {
        GALSIM_DOT def("PyInt1d", &PyInt1d);

    }

} // namespace integ
} // namespace galsim

