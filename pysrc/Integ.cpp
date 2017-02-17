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
#include "integ/Int.h"

#include <iostream>

namespace bp = boost::python;

namespace galsim {
namespace integ {
namespace {

    // A C++ function object that just calls a python function.
    class PyFunc :
        public std::unary_function<double, double>
    {
    public:
        PyFunc(const bp::object& func) : _func(func) {}
        double operator()(double x) const
        { return bp::extract<double>(_func(x)); }
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

} // anonymous


void pyExportInteg() {

    bp::def("PyInt1d",
            &PyInt1d, (bp::args("func", "min", "max"),
                       bp::arg("rel_err")=DEFRELERR, bp::arg("abs_err")=DEFABSERR),
            "Calculate the integral of the given 1-d function from min to max.");

}

} // namespace integ
} // namespace galsim

