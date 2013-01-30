// -*- c++ -*-
/*
 * Copyright 2012, 2013 The GalSim developers:
 * https://github.com/GalSim-developers
 *
 * This file is part of GalSim: The modular galaxy image simulation toolkit.
 *
 * GalSim is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * GalSim is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with GalSim.  If not, see <http://www.gnu.org/licenses/>
 */
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

