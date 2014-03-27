/* -*- c++ -*-
 * Copyright 2012-2014 The GalSim developers:
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
#ifndef __INTEL_COMPILER
#if defined(__GNUC__) && __GNUC__ >= 4 && (__GNUC__ >= 5 || __GNUC_MINOR__ >= 8)
#pragma GCC diagnostic ignored "-Wunused-local-typedefs"
#endif
#endif

#define BOOST_NO_CXX11_SMART_PTR
#include "boost/python.hpp"
#include "bessel/Roots.h"
#include <boost/math/special_functions/bessel.hpp>

namespace bp = boost::python;

namespace galsim {
namespace bessel {

    // The boost versions are templated.  Make them concrete here so they are 
    // easier to wrap.
    inline double BesselJv(double v, double x) 
    { return boost::math::cyl_bessel_j(v,x); }

    inline double BesselJn(int n, double x) 
    { return boost::math::cyl_bessel_j(n,x); }

    inline double BesselKv(double v, double x) 
    { return boost::math::cyl_bessel_k(v,x); }

    inline double BesselKn(int n, double x) 
    { return boost::math::cyl_bessel_k(n,x); }

    void pyExportBessel() {

        bp::def("j0_root", &getBesselRoot0, bp::args("s"),
                "Get the sth root of the n=0 Bessel function, J_0(x)");
        bp::def("j0", &j0, bp::args("x"),
                "Calculate the n=0 cylindrical Bessel function, J_0(x)");
        bp::def("j1", &j1, bp::args("x"),
                "Calculate the n=1 cylindrical Bessel function, J_1(x)");
        bp::def("jn", &BesselJn, bp::args("n","x"),
                "Calculate the arbitrary n cylindrical Bessel function, J_n(x)");
        bp::def("jv", &BesselJv, bp::args("v","x"),
                "Calculate the arbitrary v cylindrical Bessel function, J_v(x)");
        bp::def("kn", &BesselKn, bp::args("n","x"),
                "Calculate the modified cylindrical Bessel function, K_n(x)");
        bp::def("kv", &BesselKv, bp::args("v","x"),
                "Calculate the arbitrary v modified cylindrical Bessel function, K_v(x)");

    }

} // namespace bessel
} // namespace galsim

