/* -*- c++ -*-
 * Copyright (c) 2012-2016 by the GalSim developers team on GitHub
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
#include "math/BesselRoots.h"
#include "math/Bessel.h"
#include <boost/math/special_functions/bessel.hpp>

namespace bp = boost::python;

namespace galsim {
namespace math {

    // The boost versions are templated.  Make them concrete here so they are
    // easier to wrap.
    inline double BesselJv(double v, double x)
    { return boost::math::cyl_bessel_j(v,x); }

    inline double BesselJn(int n, double x)
    { return boost::math::cyl_bessel_j(n,x); }

    inline double BesselKv(double v, double x)
    { return math::cyl_bessel_k(v,x); }

    inline double BesselKn(int n, double x)
    { return math::cyl_bessel_k(n,x); }

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

} // namespace math
} // namespace galsim

