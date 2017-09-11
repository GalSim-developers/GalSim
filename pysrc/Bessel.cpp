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

#include "boost/python.hpp"
#include "math/BesselRoots.h"
#include "math/Bessel.h"

namespace bp = boost::python;

namespace galsim {
namespace math {

    void pyExportBessel() {

        bp::def("j0_root", &getBesselRoot0, bp::args("s"),
                "Get the sth root of the n=0 Bessel function, J_0(x)");
        // In python, with switch from mostly matching the boost names for these to matching
        // the names scipy.special uses.
        bp::def("j0", &j0, bp::args("x"),
                "Calculate the cylindrical Bessel function, J_0(x)");
        bp::def("j1", &j1, bp::args("x"),
                "Calculate the cylindrical Bessel function, J_1(x)");
        bp::def("jv", &cyl_bessel_j, bp::args("v","x"),
                "Calculate the arbitrary v cylindrical Bessel function, J_v(x)");
        bp::def("yv", &cyl_bessel_y, bp::args("v","x"),
                "Calculate the arbitrary v cylindrical Bessel function, Y_v(x)");
        bp::def("iv", &cyl_bessel_i, bp::args("v","x"),
                "Calculate the arbitrary v modified cylindrical Bessel function, I_v(x)");
        bp::def("kv", &cyl_bessel_k, bp::args("v","x"),
                "Calculate the arbitrary v modified cylindrical Bessel function, K_v(x)");

    }

} // namespace math
} // namespace galsim

