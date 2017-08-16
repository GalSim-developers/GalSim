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

        // These will be imported into the galsim.bessel namespace, so remove the Bessel
        // part of the name to avoid redundancy.  Also, go to lowercase to match the names
        // scipy.special uses.
        bp::def("j0_root", &BesselJ0Root, bp::args("s"),
                "Get the sth root of the n=0 Bessel function, J_0(x)");
        bp::def("j0", &BesselJ0, bp::args("x"),
                "Calculate the cylindrical Bessel function, J_0(x)");
        bp::def("j1", &BesselJ1, bp::args("x"),
                "Calculate the cylindrical Bessel function, J_1(x)");
        bp::def("jv", &BesselJ, bp::args("nu","x"),
                "Calculate the cylindrical Bessel function, J_nu(x)");
        bp::def("kv", &BesselK, bp::args("nu","x"),
                "Calculate the modified cylindrical Bessel function, K_nu(x)");
        bp::def("yv", &BesselY, bp::args("nu","x"),
                "Calculate the cylindrical Bessel function, Y_nu(x)");
        bp::def("iv", &BesselI, bp::args("nu","x"),
                "Calculate the modified cylindrical Bessel function, I_nu(x)");

    }

} // namespace math
} // namespace galsim

