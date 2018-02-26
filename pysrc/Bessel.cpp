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

        bp::def("j0_root", &getBesselRoot0);
        // In python, with switch from mostly matching the boost names for these to matching
        // the names scipy.special uses.
        bp::def("j0", &j0);
        bp::def("j1", &j1);
        bp::def("jv", &cyl_bessel_j);
        bp::def("yv", &cyl_bessel_y);
        bp::def("iv", &cyl_bessel_i);
        bp::def("kv", &cyl_bessel_k);

    }

} // namespace math
} // namespace galsim

