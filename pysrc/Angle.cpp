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
#include "Angle.h"
#include "NumpyHelper.h"

namespace bp = boost::python;

namespace galsim {

namespace {
    bp::tuple sincos(const double theta) {
        double sint, cost;
        Angle(theta, radians).sincos(sint, cost);
        return bp::make_tuple(sint, cost);
    }
}

void pyExportAngle()
{
    bp::class_< AngleUnit >("AngleUnit", bp::no_init)
        .def(bp::init<double>())
        .def("getValue", &AngleUnit::getValue)
        .enable_pickling()
        ;

    bp::class_< Angle >("Angle", bp::no_init)
        .def(bp::init<double, AngleUnit>())
        .def("rad", &Angle::rad)
        .enable_pickling()
        ;

    bp::def("sincos", &sincos);
}

} // namespace galsim
