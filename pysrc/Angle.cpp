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
#include "Angle.h"

namespace bp = boost::python;

namespace galsim {
namespace {

struct PyAngleUnit {

    static void wrap() {
        bp::class_< AngleUnit > pyAngleUnit("AngleUnit", bp::no_init);
        pyAngleUnit
            .def(bp::init<double>(bp::arg("val")))
            .def(bp::self == bp::self)
            .def(bp::other<double>() * bp::self)
            ;
    }

};

struct PyAngle {

    static void wrap() {
        bp::class_< Angle > pyAngle("Angle", bp::init<>());
        pyAngle
            .def(bp::init<double, AngleUnit>(bp::args("val","unit")))
            .def(bp::init<const Angle&>(bp::args("rhs")))
            .def("rad", &Angle::rad)
            .def("wrap", &Angle::wrap)
            .def(bp::self / bp::other<AngleUnit>())
            .def(bp::self * bp::other<double>())
            .def(bp::other<double>() * bp::self)
            .def(bp::self / bp::other<double>())
            .def(bp::self + bp::self)
            .def(bp::self - bp::self)
            .def(bp::self == bp::self)
            .def(bp::self != bp::self)
            .def(bp::self <= bp::self)
            .def(bp::self < bp::self)
            .def(bp::self >= bp::self)
            .def(bp::self > bp::self)
            .def(str(bp::self))
            ;
    }

};

} // anonymous

void pyExportAngle() 
{
    PyAngleUnit::wrap();
    PyAngle::wrap();

    // Also export the global variables:
    bp::scope galsim;
    galsim.attr("radians") = radians;
    galsim.attr("degrees") = degrees;
    galsim.attr("hours") = hours;
    galsim.attr("arcmin") = arcmin;
    galsim.attr("arcsec") = arcsec;
}

} // namespace galsim
