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
#include "Angle.h"
#include "NumpyHelper.h"

namespace bp = boost::python;

namespace galsim {
namespace {

struct PyAngleUnit {

    static void wrap() {
        typedef double (*div_func)(AngleUnit, AngleUnit);

        bp::class_< AngleUnit > pyAngleUnit("AngleUnit", bp::no_init);
        pyAngleUnit
            .def(bp::init<double>(bp::arg("val")))
            .def(bp::self == bp::self)
            .def("getValue", &AngleUnit::getValue)
            .def(bp::other<double>() * bp::self)
            .def("__div__", div_func(&operator/))
            .def("__truediv__", div_func(&operator/))
            .enable_pickling()
            ;
    }

};

struct PyAngle {

    static bp::handle<> sincos(const Angle& self) {
        static npy_intp dim[1] = {2};
        double ar[2];
        self.sincos(ar[0], ar[1]);
        PyObject* r = PyArray_SimpleNewFromData(1, dim, NPY_DOUBLE, ar);
        if (!r) throw bp::error_already_set();
        PyObject* r2 = PyArray_FROM_OF(r, NPY_ARRAY_ENSURECOPY);
        Py_DECREF(r);
        return bp::handle<>(r2);
    }

    static void wrap() {
        typedef double (Angle::*div_func1)(AngleUnit) const;
        typedef Angle (Angle::*div_func2)(double) const;

        bp::class_< Angle > pyAngle("Angle", bp::init<>());
        pyAngle
            .def(bp::init<double, AngleUnit>(bp::args("val","unit")))
            .def(bp::init<const Angle&>(bp::args("rhs")))
            .def("rad", &Angle::rad)
            .def("wrap", &Angle::wrap, bp::arg("center")=0.*radians)
            .def("sin", &Angle::sin)
            .def("cos", &Angle::cos)
            .def("tan", &Angle::tan)
            .def("sincos", sincos)
            .def("__div__", div_func1(&Angle::operator/))
            .def("__truediv__", div_func1(&Angle::operator/))
            .def(bp::self * bp::other<double>())
            .def(bp::other<double>() * bp::self)
            .def("__div__", div_func2(&Angle::operator/))
            .def("__truediv__", div_func2(&Angle::operator/))
            .def(bp::self + bp::self)
            .def(bp::self - bp::self)
            .def(bp::self == bp::self)
            .def(bp::self != bp::self)
            .def(bp::self <= bp::self)
            .def(bp::self < bp::self)
            .def(bp::self >= bp::self)
            .def(bp::self > bp::self)
            .def(str(bp::self))
            .enable_pickling()
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
