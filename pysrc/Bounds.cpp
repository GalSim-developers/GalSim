/* -*- c++ -*-
 * Copyright (c) 2012-2014 by the GalSim developers team on GitHub
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

#ifndef __INTEL_COMPILER
#if defined(__GNUC__) && __GNUC__ >= 4 && (__GNUC__ >= 5 || __GNUC_MINOR__ >= 8)
#pragma GCC diagnostic ignored "-Wunused-local-typedefs"
#endif
#endif

#define BOOST_NO_CXX11_SMART_PTR
#include "boost/python.hpp"
#include "Bounds.h"

namespace bp = boost::python;

namespace galsim {
namespace {

template <typename T>
struct PyPosition {

    static void wrap(std::string const & suffix) {
        
        bp::class_< Position<T> > pyPosition(("Position" + suffix).c_str(), bp::no_init);
        pyPosition.def(bp::init< const Position<T>& >(bp::args("other")))
            .def(bp::init<T,T>((bp::arg("x")=T(0), bp::arg("y")=T(0))))
            .def_readonly("x", &Position<T>::x)
            .def_readonly("y", &Position<T>::y)
            .def(bp::self * bp::other<T>())
            .def(bp::self / bp::other<T>())
            .def(bp::other<T>() * bp::self)
            .def(-bp::self)
            .def(bp::self + bp::self)
            .def(bp::self - bp::self)
            .def(bp::self == bp::self)
            .def(bp::self != bp::self)
            .def(str(bp::self))
            .def("assign", &Position<T>::operator=, bp::return_self<>())
            .enable_pickling()
            ;
    }

};

template <typename T>
struct PyBounds {

    static void wrap(std::string const & suffix) {
        bp::class_< Bounds<T> > pyBounds(("Bounds" + suffix).c_str(), bp::init<>());
        pyBounds
            .def(bp::init<T,T,T,T>(bp::args("xmin","xmax","ymin","ymax")))
            .def(bp::init< const Position<T>& >(bp::args("pos")))
            .def(bp::init< const Position<T>&, const Position<T>& >(bp::args("pos1", "pos2")))
            .def("copy", &Bounds<T>::copy)
            .def("isDefined", &Bounds<T>::isDefined)
            .def("origin", &Bounds<T>::origin)
            .def("center", &Bounds<T>::center)
            .def("trueCenter", &Bounds<T>::trueCenter)
            // Note: the python methods always use the version that returns a new bounds object.
            // This matches the typical python style of objects being immutable, and you get
            // new objects back when you want to change them.
            .def("withBorder", &Bounds<T>::withBorder)
            .def("expand", &Bounds<T>::makeExpanded, "grow by the given factor about center")
            .def(bp::self & bp::self)
            .def("shift", (Bounds<T> (Bounds<T>::*)(const Position<T>&) const)&Bounds<T>::makeShifted,
                           bp::args("delta"))
            .def("includes", (bool (Bounds<T>::*)(const Position<T>&) const)&Bounds<T>::includes)
            .def("includes", (bool (Bounds<T>::*)(const T, const T) const)&Bounds<T>::includes)
            .def("includes", (bool (Bounds<T>::*)(const Bounds<T>&) const)&Bounds<T>::includes)
            .def(bp::self == bp::self)
            .def(bp::self != bp::self)
            .def("area", &Bounds<T>::area)
            .def(str(bp::self))
            .def("assign", &Bounds<T>::operator=, bp::return_self<>())
            .def(bp::self + bp::self)
            .def(bp::self + bp::other< Position<T> >())
            .def(bp::self + bp::other<T>())
            .def("getXMin", &Bounds<T>::getXMin)
            .def("getXMax", &Bounds<T>::getXMax)
            .def("getYMin", &Bounds<T>::getYMin)
            .def("getYMax", &Bounds<T>::getYMax)
            .add_property("xmin", &Bounds<T>::getXMin)
            .add_property("xmax", &Bounds<T>::getXMax)
            .add_property("ymin", &Bounds<T>::getYMin)
            .add_property("ymax", &Bounds<T>::getYMax)
            .def("_setXMin", &Bounds<T>::setXMin)
            .def("_setXMax", &Bounds<T>::setXMax)
            .def("_setYMin", &Bounds<T>::setYMin)
            .def("_setYMax", &Bounds<T>::setYMax)
            .enable_pickling()
            ;
    }

};

} // anonymous

void pyExportBounds() {
    PyPosition<double>::wrap("D");
    PyBounds<double>::wrap("D");
    PyPosition<int>::wrap("I");
    PyBounds<int>::wrap("I");
}

} // namespace galsim
