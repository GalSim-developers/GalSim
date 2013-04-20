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
#include "Bounds.h"

namespace bp = boost::python;

#define ADD_CORNER(getter, setter, prop)\
    do {                                                            \
        bp::object fget = bp::make_function(&Bounds<T>::getter);    \
        bp::object fset = bp::make_function(&Bounds<T>::setter);    \
        pyBounds.def(#getter, fget);                                \
        pyBounds.def(#setter, fset);                                \
        pyBounds.add_property(#prop, fget, fset);                   \
    } while (false)

namespace galsim {
namespace {

template <typename T>
struct PyPosition {

    static void wrap(std::string const & suffix) {
        
        bp::class_< Position<T> >(("Position" + suffix).c_str(), bp::no_init)
            .def(bp::init< const Position<T> & >(bp::args("other")))
            .def(bp::init<T,T>((bp::arg("x")=T(0), bp::arg("y")=T(0))))
            .def_readwrite("x", &Position<T>::x)
            .def_readwrite("y", &Position<T>::y)
            .def(bp::self += bp::self)
            .def(bp::self -= bp::self)
            .def(bp::self *= bp::other<T>())
            .def(bp::self /= bp::other<T>())
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
            .def(bp::init< const Position<T> &>(bp::args("pos")))
            .def(bp::init< const Position<T> &, const Position<T> & >(bp::args("pos1", "pos2")))
            .def("isDefined", &Bounds<T>::isDefined)
            .def("center", &Bounds<T>::center)
            .def("trueCenter", &Bounds<T>::trueCenter)
            .def(bp::self += bp::self)
            .def(bp::self += bp::other< Position<T> >())
            .def(bp::self += bp::other<T>())
            .def("addBorder", &Bounds<T>::addBorder)
            .def("expand", &Bounds<T>::expand, "grow by the given factor about center")
            .def(bp::self & bp::self)
            .def("shift", (void (Bounds<T>::*)(const T, const T))&Bounds<T>::shift,
                 bp::args("dx", "dy"))
            .def("shift", (void (Bounds<T>::*)(Position<T>))&Bounds<T>::shift, bp::args("d"))
            .def("includes", (bool (Bounds<T>::*)(const Position<T> &) const)&Bounds<T>::includes)
            .def("includes", (bool (Bounds<T>::*)(const T, const T) const)&Bounds<T>::includes)
            .def("includes", (bool (Bounds<T>::*)(const Bounds<T> &) const)&Bounds<T>::includes)
            .def(bp::self == bp::self)
            .def(bp::self != bp::self)
            .def("area", &Bounds<T>::area)
            .def(str(bp::self))
            .def("assign", &Bounds<T>::operator=, bp::return_self<>())
            .enable_pickling()
            ;
        ADD_CORNER(getXMin, setXMin, xmin);
        ADD_CORNER(getXMax, setXMax, xmax);
        ADD_CORNER(getYMin, setYMin, ymin);
        ADD_CORNER(getYMax, setYMax, ymax);
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
