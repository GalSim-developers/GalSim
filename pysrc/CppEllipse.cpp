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
#include "deprecated/CppEllipse.h"
#include "NumpyHelper.h"

namespace bp = boost::python;

namespace galsim {
namespace {

struct PyCppEllipse {

    static bp::handle<> getMatrix(const CppEllipse& self) {
        static npy_intp dim[2] = {2, 2};
        // Because the C++ version sets references that are passed in, and that's not possible in
        // Python, we wrap this instead, which returns a numpy array.
        tmv::Matrix<double> m = self.getMatrix();
        PyObject* r = PyArray_SimpleNewFromData(2, dim, NPY_DOUBLE, m.ptr());
        if (!r) throw bp::error_already_set();
        PyObject* r2 = PyArray_FROM_OF(r, NPY_ARRAY_ENSURECOPY);
        Py_DECREF(r);
        return bp::handle<>(r2);
    }

    static void wrap() {
        static const char* doc = 
            "Class to describe transformation from an ellipse\n"
            "with center x0, size exp(mu), and shape s to the unit circle.\n"
            "Map from source plane to image plane is defined as\n"
            "E(x) = T(D(S(x))), where S=shear, D=dilation, T=translation.\n"
            "Conventions for order of compounding, etc., are same as for CppShear.\n"
            ;
        bp::class_<CppEllipse>("_CppEllipse", doc, bp::init<const CppEllipse &>())
            .def(
                 bp::init<const CppShear &, double, const Position<double> &>(
                     (bp::arg("s")=CppShear(), bp::arg("mu")=0., 
                      bp::arg("p")=Position<double>())
                     )
            )
            .def(-bp::self)
            .def(bp::self + bp::self)
            .def(bp::self - bp::self)
            .def(bp::self += bp::self)
            .def(bp::self -= bp::self)
            .def(bp::self == bp::self)
            .def(bp::self != bp::self)
            .def(
                "reset", (
                    void (CppEllipse::*)(
                        const CppShear&, double, const Position<double>))&CppEllipse::reset,
                 bp::args("s", "mu", "p"))
            .def("fwd", &CppEllipse::fwd, "FIXME: needs documentation!")
            .def("inv", &CppEllipse::inv, "FIXME: needs documentation!")
            .def("setS", &CppEllipse::setS, bp::return_self<>())
            .def("setMu", &CppEllipse::setMu, bp::return_self<>())
            .def("setX0", &CppEllipse::setX0, bp::return_self<>())
            .def("getS", &CppEllipse::getS)
            .def("getMu", &CppEllipse::getMu)
            .def("getX0", &CppEllipse::getX0)
            .def("getMajor", &CppEllipse::getMajor, "FIXME: is this semi-major or full major axis?")
            .def("getMinor", &CppEllipse::getMinor, "FIXME: is this semi-minor or full minor axis?")
            .def("getBeta", &CppEllipse::getMinor, "position angle FIXME: which convention?")
            .def("range", &CppEllipse::range, (bp::arg("nSigma")=1.))
            .def("getMatrix", &getMatrix)
            .def(str(bp::self))
            .def("assign", &CppEllipse::operator=, bp::return_self<>())
            ;
    }

};

} // anonymous

void pyExportCppEllipse() {
    PyCppEllipse::wrap();
}

} // namespace galsim
