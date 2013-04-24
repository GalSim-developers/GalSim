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
#include "boost/python/stl_iterator.hpp"

#include "SBBox.h"

namespace bp = boost::python;

namespace galsim {

    struct PySBBox 
    {

        static SBBox* construct(const bp::object & xw, const bp::object & yw, double flux,
                                boost::shared_ptr<GSParams> gsparams) 
        {
            if (xw.ptr() == Py_None || yw.ptr() == Py_None) {
                PyErr_SetString(PyExc_TypeError, "SBBox requires x and y width parameters");
                bp::throw_error_already_set();
            }
            return new SBBox(bp::extract<double>(xw), bp::extract<double>(yw), flux, gsparams);
        }

        static void wrap() 
        {
            bp::class_<SBBox,bp::bases<SBProfile> >("SBBox", bp::no_init)
                .def("__init__", bp::make_constructor(
                        &construct, bp::default_call_policies(),
                        (bp::arg("xw")=bp::object(), bp::arg("yw")=bp::object(), 
                         bp::arg("flux")=1.,
                         bp::arg("gsparams")=bp::object())
                ))
                .def(bp::init<const SBBox &>())
                .def("getXWidth", &SBBox::getXWidth)
                .def("getYWidth", &SBBox::getYWidth)
                ;
        }
    };

    void pyExportSBBox() 
    {
        PySBBox::wrap();
    }

} // namespace galsim
