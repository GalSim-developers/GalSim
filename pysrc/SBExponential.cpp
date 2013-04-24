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

#include "SBExponential.h"

namespace bp = boost::python;

namespace galsim {

    // Used by multiple profile classes to ensure at most one radius is given.
    static void checkRadii(const bp::object & r1, const bp::object & r2, const bp::object & r3) 
    {
        int nRad = (r1.ptr() != Py_None) + (r2.ptr() != Py_None) + (r3.ptr() != Py_None);
        if (nRad > 1) {
            PyErr_SetString(PyExc_TypeError, "Multiple radius parameters given");
            bp::throw_error_already_set();
        }
        if (nRad == 0) {
            PyErr_SetString(PyExc_TypeError, "No radius parameter given");
            bp::throw_error_already_set();
        }
    }

    struct PySBExponential 
    {

        static SBExponential* construct(
            const bp::object& half_light_radius, const bp::object& scale_radius, double flux,
            boost::shared_ptr<GSParams> gsparams) 
        {
            double s = 1.0;
            checkRadii(half_light_radius, scale_radius, bp::object());
            if (half_light_radius.ptr() != Py_None) {
                s = bp::extract<double>(half_light_radius) / 1.6783469900166605; // not analytic
            }
            if (scale_radius.ptr() != Py_None) {
                s = bp::extract<double>(scale_radius);
            }
            return new SBExponential(s, flux, gsparams);
        }

        static void wrap() 
        {
            bp::class_<SBExponential,bp::bases<SBProfile> >(
                "SBExponential",
                "SBExponential(flux=1., half_light_radius=None, scale=None)\n\n"
                "Construct an exponential profile with the given flux and either half-light\n"
                "radius or scale length.  Exactly one radius must be provided.\n",
                bp::no_init)
                .def("__init__", bp::make_constructor(
                        &construct, bp::default_call_policies(),
                        (bp::arg("half_light_radius")=bp::object(), 
                         bp::arg("scale_radius")=bp::object(), (bp::arg("flux")=1.),
                         bp::arg("gsparams")=bp::object()))
                )
                .def(bp::init<const SBExponential &>())
                .def("getScaleRadius", &SBExponential::getScaleRadius)
                ;
        }
    };

    void pyExportSBExponential() 
    {
        PySBExponential::wrap();
    }

} // namespace galsim
