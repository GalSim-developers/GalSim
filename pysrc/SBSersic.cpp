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

#include "SBSersic.h"

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

    struct PySBSersic 
    {

        static SBSersic* construct(
            double n, const bp::object & scale_radius, const bp::object & half_light_radius,
            double trunc, double flux, bool flux_untruncated,
            boost::shared_ptr<GSParams> gsparams)
        {
            double s = 1.0;
            checkRadii(half_light_radius, scale_radius, bp::object());
            SBSersic::RadiusType rType = SBSersic::HALF_LIGHT_RADIUS;
            if (half_light_radius.ptr() != Py_None) {
                s = bp::extract<double>(half_light_radius);
            }
            if (scale_radius.ptr() != Py_None) {
                s = bp::extract<double>(scale_radius);
                rType = SBSersic::SCALE_RADIUS;
            }
            return new SBSersic(n, s, rType, flux, trunc, flux_untruncated, gsparams);
        }

        static void wrap() 
        {
            bp::class_<SBSersic,bp::bases<SBProfile> >("SBSersic", bp::no_init)
                .def("__init__",
                     bp::make_constructor(
                         &construct, bp::default_call_policies(),
                         (bp::arg("n"),
                          bp::arg("scale_radius")=bp::object(),
                          bp::arg("half_light_radius")=bp::object(),
                          bp::arg("trunc")=0.,
                          bp::arg("flux")=1., bp::arg("flux_untruncated")=false,
                          bp::arg("gsparams")=bp::object())
                     )
                )
                .def(bp::init<const SBSersic &>())
                .def("getN", &SBSersic::getN)
                .def("getHalfLightRadius", &SBSersic::getHalfLightRadius)
                .def("getScaleRadius", &SBSersic::getScaleRadius)
                ;
        }
    };

    struct PySBDeVaucouleurs 
    {

        static SBDeVaucouleurs* construct(
            const bp::object & scale_radius, const bp::object & half_light_radius,
            double trunc, double flux, bool flux_untruncated,
            boost::shared_ptr<GSParams> gsparams)
        {
            double s = 1.0;
            checkRadii(half_light_radius, scale_radius, bp::object());
            SBSersic::RadiusType rType = SBSersic::HALF_LIGHT_RADIUS;
            if (half_light_radius.ptr() != Py_None) {
                s = bp::extract<double>(half_light_radius);
            }
            if (scale_radius.ptr() != Py_None) {
                s = bp::extract<double>(scale_radius);
                rType = SBSersic::SCALE_RADIUS;
            }
            return new SBDeVaucouleurs(s, rType, flux, trunc, flux_untruncated, gsparams);
        }

        static void wrap() 
        {
            bp::class_<SBDeVaucouleurs,bp::bases<SBSersic> >("SBDeVaucouleurs",bp::no_init)
                .def("__init__",
                     bp::make_constructor(
                         &construct, bp::default_call_policies(),
                         (bp::arg("scale_radius")=bp::object(),
                          bp::arg("half_light_radius")=bp::object(),
                          bp::arg("trunc")=0.,
                          bp::arg("flux")=1., bp::arg("flux_untruncated")=false,
                          bp::arg("gsparams")=bp::object())
                     )
                )
                .def(bp::init<const SBDeVaucouleurs &>())
                .def("getHalfLightRadius", &SBDeVaucouleurs::getHalfLightRadius)
                .def("getScaleRadius", &SBDeVaucouleurs::getScaleRadius)
                ;
        }
    };

    void pyExportSBSersic() 
    {
        PySBSersic::wrap();
        PySBDeVaucouleurs::wrap();
    }

} // namespace galsim
