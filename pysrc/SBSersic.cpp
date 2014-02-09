/* -*- c++ -*-
 * Copyright 2012-2014 The GalSim developers:
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
#ifndef __INTEL_COMPILER
#if defined(__GNUC__) && __GNUC__ >= 4 && (__GNUC__ >= 5 || __GNUC_MINOR__ >= 8)
#pragma GCC diagnostic ignored "-Wunused-local-typedefs"
#endif
#endif

#define BOOST_NO_CXX11_SMART_PTR
#include "boost/python.hpp"
#include "boost/python/stl_iterator.hpp"

#include "SBSersic.h"
#include "RadiusHelper.h"

namespace bp = boost::python;

namespace galsim {

    struct PySBSersic 
    {

        static SBSersic* construct(
            double n, const bp::object & scale_radius, const bp::object & half_light_radius,
            double flux, double trunc, bool flux_untruncated,
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
                          bp::arg("flux")=1.,
                          bp::arg("trunc")=0., bp::arg("flux_untruncated")=false,
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
            double flux, double trunc, bool flux_untruncated,
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
                          bp::arg("flux")=1.,
                          bp::arg("trunc")=0., bp::arg("flux_untruncated")=false,
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
