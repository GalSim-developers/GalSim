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

    struct PySBSersic 
    {

        static SBSersic * construct(
            double n,
            const bp::object & half_light_radius,
            double trunc,
            double flux,
            bool flux_untruncated
        ) {
            if (half_light_radius.ptr() == Py_None) {
                PyErr_SetString(PyExc_TypeError, "No radius parameter given");
                bp::throw_error_already_set();
            }
            return new SBSersic(n, bp::extract<double>(half_light_radius), trunc, flux,
                                flux_untruncated);
        }
        static void wrap() {
            bp::class_<SBSersic,bp::bases<SBProfile> >("SBSersic", bp::no_init)
                .def("__init__",
                     bp::make_constructor(
                         &construct, bp::default_call_policies(),
                         (bp::arg("n"), bp::arg("half_light_radius")=bp::object(),
                          bp::arg("trunc")=0., bp::arg("flux")=1., bp::arg("flux_untruncated")=1.)
                     )
                )
                .def(bp::init<const SBSersic &>())
                .def("getN", &SBSersic::getN)
                .def("getHalfLightRadius", &SBSersic::getHalfLightRadius)
                ;
        }
    };

    struct PySBDeVaucouleurs 
    {
        static SBDeVaucouleurs * construct(
            const bp::object & half_light_radius,
            double trunc,
            double flux,
            double flux_untruncated
        ) {
            if (half_light_radius.ptr() == Py_None) {
                PyErr_SetString(PyExc_TypeError, "No radius parameter given");
                bp::throw_error_already_set();
            }
            return new SBDeVaucouleurs(bp::extract<double>(half_light_radius), trunc, flux,
                                       flux_untruncated);
        }

        static void wrap() {
            bp::class_<SBDeVaucouleurs,bp::bases<SBSersic> >(
                "SBDeVaucouleurs",bp::no_init)
                .def("__init__",
                     bp::make_constructor(
                         &construct, bp::default_call_policies(),
                         (bp::arg("half_light_radius")=bp::object(),
                          bp::arg("trunc")=0., bp::arg("flux")=1., bp::arg("flux_untruncated")=1.)
                     )
                )
                .def(bp::init<const SBDeVaucouleurs &>())
                .def("getHalfLightRadius", &SBDeVaucouleurs::getHalfLightRadius)
                ;
        }
    };

    void pyExportSBSersic() 
    {
        PySBSersic::wrap();
        PySBDeVaucouleurs::wrap();
    }

} // namespace galsim
