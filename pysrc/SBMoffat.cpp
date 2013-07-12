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
#ifndef __INTEL_COMPILER
#if defined(__GNUC__) && __GNUC__ >= 4 && (__GNUC__ >= 5 || __GNUC_MINOR__ >= 8)
#pragma GCC diagnostic ignored "-Wunused-local-typedefs"
#endif
#endif

#include "boost/python.hpp"
#include "boost/python/stl_iterator.hpp"

#include "SBMoffat.h"
#include "RadiusHelper.h"

namespace bp = boost::python;

namespace galsim {

    struct PySBMoffat 
    {

        static SBMoffat* construct(
            double beta, const bp::object & fwhm, const bp::object & scale_radius,
            const bp::object & half_light_radius, double trunc, double flux,
            boost::shared_ptr<GSParams> gsparams) 
        {
            double s = 1.0;
            checkRadii(half_light_radius, scale_radius, fwhm);
            SBMoffat::RadiusType rType = SBMoffat::FWHM;
            if (fwhm.ptr() != Py_None) {
                s = bp::extract<double>(fwhm);
            }
            if (scale_radius.ptr() != Py_None) {
                s = bp::extract<double>(scale_radius);
                rType = SBMoffat::SCALE_RADIUS;
            }
            if (half_light_radius.ptr() != Py_None) {
                s = bp::extract<double>(half_light_radius);
                rType = SBMoffat::HALF_LIGHT_RADIUS;
            }
            return new SBMoffat(beta, s, rType, trunc, flux, gsparams);
        }

        static void wrap() 
        {
            bp::class_<SBMoffat,bp::bases<SBProfile> >("SBMoffat", bp::no_init)
                .def("__init__", 
                     bp::make_constructor(
                         &construct, bp::default_call_policies(),
                         (bp::arg("beta"), bp::arg("fwhm")=bp::object(), 
                          bp::arg("scale_radius")=bp::object(),
                          bp::arg("half_light_radius")=bp::object(),
                          bp::arg("trunc")=0., bp::arg("flux")=1.,
                          bp::arg("gsparams")=bp::object())
                     )
                )
                .def(bp::init<const SBMoffat &>())
                .def("getBeta", &SBMoffat::getBeta)
                .def("getScaleRadius", &SBMoffat::getScaleRadius)
                .def("getFWHM", &SBMoffat::getFWHM)
                .def("getHalfLightRadius", &SBMoffat::getHalfLightRadius)
                ;
        }
    };

    void pyExportSBMoffat() 
    {
        PySBMoffat::wrap();
    }

} // namespace galsim
