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
#include "boost/python/stl_iterator.hpp"

#include "SBMoffat.h"
#include "RadiusHelper.h"

namespace bp = boost::python;

namespace galsim {

    struct PySBMoffat
    {

        static SBMoffat* construct(
            double beta, const bp::object & scale_radius, const bp::object & half_light_radius,
            const bp::object & fwhm, double trunc, double flux,
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
                         (bp::arg("beta"), bp::arg("scale_radius")=bp::object(),
                          bp::arg("half_light_radius")=bp::object(), bp::arg("fwhm")=bp::object(),
                          bp::arg("trunc")=0., bp::arg("flux")=1.,
                          bp::arg("gsparams")=bp::object())
                     )
                )
                .def(bp::init<const SBMoffat &>())
                .def("getBeta", &SBMoffat::getBeta)
                .def("getScaleRadius", &SBMoffat::getScaleRadius)
                .def("getFWHM", &SBMoffat::getFWHM)
                .def("getHalfLightRadius", &SBMoffat::getHalfLightRadius)
                .def("getTrunc", &SBMoffat::getTrunc)
                .enable_pickling()
                ;
        }
    };

    void pyExportSBMoffat()
    {
        PySBMoffat::wrap();
    }

} // namespace galsim
