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

#include "Angle.h"
#include "SBInclinedSersic.h"
#include "RadiusHelper.h"

namespace bp = boost::python;

namespace galsim {

    struct PySBInclinedSersic
    {

        static SBInclinedSersic* construct(
            double n, Angle inclination, const bp::object & scale_radius, const bp::object & half_light_radius,
            const bp::object & scale_height, const bp::object & scale_h_over_r,
            double flux, double trunc, bool flux_untruncated,
            boost::shared_ptr<GSParams> gsparams)
        {
            // Get the scale or half-light-radius, as needed

            double s = 1.0;
            checkRadii(half_light_radius, scale_radius, bp::object());
            SBInclinedSersic::RadiusType rType = SBInclinedSersic::HALF_LIGHT_RADIUS;

            if (half_light_radius.ptr() != Py_None)
            {
                s = bp::extract<double>(half_light_radius);
            }
            if (scale_radius.ptr() != Py_None)
            {
                s = bp::extract<double>(scale_radius);
                rType = SBInclinedSersic::SCALE_RADIUS;
            }

            // Get the scale_height or scale_h_over_r, as needed

            double h = 1.0;
            SBInclinedSersic::HeightType hType = SBInclinedSersic::SCALE_H_OVER_R;

            if (scale_h_over_r.ptr() != Py_None)
            {
                h = bp::extract<double>(scale_h_over_r);
            }
            if (scale_height.ptr() != Py_None)
            {
                h = bp::extract<double>(scale_height);
                hType = SBInclinedSersic::SCALE_HEIGHT;
            }

            return new SBInclinedSersic(n, inclination, s, rType, h, hType, flux, trunc, flux_untruncated, gsparams);
        }

        static void wrap()
        {
            bp::class_<SBInclinedSersic,bp::bases<SBProfile> >("SBInclinedSersic", bp::no_init)
                .def("__init__",
                     bp::make_constructor(
                         &construct, bp::default_call_policies(),
                         (bp::arg("n"),
                          bp::arg("inclination"),
                          bp::arg("scale_radius")=bp::object(),
                          bp::arg("half_light_radius")=bp::object(),
                          bp::arg("scale_height")=bp::object(),
                          bp::arg("scale_h_over_r")=bp::object(),
                          bp::arg("flux")=1.,
                          bp::arg("trunc")=0., bp::arg("flux_untruncated")=false,
                          bp::arg("gsparams")=bp::object())
                     )
                )
                .def(bp::init<const SBInclinedSersic &>())
                .def("getN", &SBInclinedSersic::getN)
                .def("getHalfLightRadius", &SBInclinedSersic::getHalfLightRadius)
                .def("getInclination", &SBInclinedSersic::getInclination)
                .def("getScaleRadius", &SBInclinedSersic::getScaleRadius)
                .def("getScaleHeight", &SBInclinedSersic::getScaleHeight)
                .def("getTrunc", &SBInclinedSersic::getTrunc)
                .enable_pickling()
                ;
        }
    };

    void pyExportSBInclinedSersic()
    {
        PySBInclinedSersic::wrap();
    }

} // namespace galsim
