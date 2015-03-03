/* -*- c++ -*-
 * Copyright (c) 2012-2014 by the GalSim developers team on GitHub
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
#ifndef __INTEL_COMPILER
#if defined(__GNUC__) && __GNUC__ >= 4 && (__GNUC__ >= 5 || __GNUC_MINOR__ >= 8)
#pragma GCC diagnostic ignored "-Wunused-local-typedefs"
#endif
#endif

#define BOOST_NO_CXX11_SMART_PTR
#include "boost/python.hpp"
#include "boost/python/stl_iterator.hpp"

#include "SBGaussian.h"
#include "RadiusHelper.h"

namespace bp = boost::python;

namespace galsim {

    struct PySBGaussian 
    {

        static SBGaussian* construct(
            const bp::object& half_light_radius, const bp::object& sigma, const bp::object& fwhm,
            double flux, boost::shared_ptr<GSParams> gsparams) 
        {
            double s = 1.0;
            checkRadii(half_light_radius, sigma, fwhm);
            if (half_light_radius.ptr() != Py_None) {
                s = bp::extract<double>(half_light_radius) * 0.84932180028801907; // (2\ln2)^(-1/2)
            }
            if (sigma.ptr() != Py_None) {
                s = bp::extract<double>(sigma);
            }
            if (fwhm.ptr() != Py_None) {
                s = bp::extract<double>(fwhm) * 0.42466090014400953; // 1 / (2(2\ln2)^(1/2))
            }
            return new SBGaussian(s, flux, gsparams);
        }

        static void wrap() 
        {
            bp::class_<SBGaussian,bp::bases<SBProfile> > pySBGaussian(
                "SBGaussian",
                "SBGaussian(flux=1., half_light_radius=None, sigma=None, fwhm=None)\n\n"
                "Construct an exponential profile with the given flux and half-light radius,\n"
                "sigma, or FWHM.  Exactly one radius must be provided.\n",
                bp::no_init);
            pySBGaussian
                .def("__init__", bp::make_constructor(
                        &construct, bp::default_call_policies(),
                        (bp::arg("half_light_radius")=bp::object(), bp::arg("sigma")=bp::object(), 
                         bp::arg("fwhm")=bp::object(), bp::arg("flux")=1.,
                         bp::arg("gsparams")=bp::object())
                ))
                .def(bp::init<const SBGaussian &>())
                .def("getSigma", &SBGaussian::getSigma)
                ;
        }
    };

    void pyExportSBGaussian() 
    {
        PySBGaussian::wrap();
    }

} // namespace galsim
