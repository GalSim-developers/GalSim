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

#include "SBMoffatlet.h"

namespace bp = boost::python;

namespace galsim {

    struct PySBMoffatlet
    {

        static SBMoffatlet* construct(double beta, double r0, int j, int q,
                                      boost::shared_ptr<GSParams> gsparams)
        {
            return new SBMoffatlet(beta, r0, j, q, gsparams);
        }

        static void wrap()
        {
            bp::class_<SBMoffatlet,bp::bases<SBProfile> >("SBMoffatlet",bp::no_init)
                .def("__init__",
                     bp::make_constructor(
                        &construct, bp::default_call_policies(),
                        (bp::arg("beta"),
                         bp::arg("r0"),
                         bp::arg("j"),
                         bp::arg("q"),
                         bp::arg("gsparams")=bp::object()))
                )
                .def(bp::init<const SBMoffatlet &>())
                .def("getBeta", &SBMoffatlet::getBeta)
                .def("getScaleRadius", &SBMoffatlet::getScaleRadius)
                .def("getJ", &SBMoffatlet::getJ)
                .def("getQ", &SBMoffatlet::getQ)
                ;
        }
    };

    void pyExportSBMoffatlet()
    {
        PySBMoffatlet::wrap();
    }

} // namespace galsim
