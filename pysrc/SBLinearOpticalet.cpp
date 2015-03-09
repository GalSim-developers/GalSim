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

#include "SBLinearOpticalet.h"

namespace bp = boost::python;

namespace galsim {

    struct PySBLinearOpticalet
    {

        static SBLinearOpticalet* construct(double r0, int n1, int m1, int n2, int m2,
                                            boost::shared_ptr<GSParams> gsparams)
        {
            return new SBLinearOpticalet(r0, n1, m1, n2, m2, gsparams);
        }

        static void wrap()
        {
            bp::class_<SBLinearOpticalet,bp::bases<SBProfile> >("SBLinearOpticalet",bp::no_init)
                .def("__init__",
                     bp::make_constructor(
                        &construct, bp::default_call_policies(),
                        (bp::arg("scale_radius"),
                         bp::arg("n1"),
                         bp::arg("m1"),
                         bp::arg("n2"),
                         bp::arg("m2"),
                         bp::arg("gsparams")=bp::object()))
                )
                .def(bp::init<const SBLinearOpticalet &>())
                .def("getScaleRadius", &SBLinearOpticalet::getScaleRadius)
                .def("getN1", &SBLinearOpticalet::getN1)
                .def("getM1", &SBLinearOpticalet::getM1)
                .def("getN1", &SBLinearOpticalet::getN2)
                .def("getM1", &SBLinearOpticalet::getM2)
                ;
        }
    };

    void pyExportSBLinearOpticalet()
    {
        PySBLinearOpticalet::wrap();
    }

} // namespace galsim
