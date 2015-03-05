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
        static void wrap() 
        {
            bp::class_<SBGaussian,bp::bases<SBProfile> >("SBGaussian", bp::no_init)
                .def(bp::init<double,double,boost::shared_ptr<GSParams> >(
                        (bp::arg("sigma")=bp::object(), bp::arg("flux")=1.,
                         bp::arg("gsparams")=bp::object())
                ))
                .def(bp::init<const SBGaussian &>())
                .def("getSigma", &SBGaussian::getSigma)
                .enable_pickling()
                ;
        }
    };

    void pyExportSBGaussian() 
    {
        PySBGaussian::wrap();
    }

} // namespace galsim
