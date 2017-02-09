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

#include "SBAiry.h"

namespace bp = boost::python;

namespace galsim {

    struct PySBAiry
    {
        static void wrap()
        {
            bp::class_<SBAiry,bp::bases<SBProfile> >("SBAiry", bp::no_init)
                .def(bp::init<double,double,double,boost::shared_ptr<GSParams> >(
                        (bp::arg("lam_over_diam"), bp::arg("obscuration")=0., bp::arg("flux")=1.,
                         bp::arg("gsparams")=bp::object())
                ))
                .def(bp::init<const SBAiry &>())
                .def("getLamOverD", &SBAiry::getLamOverD)
                .def("getObscuration", &SBAiry::getObscuration)
                .enable_pickling()
                ;
        }
    };

    void pyExportSBAiry()
    {
        PySBAiry::wrap();
    }

} // namespace galsim
