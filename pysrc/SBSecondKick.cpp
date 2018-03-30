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

#include "SBSecondKick.h"

namespace bp = boost::python;

namespace galsim {

    struct PySBSecondKick
    {
        static void wrap()
        {
            double (SBSecondKick::*kv)(double) const = &SBSecondKick::kValue;
            double (SBSecondKick::*xv)(double) const = &SBSecondKick::xValue;

            bp::class_<SBSecondKick,bp::bases<SBProfile> >("SBSecondKick", bp::no_init)
                .def(bp::init<double,double,double,double,double,double,double,double,
                              boost::shared_ptr<GSParams> >(
                        (bp::arg("lam"), bp::arg("r0"),
                         bp::arg("diam"), bp::arg("obscuration"),
                         bp::arg("L0"), bp::arg("kcrit"), bp::arg("flux")=1.,
                         bp::arg("scale")=1.0, bp::arg("gsparams")=bp::object()))
                )
                .def(bp::init<const SBSecondKick &>())
                .def("getLam", &SBSecondKick::getLam)
                .def("getR0", &SBSecondKick::getR0)
                .def("getDiam", &SBSecondKick::getDiam)
                .def("getObscuration", &SBSecondKick::getObscuration)
                .def("getL0", &SBSecondKick::getL0)
                .def("getKCrit", &SBSecondKick::getKCrit)
                .def("getScale", &SBSecondKick::getScale)
                .def("getDelta", &SBSecondKick::getDelta)
                .def("structureFunction", &SBSecondKick::structureFunction)
                .def("kValueRaw", &SBSecondKick::kValueRaw)
                .def("xValueRaw", &SBSecondKick::xValueRaw)
                .def("kValueDouble", kv)
                .def("xValueDouble", xv)
                .def("xValueExact", &SBSecondKick::xValueExact)
                .enable_pickling()
                ;
        }
    };

    void pyExportSBSecondKick()
    {
        PySBSecondKick::wrap();
    }

} // namespace galsim
