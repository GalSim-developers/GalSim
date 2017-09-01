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
#include "boost/python.hpp"

#include "SBInclinedExponential.h"

namespace bp = boost::python;

namespace galsim {

    struct PySBInclinedExponential
    {

        static void wrap()
        {
            bp::class_<SBInclinedExponential,bp::bases<SBProfile> >(
                "SBInclinedExponential", bp::no_init)
                .def(bp::init<double,double,double,double, GSParams>(
                        (bp::arg("inclination"), bp::arg("scale_radius"), bp::arg("scale_height"),
                         bp::arg("flux"), bp::arg("gsparams"))))
                .def(bp::init<const SBInclinedExponential &>())
                .def("getInclination", &SBInclinedExponential::getInclination)
                .def("getScaleRadius", &SBInclinedExponential::getScaleRadius)
                .def("getScaleHeight", &SBInclinedExponential::getScaleHeight)
                .enable_pickling()
                ;
        }
    };

    void pyExportSBInclinedExponential()
    {
        PySBInclinedExponential::wrap();
    }

} // namespace galsim
