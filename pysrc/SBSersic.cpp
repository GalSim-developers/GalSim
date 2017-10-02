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

#include "SBSersic.h"

namespace bp = boost::python;

namespace galsim {

    struct PySBSersic
    {

        static void wrap()
        {
            bp::class_<SBSersic,bp::bases<SBProfile> >("SBSersic", bp::no_init)
                .def(bp::init<double,double,double,double, GSParams>(
                         (bp::arg("n"), bp::arg("scale_radius"), bp::arg("flux"),
                          bp::arg("trunc"), bp::arg("gsparams"))));

            bp::def("SersicTruncatedScale", &SersicTruncatedScale);
            bp::def("SersicIntegratedFlux", &SersicIntegratedFlux);
            bp::def("SersicHLR", &SersicHLR);
        }
    };

    void pyExportSBSersic()
    {
        PySBSersic::wrap();
    }

} // namespace galsim
