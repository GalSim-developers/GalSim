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

#include "SBTransform.h"

namespace bp = boost::python;

namespace galsim {

    struct PySBTransform
    {
        static void getJac(const SBTransform& self, size_t idata)
        {
            double* data = reinterpret_cast<double*>(idata);
            self.getJac(data[0], data[1], data[2], data[3]);
        }

        static void wrap()
        {
            static char const * doc =
                "SBTransform is an affine transformation of another SBProfile.\n"
                "Origin of original shape will now appear at x0.\n"
                "Flux is NOT conserved in transformation - SB is preserved."
                ;

            bp::class_< SBTransform, bp::bases<SBProfile> >("SBTransform", doc, bp::no_init)
                .def(bp::init<const SBProfile &, double, double, double, double,
                     Position<double>, double, GSParams>(
                         (bp::args("sbin", "mA", "mB", "mC", "mD"),
                          bp::arg("x0"), bp::arg("fluxScaling"), bp::arg("gsparams"))))
                .def(bp::init<const SBTransform &>())
                .def("getObj", &SBTransform::getObj)
                .def("getJac", getJac)
                .def("getOffset", &SBTransform::getOffset)
                .def("getFluxScaling", &SBTransform::getFluxScaling)
                .enable_pickling()
                ;
        }

    };

    void pyExportSBTransform()
    {
        PySBTransform::wrap();
    }

} // namespace galsim
