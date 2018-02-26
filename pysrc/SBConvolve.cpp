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

#include "SBConvolve.h"

namespace bp = boost::python;

namespace galsim {

    static SBConvolve* construct(const bp::list& slist, bool real_space, GSParams gsparams)
    {
        std::list<SBProfile> plist;
        int n = len(slist);
        for(int i=0; i<n; ++i) {
            plist.push_back(bp::extract<const SBProfile&>(slist[i]));
        }
        return new SBConvolve(plist, real_space, gsparams);
    }

    void pyExportSBConvolve()
    {
        bp::class_< SBConvolve, bp::bases<SBProfile> >("SBConvolve", bp::no_init)
            .def("__init__", bp::make_constructor(&construct, bp::default_call_policies()));

        bp::class_< SBAutoConvolve, bp::bases<SBProfile> >("SBAutoConvolve", bp::no_init)
            .def(bp::init<const SBProfile&, bool, GSParams>());

        bp::class_< SBAutoCorrelate, bp::bases<SBProfile> >("SBAutoCorrelate", bp::no_init)
            .def(bp::init<const SBProfile&, bool, GSParams>());
    }

} // namespace galsim
