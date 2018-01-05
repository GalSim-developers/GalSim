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

#include "PyBind11Helper.h"
#include "SBConvolve.h"

namespace galsim {

#ifdef USE_BOOST
    static BP_CONSTRUCTOR(construct, SBConvolve,
                          const bp::object& iterable, bool real_space, GSParams gsparams)
    {
        bp::stl_input_iterator<SBProfile> iter(iterable), end;
        std::list<SBProfile> plist;
        for(; iter != end; ++iter) plist.push_back(*iter);
        PB11_PLACEMENT_NEW SBConvolve(plist, real_space, gsparams);
    }
#else
    static BP_CONSTRUCTOR(construct, SBConvolve,
                          const std::list<SBProfile>& plist, bool real_space, GSParams gsparams)
    {
        PB11_PLACEMENT_NEW SBConvolve(plist, real_space, gsparams);
    }
#endif

    void pyExportSBConvolve(PB11_MODULE& _galsim)
    {
        bp::class_<SBConvolve BP_BASES(SBProfile)>(GALSIM_COMMA "SBConvolve" BP_NOINIT)
            .def("__init__", BP_MAKE_CONSTRUCTOR( &construct));
        bp::class_<SBAutoConvolve BP_BASES(SBProfile)>(GALSIM_COMMA "SBAutoConvolve" BP_NOINIT)
            .def(bp::init<const SBProfile&, bool, GSParams>());
        bp::class_<SBAutoCorrelate BP_BASES(SBProfile)>(GALSIM_COMMA "SBAutoCorrelate" BP_NOINIT)
            .def(bp::init<const SBProfile&, bool, GSParams>());
    }

} // namespace galsim
