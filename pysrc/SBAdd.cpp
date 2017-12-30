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
#include "SBAdd.h"

namespace galsim {

#ifdef USE_BOOST
    static BP_CONSTRUCTOR(construct, SBAdd, const bp::object& iterable, GSParams gsparams)
    {
        bp::stl_input_iterator<SBProfile> iter(iterable), end;
        std::list<SBProfile> plist;
        for(; iter != end; ++iter) plist.push_back(*iter);
        PYBIND11_PLACEMENT_NEW SBAdd(plist, gsparams);
    }
#else
    static BP_CONSTRUCTOR(construct, SBAdd, const std::list<SBProfile>& plist, GSParams gsparams)
    {
        PYBIND11_PLACEMENT_NEW SBAdd(plist, gsparams);
    }
#endif

    void pyExportSBAdd(PYBIND11_MODULE& _galsim)
    {
        bp::class_<SBAdd BP_BASES(SBProfile)>(GALSIM_COMMA "SBAdd" BP_NOINIT)
            .def("__init__", BP_MAKE_CONSTRUCTOR(&construct));
    }

} // namespace galsim
