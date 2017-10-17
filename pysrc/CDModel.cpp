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

#include "CDModel.h"

namespace bp = boost::python;

namespace galsim {

    struct PyCDModels
    {

        template <typename U>
        static void wrapTemplates() {

            typedef void (*ApplyCD_func)(
                ImageView<U>& , const BaseImage<U>& ,
                const BaseImage<double>& , const BaseImage<double>& ,
                const BaseImage<double>& , const BaseImage<double>& ,
                const int , const double );
            bp::def("_ApplyCD", ApplyCD_func(&ApplyCD));

        };

        static void wrap(){
            wrapTemplates<float>();
            wrapTemplates<double>();
        }

    };

    void pyExportCDModel()
    {
        PyCDModels::wrap();
    }

} // namespace galsim
