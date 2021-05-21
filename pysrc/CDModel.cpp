/* -*- c++ -*-
 * Copyright (c) 2012-2021 by the GalSim developers team on GitHub
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
#include "CDModel.h"

namespace galsim {

    template <typename T>
    static void WrapTemplates(PY_MODULE& _galsim)
    {
        typedef void (*ApplyCD_func)(ImageView<T>& , const BaseImage<T>& ,
                                     const BaseImage<double>& , const BaseImage<double>& ,
                                     const BaseImage<double>& , const BaseImage<double>& ,
                                     const int , const double );
        GALSIM_DOT def("_ApplyCD", ApplyCD_func(&ApplyCD));
    }

    void pyExportCDModel(PY_MODULE& _galsim)
    {
        WrapTemplates<float>(_galsim);
        WrapTemplates<double>(_galsim);
    }

} // namespace galsim
