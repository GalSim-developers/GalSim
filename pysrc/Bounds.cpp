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
#include "Bounds.h"

namespace galsim {

    template <typename T>
    static void WrapPosition(PY_MODULE& _galsim, const std::string& suffix)
    {
        py::class_<Position<T> >(GALSIM_COMMA ("Position" + suffix).c_str() BP_NOINIT)
            .def(py::init<T,T>())
            .def_readonly("x", &Position<T>::x)
            .def_readonly("y", &Position<T>::y);
    }

    template <typename T>
    static void WrapBounds(PY_MODULE& _galsim, const std::string& suffix)
    {
        py::class_< Bounds<T> >(GALSIM_COMMA ("Bounds" + suffix).c_str() BP_NOINIT)
            .def(py::init<T,T,T,T>())
            .def_property_readonly("xmin", &Bounds<T>::getXMin)
            .def_property_readonly("xmax", &Bounds<T>::getXMax)
            .def_property_readonly("ymin", &Bounds<T>::getYMin)
            .def_property_readonly("ymax", &Bounds<T>::getYMax);
    }

    void pyExportBounds(PY_MODULE& _galsim)
    {
        WrapPosition<double>(_galsim, "D");
        WrapPosition<int>(_galsim, "I");
        WrapBounds<double>(_galsim, "D");
        WrapBounds<int>(_galsim, "I");
    }

} // namespace galsim
