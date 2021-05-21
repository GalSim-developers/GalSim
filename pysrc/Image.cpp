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
#include "Image.h"

// Note that docstrings are now added in galsim/image.py
namespace galsim {

    template <typename T>
    static ImageView<T>* MakeFromArray(
        size_t idata, int step, int stride, const Bounds<int>& bounds)
    {
        T* data = reinterpret_cast<T*>(idata);
        shared_ptr<T> owner;
        return new ImageView<T>(data, owner, step, stride, bounds);
    }

    template <typename T>
    static void WrapImage(PY_MODULE& _galsim, const std::string& suffix)
    {
        py::class_<BaseImage<T> BP_NONCOPYABLE>(
            GALSIM_COMMA ("BaseImage" + suffix).c_str() BP_NOINIT);

        typedef ImageView<T>* (*Make_func)(size_t, int, int, const Bounds<int>&);
        py::class_<ImageView<T>, BP_BASES(BaseImage<T>)>(
            GALSIM_COMMA ("ImageView" + suffix).c_str() BP_NOINIT)
            .def(PY_INIT((Make_func)&MakeFromArray));

        typedef void (*rfft_func_type)(const BaseImage<T>&, ImageView<std::complex<double> >,
                                       bool, bool);
        typedef void (*irfft_func_type)(const BaseImage<T>&, ImageView<double>, bool, bool);
        typedef void (*cfft_func_type)(const BaseImage<T>&, ImageView<std::complex<double> >,
                                       bool, bool, bool);
        GALSIM_DOT def("rfft", rfft_func_type(&rfft));
        GALSIM_DOT def("irfft", irfft_func_type(&irfft));
        GALSIM_DOT def("cfft", cfft_func_type(&cfft));

        typedef void (*wrap_func_type)(ImageView<T>, const Bounds<int>&, bool, bool);
        GALSIM_DOT def("wrapImage", wrap_func_type(&wrapImage));

        typedef void (*invert_func_type)(ImageView<T>);
        GALSIM_DOT def("invertImage", invert_func_type(&invertImage));
    }

    void pyExportImage(PY_MODULE& _galsim)
    {
        WrapImage<uint16_t>(_galsim, "US");
        WrapImage<uint32_t>(_galsim, "UI");
        WrapImage<int16_t>(_galsim, "S");
        WrapImage<int32_t>(_galsim, "I");
        WrapImage<float>(_galsim, "F");
        WrapImage<double>(_galsim, "D");
        WrapImage<std::complex<double> >(_galsim, "CD");
        WrapImage<std::complex<float> >(_galsim, "CF");

        GALSIM_DOT def("goodFFTSize", &goodFFTSize);
    }

} // namespace galsim
