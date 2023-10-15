/* -*- c++ -*-
 * Copyright (c) 2012-2023 by the GalSim developers team on GitHub
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
        // Note: use ptrdiff_t for these to make sure the product that becomes
        // the maxptr offset or nElements doesn't overflow.
        ptrdiff_t ncol = bounds.getXMax() - bounds.getXMin() + 1;
        ptrdiff_t nrow = bounds.getYMax() - bounds.getYMin() + 1;
        const T* maxptr = data + (ncol-1) * step + (nrow-1) * stride + 1;
        shared_ptr<T> owner;
        return new ImageView<T>(data, maxptr, ncol*nrow, owner, step, stride, bounds);
    }

    template <typename T>
    static void DepixelizeImage(ImageView<T> im, size_t iunit_integrals, const int n)
    {
        const double* unit_integrals = reinterpret_cast<const double*>(iunit_integrals);
        im.depixelizeSelf(unit_integrals, n);
    }

    template <typename T>
    static void WrapImage(py::module& _galsim, const std::string& suffix)
    {
        py::class_<BaseImage<T> >(_galsim, ("BaseImage" + suffix).c_str());

        typedef ImageView<T>* (*Make_func)(size_t, int, int, const Bounds<int>&);
        py::class_<ImageView<T>, BaseImage<T> >(_galsim, ("ImageView" + suffix).c_str())
            .def(py::init((Make_func)&MakeFromArray));

        typedef void (*rfft_func_type)(const BaseImage<T>&, ImageView<std::complex<double> >,
                                       bool, bool);
        typedef void (*irfft_func_type)(const BaseImage<T>&, ImageView<double>, bool, bool);
        typedef void (*cfft_func_type)(const BaseImage<T>&, ImageView<std::complex<double> >,
                                       bool, bool, bool);
        _galsim.def("rfft", rfft_func_type(&rfft));
        _galsim.def("irfft", irfft_func_type(&irfft));
        _galsim.def("cfft", cfft_func_type(&cfft));

        typedef void (*wrap_func_type)(ImageView<T>, const Bounds<int>&, bool, bool);
        _galsim.def("wrapImage", wrap_func_type(&wrapImage));

        typedef void (*invert_func_type)(ImageView<T>);
        _galsim.def("invertImage", invert_func_type(&invertImage));

        typedef void (*depix_func_type)(ImageView<T>, size_t, const int);
        _galsim.def("depixelizeImage", depix_func_type(&DepixelizeImage));
    }

    void pyExportImage(py::module& _galsim)
    {
        WrapImage<uint16_t>(_galsim, "US");
        WrapImage<uint32_t>(_galsim, "UI");
        WrapImage<int16_t>(_galsim, "S");
        WrapImage<int32_t>(_galsim, "I");
        WrapImage<float>(_galsim, "F");
        WrapImage<double>(_galsim, "D");
        WrapImage<std::complex<double> >(_galsim, "CD");
        WrapImage<std::complex<float> >(_galsim, "CF");

        _galsim.def("goodFFTSize", &goodFFTSize);
        _galsim.def("ClearDepixelizeCache", &ClearDepixelizeCache);
    }

} // namespace galsim
