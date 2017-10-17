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
#include "boost/python.hpp" // header that includes Python.h always needs to come first

#include "Image.h"

namespace bp = boost::python;

// Note that docstrings are now added in galsim/image.py
namespace galsim {

    template <typename T>
    static ImageView<T>* MakeFromArray(size_t idata, int step, int stride,
                                       const Bounds<int>& bounds)
    {
        T* data = reinterpret_cast<T*>(idata);
        shared_ptr<T> owner;
        return new ImageView<T>(data, owner, step, stride, bounds);
    }

    template <typename T>
    static void WrapImage(const std::string& suffix)
    {
        bp::class_< BaseImage<T>, boost::noncopyable >(("BaseImage" + suffix).c_str(), bp::no_init);

        typedef ImageView<T>* (*Make_func)(size_t, int, int, const Bounds<int>&);
        bp::class_< ImageView<T>, bp::bases< BaseImage<T> > >(("ImageView" + suffix).c_str(),
                                                              bp::no_init)
            .def("__init__", bp::make_constructor((Make_func)&MakeFromArray,
                                                  bp::default_call_policies()));

        typedef void (*rfft_func_type)(const BaseImage<T>&, ImageView<std::complex<double> >,
                                       bool, bool);
        typedef void (*irfft_func_type)(const BaseImage<T>&, ImageView<double>, bool, bool);
        typedef void (*cfft_func_type)(const BaseImage<T>&, ImageView<std::complex<double> >,
                                       bool, bool, bool);
        bp::def("rfft", rfft_func_type(&rfft));
        bp::def("irfft", irfft_func_type(&irfft));
        bp::def("cfft", cfft_func_type(&cfft));

        typedef void (*wrap_func_type)(ImageView<T>, const Bounds<int>&, bool, bool);
        bp::def("wrapImage", wrap_func_type(&wrapImage));

        typedef void (*invert_func_type)(ImageView<T>);
        bp::def("invertImage", invert_func_type(&invertImage));
    }

    void pyExportImage()
    {
        WrapImage<uint16_t>("US");
        WrapImage<uint32_t>("UI");
        WrapImage<int16_t>("S");
        WrapImage<int32_t>("I");
        WrapImage<float>("F");
        WrapImage<double>("D");
        WrapImage<std::complex<double> >("CD");
        WrapImage<std::complex<float> >("CF");

        bp::def("goodFFTSize", &goodFFTSize);
    }

} // namespace galsim
