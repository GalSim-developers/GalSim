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
    struct PyImage
    {
        static ImageView<T>* MakeFromArray(
            size_t idata, int step, int stride, const Bounds<int>& bounds)
        {
            T* data = reinterpret_cast<T*>(idata);
            shared_ptr<T> owner;
            return new ImageView<T>(data, owner, step, stride, bounds);
        }

        static ConstImageView<T>* MakeConstFromArray(
            size_t idata, int step, int stride, const Bounds<int>& bounds)
        {
            T* data = reinterpret_cast<T*>(idata);
            shared_ptr<T> owner;
            return new ConstImageView<T>(data, owner, step, stride, bounds);
        }

        static void wrapBaseImage(const std::string& suffix)
        {
            bp::class_< BaseImage<T>, boost::noncopyable >
                pyBaseImage(("BaseImage" + suffix).c_str(), "", bp::no_init);

            typedef void (*rfft_func_type)(const BaseImage<T>&, ImageView<std::complex<double> >,
                                           bool, bool);
            typedef void (*irfft_func_type)(const BaseImage<T>&, ImageView<double>, bool, bool);
            typedef void (*cfft_func_type)(const BaseImage<T>&, ImageView<std::complex<double> >,
                                           bool, bool, bool);

            bp::def("rfft", rfft_func_type(&rfft));
            bp::def("irfft", irfft_func_type(&irfft));
            bp::def("cfft", cfft_func_type(&cfft));
        }

        static bp::object wrapImageView(const std::string& suffix)
        {
            bp::class_< ImageView<T>, bp::bases< BaseImage<T> > >
                pyImageView(("ImageView" + suffix).c_str(), "", bp::no_init);
            pyImageView
                .def("__init__", bp::make_constructor(&MakeFromArray, bp::default_call_policies()));

            typedef void (*wrap_func_type)(ImageView<T>, const Bounds<int>&, bool, bool);
            bp::def("wrapImage", wrap_func_type(&wrapImage));
            typedef void (*invert_func_type)(ImageView<T>);
            bp::def("invertImage", invert_func_type(&invertImage));

            return pyImageView;
        }

        static bp::object wrapConstImageView(const std::string& suffix)
        {
            bp::class_< ConstImageView<T>, bp::bases< BaseImage<T> > >
                pyConstImageView(("ConstImageView" + suffix).c_str(), "", bp::no_init);
            pyConstImageView
                .def("__init__", bp::make_constructor(
                        &MakeConstFromArray, bp::default_call_policies()));

            return pyConstImageView;
        }
    };

    void pyExportImage()
    {
        PyImage<uint16_t>::wrapBaseImage("US");
        PyImage<uint32_t>::wrapBaseImage("UI");
        PyImage<int16_t>::wrapBaseImage("S");
        PyImage<int32_t>::wrapBaseImage("I");
        PyImage<float>::wrapBaseImage("F");
        PyImage<double>::wrapBaseImage("D");
        PyImage<std::complex<double> >::wrapBaseImage("CD");
        PyImage<std::complex<float> >::wrapBaseImage("CF");

        bp::dict pyConstImageViewDict;

        pyConstImageViewDict["US"] = PyImage<uint16_t>::wrapConstImageView("US");
        pyConstImageViewDict["UI"] = PyImage<uint32_t>::wrapConstImageView("UI");
        pyConstImageViewDict["S"] = PyImage<int16_t>::wrapConstImageView("S");
        pyConstImageViewDict["I"] = PyImage<int32_t>::wrapConstImageView("I");
        pyConstImageViewDict["F"] = PyImage<float>::wrapConstImageView("F");
        pyConstImageViewDict["D"] = PyImage<double>::wrapConstImageView("D");
        pyConstImageViewDict["CD"] = PyImage<std::complex<double> >::wrapConstImageView("CD");
        pyConstImageViewDict["CF"] = PyImage<std::complex<float> >::wrapConstImageView("CF");

        bp::dict pyImageViewDict;

        pyImageViewDict["US"] = PyImage<uint16_t>::wrapImageView("US");
        pyImageViewDict["UI"] = PyImage<uint32_t>::wrapImageView("UI");
        pyImageViewDict["S"] = PyImage<int16_t>::wrapImageView("S");
        pyImageViewDict["I"] = PyImage<int32_t>::wrapImageView("I");
        pyImageViewDict["F"] = PyImage<float>::wrapImageView("F");
        pyImageViewDict["D"] = PyImage<double>::wrapImageView("D");
        pyImageViewDict["CD"] = PyImage<std::complex<double> >::wrapImageView("CD");
        pyImageViewDict["CF"] = PyImage<std::complex<float> >::wrapImageView("CF");

        bp::scope scope;  // a default constructed scope represents the module we're creating
        scope.attr("ConstImageView") = pyConstImageViewDict;
        scope.attr("ImageView") = pyImageViewDict;

        bp::def("goodFFTSize", &goodFFTSize);
    }

} // namespace galsim
