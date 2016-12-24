/* -*- c++ -*-
 * Copyright (c) 2012-2016 by the GalSim developers team on GitHub
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

#define BOOST_NO_CXX11_SMART_PTR
#include "boost/python.hpp" // header that includes Python.h always needs to come first

#include "NumpyHelper.h"
#include "Image.h"

namespace bp = boost::python;

// Note that docstrings are now added in galsim/image.py
namespace galsim {


template <typename T>
struct PyImage {

    static void BuildConstructorArgs(
        const bp::object& array, int xmin, int ymin, bool isConst,
        T*& data, boost::shared_ptr<T>& owner, int& step, int& stride, Bounds<int>& bounds)
    {
        CheckNumpyArray(array,2,isConst,data,owner,step,stride);
        bounds = Bounds<int>(
            xmin, xmin + GetNumpyArrayDim(array.ptr(), 1) - 1,
            ymin, ymin + GetNumpyArrayDim(array.ptr(), 0) - 1
        );
    }

    static ImageView<T>* MakeFromArray(
        const bp::object& array, int xmin, int ymin)
    {
        Bounds<int> bounds;
        int step = 0;
        int stride = 0;
        T* data = 0;
        boost::shared_ptr<T> owner;
        BuildConstructorArgs(array, xmin, ymin, false, data, owner, step, stride, bounds);
        return new ImageView<T>(data, owner, step, stride, bounds);
    }

    static ConstImageView<T>* MakeConstFromArray(
        const bp::object& array, int xmin, int ymin)
    {
        Bounds<int> bounds;
        int step = 0;
        int stride = 0;
        T* data = 0;
        boost::shared_ptr<T> owner;
        BuildConstructorArgs(array, xmin, ymin, true, data, owner, step, stride, bounds);
        return new ConstImageView<T>(data, owner, step, stride, bounds);
    }

    static bp::object wrapBaseImage(const std::string& suffix) {

        bp::object getBounds = bp::make_function(
            &BaseImage<T>::getBounds,
            bp::return_value_policy<bp::copy_const_reference>()
        );

        bp::class_< BaseImage<T>, boost::noncopyable >
            pyBaseImage(("BaseImage" + suffix).c_str(), "", bp::no_init);

        typedef void (*rfft_func_type)(const BaseImage<T>&, ImageView<std::complex<double> >,
                                       bool, bool);
        typedef void (*irfft_func_type)(const BaseImage<T>&, ImageView<double>, bool, bool);
        typedef void (*cfft_func_type)(const BaseImage<T>&, ImageView<std::complex<double> >,
                                       bool, bool, bool);

        bp::def("rfft", rfft_func_type(&rfft),
            (bp::arg("in"), bp::arg("out"), bp::arg("shift_in")=true, bp::arg("shift_out")=true));
        bp::def("irfft", irfft_func_type(&irfft),
            (bp::arg("in"), bp::arg("out"), bp::arg("shift_in")=true, bp::arg("shift_out")=true));
        bp::def("cfft", cfft_func_type(&cfft),
            (bp::arg("in"), bp::arg("out"), bp::arg("inverse")=false,
            bp::arg("shift_in")=true, bp::arg("shift_out")=true));

        return pyBaseImage;
    }

    static bp::object wrapImageView(const std::string& suffix) {

        typedef T& (ImageView<T>::*at_func_type)(int, int);
        typedef T& (ImageView<T>::*at_pos_func_type)(const Position<int>&);

        bp::class_< ImageView<T>, bp::bases< BaseImage<T> > >
            pyImageView(("ImageView" + suffix).c_str(), "", bp::no_init);
        pyImageView
            .def("__init__", bp::make_constructor(
                    &MakeFromArray, bp::default_call_policies(),
                    (bp::arg("array"), bp::arg("xmin")=1, bp::arg("ymin")=1)))
            ;

        typedef void (*wrap_func_type)(ImageView<T>, const Bounds<int>&, bool, bool);
        bp::def("wrapImage", wrap_func_type(&wrapImage),
            (bp::arg("im"), bp::arg("bounds"), bp::arg("hermx"), bp::arg("hermy")));
        typedef void (*invert_func_type)(ImageView<T>);
        bp::def("invertImage", invert_func_type(&invertImage));

        return pyImageView;
    }

    static bp::object wrapConstImageView(const std::string& suffix) {
        typedef const T& (BaseImage<T>::*at_func_type)(int, int) const;
        typedef const T& (BaseImage<T>::*at_pos_func_type)(const Position<int>&) const;

        bp::class_< ConstImageView<T>, bp::bases< BaseImage<T> > >
            pyConstImageView(("ConstImageView" + suffix).c_str(), "", bp::no_init);
        pyConstImageView
            .def("__init__", bp::make_constructor(
                    &MakeConstFromArray, bp::default_call_policies(),
                    (bp::arg("array"), bp::arg("xmin")=1, bp::arg("ymin")=1)))
            ;

        return pyConstImageView;
    }

};

void pyExportImage() {

    PyImage<uint16_t>::wrapBaseImage("US");
    PyImage<uint32_t>::wrapBaseImage("UI");
    PyImage<int16_t>::wrapBaseImage("S");
    PyImage<int32_t>::wrapBaseImage("I");
    PyImage<float>::wrapBaseImage("F");
    PyImage<double>::wrapBaseImage("D");
    PyImage<std::complex<double> >::wrapBaseImage("C");

    bp::dict pyConstImageViewDict;

    pyConstImageViewDict[GetNumPyType<uint16_t>()] = PyImage<uint16_t>::wrapConstImageView("US");
    pyConstImageViewDict[GetNumPyType<uint32_t>()] = PyImage<uint32_t>::wrapConstImageView("UI");
    pyConstImageViewDict[GetNumPyType<int16_t>()] = PyImage<int16_t>::wrapConstImageView("S");
    pyConstImageViewDict[GetNumPyType<int32_t>()] = PyImage<int32_t>::wrapConstImageView("I");
    pyConstImageViewDict[GetNumPyType<float>()] = PyImage<float>::wrapConstImageView("F");
    pyConstImageViewDict[GetNumPyType<double>()] = PyImage<double>::wrapConstImageView("D");
    pyConstImageViewDict[GetNumPyType<std::complex<double> >()] =
        PyImage<std::complex<double> >::wrapConstImageView("C");

    bp::dict pyImageViewDict;

    pyImageViewDict[GetNumPyType<uint16_t>()] = PyImage<uint16_t>::wrapImageView("US");
    pyImageViewDict[GetNumPyType<uint32_t>()] = PyImage<uint32_t>::wrapImageView("UI");
    pyImageViewDict[GetNumPyType<int16_t>()] = PyImage<int16_t>::wrapImageView("S");
    pyImageViewDict[GetNumPyType<int32_t>()] = PyImage<int32_t>::wrapImageView("I");
    pyImageViewDict[GetNumPyType<float>()] = PyImage<float>::wrapImageView("F");
    pyImageViewDict[GetNumPyType<double>()] = PyImage<double>::wrapImageView("D");
    pyImageViewDict[GetNumPyType<std::complex<double> >()] =
        PyImage<std::complex<double> >::wrapImageView("C");

    bp::scope scope;  // a default constructed scope represents the module we're creating
    scope.attr("ConstImageView") = pyConstImageViewDict;
    scope.attr("ImageView") = pyImageViewDict;

    bp::def("goodFFTSize", &goodFFTSize, (bp::arg("input_size")),
            "Round up to the next larger 2^n or 3x2^n.");
}

} // namespace galsim
