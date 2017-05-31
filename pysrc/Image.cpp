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

#define BOOST_NO_CXX11_SMART_PTR
#include "boost/python.hpp" // header that includes Python.h always needs to come first

#include "NumpyHelper.h"
#include "Image.h"

namespace bp = boost::python;

#define ADD_CORNER(wrapper, getter, prop)                                      \
    do {                                                                \
        bp::object fget = bp::make_function(&BaseImage<T>::getter);   \
        wrapper.def(#getter, fget);                                \
        wrapper.add_property(#prop, fget);                   \
    } while (false)

// Note that docstrings are now added in galsim/image.py
namespace galsim {


template <typename T, typename U>
static ImageAlloc<T>* MakeFromImage(const BaseImage<U>& rhs)
{ return new ImageAlloc<T>(rhs); }

template <typename T, typename U, typename W>
static void doWrapImageAllocTemplates(W& wrapper) {
    typedef ImageAlloc<T>* (*constructFrom_func_type)(const BaseImage<U>&);
    typedef void (ImageAlloc<T>::* copyFrom_func_type)(const BaseImage<U>&);
    wrapper
        .def(
            "__init__",
            bp::make_constructor(
                constructFrom_func_type(&MakeFromImage),
                bp::default_call_policies(), bp::args("other")
            )
        )
        .def("copyFrom", copyFrom_func_type(&ImageAlloc<T>::copyFrom));
}

template <typename T, typename U, typename W>
static void doWrapImageViewTemplates(W& wrapper) {
    typedef void (ImageView<T>::* copyFrom_func_type)(const BaseImage<U>&);
    wrapper
        .def("copyFrom", copyFrom_func_type(&ImageView<T>::copyFrom));
}

template <typename T, typename U>
struct WrapHelper {
    // Normally do the above functions
    template <typename W>
    static void wrapImageAllocTemplates(W& wrapper)
    { doWrapImageAllocTemplates<T,U,W>(wrapper); }
    template <typename W>
    static void wrapImageViewTemplates(W& wrapper)
    { doWrapImageViewTemplates<T,U,W>(wrapper); }
};

// Overload complex -> real copies to do nothing.
template <typename T, typename U>
struct WrapHelper<T,std::complex<U> > {
    template <typename W>
    static void wrapImageAllocTemplates(W& wrapper) {}
    template <typename W>
    static void wrapImageViewTemplates(W& wrapper) {}
};

template <typename T, typename U>
struct WrapHelper<std::complex<T>, std::complex<U> > {
    // But now re-enable complex -> complex
    typedef std::complex<T> CT;
    typedef std::complex<U> CU;
    template <typename W>
    static void wrapImageAllocTemplates(W& wrapper)
    { doWrapImageAllocTemplates<CT,CU,W>(wrapper); }
    template <typename W>
    static void wrapImageViewTemplates(W& wrapper)
    { doWrapImageViewTemplates<CT,CU,W>(wrapper); }
};

template <typename T>
struct PyImage {

    // This one is mostly just used to enable a repr that actually gets back to the original.
    static ImageAlloc<T>* MakeAllocFromArray(const Bounds<int>& bounds, const bp::object& array)
    {
        int step = 0;
        int stride = 0;
        T* data = 0;
        boost::shared_ptr<T> owner;
        Bounds<int> bounds2;
        BuildConstructorArgs(array, bounds.getXMin(), bounds.getYMin(), false, data, owner,
                             step, stride, bounds2);
        ImageView<T>* imview = new ImageView<T>(data, owner, step, stride, bounds2);
        return new ImageAlloc<T>(*imview);
    }

    static bp::object GetArrayImpl(bp::object self, bool isConst)
    {
        // --- Try to get cached array ---
        // NB: self.attr("_array") != bp::object() no longer works, since it calls the numpy
        //     overload of _array != None, which compares term by term rather than checking
        //     if _array is not None.  Instead check if the attribute exists and is not None.
        if (PyObject_HasAttrString(self.ptr(), "_array")
            && bp::object(self.attr("_array")).ptr() != Py_None) {
            return self.attr("_array");
        }

        const BaseImage<T>& image = bp::extract<const BaseImage<T>&>(self);

        bp::object numpy_array = MakeNumpyArray(
            image.getData(),
            image.getYMax() - image.getYMin() + 1,
            image.getXMax() - image.getXMin() + 1,
            image.getStep(), image.getStride(), isConst, image.getOwner());

        self.attr("_array") = numpy_array;
        return numpy_array;
    }

    static void CallResize(bp::object self, const Bounds<int>& new_bounds)
    {
        // We need to make sure the _array attribute is deleted
        if (PyObject_HasAttrString(self.ptr(), "_array"))
            self.attr("_array") = bp::object();

        // Now call the regular resize method.
        ImageAlloc<T>& image = bp::extract<ImageAlloc<T>&>(self);
        image.resize(new_bounds);
    }

    static bp::object GetArray(bp::object image) { return GetArrayImpl(image, false); }
    static bp::object GetConstArray(bp::object image) { return GetArrayImpl(image, true); }

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

    static bp::object wrapImageAlloc(const std::string& suffix) {

        // Need some typedefs and explicit casts here to resolve overloads of methods
        // that have both const and non-const versions or (x,y) and pos version
        typedef const T& (ImageAlloc<T>::* at_func_type)(const int, const int) const;
        typedef const T& (ImageAlloc<T>::* at_pos_func_type)(const Position<int>&) const;
        typedef ImageView<T> (ImageAlloc<T>::* subImage_func_type)(const Bounds<int>&);
        typedef ImageView<T> (ImageAlloc<T>::* view_func_type)();

        bp::object at = bp::make_function(
            at_func_type(&ImageAlloc<T>::at),
            bp::return_value_policy<bp::copy_const_reference>(),
            bp::args("x", "y")
        );
        bp::object at_pos = bp::make_function(
            at_pos_func_type(&ImageAlloc<T>::at),
            bp::return_value_policy<bp::copy_const_reference>(),
            bp::args("pos")
        );
        bp::object getBounds = bp::make_function(
            &BaseImage<T>::getBounds,
            bp::return_value_policy<bp::copy_const_reference>()
        );

        bp::class_< BaseImage<T>, boost::noncopyable >
            pyBaseImage(("BaseImage" + suffix).c_str(), "", bp::no_init);
        pyBaseImage
            .def("subImage", &BaseImage<T>::subImage, bp::args("bounds"))
            .add_property("array", &GetConstArray)
            .def("getBounds", getBounds)
            .add_property("bounds", getBounds)
            .def("rfft", &BaseImage<T>::fft,
                 (bp::arg("shift_in")=true, bp::arg("shift_out")=true))
            .def("irfft", &BaseImage<T>::inverse_fft,
                 (bp::arg("shift_in")=true, bp::arg("shift_out")=true))
            .def("cfft", &BaseImage<T>::cfft,
                 (bp::arg("inverse")=false, bp::arg("shift_in")=true, bp::arg("shift_out")=true))
            ;
        ADD_CORNER(pyBaseImage, getXMin, xmin);
        ADD_CORNER(pyBaseImage, getYMin, ymin);
        ADD_CORNER(pyBaseImage, getXMax, xmax);
        ADD_CORNER(pyBaseImage, getYMax, ymax);

        bp::class_< ImageAlloc<T>, bp::bases< BaseImage<T> > >
            pyImageAlloc(("ImageAlloc" + suffix).c_str(), "", bp::no_init);
        pyImageAlloc
            .def(bp::init<>())
            .def(bp::init<int,int,T>(
                    (bp::args("ncol","nrow"), bp::arg("init_value")=T(0))
            ))
            .def(bp::init<const Bounds<int>&, T>(
                    (bp::arg("bounds"), bp::arg("init_value")=T(0))
            ))
            .def("__init__", bp::make_constructor(
                    &MakeAllocFromArray, bp::default_call_policies(),
                    (bp::arg("bounds"), bp::arg("array"))))
            .def("subImage", subImage_func_type(&ImageAlloc<T>::subImage), bp::args("bounds"))
            .def("wrap", &ImageAlloc<T>::wrap,
                 (bp::arg("bounds"), bp::arg("hermx")=false, bp::arg("hermy")=false))
            .def("view", view_func_type(&ImageAlloc<T>::view))
            .add_property("array", &GetArray)
            // In python, there is no way to have a function return a mutable reference
            // so you can't make im(x,y) = val work correctly.  Thus, the __call__
            // function (which is the im(x,y) syntax) is just the const version.
            .def("__call__", at) // always used checked accessors in Python
            .def("__call__", at_pos)
            .def("setValue", &ImageAlloc<T>::setValue, bp::args("x","y","value"))
            .def("fill", &ImageAlloc<T>::fill)
            .def("setZero", &ImageAlloc<T>::setZero)
            .def("invertSelf", &ImageAlloc<T>::invertSelf)
            .def("shift", &ImageAlloc<T>::shift, bp::args("delta"))
            .def("resize", &CallResize)
            .enable_pickling()
            ;
        WrapHelper<T,float>::wrapImageAllocTemplates(pyImageAlloc);
        WrapHelper<T,double>::wrapImageAllocTemplates(pyImageAlloc);
        WrapHelper<T,int16_t>::wrapImageAllocTemplates(pyImageAlloc);
        WrapHelper<T,int32_t>::wrapImageAllocTemplates(pyImageAlloc);
        WrapHelper<T,uint16_t>::wrapImageAllocTemplates(pyImageAlloc);
        WrapHelper<T,uint32_t>::wrapImageAllocTemplates(pyImageAlloc);
        WrapHelper<T,std::complex<float> >::wrapImageAllocTemplates(pyImageAlloc);
        WrapHelper<T,std::complex<double> >::wrapImageAllocTemplates(pyImageAlloc);

        return pyImageAlloc;
    }

    static bp::object wrapImageView(const std::string& suffix) {

        typedef T& (ImageView<T>::*at_func_type)(int, int);
        typedef T& (ImageView<T>::*at_pos_func_type)(const Position<int>&);

        bp::object at = bp::make_function(
            at_func_type(&ImageView<T>::at),
            bp::return_value_policy<bp::copy_non_const_reference>(),
            bp::args("x", "y")
        );
        bp::object at_pos = bp::make_function(
            at_pos_func_type(&ImageView<T>::at),
            bp::return_value_policy<bp::copy_non_const_reference>(),
            bp::args("pos")
        );
        bp::class_< ImageView<T>, bp::bases< BaseImage<T> > >
            pyImageView(("ImageView" + suffix).c_str(), "", bp::no_init);
        pyImageView
            .def("__init__", bp::make_constructor(
                    &MakeFromArray, bp::default_call_policies(),
                    (bp::arg("array"), bp::arg("xmin")=1, bp::arg("ymin")=1)))
            .def(bp::init<const ImageView<T>&>(bp::args("other")))
            .def("subImage", &ImageView<T>::subImage, bp::args("bounds"))
            .def("wrap", &ImageView<T>::wrap, bp::args("bounds"))
            .def("view", &ImageView<T>::view)
            .add_property("array", &GetArray)
            .def("__call__", at) // always used checked accessors in Python
            .def("__call__", at_pos)
            .def("setValue", &ImageView<T>::setValue, bp::args("x","y","value"))
            .def("fill", &ImageView<T>::fill)
            .def("setZero", &ImageView<T>::setZero)
            .def("invertSelf", &ImageView<T>::invertSelf)
            .def("shift", &ImageView<T>::shift, bp::args("delta"))
            .enable_pickling()
            ;
        WrapHelper<T,float>::wrapImageViewTemplates(pyImageView);
        WrapHelper<T,double>::wrapImageViewTemplates(pyImageView);
        WrapHelper<T,int16_t>::wrapImageViewTemplates(pyImageView);
        WrapHelper<T,int32_t>::wrapImageViewTemplates(pyImageView);
        WrapHelper<T,uint16_t>::wrapImageViewTemplates(pyImageView);
        WrapHelper<T,uint32_t>::wrapImageViewTemplates(pyImageView);
        WrapHelper<T,std::complex<float> >::wrapImageViewTemplates(pyImageView);
        WrapHelper<T,std::complex<double> >::wrapImageViewTemplates(pyImageView);

        return pyImageView;
    }

    static bp::object wrapConstImageView(const std::string& suffix) {
        typedef const T& (BaseImage<T>::*at_func_type)(int, int) const;
        typedef const T& (BaseImage<T>::*at_pos_func_type)(const Position<int>&) const;

        bp::object at = bp::make_function(
            at_func_type(&BaseImage<T>::at),
            bp::return_value_policy<bp::copy_const_reference>(),
            bp::args("x", "y")
        );
        bp::object at_pos = bp::make_function(
            at_pos_func_type(&BaseImage<T>::at),
            bp::return_value_policy<bp::copy_const_reference>(),
            bp::args("pos")
        );
        bp::class_< ConstImageView<T>, bp::bases< BaseImage<T> > >
            pyConstImageView(("ConstImageView" + suffix).c_str(), "", bp::no_init);
        pyConstImageView
            .def("__init__", bp::make_constructor(
                    &MakeConstFromArray, bp::default_call_policies(),
                    (bp::arg("array"), bp::arg("xmin")=1, bp::arg("ymin")=1)))
            .def(bp::init<const BaseImage<T>&>(bp::args("other")))
            .def("view", &ConstImageView<T>::view)
            .def("__call__", at) // always used checked accessors in Python
            .def("__call__", at_pos)
            .enable_pickling()
            ;

        return pyConstImageView;
    }

};

void pyExportImage() {
    bp::dict pyImageAllocDict;  // dict that lets us say "Image[numpy.float32]", etc.

    pyImageAllocDict[GetNumPyType<uint16_t>()] = PyImage<uint16_t>::wrapImageAlloc("US");
    pyImageAllocDict[GetNumPyType<uint32_t>()] = PyImage<uint32_t>::wrapImageAlloc("UI");
    pyImageAllocDict[GetNumPyType<int16_t>()] = PyImage<int16_t>::wrapImageAlloc("S");
    pyImageAllocDict[GetNumPyType<int32_t>()] = PyImage<int32_t>::wrapImageAlloc("I");
    pyImageAllocDict[GetNumPyType<float>()] = PyImage<float>::wrapImageAlloc("F");
    pyImageAllocDict[GetNumPyType<double>()] = PyImage<double>::wrapImageAlloc("D");
    pyImageAllocDict[GetNumPyType<std::complex<float> >()] =
        PyImage<std::complex<float> >::wrapImageAlloc("CF");
    pyImageAllocDict[GetNumPyType<std::complex<double> >()] =
        PyImage<std::complex<double> >::wrapImageAlloc("CD");

    bp::dict pyConstImageViewDict;

    pyConstImageViewDict[GetNumPyType<uint16_t>()] = PyImage<uint16_t>::wrapConstImageView("US");
    pyConstImageViewDict[GetNumPyType<uint32_t>()] = PyImage<uint32_t>::wrapConstImageView("UI");
    pyConstImageViewDict[GetNumPyType<int16_t>()] = PyImage<int16_t>::wrapConstImageView("S");
    pyConstImageViewDict[GetNumPyType<int32_t>()] = PyImage<int32_t>::wrapConstImageView("I");
    pyConstImageViewDict[GetNumPyType<float>()] = PyImage<float>::wrapConstImageView("F");
    pyConstImageViewDict[GetNumPyType<double>()] = PyImage<double>::wrapConstImageView("D");
    pyConstImageViewDict[GetNumPyType<std::complex<float> >()] =
        PyImage<std::complex<float> >::wrapConstImageView("CF");
    pyConstImageViewDict[GetNumPyType<std::complex<double> >()] =
        PyImage<std::complex<double> >::wrapConstImageView("CD");

    bp::dict pyImageViewDict;

    pyImageViewDict[GetNumPyType<uint16_t>()] = PyImage<uint16_t>::wrapImageView("US");
    pyImageViewDict[GetNumPyType<uint32_t>()] = PyImage<uint32_t>::wrapImageView("UI");
    pyImageViewDict[GetNumPyType<int16_t>()] = PyImage<int16_t>::wrapImageView("S");
    pyImageViewDict[GetNumPyType<int32_t>()] = PyImage<int32_t>::wrapImageView("I");
    pyImageViewDict[GetNumPyType<float>()] = PyImage<float>::wrapImageView("F");
    pyImageViewDict[GetNumPyType<double>()] = PyImage<double>::wrapImageView("D");
    pyImageViewDict[GetNumPyType<std::complex<float> >()] =
        PyImage<std::complex<float> >::wrapImageView("CF");
    pyImageViewDict[GetNumPyType<std::complex<double> >()] =
        PyImage<std::complex<double> >::wrapImageView("CD");

    bp::scope scope;  // a default constructed scope represents the module we're creating
    scope.attr("ImageAlloc") = pyImageAllocDict;
    scope.attr("ConstImageView") = pyConstImageViewDict;
    scope.attr("ImageView") = pyImageViewDict;

    bp::def("goodFFTSize", &goodFFTSize, (bp::arg("input_size")),
            "Round up to the next larger 2^n or 3x2^n.");
}

} // namespace galsim
