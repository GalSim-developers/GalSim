// -*- c++ -*-
/*
 * Copyright 2012, 2013 The GalSim developers:
 * https://github.com/GalSim-developers
 *
 * This file is part of GalSim: The modular galaxy image simulation toolkit.
 *
 * GalSim is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * GalSim is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with GalSim.  If not, see <http://www.gnu.org/licenses/>
 */

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

template <typename T>
struct PyImage {

    template <typename U, typename W>
    static void wrapImageTemplates(W& wrapper) {
        typedef void (Image<T>::* copyFrom_func_type)(const BaseImage<U>&);
        wrapper
            .def("copyFrom", copyFrom_func_type(&Image<T>::copyFrom));
    }

    template <typename U, typename W>
    static void wrapImageViewTemplates(W& wrapper) {
        typedef void (ImageView<T>::* copyFrom_func_type)(const BaseImage<U>&) const;
        wrapper
            .def("copyFrom", copyFrom_func_type(&ImageView<T>::copyFrom));
    }

    static bp::object GetArrayImpl(bp::object self, bool isConst) 
    {
        // --- Try to get cached array ---
        if (PyObject_HasAttrString(self.ptr(), "_array")) return self.attr("_array");

        const BaseImage<T>& image = bp::extract<const BaseImage<T>&>(self);

        bp::object numpy_array = MakeNumpyArray(
            image.getData(), 
            image.getYMax() - image.getYMin() + 1,
            image.getXMax() - image.getXMin() + 1,
            image.getStride(), isConst, image.getOwner());

        self.attr("_array") = numpy_array;
        return numpy_array;
    }

    static bp::object GetArray(bp::object image) { return GetArrayImpl(image, false); }
    static bp::object GetConstArray(bp::object image) { return GetArrayImpl(image, true); }

    static void BuildConstructorArgs(
        const bp::object& array, int xmin, int ymin, bool isConst,
        T*& data, boost::shared_ptr<T>& owner, int& stride, Bounds<int>& bounds) 
    {
        CheckNumpyArray(array,2,isConst,data,owner,stride);
        bounds = Bounds<int>(
            xmin, xmin + GetNumpyArrayDim(array.ptr(), 1) - 1,
            ymin, ymin + GetNumpyArrayDim(array.ptr(), 0) - 1
        );
    }

    static ImageView<T>* MakeFromArray(
        const bp::object& array, int xmin, int ymin, double scale) 
    {
        Bounds<int> bounds;
        int stride = 0;
        T* data = 0;
        boost::shared_ptr<T> owner;
        BuildConstructorArgs(array, xmin, ymin, false, data, owner, stride, bounds);
        return new ImageView<T>(data, owner, stride, bounds, scale);
    }

    static ConstImageView<T>* MakeConstFromArray(
        const bp::object& array, int xmin, int ymin, double scale) 
    {
        Bounds<int> bounds;
        int stride = 0;
        T* data = 0;
        boost::shared_ptr<T> owner;
        BuildConstructorArgs(array, xmin, ymin, true, data, owner, stride, bounds);
        return new ConstImageView<T>(data, owner, stride, bounds, scale);
    }

    static bp::object wrapImage(const std::string& suffix) {
        bp::object getScale = bp::make_function(&BaseImage<T>::getScale);
        bp::object setScale = bp::make_function(&BaseImage<T>::setScale);

        // Need some typedefs and explicit casts here to resolve overloads of methods
        // that have both const and non-const versions or (x,y) and pos version
        typedef const T& (Image<T>::* at_func_type)(const int, const int) const;
        typedef const T& (Image<T>::* at_pos_func_type)(const Position<int>&) const;
        typedef void (BaseImage<T>::* shift_func_type)(const int, const int);
        typedef void (BaseImage<T>::* shift_pos_func_type)(const Position<int>&);
        typedef void (BaseImage<T>::* setOrigin_func_type)(const int, const int);
        typedef void (BaseImage<T>::* setOrigin_pos_func_type)(const Position<int>&);
        typedef void (BaseImage<T>::* setCenter_func_type)(const int, const int);
        typedef void (BaseImage<T>::* setCenter_pos_func_type)(const Position<int>&);
        typedef ImageView<T> (Image<T>::* subImage_func_type)(const Bounds<int>&);
        typedef ImageView<T> (Image<T>::* view_func_type)();

        bp::object at = bp::make_function(
            at_func_type(&Image<T>::at),
            bp::return_value_policy<bp::copy_const_reference>(),
            bp::args("x", "y")
        );
        bp::object at_pos = bp::make_function(
            at_pos_func_type(&Image<T>::at),
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
            .def("getScale", getScale)
            .def("setScale", setScale)
            .add_property("scale", getScale, setScale)
            .def("subImage", &BaseImage<T>::subImage, bp::args("bounds"))
            .add_property("array", &GetConstArray)
            .def("shift", shift_func_type(&Image<T>::shift), bp::args("x", "y"))
            .def("shift", shift_pos_func_type(&Image<T>::shift), bp::args("pos"))
            .def("setCenter", setCenter_func_type(&Image<T>::setCenter), bp::args("x", "y"))
            .def("setCenter", setCenter_pos_func_type(&Image<T>::setCenter), bp::args("pos"))
            .def("setOrigin", setOrigin_func_type(&Image<T>::setOrigin), bp::args("x", "y"))
            .def("setOrigin", setOrigin_pos_func_type(&Image<T>::setOrigin), bp::args("pos"))
            .def("getBounds", getBounds)
            .def("getPaddedSize", &BaseImage<T>::getPaddedSize, bp::args("pad_factor"))
            .add_property("bounds", getBounds)
            ;
        ADD_CORNER(pyBaseImage, getXMin, xmin);
        ADD_CORNER(pyBaseImage, getYMin, ymin);
        ADD_CORNER(pyBaseImage, getXMax, xmax);
        ADD_CORNER(pyBaseImage, getYMax, ymax);

        bp::class_< Image<T>, bp::bases< BaseImage<T> > >
            pyImage(("Image" + suffix).c_str(), "", bp::no_init);
        pyImage
            .def(bp::init<int,int,T>(
                    (bp::args("ncol","nrow"), bp::arg("init_value")=T(0))
            ))
            .def(bp::init<const Bounds<int>&, T>(
                    (bp::arg("bounds")=Bounds<int>(), bp::arg("init_value")=T(0))
            ))
            .def(bp::init<const BaseImage<T>&>(bp::args("other")))
            .def("subImage", subImage_func_type(&Image<T>::subImage), bp::args("bounds"))
            .def("view", view_func_type(&Image<T>::view))
            .add_property("array", &GetArray)
            // In python, there is no way to have a function return a mutable reference
            // so you can't make im(x,y) = val work correctly.  Thus, the __call__
            // function (which is the im(x,y) syntax) is just the const version.
            .def("__call__", at) // always used checked accessors in Python
            .def("at", at)
            .def("__call__", at_pos)
            .def("at", at_pos)
            .def("setValue", &Image<T>::setValue, bp::args("x","y","value"))
            .def("fill", &Image<T>::fill)
            .def("setZero", &Image<T>::setZero)
            .def("invertSelf", &Image<T>::invertSelf)
            .def("resize", &Image<T>::resize)
            .enable_pickling()
            ;
        wrapImageTemplates<float>(pyImage);
        wrapImageTemplates<double>(pyImage);
        wrapImageTemplates<int16_t>(pyImage);
        wrapImageTemplates<int32_t>(pyImage);

        return pyImage;
    }

    static bp::object wrapImageView(const std::string& suffix) {

        typedef T& (ImageView<T>::*at_func_type)(int, int) const;
        typedef T& (ImageView<T>::*at_pos_func_type)(const Position<int>&) const;

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
            .def(
                "__init__",
                bp::make_constructor(
                    &MakeFromArray, bp::default_call_policies(),
                    (bp::arg("array"), bp::arg("xmin")=1, bp::arg("ymin")=1, 
                     bp::arg("scale")=1.0)
                )
            )
            .def(bp::init<const ImageView<T>&>(bp::args("other")))
            .def("subImage", &ImageView<T>::subImage, bp::args("bounds"))
            .def("view", &ImageView<T>::view)
            .add_property("array", &GetArray)
            .def("__call__", at) // always used checked accessors in Python
            .def("at", at)
            .def("__call__", at_pos)
            .def("at", at_pos)
            .def("setValue", &ImageView<T>::setValue, bp::args("x","y","value"))
            .def("fill", &ImageView<T>::fill)
            .def("setZero", &ImageView<T>::setZero)
            .def("invertSelf", &Image<T>::invertSelf)
            .enable_pickling()
            ;
        wrapImageViewTemplates<float>(pyImageView);
        wrapImageViewTemplates<double>(pyImageView);
        wrapImageViewTemplates<int16_t>(pyImageView);
        wrapImageViewTemplates<int32_t>(pyImageView);

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
            .def(
                "__init__",
                bp::make_constructor(
                    &MakeConstFromArray, bp::default_call_policies(),
                    (bp::arg("array"), bp::arg("xmin")=1, bp::arg("ymin")=1,
                     bp::arg("scale")=1.0)
                )
            )
            .def(bp::init<const BaseImage<T>&>(bp::args("other")))
            .def("view", &ConstImageView<T>::view)
            .def("__call__", at) // always used checked accessors in Python
            .def("at", at)
            .def("__call__", at_pos)
            .def("at", at_pos)
            .enable_pickling()
            ;

        return pyConstImageView;
    }

};

void pyExportImage() {
    bp::dict pyImageDict;  // dict that lets us say "Image[numpy.float32]", etc.

    pyImageDict[GetNumPyType<int16_t>()] = PyImage<int16_t>::wrapImage("S");
    pyImageDict[GetNumPyType<int32_t>()] = PyImage<int32_t>::wrapImage("I");
    pyImageDict[GetNumPyType<float>()] = PyImage<float>::wrapImage("F");
    pyImageDict[GetNumPyType<double>()] = PyImage<double>::wrapImage("D");

    bp::dict pyConstImageViewDict; 

    pyConstImageViewDict[GetNumPyType<int16_t>()] = PyImage<int16_t>::wrapConstImageView("S");
    pyConstImageViewDict[GetNumPyType<int32_t>()] = PyImage<int32_t>::wrapConstImageView("I");
    pyConstImageViewDict[GetNumPyType<float>()] = PyImage<float>::wrapConstImageView("F");
    pyConstImageViewDict[GetNumPyType<double>()] = PyImage<double>::wrapConstImageView("D");

    bp::dict pyImageViewDict;

    pyImageViewDict[GetNumPyType<int16_t>()] = PyImage<int16_t>::wrapImageView("S");
    pyImageViewDict[GetNumPyType<int32_t>()] = PyImage<int32_t>::wrapImageView("I");
    pyImageViewDict[GetNumPyType<float>()] = PyImage<float>::wrapImageView("F");
    pyImageViewDict[GetNumPyType<double>()] = PyImage<double>::wrapImageView("D");

    bp::scope scope;  // a default constructed scope represents the module we're creating
    scope.attr("Image") = pyImageDict;
    scope.attr("ConstImageView") = pyConstImageViewDict;
    scope.attr("ImageView") = pyImageViewDict;
}

} // namespace galsim
