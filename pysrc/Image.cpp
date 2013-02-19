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

#ifdef __INTEL_COMPILER
#pragma warning (disable : 47)
#endif

#include <stdint.h>

#ifdef __INTEL_COMPILER
#pragma warning (default : 47)
#endif

#define PY_ARRAY_UNIQUE_SYMBOL SBPROFILE_ARRAY_API
#define NO_IMPORT_ARRAY
#include "numpy/arrayobject.h"

#include "Image.h"

namespace bp = boost::python;

#define ADD_CORNER(wrapper, getter, prop)                                      \
    do {                                                                \
        bp::object fget = bp::make_function(&BaseImage<T>::getter);   \
        wrapper.def(#getter, fget);                                \
        wrapper.add_property(#prop, fget);                   \
    } while (false)

namespace galsim {
namespace {

template <typename T> struct NumPyTraits;
template <> struct NumPyTraits<int16_t> { static int getCode() { return NPY_INT16; } };
template <> struct NumPyTraits<int32_t> { static int getCode() { return NPY_INT32; } };
//template <> struct NumPyTraits<int64_t> { static int getCode() { return NPY_INT64; } };
template <> struct NumPyTraits<float> { static int getCode() { return NPY_FLOAT32; } };
template <> struct NumPyTraits<double> { static int getCode() { return NPY_FLOAT64; } };

int Normalize(int code) 
{
    // Normally the return of PyArray_TYPE is a code that indicates what 
    // type the data is.  However, this gets confusing for integer types, since
    // different integer types may be equivalent.  In particular int and long
    // might be the same thing (typically on 32 bit machines, they can both
    // be 4 bytes).  For some reason in this case, PyArray_TYPE sometimes returns
    // NPY_INT and sometimes NPY_LONG.  So this function normalizes these answers
    // to make sure that if they are equivalent, we convert NPY_INT to the
    // equivalent other type.
    if (sizeof(int) == sizeof(int16_t) && code == NPY_INT) return NPY_INT16;
    if (sizeof(int) == sizeof(int32_t) && code == NPY_INT) return NPY_INT32;
    if (sizeof(int) == sizeof(int64_t) && code == NPY_INT) return NPY_INT64;
    return code;
}

// return the NumPy type for a C++ class (e.g. float -> numpy.float32)
template <typename T>
bp::object getNumPyType() {
    bp::handle<> h(reinterpret_cast<PyObject*>(PyArray_DescrFromType(NumPyTraits<T>::getCode())));
    return bp::object(h).attr("type");
}

template <typename T>
struct PyImage {

    static void destroyCObjectOwner(void * p) {
        boost::shared_ptr<T> * owner = reinterpret_cast< boost::shared_ptr<T> *>(p);
        delete owner;
    }

    struct PythonDeleter {
        void operator()(T * p) { owner.reset(); }

        explicit PythonDeleter(PyObject * o) : owner(bp::borrowed(o)) {}

        bp::handle<> owner;
    };

    static bp::object getArrayImpl(bp::object self, bool isConst) {

        // --- Try to get cached array ---
        if (PyObject_HasAttrString(self.ptr(), "_array")) return self.attr("_array");

        BaseImage<T> const & image = bp::extract<BaseImage<T> const &>(self);

        // --- Create array ---
        int flags = NPY_ALIGNED;
        if (!isConst) flags |= NPY_WRITEABLE;
        npy_intp shape[2] = {
            image.getYMax() - image.getYMin() + 1,
            image.getXMax() - image.getXMin() + 1
        };
        npy_intp strides[2];
        strides[0] = image.getStride() * sizeof(T);
        strides[1] = sizeof(T);
        bp::object result(
            bp::handle<>(
                PyArray_New(
                    &PyArray_Type, 2, shape, NumPyTraits<T>::getCode(), strides,
                    const_cast<T*>(image.getData()), sizeof(T), flags, NULL
                )
            )
        );

        // --- Manage ownership ---
        boost::shared_ptr<T> owner = image.getOwner();
        PythonDeleter * pyDeleter = boost::get_deleter<PythonDeleter>(owner);
        bp::handle<> pyOwner;
        if (pyDeleter) {
            // If memory was original allocated by Python, we use that Python object as the owner...
            pyOwner = pyDeleter->owner;
        } else {
            // ..if not, we put a shared_ptr in an opaque Python object.
            pyOwner = bp::handle<>(
                PyCObject_FromVoidPtr(new boost::shared_ptr<T>(owner), &destroyCObjectOwner)
            );
        }
        reinterpret_cast<PyArrayObject*>(result.ptr())->base = pyOwner.release();

        self.attr("_array") = result;
        return result;
    }

    static bp::object getArray(bp::object image) { return getArrayImpl(image, false); }
    static bp::object getConstArray(bp::object image) { return getArrayImpl(image, true); }

    static void buildConstructorArgs(
        bp::object const & array, int xmin, int ymin, bool isConst,
        T * & data, boost::shared_ptr<T> & owner, int & stride, Bounds<int> & bounds
    ) {
        if (!PyArray_Check(array.ptr())) {
            PyErr_SetString(PyExc_TypeError, "numpy.ndarray argument required");
            bp::throw_error_already_set();
        }
        int actualType = Normalize(PyArray_TYPE(array.ptr()));
        int requiredType = NumPyTraits<T>::getCode();
        if (actualType != requiredType) {
            std::ostringstream oss;
            oss<<"numpy.ndarray argument has incorrect data type\n";
            oss<<"T = "<<typeid(T).name()<<"\n";
            oss<<"actualType = "<<actualType<<"\n";
            oss<<"requiredType = "<<requiredType<<"\n";
            oss<<"For reference: \n";
            oss<<"  NPY_SHORT   = "<<NPY_SHORT<<"\n";
            oss<<"  NPY_INT     = "<<NPY_INT<<"\n";
            oss<<"  NPY_LONG    = "<<NPY_LONG<<"\n";
            oss<<"  NPY_INT16   = "<<NPY_INT16<<"\n";
            oss<<"  NPY_INT32   = "<<NPY_INT32<<"\n";
            oss<<"  NPY_INT64   = "<<NPY_INT64<<"\n";
            oss<<"  NPY_FLOAT   = "<<NPY_FLOAT<<"\n";
            oss<<"  NPY_DOUBLE  = "<<NPY_DOUBLE<<"\n";
            oss<<"  sizeof(int16_t) = "<<sizeof(int16_t)<<"\n";
            oss<<"  sizeof(int32_t) = "<<sizeof(int32_t)<<"\n";
            oss<<"  sizeof(int64_t) = "<<sizeof(int64_t)<<"\n";
            oss<<"  sizeof(short) = "<<sizeof(short)<<"\n";
            oss<<"  sizeof(int) = "<<sizeof(int)<<"\n";
            oss<<"  sizeof(long) = "<<sizeof(long)<<"\n";
            oss<<"  sizeof(npy_int16) = "<<sizeof(npy_int16)<<"\n";
            oss<<"  sizeof(npy_int32) = "<<sizeof(npy_int32)<<"\n";
            oss<<"  sizeof(npy_int64) = "<<sizeof(npy_int64)<<"\n";
            PyErr_SetString(PyExc_ValueError, oss.str().c_str());
            bp::throw_error_already_set();
        }
        if (PyArray_NDIM(array.ptr()) != 2) {
            PyErr_SetString(PyExc_ValueError, "numpy.ndarray argument has must be 2-d");
            bp::throw_error_already_set();
        }
        if (!isConst && !(PyArray_FLAGS(array.ptr()) & NPY_WRITEABLE)) {
            PyErr_SetString(PyExc_TypeError, "numpy.ndarray argument must be writeable");
            bp::throw_error_already_set();
        }
        if (PyArray_STRIDE(array.ptr(), 1) != sizeof(T)) {
            PyErr_SetString(PyExc_ValueError, "numpy.ndarray argument must have contiguous rows");
            bp::throw_error_already_set();
        }
        stride = PyArray_STRIDE(array.ptr(), 0) / sizeof(T);
        data = reinterpret_cast<T*>(PyArray_DATA(array.ptr()));
        PyObject * pyOwner = PyArray_BASE(array.ptr());
        if (pyOwner) {
            if (PyArray_Check(pyOwner) && PyArray_TYPE(pyOwner) == requiredType) {
                // Not really important, but we try to use the full array for 
                // the owner pointer if this is a subarray, just to be consistent
                // with how it works for subimages.
                // The deleter is really all that matters.
                owner = boost::shared_ptr<T>(
                    reinterpret_cast<T*>(PyArray_DATA(pyOwner)),
                    PythonDeleter(pyOwner)
                );
            } else {
                owner = boost::shared_ptr<T>(
                    reinterpret_cast<T*>(PyArray_DATA(array.ptr())),
                    PythonDeleter(pyOwner)
                );
            }
        } else {
            owner = boost::shared_ptr<T>(
                reinterpret_cast<T*>(PyArray_DATA(array.ptr())),
                PythonDeleter(array.ptr())
            );
        }
        bounds = Bounds<int>(
            xmin, xmin + PyArray_DIM(array.ptr(), 1) - 1,
            ymin, ymin + PyArray_DIM(array.ptr(), 0) - 1
        );
    }

    static ImageView<T>* makeFromArray(
        const bp::object& array, int xmin, int ymin, double scale) 
    {
        Bounds<int> bounds;
        int stride = 0;
        T * data = 0;
        boost::shared_ptr<T> owner;
        buildConstructorArgs(array, xmin, ymin, false, data, owner, stride, bounds);
        return new ImageView<T>(data, owner, stride, bounds, scale);
    }

    static ConstImageView<T>* makeConstFromArray(
        const bp::object& array, int xmin, int ymin, double scale) 
    {
        Bounds<int> bounds;
        int stride = 0;
        T * data = 0;
        boost::shared_ptr<T> owner;
        buildConstructorArgs(array, xmin, ymin, true, data, owner, stride, bounds);
        return new ConstImageView<T>(data, owner, stride, bounds, scale);
    }

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

    static bp::object wrapImage(std::string const & suffix) {
        
        // Note that docstrings are now added in galsim/image.py

        bp::object getScale = bp::make_function(&BaseImage<T>::getScale);
        bp::object setScale = bp::make_function(&BaseImage<T>::setScale);

        // Need some typedefs and explicit casts here to resolve overloads of methods
        // that have both const and non-const versions:
        typedef const T& (Image<T>::* at_func_type)(const int, const int) const;
        typedef ImageView<T> (Image<T>::* subImage_func_type)(const Bounds<int>&);
        typedef ImageView<T> (Image<T>::* view_func_type)();

        bp::object at = bp::make_function(
            at_func_type(&Image<T>::at),
            bp::return_value_policy<bp::copy_const_reference>(),
            bp::args("x", "y")
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
            .add_property("array", &getConstArray)
            .def("shift", &BaseImage<T>::shift, bp::args("dx", "dy"))
            .def("setOrigin", &BaseImage<T>::setOrigin, bp::args("x0", "y0"))
            .def("setCenter", &BaseImage<T>::setCenter, bp::args("x0", "y0"))
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
            .def(bp::init<const Bounds<int> &, T>(
                    (bp::arg("bounds")=Bounds<int>(), bp::arg("init_value")=T(0))
            ))
            .def(bp::init<BaseImage<T> const &>(bp::args("other")))
            .def("subImage", subImage_func_type(&Image<T>::subImage), bp::args("bounds"))
            .def("view", view_func_type(&Image<T>::view))
            //.def("assign", &Image<T>::operator=, bp::return_self<>())
            .add_property("array", &getArray)
            // In python, there is no way to have a function return a mutable reference
            // so you can't make im(x,y) = val work correctly.  Thus, the __call__
            // funtion (which is the im(x,y) syntax) is just the const version.
            .def("__call__", at) // always used checked accessors in Python
            .def("at", at)
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

    static bp::object wrapImageView(std::string const & suffix) {
        
        // Note that docstrings are now added in galsim/image.py

        bp::object at = bp::make_function(
            &ImageView<T>::at,
            bp::return_value_policy<bp::copy_non_const_reference>(),
            bp::args("x", "y")
        );
        bp::class_< ImageView<T>, bp::bases< BaseImage<T> > >
            pyImageView(("ImageView" + suffix).c_str(), "", bp::no_init);
        pyImageView
            .def(
                "__init__",
                bp::make_constructor(
                    makeFromArray, bp::default_call_policies(),
                    (bp::arg("array"), bp::arg("xmin")=1, bp::arg("ymin")=1, 
                     bp::arg("scale")=1.0)
                )
            )
            .def(bp::init<ImageView<T> const &>(bp::args("other")))
            .def("subImage", &ImageView<T>::subImage, bp::args("bounds"))
            .def("view", &ImageView<T>::view)
            //.def("assign", &ImageView<T>::operator=, bp::return_self<>())
            .add_property("array", &getArray)
            .def("__call__", at) // always used checked accessors in Python
            .def("at", at)
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

    static bp::object wrapConstImageView(std::string const & suffix) {
        
        // Note that docstrings are now added in galsim/image.py

        bp::object at = bp::make_function(
            &BaseImage<T>::at,
            bp::return_value_policy<bp::copy_const_reference>(),
            bp::args("x", "y")
        );
        bp::class_< ConstImageView<T>, bp::bases< BaseImage<T> > >
            pyConstImageView(("ConstImageView" + suffix).c_str(), "", bp::no_init);
        pyConstImageView
            .def(
                "__init__",
                bp::make_constructor(
                    makeConstFromArray, bp::default_call_policies(),
                    (bp::arg("array"), bp::arg("xmin")=1, bp::arg("ymin")=1,
                     bp::arg("scale")=1.0)
                )
            )
            .def(bp::init<BaseImage<T> const &>(bp::args("other")))
            .def("view", &ConstImageView<T>::view)
            .def("__call__", at) // always used checked accessors in Python
            .def("at", at)
            .enable_pickling()
            ;

        return pyConstImageView;
    }

};

} // anonymous

void pyExportImage() {
    bp::dict pyImageDict;  // dict that lets us say "Image[numpy.float32]", etc.

    pyImageDict[getNumPyType<int16_t>()] = PyImage<int16_t>::wrapImage("S");
    pyImageDict[getNumPyType<int32_t>()] = PyImage<int32_t>::wrapImage("I");
    pyImageDict[getNumPyType<float>()] = PyImage<float>::wrapImage("F");
    pyImageDict[getNumPyType<double>()] = PyImage<double>::wrapImage("D");

    bp::dict pyConstImageViewDict; 

    pyConstImageViewDict[getNumPyType<int16_t>()] = PyImage<int16_t>::wrapConstImageView("S");
    pyConstImageViewDict[getNumPyType<int32_t>()] = PyImage<int32_t>::wrapConstImageView("I");
    pyConstImageViewDict[getNumPyType<float>()] = PyImage<float>::wrapConstImageView("F");
    pyConstImageViewDict[getNumPyType<double>()] = PyImage<double>::wrapConstImageView("D");

    bp::dict pyImageViewDict;

    pyImageViewDict[getNumPyType<int16_t>()] = PyImage<int16_t>::wrapImageView("S");
    pyImageViewDict[getNumPyType<int32_t>()] = PyImage<int32_t>::wrapImageView("I");
    pyImageViewDict[getNumPyType<float>()] = PyImage<float>::wrapImageView("F");
    pyImageViewDict[getNumPyType<double>()] = PyImage<double>::wrapImageView("D");

    bp::scope scope;  // a default constructed scope represents the module we're creating
    scope.attr("Image") = pyImageDict;
    scope.attr("ConstImageView") = pyConstImageViewDict;
    scope.attr("ImageView") = pyImageViewDict;
}

} // namespace galsim
