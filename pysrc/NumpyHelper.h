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

#define PY_ARRAY_UNIQUE_SYMBOL GALSIM_ARRAY_API
#define NO_IMPORT_ARRAY
#include "numpy/arrayobject.h"

namespace bp = boost::python;

namespace galsim {

template <typename T> struct NumPyTraits;
template <> struct NumPyTraits<int16_t> { static int getCode() { return NPY_INT16; } };
template <> struct NumPyTraits<int32_t> { static int getCode() { return NPY_INT32; } };
//template <> struct NumPyTraits<int64_t> { static int getCode() { return NPY_INT64; } };
template <> struct NumPyTraits<float> { static int getCode() { return NPY_FLOAT32; } };
template <> struct NumPyTraits<double> { static int getCode() { return NPY_FLOAT64; } };

static int Normalize(int code) 
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
static bp::object GetNumPyType() 
{
    bp::handle<> h(reinterpret_cast<PyObject*>(PyArray_DescrFromType(NumPyTraits<T>::getCode())));
    return bp::object(h).attr("type");
}

template <typename T>
static void DestroyCObjectOwner(T* p) 
{
    boost::shared_ptr<T> * owner = reinterpret_cast< boost::shared_ptr<T> *>(p);
    delete owner;
}

template <typename T>
struct PythonDeleter {
    void operator()(T * p) { owner.reset(); }
    explicit PythonDeleter(PyObject * o) : owner(bp::borrowed(o)) {}
    bp::handle<> owner;
};

template <typename T>
static bp::object MakeNumpyArray(
    const T* data, int n1, int n2, int stride, bool isConst,
    boost::shared_ptr<T> owner = boost::shared_ptr<T>())
{
    // --- Create array ---
    int flags = NPY_ALIGNED;
    if (!isConst) flags |= NPY_WRITEABLE;
    npy_intp shape[2] = { n1, n2 };
    npy_intp strides[2] = { stride * int(sizeof(T)), int(sizeof(T)) };
    bp::object result(
        bp::handle<>(
            PyArray_New(
                &PyArray_Type, 2, shape, NumPyTraits<T>::getCode(), strides,
                const_cast<T*>(data), sizeof(T), flags, NULL
            )
        )
    );

    // --- Manage ownership ---
    PythonDeleter<T>* pyDeleter = boost::get_deleter<PythonDeleter<T> >(owner);
    bp::handle<> pyOwner;
    if (pyDeleter) {
        // If memory was original allocated by Python, we use that Python object as the owner...
        pyOwner = pyDeleter->owner;
    } else {
        // ..if not, we put a shared_ptr in an opaque Python object.
        pyOwner = bp::handle<>(
            PyCObject_FromVoidPtr(new boost::shared_ptr<T>(owner), &DestroyCObjectOwner)
        );
    }
    reinterpret_cast<PyArrayObject*>(result.ptr())->base = pyOwner.release();

    return result;
}

template <typename T>
static bp::object MakeNumpyArray(
    const T* data, int n1, int stride, bool isConst,
    boost::shared_ptr<T> owner = boost::shared_ptr<T>())
{
    // --- Create array ---
    int flags = NPY_ALIGNED;
    if (!isConst) flags |= NPY_WRITEABLE;
    npy_intp shape[1] = { n1 };
    npy_intp strides[1] = { stride * int(sizeof(T)) };
    bp::object result(
        bp::handle<>(
            PyArray_New(
                &PyArray_Type, 1, shape, NumPyTraits<T>::getCode(), strides,
                const_cast<T*>(data), sizeof(T), flags, NULL
            )
        )
    );

    // --- Manage ownership ---
    PythonDeleter<T>* pyDeleter = boost::get_deleter<PythonDeleter<T> >(owner);
    bp::handle<> pyOwner;
    if (pyDeleter) {
        // If memory was original allocated by Python, we use that Python object as the owner...
        pyOwner = pyDeleter->owner;
    } else {
        // ..if not, we put a shared_ptr in an opaque Python object.
        pyOwner = bp::handle<>(
            PyCObject_FromVoidPtr(new boost::shared_ptr<T>(owner), &DestroyCObjectOwner)
        );
    }
    reinterpret_cast<PyArrayObject*>(result.ptr())->base = pyOwner.release();

    return result;
}

// Check the type of the numpy array, input as array.
// - It should be the same type as required for data (T).
// - It should have dimensions dim
// - It should be writeable if isConst=true
// - It should have unit stride on the rows if ndim == 2
// Also sets data, owner, stride to the appropriate values before returning.
template <typename T>
static void CheckNumpyArray(const bp::object& array, int ndim, bool isConst,
    T*& data, boost::shared_ptr<T>& owner, int& stride)
{
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
    if (PyArray_NDIM(array.ptr()) != ndim) {
        PyErr_SetString(PyExc_ValueError, "numpy.ndarray argument has must be 2-d");
        bp::throw_error_already_set();
    }
    if (!isConst && !(PyArray_FLAGS(array.ptr()) & NPY_WRITEABLE)) {
        PyErr_SetString(PyExc_TypeError, "numpy.ndarray argument must be writeable");
        bp::throw_error_already_set();
    }
    if (ndim == 2 && PyArray_STRIDE(array.ptr(), 1) != sizeof(T)) {
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
                PythonDeleter<T>(pyOwner)
            );
        } else {
            owner = boost::shared_ptr<T>(
                reinterpret_cast<T*>(PyArray_DATA(array.ptr())),
                PythonDeleter<T>(pyOwner)
            );
        }
    } else {
        owner = boost::shared_ptr<T>(
            reinterpret_cast<T*>(PyArray_DATA(array.ptr())),
            PythonDeleter<T>(array.ptr())
        );
    }
}

} // namespace galsim
