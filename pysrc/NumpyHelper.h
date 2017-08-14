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
#ifndef NumpyHelper_H
#define NumpyHelper_H

#include <iostream>
#include "Python.h"
#include "capsulethunk.h" // cf. https://docs.python.org/3/howto/cporting.html#cobject-replaced-with-capsule

#include "boost/python.hpp"

#ifdef __INTEL_COMPILER
#pragma warning (disable : 47)
#endif

#include <stdint.h>

#ifdef __INTEL_COMPILER
#pragma warning (default : 47)
#endif

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#define PY_ARRAY_UNIQUE_SYMBOL GALSIM_ARRAY_API
#define NO_IMPORT_ARRAY
#include "numpy/arrayobject.h"

#if !defined(NPY_API_VERSION) || NPY_API_VERSION < 7
    #define NPY_OLD_API
    #define NPY_ARRAY_ALIGNED NPY_ALIGNED
    #define NPY_ARRAY_WRITEABLE NPY_WRITEABLE
    #define NPY_ARRAY_ENSURECOPY NPY_ENSURECOPY
#endif

namespace bp = boost::python;

namespace galsim {

template <typename T> struct NumPyTraits;
template <> struct NumPyTraits<uint16_t> { static int getCode() { return NPY_UINT16; } };
template <> struct NumPyTraits<uint32_t> { static int getCode() { return NPY_UINT32; } };
template <> struct NumPyTraits<int16_t> { static int getCode() { return NPY_INT16; } };
template <> struct NumPyTraits<int32_t> { static int getCode() { return NPY_INT32; } };
//template <> struct NumPyTraits<int64_t> { static int getCode() { return NPY_INT64; } };
template <> struct NumPyTraits<float> { static int getCode() { return NPY_FLOAT32; } };
template <> struct NumPyTraits<double> { static int getCode() { return NPY_FLOAT64; } };
template <> struct NumPyTraits<std::complex<float> >
{ static int getCode() { return NPY_COMPLEX64; } };
template <> struct NumPyTraits<std::complex<double> >
{ static int getCode() { return NPY_COMPLEX128; } };

inline int GetNumpyArrayTypeCode(PyObject* array)
{
    PyArrayObject* numpy_array = reinterpret_cast<PyArrayObject*>(array);
    int code = PyArray_TYPE(numpy_array);
    // Normally the return of PyArray_TYPE is a code that indicates what type the data is.  However,
    // this gets confusing for integer types, since different integer types may be equivalent.  In
    // particular int and long might be the same thing (typically on 32 bit machines, they can both
    // be 4 bytes).  For some reason in this case, PyArray_TYPE sometimes returns NPY_INT and
    // sometimes NPY_LONG.  So this function normalizes these answers to make sure that if they are
    // equivalent, we convert NPY_INT to the equivalent other type.
    if (sizeof(int) == sizeof(int16_t) && code == NPY_INT) return NPY_INT16;
    if (sizeof(int) == sizeof(int32_t) && code == NPY_INT) return NPY_INT32;
    if (sizeof(int) == sizeof(int64_t) && code == NPY_INT) return NPY_INT64;
    if (sizeof(int) == sizeof(uint16_t) && code == NPY_INT) return NPY_UINT16;
    if (sizeof(int) == sizeof(uint32_t) && code == NPY_INT) return NPY_UINT32;
    return code;
}

// return the NumPy type for a C++ class (e.g. float -> numpy.float32)
template <typename T>
inline bp::object GetNumPyType()
{
    bp::handle<> h(reinterpret_cast<PyObject*>(PyArray_DescrFromType(NumPyTraits<T>::getCode())));
    return bp::object(h).attr("type");
}

inline int GetNumpyArrayNDim(PyObject* array)
{
    PyArrayObject* numpy_array = reinterpret_cast<PyArrayObject*>(array);
    return PyArray_NDIM(numpy_array);
}

inline int GetNumpyArrayDim(PyObject* array, int i)
{
    PyArrayObject* numpy_array = reinterpret_cast<PyArrayObject*>(array);
    return PyArray_DIM(numpy_array,i);
}

inline int GetNumpyArrayFlags(PyObject* array)
{
    PyArrayObject* numpy_array = reinterpret_cast<PyArrayObject*>(array);
    return PyArray_FLAGS(numpy_array);
}

inline PyObject* GetNumpyArrayBase(PyObject* array)
{
    PyArrayObject* numpy_array = reinterpret_cast<PyArrayObject*>(array);
    return PyArray_BASE(numpy_array);
}

template <typename T>
inline int GetNumpyArrayStride(PyObject* array, int i)
{
    PyArrayObject* numpy_array = reinterpret_cast<PyArrayObject*>(array);
    return PyArray_STRIDE(numpy_array,i) / sizeof(T);
}

template <typename T>
inline T* GetNumpyArrayData(PyObject* array)
{
    PyArrayObject* numpy_array = reinterpret_cast<PyArrayObject*>(array);
    return reinterpret_cast<T*>(PyArray_DATA(numpy_array));
}

#if (PY_VERSION_HEX < 0x02070000)
template <typename T>
inline void DestroyCObjectOwner(T* p)
{
    boost::shared_ptr<T>* owner = reinterpret_cast<boost::shared_ptr<T>*>(p);
    delete owner;
}
#else
template <typename T>
inline void DestroyCapsule(PyObject* capsule)
{
    void* p = PyCapsule_GetPointer(capsule, NULL);
    boost::shared_ptr<T>* owner = reinterpret_cast<boost::shared_ptr<T>*>(p);
    delete owner;
}
#endif

template <typename T>
struct PythonDeleter {
    void operator()(T* p) { owner.reset(); }
    explicit PythonDeleter(PyObject* p) : owner(bp::borrowed(p)) {}
    bp::handle<> owner;
};

template <typename T>
static bp::object ManageNumpyArray(PyObject* array, boost::shared_ptr<T> owner)
{
    // --- Manage ownership ---
    PythonDeleter<T>* pyDeleter = boost::get_deleter<PythonDeleter<T> >(owner);
    // If memory was originally allocated by Python, we don't need to do anything here.
    // Just let the python Image class keep a pointer to the original numpy array.
    if (!pyDeleter) {
        // ..if not, we put a shared_ptr in an opaque Python object.
        boost::shared_ptr<T>* sp = new boost::shared_ptr<T>(owner);
#if (PY_VERSION_HEX < 0x02070000)
        PyObject* pyOwner = PyCapsule_New(sp, NULL, &DestroyCObjectOwner);
#else
        PyObject* pyOwner = PyCapsule_New(sp, NULL, &DestroyCapsule<T>);
#endif

#ifdef NPY_OLD_API
        reinterpret_cast<PyArrayObject*>(array)->base = pyOwner;
#else
        PyArray_SetBaseObject(reinterpret_cast<PyArrayObject*>(array),pyOwner);
#endif
    }

    return bp::object(bp::handle<>(array));
}

template <typename T>
static bp::object MakeNumpyArray(
    const T* data, int n1, int n2, int step, int stride, bool isConst,
    boost::shared_ptr<T> owner = boost::shared_ptr<T>())
{
    // --- Create array ---
    int flags = NPY_ARRAY_ALIGNED;
    if (!isConst) flags |= NPY_ARRAY_WRITEABLE;
    npy_intp shape[2] = { n1, n2 };
    npy_intp strides[2] = { stride * int(sizeof(T)), step * int(sizeof(T)) };
    PyObject* array = PyArray_New(
        &PyArray_Type, 2, shape, NumPyTraits<T>::getCode(), strides,
        const_cast<T*>(data), sizeof(T), flags, NULL);

    return ManageNumpyArray(array, owner);
}

template <typename T>
static bp::object MakeNumpyArray(
    const T* data, int n1, int stride, bool isConst,
    boost::shared_ptr<T> owner = boost::shared_ptr<T>())
{
    // --- Create array ---
    int flags = NPY_ARRAY_ALIGNED;
    if (!isConst) flags |= NPY_ARRAY_WRITEABLE;
    npy_intp shape[1] = { n1 };
    npy_intp strides[1] = { stride * int(sizeof(T)) };
    PyObject* array = PyArray_New(
        &PyArray_Type, 1, shape, NumPyTraits<T>::getCode(), strides,
        const_cast<T*>(data), sizeof(T), flags, NULL);

    return ManageNumpyArray(array, owner);
}

// Check the type of the numpy array, input as array.
// - It should be the same type as required for data (T).
// - It should have dimensions dim
// - It should be writeable if isConst=false
// Also sets data, owner, step, stride to the appropriate values before returning.
template <typename T>
static void CheckNumpyArray(const bp::object& array, int ndim, bool isConst,
    T*& data, boost::shared_ptr<T>& owner, int& step, int& stride)
{
    if (!PyArray_Check(array.ptr())) {
        PyErr_SetString(PyExc_TypeError, "numpy.ndarray argument required");
        bp::throw_error_already_set();
    }
    int actualType = GetNumpyArrayTypeCode(array.ptr());
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
        oss<<"  NPY_UINT16   = "<<NPY_UINT16<<"\n";
        oss<<"  NPY_UINT32   = "<<NPY_UINT32<<"\n";
        oss<<"  NPY_FLOAT   = "<<NPY_FLOAT<<"\n";
        oss<<"  NPY_DOUBLE  = "<<NPY_DOUBLE<<"\n";
        oss<<"  sizeof(int16_t) = "<<sizeof(int16_t)<<"\n";
        oss<<"  sizeof(int32_t) = "<<sizeof(int32_t)<<"\n";
        oss<<"  sizeof(int64_t) = "<<sizeof(int64_t)<<"\n";
        oss<<"  sizeof(uint16_t) = "<<sizeof(uint16_t)<<"\n";
        oss<<"  sizeof(uint32_t) = "<<sizeof(uint32_t)<<"\n";
        oss<<"  sizeof(short) = "<<sizeof(short)<<"\n";
        oss<<"  sizeof(int) = "<<sizeof(int)<<"\n";
        oss<<"  sizeof(long) = "<<sizeof(long)<<"\n";
        oss<<"  sizeof(npy_int16) = "<<sizeof(npy_int16)<<"\n";
        oss<<"  sizeof(npy_int32) = "<<sizeof(npy_int32)<<"\n";
        oss<<"  sizeof(npy_int64) = "<<sizeof(npy_int64)<<"\n";
        oss<<"  sizeof(npy_uint16) = "<<sizeof(npy_uint16)<<"\n";
        oss<<"  sizeof(npy_uint32) = "<<sizeof(npy_uint32)<<"\n";
        PyErr_SetString(PyExc_ValueError, oss.str().c_str());
        bp::throw_error_already_set();
    }
    if (GetNumpyArrayNDim(array.ptr()) != ndim) {
        std::ostringstream oss;
        oss<<"numpy.ndarray argument must be "<<ndim<<"-d"<<"\n";
        PyErr_SetString(PyExc_ValueError, oss.str().c_str());
        bp::throw_error_already_set();
    }
    if (!isConst && !(GetNumpyArrayFlags(array.ptr()) & NPY_ARRAY_WRITEABLE)) {
        PyErr_SetString(PyExc_TypeError, "numpy.ndarray argument must be writeable");
        bp::throw_error_already_set();
    }
    if (ndim == 2)
        step = GetNumpyArrayStride<T>(array.ptr(), 1);
    else
        step = 1;
    stride = GetNumpyArrayStride<T>(array.ptr(), 0);
    data = GetNumpyArrayData<T>(array.ptr());
    PyObject* pyOwner = GetNumpyArrayBase(array.ptr());
    if (pyOwner == NULL) pyOwner = array.ptr();
    if (PyArray_Check(pyOwner) && GetNumpyArrayTypeCode(pyOwner) == requiredType) {
        // Not really important, but we try to use the full array for
        // the owner pointer if this is a subarray, just to be consistent
        // with how it works for subimages.
        // The deleter is really all that matters.
        owner = boost::shared_ptr<T>(GetNumpyArrayData<T>(pyOwner),
                                     PythonDeleter<T>(pyOwner));
    } else {
        owner = boost::shared_ptr<T>(GetNumpyArrayData<T>(array.ptr()),
                                     PythonDeleter<T>(pyOwner));
    }
}

} // namespace galsim

#endif
