/* -*- c++ -*-
 * Copyright 2012-2014 The GalSim developers:
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

#ifndef __INTEL_COMPILER
#if defined(__GNUC__) && __GNUC__ >= 4 && (__GNUC__ >= 5 || __GNUC_MINOR__ >= 8)
#pragma GCC diagnostic ignored "-Wunused-local-typedefs"
#endif
#endif

#define BOOST_NO_CXX11_SMART_PTR
#include "boost/python.hpp"

namespace pyconv = boost::python::converter;

template <typename T>
class array_scalar_converter {
public:

  static PyTypeObject const * get_pytype() {
    // This implementation depends on the fact that get_builtin returns pointers to objects
    // NumPy has declared statically, and that the typeobj member also refers to a static
    // object.  That means we don't need to do any reference counting.
    // In fact, I'm somewhat concerned that increasing the reference count of any of these
    // might cause leaks, because I don't think Boost.Python ever decrements it, but it's
    // probably a moot point if everything is actually static.
    return reinterpret_cast<PyArray_Descr*>(dtype::get_builtin<T>().ptr())->typeobj;
  }

  static void * convertible(PyObject * obj) {
    if (obj->ob_type == get_pytype()) {
      return obj;
    } else { 
      return 0;
    }
  }

  static void convert(PyObject * obj, pyconv::rvalue_from_python_stage1_data* data) {
    void * storage = reinterpret_cast<pyconv::rvalue_from_python_storage<T>*>(data)->storage.bytes;
    // We assume std::complex is a "standard layout" here and elsewhere; not guaranteed by
    // C++03 standard, but true in every known implementation (and guaranteed by C++11).
    PyArray_ScalarAsCtype(obj, reinterpret_cast<T*>(storage));
    data->convertible = storage;
  }

  static void declare() {
    pyconv::registry::push_back(
      &convertible, &convert, python::type_id<T>()
#ifndef BOOST_PYTHON_NO_PY_SIGNATURES
      , &get_pytype
#endif
    );
  }

};