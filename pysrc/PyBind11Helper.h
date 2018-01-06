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
#ifndef PyBind11Helper_H
#define PyBind11Helper_H

#ifdef USE_BOOST

#include "galsim/IgnoreWarnings.h"

#define BOOST_PYTHON_MAX_ARITY 22  // We have a function with 21 params in HSM.cpp
                                   // c.f. www.boost.org/libs/python/doc/v2/configuration.html

#define BOOST_NO_CXX11_SMART_PTR
#include <boost/python.hpp>
#include <boost/python/stl_iterator.hpp>
namespace bp = boost::python;

#define PB11_MAKE_MODULE(x) BOOST_PYTHON_MODULE(x)
#define PB11_START_MODULE(x)
#define PB11_END_MODULE(x)

#define TUPLE(args...) bp::tuple
#define MAKE_TUPLE bp::make_tuple

#define GALSIM_DOT bp::
#define GALSIM_COMMA
#define PB11_MODULE bp::scope
#define BP_HANDLE bp::handle<>
#define BP_THROW bp::throw_error_already_set()
#define BP_NOINIT , bp::no_init
#define ENABLE_PICKLING .enable_pickling()
#define PB11_CAST(x) x
#define BP_OTHER(T) bp::other<T>()
#define ADD_PROPERTY(name, func) add_property(name, func)
#define BP_REGISTER(T) bp::register_ptr_to_python< boost::shared_ptr<T> >()
#define BOOST_NONCOPYABLE , boost::noncopyable
#define BP_BASES(T) , bp::bases<T>
#define BP_MAKE_CONSTRUCTOR(args...) bp::make_constructor(args, bp::default_call_policies())
#define BP_CONSTRUCTOR(f,x,args...) x* f(args)
#define PB11_PLACEMENT_NEW return new
#define CAST bp::extract
#define BP_COPY_CONST_REFERENCE bp::return_value_policy<bp::copy_const_reference>()
#define def_property_readonly add_property

#else

#include <pybind11/pybind11.h>
#include <pybind11/operators.h>
#include <pybind11/stl.h>
#include <pybind11/complex.h>
namespace bp = pybind11;

#if PYBIND11_VERSION_MAJOR >= 3 || (PYBIND11_VERSION_MAJOR == 2 && PYBIND11_VERSION_MINOR >= 2)
    #define PB11_MAKE_MODULE(x) PYBIND11_MODULE(x,x)
    #define PB11_START_MODULE(x)
    #define PB11_END_MODULE(x)
#else
    #define PB11_MAKE_MODULE(x) PYBIND11_PLUGIN(x)
    #define PB11_START_MODULE(x) pybind11::module x(#x)
    #define PB11_END_MODULE(x) return x.ptr()
#endif

#define TUPLE(args...) std::tuple<args>
#define MAKE_TUPLE std::make_tuple

#define GALSIM_DOT _galsim.
#define GALSIM_COMMA _galsim,
#define PB11_MODULE pybind11::module
#define BP_HANDLE pybind11::handle
#define BP_THROW throw pybind11::error_already_set()
#define BP_NOINIT
#define ENABLE_PICKLING
#define PB11_CAST(x) pybind11::cast(x)
#define BP_OTHER(T) T()
#define ADD_PROPERTY(name, func) def_property_readonly(name, func)
#define BP_REGISTER(T)
#define BOOST_NONCOPYABLE
#define BP_BASES(T) , T
#define BP_MAKE_CONSTRUCTOR(args...) args
#define BP_CONSTRUCTOR(f,x,args...) void f(x& instance, args)
#define PB11_PLACEMENT_NEW new (&instance)
#define CAST pybind11::cast
#define BP_COPY_CONST_REFERENCE pybind11::return_value_policy::reference

#endif

#endif
