/* -*- c++ -*-
 * Copyright (c) 2012-2021 by the GalSim developers team on GitHub
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

#include "galsim/IgnoreWarnings.h"

// Python.h has to be included beore anything else, since they make some idiotic choices in
// how they structure their #defines.  But they refuse to fix.
// cf. https://bugs.python.org/issue1045893),
#include "Python.h"

#ifdef USE_BOOST

#define BOOST_PYTHON_MAX_ARITY 22  // We have a function with 21 params in HSM.cpp
                                   // c.f. www.boost.org/libs/python/doc/v2/configuration.html

#define BOOST_NO_CXX11_SMART_PTR
#include <boost/python.hpp>
#include <boost/python/stl_iterator.hpp>
namespace py = boost::python;

// Boost Python and PyBind11 work fairly similarly.  There are a few differences though.
// In some cases, pybind11 simplified things, or changed how some things work.  So these
// macros allow us to write code that works for either boost python or pybind11.

// First some things where the boost equivalent of some pybind11 function is different:
#define PYBIND11_MODULE(x,y) BOOST_PYTHON_MODULE(x)
#define PY_MODULE py::scope
#define PY_CAST py::extract
#define PY_INIT(args...) "__init__", py::make_constructor(args, py::default_call_policies())
#define def_property_readonly add_property

// PyBind11 requires the module object to be written some places where boost python does not.
// Our module name is always _galsim, so where we would write _galsim. or _galsim, we write these
// instead so in boost python, the module name goes away.
#define GALSIM_DOT py::
#define GALSIM_COMMA

// Finally, there are somethings that are only needed for boost python.  These are not required
// at all for pybind11.
#define BP_SCOPE(x) py::scope x;
#define BP_NOINIT , py::no_init
#define BP_NONCOPYABLE , boost::noncopyable
#define BP_BASES(T) py::bases<T>

#else

#include <pybind11/pybind11.h>
#include <pybind11/operators.h>
#include <pybind11/stl.h>
#include <pybind11/complex.h>
namespace py = pybind11;

#define PY_MODULE py::module
#define PY_CAST py::cast
#define PY_INIT(args...) py::init(args)

#define GALSIM_DOT _galsim.
#define GALSIM_COMMA _galsim,

#define BP_SCOPE(x)
#define BP_NOINIT
#define BP_NONCOPYABLE
#define BP_BASES(T) T

#endif

#endif
