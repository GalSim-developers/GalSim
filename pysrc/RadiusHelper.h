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
#include "boost/python.hpp"

namespace bp = boost::python;

namespace galsim {

    // Used by multiple profile classes to ensure at most one radius is given.
    static void checkRadii(const bp::object & r1, const bp::object & r2, const bp::object & r3)
    {
        int nRad = (r1.ptr() != Py_None) + (r2.ptr() != Py_None) + (r3.ptr() != Py_None);
        if (nRad > 1) {
            PyErr_SetString(PyExc_TypeError, "Multiple radius parameters given");
            bp::throw_error_already_set();
        }
        if (nRad == 0) {
            PyErr_SetString(PyExc_TypeError, "No radius parameter given");
            bp::throw_error_already_set();
        }
    }
}
