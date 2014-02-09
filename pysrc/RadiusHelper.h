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
