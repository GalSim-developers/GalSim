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

#ifndef __INTEL_COMPILER
#if defined(__GNUC__) && __GNUC__ >= 4 && (__GNUC__ >= 5 || __GNUC_MINOR__ >= 8)
#pragma GCC diagnostic ignored "-Wunused-local-typedefs"
#endif
#endif

#include <boost/python.hpp> // header that includes Python.h always needs to come first
#include <boost/python/stl_iterator.hpp>

#include "Table.h"

namespace bp = boost::python;

namespace galsim {
namespace {

    // We only export the Table<double,double>, so don't bother templatizing PyTable.
    struct PyTable {

        static Table<double,double>* makeTable(
            const bp::object& args, const bp::object& vals, const std::string& interp)
        {
            std::vector<double> vargs, vvals;
            try {
                bp::stl_input_iterator<double> args_it(args);
                bp::stl_input_iterator<double> end;
                vargs.insert(vargs.end(),args_it,end);
            } catch (std::exception& e) {
                PyErr_SetString(PyExc_ValueError, "Unable to convert args to C++ vector");
                bp::throw_error_already_set();
            }
            try {
                bp::stl_input_iterator<double> vals_it(vals);
                bp::stl_input_iterator<double> end;
                vvals.insert(vvals.end(),vals_it,end);
            } catch (std::exception& e) {
                PyErr_SetString(PyExc_ValueError, "Unable to convert vals to C++ vector");
                bp::throw_error_already_set();
            }
            if (vargs.size() != vvals.size()) {
                PyErr_SetString(PyExc_ValueError, "args and vals must be the same size");
                bp::throw_error_already_set();
            }

            Table<double,double>::interpolant i = Table<double,double>::linear;
            if (interp == "linear") i = Table<double,double>::linear;
            else if (interp == "spline") i = Table<double,double>::spline;
            else if (interp == "floor") i = Table<double,double>::floor;
            else if (interp == "ceil") i = Table<double,double>::ceil;
            else {
                PyErr_SetString(PyExc_ValueError, "Invalid interpolant");
                bp::throw_error_already_set();
            }

            return new Table<double,double>(vargs,vvals,i);
        }

        static bp::list convertGetArgs(const Table<double,double>& table)
        {
            const std::vector<Table<double,double>::Entry>& v = table.getV();
            bp::list l;
            for (size_t i=0; i!=v.size(); ++i) l.append(v[i].arg);
            return l;
        }

        static bp::list convertGetVals(const Table<double,double>& table)
        {
            const std::vector<Table<double,double>::Entry>& v = table.getV();
            bp::list l;
            for (size_t i=0; i!=v.size(); ++i) l.append(v[i].val);
            return l;
        }

        static std::string convertGetInterp(const Table<double,double>& table)
        {
            Table<double,double>::interpolant i = table.getInterp();
            switch (i) {
                case Table<double,double>::linear:
                     return std::string("linear");
                case Table<double,double>::spline:
                     return std::string("spline");
                case Table<double,double>::floor:
                     return std::string("floor");
                case Table<double,double>::ceil:
                     return std::string("ceil");
                default:
                     PyErr_SetString(PyExc_ValueError, "Invalid interpolant");
                     bp::throw_error_already_set();
            }
            // Shouldn't get here...
            return std::string("");
        }

        static void wrap() 
        {
            // docstrings are in galsim/table.py
            bp::class_<Table<double,double> > pyTable("_LookupTable", bp::no_init);
            pyTable
                .def("__init__",
                     bp::make_constructor(
                         &makeTable, bp::default_call_policies(),
                         (bp::arg("args"), bp::arg("vals"), bp::arg("interp"))
                     )
                )
                .def(bp::init<const Table<double,double> &>(bp::args("other")))
                .def("argMin", &Table<double,double>::argMin)
                .def("argMax", &Table<double,double>::argMax)

                // Use version that throws expection if out of bounds
                .def("__call__", &Table<double,double>::lookup) 

                .def("getArgs", &convertGetArgs)
                .def("getVals", &convertGetVals)
                .def("getInterp", &convertGetInterp)
                .enable_pickling()
                ;
        }

    };

} // anonymous

void pyExportTable() 
{
    PyTable::wrap();
}

} // namespace galsim
