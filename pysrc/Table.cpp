/* -*- c++ -*-
 * Copyright (c) 2012-2015 by the GalSim developers team on GitHub
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

#ifndef __INTEL_COMPILER
#if defined(__GNUC__) && __GNUC__ >= 4 && (__GNUC__ >= 5 || __GNUC_MINOR__ >= 8)
#pragma GCC diagnostic ignored "-Wunused-local-typedefs"
#endif
#endif

#define BOOST_NO_CXX11_SMART_PTR
#include <boost/python.hpp> // header that includes Python.h always needs to come first
#include <boost/python/stl_iterator.hpp>

#include "Table.h"
#include "NumpyHelper.h"

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

        static void interpMany(const Table<double,double>& table,
                               const bp::object& args, const bp::object& vals)
        {
            const double* argvec = GetNumpyArrayData<double>(args.ptr());
            double* valvec = GetNumpyArrayData<double>(vals.ptr());
            int N = GetNumpyArrayDim(args.ptr(), 0);
            table.interpMany(argvec, valvec, N);
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
                .def("interpMany", &interpMany)
                .def("getArgs", &convertGetArgs)
                .def("getVals", &convertGetVals)
                .def("getInterp", &convertGetInterp)
                .enable_pickling()
                ;
        }

    }; // struct PyTable

    struct PyTable2D{
        // static Table2D<double,double>* makeTable2D(
        //     double x0, double y0, double dx, double dy, const bp::object& valarray,
        //     const std::string& interp)
        // {
        //     const int Nx = GetNumpyArrayDim(valarray.ptr(), 1);
        //     const int Ny = GetNumpyArrayDim(valarray.ptr(), 0);
        //     const double* vals = GetNumpyArrayData<double>(valarray.ptr());
        //     Table2D<double,double>::interpolant i = Table2D<double,double>::linear;
        //     if (interp == "linear") i = Table2D<double,double>::linear;
        //     else if (interp == "floor") i = Table2D<double,double>::floor;
        //     else if (interp == "ceil") i = Table2D<double,double>::ceil;
        //     else {
        //         PyErr_SetString(PyExc_ValueError, "Invalid interpolant");
        //         bp::throw_error_already_set();
        //     }
        //     return new Table2D<double,double>(x0, y0, dx, dy, Nx, Ny, vals, i);
        // }

        static Table2D<double, double>* makeTable2D(
            const bp::object& xs, const bp::object& ys, const bp::object& valarray,
            const std::string& interp)
        {
            const int Nx = GetNumpyArrayDim(valarray.ptr(), 1);
            const int Ny = GetNumpyArrayDim(valarray.ptr(), 0);
            assert(Nx == GetNumpyArrayDim(xs.ptr(), 0));
            assert(Ny == GetNumpyArrayDim(ys.ptr(), 0));
            const double* xargs = GetNumpyArrayData<double>(xs.ptr());
            const double* yargs = GetNumpyArrayData<double>(ys.ptr());
            const double* vals = GetNumpyArrayData<double>(valarray.ptr());
            Table2D<double,double>::interpolant i = Table2D<double,double>::linear;
            if (interp == "linear") i = Table2D<double,double>::linear;
            else if (interp == "floor") i = Table2D<double,double>::floor;
            else if (interp == "ceil") i = Table2D<double,double>::ceil;
            else {
                PyErr_SetString(PyExc_ValueError, "Invalid interpolant");
                bp::throw_error_already_set();
            }
            return new Table2D<double,double>(xargs, yargs, vals, Nx, Ny, i);
        }

        static void interpManyScatter(const Table2D<double,double>& table2d,
                                      const bp::object& x, const bp::object& y,
                                      const bp::object& vals)
        {
            const double* xvec = GetNumpyArrayData<double>(x.ptr());
            const double* yvec = GetNumpyArrayData<double>(y.ptr());
            double* valvec = GetNumpyArrayData<double>(vals.ptr());
            int N = GetNumpyArrayDim(x.ptr(), 0);
            table2d.interpManyScatter(xvec, yvec, valvec, N);
        }

        static void interpManyOuter(const Table2D<double,double>& table2d,
                                    const bp::object& x, const bp::object& y,
                                    const bp::object& vals)
        {
            const double* xvec = GetNumpyArrayData<double>(x.ptr());
            const double* yvec = GetNumpyArrayData<double>(y.ptr());
            double* valvec = GetNumpyArrayData<double>(vals.ptr());
            int Nx = GetNumpyArrayDim(x.ptr(), 0);
            int Ny = GetNumpyArrayDim(y.ptr(), 0);
            int outNx = GetNumpyArrayDim(vals.ptr(), 1);
            int outNy = GetNumpyArrayDim(vals.ptr(), 0);
            assert(Nx == outNx);
            assert(Ny == outNy);
            table2d.interpManyOuter(xvec, yvec, valvec, Nx, Ny);
        }

        static void wrap()
        {
            bp::class_<Table2D<double,double> > pyTable2D("_LookupTable2D", bp::no_init);
            pyTable2D
                .def("__init__",
                    bp::make_constructor(
                        &makeTable2D, bp::default_call_policies(),
                        (bp::arg("xs"), bp::arg("ys"), bp::arg("valarray"), bp::arg("interp"))
                    )
                )
                .def("__call__", &Table2D<double,double>::lookup)
                .def("interpManyScatter", &interpManyScatter)
                .def("interpManyOuter", &interpManyOuter)
                ;
        }
    };

} // anonymous

void pyExportTable()
{
    PyTable::wrap();
}

void pyExportTable2D()
{
    PyTable2D::wrap();
}

} // namespace galsim
