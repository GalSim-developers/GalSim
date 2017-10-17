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

#include "galsim/IgnoreWarnings.h"
#include "boost/python.hpp" // header that includes Python.h always needs to come first

#include "Table.h"

namespace bp = boost::python;

namespace galsim {
namespace {

    struct PyTable {

        static Table* makeTable(
            size_t iargs, size_t ivals, int N, const char* interp_c)
        {
            const double* args = reinterpret_cast<double*>(iargs);
            const double* vals = reinterpret_cast<double*>(ivals);
            std::string interp(interp_c);

            Table::interpolant i = Table::linear;
            if (interp == "spline") i = Table::spline;
            else if (interp == "floor") i = Table::floor;
            else if (interp == "ceil") i = Table::ceil;
            else if (interp == "nearest") i = Table::nearest;

            return new Table(args, vals, N, i);
        }

        static void interpMany(const Table& table, size_t iargs, size_t ivals, int N)
        {
            const double* args = reinterpret_cast<const double*>(iargs);
            double* vals = reinterpret_cast<double*>(ivals);
            table.interpMany(args, vals, N);
        }

        static void wrap()
        {
            // docstrings are in galsim/table.py
            bp::class_<Table > pyTable("_LookupTable", bp::no_init);
            pyTable
                .def("__init__",
                     bp::make_constructor(
                         &makeTable, bp::default_call_policies(),
                         (bp::arg("args"), bp::arg("vals"), bp::args("N"), bp::arg("interp"))
                     )
                )
                .def(bp::init<const Table &>(bp::args("other")))

                // Use version that throws expection if out of bounds
                .def("__call__", &Table::lookup)
                .def("interpMany", &interpMany)
                ;
        }

    }; // struct PyTable

    struct PyTable2D{
        static Table2D* makeTable2D(
            size_t ix, size_t iy, size_t ivals, int Nx, int Ny,
            const char* interp_c)
        {
            const double* x = reinterpret_cast<const double*>(ix);
            const double* y = reinterpret_cast<const double*>(iy);
            const double* vals = reinterpret_cast<const double*>(ivals);
            std::string interp(interp_c);

            Table2D::interpolant i = Table2D::linear;
            if (interp == "floor") i = Table2D::floor;
            else if (interp == "ceil") i = Table2D::ceil;
            else if (interp == "nearest") i = Table2D::nearest;

            return new Table2D(x, y, vals, Nx, Ny, i);
        }

        static void interpMany(const Table2D& table2d,
                               size_t ix, size_t iy, size_t ivals, int N)
        {
            const double* x = reinterpret_cast<const double*>(ix);
            const double* y = reinterpret_cast<const double*>(iy);
            double* vals = reinterpret_cast<double*>(ivals);
            table2d.interpMany(x, y, vals, N);
        }

        static void Gradient(const Table2D& table2d,
                             double x, double y, size_t igrad)
        {
            double* grad = reinterpret_cast<double*>(igrad);
            table2d.gradient(x, y, grad[0], grad[1]);
        }

        static void GradientMany(const Table2D& table2d,
                                 size_t ix, size_t iy, size_t idfdx, size_t idfdy, int N)
        {
            const double* x = reinterpret_cast<const double*>(ix);
            const double* y = reinterpret_cast<const double*>(iy);
            double* dfdx = reinterpret_cast<double*>(idfdx);
            double* dfdy = reinterpret_cast<double*>(idfdy);
            table2d.gradientMany(x, y, dfdx, dfdy, N);
        }

        static void wrap()
        {
            bp::class_<Table2D > pyTable2D("_LookupTable2D", bp::no_init);
            pyTable2D
                .def("__init__",
                    bp::make_constructor(
                        &makeTable2D, bp::default_call_policies(),
                        (bp::arg("x"), bp::arg("y"), bp::arg("f"),
                         bp::arg("Nx"), bp::arg("Ny"), bp::arg("interp"))
                    )
                )
                .def("__call__", &Table2D::lookup)
                .def("interpMany", &interpMany)
                .def("gradient", &Gradient)
                .def("gradientMany", &GradientMany)
                ;
        }
    };

} // anonymous

void pyExportTable()
{
    PyTable::wrap();
    PyTable2D::wrap();
}

} // namespace galsim
