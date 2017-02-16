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

#define BOOST_NO_CXX11_SMART_PTR
#include "boost/python.hpp"
#include "boost/python/stl_iterator.hpp"

#include "SBTransform.h"
#include "NumpyHelper.h"

namespace bp = boost::python;

namespace galsim {

    struct PySBTransform
    {
        static bp::handle<> getJac(const SBTransform& self) {
            static npy_intp dim[1] = {4};
            // Because the C++ version sets references that are passed in, and that's not possible
            // in Python, we wrap this instead, which returns a numpy array.
            double a=0., b=0., c=0., d=0.;
            self.getJac(a, b, c, d);
            double ar[4] = { a, b, c, d };
            PyObject* r = PyArray_SimpleNewFromData(1, dim, NPY_DOUBLE, ar);
            if (!r) throw bp::error_already_set();
            PyObject* r2 = PyArray_FROM_OF(r, NPY_ARRAY_ENSURECOPY);
            Py_DECREF(r);
            return bp::handle<>(r2);
        }

        static void wrap()
        {
            static char const * doc =
                "SBTransform is an affine transformation of another SBProfile.\n"
                "Origin of original shape will now appear at x0.\n"
                "Flux is NOT conserved in transformation - SB is preserved."
                ;

            bp::class_< SBTransform, bp::bases<SBProfile> >("SBTransform", doc, bp::no_init)
                .def(bp::init<const SBProfile &, double, double, double, double,
                     Position<double>, double,boost::shared_ptr<GSParams> >(
                         (bp::args("sbin", "mA", "mB", "mC", "mD"),
                          bp::arg("x0")=Position<double>(0.,0.),
                          bp::arg("fluxScaling")=1.,
                          bp::arg("gsparams")=bp::object())
                     )
                )
                .def(bp::init<const SBTransform &>())
                .def("getObj", &SBTransform::getObj)
                .def("getJac", getJac)
                .def("getOffset", &SBTransform::getOffset)
                .def("getFluxScaling", &SBTransform::getFluxScaling)
                .enable_pickling()
                ;
        }

    };

    void pyExportSBTransform()
    {
        PySBTransform::wrap();
    }

} // namespace galsim
