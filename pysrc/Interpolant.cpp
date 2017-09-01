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
#include "boost/python.hpp"

#include "Interpolant.h"

namespace bp = boost::python;

namespace galsim {

    template<typename T>
    T *get_pointer(shared_ptr<T> p) { return p.get(); }

    struct PyInterpolant
    {

        static void wrap()
        {
            bp::class_<Interpolant, shared_ptr<Interpolant>, boost::noncopyable>(
                "Interpolant", bp::no_init);

            bp::class_<Delta, shared_ptr<Delta>, bp::bases<Interpolant> >("Delta", bp::no_init)
                .def(bp::init<double,GSParams>((bp::arg("tol")=1e-4,
                                                bp::arg("gsparams")=GSParams())));

            bp::class_<Nearest, shared_ptr<Nearest>, bp::bases<Interpolant> >(
                "Nearest", bp::no_init)
                .def(bp::init<double,GSParams>((bp::arg("tol")=1e-4,
                                                bp::arg("gsparams")=GSParams())));

            bp::class_<SincInterpolant, shared_ptr<SincInterpolant>, bp::bases<Interpolant> >(
                "SincInterpolant", bp::no_init)
                .def(bp::init<double,GSParams>((bp::arg("tol")=1e-4,
                                                bp::arg("gsparams")=GSParams())));

            bp::class_<Lanczos, shared_ptr<Lanczos>, bp::bases<Interpolant> >(
                "Lanczos", bp::no_init)
                .def(bp::init<int,bool,double,GSParams>(
                    (bp::arg("n"), bp::arg("conserve_dc")=true, bp::arg("tol")=1e-4,
                     bp::arg("gsparams")=GSParams())));

            bp::class_<Linear, shared_ptr<Linear>, bp::bases<Interpolant> >("Linear", bp::no_init)
                .def(bp::init<double,GSParams>((bp::arg("tol")=1e-4,
                                                bp::arg("gsparams")=GSParams())));

            bp::class_<Cubic, shared_ptr<Cubic>, bp::bases<Interpolant> >("Cubic", bp::no_init)
                .def(bp::init<double,GSParams>((bp::arg("tol")=1e-4,
                                                bp::arg("gsparams")=GSParams())));

            bp::class_<Quintic, shared_ptr<Quintic>, bp::bases<Interpolant> >(
                "Quintic", bp::no_init)
                .def(bp::init<double,GSParams>((bp::arg("tol")=1e-4,
                                                bp::arg("gsparams")=GSParams())));

            bp::implicitly_convertible<shared_ptr<Delta>,shared_ptr<Interpolant> >();
            bp::implicitly_convertible<shared_ptr<Nearest>,shared_ptr<Interpolant> >();
            bp::implicitly_convertible<shared_ptr<SincInterpolant>,shared_ptr<Interpolant> >();
            bp::implicitly_convertible<shared_ptr<Lanczos>,shared_ptr<Interpolant> >();
            bp::implicitly_convertible<shared_ptr<Linear>,shared_ptr<Interpolant> >();
            bp::implicitly_convertible<shared_ptr<Cubic>,shared_ptr<Interpolant> >();
            bp::implicitly_convertible<shared_ptr<Quintic>,shared_ptr<Interpolant> >();
        }

    };

    void pyExportInterpolant()
    { PyInterpolant::wrap(); }

} // namespace galsim
