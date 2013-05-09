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
#include "boost/python.hpp"
#include "boost/python/stl_iterator.hpp"

#include "SBTransform.h"

namespace bp = boost::python;

namespace galsim {

    struct PySBTransform 
    {
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
                ;
        }

    };

    void pyExportSBTransform() 
    {
        PySBTransform::wrap();
    }

} // namespace galsim
