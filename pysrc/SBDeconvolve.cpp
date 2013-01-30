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

#include "SBDeconvolve.h"

namespace bp = boost::python;

namespace galsim {

    struct PySBDeconvolve 
    {

        static void wrap() {
            bp::class_< SBDeconvolve, bp::bases<SBProfile> >("SBDeconvolve", bp::no_init)
                .def(bp::init<const SBProfile &>(bp::args("adaptee")))
                .def(bp::init<const SBDeconvolve &>())
                ;
        }

    };

    void pyExportSBDeconvolve() 
    {
        PySBDeconvolve::wrap();
    }

} // namespace galsim
