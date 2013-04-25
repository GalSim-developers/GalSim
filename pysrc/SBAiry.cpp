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

#include "SBAiry.h"

namespace bp = boost::python;

namespace galsim {

    struct PySBAiry 
    {
        static void wrap() 
        {
            bp::class_<SBAiry,bp::bases<SBProfile> >("SBAiry", bp::no_init)
                .def(bp::init<double,double,double,boost::shared_ptr<GSParams> >(
                        (bp::arg("lam_over_diam"), bp::arg("obscuration")=0., bp::arg("flux")=1.,
                         bp::arg("gsparams")=bp::object())
                ))
                .def(bp::init<const SBAiry &>())
                .def("getLamOverD", &SBAiry::getLamOverD)
                .def("getObscuration", &SBAiry::getObscuration)
                ;
        }
    };

    void pyExportSBAiry() 
    {
        PySBAiry::wrap();
    }

} // namespace galsim
