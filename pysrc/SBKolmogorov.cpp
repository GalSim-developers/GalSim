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

#include "SBKolmogorov.h"

namespace bp = boost::python;

namespace galsim {

    struct PySBKolmogorov 
    {
        static void wrap() 
        {
            bp::class_<SBKolmogorov,bp::bases<SBProfile> >("SBKolmogorov", bp::no_init)
                .def(bp::init<double,double,boost::shared_ptr<GSParams> >(
                        (bp::arg("lam_over_r0"), bp::arg("flux")=1.,
                         bp::arg("gsparams")=bp::object()))
                )
                .def(bp::init<const SBKolmogorov &>())
                .def("getLamOverR0", &SBKolmogorov::getLamOverR0)
                ;
        }
    };

    void pyExportSBKolmogorov() 
    {
        PySBKolmogorov::wrap();
    }

} // namespace galsim
