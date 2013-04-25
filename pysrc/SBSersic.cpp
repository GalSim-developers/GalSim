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

#include "SBSersic.h"

namespace bp = boost::python;

namespace galsim {

    struct PySBSersic 
    {

        static void wrap() 
        {
            bp::class_<SBSersic,bp::bases<SBProfile> >("SBSersic", bp::no_init)
                .def(bp::init<double,double,double,double,bool,boost::shared_ptr<GSParams> >(
                        (bp::args("n", "half_light_radius"), bp::arg("flux")=1.,
                         bp::arg("trunc")=0., bp::arg("flux_untruncated")=false,
                         bp::arg("gsparams")=bp::object()))
                )
                .def(bp::init<const SBSersic &>())
                .def("getN", &SBSersic::getN)
                .def("getHalfLightRadius", &SBSersic::getHalfLightRadius)
                ;
        }
    };

    struct PySBDeVaucouleurs 
    {
        static void wrap() 
        {
            bp::class_<SBDeVaucouleurs,bp::bases<SBSersic> >("SBDeVaucouleurs",bp::no_init)
                .def(bp::init<double,double,double,bool,boost::shared_ptr<GSParams> >(
                        (bp::arg("half_light_radius"), bp::arg("flux")=1.,
                         bp::arg("trunc")=0., bp::arg("flux_untruncated")=false,
                         bp::arg("gsparams")=bp::object()))
                )
                .def(bp::init<const SBDeVaucouleurs &>())
                .def("getHalfLightRadius", &SBDeVaucouleurs::getHalfLightRadius)
                ;
        }
    };

    void pyExportSBSersic() 
    {
        PySBSersic::wrap();
        PySBDeVaucouleurs::wrap();
    }

} // namespace galsim
