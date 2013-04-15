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

#include "SBConvolve.h"

namespace bp = boost::python;

namespace galsim {

    struct PySBConvolve 
    {

        // This will be wrapped as a Python constructor; it accepts an arbitrary Python iterable.
        static SBConvolve * construct(bp::object const & iterable, bool real_space,
                                      boost::shared_ptr<GSParams> gsparams) 
        {
            bp::stl_input_iterator<SBProfile> begin(iterable), end;
            std::list<SBProfile> plist(begin, end);
            return new SBConvolve(plist, real_space, gsparams);
        }

        static void wrap() 
        {
            bp::class_< SBConvolve, bp::bases<SBProfile> >("SBConvolve", bp::no_init)
                // bp tries the overloads in reverse order, so we wrap the most general one first
                // to ensure we try it last
                .def("__init__", bp::make_constructor(
                        &construct, bp::default_call_policies(), 
                        (bp::arg("slist"), bp::arg("real_space")=false,
                         bp::arg("gsparams")=bp::object()))
                )
                .def(bp::init<const SBConvolve &>())
                ;
        }

    };

    struct PySBAutoConvolve 
    {
        static void wrap() {
            bp::class_< SBAutoConvolve, bp::bases<SBProfile> >("SBAutoConvolve", bp::no_init)
                .def(bp::init<const SBProfile &, boost::shared_ptr<GSParams> >(
                        (bp::arg("adaptee"),
                         bp::arg("gsparams")=bp::object())))
                .def(bp::init<const SBAutoConvolve &>())
                ;
        }

    };

    struct PySBAutoCorrelate 
    {
        static void wrap() {
            bp::class_< SBAutoCorrelate, bp::bases<SBProfile> >("SBAutoCorrelate", bp::no_init)
                .def(bp::init<const SBProfile &, boost::shared_ptr<GSParams> >(
                        (bp::arg("adaptee"),
                         bp::arg("gsparams")=bp::object())))
                .def(bp::init<const SBAutoCorrelate &>())
                ;
        }

    };

    void pyExportSBConvolve() 
    {
        PySBConvolve::wrap();
        PySBAutoConvolve::wrap();
        PySBAutoCorrelate::wrap();
    }

} // namespace galsim
