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

#include "SBAdd.h"

namespace bp = boost::python;

namespace galsim {

    struct PySBAdd 
    {

        // This will be wrapped as a Python constructor; it accepts an arbitrary Python iterable.
        static SBAdd * construct(bp::object const & iterable) {
            bp::stl_input_iterator<SBProfile> begin(iterable), end;
            std::list<SBProfile> plist(begin, end);
            return new SBAdd(plist);
        }

        static void wrap() {
            static char const * doc = 
                "Sum of SBProfiles."
                ;

            bp::class_< SBAdd, bp::bases<SBProfile> >("SBAdd", doc, bp::no_init)
                // bp tries the overloads in reverse order, so we wrap the most general one first
                // to ensure we try it last
                .def("__init__", bp::make_constructor(&construct, bp::default_call_policies(),
                                                      bp::args("slist")))
                .def(bp::init<const SBProfile &, const SBProfile &>(bp::args("s1", "s2")))
                .def(bp::init<const SBAdd &>())
                ;
        }

    };

    void pyExportSBAdd() 
    {
        PySBAdd::wrap();
    }

} // namespace galsim
