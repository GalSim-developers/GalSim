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
#include "SBInterpolatedImage.h"

namespace bp = boost::python;

namespace galsim {

    struct PySBInterpolatedImage 
    {

        template <typename U, typename W>
        static void wrapTemplates_Multi(W& wrapper) 
        {
            wrapper
                .def(bp::init<const std::vector<boost::shared_ptr<BaseImage<U> > >&, 
                     double, double>(
                        (bp::arg("images"),
                         bp::arg("dx")=0., bp::arg("pad_factor")=0.)
                ))
                .def(bp::init<const BaseImage<U>&, double, double, boost::shared_ptr<Image<U> > >(
                        (bp::arg("image"),
                         bp::arg("dx")=0., bp::arg("pad_factor")=0.,
                         bp::arg("pad_image")=bp::object())
                ))
                ;
        }

        template <typename U, typename W>
        static void wrapTemplates(W& wrapper) 
        {
            wrapper
                .def(bp::init<const BaseImage<U> &,
                     boost::shared_ptr<InterpolantXY>,
                     boost::shared_ptr<InterpolantXY>,
                     double, double, boost::shared_ptr<Image<U> >,
                     boost::shared_ptr<GSParams> >(
                         (bp::arg("image"),
                          bp::arg("xInterp")=bp::object(),
                          bp::arg("kInterp")=bp::object(),
                          bp::arg("dx")=0., bp::arg("pad_factor")=0.,
                          bp::arg("pad_image")=bp::object(),
                          bp::arg("gsparams")=bp::object())
                     )
                )
                ;
        }

        static void wrap() 
        {
            bp::class_< MultipleImageHelper > pyMultipleImageHelper(
                "MultipleImageHelper", bp::init<const MultipleImageHelper &>()
            );
            wrapTemplates_Multi<float>(pyMultipleImageHelper);
            wrapTemplates_Multi<double>(pyMultipleImageHelper);
            wrapTemplates_Multi<int32_t>(pyMultipleImageHelper);
            wrapTemplates_Multi<int16_t>(pyMultipleImageHelper);

            bp::class_< SBInterpolatedImage, bp::bases<SBProfile> > pySBInterpolatedImage(
                "SBInterpolatedImage", bp::init<const SBInterpolatedImage &>()
            );
            pySBInterpolatedImage
                .def(bp::init<const MultipleImageHelper&, const std::vector<double>&,
                     boost::shared_ptr<InterpolantXY>,
                     boost::shared_ptr<InterpolantXY>,
                     boost::shared_ptr<GSParams> >(
                         (bp::args("multi","weights"),
                          bp::arg("xInterp")=bp::object(),
                          bp::arg("kInterp")=bp::object(),
                          bp::arg("gsparams")=bp::object())
                     )
                )
                .def("calculateStepK", &SBInterpolatedImage::calculateStepK)
                .def("calculateMaxK", &SBInterpolatedImage::calculateMaxK)
                ;
            wrapTemplates<float>(pySBInterpolatedImage);
            wrapTemplates<double>(pySBInterpolatedImage);
            wrapTemplates<int32_t>(pySBInterpolatedImage);
            wrapTemplates<int16_t>(pySBInterpolatedImage);
        }

    };

    void pyExportSBInterpolatedImage() 
    {
        PySBInterpolatedImage::wrap();
    }

} // namespace galsim
