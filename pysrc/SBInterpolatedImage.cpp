#include "boost/python.hpp"
#include "SBInterpolatedImage.h"

namespace bp = boost::python;

namespace galsim {

    struct PySBInterpolatedImage 
    {

        template <typename U, typename W>
        static void wrapTemplates_Multi(W & wrapper) {
            wrapper
                .def(bp::init<const std::vector<boost::shared_ptr<BaseImage<U> > >&, 
                     double, double, boost::shared_ptr<ImageView<U> > >(
                        (bp::arg("images"),
                         bp::arg("dx")=0., bp::arg("pad_factor")=0.,
                         bp::arg("pad_image")=bp::object())
                ))
                .def(bp::init<const BaseImage<U> &, double, double, boost::shared_ptr<ImageView<U> > >(
                        (bp::arg("image"),
                         bp::arg("dx")=0., bp::arg("pad_factor")=0.,
                         bp::arg("pad_image")=bp::object())
                ))
                ;
        }

        template <typename U, typename W>
        static void wrapTemplates(W & wrapper) {
            wrapper
                .def(bp::init<const BaseImage<U> &,
                     boost::shared_ptr<InterpolantXY>,
                     boost::shared_ptr<InterpolantXY>,
                     double, double, boost::shared_ptr<ImageView <U> > >(
                         (bp::arg("image"),
                          bp::arg("xInterp")=bp::object(),
                          bp::arg("kInterp")=bp::object(),
                          bp::arg("dx")=0., bp::arg("pad_factor")=0.,
                          bp::arg("pad_image")=bp::object())
                     ))
                ;
        }

        static void wrap() {
            bp::class_< MultipleImageHelper > pyMultipleImageHelper(
                "MultipleImageHelper", bp::init<const MultipleImageHelper &>()
            );
            wrapTemplates_Multi<float>(pyMultipleImageHelper);
            wrapTemplates_Multi<double>(pyMultipleImageHelper);
            wrapTemplates_Multi<short>(pyMultipleImageHelper);
            wrapTemplates_Multi<int>(pyMultipleImageHelper);

            bp::class_< SBInterpolatedImage, bp::bases<SBProfile> > pySBInterpolatedImage(
                "SBInterpolatedImage", bp::init<const SBInterpolatedImage &>()
            );
            pySBInterpolatedImage
                .def(bp::init<const MultipleImageHelper&, const std::vector<double>&,
                     boost::shared_ptr<InterpolantXY>,
                     boost::shared_ptr<InterpolantXY> >(
                         (bp::args("multi","weights"),
                          bp::arg("xInterp")=bp::object(),
                          bp::arg("kInterp")=bp::object())
                     ))
                .def("calculateStepK", &SBInterpolatedImage::calculateStepK)
                .def("calculateMaxK", &SBInterpolatedImage::calculateMaxK)
                ;
            wrapTemplates<float>(pySBInterpolatedImage);
            wrapTemplates<double>(pySBInterpolatedImage);
            wrapTemplates<short>(pySBInterpolatedImage);
            wrapTemplates<int>(pySBInterpolatedImage);
        }

    };

    void pyExportSBInterpolatedImage() 
    {
        PySBInterpolatedImage::wrap();
    }

} // namespace galsim
