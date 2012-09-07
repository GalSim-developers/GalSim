#include "boost/python.hpp"
#include "Interpolant.h"
#include "SBInterpolatedImage.h"

namespace bp = boost::python;

namespace galsim {

    struct PySBInterpolatedImage 
    {

        template <typename U, typename W>
        static void wrapTemplates_Multi(W & wrapper) {
            wrapper
                .def(bp::init<const std::vector<boost::shared_ptr<BaseImage<U> > >&, 
                     double, double, double>(
                        (bp::arg("images"),
                         bp::arg("dx")=0., bp::arg("pad_factor")=0., bp::arg("pad_variance")=0.)
                ))
              .def(bp::init<const BaseImage<U> &, double, double, double>(
                        (bp::arg("image"),
                         bp::arg("dx")=0., bp::arg("pad_factor")=0., bp::arg("pad_variance")=0.)
                ))
                ;
        }

        template <typename U, typename W>
        static void wrapTemplates(W & wrapper) {
            wrapper
                .def(bp::init<const BaseImage<U> &,
                     boost::shared_ptr<InterpolantXY>,
                     boost::shared_ptr<InterpolantXY>,
                     double, double, double>(
                         (bp::arg("image"),
                          bp::arg("xInterp")=bp::object(),
                          bp::arg("kInterp")=bp::object(),
                          bp::arg("dx")=0., bp::arg("pad_factor")=0., bp::arg("pad_variance")=0.)
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
        // We wrap Interpolant classes as opaque, construct-only objects; we just
        // need to be able to make them from Python and pass them to C++.
        bp::class_<Interpolant,boost::noncopyable>("Interpolant", bp::no_init);
        bp::class_<Interpolant2d,boost::noncopyable>("Interpolant2d", bp::no_init);
        bp::class_<InterpolantXY,bp::bases<Interpolant2d>,boost::noncopyable>(
            "InterpolantXY",
            bp::init<boost::shared_ptr<Interpolant> >(bp::arg("i1d"))
        );
        bp::class_<Delta,bp::bases<Interpolant>,boost::noncopyable>(
            "Delta", bp::init<double>(bp::arg("tol")=1E-3)
        );
        bp::class_<Nearest,bp::bases<Interpolant>,boost::noncopyable>(
            "Nearest", bp::init<double>(bp::arg("tol")=1E-3)
        );
        bp::class_<SincInterpolant,bp::bases<Interpolant>,boost::noncopyable>(
            "SincInterpolant", bp::init<double>(bp::arg("tol")=1E-3)
        );
        bp::class_<Linear,bp::bases<Interpolant>,boost::noncopyable>(
            "Linear", bp::init<double>(bp::arg("tol")=1E-3)
        );
        bp::class_<Lanczos,bp::bases<Interpolant>,boost::noncopyable>(
            "Lanczos", bp::init<int,bool,double>(
                (bp::arg("n"), bp::arg("conserve_flux")=false, bp::arg("tol")=1E-3)
            )
        );
        bp::class_<Cubic,bp::bases<Interpolant>,boost::noncopyable>(
            "Cubic", bp::init<double>(bp::arg("tol")=1E-4)
        );
        bp::class_<Quintic,bp::bases<Interpolant>,boost::noncopyable>(
            "Quintic", bp::init<double>(bp::arg("tol")=1E-4)
        );

        PySBInterpolatedImage::wrap();
    }

} // namespace galsim
