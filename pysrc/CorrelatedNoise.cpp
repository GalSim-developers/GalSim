#include "boost/python.hpp"
#include "Interpolant.h"
#include "CorrelatedNoise.h"

namespace bp = boost::python;

namespace galsim {

    struct PyCorrelationFunction
    {

        template <typename U, typename W>
        static void wrapTemplates(W & wrapper) {
            wrapper
                .def(
                    bp::init<const BaseImage<U> &,
                    boost::shared_ptr<InterpolantXY>,
                    boost::shared_ptr<InterpolantXY>,
                    double, double>(
                        (
                            bp::arg("image"), bp::arg("xInterp")=bp::object(),
                            bp::arg("kInterp")=bp::object(),
                            bp::arg("dx")=0., bp::arg("pad_factor")=0.
                        )
                     )
               )
               .def(
                     "getCovarianceMatrix", (
                         Image<double> (CorrelationFunction::*) (ImageView<U>, double) 
                         const)&CorrelationFunction::getCovarianceMatrix, 
                      (bp::arg("image"), bp::arg("dx")=0.)
               )
               .def("drawShoot", 
                   (double (CorrelationFunction::*)(
                       ImageView<U>, double, UniformDeviate, double, double, bool
                   ) const)&CorrelationFunction::drawShoot, (
                       bp::arg("image"), bp::arg("N")=0., bp::arg("ud"), bp::arg("gain")=1., 
                       bp::arg("max_extra_noise")=0., bp::arg("poisson_flux")=true
                   ),
                   "Draw object into existing image using photon shooting.\n"
                   "\n"
                   "Setting optional integer arg possionFlux != 0 allows profile flux to vary \n"
                   "according to Poisson statistics for N samples.\n"
                   "\n"
                   "Returns total flux of photons that landed inside image bounds."
                )
                .def("draw", 
                     (
                         double (CorrelationFunction::*)(ImageView<U>, double, double) const
                     )&CorrelationFunction::draw, (
                         bp::arg("image"), bp::arg("gain")=1., bp::arg("wmult")=1.
                     ),
                     "Draw in-place and return the summed flux."
                )
                .def("drawK", 
                     (void (CorrelationFunction::*)(
                         ImageView<U>, ImageView<U>, double, double
                     ) const)&CorrelationFunction::drawK, (
                         bp::arg("re"), bp::arg("im"), bp::arg("gain")=1., bp::arg("wmult")=1.
                     ),
                     "Draw k-space image (real and imaginary components)."
                )
                ;
        }

        static void wrap() {
            bp::class_< CorrelationFunction > pyCorrelationFunction(
                "_CorrelationFunction", bp::init<const CorrelationFunction &>()
            );
            pyCorrelationFunction
                .def("xValue", &CorrelationFunction::xValue, bp::args("p"))
                .def("kValue", &CorrelationFunction::kValue, bp::args("k"))
                .def("applyTransformation",
                     &CorrelationFunction::applyTransformation, bp::args("e"))
                .def("applyShear",
                     (void (CorrelationFunction::*)(CppShear))&CorrelationFunction::applyShear,
                     bp::arg("s"))
                .def("applyRotation", &CorrelationFunction::applyRotation, bp::args("theta"))
            ;
            wrapTemplates<float>(pyCorrelationFunction);
            wrapTemplates<double>(pyCorrelationFunction);
            wrapTemplates<short>(pyCorrelationFunction);
            wrapTemplates<int>(pyCorrelationFunction);
        }

    };

    void pyExportCorrelationFunction()
    {
        PyCorrelationFunction::wrap();
    }

} // namespace galsim
