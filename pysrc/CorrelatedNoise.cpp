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
                .def(bp::init<const BaseImage<U> &,
                     boost::shared_ptr<InterpolantXY>,
                     boost::shared_ptr<InterpolantXY>,
                     double, double>(
                         (bp::arg("image"),
                          bp::arg("xInterp")=bp::object(),
                          bp::arg("kInterp")=bp::object(),
                          bp::arg("dx")=0., bp::arg("pad_factor")=0.)
                     ))
               .def(
                     "getCovarianceMatrix", (
                         Image<double> (CorrelationFunction::*) (ImageView<U>, double) 
                         const)&CorrelationFunction::getCovarianceMatrix, 
                      (bp::arg("image"), bp::arg("dx")=0.))
                  ;
        }

        static void wrap() {
	  bp::class_< CorrelationFunction, bp::bases<SBInterpolatedImage> > pyCorrelationFunction(
                "CorrelationFunction", bp::init<const CorrelationFunction &>()
            );
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
