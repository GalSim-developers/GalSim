#include "boost/python.hpp"
#include "Interpolant.h"
#include "SBInterpolatedImage.h"
#include "CorrelatedNoise.h"

namespace bp = boost::python;

namespace galsim {

    struct PySBCorrFunc
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
                         Image<double> (SBCorrFunc::*) (ImageView<U>, double) 
                         const)&SBCorrFunc::getCovarianceMatrix, 
                      (bp::arg("image"), bp::arg("dx")=0.))
                  ;
        }

        static void wrap() {
            bp::class_< SBCorrFunc, bp::bases<SBProfile> > pySBCorrFunc(
                "SBCorrFunc", bp::init<const SBCorrFunc &>()
            );
            wrapTemplates<float>(pySBCorrFunc);
            wrapTemplates<double>(pySBCorrFunc);
            wrapTemplates<short>(pySBCorrFunc);
            wrapTemplates<int>(pySBCorrFunc);
        }

    };

    void pyExportSBCorrFunc() 
    {
        PySBCorrFunc::wrap();
    }

} // namespace galsim
