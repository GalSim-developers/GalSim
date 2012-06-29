#include "boost/python.hpp"

#define PY_ARRAY_UNIQUE_SYMBOL SBPROFILE_ARRAY_API
#include "numpy/arrayobject.h"

namespace galsim {

    void pyExportAngle();
    void pyExportBounds();
    void pyExportShear();
    void pyExportImage();
    void pyExportPhotonArray();
    void pyExportSBProfile();
    void pyExportSBAdd();
    void pyExportSBConvolve();
    void pyExportSBDeconvolve();
    void pyExportSBTransform();
    void pyExportSBBox();
    void pyExportSBGaussian();
    void pyExportSBExponential();
    void pyExportSBSersic();
    void pyExportSBMoffat();
    void pyExportSBAiry();
    void pyExportSBLaguerre();
    void pyExportSBInterpolatedImage();
    void pyExportRandom();

    namespace hsm{
        void pyExportPSFCorr();
    } // namespace hsm

} // namespace galsim

BOOST_PYTHON_MODULE(_galsim) {
    import_array(); // for numpy
    galsim::pyExportAngle();
    galsim::pyExportBounds();
    galsim::pyExportShear();
    galsim::pyExportImage();
    galsim::pyExportPhotonArray();
    galsim::pyExportSBProfile();
    galsim::pyExportSBAdd();
    galsim::pyExportSBConvolve();
    galsim::pyExportSBDeconvolve();
    galsim::pyExportSBTransform();
    galsim::pyExportSBBox();
    galsim::pyExportSBGaussian();
    galsim::pyExportSBExponential();
    galsim::pyExportSBSersic();
    galsim::pyExportSBMoffat();
    galsim::pyExportSBAiry();
    galsim::pyExportSBLaguerre();
    galsim::pyExportSBInterpolatedImage();
    galsim::pyExportRandom();
    galsim::hsm::pyExportPSFCorr();
}
