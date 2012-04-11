#include "boost/python.hpp"

#define PY_ARRAY_UNIQUE_SYMBOL SBPROFILE_ARRAY_API
#include "numpy/arrayobject.h"

namespace galsim {

void pyExportBounds();
void pyExportShear();
void pyExportImage();
void pyExportSBProfile();
void pyExportSBInterpolatedImage();
void pyExportRandom();

} // namespace galsim

BOOST_PYTHON_MODULE(_galsim) {
    import_array(); // for numpy
    galsim::pyExportBounds();
    galsim::pyExportShear();
    galsim::pyExportImage();
    galsim::pyExportSBProfile();
    galsim::pyExportSBInterpolatedImage();
    galsim::pyExportRandom();
}
