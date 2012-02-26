#include "boost/python.hpp"

#define PY_ARRAY_UNIQUE_SYMBOL SBPROFILE_ARRAY_API
#include "numpy/arrayobject.h"

namespace sbp {

void pyExportBounds();
void pyExportShear();

} // namespace sbp

BOOST_PYTHON_MODULE(_sbprofile) {
    import_array(); // for numpy
    sbp::pyExportBounds();
    sbp::pyExportShear();
}
