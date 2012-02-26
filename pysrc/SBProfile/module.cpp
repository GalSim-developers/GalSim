#include "boost/python.hpp"

namespace sbp {

void pyExportBounds();

} // namespace sbp

BOOST_PYTHON_MODULE(_sbprofile) {
    sbp::pyExportBounds();
}
