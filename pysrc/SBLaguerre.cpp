#include "boost/python.hpp"
#include "boost/python/stl_iterator.hpp"

#include "SBLaguerre.h"

namespace bp = boost::python;

namespace galsim {

    struct PySBLaguerre 
    {
        static void wrap() {
            // TODO: Need to wrap LVector before this will work.
#if 0
            bp::class_<SBLaguerre,bp::bases<SBProfile> >("SBLaguerre", bp::no_init)
                .def(bp::init<LVector,double>(
                        (bp::arg("bvec"), bp::arg("sigma")=1.))
                )
                .def(bp::init<const SBLaguerre &>())
                ;
#endif
        }
    };

    void pyExportSBLaguerre() 
    {
        PySBLaguerre::wrap();
    }

} // namespace galsim
