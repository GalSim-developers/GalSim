#include "boost/python.hpp"
#include "boost/python/stl_iterator.hpp"

#include "PhotonArray.h"

namespace bp = boost::python;

namespace galsim {

    struct PyPhotonArray 
    {

        static PhotonArray * construct(bp::object const & vx, bp::object const & vy,
                                       bp::object const & vflux) {
            Py_ssize_t size = bp::len(vx);
            if (size != bp::len(vx)) {
                PyErr_SetString(PyExc_ValueError,
                                "Length of vx array does not match  length of vy array");
                bp::throw_error_already_set();
            }
            if (size != bp::len(vflux)) {
                PyErr_SetString(PyExc_ValueError,
                                "Length of vx array does not match length of vflux array");
                bp::throw_error_already_set();
            }
            std::vector<double> vx_(size);
            std::vector<double> vy_(size);
            std::vector<double> vflux_(size);
            for (Py_ssize_t n = 0; n < size; ++n) {
                vx_[n] = bp::extract<double>(vx[n]);
                vy_[n] = bp::extract<double>(vy[n]);
                vflux_[n] = bp::extract<double>(vflux[n]);
            }
            return new PhotonArray(vx_, vy_, vflux_);
        }

        static void wrap() {
            const char * doc = 
                "\n"
                "Class to hold a list of 'photon' arrival positions\n"
                "\n"
                "Class holds a vector of information about photon arrivals: x\n"
                "and y positions, and a flux carried by each photon.  It is the\n"
                "intention that fluxes of photons be nearly equal in absolute\n"
                "value so that noise statistics can be estimated by counting\n"
                "number of positive and negative photons.  This class holds the\n"
                "code that allows its flux to be added to a surface-brightness\n"
                "Image.\n"
                ;
            bp::class_<PhotonArray> pyPhotonArray("PhotonArray", doc, bp::no_init);
            pyPhotonArray
                .def(
                    "__init__",
                    bp::make_constructor(&construct, bp::default_call_policies(),
                                         bp::args("vx", "vy", "vflux"))
                )
                .def(bp::init<int>(bp::args("n")))
                .def("__len__", &PhotonArray::size)
                .def("reserve", &PhotonArray::reserve)
                .def("setPhoton", &PhotonArray::setPhoton, bp::args("i", "x", "y", "flux"))
                .def("getX", &PhotonArray::getX)
                .def("getY", &PhotonArray::getY)
                .def("getFlux", &PhotonArray::getFlux)
                .def("getTotalFlux", &PhotonArray::getTotalFlux)
                .def("setTotalFlux", &PhotonArray::setTotalFlux)
                .def("append", &PhotonArray::append)
                .def("convolve", &PhotonArray::convolve)
                .def("addTo", 
                     (double(PhotonArray::*)(ImageView<float> &) const)&PhotonArray::addTo,
                     bp::arg("image"),
                     "Add photons' fluxes into image. Returns total flux of photons falling inside "
                     "image bounds.")
                .def("addTo", 
                     (double(PhotonArray::*)(ImageView<double> &) const)&PhotonArray::addTo,
                     bp::arg("image"),
                     "Add photons' fluxes into image. Returns total flux of photons falling inside "
                     "image bounds.")
                ;
        }

    };

    void pyExportPhotonArray() 
    {
        PyPhotonArray::wrap();
    }

} // namespace galsim
