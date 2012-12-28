
#include "boost/python.hpp" // header that includes Python.h always needs to come first

#define PY_ARRAY_UNIQUE_SYMBOL SBPROFILE_ARRAY_API
#define NO_IMPORT_ARRAY
#include "numpy/arrayobject.h"

#include "Table.h"

namespace bp = boost::python;

namespace galsim {
namespace {

    // We only export the Table<double,double>, so don't baother templatizing PyTable.
    struct PyTable {

        static void destroyCObjectOwner(void* p) {
            boost::shared_ptr<double>* owner = reinterpret_cast< boost::shared_ptr<double>*>(p);
            delete owner;
        }

        struct PythonDeleter {
            void operator()(double* p) { owner.reset(); }
            explicit PythonDeleter(PyObject* o) : owner(bp::borrowed(o)) {}
            bp::handle<> owner;
        };

        static Table<double,double>* makeTable(
            const bp::object& arg_array, const bp::object& val_array, 
            const std::string& interp_str)
        {
            if (!PyArray_Check(arg_array.ptr())) {
                PyErr_SetString(PyExc_TypeError, "numpy.ndarray argument required for arg_array");
                bp::throw_error_already_set();
            }
            if (!PyArray_Check(val_array.ptr())) {
                PyErr_SetString(PyExc_TypeError, "numpy.ndarray argument required for val_array");
                bp::throw_error_already_set();
            }
            if (PyArray_NDIM(arg_array.ptr()) != 1) {
                PyErr_SetString(PyExc_ValueError, "arg_array argument must be 1-d");
                bp::throw_error_already_set();
            }
            if (PyArray_NDIM(val_array.ptr()) != 1) {
                PyErr_SetString(PyExc_ValueError, "val_array argument must be 1-d");
                bp::throw_error_already_set();
            }
            if (PyArray_STRIDE(arg_array.ptr(), 0) != sizeof(double)) {
                PyErr_SetString(PyExc_ValueError, "arg_array argument must have contiguous data");
                bp::throw_error_already_set();
            }
            if (PyArray_STRIDE(val_array.ptr(), 0) != sizeof(double)) {
                PyErr_SetString(PyExc_ValueError, "val_array argument must have contiguous data");
                bp::throw_error_already_set();
            }
            const double* arg_data = reinterpret_cast<const double*>(PyArray_DATA(arg_array.ptr()));
            const double* val_data = reinterpret_cast<const double*>(PyArray_DATA(val_array.ptr()));

            int n = PyArray_DIM(arg_array.ptr(), 0);
            int n2 = PyArray_DIM(val_array.ptr(), 1);
            if (n != n2) {
                PyErr_SetString(PyExc_ValueError, "arg_array and val_array must be the same size");
                bp::throw_error_already_set();
            }

            Table<double,double>::interpolant i = Table<double,double>::linear;
            if (interp_str == "linear") i = Table<double,double>::linear;
            else if (interp_str == "spline") i = Table<double,double>::spline;
            else if (interp_str == "floor") i = Table<double,double>::floor;
            else if (interp_str == "ceil") i = Table<double,double>::ceil;
            else {
                PyErr_SetString(PyExc_ValueError, "Invalid interpolant");
                bp::throw_error_already_set();
            }

            return new Table<double,double>(arg_data,val_data,n,i);
        }

        static double evalTable(const Table<double,double>& t, double x)
        { return t(x); }

        static void wrap() 
        {
            // docstrings are in galsim/table.py
            bp::class_<Table<double,double> > pyTable("LookupTable", bp::no_init);
            pyTable
                .def("__init__",
                     bp::make_constructor(
                         &makeTable, bp::default_call_policies(),
                         (bp::arg("arg_array"), bp::arg("val_array"), bp::arg("interp"))
                     )
                )
                .def(bp::init<const Table<double,double> &>(bp::args("other")))
                .def("argMin", &Table<double,double>::argMin)
                .def("argMax", &Table<double,double>::argMax)
                .def("__call__", &Table<double,double>::operator())
                .enable_pickling()
                ;
        }

    };

} // anonymous

void pyExportTable() 
{
    PyTable::wrap();
}

} // namespace galsim
