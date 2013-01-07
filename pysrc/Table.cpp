
#include "boost/python.hpp" // header that includes Python.h always needs to come first
#include "boost/python/stl_iterator.hpp"

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
            const bp::object& args, const bp::object& vals, const std::string& interp)
        {
            std::vector<double> vargs, vvals;
            try {
                bp::stl_input_iterator<double> args_it(args);
                bp::stl_input_iterator<double> end;
                vargs.insert(vargs.end(),args_it,end);
            } catch (std::exception& e) {
                PyErr_SetString(PyExc_ValueError, "Unable to convert args to C++ vector");
                bp::throw_error_already_set();
            }
            try {
                bp::stl_input_iterator<double> vals_it(vals);
                bp::stl_input_iterator<double> end;
                vvals.insert(vvals.end(),vals_it,end);
            } catch (std::exception& e) {
                PyErr_SetString(PyExc_ValueError, "Unable to convert vals to C++ vector");
                bp::throw_error_already_set();
            }
            if (vargs.size() != vvals.size()) {
                PyErr_SetString(PyExc_ValueError, "args and vals must be the same size");
                bp::throw_error_already_set();
            }

            Table<double,double>::interpolant i = Table<double,double>::linear;
            if (interp == "linear") i = Table<double,double>::linear;
            else if (interp == "spline") i = Table<double,double>::spline;
            else if (interp == "floor") i = Table<double,double>::floor;
            else if (interp == "ceil") i = Table<double,double>::ceil;
            else {
                PyErr_SetString(PyExc_ValueError, "Invalid interpolant");
                bp::throw_error_already_set();
            }

            return new Table<double,double>(vargs,vvals,i);
        }

        static bp::list convertGetArgs(const Table<double,double>& table)
        {
            bp::object get_iter = bp::iterator<std::vector<double> >();
            bp::object iter = get_iter(table.getArgs());
            bp::list l(iter);
            return l;
        }

        static bp::object convertGetVals(const Table<double,double>& table)
        {
            bp::object get_iter = bp::iterator<std::vector<double> >();
            bp::object iter = get_iter(table.getVals());
            bp::list l(iter);
            return l;
        }

        static std::string convertGetIType(const Table<double,double>& table)
        {
            Table<double,double>::interpolant i = table.getIType();
            switch (i) {
                case Table<double,double>::linear:
                     return std::string("linear");
                case Table<double,double>::spline:
                     return std::string("spline");
                case Table<double,double>::floor:
                     return std::string("floor");
                case Table<double,double>::ceil:
                     return std::string("ceil");
                default:
                     PyErr_SetString(PyExc_ValueError, "Invalid interpolant");
                     bp::throw_error_already_set();
            }
            // Shouldn't get here...
            return std::string("");
        }

        static void wrap() 
        {
            // docstrings are in galsim/table.py
            bp::class_<Table<double,double> > pyTable("LookupTable", bp::no_init);
            pyTable
                .def("__init__",
                     bp::make_constructor(
                         &makeTable, bp::default_call_policies(),
                         (bp::arg("args"), bp::arg("vals"), bp::arg("interp"))
                     )
                )
                .def(bp::init<const Table<double,double> &>(bp::args("other")))
                .def("argMin", &Table<double,double>::argMin)
                .def("argMax", &Table<double,double>::argMax)

                // Use version that throws expection if out of bounds
                .def("__call__", &Table<double,double>::lookup) 

                .def("getArgs", &Table<double,double>::getArgs)
                .def("getVals", &Table<double,double>::getVals)
                .def("getInterp", &convertGetIType)
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
