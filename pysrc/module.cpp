#include "Python.h"

PyMODINIT_FUNC init_sbprofile(void) {
    PyObject * m = Py_InitModule("_sbprofile", 0);
};
