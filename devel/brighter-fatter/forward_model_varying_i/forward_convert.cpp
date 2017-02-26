/*
Craig Lage - 15-Jul-15

C++ code to calculate forward model fit
for Gaussian spots
This file converts the Python class to a C++ class
and builds the .so extension.

file: forward_convert.cpp

*/
#include <stdio.h>
#include <stdlib.h>
#include <algorithm>            // for min_element
#include <string.h>
#include <math.h>
#include <Python.h>
#include "numpy/arrayobject.h"
#define pi 4.0 * atan(1.0)     // Pi
#define max(x,y) (x>y?x:y)      // max macro definition
#define min(x,y) (x>y?y:x)      // min macro definition

using namespace std;

//  DATA STRUCTURES:  

/*  This is the Python class being converted: 

class Array2dSet:
    def __init__(self,xmin,xmax,nx,ymin,ymax,ny):
        self.nx=nx
        self.ny=ny
        self.nstamps=nstamps

        self.xmin=xmin
        self.ymin=ymin
        
        self.xmax=xmax
        self.ymax=ymax
        
        self.dx=(xmax-xmin)/nx
        self.dy=(ymax-ymin)/ny
        
        self.x=linspace(xmin+self.dx/2,xmax-self.dx/2,nx)
        self.y=linspace(ymin+self.dy/2,ymax-self.dy/2,ny)

        self.data=zeros([nx,ny,nstamps])
        self.xoffset=zeros([nstamps])
        self.yoffset=zeros([nstamps])
        self.imax=zeros([nstamps])
*/

class Array //This packages the 2D data sets which are brought from Python
{
 public:
  long nx, ny, nstamps;
  double xmin, xmax, ymin, ymax, dx, dy, *x, *y, *xoffset, *yoffset, *imax, *data;
  Array() {};
  ~Array();
};

Array::~Array() //Destructor
{
  Py_DECREF(this->x);
  Py_DECREF(this->y);
  Py_DECREF(this->xoffset);
  Py_DECREF(this->yoffset);
  Py_DECREF(this->imax);    
  Py_DECREF(this->data);  
}

// This is the top level method:

double FOM(Array*,double,double);

/* class converter */

Array* ClassConvert(PyObject* arg)
{
  /* This converts a pointer to a Python class object into a pointer to a C++ class */
  Array* arr = new Array();
    
    PyObject* pynx = PyObject_GetAttrString(arg, "nx");
    arr->nx = PyLong_AsLong(pynx);

    PyObject* pyny = PyObject_GetAttrString(arg, "ny");
    arr->ny = PyLong_AsLong(pyny);

    PyObject* pynstamps = PyObject_GetAttrString(arg, "nstamps");
    arr->nstamps = PyLong_AsLong(pynstamps);

    PyObject* pydx = PyObject_GetAttrString(arg, "dx");
    arr->dx = PyFloat_AsDouble(pydx);

    PyObject* pydy = PyObject_GetAttrString(arg, "dy");
    arr->dy = PyFloat_AsDouble(pydy);

    PyObject* pyxmin = PyObject_GetAttrString(arg, "xmin");
    arr->xmin = PyFloat_AsDouble(pyxmin);

    PyObject* pyxmax = PyObject_GetAttrString(arg, "xmax");
    arr->xmax = PyFloat_AsDouble(pyxmax);

    PyObject* pyymin = PyObject_GetAttrString(arg, "ymin");
    arr->ymin = PyFloat_AsDouble(pyymin);

    PyObject* pyymax = PyObject_GetAttrString(arg, "ymax");
    arr->ymax = PyFloat_AsDouble(pyymax);

    PyObject* xobj = PyObject_GetAttrString(arg, "x");    
    PyObject *xarr = PyArray_FROM_OTF(xobj, NPY_DOUBLE, NPY_IN_ARRAY);
    arr->x = (double*)PyArray_DATA(xarr);

    PyObject* yobj = PyObject_GetAttrString(arg, "y");    
    PyObject *yarr = PyArray_FROM_OTF(yobj, NPY_DOUBLE, NPY_IN_ARRAY);
    arr->y = (double*)PyArray_DATA(yarr);

    PyObject* xoffsetobj = PyObject_GetAttrString(arg, "xoffset");    
    PyObject *xoffsetarr = PyArray_FROM_OTF(xoffsetobj, NPY_DOUBLE, NPY_IN_ARRAY);
    arr->xoffset = (double*)PyArray_DATA(xoffsetarr);

    PyObject* yoffsetobj = PyObject_GetAttrString(arg, "yoffset");    
    PyObject *yoffsetarr = PyArray_FROM_OTF(yoffsetobj, NPY_DOUBLE, NPY_IN_ARRAY);
    arr->yoffset = (double*)PyArray_DATA(yoffsetarr);

    PyObject* imaxobj = PyObject_GetAttrString(arg, "imax");    
    PyObject *imaxarr = PyArray_FROM_OTF(imaxobj, NPY_DOUBLE, NPY_INOUT_ARRAY);
    arr->imax = (double*)PyArray_DATA(imaxarr);

    PyObject* dataobj = PyObject_GetAttrString(arg, "data");    
    PyObject *dataarr = PyArray_FROM_OTF(dataobj, NPY_DOUBLE, NPY_IN_ARRAY);
    arr->data = (double*)PyArray_DATA(dataarr);

    return arr;
}

/* module functions */

static PyObject *
forward(PyObject *self, PyObject *args)
{

  PyObject *arg1=NULL;
  double sigmax, sigmay, result;
  if (!PyArg_ParseTuple(args, "Odd", &arg1, &sigmax, &sigmay)) return NULL;

  else
    {

    Array* arr1 = ClassConvert(arg1);
    result = FOM(arr1, sigmax, sigmay);

    /* Build the output tuple */
    PyObject *ret = Py_BuildValue("d", result);
    return ret;
    }

}

/* registration table */

static struct PyMethodDef forward_methods[] =
  {
    {"forward", forward, METH_VARARGS, "descript of example"},
    {NULL, NULL, 0, NULL}
  };

/* module initializer */

PyMODINIT_FUNC
initforward (void)
{
    (void)Py_InitModule("forward", forward_methods);
    import_array();
}




