"""@file integ.py
A Python layer version of the C++ int1d function in galim::integ
"""

from . import _galsim

def int1d(func, min, max, rel_err=1.e-6, abs_err=1.e-12):
    """Integrate a 1-dimensional function from min to max.

    @param func     The function to be integrated.  y = func(x) should be valid.
    @param min      The lower end of the integration bounds
    @param max      The upper end of the integration bounds
    @param rel_err  The desired relative error (default `rel_err = 1.e-6`)
    @param abs_err  The desired absolute error (default `rel_err = 1.e-12`)
    """
    print 'Start python int1d'
    min = float(min)
    max = float(max)
    rel_err = float(rel_err)
    abs_err = float(abs_err)
    print 'min = ',min
    print 'max = ',max
    print 'rel_err = ',rel_err
    print 'abs_err = ',abs_err
    return _galsim.PyInt1d(func,min,max,rel_err,abs_err)

