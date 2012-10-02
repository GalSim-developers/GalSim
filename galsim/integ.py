"""@file integ.py
A Python layer version of the C++ int1d function in galim::integ
"""

from . import _galsim

def int1d(func, min, max, rel_err=1.e-6, abs_err=1.e-12):
    """Integrate a 1-dimensional function from min to max.

    @param func     The function to be integrated.  y = func(x) should be valid.
    @param min      The lower end of the integration bounds (anything < -1.e10 is treated as
                    negative infinity).
    @param max      The upper end of the integration bounds (anything > 1.e10 is treated as positive
                    infinity).
    @param rel_err  The desired relative error (default `rel_err = 1.e-6`)
    @param abs_err  The desired absolute error (default `abs_err = 1.e-12`)
    """
    min = float(min)
    max = float(max)
    rel_err = float(rel_err)
    abs_err = float(abs_err)
    return _galsim.PyInt1d(func,min,max,rel_err,abs_err)

