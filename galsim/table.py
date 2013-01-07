"""@file table.py
A few adjustments to galsim.LookupTable at the Python layer, including the addition of 
the docstring.
"""
from . import _galsim

# First of all we add docstrings to the LookupTable
_galsim.LookupTable.__doc__ = """
    LookupTable represents a lookup table to store function values that may be slow to calculate,
    for which interpolating from a lookup table is sufficiently accurate.

    A LookupTable is constructed from two arrays and a string indicating what kind of 
    interpolation to use.

        args = [...]
        vals = []
        for arg in args:
            val = calculateVal(arg)
            vals.append(val)
        interp = 'spline'  # Other options are 'linear', 'floor' and 'ceil'. 
        table = galsim.LookupTable(args,vals,interp)

    Then you can use this table as a replacement for the slow calculation:

        other_args = [...]
        for arg in other_args:
            val = table(arg)
            [... use val ...]
    """

# Some functions to enable pickling of tables
def LookupTable_getinitargs(self):
    return self.getArgs(), self.getVals(), self.getInterp()

_galsim.LookupTable.__getinitargs__ = LookupTable_getinitargs
