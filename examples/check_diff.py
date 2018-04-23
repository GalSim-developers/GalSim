# Copyright (c) 2012-2018 by the GalSim developers team on GitHub
# https://github.com/GalSim-developers
#
# This file is part of GalSim: The modular galaxy image simulation toolkit.
# https://github.com/GalSim-developers/GalSim
#
# GalSim is free software: redistribution and use in source and binary forms,
# with or without modification, are permitted provided that the following
# conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions, and the disclaimer given in the accompanying LICENSE
#    file.
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions, and the disclaimer given in the documentation
#    and/or other materials provided with the distribution.
#


from __future__ import print_function
import sys
import subprocess

def same(file_name1, file_name2):
    cmd = "diff -q %s %s"%(file_name1, file_name2)
    p = subprocess.Popen([cmd],stdout=subprocess.PIPE, stderr=subprocess.STDOUT, shell=True)
    diff_output = p.stdout.read()
    if len(diff_output) > 0:
        print(diff_output.strip())
    return (len(diff_output) == 0)


def report_txt(file_name1, file_name2):
    # NB. No -q here:
    cmd = "diff %s %s"%(file_name1, file_name2)
    p = subprocess.Popen([cmd],stdout=subprocess.PIPE, stderr=subprocess.STDOUT, shell=True)
    diff_output = p.stdout.read()
    print(diff_output.strip())


def report(file_name1, file_name2):
    try:
        try:
            import astropy.io.fits as pyfits
        except:
            import pyfits
    except ImportError as e:
        print('Unable to import pyfits')
        # Then /usr/bin/env python doesn't have pyfits installed.  Oh well.
        # Let diff do the best it can.
        return report_txt(file_name1, file_name2)

    # Now give more information for fits files
    try:
        f1 = pyfits.open(file_name1)
        f2 = pyfits.open(file_name2)
    except (IOError, OSError) as e:
        # Then either at least one of the files doesn't exist, which diff can report for us,
        # or the files are txt files, which diff can also do.
        return report_txt(file_name1, file_name2)

    for hdu in range(len(f1)):
        d0 = f1[hdu].data
        d1 = f2[hdu].data
        if d0 is None and d1 is None:
            pass
        elif hasattr(d0,'names'):
            if d0.names != d1.names:
                print('    The column names are different in HDU %d:'%hdu)
                print('   ',d0.dtype)
                print('   ',d1.dtype)
            elif d0.dtype != d1.dtype:
                print('    The dtypes are different in HDU %d:'%hdu)
                print('   ',d0.dtype)
                print('   ',d1.dtype)
            else:
                for key in d0.names:
                    if (d0[key] != d1[key]).any():
                        print('    HDU %d shows differences in key %s'%(hdu, key))
        elif (d0 != d1).any():
            print('    HDU %d shows differences in %d pixels'%(hdu, (d0!=d1).sum()))
            print('    The maximum absolute difference is %e.'%(abs(d0-d1).max()))
            print('    The maximum relative difference is %e.'%(abs((d0-d1)/(d0+1.e-10)).max()))


if __name__ == "__main__":
    if not same(*sys.argv[1:]):
        report(*sys.argv[1:])
