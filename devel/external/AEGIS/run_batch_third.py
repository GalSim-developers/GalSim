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

import subprocess
import glob
import os
import numpy as np


def run_clean_pstamps():
    for fl in glob.glob('outfile/out_3_seg_*'):
        os.remove(fl)
    file_name = '/nfs/slac/g/ki/ki19/deuce/AEGIS/unzip/seg_ids.txt'
    all_seg_ids = np.loadtxt(file_name, delimiter=" ", dtype='S2')
    for seg_id in all_seg_ids:
        print 'SEG ID ', seg_id
        outfile = 'outfile/out_3_seg_{0}.txt'.format(seg_id)
        com1 = '/nfs/slac/g/ki/ki19/deuce/AEGIS/AEGIS_catalog_full/'
        com = 'python run_clean_seg.py --seg_id=' + seg_id + ' --main_path=' + com1
        final_args = ['bsub', '-W', '1:55', '-o', outfile, com]
        print outfile
        subprocess.call(final_args)


if __name__ == '__main__':
    run_clean_pstamps()
