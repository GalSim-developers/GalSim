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
    main_path = '/nfs/slac/g/ki/ki19/deuce/AEGIS/AEGIS_catalog_full/'
    run_again = []
    all_seg_ids = np.loadtxt(file_name, delimiter=" ", dtype='S2')
    for i, seg_id in enumerate(all_seg_ids):
        file1 = main_path + str(seg_id) + '/' + 'objects_with_p_stamps.txt'
        objs = np.loadtxt(file1)
        for obj in objs:
            path = main_path + str(seg_id) + '/' + 'postage_stamps'
            test = path + '/f814w_' + seg_id + '_{0}_gal.fits'.format(int(obj))
            if os.path.isfile(test) is False:
                print "clean pstamp didnt work for seg, obj: ", seg_id, int(obj)
                run_again.append([seg_id, int(obj)])
                com = 'python clean_pstamp.py --seg_id=' + seg_id
                final_args = ['bsub', '-W', '0:30', com]
                final_args.append("--num=" + str(int(obj)))
                final_args.append("--main_path=" + main_path)
                subprocess.call(final_args)


if __name__ == '__main__':
    run_clean_pstamps()
