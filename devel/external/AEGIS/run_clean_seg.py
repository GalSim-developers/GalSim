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
import numpy as np
import glob
import os


def run_clean_seg(args):
    seg_id = args.seg_id
    for fl in glob.glob('outfile/out_3_' + seg_id + '_*'):
        os.remove(fl)
    print 'SEG ID ', seg_id
    obj_file = args.main_path + seg_id + '/objects_with_p_stamps.txt'
    obj_list = np.loadtxt(obj_file, dtype=int)
    for num in obj_list:
        com = 'python clean_pstamp.py --main_path=' + args.main_path
        final_args = ['bsub', '-W', '0:35', com]
        final_args.append("--seg_id=" + seg_id)
        final_args.append("--num=" + str(num))
        subprocess.call(final_args)


if __name__ == '__main__':
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument('--seg_id', default='0a',
                        help="id of segment to run [Default:'0a']")
    parser.add_argument('--main_path',
                        default='/nfs/slac/g/ki/ki19/deuce/AEGIS/AEGIS_catalog_full/',
                        help="Path where image files are stored \
                        [Default:'/nfs/slac/g/ki/ki19/deuce/AEGIS/AEGIS_catalog_full/']")
    args = parser.parse_args()
    run_clean_seg(args)
