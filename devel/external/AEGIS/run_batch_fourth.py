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


def main(args):
    for fl in glob.glob('outfile/out_4_*'):
        os.remove(fl)
    all_seg = np.loadtxt(args.seg_file_name, delimiter=" ", dtype='S2')
    print "path is", args.main_path
    for f, filt in enumerate(args.filter_names):
        for seg_id in all_seg:
            print 'SEG ID: ', seg_id, ' filter: ', filt
            outfile = 'outfile/out_4_{0}.txt'.format(seg_id)
            com1 = args.main_path
            com = 'python get_cat_seg.py --seg_id=' + seg_id + ' --main_path=' + com1
            final_args = ['bsub', '-W', '0:35', '-o', outfile, com]
            final_args.append("--filter=" + filt)
            final_args.append("--file_filter_name=" + args.filter_file_names[f])
            subprocess.call(final_args)


if __name__ == '__main__':
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument('--filter_names', default=['f606w', 'f814w'],
                        help="names of filters [Default: ['f606w','f814w']]")
    parser.add_argument('--filter_file_names', default=['V', 'I'],
                        help="Names of filters to write inf file [Default: ['V','I']]")
    parser.add_argument('--main_path',
                        default='/nfs/slac/g/ki/ki19/deuce/AEGIS/AEGIS_catalog_full/')
    parser.add_argument('--seg_file_name', default='/nfs/slac/g/ki/ki19/deuce/AEGIS/unzip/seg_ids.txt',
                        help="file with all seg id names")
    args = parser.parse_args()
    main(args)
