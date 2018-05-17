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

""" Program Number: 2

Program to remove multiple detections of an object in overlapping segments.
Searches of objects within 2 arcseconds. Object with smaller flag or higher SNR
is kept. Every pair of segments are checked for overlapping objects once.
"""
from astropy.table import Table
from scipy import spatial


def main(args):
    all_seg = np.loadtxt(args.seg_file_name, delimiter=" ", dtype='S2')
    for f, filt in enumerate(args.filter_names):
        check_segs = list(all_seg)
        for seg in all_seg:
            cat_name = args.main_path + seg + '/' + filt + '_clean.cat'
            cat = Table.read(cat_name, format='ascii.basic')
            q1, = np.where(cat['IN_BOUNDARY'] == 1)
            x = cat['ALPHA_J2000'][q1] * np.cos(np.radians(cat['DELTA_J2000'][q1]))
            y = cat['DELTA_J2000'][q1]
            tree = spatial.KDTree(zip(x, y))
            check_segs.remove(seg)
            for check_seg in check_segs:
                ch_cat_name = args.main_path + check_seg + '/' + filt + '_clean.cat'
                ch_cat = Table.read(ch_cat_name, format='ascii.basic')
                q2, = np.where(ch_cat['IN_BOUNDARY'] == 1)
                x = ch_cat['ALPHA_J2000'][q2] * np.cos(np.radians(ch_cat['DELTA_J2000'][q2]))
                y = ch_cat['DELTA_J2000'][q2]
                pts = zip(x, y)
                s = tree.query(pts, distance_upper_bound=args.tolerance)
                ch_q, = np.where(s[0] != np.inf)
                if len(ch_q) == 0:
                    continue
                print "Detected duplicate objects: ", seg, check_seg
                q = s[1][ch_q]
                # If the object in check seg is worse
                cond1 = list(cat['FLAGS'][q] < ch_cat['FLAGS'][ch_q])
                cond2 = list(cat['SNR'][q] > ch_cat['SNR'][ch_q])
                multi, = np.where(cond1 or cond2)
                ids = np.array(ch_q[multi], dtype='int')
                ch_cat['MULTI_DET'][ids] = 1
                obj = [seg + '.' + str(cat['NUMBER'][q][num]) for num in multi]
                print multi, ids
                ch_cat['MULTI_DET_OBJ'][ids] = obj
                print ch_cat['MULTI_DET_OBJ'][ids]
                q = np.delete(q, multi)
                ch_q = np.delete(ch_q, multi)
                # object in  seg is worse
                cat['MULTI_DET'][q] = 1
                obj = [check_seg + '.' + str(ch_cat['NUMBER'][c]) for c in ch_q]
                print q, obj
                cat['MULTI_DET_OBJ'][q] = obj
                ch_cat.write(ch_cat_name, format='ascii.basic')
            cat.write(cat_name, format='ascii.basic')


if __name__ == '__main__':
    import numpy as np
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument('--filter_names', default=['f606w', 'f814w'],
                        help="names of filters [Default: ['f606w','f814w']]")
    parser.add_argument('--main_path',
                        default='/nfs/slac/g/ki/ki19/deuce/AEGIS/AEGIS_catalog_full/',
                        help="Path to where you want the images are stored \
                        [Default: /nfs/slac/g/ki/ki19/deuce/AEGIS/AEGIS_catalog_full]")
    parser.add_argument('--seg_file_name',
                        default ='/nfs/slac/g/ki/ki19/deuce/AEGIS/unzip/seg_ids.txt',
                        help="File with list of all all segment id names")
    parser.add_argument('--tolerance', type=float, default=1 / 1800.,
                        help="tolerance (in degrees) allowed while comparing \
                        objects in 2 seg fields [Default:1/1800.]")
    args = parser.parse_args()
    main(args)
