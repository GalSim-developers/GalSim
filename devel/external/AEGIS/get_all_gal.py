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

import numpy as np
from astropy.table import Table, vstack, join, Column


def get_all_gal(args):
    """Makes the parent catalog of galaxies by combining galaxies in all
    segments and writng it to a single file. The individual galaxies are
    taken from the *_with_pstamp.fits file for each segment. This file includes
    only objects that were detected in the Hot-Cold method with SExtractor,
    classified as galaxy in all bands, and not in any of the masked regions.
    """
    index_table = Table.read(args.index_table_file, format='fits')
    all_seg_ids = np.loadtxt(args.seg_list_file, delimiter=" ", dtype='S2')
    for f, filt in enumerate(args.filter_names):
        complete_table = Table()
        for seg_id in all_seg_ids:
            file_name = args.main_path + seg_id + '/' + filt + '_with_pstamp.fits'
            seg_cat = Table.read(file_name, format='fits')
            col = Column([seg_id] * len(seg_cat), name='SEG_ID')
            seg_cat.add_column(col)
            q, = np.where(index_table['SEG_ID'] == seg_id)
            indx_seg = index_table[q]
            temp = join(seg_cat, indx_seg, keys=['NUMBER', 'SEG_ID'],
                        join_type='outer')
            col = Column(temp['MAG_CORR'], name='SEG_ID')
            temp.add_column(col, name='MAG')
            complete_table = vstack([complete_table, temp])
        path = args.main_path
        cat_name = path + args.cat_name.replace('filter', args.file_filter_name[f])
        print "Savings fits file at ", cat_name
        complete_table.sort('ORDER')
        complete_table.write(cat_name, format='fits',
                             overwrite=True)


if __name__ == '__main__':
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument('--filter_names', default=['f606w','f814w'],
                        help="names of filters [Default: ['f606w','f814w']]")
    parser.add_argument('--file_filter_name', default=['V', 'I'],
                        help="Name of filter to use ")
    parser.add_argument('--main_path',
                        default='/nfs/slac/g/ki/ki19/deuce/AEGIS/AEGIS_catalog_full/')
    parser.add_argument('--seg_list_file', default='/nfs/slac/g/ki/ki19/deuce/AEGIS/unzip/seg_ids.txt',
                        help="file with all seg id names" )
    parser.add_argument('--index_table_file', default='/nfs/slac/g/ki/ki19/deuce/AEGIS/AEGIS_catalog_full/index_AEGIS_galaxy_catalog_all_25.2.fits',
                        help="file with all seg id names")
    parser.add_argument('--cat_name', default='all_AEGIS_galaxy_filter_25.2.fits',
                        help="name of catalog with all galxies")
    args = parser.parse_args()
    get_all_gal(args)
