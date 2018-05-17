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

"""Program 7
Script to add WEIGHT factors to galaxies to account for any selection bias
when making the final catalog. The fraction of galaxies that are included in
the final catalog compared to the parent catalog was measured as a function
of several parameters like FLUX_RADIUS, ELLIPTICITY, MAGNITUDE etc., to check
for selection bias. The distribution was observed to show a dependance on the
half light radius of the galaxy measured by SExtractor. The weight is defined
as the ratio of the number of galaxies in the parent sample to the number in
the final catalog, at a given half light radius. Due to the small number of
galaxies at high half light radius in the parent sample, this definition
breaks down. Thus weight for large galaxies is set to 1, and the weights of
other galaxies were renormalized.
Since the magnitude cut in the parent sample was placed in the I band, the
weight for each galxy was computed in this band and set the same for all other
bands.
"""


def get_efficiency_with_error(val_s, val_p,
                              bins=10):
    """ Returns efficiency=(# in selection)/(# in parent)
    @input param
    val_s: Distribution in selection sample
    val_p: Distribution in parent sample
    returns efficiency of selection and error on efficiency
    """
    num_p, bins2 = np.histogram(val_p, bins=bins)
    num_s, bins1 = np.histogram(val_s, bins=bins)
    eff = num_s / np.array(num_p, dtype=float)
    err = (num_s * (1 - eff))**0.5 / num_p
    return eff, err


def main(args):
    """Compute weight for galaxies in args.meas_filter band and save it the
    main catalogs, args.cat_name, in all bands. The half light information is
    stored in the args.fits_file_name files.
    """
    parent_name = args.parent_file_name.replace('filter', args.meas_filter)
    parent_cat = Table.read(args.main_path + parent_name,
                            format='fits')
    select_name = args.fits_file_name.replace('filter', args.meas_filter)
    select_cat = Table.read(args.main_path + args.out_dir + select_name,
                            format='fits')
    # error in efficiency was high beyond half light radius of 55 pixels
    # function is computed only till there
    bins = np.linspace(0, args.max_hlr, 10)
    eff, err = get_efficiency_with_error(select_cat['flux_radius'],
                                         parent_cat['FLUX_RADIUS'],
                                         bins=bins)
    hlr = 0.5 * (bins[1:] + bins[:-1])
    z = np.polyfit(hlr, eff, 3)
    p = np.poly1d(z)
    norm = p(args.max_hlr)
    eff_new = p(select_cat['flux_radius'])
    weight = 1 / eff_new * norm
    # Set weight to 1 for large galaxies
    q, = np.where(select_cat['flux_radius'] > args.max_hlr)
    weight[q] = 1.0
    for f, filt in enumerate(args.file_filter_name):
        cat_name = args.main_path + args.out_dir + args.cat_name.replace('filter', filt)
        cat = Table.read(cat_name,
                         format='fits')
        cat['WEIGHT'] = weight
        print "Savings fits file at ", cat_name
        cat.write(cat_name, format='fits',
                  overwrite=True)


if __name__ == '__main__':
    from argparse import ArgumentParser
    import numpy as np
    from astropy.table import Table
    parser = ArgumentParser()
    parser.add_argument('--meas_filter', default='I',
                        help="Name of filter to compute weights for [Default:'I']")
    parser.add_argument('--max_hlr', default=55,
                        help="maximum half light radius (in pixels) to which\
                        weight definition is valid [Default:55]")
    parser.add_argument('--file_filter_name', default=['V', 'I'],
                        help="Name of filter to use ")
    parser.add_argument('--main_path',
                        default='/nfs/slac/g/ki/ki19/deuce/AEGIS/AEGIS_catalog_full/')
    parser.add_argument('--out_dir', default="AEGIS_training_sample/",
                        help="directory containing the final catalog")
    parser.add_argument('--cat_name', default="AEGIS_galaxy_catalog_filter_25.2.fits",
                        help="Final catalog name")
    parser.add_argument('--fits_file_name', default="AEGIS_galaxy_catalog_filter_25.2_fits.fits",
                        help="Name of Catalog with fit information")
    parser.add_argument('--parent_file_name', default="all_AEGIS_galaxy_filter_25.2.fits",
                        help="Name of file with the parent catalog.")
    args = parser.parse_args()
    main(args)
