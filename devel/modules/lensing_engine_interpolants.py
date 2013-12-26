#!/usr/bin/env python
"""
A script for checking the effects of interpolating gridded shears on the output correlation function
/ power spectrum.  In terms of the power spectrum, we expect behavior that is effectively
multiplying by the square of the Fourier transform of the interpolant, but we check this directly,
and also check the impact on the correlation function.
"""

import galsim
import numpy as np
import os
import subprocess
import pyfits

# Set some important quantities up top:
# Which interpolants do we want to test?
interpolant_list = ['linear', 'cubic', 'quintic', 'nearest']
# Define shear grid
grid_size = 5. # degrees
ngrid = 50 # grid points in nominal grid
kmin_factor = 3 # factor by which to have the lensing engine internally expand the grid, to get
                # large-scale shear correlations.
# Define shear power spectrum file
pk_file = os.path.join('..','..','examples','data','cosmo-fid.zmed1.00.out')
# Define binning for PS / corr func.
default_n_output_bins = 12
tmp_cf_file = 'tmp.cf.dat'
# Set defaults for command-line arguments
## subsampling factor for the finer grid for which we wish to test results.
default_subsampling = 5
## number of realizations to run.
default_n = 100

# Utility functions go here, above main():
def getCF(x, y, g1, g2, dtheta, ngrid, n_output_bins):
    """Routine to estimate shear correlation functions using corr2.

    This routine takes information about positions and shears, and writes to temporary FITS files
    before calling the corr2 executable to get the shear correlation functions.  We read the results
    back in and return them.

    Arguments:

        x --------------- x grid position (1d NumPy array).

        y --------------- y grid position (1d NumPy array).

        g1 -------------- g1 (1d NumPy array).

        g2 -------------- g2 (1d NumPy array).

        ngrid ----------- Linear array size (i.e., number of points in each dimension).

        dtheta ---------- Array spacing, in degrees.

        n_output_bins --- Number of bins for calculation of correlatio function.

    """
    # Basic sanity checks of inputs.
    assert x.shape == y.shape
    assert x.shape == g1.shape
    assert x.shape == g2.shape

    # Set up a FITS table for output file.
    x_col = pyfits.Column(name='x', format='1D', array=x)
    y_col = pyfits.Column(name='y', format='1D', array=y)
    g1_col = pyfits.Column(name='g1', format='1D', array=g1)
    g2_col = pyfits.Column(name='g2', format='1D', array=g2)
    cols = pyfits.ColDefs([x_col, y_col, g1_col, g2_col])
    table = pyfits.new_table(cols)
    phdu = pyfits.PrimaryHDU()
    hdus = pyfits.HDUList([phdu,table])
    hdus.writeto('temp.fits',clobber=True)
    
    # Define some variables that corr2 needs: range of separations to use.
    min_sep = dtheta
    max_sep = ngrid * np.sqrt(2) * dtheta
    subprocess.Popen(['corr2','corr2.params',
                      'file_name=temp.fits',
                      'e2_file_name=%s'%tmp_cf_file,
                      'min_sep=%f'%min_sep,'max_sep=%f'%max_sep,
                      'nbins=%f'%n_output_bins]).wait()
    results = np.loadtxt(tmp_cf_file)
    os.remove('temp.fits')

    # Parse and return results
    r = results[:,0]
    xip = results[:,2]
    xip_err = results[:,6]
    return r, xip, xip_err

def main(n_realizations=default_n, subsampling=default_subsampling,
         n_output_bins=default_n_output_bins):
    """Main routine to drive all tests.

    Arguments:

        n_realizations ----- Number of random realizations of each shear field.

        subsampling -------- Factor by which to subsample the default grid.

        n_output_bins ------ Number of bins for calculation of 2-point functions.
    """
    # Set up PowerSpectrum object.
    ps = galsim.PowerSpectrum(pk_file, units = galsim.radians)

    # Loop over interpolants.
    for interpolant in interpolant_list:
        print "Beginning tests for interpolant %r:"%interpolant
        print "  Generating %d realizations..."%n_realizations

        # Initialize arrays for two-point functions.
        mean_interpolated_ps = np.zeros(n_output_bins)
        mean_ps = np.zeros(n_output_bins)
        mean_interpolated_cf = np.zeros(n_output_bins)
        mean_cf = np.zeros(n_output_bins)

        # Loop over realizations.
        for i_real in range(n_realizations):

            # Get shears on default grid.
            default_grid_spacing = grid_size / ngrid
            default_ngrid = ngrid
            # Note: kmax_factor = subsampling is given, so that when we rerun this later with the
            # subsampled grid, the same range of k will be used in both cases, and the only
            # difference is the interpolation.
            ps.buildGrid(grid_spacing = default_grid_spacing,
                         ngrid = default_ngrid,
                         units = galsim.degrees,
                         interpolant = interpolant,
                         kmin_factor = kmin_factor,
                         kmax_factor = subsampling)

            # Set up subsampled grid.
            subsampled_grid_spacing = grid_size / (ngrid*subsampling)
            subsampled_ngrid = ngrid*subsampling
            min = (-subsampled_ngrid/2 + 0.5) * subsampled_grid_spacing
            max = (subsampled_ngrid/2 - 0.5) * subsampled_grid_spacing
            x, y = np.meshgrid(
                np.arange(min,max+subsampled_grid_spacing,subsampled_grid_spacing),
                np.arange(min,max+subsampled_grid_spacing,subsampled_grid_spacing))
            x = list(x.flatten())
            y = list(y.flatten())

            # Interpolate shears on the subsampled grid.
            interpolated_g1, interpolated_g2 = ps.getShear(pos=(x,y),
                                                           units=galsim.degrees)
            # And put back into the format that the PowerSpectrumEstimator will want.
            interpolated_g1 = np.array(interpolated_g1).reshape((subsampled_ngrid,
                                                                 subsampled_ngrid))
            interpolated_g2 = np.array(interpolated_g2).reshape((subsampled_ngrid,
                                                                 subsampled_ngrid))

            # Directly get shears on subsampled grid.
            g1, g2 = ps.buildGrid(grid_spacing = subsampled_grid_spacing,
                                  ngrid = subsampled_ngrid,
                                  units = galsim.degrees,
                                  kmin_factor = kmin_factor)

            # Get statistics: PS, correlation function.
            # Set up PowerSpectrumEstimator first.
            if i_real == 0:
                pse = galsim.pse.PowerSpectrumEstimator(subsampled_ngrid,
                                                        grid_size,
                                                        n_output_bins)            
            ell, interpolated_ps_ee, _, _ = \
                pse.estimate(interpolated_g1, interpolated_g2)
            ell, ps_ee, _, _ = pse.estimate(g1, g2)
            x = np.array(x)
            y = np.array(y)
            th, interpolated_cf, cf_err = getCF(x, y, interpolated_g1.flatten(),
                                                interpolated_g2.flatten(),
                                                subsampled_grid_spacing,
                                                subsampled_ngrid,
                                                n_output_bins)
            _, cf, _ = getCF(x, y, g1.flatten(), g2.flatten(),
                             subsampled_grid_spacing, subsampled_ngrid,
                             n_output_bins)

            # Accumulate statistics.
            mean_interpolated_ps += interpolated_ps_ee
            mean_ps += ps_ee
            mean_interpolated_cf += interpolated_cf
            mean_cf += cf

        # Now get the average over all realizations
        print "Done generating realizations, now getting mean 2-point functions"
        mean_interpolated_ps /= n_realizations
        mean_ps /= n_realizations
        mean_interpolated_cf /= n_realizations
        mean_cf /= n_realizations

        # Plot statistics, and ratios with vs. without interpolants.
        print "Running plotting routines for interpolant=%s..."%interpolant
        generate_ps_plots(mean_ps, mean_interpolated_ps, interpolant)
        generate_cf_plots(mean_cf, mean_interpolated_cf, interpolant)

        # Output results.
        print "Outputting tables of results..."
        write_ps_output(mean_ps, mean_interpolated_ps, interpolant)
        write_cf_output(mean_cf, mean_interpolated_cf, interpolant)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        description='Run tests of GalSim interpolants on lensing engine outputs.')
    parser.add_argument('-n','--number-of-realizations', 
                        help='Number of objects to run tests on (default: %i)'%default_n,
                        default=default_n, type=int, dest='n_realizations')
    parser.add_argument('-s','--subsampling', 
                        help='Subsampling factor to test (default: %i)'%default_subsampling,
                        default=default_subsampling, type=int, dest='subsampling')
    parser.add_argument('-n_out','--n_output_bins',
                        help='Number of bins for calculating 2-point functions '
                        '(default: %i)'%default_n_output_bins,
                        default=default_n_output_bins, type=int,
                        dest='n_output_bins')
    args = parser.parse_args()
    main(n_realizations=args.n_realizations,
         subsampling=args.subsampling,
         n_output_bins=args.n_output_bins)
