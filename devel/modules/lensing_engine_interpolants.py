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
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties

# Set some important quantities up top:
# Which interpolants do we want to test?
interpolant_list = ['linear', 'cubic', 'quintic', 'sinc']
# Define shear grid
grid_size = 10. # degrees
ngrid = 100 # grid points in nominal grid
kmin_factor = 3 # factor by which to have the lensing engine internally expand the grid, to get
                # large-scale shear correlations.
# Define shear power spectrum file
pk_file = os.path.join('..','..','examples','data','cosmo-fid.zmed1.00.out')
# Define files for PS / corr func.
tmp_cf_file = 'tmp.cf.dat'
# Set defaults for command-line arguments
## grid offsets: 'random' (i.e., random sub-pixel amounts) or a specific fraction of a pixel
default_dithering = 'random'
## number of realizations to run.
default_n = 100
## number of bins for output PS / corrfunc
default_n_output_bins = 12
## output prefix for power spectrum, correlation function plots
default_ps_plot_prefix = "plots/interpolated_ps_"
default_cf_plot_prefix = "plots/interpolated_cf_"

# make small fonts possible in legend
fontP = FontProperties()
fontP.set_size('small')

# Utility functions go here, above main():
def check_dir(dir):
    """Utility to make an output directory if necessary.

    Arguments:

      dir ----- Desired output directory.

    """
    try:
        os.makedirs(dir)
        print "Created directory %s for outputs"%dir
    except OSError:
        if os.path.isdir(dir):
            # It was already a directory.
            pass
        else:
            # There was a problem, so exist.
            raise

def ft_interp(uvals, interpolant):
    """Utility to calculate the Fourier transform of an interpolant.

    The uvals are not the standard k or ell, but rather cycles per interpolation unit.

    If we do not have a form available for a particular interpolant, this will raise a
    NotImplementedError.

    Arguments:

      uvals --------- NumPy array of u values at which to get the FT of the interpolant.

      interpolant --- String specifying the interpolant.

    """
    s = np.sinc(uvals)

    if interpolant=='linear':
        return s**2
    elif interpolant=='nearest':
        return s
    elif interpolant=='cubic':
        c = np.cos(np.pi*uvals)
        return (s**3)*(3.*s-2.*c)
    elif interpolant=='quintic':
        piu = np.pi*uvals
        c = np.cos(piu)
        piusq = piu**2
        return (s**5)*(s*(55.-19.*piusq) + 2.*c*(piusq-27.))
    else:
        raise NotImplementedError


def generate_ps_plots(ell, ps, interpolated_ps, interpolant, ps_plot_prefix,
                      dth, type='EE'):
    """Routine to make power spectrum plots and write them to file.

    This routine makes a two-panel plot, with the first panel showing the two power spectra,
    and the second showing their ratio.

    Arguments:

      ell ----------------- Wavenumber k (flat-sky version of ell) in 1/radians.

      ps ------------------ Actual power spectrum, in radians^2.

      interpolated_ps ----- Power spectrum including effects of the interpolant.

      interpolant --------- Which interpolant was used?

      ps_plot_prefix ------ Prefix to use for power spectrum plots.

      dth ----------------- Grid spacing (degrees).

      type ---------------- Type of power spectrum?  Options are 'EE', 'BB', 'EB'.  This affects the
                            choice of plots to make.

    """
    # Sanity checks on acceptable inputs
    assert ell.shape == ps.shape
    assert ell.shape == interpolated_ps.shape

    # Calculate maximum k values, in 1/radians
    # kmax = pi / (theta in radians)
    #      = pi / (2pi * (theta in degrees) / 180.)
    #      = 90 / (theta in degrees)
    kmax = 90. / dth

    # Set up plot
    fig = plt.figure()
    # Set up first panel with power spectra.
    if type=='EE':
        ax = fig.add_subplot(211)
    else:
        ax = fig.add_subplot(111)
    ax.plot(ell, ell*(ell+1)*ps/(2.*np.pi), color='b', label='Power spectrum')
    ax.plot(ell, ell*(ell+1)*interpolated_ps/(2.*np.pi), color='r',
            label='Interpolated')
    kmax_x_markers = np.array((kmax, kmax))
    if type=='EE':
        kmax_y_markers = np.array((
                min(np.min((ell**2)*ps[ps>0]/(2.*np.pi)),
                    np.min((ell**2)*interpolated_ps[interpolated_ps>0]/(2.*np.pi))),
                max(np.max((ell**2)*ps/(2.*np.pi)),
                    np.max((ell**2)*interpolated_ps/(2.*np.pi)))))
    else:
        kmax_y_markers = np.array((
                min(np.min((ell**2)*ps/(2.*np.pi)),
                    np.min((ell**2)*interpolated_ps/(2.*np.pi))),
                max(np.max((ell**2)*ps/(2.*np.pi)),
                    np.max((ell**2)*interpolated_ps/(2.*np.pi)))))
    ax.plot(kmax_x_markers, kmax_y_markers, '--', color='k', label='Grid kmax')
    ax.set_ylabel('Dimensionless %s power'%type)
    ax.set_title('Interpolant: %s'%interpolant)
    ax.set_xscale('log')
    if type=='EE':
        ax.set_yscale('log')
    plt.legend(loc='lower left', prop=fontP)

    if type=='EE':
        # Set up second panel with ratio.
        ax = fig.add_subplot(212)
        ratio = interpolated_ps/ps
        ax.plot(ell, ratio, color='k')
        fine_ell = np.arange(20000.)
        fine_ell = fine_ell[(fine_ell > np.min(ell)) & (fine_ell < np.max(ell))]
        u = fine_ell*big_dth/180./np.pi # check factors of pi and so on; the big_dth is needed to
        # convert from actual distances to "interpolation units"
        try:
            theor_ratio = (ft_interp(u, interpolant))**2
            ax.plot(fine_ell, theor_ratio, '--', color='g', label='Theory prediction')
        except:
            print "Could not get theoretical prediction for interpolant %s"%interpolant
        ax.plot(kmax_x_markers, np.array((np.min(ratio), 1.3*np.max(ratio))), '--',
                color='k')
        ax.plot(ell, np.ones_like(ell), '--', color='r')
        ax.set_xlabel('ell [1/radians]')
        ax.set_ylabel('Interpolated / direct power')
        ax.set_xscale('log')
        plt.legend(loc='upper right', prop=fontP)

    # Write to file.
    outfile = ps_plot_prefix + interpolant + '_' + type + '.png'
    plt.savefig(outfile)
    print "Wrote power spectrum plots to file %r"%outfile

def generate_cf_plots(th, cf, interpolated_cf, interpolant, cf_plot_prefix,
                      dth, type='p'):
    """Routine to make correlation function plots and write them to file.

    This routine makes a two-panel plot, with the first panel showing the two correlation functions,
    and the second showing their ratio.

    Arguments:

      th ------------------ Angle theta (separation on sky), in degrees.

      cf ------------------ Correlation function xi_+ (dimensionless).

      interpolated_cf ----- Correlation function including effects of the interpolant.

      interpolant --------- Which interpolant was used?

      cf_plot_prefix ------ Prefix to use for correlation function plots.

      dth ----------------- Grid spacing (degrees).

      type ---------------- Type of correlation function?  Options are 'p' and 'm' for xi_+ and
                            xi_-.
    """
    # Sanity checks on acceptable inputs
    assert th.shape == cf.shape
    assert th.shape == interpolated_cf.shape

    # Set up 2-panel plot
    fig = plt.figure()
    # Set up first panel with power spectra.
    ax = fig.add_subplot(211)
    ax.plot(th, cf, color='b', label='Correlation function')
    ax.plot(th, interpolated_cf, color='r',
            label='Interpolated')
    dth_x_markers = np.array((dth, dth))
    dth_y_markers = np.array(( min(np.min(cf[cf>0]),
                                   np.min(interpolated_cf[interpolated_cf>0])),
                               2*max(np.max(cf), np.max(interpolated_cf))))
    ax.plot(dth_x_markers, dth_y_markers, '--', color='k', label='Grid spacing')
    if type=='p':
        ax.set_ylabel('xi_+')
    else:
        ax.set_ylabel('xi_-')
    ax.set_title('Interpolant: %s'%interpolant)
    ax.set_xscale('log')
    ax.set_yscale('log')
    plt.legend(loc='upper right', prop=fontP)

    # Set up second panel with ratio.
    ax = fig.add_subplot(212)
    ratio = interpolated_cf/cf
    ax.plot(th, ratio, color='k')
    ax.plot(dth_x_markers, np.array((np.min(ratio), 1.3*np.max(ratio))), '--', color='k')
    ax.plot(th, np.ones_like(th), '--', color='r')
    ax.set_ylim(0.0,1.6)
    ax.set_xlabel('Separation [degrees]')
    ax.set_ylabel('Interpolated / direct xi')
    ax.set_xscale('log')

    # Write to file.
    outfile = cf_plot_prefix + interpolant + '_' + type + '.png'
    plt.savefig(outfile)
    print "Wrote correlation function plots to file %r"%outfile

def write_ps_output(ell, ps, interpolated_ps, interpolant, ps_plot_prefix, type='EE'):
    """Routine to write final power spectra to file.

    This routine makes two output files: one ascii (.dat) and one FITS table (.fits).

    Arguments:

      ell ----------------- Wavenumber k (flat-sky version of ell) in 1/radians.

      ps ------------------ Actual power spectrum, in radians^2.

      interpolated_ps ----- Power spectrum including effects of the interpolant.

      interpolant --------- Which interpolant was used?

      ps_plot_prefix ------ Prefix to use for power spectrum data.

      type ---------------- Type of power spectrum?  (e.g., 'EE', 'BB', etc.)

    """
    # Make ascii output.
    outfile = ps_plot_prefix + interpolant + '_' + type + '.dat'
    np.savetxt(outfile, np.column_stack((ell, ps, interpolated_ps)), fmt='%12.4e')
    print "Wrote ascii output to file %r"%outfile

    # Set up a FITS table for output file.
    ell_col = pyfits.Column(name='ell', format='1D', array=ell)
    ps_col = pyfits.Column(name='ps', format='1D', array=ps)
    interpolated_ps_col = pyfits.Column(name='interpolated_ps', format='1D',
                                        array=interpolated_ps)
    cols = pyfits.ColDefs([ell_col, ps_col, interpolated_ps_col])
    table = pyfits.new_table(cols)
    phdu = pyfits.PrimaryHDU()
    hdus = pyfits.HDUList([phdu,table])
    outfile = ps_plot_prefix + interpolant + '_' + type + '.fits'
    hdus.writeto(outfile, clobber=True)
    print "Wrote FITS output to file %r"%outfile

def write_cf_output(th, cf, interpolated_cf, interpolant, cf_plot_prefix, type='p'):
    """Routine to write final correlation functions to file.

    This routine makes two output files: one ascii (.dat) and one FITS table (.fits).

    Arguments:

      th ------------------ Angle theta (separation on sky), in degrees.

      cf ------------------ Correlation function xi_+ (dimensionless).

      interpolated_cf ----- Correlation function including effects of the interpolant.

      interpolant --------- Which interpolant was used?

      cf_plot_prefix ------ Prefix to use for correlation function plots.

      type ---------------- Type of correlation function?  Options are 'p' and 'm' for xi_+ and
                            xi_-.
    """
    # Make ascii output.
    outfile = cf_plot_prefix + interpolant + '_' + type + '.dat'
    np.savetxt(outfile, np.column_stack((th, cf, interpolated_cf)), fmt='%12.4e')
    print "Wrote ascii output to file %r"%outfile

    # Set up a FITS table for output file.
    th_col = pyfits.Column(name='theta', format='1D', array=th)
    cf_col = pyfits.Column(name='xip', format='1D', array=cf)
    interpolated_cf_col = pyfits.Column(name='interpolated_xip', format='1D',
                                        array=interpolated_cf)
    cols = pyfits.ColDefs([th_col, cf_col, interpolated_cf_col])
    table = pyfits.new_table(cols)
    phdu = pyfits.PrimaryHDU()
    hdus = pyfits.HDUList([phdu,table])
    outfile = cf_plot_prefix + interpolant + '_' + type + '.fits'
    hdus.writeto(outfile, clobber=True)
    print "Wrote FITS output to file %r"%outfile

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
    subprocess.Popen(['/Users/rmandelb/svn/mjarvis-read-only/corr2','corr2.params',
                      'file_name=temp.fits',
                      'e2_file_name=%s'%tmp_cf_file,
                      'min_sep=%f'%min_sep,'max_sep=%f'%max_sep,
                      'nbins=%f'%n_output_bins]).wait()
    results = np.loadtxt(tmp_cf_file)
    os.remove('temp.fits')

    # Parse and return results
    r = results[:,0]
    xip = results[:,2]
    xim = results[:,3]
    xip_err = results[:,6]
    return r, xip, xim, xip_err

def main(n_realizations, dithering, n_output_bins, ps_plot_prefix, cf_plot_prefix):
    """Main routine to drive all tests.

    Arguments:

        n_realizations ----- Number of random realizations of each shear field.

        dithering ---------- Sub-pixel dither to apply to the grid onto which we interpolate, with
                             respect to the original grid on which shears are determined.  If
                             'random' (default) then each realization has a random (x, y) sub-pixel
                             offset.  If a string that converts to a float, then that float is used
                             to determine the x, y offsets for each realization.

        n_output_bins ------ Number of bins for calculation of 2-point functions.

        ps_plot_prefix ----- Prefix to use for power-spectrum outputs.

        cf_plot_prefix ----- Prefix to use for correlation function outputs.

    """
    # Get basic grid information
    grid_spacing = grid_size / ngrid

    # Set up PowerSpectrum object.  We have to be careful to watch out for aliasing due to our
    # initial P(k) including power on scales above those that can be represented by our grid.  In
    # order to deal with that, we will define a power spectrum function that equals a cosmological
    # one for k < k_max and is zero above that.  In principle, hard edges could cause a problem, but
    # we will cross (burn?) that bridge when we come to it.
    raw_ps_data = np.loadtxt(pk_file).transpose()
    raw_ps_k = raw_ps_data[0,:]
    raw_ps_p = raw_ps_data[1,:]
    # Find k_max, taking into account that grid spacing is in degrees and our power spectrum is
    # defined in radians. So
    #    k_max = pi / (grid_spacing in radians) = pi / [2 pi (grid_spacing in degrees) / 180]
    #          = 90 / (grid spacing in degrees)
    k_max = 90. / grid_spacing
    # Now define a power spectrum that is raw_ps below k_max and zero above that.
    # If hard edges are a problem, we could use some function that goes smoothly to zero instead.
    ps = galsim.PowerSpectrum(
        galsim.LookupTable(raw_ps_k, raw_ps_p*(raw_ps_k<k_max), interpolant='linear'),
        units = galsim.radians
        )

    # Set up grid and the corresponding x, y lists.
    min = (-ngrid/2 + 0.5) * grid_spacing
    max = (ngrid/2 - 0.5) * grid_spacing
    x, y = np.meshgrid(
        np.arange(min,max+grid_spacing,grid_spacing),
        np.arange(min,max+grid_spacing,grid_spacing))

    # Parse the input dithering
    if dithering != 'random':
        try:
            dithering = float(dithering)
        except:
            raise RuntimeError("Dithering should be 'random' or a float!")
        if dithering<0 or dithering>1:
            import warnings
            dithering = dithering % 1
            warnings.warn("Dithering converted to a value between 0-1: %f"%dithering)
        # Now, set up the dithered grid.  Will be (ngrid-1) x (ngrid-1) so as to not go off the
        # edge.
        dither_x = x[0:ngrid-1,0:ngrid-1] + dithering*grid_spacing
        dither_y = y[0:ngrid-1,0:ngrid-1] + dithering*grid_spacing
        dither_x = list(dither_x.flatten())
        dither_y = list(dither_y.flatten())
    else:
        # Just set up the uniform deviate.
        u = galsim.UniformDeviate()

    # Loop over interpolants.
    for interpolant in interpolant_list:
        print "Beginning tests for interpolant %r:"%interpolant
        print "  Generating %d realizations..."%n_realizations

        # Initialize arrays for two-point functions.
        mean_interpolated_ps_ee = np.zeros(n_output_bins)
        mean_ps_ee = np.zeros(n_output_bins)
        mean_interpolated_ps_bb = np.zeros(n_output_bins)
        mean_ps_bb = np.zeros(n_output_bins)
        mean_interpolated_ps_eb = np.zeros(n_output_bins)
        mean_ps_eb = np.zeros(n_output_bins)
        mean_interpolated_cfp = np.zeros(n_output_bins)
        mean_cfp = np.zeros(n_output_bins)
        mean_interpolated_cfm = np.zeros(n_output_bins)
        mean_cfm = np.zeros(n_output_bins)

        # Loop over realizations.
        for i_real in range(n_realizations):

            # Get shears on default grid.
            g1, g2 = ps.buildGrid(grid_spacing = grid_spacing,
                                  ngrid = ngrid,
                                  units = galsim.degrees,
                                  interpolant = interpolant,
                                  kmin_factor = kmin_factor)

            # Set up the grid for interpolation for this realization, if we're using random
            # dithers.  Will be (ngrid-1) x (ngrid-1) so as to not go off the edge.
            if dithering == 'random':
                dither_x = x[0:ngrid-1,0:ngrid-1] + u()*grid_spacing
                dither_y = y[0:ngrid-1,0:ngrid-1] + u()*grid_spacing
                dither_x = list(dither_x.flatten())
                dither_y = list(dither_y.flatten())

            # Interpolate shears on the offset grid.
            interpolated_g1, interpolated_g2 = ps.getShear(pos=(dither_x,dither_y),
                                                           units=galsim.degrees)
            # And put back into the format that the PowerSpectrumEstimator will want.
            interpolated_g1 = np.array(interpolated_g1).reshape((ngrid-1, ngrid-1))
            interpolated_g2 = np.array(interpolated_g2).reshape((ngrid-1, ngrid-1))

            # Get statistics: PS, correlation function. 
            # Set up PowerSpectrumEstimator first.  Reminder: for interpolated points, we need to
            # omit the last row, since the interpolated grid has gotten shifted outside the original
            # grid bounds, and therefore was set to zero.
            if i_real == 0:
                interpolated_pse = galsim.pse.PowerSpectrumEstimator(ngrid-1,
                                                                    grid_size,
                                                                    n_output_bins)           
                pse = galsim.pse.PowerSpectrumEstimator(ngrid,
                                                        grid_size,
                                                        n_output_bins)           
            ell, interpolated_ps_ee, interpolated_ps_bb, interpolated_ps_eb = \
                interpolated_pse.estimate(interpolated_g1,
                                          interpolated_g2)
            ell, ps_ee, ps_bb, ps_eb = pse.estimate(g1, g2)
            th, interpolated_cfp, interpolated_cfm, cf_err = \
                getCF(np.array(dither_x), np.array(dither_y),
                      interpolated_g1.flatten(),
                      interpolated_g2.flatten(), grid_spacing,
                      ngrid, n_output_bins)
            _, cfp, cfm, _ = \
                getCF(x.flatten(), y.flatten(), g1.flatten(), g2.flatten(),
                      grid_spacing, ngrid, n_output_bins)

            # Accumulate statistics.
            mean_interpolated_ps_ee += interpolated_ps_ee
            mean_ps_ee += ps_ee
            mean_interpolated_ps_bb += interpolated_ps_bb
            mean_ps_bb += ps_bb
            mean_interpolated_ps_eb += interpolated_ps_eb
            mean_ps_eb += ps_eb
            mean_interpolated_cfp += interpolated_cfp
            mean_cfp += cfp
            mean_interpolated_cfm += interpolated_cfm
            mean_cfm += cfm

        # Now get the average over all realizations
        print "Done generating realizations, now getting mean 2-point functions"
        mean_interpolated_ps_ee /= n_realizations
        mean_ps_ee /= n_realizations
        mean_interpolated_ps_bb /= n_realizations
        mean_ps_bb /= n_realizations
        mean_interpolated_ps_eb /= n_realizations
        mean_ps_eb /= n_realizations
        mean_interpolated_cfp /= n_realizations
        mean_cfp /= n_realizations
        mean_interpolated_cfm /= n_realizations
        mean_cfm /= n_realizations

        # Plot statistics, and ratios with vs. without interpolants.
        print "Running plotting routines for interpolant=%s..."%interpolant
        generate_ps_plots(ell, mean_ps_ee, mean_interpolated_ps_ee, interpolant, ps_plot_prefix,
                          grid_spacing, type='EE')
        generate_ps_plots(ell, mean_ps_bb, mean_interpolated_ps_bb, interpolant, ps_plot_prefix,
                          grid_spacing, type='BB')
        generate_ps_plots(ell, mean_ps_eb, mean_interpolated_ps_eb, interpolant, ps_plot_prefix,
                          grid_spacing, type='EB')
        generate_cf_plots(th, mean_cfp, mean_interpolated_cfp, interpolant, cf_plot_prefix,
                          grid_spacing, type='p')
        generate_cf_plots(th, mean_cfm, mean_interpolated_cfm, interpolant, cf_plot_prefix,
                          grid_spacing, type='m')

        # Output results.
        print "Outputting tables of results..."
        write_ps_output(ell, mean_ps_ee, mean_interpolated_ps_ee, interpolant, ps_plot_prefix,
                        type='EE')
        write_ps_output(ell, mean_ps_bb, mean_interpolated_ps_bb, interpolant, ps_plot_prefix,
                        type='BB')
        write_ps_output(ell, mean_ps_eb, mean_interpolated_ps_eb, interpolant, ps_plot_prefix,
                        type='EB')
        write_cf_output(th, mean_cfp, mean_interpolated_cfp, interpolant, cf_plot_prefix, type='p')
        write_cf_output(th, mean_cfm, mean_interpolated_cfm, interpolant, cf_plot_prefix, type='m')
        print ""

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        description='Run tests of GalSim interpolants on lensing engine outputs.')
    parser.add_argument('-n','--number-of-realizations', 
                        help='Number of objects to run tests on (default: %i)'%default_n,
                        default=default_n, type=int, dest='n_realizations')
    parser.add_argument('-s','--subsampling', 
                        help='Grid dithering to test (default: %s)'%default_dithering,
                        default=default_dithering, type=str, dest='dithering')
    parser.add_argument('-n_out','--n_output_bins',
                        help='Number of bins for calculating 2-point functions '
                        '(default: %i)'%default_n_output_bins,
                        default=default_n_output_bins, type=int,
                        dest='n_output_bins')
    parser.add_argument('--ps_plot_prefix',
                        help='Prefix for output power spectrum plots '
                        '(default: %s)'%default_ps_plot_prefix,
                        default=default_ps_plot_prefix, type=str,
                        dest='ps_plot_prefix')
    parser.add_argument('--cf_plot_prefix',
                        help='Prefix for output correlation function plots '
                        '(default: %s)'%default_cf_plot_prefix,
                        default=default_cf_plot_prefix, type=str,
                        dest='cf_plot_prefix')
    args = parser.parse_args()
    # Make directories if necessary.
    dir = os.path.dirname(args.ps_plot_prefix)
    if dir is not '':
        check_dir(dir)
    dir = os.path.dirname(args.cf_plot_prefix)
    if dir is not '':
        check_dir(dir)

    main(n_realizations=args.n_realizations,
         dithering=args.dithering,
         n_output_bins=args.n_output_bins,
         ps_plot_prefix=args.ps_plot_prefix,
         cf_plot_prefix=args.cf_plot_prefix)
