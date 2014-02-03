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
import optparse
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties

# Set some important quantities up top:
# Which interpolants do we want to test?
interpolant_list = ['linear', 'cubic', 'quintic', 'nearest']
# Define shear grid
grid_size = 10. # degrees
ngrid = 100 # grid points in nominal grid
kmin_factor = 3 # factor by which to have the lensing engine internally expand the grid, to get
                # large-scale shear correlations.
# Define shear power spectrum file
pk_file = os.path.join('..','..','examples','data','cosmo-fid.zmed1.00_smoothed.out')
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
def cutoff_func(k_ratio):
    """Softening function for the power spectrum cutoff, instead of a hard cutoff.

    The argument is the ratio of k to k_max for this grid.  We use an arctan function to go smoothly
    from 1 to 0 above k_max.
    """
    # The magic numbers in the code below come from the following:
    # We define the function as
    #     (arctan[A log(k/k_max) + B] + pi/2)/pi
    # For our current purposes, we will define A and B by requiring that this function go to 0.95
    # (0.05) for k/k_max = 0.95 (1).  This gives two equations:
    #     0.95 = (arctan[log(0.95) A + B] + pi/2)/pi
    #     0.05 = (arctan[B] + pi/2)/pi.
    # We will solve the second equation:
    #     -0.45 pi = arctan(B), or
    #     B = tan(-0.45 pi).
    b = np.tan(-0.45*np.pi)
    # Then, we get A from the first equation:
    #     0.45 pi = arctan[log(0.95) A + B]
    #     tan(0.45 pi) = log(0.95) A  + B
    a = (np.tan(0.45*np.pi)-b) / np.log(0.95)
    return (np.arctan(a*np.log(k_ratio)+b) + np.pi/2.)/np.pi

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

def generate_ps_cutoff_plots(ell, ps, theory_ps,
                             nocutoff_ell, nocutoff_ps, nocutoff_theory_ps,
                             interpolant, ps_plot_prefix, type='EE'):
    """Routine to make power spectrum plots for edge cutoff vs. not (in both cases, without
       interpolation) and write them to file.

    Arguments:

      ell ----------------- Wavenumber k (flat-sky version of ell) in 1/radians, after cutting off
                            grid edges.

      ps ------------------ Power spectrum in radians^2 after cutting off grid edges.

      theory_ps ----------- Rebinned theory power spectrum after cutting off grid edges.

      nocutoff_ell -------- Wavenumber k (flat-sky version of ell) in 1/radians, before cutting off
                            grid edges.

      nocutoff_ps --------- Power spectrum in radian^2 before cutting off grid edges.

      nocutoff_theory_ps -- Rebinned theory power spectrum before cutting off grid edges.

      interpolant --------- Which interpolant was used?

      ps_plot_prefix ------ Prefix to use for power spectrum plots.

      type ---------------- Type of power spectrum?  Options are 'EE', 'BB', 'EB'.

    """
    # Sanity checks on acceptable inputs
    assert ell.shape == ps.shape
    assert ell.shape == theory_ps.shape
    assert nocutoff_ell.shape == nocutoff_ps.shape
    assert nocutoff_ell.shape == nocutoff_theory_ps.shape

    # Set up plot
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(ell, ell*(ell+1)*ps/(2.*np.pi), color='b', label='Power spectrum')
    ax.plot(ell, ell*(ell+1)*theory_ps/(2.*np.pi), color='c', label='Theory power spectrum')
    ax.plot(nocutoff_ell, nocutoff_ell*(nocutoff_ell+1)*nocutoff_ps/(2.*np.pi),
            color='r', label='Power spectrum without cutoff')
    ax.plot(nocutoff_ell, nocutoff_ell*(nocutoff_ell+1)*nocutoff_theory_ps/(2.*np.pi),
            color='m', label='Theory power spectrum without cutoff')
    if type=='EE':
        ax.set_yscale('log')
    plt.legend(loc='upper right', prop=fontP)

    # Write to file.
    outfile = ps_plot_prefix + interpolant + '_grid_cutoff_' + type + '.png'
    plt.savefig(outfile)
    print "Wrote power spectrum (grid cutoff) plot to file %r"%outfile


def generate_ps_plots(ell, ps, interpolated_ps, interpolant, ps_plot_prefix,
                      dth, type='EE'):
    """Routine to make power spectrum plots and write them to file.

    This routine makes a two-panel plot, with the first panel showing the two power spectra,
    and the second showing their ratio.

    Arguments:

      ell ----------------- Wavenumber k (flat-sky version of ell) in 1/radians.

      ps ------------------ Actual power spectrum for original grid, pre-interpolation,, in
                            radians^2.

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
        u = fine_ell*dth/180./np.pi # check factors of pi and so on; the dth is needed to
        # convert from actual distances to "interpolation units"
        try:
            theor_ratio = (ft_interp(u, interpolant))**2
            ax.plot(fine_ell, theor_ratio, '--', color='g', label='|FT interpolant|^2')
        except:
            print "Could not get theoretical prediction for interpolant %s"%interpolant
        ax.plot(kmax_x_markers, np.array((np.min(ratio), np.max(ratio))), '--',
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

def generate_cf_cutoff_plots(th, cf, nocutoff_th, nocutoff_cf, interpolant, cf_plot_prefix,
                             type='p'):
    """Routine to make correlation function plots for edge cutoff vs. not (in both cases, without
       interpolation) and write them to file.

    Arguments:

      th ------------------ Angle theta (separation on sky), in degrees, after grid cutoff

      cf ------------------ Correlation function xi_+ (dimensionless), after grid cutoff.

      nocutoff_th --------- Angle theta (separation on sky), in degrees, before grid cutoff

      nocutoff_cf --------- Correlation function xi_+ (dimensionless), before grid cutoff.

      interpolant --------- Which interpolant was used?

      cf_plot_prefix ------ Prefix to use for correlation function plots.

      type ---------------- Type of correlation function?  Options are 'p' and 'm' for xi_+ and
                            xi_-.
    """
    # Sanity checks on acceptable inputs
    assert th.shape == cf.shape
    assert nocutoff_th.shape == nocutoff_cf.shape

    # Set up plot
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(th, cf, color='b', label='Correlation function')
    ax.plot(nocutoff_th, nocutoff_cf, color='r',
            label='Correlation function with grid cutoff')
    if type=='p':
        ax.set_ylabel('xi_+')
    else:
        ax.set_ylabel('xi_-')
    ax.set_title('Interpolant: %s'%interpolant)
    ax.set_xscale('log')
    ax.set_yscale('log')
    plt.legend(loc='upper right', prop=fontP)

    # Write to file.
    outfile = cf_plot_prefix + interpolant + '_grid_cutoff_' + type + '.png'
    plt.savefig(outfile)
    print "Wrote correlation function (grid cutoff) plots to file %r"%outfile

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
    ax.set_ylim(0.85,1.15)
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
    xim = results[:,3]
    xip_err = results[:,6]
    return r, xip, xim, xip_err

def nCutoff(interpolant):
    """How many rows/columns around the edge to cut off for a given interpolant?

    Arguments:

        interpolant ------- String indicating which interpolant to use.  Options are "nearest",
                            "linear", "cubic", "quintic".
    """
    options = {"nearest" : 1,
               "linear" : 1,
               "cubic" : 2,
               "quintic" : 3}

    try:
        return options[interpolant]
    except KeyError:
        raise RuntimeError("No cutoff scheme was defined for interpolant %s!"%interpolant)

def main(n_realizations, dithering, n_output_bins, ps_plot_prefix, cf_plot_prefix,
    edge_cutoff=False):
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

        edge_cutoff -------- Cut off grid edges when comparing original vs. interpolation?  The
                             motivation for doing so would be to check for effects due to the edge
                             pixels being affected most by interpolation.
                             (default=False)
    """
    # Get basic grid information
    grid_spacing = grid_size / ngrid

    # Set up PowerSpectrum object.  We have to be careful to watch out for aliasing due to our
    # initial P(k) including power on scales above those that can be represented by our grid.  In
    # order to deal with that, we will define a power spectrum function that equals a cosmological
    # one for k < k_max and is zero above that, with some smoothing function rather than a hard
    # cutoff.
    raw_ps_data = np.loadtxt(pk_file).transpose()
    raw_ps_k = raw_ps_data[0,:]
    raw_ps_p = raw_ps_data[1,:]
    # Find k_max, taking into account that grid spacing is in degrees and our power spectrum is
    # defined in radians. So
    #    k_max = pi / (grid_spacing in radians) = pi / [2 pi (grid_spacing in degrees) / 180]
    #          = 90 / (grid spacing in degrees)
    k_max = 90. / grid_spacing
    # Now define a power spectrum that is raw_ps below k_max and goes smoothly to zero above that.
    ps_table = galsim.LookupTable(raw_ps_k, raw_ps_p*cutoff_func(raw_ps_k/k_max),
                                  interpolant='linear')
    ps = galsim.PowerSpectrum(ps_table, units = galsim.radians)

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
        if edge_cutoff:
            # If we are cutting off edge points, then all the functions we store will include that
            # cutoff.  However, we will save results for the original grids without the cutoffs just
            # in order to test what happens with / without a cutoff in the no-interpolation case.
            mean_nocutoff_ps_ee = np.zeros(n_output_bins)
            mean_nocutoff_ps_bb = np.zeros(n_output_bins)
            mean_nocutoff_ps_eb = np.zeros(n_output_bins)
            mean_nocutoff_cfp = np.zeros(n_output_bins)
            mean_nocutoff_cfm = np.zeros(n_output_bins)
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
            # dithers.
            if dithering == 'random':
                dither_x = x + u()*grid_spacing
                dither_y = y + u()*grid_spacing
                dither_x_list = list(dither_x.flatten())
                dither_y_list = list(dither_y.flatten())

            # Interpolate shears on the offset grid, with periodic interpolation.
            # Note: setting 'reduced=False' here, so as to compare g1 and g2 from original grid with
            # the interpolated g1 and g2 rather than reduced shear.
            interpolated_g1, interpolated_g2 = ps.getShear(pos=(dither_x_list,dither_y_list),
                                                           units=galsim.degrees,
                                                           periodic=True, reduced=False)
            # And put back into the format that the PowerSpectrumEstimator will want.
            interpolated_g1 = np.array(interpolated_g1).reshape((ngrid, ngrid))
            interpolated_g2 = np.array(interpolated_g2).reshape((ngrid, ngrid))

            # Now, we consider the question of whether we want cutoff grids.  If so, then we should
            # (a) store some results for the non-cutoff grids, and (b) cutoff the original and
            # interpolated grids before doing any further calculations.  If not, then nothing else
            # is really needed here.
            if edge_cutoff:
                # Get the PS for non-cutoff grid, and store results.
                nocutoff_pse = galsim.pse.PowerSpectrumEstimator(ngrid, grid_size, n_output_bins)
                nocutoff_ell, tmp_ps_ee, tmp_ps_bb, tmp_ps_eb, nocutoff_ps_ee_theory = \
                    nocutoff_pse.estimate(g1, g2, theory_func=ps_table)
                mean_nocutoff_ps_ee += tmp_ps_ee
                mean_nocutoff_ps_bb += tmp_ps_bb
                mean_nocutoff_ps_eb += tmp_ps_eb

                # Get the corr func for non-cutoff grid, and store results.
                nocutoff_th, tmp_cfp, tmp_cfm, _ = \
                    getCF(x.flatten(), y.flatten(), g1.flatten(), g2.flatten(),
                          grid_spacing, ngrid, n_output_bins)
                mean_nocutoff_cfp += tmp_cfp
                mean_nocutoff_cfm += tmp_cfm

                # Cut off the original, interpolated grids before doing any more calculations.
                # Store grid size that we actually use for everything else in future, post-cutoff.
                n_cutoff = nCutoff(interpolant)
                ngrid_use = ngrid - 2*n_cutoff
                grid_size_use = grid_size * float(ngrid_use)/ngrid
                if ngrid_use <= 2:
                    raise RuntimeError("After applying edge cutoff, grid is too small!"
                                       "Increase grid size or remove cutoff, or both.")
                g1 = g1[n_cutoff:ngrid-n_cutoff, n_cutoff:ngrid-n_cutoff]
                g2 = g2[n_cutoff:ngrid-n_cutoff, n_cutoff:ngrid-n_cutoff]
                interpolated_g1 = interpolated_g1[n_cutoff:ngrid-n_cutoff, n_cutoff:ngrid-n_cutoff]
                interpolated_g2 = interpolated_g2[n_cutoff:ngrid-n_cutoff, n_cutoff:ngrid-n_cutoff]
                x_use = x[n_cutoff:ngrid-n_cutoff, n_cutoff:ngrid-n_cutoff]
                y_use = y[n_cutoff:ngrid-n_cutoff, n_cutoff:ngrid-n_cutoff]
                dither_x_use = dither_x[n_cutoff:ngrid-n_cutoff, n_cutoff:ngrid-n_cutoff]
                dither_y_use = dither_y[n_cutoff:ngrid-n_cutoff, n_cutoff:ngrid-n_cutoff]
            else:
                # Just store the grid size and other quantities that we actually use.
                ngrid_use = ngrid
                grid_size_use = grid_size
                x_use = x
                y_use = y
                dither_x_use = dither_x
                dither_y_use = dither_y

            # Get statistics: PS, correlation function. 
            # Set up PowerSpectrumEstimator first, with the grid size and so on depending on whether
            # we have cut off the edges using the edge_cutoff.  The block of code just above this
            # one should have set all variables appropriately.
            if i_real == 0:
                interpolated_pse = galsim.pse.PowerSpectrumEstimator(ngrid_use,
                                                                     grid_size_use,
                                                                     n_output_bins)           
                pse = galsim.pse.PowerSpectrumEstimator(ngrid_use,
                                                        grid_size_use,
                                                        n_output_bins)           
            int_ell, interpolated_ps_ee, interpolated_ps_bb, interpolated_ps_eb, \
                interpolated_ps_ee_theory = \
                interpolated_pse.estimate(interpolated_g1,
                                          interpolated_g2,
                                          theory_func=ps_table)
            ell, ps_ee, ps_bb, ps_eb, ps_ee_theory = pse.estimate(g1, g2, theory_func=ps_table)
            int_th, interpolated_cfp, interpolated_cfm, cf_err = \
                getCF(dither_x_use.flatten(), dither_y_use.flatten(),
                      interpolated_g1.flatten(), interpolated_g2.flatten(),
                      grid_spacing, ngrid_use, n_output_bins)
            th, cfp, cfm, _ = \
                getCF(x_use.flatten(), y_use.flatten(), g1.flatten(), g2.flatten(),
                      grid_spacing, ngrid_use, n_output_bins)

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
        if edge_cutoff:
            mean_nocutoff_ps_ee /= n_realizations
            mean_nocutoff_ps_bb /= n_realizations
            mean_nocutoff_ps_eb /= n_realizations
            mean_nocutoff_cfp /= n_realizations
            mean_nocutoff_cfm /= n_realizations

        # Plot statistics, and ratios with vs. without interpolants.
        print "Running plotting routines for interpolant=%s..."%interpolant
        if edge_cutoff:
            generate_ps_cutoff_plots(ell, mean_ps_ee, ps_ee_theory,
                                     nocutoff_ell, mean_nocutoff_ps_ee, nocutoff_ps_ee_theory,
                                     interpolant, ps_plot_prefix, type='EE')
        generate_ps_plots(ell, mean_ps_ee, mean_interpolated_ps_ee, interpolant, ps_plot_prefix,
                          grid_spacing, type='EE')
        generate_ps_plots(ell, mean_ps_bb, mean_interpolated_ps_bb, interpolant, ps_plot_prefix,
                          grid_spacing, type='BB')
        generate_ps_plots(ell, mean_ps_eb, mean_interpolated_ps_eb, interpolant, ps_plot_prefix,
                          grid_spacing, type='EB')
        if edge_cutoff:
            generate_cf_cutoff_plots(th, mean_cfp, 
                                     nocutoff_th, mean_nocutoff_cfp,
                                     interpolant, cf_plot_prefix, type='p')
            generate_cf_cutoff_plots(th, mean_cfm, 
                                     nocutoff_th, mean_nocutoff_cfm,
                                     interpolant, cf_plot_prefix, type='m')
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
    description='Run tests of GalSim interpolants on lensing engine outputs.'
    usage='usage: %prog [options]'
    parser = optparse.OptionParser(usage=usage, description=description)
    parser.add_option('--n', dest="n_realizations", type=int,
                      default=default_n,
                      help='Number of objects to run tests on (default: %i)'%default_n)
    parser.add_option('--subsampling', dest='dithering', type=str,
                        help='Grid dithering to test (default: %s)'%default_dithering,
                        default=default_dithering)
    parser.add_option('--n_output_bins',
                      help='Number of bins for calculating 2-point functions '
                      '(default: %i)'%default_n_output_bins,
                      default=default_n_output_bins, type=int,
                      dest='n_output_bins')
    parser.add_option('--ps_plot_prefix',
                      help='Prefix for output power spectrum plots '
                      '(default: %s)'%default_ps_plot_prefix,
                      default=default_ps_plot_prefix, type=str,
                      dest='ps_plot_prefix')
    parser.add_option('--cf_plot_prefix',
                      help='Prefix for output correlation function plots '
                      '(default: %s)'%default_cf_plot_prefix,
                      default=default_cf_plot_prefix, type=str,
                      dest='cf_plot_prefix')
    parser.add_option('--edge_cutoff', dest="edge_cutoff", action='store_true', default=False,
                      help='Cut off edges of grid that are affected by interpolation')
    opts, args = parser.parse_args()
    # Make directories if necessary.
    dir = os.path.dirname(opts.ps_plot_prefix)
    if dir is not '':
        check_dir(dir)
    dir = os.path.dirname(opts.cf_plot_prefix)
    if dir is not '':
        check_dir(dir)

    main(n_realizations=opts.n_realizations,
         dithering=opts.dithering,
         n_output_bins=opts.n_output_bins,
         ps_plot_prefix=opts.ps_plot_prefix,
         cf_plot_prefix=opts.cf_plot_prefix,
         edge_cutoff=opts.edge_cutoff)
