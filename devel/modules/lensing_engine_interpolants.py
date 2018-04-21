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
"""
A script for checking the effects of interpolating gridded shears on the output correlation function
/ power spectrum.  In terms of the power spectrum, we expect behavior that is effectively
multiplying by the square of the Fourier transform of the interpolant, but we check this directly,
and also check the impact on the correlation function.

This script includes an implementation of two different tests of interpolation of lensing shears:

(1) A test that uses gridded points only, in interpolant_test_grid().  We set up some initial grid
of shear values, and interpolate to a grid that has the same spacing and number of points, but is
offset by some fraction of a grid unit.  While this setup is somewhat artificial, the benefit of
this test is that our power spectrum estimator (which works only for gridded points) can be used to
estimate the effect of interpolation on the shear power spectrum.  It also calls corr2 to check the
effects on the shear correlation function.

(2) A test of interpolation to random positions, in interpolant_test_random().  We set up some
fiducial grid that is used to interpolate to random positions, and compare with what we get if we
interpolate from a grid that has 10x smaller grid spacing (for which interpolant effects should be
minimal).  Since we interpolate to random positions, we cannot check the output power spectrum, only
the correlation function.

Any given call to this script will result in only one of the above functions being called, depending
on the user-supplied test parameters.
"""

import galsim
import numpy as np
import os
import subprocess
import pyfits
import optparse
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties

############################### Set some important quantities up top: ###########################
######## These are some hard-coded parameters (user cannot change from the command-line) ########
# Which interpolants do we want to test?
interpolant_list = ['linear', 'cubic', 'quintic', 'nearest','lanczos3','lanczos5']
n_interpolants = len(interpolant_list)
# Define the default grid for all tests.
grid_size = 10. # Total grid extent, in degrees
ngrid = 100 # Number of grid points per dimension
# Factor by which to upsample the grid in order to make a "reference" grid from which we interpolate
# to the positions of random points.
random_upsample = 10
# File containing the tabulated shear power spectrum to use for the tests.
pk_file = os.path.join('..','..','examples','data','cosmo-fid.zmed1.00_smoothed.out')
# Define a temporary file for correlation function outputs from corr2
tmp_cf_file = 'tmp.cf.dat'
# Make small fonts possible in legend, so they don't take up the whole plot.
fontP = FontProperties()
fontP.set_size('small')

######################### Set defaults for command-line arguments ################################
############### The user can change these easily from the command-line ###########################
# Factor by which to have the lensing engine internally expand the grid, to get large-scale shear
# correlations right.
default_kmin_factor = 3
# Grid offsets, for the tests that use a grid: 'random' (i.e., random sub-pixel amounts) or a
# specific fraction of a pixel.
default_dithering = 'random'
# Number of realizations to run.
default_n = 100
# Number of bins for calculating the power spectrum or correlation function.
default_n_output_bins = 12
# Prefix for filenames into which we will put the power spectrum, correlation function plots.
default_ps_plot_prefix = "plots/interpolated_ps_"
default_cf_plot_prefix = "plots/interpolated_cf_"

############### Utility function #############
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
            # There was a problem, so exit.
            raise

######################### Routines to make plots of results ##############################
def generate_ps_cutoff_plots(ell, ps, theory_ps,
                             nocutoff_ell, nocutoff_ps, nocutoff_theory_ps,
                             interpolant, ps_plot_prefix, type='EE'):
    """Routine to make power spectrum plots for grid edge cutoff vs. not (in both cases, without
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
    """Routine to make power spectrum plots for gridded points before and after interpolation, and
       write them to file.

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
        u = fine_ell*dth*np.pi/180. # check factors of pi and so on; the dth is needed to convert
        # from actual distances to cycles per pixel.  Since fine_ell is in 1/radians and dth is in
        # degrees, I think this is right.  However, I'm not sure that the expression here is right
        # in general; there was some discussion of FT(interp)^2 but I think it should be some
        # average over the values of FT(interp(u, v))^2 in the regions of the (u, v) plane where we
        # could get the right value for sqrt(u^2+v^2) to fall into this bin in ell.  What's below
        # just uses some value assuming that u=v always, which must be wrong at some level.
        tmp_interp = galsim.Interpolant2d(interpolant)
        theor_ratio = np.zeros_like(u)
        for indx in range(len(fine_ell)):
            theor_ratio[indx] = (tmp_interp.uval(u[indx]/np.pi, u[indx]/np.pi))
        ax.plot(fine_ell, theor_ratio, '--', color='g', label='|FT interpolant|^2')
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
                      dth, type='p', theory_raw=None, theory_binned=None, theory_rand=None):
    """Routine to make correlation function plots for interpolation tests and write them to file.

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

      theory_raw ---------- Theory prediction at finely-spaced positions.

      theory_binned ------- Theory prediction averaged within bin, for gridded points.

      theory_rand --------- Theory prediction averaged within bin, for random points
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
    if theory_raw is not None:
        ax.plot(theory_raw[0], theory_raw[1], color='g', label='Theory (unbinned)')
    if theory_binned is not None:
        ax.plot(th, theory_binned, color='k', label='Theory (binned from grid)')
    if theory_rand is not None:
        ax.plot(th, theory_rand, color='m', label='Theory (binned from random)')
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
    back in and return them as a set of NumPy arrays.

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
                            "linear", "cubic", "quintic", "lanczos3", "lanczos5".
    """
    options = {"nearest" : 1,
               "linear" : 1,
               "cubic" : 2,
               "quintic" : 3,
               "lanczos3" : 3,
               "lanczos5" : 5}

    try:
        return options[interpolant]
    except KeyError:
        raise RuntimeError("No cutoff scheme was defined for interpolant %s!"%interpolant)


################### Below are the two core functions for interpolant tests ######################
############################## using gridded and random points ##################################

def interpolant_test_grid(n_realizations, dithering, n_output_bins, kmin_factor, ps_plot_prefix,
                          cf_plot_prefix, edge_cutoff=False, periodic=False):
    """Main routine to drive all tests for interpolation to dithered grid positions.

    Arguments:

        n_realizations ----- Number of random realizations of each shear field.

        dithering ---------- Sub-pixel dither to apply to the grid onto which we interpolate, with
                             respect to the original grid on which shears are determined.  If
                             'random' (default) then each realization has a random (x, y) sub-pixel
                             offset.  If a string that converts to a float, then that float is used
                             to determine the x, y offsets for each realization.

        n_output_bins ------ Number of bins for calculation of 2-point functions.

        kmin_factor -------- Factor by which to divide the native kmin of the grid (as an argument
                             to the lensing engine).  Default: 3.

        ps_plot_prefix ----- Prefix to use for power-spectrum outputs.

        cf_plot_prefix ----- Prefix to use for correlation function outputs.

        edge_cutoff -------- Cut off grid edges when comparing original vs. interpolation?  The
                             motivation for doing so would be to check for effects due to the edge
                             pixels being affected most by interpolation.
                             (default=False)

        periodic ----------- Use periodic interpolation when getting the shears at the dithered grid
                             positions? (default=False)
    """
    # Get basic grid information
    grid_spacing = grid_size / ngrid

    print "Doing initial PS / correlation function setup..."
    # Set up PowerSpectrum object.  We have to be careful to watch out for aliasing due to our
    # initial P(k) including power on scales above those that can be represented by our grid.  In
    # order to deal with that, we will define a power spectrum function that equals a cosmological
    # one.  Later, when getting the shears, we will use a keyword to ensure that it is
    # band-limited.
    ps_table = galsim.LookupTable(file=pk_file, interpolant='linear')
    ps = galsim.PowerSpectrum(pk_file, units = galsim.radians)
    # Let's also get a theoretical correlation function for later use.
    theory_th_vals, theory_cfp_vals, theory_cfm_vals = \
        ps.calculateXi(grid_spacing=grid_spacing, ngrid=ngrid, units=galsim.degrees,
                       bandlimit="soft", kmin_factor=kmin_factor, n_theta=100)

    # Set up grid and the corresponding x, y lists.
    min_val = (-ngrid/2 + 0.5) * grid_spacing
    max_val = (ngrid/2 - 0.5) * grid_spacing
    x, y = np.meshgrid(
        np.arange(min_val,max_val+grid_spacing,grid_spacing),
        np.arange(min_val,max_val+grid_spacing,grid_spacing))

    # Parse the target position information: grids that are dithered by some fixed amount, or
    # randomly.
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
        target_x = x[0:ngrid-1,0:ngrid-1] + dithering*grid_spacing
        target_y = y[0:ngrid-1,0:ngrid-1] + dithering*grid_spacing
        target_x = list(target_x.flatten())
        target_y = list(target_y.flatten())
    else:
        # Just set up the uniform deviate.
        u = galsim.UniformDeviate()

    # Initialize arrays for two-point functions.
    if edge_cutoff:
        # If we are cutting off edge points, then all the functions we store will include that
        # cutoff.  However, we will save results for the original grids without the cutoffs just in
        # order to test what happens with / without a cutoff in the no-interpolation case.
        mean_nocutoff_ps_ee = np.zeros(n_output_bins)
        mean_nocutoff_ps_bb = np.zeros(n_output_bins)
        mean_nocutoff_ps_eb = np.zeros(n_output_bins)
        mean_nocutoff_cfp = np.zeros(n_output_bins)
        mean_nocutoff_cfm = np.zeros(n_output_bins)
    mean_interpolated_ps_ee = np.zeros((n_output_bins, n_interpolants))
    mean_ps_ee = np.zeros(n_output_bins)
    mean_interpolated_ps_bb = np.zeros((n_output_bins, n_interpolants))
    mean_ps_bb = np.zeros(n_output_bins)
    mean_interpolated_ps_eb = np.zeros((n_output_bins, n_interpolants))
    mean_ps_eb = np.zeros(n_output_bins)
    mean_interpolated_cfp = np.zeros((n_output_bins, n_interpolants))
    mean_cfp = np.zeros(n_output_bins)
    mean_interpolated_cfm = np.zeros((n_output_bins, n_interpolants))
    mean_cfm = np.zeros(n_output_bins)

    print "Test type: offset/dithered grids, correlation function and power spectrum."
    print "Doing calculations for %d realizations"%n_realizations
    # Loop over realizations.
    for i_real in range(n_realizations):

        # Get shears on default grid.
        g1, g2 = ps.buildGrid(grid_spacing = grid_spacing, ngrid = ngrid, units = galsim.degrees,
                              kmin_factor = kmin_factor, bandlimit = 'soft')

        # Set up the target positions for this interpolation, if we're using random grid dithers.
        if dithering == 'random':
            target_x = x + u()*grid_spacing
            target_y = y + u()*grid_spacing
            target_x_list = list(target_x.flatten())
            target_y_list = list(target_y.flatten())

        # Interpolate shears to the target positions, with periodic interpolation if specified at
        # the command-line.  Do this for each interpolant in turn.  But first we have to set up some
        # arrays to store results.
        interpolated_g1 = np.zeros((ngrid, ngrid, n_interpolants))
        interpolated_g2 = np.zeros((ngrid, ngrid, n_interpolants))
        for i_int in range(n_interpolants):
            # Note: setting 'reduced=False' here, so as to compare g1 and g2 from original grid with
            # the interpolated g1 and g2 rather than with the reduced shear, which is what getShear
            # returns by default.
            tmp_g1, tmp_g2 = \
                ps.getShear(pos=(target_x_list,target_y_list), units=galsim.degrees,
                            periodic=periodic, interpolant = interpolant_list[i_int], reduced=False)
            interpolated_g1[:,:,i_int] = np.array(tmp_g1).reshape(ngrid,ngrid)
            interpolated_g2[:,:,i_int] = np.array(tmp_g2).reshape(ngrid,ngrid)

        # Now, we consider the question of whether we want cutoff grids.  If so, then we should (a)
        # store some results for the non-cutoff grids, and (b) cutoff the original and interpolated
        # results before doing any further calculations.  If not, then nothing else is really needed
        # here.
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

            # Cut off the original, interpolated set of positions before doing any more calculations.
            # Store grid size that we actually use for everything else in future, post-cutoff.
            # We will conservatively cut off the maximum required for our most tricky interpolant.
            n_cutoff = 0
            for interpolant in interpolant_list:
                n_cutoff = max(n_cutoff, nCutoff(interpolant))
            ngrid_use = ngrid - 2*n_cutoff
            grid_size_use = grid_size * float(ngrid_use)/ngrid
            if ngrid_use <= 2:
                raise RuntimeError("After applying edge cutoff, grid is too small!"
                                   "Increase grid size or remove cutoff, or both.")
            g1 = g1[n_cutoff:ngrid-n_cutoff, n_cutoff:ngrid-n_cutoff]
            g2 = g2[n_cutoff:ngrid-n_cutoff, n_cutoff:ngrid-n_cutoff]
            x_use = x[n_cutoff:ngrid-n_cutoff, n_cutoff:ngrid-n_cutoff]
            y_use = y[n_cutoff:ngrid-n_cutoff, n_cutoff:ngrid-n_cutoff]

            interpolated_g1 = interpolated_g1[n_cutoff:ngrid-n_cutoff, n_cutoff:ngrid-n_cutoff, :]
            interpolated_g2 = interpolated_g2[n_cutoff:ngrid-n_cutoff, n_cutoff:ngrid-n_cutoff, :]
            target_x_use = target_x[n_cutoff:ngrid-n_cutoff, n_cutoff:ngrid-n_cutoff]
            target_y_use = target_y[n_cutoff:ngrid-n_cutoff, n_cutoff:ngrid-n_cutoff]
        else:
            # Just store the grid size and other quantities that we actually use.
            ngrid_use = ngrid
            grid_size_use = grid_size
            x_use = x
            y_use = y
            target_x_use = target_x
            target_y_use = target_y
                
        # Get statistics: PS, correlation function.
        # Set up PowerSpectrumEstimator first, with the grid size and so on depending on whether we
        # have cut off the edges using the edge_cutoff.  The block of code just above this one
        # should have set all variables appropriately.
        if i_real == 0:
            interpolated_pse = galsim.pse.PowerSpectrumEstimator(ngrid_use, grid_size_use,
                                                                 n_output_bins)
            pse = galsim.pse.PowerSpectrumEstimator(ngrid_use, grid_size_use, n_output_bins)           

        # First do the estimation and store results for the original grid.
        ell, ps_ee, ps_bb, ps_eb, ps_ee_theory = pse.estimate(g1, g2, theory_func=ps_table)
        th, cfp, cfm, _ = \
            getCF(x_use.flatten(), y_use.flatten(), g1.flatten(), g2.flatten(),
                  grid_spacing, ngrid_use, n_output_bins)
        mean_ps_ee += ps_ee
        mean_ps_bb += ps_bb
        mean_ps_eb += ps_eb
        mean_cfp += cfp
        mean_cfm += cfm
        # Now do this for each set of interpolated results.
        for i_int in range(n_interpolants):
            int_ell, interpolated_ps_ee, interpolated_ps_bb, interpolated_ps_eb, \
                interpolated_ps_ee_theory = \
                interpolated_pse.estimate(interpolated_g1[:,:,i_int], interpolated_g2[:,:,i_int],
                                          theory_func=ps_table)
            int_th, interpolated_cfp, interpolated_cfm, cf_err = \
                getCF(target_x_use.flatten(), target_y_use.flatten(),
                      interpolated_g1[:,:,i_int].flatten(), interpolated_g2[:,:,i_int].flatten(),
                      grid_spacing, ngrid_use, n_output_bins)

            # Accumulate statistics.
            mean_interpolated_ps_ee[:,i_int] += interpolated_ps_ee
            mean_interpolated_ps_bb[:,i_int] += interpolated_ps_bb
            mean_interpolated_ps_eb[:,i_int] += interpolated_ps_eb
            mean_interpolated_cfp[:,i_int] += interpolated_cfp
            mean_interpolated_cfm[:,i_int] += interpolated_cfm

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
    if edge_cutoff:
        mean_nocutoff_ps_ee /= n_realizations
        mean_nocutoff_ps_bb /= n_realizations
        mean_nocutoff_ps_eb /= n_realizations
        mean_nocutoff_cfp /= n_realizations
        mean_nocutoff_cfm /= n_realizations
        generate_ps_cutoff_plots(ell, mean_ps_ee, ps_ee_theory,
                                 nocutoff_ell, mean_nocutoff_ps_ee, nocutoff_ps_ee_theory,
                                 interpolant, ps_plot_prefix, type='EE')
        generate_cf_cutoff_plots(th, mean_cfp, 
                                 nocutoff_th, mean_nocutoff_cfp,
                                 interpolant, cf_plot_prefix, type='p')
        generate_cf_cutoff_plots(th, mean_cfm, 
                                 nocutoff_th, mean_nocutoff_cfm,
                                 interpolant, cf_plot_prefix, type='m')

    for i_int in range(n_interpolants):
        interpolant = interpolant_list[i_int]
        print "Running plotting routines for interpolant=%s..."%interpolant
        generate_ps_plots(ell, mean_ps_ee, mean_interpolated_ps_ee[:,i_int], interpolant,
                          ps_plot_prefix, grid_spacing, type='EE')
        generate_ps_plots(ell, mean_ps_bb, mean_interpolated_ps_bb[:,i_int], interpolant,
                          ps_plot_prefix, grid_spacing, type='BB')
        generate_ps_plots(ell, mean_ps_eb, mean_interpolated_ps_eb[:,i_int], interpolant,
                          ps_plot_prefix, grid_spacing, type='EB')
        generate_cf_plots(th, mean_cfp, mean_interpolated_cfp[:,i_int], interpolant,
                          cf_plot_prefix, grid_spacing, type='p',
                          theory_raw=(theory_th_vals, theory_cfp_vals))
        generate_cf_plots(th, mean_cfm, mean_interpolated_cfm[:,i_int], interpolant,
                          cf_plot_prefix, grid_spacing, type='m',
                          theory_raw=(theory_th_vals, theory_cfm_vals))

        # Output results.
        print "Outputting tables of results..."
        write_ps_output(ell, mean_ps_ee, mean_interpolated_ps_ee[:,i_int],
                        interpolant, ps_plot_prefix, type='EE')
        write_ps_output(ell, mean_ps_bb, mean_interpolated_ps_bb[:,i_int],
                        interpolant, ps_plot_prefix, type='BB')
        write_ps_output(ell, mean_ps_eb, mean_interpolated_ps_eb[:,i_int],
                        interpolant, ps_plot_prefix, type='EB')
        write_cf_output(th, mean_cfp, mean_interpolated_cfp[:,i_int], interpolant,
                        cf_plot_prefix, type='p')
        write_cf_output(th, mean_cfm, mean_interpolated_cfm[:,i_int], interpolant,
                        cf_plot_prefix, type='m')
        print ""

def interpolant_test_random(n_realizations, n_output_bins, kmin_factor,
                            cf_plot_prefix, edge_cutoff=False, periodic=False):
    """Main routine to drive all tests of interpolation to random points.

    Arguments:

        n_realizations ----- Number of random realizations of each shear field.  We will actually
                             cheat and just make a fixed number of realizations, but sample at more
                             random points according to the value of `n_realizations`.

        n_output_bins ------ Number of bins for calculation of 2-point functions.

        kmin_factor -------- Factor by which to divide the native kmin of the grid (as an argument
                             to the lensing engine).  Default: 3.

        cf_plot_prefix ----- Prefix to use for correlation function outputs.

        edge_cutoff -------- Cut off grid edges when comparing original vs. interpolation?  The
                             motivation for doing so would be to check for effects due to the edge
                             pixels being affected most by interpolation.
                             (default=False)

        periodic ----------- Use periodic interpolation when getting the shears at the dithered grid
                             positions? (default=False)
    """
    # Get basic grid information
    grid_spacing = grid_size / ngrid

    print "Doing initial PS / correlation function setup..."
    # Set up PowerSpectrum object.  We have to be careful to watch out for aliasing due to our
    # initial P(k) including power on scales above those that can be represented by our grid.  In
    # order to deal with that, we will define a power spectrum function that equals a cosmological
    # one.  Unlike for the previous function, in this case we have to explicitly include the cutoff
    # in the power.  The reason for this is that we want the cutoff to be the same for the default
    # and the subsampled grid, and the only way to make this happen is to cut it off and hand in the
    # same power spectrum in both cases.
    raw_ps_data = np.loadtxt(pk_file).transpose()
    raw_ps_k = raw_ps_data[0,:]
    raw_ps_p = raw_ps_data[1,:]
    # Find k_max, taking into account that grid spacing is in degrees and our power spectrum is
    # defined in radians. So
    #    k_max = pi / (grid_spacing in radians) = pi / [2 pi (grid_spacing in degrees) / 180]
    #          = 90 / (grid spacing in degrees)
    # Also find k_min, for correlation function prediction.
    #    k_min = 2*pi / (total grid extent) = 180. / (grid extent)
    k_max = 90.*random_upsample / grid_spacing # factor of 10 because we're actually going to use a
                                               # finer grid
    k_min = 180. / (kmin_factor*grid_size)
    # Now define a power spectrum that is raw_ps.  All softening / cutting off above k_max will
    # happen internally in the galsim.PowerSpectrum class.
    ps_table = galsim.LookupTable(raw_ps_k, raw_ps_p*(raw_ps_k<k_max).astype(float),
                                  interpolant='linear')

    ps = galsim.PowerSpectrum(ps_table, units = galsim.radians)
    # Let's also get a theoretical correlation function for later use.
    theory_th_vals, theory_cfp_vals, theory_cfm_vals = \
        ps.calculateXi(grid_spacing=grid_spacing, ngrid=ngrid, units=galsim.degrees,
                       bandlimit="soft", kmin_factor=kmin_factor, n_theta=100)

    # Set up grid and the corresponding x, y lists.
    ngrid_fine = random_upsample*ngrid
    grid_spacing_fine = grid_spacing/random_upsample
    min_fine = (-ngrid_fine/2 + 0.5) * grid_spacing_fine
    max_fine = (ngrid_fine/2 - 0.5) * grid_spacing_fine
    x_fine, y_fine = np.meshgrid(
        np.arange(min_fine,max_fine+grid_spacing_fine,grid_spacing_fine),
        np.arange(min_fine,max_fine+grid_spacing_fine,grid_spacing_fine))

    # Set up uniform deviate and min/max values for random positions
    u = galsim.UniformDeviate()
    random_min_val = np.min(x_fine)
    random_max_val = np.max(x_fine)

    # Initialize arrays for two-point functions.
    mean_interpolated_cfp = np.zeros((n_output_bins, n_interpolants))
    mean_cfp = np.zeros((n_output_bins, n_interpolants))
    mean_interpolated_cfm = np.zeros((n_output_bins, n_interpolants))
    mean_cfm = np.zeros((n_output_bins, n_interpolants))

    # Sort out number of realizations:
    # We make a maximum of 20 realizations here.  If more are requested, then we simply increase the
    # number density of random points within each realizations.
    if n_realizations > 20:
        n_points = int(float(n_realizations)/20*ngrid**2)
        n_realizations = 20
    else:
        n_points = ngrid**2

    print "Test type: random target positions, correlation function only."
    print "Doing calculations for %d realizations with %d points each."%(n_realizations,n_points)

    # Loop over realizations.
    for i_real in range(n_realizations):

        # Set up the target positions for this interpolation.
        target_x = np.zeros(n_points)
        target_y = np.zeros(n_points)
        for ind in range(n_points):
            target_x[ind] = random_min_val+(random_max_val-random_min_val)*u()
            target_y[ind] = random_min_val+(random_max_val-random_min_val)*u()
        target_x_list = list(target_x)
        target_y_list = list(target_y)

        # Get shears on default grid and fine grid.  Interpolation from the former is going to be
        # our test case, interpolation from the latter will be treated like ground truth.  Note that
        # these would nominally have different kmax, which would result in different correlation
        # functions, but really since we cut off the power above kmax for the default grid already,
        # the effective kmax for the correlation function is the same in both cases.
        g1_fine, g2_fine = ps.buildGrid(grid_spacing = grid_spacing/random_upsample,
                                        ngrid = random_upsample*ngrid,
                                        units = galsim.degrees,
                                        kmin_factor = kmin_factor, bandlimit = 'soft')
        interpolated_g1_fine = np.zeros((len(target_x_list), n_interpolants))
        interpolated_g2_fine = np.zeros((len(target_x_list), n_interpolants))
        interpolated_g1 = np.zeros((len(target_x_list), n_interpolants))
        interpolated_g2 = np.zeros((len(target_x_list), n_interpolants))
        for i_int in range(n_interpolants):
            # Interpolate shears to the target positions, with periodic interpolation if specified
            # at the command-line.
            # Note: setting 'reduced=False' here, so as to compare g1 and g2 from original grid with
            # the interpolated g1 and g2 rather than the reduced shear.
            interpolated_g1_fine[:,i_int], interpolated_g2_fine[:,i_int] = \
                ps.getShear(pos=(target_x_list, target_y_list), units=galsim.degrees,
                            periodic=periodic, reduced=False, interpolant=interpolant_list[i_int])

        g1, g2 = ps.subsampleGrid(random_upsample)
        for i_int in range(n_interpolants):
            interpolated_g1[:,i_int], interpolated_g2[:,i_int] = \
                ps.getShear(pos=(target_x_list,target_y_list), units=galsim.degrees,
                            periodic=periodic, reduced=False, interpolant=interpolant_list[i_int])

        # Now, we consider the question of whether we want cutoff grids.  If so, then we should (a)
        # store some results for the non-cutoff grids, and (b) cutoff the results before doing any
        # further calculations.  If not, then nothing else is really needed here.
        if edge_cutoff:
            # Cut off the original, interpolated set of positions before doing any more calculations.
            # Store grid size that we actually use for everything else in future, post-cutoff.  To
            # be fair, just cut off all at the maximum required value over all interpolants.
            n_cutoff = 0
            for i_int in range(n_interpolants):
                n_cutoff = max(n_cutoff, nCutoff(interpolant_list[i_int]))
            ngrid_use = ngrid - 2*n_cutoff
            grid_size_use = grid_size * float(ngrid_use)/ngrid
            if ngrid_use <= 2:
                raise RuntimeError("After applying edge cutoff, grid is too small!"
                                   "Increase grid size or remove cutoff, or both.")
            # Remember that x_fine, y_fine have more points compared to default grid.
            min_val_fine = random_upsample*n_cutoff
            max_val_fine = random_upsample*ngrid - random_upsample*n_cutoff
            x_use = x_fine[min_val_fine:max_val_fine, min_val_fine:max_val_fine]
            y_use = y_fine[min_val_fine:max_val_fine, min_val_fine:max_val_fine]

            targets_to_use = np.logical_and.reduce(
                [target_x >= np.min(x_use),
                 target_x < np.max(x_use),
                 target_y >= np.min(y_use),
                 target_y < np.max(y_use)
                 ])
            target_x_use = target_x[targets_to_use]
            target_y_use = target_y[targets_to_use]
            interpolated_g1_use = interpolated_g1[targets_to_use,:]
            interpolated_g2_use = interpolated_g2[targets_to_use,:]
            interpolated_g1_fine_use = interpolated_g1_fine[targets_to_use,:]
            interpolated_g2_fine_use = interpolated_g2_fine[targets_to_use,:]
        else:
            # Just store the quantities that we actually use, from before.
            ngrid_use = ngrid
            target_x_use = target_x
            target_y_use = target_y
            interpolated_g1_use = interpolated_g1
            interpolated_g2_use = interpolated_g2
            interpolated_g1_fine_use = interpolated_g1_fine
            interpolated_g2_fine_use = interpolated_g2_fine

        # Get statistics: correlation function.  Do this for all sets of interpolated points.
        for i_int in range(n_interpolants):
            int_th, interpolated_cfp, interpolated_cfm, cf_err = \
                getCF(target_x_use, target_y_use, interpolated_g1_use[:,i_int], interpolated_g2_use[:,i_int],
                      grid_spacing, ngrid_use, n_output_bins)
            th, cfp, cfm, _ = \
                getCF(target_x_use, target_y_use, interpolated_g1_fine_use[:,i_int], interpolated_g2_fine_use[:,i_int],
                      grid_spacing, ngrid_use, n_output_bins)

            # Accumulate statistics.
            mean_interpolated_cfp[:,i_int] += interpolated_cfp
            mean_cfp[:,i_int] += cfp
            mean_interpolated_cfm[:,i_int] += interpolated_cfm
            mean_cfm[:,i_int] += cfm

    # Now get the average over all realizations
    print "Done generating realizations, now getting mean 2-point functions"
    mean_interpolated_cfp /= n_realizations
    mean_cfp /= n_realizations
    mean_interpolated_cfm /= n_realizations
    mean_cfm /= n_realizations

    # Plot statistics, and ratios with vs. without interpolants.
    for i_int in range(n_interpolants):
        interpolant = interpolant_list[i_int]
        print "Running plotting routines for interpolant=%s..."%interpolant

        generate_cf_plots(th, mean_cfp[:,i_int], mean_interpolated_cfp[:,i_int], interpolant,
                          cf_plot_prefix, grid_spacing, type='p',
                          theory_raw=(theory_th_vals, theory_cfp_vals))
        generate_cf_plots(th, mean_cfm[:,i_int], mean_interpolated_cfm[:,i_int], interpolant,
                          cf_plot_prefix, grid_spacing, type='m',
                          theory_raw=(theory_th_vals, theory_cfm_vals))

        # Output results.
        print "Outputting tables of results..."
        write_cf_output(th, mean_cfp[:,i_int], mean_interpolated_cfp[:,i_int], interpolant,
                        cf_plot_prefix, type='p')
        write_cf_output(th, mean_cfm[:,i_int], mean_interpolated_cfm[:,i_int], interpolant,
                        cf_plot_prefix, type='m')
        print ""

if __name__ == "__main__":
    description='Run tests of GalSim interpolants on lensing engine outputs.'
    usage='usage: %prog [options]'
    parser = optparse.OptionParser(usage=usage, description=description)
    parser.add_option('--n', dest="n_realizations", type=int,
                      default=default_n,
                      help='Number of objects to run tests on (default: %i)'%default_n)
    parser.add_option('--dithering', dest='dithering', type=str,
                        help='Grid dithering to test (default: %s)'%default_dithering,
                        default=default_dithering)
    parser.add_option('--random', dest="random", action='store_true', default=False,
                      help='Distribute points completely randomly? (default: False)')
    parser.add_option('--n_output_bins',
                      help='Number of bins for calculating 2-point functions '
                      '(default: %i)'%default_n_output_bins,
                      default=default_n_output_bins, type=int,
                      dest='n_output_bins')
    parser.add_option('--kmin_factor',
                      help='Factor by which to multiply kmin (default: %i)'%default_kmin_factor,
                      default=default_kmin_factor, type=int,
                      dest='kmin_factor')
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
                      help='Cut off edges of grid that are affected by interpolation (default: false)')
    parser.add_option('--periodic', dest="periodic", action='store_true', default=False,
                      help='Do interpolation assuming a periodic grid (default: false)')
    v = optparse.Values()
    opts, args = parser.parse_args()
    # Check for mutually inconsistent args
    if 'random' in v.__dict__.keys() and 'dithering' in v.__dict__.keys():
        raise RuntimeError("Either grid dithering or random points should be specified, not both!")
    # Make directories if necessary.
    dir = os.path.dirname(opts.ps_plot_prefix)
    if dir is not '':
        check_dir(dir)
    dir = os.path.dirname(opts.cf_plot_prefix)
    if dir is not '':
        check_dir(dir)

    if opts.random:
        interpolant_test_random(
                n_realizations=opts.n_realizations,
                n_output_bins=opts.n_output_bins,
                kmin_factor=opts.kmin_factor,
                cf_plot_prefix=opts.cf_plot_prefix,
                edge_cutoff=opts.edge_cutoff,
                periodic=opts.periodic
                )
    else:
        interpolant_test_grid(
                n_realizations=opts.n_realizations,
                dithering=opts.dithering,
                n_output_bins=opts.n_output_bins,
                kmin_factor=opts.kmin_factor,
                ps_plot_prefix=opts.ps_plot_prefix,
                cf_plot_prefix=opts.cf_plot_prefix,
                edge_cutoff=opts.edge_cutoff,
                periodic=opts.periodic
                )
