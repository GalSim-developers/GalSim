"""@file reconvolution_validation_plots.py 
Create various plot from data produced by reconvolution_validation.py.
"""

import logging
import sys
import os
import numpy
import argparse
import yaml
import pdb
import pylab
import math
import galsim

HSM_ERROR_VALUE = -99
NO_PSF_VALUE    = -98

def PlotEllipticityBiases(filename_output):
    """
    This function is a prototype for a plotting function. 
    I am not sure what is the best plots/subplots combination to show what we want to see, so for now 
    let's use this form.
    Arguments
    ---------
    filename_output     file containing results from photon_vs_fft
    """

    data = numpy.loadtxt(filename_output,ndmin=2)

    n_test_gals = data.shape[0]

    g1_photon=data[:,4]
    g2_photon=data[:,5]
    g1_fft=data[:,2]
    g2_fft=data[:,3]

    de1 = g1_fft-g1_photon
    de2 = g2_fft-g2_photon 
    
    pylab.plot(de1/g1_photon,'x',label='E1')
    pylab.plot(de2/g2_photon,'+',label='E2')
                
    pylab.xlabel('test galaxy #')
    pylab.ylabel('de/e')
    pylab.xlim([-1,n_test_gals])

    pylab.gcf().set_size_inches(10,5)
    pylab.legend()

    filename_fig = os.path.join(dirname_figs,'photon_vs_fft_differences.png');
    pylab.savefig(filename_fig)
    pylab.close()

    logger.info('saved figure %s' % filename_fig)

def PlotEllipticityBiasesHistogram(filename_output):
    """
    Plot histogram for differences between photon and FFT images.
    Arguments
    ---------
    filename_output     file containing results from photon_vs_fft
    """

    # id
    # max_diff_over_max_image
    # E1_moments_fft
    # E2_moments_fft
    # E1_moments_photon
    # E2_moments_photon
    # E1_hsm_corr_fft
    # E2_hsm_corr_fft
    # E1_hsm_corr_photon
    # E2_hsm_corr_photon
    # moments_fft_sigma
    # moments_shoot_sigma
    # hsm_fft_sigma
    # hsm_shoot_sigma

    data = numpy.loadtxt(filename_output)
    n_test_gals = data.shape[0]
    n_bins = 25
    n_ticks= 3

    E1_fft=data[:,2]
    E2_fft=data[:,3]
    E1_photon=data[:,4]
    E2_photon=data[:,5]
    
    pylab.subplot(231)
    pylab.hist((E1_fft-E1_photon),label='E1',bins=n_bins)
    pylab.title('adaptive moments')
    pylab.xlabel('E1_fft - E1_photon')
    pylab.ylabel('count')
    pylab.xticks(numpy.linspace(min(pylab.xticks()[0]),max(pylab.xticks()[0]),n_ticks))
    
    pylab.subplot(234)
    pylab.hist((E2_fft-E2_photon),label='E2',bins=n_bins)
    pylab.xlabel('E2_fft - E2_photon')
    pylab.ylabel('count')
    pylab.xticks(numpy.linspace(min(pylab.xticks()[0]),max(pylab.xticks()[0]),n_ticks))


    E1_fft=data[:,6]
    E2_fft=data[:,7]
    E1_photon=data[:,8]
    E2_photon=data[:,9]

    pylab.subplot(232)
    pylab.hist((E1_fft-E1_photon),label='E1',bins=n_bins)
    pylab.title('HSM corrected')
    pylab.xlabel('E1_fft - E1_photon')
    pylab.ylabel('count')
    pylab.xticks(numpy.linspace(min(pylab.xticks()[0]),max(pylab.xticks()[0]),n_ticks))
    
    pylab.subplot(235)
    pylab.hist((E2_fft-E2_photon),label='E2',bins=n_bins)
    pylab.xlabel('E2_fft - E2_photon')
    pylab.ylabel('count')
    pylab.xticks(numpy.linspace(min(pylab.xticks()[0]),max(pylab.xticks()[0]),n_ticks))

    # plot moments differnces

    sigma_moments_fft=data[:,10]
    sigma_moments_photon=data[:,11]
    sigma_hsm_fft=data[:,12]
    sigma_hsm_photon=data[:,13]

    pylab.subplot(233)
    pylab.hist((sigma_moments_fft-sigma_moments_photon),label='sigma',bins=n_bins)
    pylab.title('moments')
    pylab.xlabel('sigma_fft - sigma_photon')
    pylab.ylabel('count')
    pylab.xticks(numpy.linspace(min(pylab.xticks()[0]),max(pylab.xticks()[0]),n_ticks))

    pylab.subplot(236)
    pylab.hist((sigma_hsm_fft-sigma_hsm_photon),label='sigma',bins=n_bins)           
    pylab.title('HSM')
    pylab.xlabel('sigma_fft - sigma_photon')
    pylab.ylabel('count')
    pylab.xticks(numpy.linspace(min(pylab.xticks()[0]),max(pylab.xticks()[0]),n_ticks))
      
    pylab.gcf().set_size_inches(20,10)

    filename_fig = os.path.join(dirname_figs,'photon_vs_fft_histogram.png');
    pylab.savefig(filename_fig)
    pylab.close()

    logger.info('saved figure %s' % filename_fig)

def PlotStatsForParam(config,param_name):
    """
    Save plots for the results of photon_vs_fft, when param_name is varied.
    Arguments
    ---------
    config               galsim yaml config, which was used to produce the results, read by yaml
    param_name           varied parameter name, listed under config['vary_params'], for which
                            to create the plots
    """

    # get the shortcut to the dict corresponding to current varied parameter
    param = config['vary_params'][param_name]

    # initialise the arrays for results
    moments_E1_diff_mean = []
    moments_E1_diff_stdm = []
    moments_E1_diff_medn = []
    moments_E2_diff_mean = []
    moments_E2_diff_stdm = []
    moments_E2_diff_medn = []

    hsmcorr_E1_diff_mean = []
    hsmcorr_E1_diff_stdm = []
    hsmcorr_E1_diff_medn = []
    hsmcorr_E2_diff_mean = []
    hsmcorr_E2_diff_stdm = []
    hsmcorr_E2_diff_medn = []

    moments_sigma_diff_mean = []
    moments_sigma_diff_stdm = []
    moments_sigma_diff_medn = []
    hsmcorr_sigma_diff_mean = []
    hsmcorr_sigma_diff_stdm = []
    hsmcorr_sigma_diff_medn = []

    # loop over values changed for the varied parameter
    for iv,value in enumerate(param['values']):

        # get the filename for the results file
        filename_out = 'results.%s.%s.%03d.cat' % (config['filename_config'],param_name,iv)
        # load the results file
        results = numpy.loadtxt(filename_out)
        # get the number of galaxies in the results file
        n_res_all = results.shape[0]
        # clean the HSM errors
        for col in range(2,14):
            select = results[:,col] != HSM_ERROR_VALUE
            results = results[select,:]
        n_res = results.shape[0]

        logging.info('opened file %s with %d valid measurements (out of %d) for %s=%e',filename_out,n_res,n_res_all,param_name,iv)

        # get all the stats
        moments_E1_diff_mean.append( numpy.mean(results[:,2] - results[:,4])  )
        moments_E1_diff_stdm.append( numpy.std(results[:,2] - results[:,4],ddof=1)/numpy.sqrt(n_res)  )
        moments_E1_diff_medn.append( numpy.median(results[:,2] - results[:,4])  )
        moments_E2_diff_mean.append( numpy.mean(results[:,3] - results[:,5])  )
        moments_E2_diff_stdm.append( numpy.std(results[:,3] - results[:,5],ddof=1)/numpy.sqrt(n_res)  )
        moments_E2_diff_medn.append( numpy.median(results[:,3] - results[:,5])  )

        hsmcorr_E1_diff_mean.append( numpy.mean(results[:,6] - results[:,8])  )
        hsmcorr_E1_diff_stdm.append( numpy.std(results[:,6] - results[:,8],ddof=1)/numpy.sqrt(n_res)  )
        hsmcorr_E1_diff_medn.append( numpy.median(results[:,6] - results[:,8])  )
        hsmcorr_E2_diff_mean.append( numpy.mean(results[:,7] - results[:,9])  )
        hsmcorr_E2_diff_stdm.append( numpy.std(results[:,7] - results[:,9],ddof=1)/numpy.sqrt(n_res)  )
        hsmcorr_E2_diff_medn.append( numpy.median(results[:,7] - results[:,9])  )

        moments_sigma_diff_mean.append( numpy.mean(results[:,10] - results[:,11])  )
        moments_sigma_diff_stdm.append( numpy.std(results[:,10] - results[:,11],ddof=1)/numpy.sqrt(n_res)  )
        moments_sigma_diff_medn.append( numpy.median(results[:,10] - results[:,11])  )
        hsmcorr_sigma_diff_mean.append( numpy.mean(results[:,12] - results[:,13])  )
        hsmcorr_sigma_diff_stdm.append( numpy.std(results[:,12] - results[:,13],ddof=1)/numpy.sqrt(n_res)  )
        hsmcorr_sigma_diff_medn.append( numpy.median(results[:,12] - results[:,13])  )

    # yaml is bad at converting lists of floats in scientific notation to floats
    values_float = map(float,param['values'])

    # set some plot parameters
    fig_xsize,fig_ysize,legend_ncol,legend_loc = 12,10,2,3

    # plot figures for moments, hsmcorr, and sigma
    pylab.figure(1,figsize=(fig_xsize,fig_ysize))
    pylab.title('Weighted moments - uncorrected')
    pylab.xscale('log')
    pylab.errorbar(param['values'],moments_E1_diff_mean,yerr=moments_E1_diff_stdm,fmt='b+-',label='mean (fft-phot) e1')
    pylab.errorbar(param['values'],moments_E2_diff_mean,yerr=moments_E2_diff_stdm,fmt='rx-',label='mean (fft-phot) e2')
    pylab.plot(param['values'],moments_E1_diff_medn,'b+--',label='median (fft-phot) e1')
    pylab.plot(param['values'],moments_E2_diff_medn,'rx--',label='median (fft-phot) e2')
    pylab.ylabel('Ei_fft - Ei_photon')
    pylab.xlabel(param_name)
    pylab.xlim([min(values_float)*0.5, max(values_float)*1.5])
    pylab.legend(ncol=legend_ncol,loc=legend_loc,mode="expand")
    filename_fig = 'fig.moments.%s.%s.png' % (config['filename_config'],param_name)
    pylab.savefig(filename_fig)
    if config['debug']: pylab.show()
    pylab.close()
    logger.info('saved figure %s' % filename_fig)

    pylab.figure(2,figsize=(fig_xsize,fig_ysize))
    pylab.title('Weighted moments - corrected')
    pylab.xscale('log')
    pylab.errorbar(param['values'],hsmcorr_E1_diff_mean,yerr=hsmcorr_E1_diff_stdm,fmt='b+-',label='mean (fft-phot) e1')
    pylab.errorbar(param['values'],hsmcorr_E2_diff_mean,yerr=hsmcorr_E2_diff_stdm,fmt='rx-',label='mean (fft-phot) e2')
    pylab.plot(param['values'],hsmcorr_E1_diff_medn,'b+--',label='median (fft-phot) e1')
    pylab.plot(param['values'],hsmcorr_E2_diff_medn,'rx--',label='median (fft-phot) e2')
    pylab.ylabel('Ei_fft - Ei_photon')
    pylab.xlabel(param_name)
    pylab.xlim([min(values_float)*0.5, max(values_float)*1.5])
    pylab.legend(ncol=legend_ncol,loc=legend_loc,mode="expand")
    filename_fig = 'fig.hsmcorr.%s.%s.png' % (config['filename_config'],param_name)
    pylab.savefig(filename_fig)
    if config['debug']: pylab.show()
    pylab.close()
    logger.info('saved figure %s' % filename_fig)

    pylab.figure(3,figsize=(fig_xsize,fig_ysize))
    pylab.title('Measured size')
    pylab.xscale('log')
    pylab.errorbar(param['values'],moments_sigma_diff_mean,yerr=moments_sigma_diff_stdm,fmt='b+-',label='mean (fft-phot) moments')
    pylab.errorbar(param['values'],hsmcorr_sigma_diff_mean,yerr=hsmcorr_sigma_diff_stdm,fmt='rx-',label='mean (fft-phot) hsmcorr')
    pylab.plot(param['values'],moments_sigma_diff_medn,'b+--',label='median (fft-phot) moments')
    pylab.plot(param['values'],hsmcorr_sigma_diff_medn,'rx--',label='median (fft-phot) hsmcorr')
    pylab.ylabel('sigma_fft - sigma_photon')
    pylab.xlabel(param_name)
    pylab.xlim([min(values_float)*0.5, max(values_float)*1.5])
    pylab.legend(ncol=legend_ncol,loc=legend_loc,mode="expand")
    filename_fig = 'fig.sigma.%s.%s.png' % (config['filename_config'],param_name)
    pylab.savefig(filename_fig)
    if config['debug']: pylab.show()
    pylab.close()
    logger.info('saved figure %s' % filename_fig)

if __name__ == "__main__":


    description = 'Use data produced by reconvolution_validation.py to produce various plots.'

    # parse arguments
    parser = argparse.ArgumentParser(description=description, add_help=True)
    parser.add_argument('filename_config', type=str, help='yaml config file, see photon_vs_fft.yaml for example.')
    parser.add_argument('--debug', action="store_true", help='run with debug verbosity', default=False)
    args = parser.parse_args()

    # set up logger
    if args.debug: logger_level = 'logging.DEBUG'
    else:  logger_level = 'logging.INFO'
    logging.basicConfig(format="%(message)s", level=eval(logger_level), stream=sys.stdout)
    logger = logging.getLogger("photon_vs_fft") 

    # load the configuration file
    config = yaml.load(open(args.filename_config,'r'))
    config['debug'] = args.debug
    config['filename_config'] = args.filename_config

    # save a plot for each of the varied parameters
    for param_name in config['vary_params'].keys():
        PlotStatsForParam(config,param_name)
