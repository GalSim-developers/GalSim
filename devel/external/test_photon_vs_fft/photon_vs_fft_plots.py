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

"""@file photon_vs_fft_plots.py 
Create various plots from data produced by photon_vs_fft.py.
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
import matplotlib.pyplot as plt
import galsim
import copy

HSM_ERROR_VALUE = -99
NO_PSF_VALUE    = -98

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

    # prepare the output dict and initialise lists
    bias_list = {'m1' : [], 'm2' : [], 'c1' : [], 'c2' : [], 
                        'm1_std' : [], 'm2_std' : [], 'c1_std' : [], 'c2_std' : [],
                        'cs' : [], 'ms' : [], 'cs_std' : [], 'ms_std' : []}
    bias_moments_list = copy.deepcopy(bias_list)
    bias_hsmcorr_list = copy.deepcopy(bias_list)

    # loop over values changed for the varied parameter
    for iv,value in enumerate(param['values']):

        # get the filename for the results file
        filename_results_pht = 'results.%s.%s.%03d.pht.cat' % (
            config['filename_config'],param_name,iv)
        filename_results_fft = 'results.%s.%s.%03d.fft.cat' % (
            config['filename_config'],param_name,iv)

        # get the path for the results files
        filepath_results_fft = os.path.join(config['results_dir'],filename_results_fft)
        filepath_results_pht = os.path.join(config['results_dir'],filename_results_pht)

        logger.debug('parameter %s, index %03d, value %2.4e' % (param_name,iv,float(value)))

        # if there is no .fft or .pht file, look for the default to compare it against
        if not os.path.isfile(filepath_results_pht):
            logger.info('file %s not found, looking for defaults' % filepath_results_pht)
            filename_results_pht = 'results.%s.default.pht.cat' % (config['filepath_config'])
            filepath_results_pht = os.path.join(config['results_dir'],filename_results_pht)     
            if not os.path.isfile(filepath_results_pht):
                raise NameError('file %s not found' % filepath_results_pht)

        if not os.path.isfile(filepath_results_fft):
            logger.info('file %s not found, looking for defaults' % filepath_results_fft)
            filename_results_fft = 'results.%s.default.fft.cat' % (config['filepath_config'])
            filepath_results_fft = os.path.join(config['results_dir'],filename_results_fft)
            if not os.path.isfile(filepath_results_fft):
                raise NameError('file %s not found' % filepath_results_fft)

        # measure m and c biases
        bias_moments,bias_hsmcorr = GetBias(config,filepath_results_pht,filepath_results_fft)

        # append results lists  - slightly clunky way
        bias_moments_list['m1'].append(bias_moments['m1'])
        bias_moments_list['m2'].append(bias_moments['m2'])
        bias_moments_list['m1_std'].append(bias_moments['m1_std'])
        bias_moments_list['m2_std'].append(bias_moments['m2_std'])
        bias_moments_list['c1'].append(bias_moments['c1'])
        bias_moments_list['c2'].append(bias_moments['c2'])
        bias_moments_list['c1_std'].append(bias_moments['c1_std'])
        bias_moments_list['c2_std'].append(bias_moments['c2_std'])
        bias_moments_list['cs'].append(bias_moments['cs'])
        bias_moments_list['ms'].append(bias_moments['ms'])
        bias_moments_list['cs_std'].append(bias_moments['cs_std'])
        bias_moments_list['ms_std'].append(bias_moments['ms_std'])

        bias_hsmcorr_list['m1'].append(bias_hsmcorr['m1'])
        bias_hsmcorr_list['m2'].append(bias_hsmcorr['m2'])
        bias_hsmcorr_list['m1_std'].append(bias_hsmcorr['m1_std'])
        bias_hsmcorr_list['m2_std'].append(bias_hsmcorr['m2_std'])
        bias_hsmcorr_list['c1'].append(bias_hsmcorr['c1'])
        bias_hsmcorr_list['c2'].append(bias_hsmcorr['c2'])
        bias_hsmcorr_list['c1_std'].append(bias_hsmcorr['c1_std'])
        bias_hsmcorr_list['c2_std'].append(bias_hsmcorr['c2_std'])
        bias_hsmcorr_list['cs'].append(bias_hsmcorr['cs'])
        bias_hsmcorr_list['ms'].append(bias_hsmcorr['ms'])
        bias_hsmcorr_list['cs_std'].append(bias_hsmcorr['cs_std'])
        bias_hsmcorr_list['ms_std'].append(bias_hsmcorr['ms_std'])

    # yaml is bad at converting lists of floats in scientific notation to floats
    values_float = map(float,param['values'])

    # set some plot parameters
    fig_xsize,fig_ysize,legend_ncol,legend_loc = 12,10,2,3

    # plot figures for moments, hsmcorr, and sigma
    pylab.figure(1,figsize=(fig_xsize,fig_ysize))
    pylab.title('Weighted moments - uncorrected')
    pylab.xscale('log')
    pylab.errorbar(param['values'],bias_moments_list['m1'],yerr=bias_moments_list['m1_std'],
        fmt='b+-',label='G1')
    pylab.errorbar(param['values'],bias_moments_list['m2'],yerr=bias_moments_list['m2_std'],
        fmt='rx-',label='G2')
    pylab.ylabel('slope of (Gi_phot-Gi_fft) vs Gi_fft')
    pylab.xlabel(param_name)
    pylab.xlim([min(values_float)*0.5, max(values_float)*1.5])
    pylab.legend(ncol=legend_ncol,loc=legend_loc,mode="expand")
    filename_fig = 'fig.moments.%s.%s.png' % (config['filename_config'],param_name)
    pylab.savefig(filename_fig)
    # if config['debug']: pylab.show()
    pylab.close()
    logger.info('saved figure %s' % filename_fig)

    # plot figures for moments, hsmcorr, and sigma
    pylab.figure(2,figsize=(fig_xsize,fig_ysize))
    pylab.title('Weighted moments - corrected')
    pylab.xscale('log')
    pylab.errorbar(param['values'],bias_hsmcorr_list['m1'],yerr=bias_hsmcorr_list['m1_std'],
        fmt='b+-',label='G1')
    pylab.errorbar(param['values'],bias_hsmcorr_list['m2'],yerr=bias_hsmcorr_list['m2_std'],
        fmt='rx-',label='G2')
    pylab.ylabel('slope of (Gi_phot-Gi_fft) vs Gi_fft')
    pylab.xlabel(param_name)
    pylab.xlim([min(values_float)*0.5, max(values_float)*1.5])
    pylab.legend(ncol=legend_ncol,loc=legend_loc,mode="expand")
    filename_fig = 'fig.hsmcorr.%s.%s.png' % (config['filename_config'],param_name)
    pylab.savefig(filename_fig)
    # if config['debug']: pylab.show()
    pylab.close()
    logger.info('saved figure %s' % filename_fig)

    # plot figures for moments, hsmcorr, and sigma
    pylab.figure(3,figsize=(fig_xsize,fig_ysize))
    pylab.title('measured size')
    pylab.xscale('log')
    pylab.errorbar(param['values'],bias_moments_list['ms'],yerr=bias_moments_list['ms_std'],
        fmt='b+-',label='G1')
    pylab.errorbar(param['values'],bias_hsmcorr_list['ms'],yerr=bias_hsmcorr_list['ms_std'],
        fmt='rx-',label='G2')
    pylab.ylabel('slope of (size_phot-size_fft) vs size_fft')
    pylab.xlabel(param_name)
    pylab.xlim([min(values_float)*0.5, max(values_float)*1.5])
    pylab.legend(ncol=legend_ncol,loc=legend_loc,mode="expand")
    filename_fig = 'fig.sigma.%s.%s.png' % (config['filename_config'],param_name)
    pylab.savefig(filename_fig)
    # if config['debug']: pylab.show()
    pylab.close()
    logger.info('saved figure %s' % filename_fig)

def _getLineFit(X,y,sig):
        """
        @brief get linear least squares fit with uncertainty estimates
        y(X) = b*X + a
        see numerical recipes 15.2.9
        @param X    function arguments 
        @param y    function values
        @param sig  function values standard deviations   
        @return a - additive 
        @return b - multiplicative
        @return C - covariance matrix
        """
        
        import numpy

        invsig2 = sig**-2;
        
        S  = numpy.sum(invsig2)
        Sx = numpy.dot(X,invsig2)
        Sy = numpy.dot(y,invsig2)
        Sxx = numpy.dot(invsig2*X,X)
        Sxy = numpy.dot(invsig2*X,y)
        
        D = S*Sxx - Sx**2
        a = (Sxx*Sy - Sx*Sxy)/D
        b = (S*Sxy  - Sx*Sy)/D
        
        Cab = numpy.zeros((2,2))
        Cab[0,0] = Sxx/D
        Cab[1,1] = S/D
        Cab[1,0] = -Sx/D
        Cab[0,1] = Cab[1,0]
        
        return a,b,Cab

def GetBias(config,filename_results_pht,filename_results_fft):
    """
    @brief load results, calculate the slope and offset for bias (g1_pht-g1_fft) vs g1_fft
    @param config dict used to create the simulations
    @param filename_results_pht results file for the fft 
    @param filename_results_fft results file for the pht
    @return dict with fields c1,m1,c2,m2,cs,ms,c1_std,c2_std,m1_std,m2_std,cs_std,ms_std
            ms and cs - line fit parameters for size
    """

    # load results
    results_pht = numpy.loadtxt(filename_results_pht)
    results_fft = numpy.loadtxt(filename_results_fft)
    logging.debug('opening files %s, %s ' % ( filename_results_pht, filename_results_fft ))

    # intersection of results in case some data is missing
    common_ids = list(set(results_pht[:,0].astype(numpy.int64)).intersection( set(results_fft[:,0].astype(numpy.int64))))  
    indices_pht = [list(results_pht[:,0].astype(numpy.int64)).index(cid) for cid in common_ids]
    indices_fft = [list(results_fft[:,0].astype(numpy.int64)).index(cid) for cid in common_ids]
    results_pht = results_pht[indices_pht,:]
    results_fft = results_fft[indices_fft,:]



    # get the number of points
    n_res_all = results_pht.shape[0]
    # clean the HSM errors
    for col in range(1,12):
        select1 = results_fft[:,col] != HSM_ERROR_VALUE 
        select2 = results_pht[:,col] != HSM_ERROR_VALUE
        select = numpy.logical_and(select1,select2)
        results_pht = results_pht[select,:]
        results_fft = results_fft[select,:]
    n_res = results_pht.shape[0]

    logging.info('opened files %s, %s with %d valid measurements (out of %d)',filename_results_pht,
        filename_results_fft,n_res,n_res_all)

    # unpack the table
    #id,G1_moments,G2_moments,G1_hsmcorr,G2_hsmcorr,moments_sigma,hsmcorr_sigma,err_g1obs,err_g2obs,
    #err_g1hsm,err_g2hsm,err_sigma,err_sigma_hsm
    moments_fft_G1         = results_fft[:,1]
    moments_fft_G2         = results_fft[:,2]
    hsmcorr_fft_G1         = results_fft[:,3]
    hsmcorr_fft_G2         = results_fft[:,4]
    moments_fft_sigma      = results_fft[:,5]
    hsmcorr_fft_sigma      = results_fft[:,6]
    moments_pht_G1         = results_pht[:,1]
    moments_pht_G2         = results_pht[:,2]
    hsmcorr_pht_G1         = results_pht[:,3]
    hsmcorr_pht_G2         = results_pht[:,4] 
    moments_pht_sigma      = results_pht[:,5]
    hsmcorr_pht_sigma      = results_pht[:,6]
    moments_pht_G1_std     = results_pht[:,7]
    moments_pht_G2_std     = results_pht[:,8]
    hsmcorr_pht_G1_std     = results_pht[:,9]
    hsmcorr_pht_G2_std     = results_pht[:,10]
    moments_pht_sigma_std  = results_pht[:,11]
    hsmcorr_pht_sigma_std  = results_pht[:,12]

    # if in debug mode save the plots of (g1_pht - g1_fft) vs g1_fft, etc
    if config['debug']:

        pylab.figure(figsize=(20,10))
        
        pylab.subplot(1,3,1)
        pylab.errorbar(moments_fft_G1, moments_pht_G1-moments_fft_G1, yerr=moments_pht_G1_std, 
            fmt='b.')
        pylab.xlabel('g1_fft')
        pylab.ylabel('g1_pht - g1_fft')

        pylab.subplot(1,3,2)
        pylab.errorbar(moments_fft_G2, moments_pht_G2-moments_fft_G2, yerr=moments_pht_G2_std, 
            fmt='b.')
        pylab.xlabel('g2_fft')
        pylab.ylabel('g2_pht - g2_fft')

        pylab.subplot(1,3,3)
        pylab.errorbar(moments_fft_sigma, moments_pht_sigma-moments_fft_sigma,
             yerr=moments_pht_sigma_std, fmt='b.')
        pylab.xlabel('sigma_fft')
        pylab.ylabel('sigma_pht - sigma_fft')

        # save the files
        name1 = os.path.basename(filename_results_pht).replace('photon_vs_fft.','').replace('yaml.',
            '').replace('results.','')
        name2 = os.path.basename(filename_results_fft).replace('photon_vs_fft.','').replace('yaml.',
            '').replace('results.','')
        filename_fig = 'fig.bias.%s.%s.png' % (name1,name2)
        pylab.savefig(filename_fig)
        pylab.close()
        logging.debug('saved figure %s' % filename_fig) 

    # get line fits for moments
    c1,m1,cov1 = _getLineFit(moments_fft_G1,moments_pht_G1-moments_fft_G1,moments_pht_G1_std)
    c2,m2,cov2 = _getLineFit(moments_fft_G2,moments_pht_G2-moments_fft_G2,moments_pht_G2_std)
    cs,ms,covs = _getLineFit(moments_fft_sigma,moments_pht_sigma-moments_fft_sigma,
        moments_pht_sigma_std)
    # create result dict
    bias_moments = {'c1' : c1, 'm1': m1,  'c2' : c2, 'm2': m2, 'cs' : cs, 'ms' : ms , 
                    'c1_std' : numpy.sqrt(cov1[0,0]),
                    'c2_std' : numpy.sqrt(cov2[0,0]),
                    'm1_std' : numpy.sqrt(cov1[1,1]),
                    'm2_std' : numpy.sqrt(cov2[1,1]),
                    'cs_std' : numpy.sqrt(covs[0,0]),
                    'ms_std' : numpy.sqrt(covs[1,1]) }
    
    # get line fits for hsmcorr
    c1,m1,cov1 = _getLineFit(hsmcorr_fft_G1,hsmcorr_pht_G1-hsmcorr_fft_G1,hsmcorr_pht_G1_std)
    c2,m2,cov2 = _getLineFit(hsmcorr_fft_G2,hsmcorr_pht_G2-hsmcorr_fft_G2,hsmcorr_pht_G2_std)
    cs,ms,covs = _getLineFit(hsmcorr_fft_sigma,hsmcorr_pht_sigma-hsmcorr_fft_sigma,
        hsmcorr_pht_sigma_std)
    # create result dict
    bias_hsmcorr = {'c1' : c1, 'm1': m1,  'c2' : c2, 'm2': m2, 'cs' : cs, 'ms' : ms , 
                    'c1_std' : numpy.sqrt(cov1[0,0]),
                    'c2_std' : numpy.sqrt(cov2[0,0]),
                    'm1_std' : numpy.sqrt(cov1[1,1]),
                    'm2_std' : numpy.sqrt(cov2[1,1]),
                    'cs_std' : numpy.sqrt(covs[0,0]),
                    'ms_std' : numpy.sqrt(covs[1,1]) }

    return bias_moments,bias_hsmcorr
    
if __name__ == "__main__":


    description = 'Use data produced by photon_vs_fft.py to produce various plots.'

    # parse arguments
    parser = argparse.ArgumentParser(description=description, add_help=True)
    parser.add_argument('filename_config', type=str, 
        help='yaml config file, see photon_vs_fft.yaml for example.')
    parser.add_argument('--debug', action="store_true", 
        help='run with debug verbosity', default=False)
    parser.add_argument('-rd','--results_dir',type=str, action='store', default='.', 
        help="dir where the results files are, default '.'")    
    args = parser.parse_args()

    # set up logger
    if args.debug: logger_level = 'logging.DEBUG'
    else:  logger_level = 'logging.INFO'
    logging.basicConfig(format="%(message)s", level=eval(logger_level), stream=sys.stdout)
    logger = logging.getLogger("photon_vs_fft") 

    # load the configuration file
    config = yaml.load(open(args.filename_config,'r'))
    config['debug'] = args.debug
    config['filepath_config'] = args.filename_config
    config['filename_config'] = os.path.basename(config['filepath_config'])
    config['results_dir'] = args.results_dir

    # analyse the defaults, useful only to make the debug plot of g1_pht-g1_fft vs g1_fft
    filename_results_fft = 'results.%s.default.fft.cat' % (config['filename_config'])
    filename_results_pht = 'results.%s.default.pht.cat' % (config['filename_config'])
    filepath_results_fft = os.path.join(args.results_dir,filename_results_fft)
    filepath_results_pht = os.path.join(args.results_dir,filename_results_pht)
    GetBias(config,filepath_results_pht,filepath_results_fft)
    
    # save a plot for each of the varied parameters
    for param_name in config['vary_params'].keys():
        PlotStatsForParam(config,param_name)


