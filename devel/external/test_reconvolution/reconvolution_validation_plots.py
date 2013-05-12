"""@file photon_vs_reconv_plots.py 
Create various plot from data produced by photon_vs_reconv.py.
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
    Save plots for the results of photon_vs_reconv, when param_name is varied.
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
        filename_results_direct = 'results.%s.%s.%03d.direct.cat' % (
            config['filename_config'],param_name,iv)
        filename_results_reconv = 'results.%s.%s.%03d.reconv.cat' % (
            config['filename_config'],param_name,iv)

        # get the path for the results files
        filepath_results_reconv = os.path.join(config['results_dir'],filename_results_reconv)
        filepath_results_direct = os.path.join(config['results_dir'],filename_results_direct)

        logger.debug('parameter %s, index %03d, value %2.4e' % (param_name,iv,value))

        # if there is no .reconv or .direct file, look for the default to compare it against
        if not os.path.isfile(filepath_results_direct):
            logger.info('file %s not found, looking for defaults' % filepath_results_direct)
            filename_results_direct = 'results.%s.default.direct.cat' % (config['filepath_config'])
            filepath_results_direct = os.path.join(config['results_dir'],filename_results_direct)     
            if not os.path.isfile(filepath_results_direct):
                raise NameError('file %s not found' % filepath_results_direct)

        if not os.path.isfile(filepath_results_reconv):
            logger.info('file %s not found, looking for defaults' % filepath_results_reconv)
            filename_results_reconv = 'results.%s.default.reconv.cat' % (config['filepath_config'])
            filepath_results_reconv = os.path.join(config['results_dir'],filename_results_reconv)
            if not os.path.isfile(filepath_results_reconv):
                raise NameError('file %s not found' % filepath_results_reconv)

        # measure m and c biases
        bias_moments,bias_hsmcorr = GetBias(config,filepath_results_direct,filepath_results_reconv)

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

    # plot figures for moments
    pylab.figure(1,figsize=(fig_xsize,fig_ysize))
    pylab.title('Weighted moments - uncorrected')
    pylab.xscale('log')
    pylab.errorbar(param['values'],bias_moments_list['m1'],yerr=bias_moments_list['m1_std'],
        fmt='b+-',label='G1')
    pylab.errorbar(param['values'],bias_moments_list['m2'],yerr=bias_moments_list['m2_std'],
        fmt='rx-',label='G2')
    pylab.ylabel('slope of (Gi_phot-Gi_reconv) vs Gi_reconv')
    pylab.xlabel(param_name)
    pylab.xlim([min(values_float)*0.5, max(values_float)*1.5])
    pylab.legend(ncol=legend_ncol,loc=legend_loc,mode="expand")
    filename_fig = 'fig.moments.%s.%s.png' % (config['filename_config'],param_name)
    pylab.savefig(filename_fig)
    # if config['debug']: pylab.show()
    pylab.close()
    logger.info('saved figure %s' % filename_fig)

    # plot figures for hsmcorr
    pylab.figure(2,figsize=(fig_xsize,fig_ysize))
    pylab.title('Weighted moments - corrected')
    pylab.xscale('log')
    pylab.errorbar(param['values'],bias_hsmcorr_list['m1'],yerr=bias_hsmcorr_list['m1_std'],
        fmt='b+-',label='G1')
    pylab.errorbar(param['values'],bias_hsmcorr_list['m2'],yerr=bias_hsmcorr_list['m2_std'],
        fmt='rx-',label='G2')
    pylab.ylabel('slope of (Gi_phot-Gi_reconv) vs Gi_reconv')
    pylab.xlabel(param_name)
    pylab.xlim([min(values_float)*0.5, max(values_float)*1.5])
    pylab.legend(ncol=legend_ncol,loc=legend_loc,mode="expand")
    filename_fig = 'fig.hsmcorr.%s.%s.png' % (config['filename_config'],param_name)
    pylab.savefig(filename_fig)
    # if config['debug']: pylab.show()
    pylab.close()
    logger.info('saved figure %s' % filename_fig)

    # # plot figures for moments, hsmcorr, and sigma
    # pylab.figure(3,figsize=(fig_xsize,fig_ysize))
    # pylab.title('measured size')
    # pylab.xscale('log')
    # pylab.errorbar(param['values'],bias_moments_list['ms'],yerr=bias_moments_list['ms_std'],
    #     fmt='b+-',label='G1')
    # pylab.errorbar(param['values'],bias_hsmcorr_list['ms'],yerr=bias_hsmcorr_list['ms_std'],
    #     fmt='rx-',label='G2')
    # pylab.ylabel('slope of (size_phot-size_reconv) vs size_reconv')
    # pylab.xlabel(param_name)
    # pylab.xlim([min(values_float)*0.5, max(values_float)*1.5])
    # pylab.legend(ncol=legend_ncol,loc=legend_loc,mode="expand")
    # filename_fig = 'fig.sigma.%s.%s.png' % (config['filename_config'],param_name)
    # pylab.savefig(filename_fig)
    # # if config['debug']: pylab.show()
    # pylab.close()
    # logger.info('saved figure %s' % filename_fig)

def _getLineFit(X,y,sig):
        """
        @brief get linear least squares fit with uncertainity estimates
        y(X) = b*X + a
        see numerical recipies 15.2.9
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

def GetBias(config,filename_results_direct,filename_results_reconv):
    """
    @brief load results, calculate the slope and offset for bias (g1_direct-g1_reconv) vs g1_reconv
    @param config dict used to create the simulations
    @param filename_results_direct results file for the reconv 
    @param filename_results_reconv results file for the direct
    @return dict with fields c1,m1,c2,m2,cs,ms,c1_std,c2_std,m1_std,m2_std,cs_std,ms_std
            ms and cs - line fit parameters for size
    """

    n_shears = config['reconvolution_validation_settings']['n_shears']
    n_angles = config['reconvolution_validation_settings']['n_angles']
    n_gals = config['reconvolution_validation_settings']['n_gals']

    bias_moments_list = []
    bias_hsmcorr_list = []

    # load results
    results_direct = numpy.loadtxt(filename_results_direct)
    results_reconv = numpy.loadtxt(filename_results_reconv)

    # check if ring test is complete, we should have n_angles results for each galaxy and each shear
    for gi in range(n_gals):
    
        moments_reconv_G1 = []
        moments_reconv_G2 = []
        hsmcorr_reconv_G1 = []
        hsmcorr_reconv_G2 = []
        moments_direct_G1 = []
        moments_direct_G2 = []
        hsmcorr_direct_G1 = []
        hsmcorr_direct_G2 = []
        true_G1 = []
        true_G2 = []
    
        n_used_shears = 0
        for si in range(n_shears):


            start_id = gi*si
            end_id = gi*si + n_angles
            select_reconv = numpy.logical_and(results_reconv[:,0] >= start_id,results_reconv[:,0] < end_id)
            select_direct = numpy.logical_and(results_direct[:,0] >= start_id,results_direct[:,0] < end_id)
            n_found_angles_reconv = sum(select_reconv)
            n_found_angles_direct = sum(select_direct)
            
            if (n_found_angles_reconv != n_angles) or (n_found_angles_direct != n_angles):
                logging.debug('gal %d shear %d found %d reconv and %d direct angles',(gi,si,n_found_angles_reconv,n_found_angles_direct))
                continue

            n_used_shears += 1

            moments_reconv_G1.append(         numpy.mean( results_reconv[select_reconv,1]) )
            moments_reconv_G2.append(         numpy.mean( results_reconv[select_reconv,2]) )
            hsmcorr_reconv_G1.append(         numpy.mean( results_reconv[select_reconv,3]) )
            hsmcorr_reconv_G2.append(         numpy.mean( results_reconv[select_reconv,4]) )
            moments_direct_G1.append(         numpy.mean( results_direct[select_direct,1]) )
            moments_direct_G2.append(         numpy.mean( results_direct[select_direct,2]) )
            hsmcorr_direct_G1.append(         numpy.mean( results_direct[select_direct,3]) )
            hsmcorr_direct_G2.append(         numpy.mean( results_direct[select_direct,4]) )
            true_G1.append( config['reconvolved_images']['gal']['shear']['items'][si]['g1'] )
            true_G2.append( config['reconvolved_images']['gal']['shear']['items'][si]['g2'] )

        moments_reconv_G1 = numpy.asarray(moments_reconv_G1)
        moments_reconv_G2 = numpy.asarray(moments_reconv_G2)
        hsmcorr_reconv_G1 = numpy.asarray(hsmcorr_reconv_G1)
        hsmcorr_reconv_G2 = numpy.asarray(hsmcorr_reconv_G2)
        moments_direct_G1 = numpy.asarray(moments_direct_G1)
        moments_direct_G2 = numpy.asarray(moments_direct_G2)
        hsmcorr_direct_G1 = numpy.asarray(hsmcorr_direct_G1)
        hsmcorr_direct_G2 = numpy.asarray(hsmcorr_direct_G2)
        true_G1 = numpy.asarray(true_G1)
        true_G2 = numpy.asarray(true_G2)

        # get line fits for moments
        c1,m1,cov1 = _getLineFit(true_G1,moments_direct_G1-moments_reconv_G1,
            numpy.ones(moments_direct_G1.shape))
        c2,m2,cov2 = _getLineFit(true_G2,moments_direct_G2-moments_reconv_G2,
            numpy.ones(moments_direct_G2.shape))
        
        # create result dict
        bias_moments = {'c1' : c1, 'm1': m1,  'c2' : c2, 'm2': m2, 
                        'c1_std' : 0. ,
                        'c2_std' : 0. ,
                        'm1_std' : 0. ,
                        'm2_std' : 0. }
        
        # get line fits for hsmcorr
        c1,m1,cov1 = _getLineFit(hsmcorr_reconv_G1,hsmcorr_direct_G1-hsmcorr_reconv_G1,
            numpy.ones(hsmcorr_direct_G1.shape))
        c2,m2,cov2 = _getLineFit(hsmcorr_reconv_G2,hsmcorr_direct_G2-hsmcorr_reconv_G2,
            numpy.ones(hsmcorr_direct_G2.shape))
        
        # create result dict
        bias_hsmcorr = {'c1' : c1, 'm1': m1,  'c2' : c2, 'm2': m2, 
                        'c1_std' : 0. ,
                        'c2_std' : 0. ,
                        'm1_std' : 0. ,
                        'm2_std' : 0. }
                        
        logging.debug( 'gal %d used %d, m1 = %2.3e, m2=%2.3e ' % (gi,n_used_shears,bias_moments['m1'],bias_moments['m2']))


        bias_moments_list.append(bias_moments)
        bias_hsmcorr_list.append(bias_hsmcorr)

    return bias_moments_list,bias_hsmcorr_list
    
if __name__ == "__main__":


    description = 'Use data produced by photon_vs_reconv.py to produce various plots.'

    # parse arguments
    parser = argparse.ArgumentParser(description=description, add_help=True)
    parser.add_argument('filename_config', type=str, 
        help='yaml config file, see photon_vs_reconv.yaml for example.')
    parser.add_argument('--debug', action="store_true", 
        help='run with debug verbosity', default=False)
    parser.add_argument('-rd','--results_dir',type=str, action='store', default='.', 
        help="dir where the results files are, default '.'")    
    args = parser.parse_args()

    # set up logger
    if args.debug: logger_level = 'logging.DEBUG'
    else:  logger_level = 'logging.INFO'
    logging.basicConfig(format="%(message)s", level=eval(logger_level), stream=sys.stdout)
    logger = logging.getLogger("photon_vs_reconv") 

    # load the configuration file
    config = yaml.load(open(args.filename_config,'r'))
    config['debug'] = args.debug
    config['filepath_config'] = args.filename_config
    config['filename_config'] = os.path.basename(config['filepath_config'])
    config['results_dir'] = args.results_dir

    # analyse the defaults, useful only to make the debug plot of g1_direct-g1_reconv vs g1_reconv
    filename_results_reconv = 'results.%s.default.reconv.cat' % (config['filename_config'])
    filename_results_direct = 'results.%s.default.direct.cat' % (config['filename_config'])
    filepath_results_reconv = os.path.join(args.results_dir,filename_results_reconv)
    filepath_results_direct = os.path.join(args.results_dir,filename_results_direct)
    GetBias(config,filepath_results_direct,filepath_results_reconv)
    
    # save a plot for each of the varied parameters
    for param_name in config['vary_params'].keys():
        PlotStatsForParam(config,param_name)


