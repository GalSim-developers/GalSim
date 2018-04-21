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
import matplotlib.pyplot as plt
import galsim
import copy

HSM_ERROR_VALUE = -99
NO_PSF_VALUE    = -98

def PlotStatsForParam(config,param_name):
    """
    @brief Save plots for the results of reconvolution_validation, when param_name is varied.
    @param config         galsim yaml config, which was used to produce the results, read by yaml
    @param param_name     varied parameter name, listed under config['vary_params'], for which
                          to create the plots
    """

    # get the shortcut to the dict corresponding to current varied parameter
    param = config['vary_params'][param_name]

    # prepare the output dict and initialise lists
    bias_list = {'m1' : [], 'm2' : [], 'c1' : [], 'c2' : [], 
                        'm1_std' : [], 'm2_std' : [], 'c1_std' : [], 'c2_std' : [] }
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

        logging.info('parameter %s, index %03d, value %2.4e' % (param_name,iv,float(value)))

        # if there is no .reconv or .direct file, look for the default to compare it against
        if not os.path.isfile(filepath_results_direct):
            logging.info('file %s not found, looking for defaults' % filepath_results_direct)
            filename_results_direct = 'results.%s.default.direct.cat' % (config['filepath_config'])
            filepath_results_direct = os.path.join(config['results_dir'],filename_results_direct)     
            if not os.path.isfile(filepath_results_direct):
                raise NameError('file %s not found' % filepath_results_direct)

        if not os.path.isfile(filepath_results_reconv):
            logging.info('file %s not found, looking for defaults' % filepath_results_reconv)
            filename_results_reconv = 'results.%s.default.reconv.cat' % (config['filepath_config'])
            filepath_results_reconv = os.path.join(config['results_dir'],filename_results_reconv)
            if not os.path.isfile(filepath_results_reconv):
                raise NameError('file %s not found' % filepath_results_reconv)

        # measure m and c biases
        bias_moments,bias_hsmcorr = GetBias(config,filepath_results_direct,filepath_results_reconv)

        logging.info('bias_moments has %d points ' % len(bias_moments))

        # append results lists  - slightly clunky way
        bias_moments_list[iv] = bias_moments
        bias_hsmcorr_list[iv] = bias_hsmcorr


    # yaml is bad at converting lists of floats in scientific notation to floats
    values_float = map(float,param['values'])

    # get the tick labels
    values_float_ticklabels = map(str,values_float)

    # if very large value is used, put it closer to other points
    for ivf,vf in enumerate(values_float):
        if vf > 1e10:
            values_float_ticklabels[ivf] = str(vf)
            values_float_sorted = sorted(values_float) 
            values_float[ivf] = values_float_sorted[-2]*10

    # set some plot parameters
    fig_xsize,fig_ysize,legend_ncol,legend_loc = 12,10,2,3

    # plot figures for moments
    pylab.figure(1,figsize=(fig_xsize,fig_ysize))
    pylab.title('Weighted moments - uncorrected')
    pylab.xscale('log')

    # add the scattered m values to plot
    for iv,value in enumerate(param['values']):

        m1 = [b['m1'] for b in bias_moments_list[iv]]
        # m2 = [b['m2'] for b in bias_moments_list[iv]]

        print any(numpy.isnan(m1))

        pylab.plot(numpy.ones([len(m1)])*values_float[iv],m1,'x')
        pylab.errorbar(values_float[iv],numpy.mean(m1),yerr=numpy.std(m1,ddof=1),fmt='o',capsize=30)
        # pylab.plot(numpy.ones([len(m2)])*values_float[iv],m2,'o')

    print values_float
    print values_float_ticklabels
    pylab.xticks(values_float , values_float_ticklabels)
    pylab.yscale('symlog',linthreshy=1e-2)
    pylab.ylabel('m1')
    pylab.xlabel(param_name)
    pylab.xlim([min(values_float)*0.5, max(values_float)*1.5])
    pylab.legend(ncol=legend_ncol,loc=legend_loc,mode="expand")
    filename_fig = 'fig.moments.%s.%s.png' % (config['filename_config'],param_name)
    pylab.savefig(filename_fig)
    pylab.close()
    logging.info('saved figure %s' % filename_fig)

    # plot for HSM
    pylab.figure(2,figsize=(fig_xsize,fig_ysize))
    pylab.title('Weighted moments - corrected')
    pylab.xscale('log')
    pylab.yscale('symlog',linthreshy=1e-3)

    for iv,value in enumerate(param['values']):

        m1 = [b['m1'] for b in bias_hsmcorr_list[iv]]
        # m2 = [b['m2'] for b in bias_moments_list[iv]]

        pylab.plot(numpy.ones([len(m1)])*values_float[iv],m1,'x')
        pylab.errorbar(values_float[iv],numpy.mean(m1),yerr=numpy.std(m1,ddof=1),fmt='o',capsize=30)
        # pylab.plot(numpy.ones([len(m2)])*values_float[iv],m2,'o')

    pylab.ylabel('m1')
    pylab.xlabel(param_name)
    pylab.xlim([min(values_float)*0.5, max(values_float)*1.5])
    pylab.legend(ncol=legend_ncol,loc=legend_loc,mode="expand")
    filename_fig = 'fig.hsmcorr.%s.%s.png' % (config['filename_config'],param_name)
    pylab.savefig(filename_fig)
    pylab.close()
    logging.info('saved figure %s' % filename_fig)

    # now plot the log std of m1 as a function of parameter

    for iv,value in enumerate(param['values']):
            m1 = [b['m1'] for b in bias_moments_list[iv]]
            pylab.plot(values_float[iv],numpy.std(m1,ddof=1),'x')
           
    pylab.ylabel('std(m_1)',interpreter='latex')
    pylab.xlabel(param_name)
    pylab.xlim([min(values_float)*0.5, max(values_float)*1.5])
    # pylab.legend(ncol=legend_ncol,loc=legend_loc,mode="expand")
    filename_fig = 'fig.moments.stdm1.%s.%s.png' % (config['filename_config'],param_name)
    pylab.savefig(filename_fig)
    pylab.close()
    logging.info('saved figure %s' % filename_fig)

    
def _getLineFit(X,y,sig):
        """
        @brief get linear least squares fit with uncertainty estimates
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
    @return dict with fields c1,m1,c2,m2,c1_std,c2_std,m1_std,m2_std
    Errors on m and c are empty for now.
    """

    name1 = os.path.basename(filename_results_reconv).replace('results','').replace('yaml',
        '').replace('cat','').replace('..','.')
    name1= name1.strip('.')
    name2 = os.path.basename(filename_results_direct).replace('results','').replace('yaml',
        '').replace('cat','').replace('..','.')
    name2= name2.strip('.')

    filename_pickle = 'results.%s.%s.pickle' % (name1,name2)
    import cPickle as pickle

    if os.path.isfile(filename_pickle):
        logging.info('using existing results file %s' % filename_pickle)
        pickle_dict = pickle.load(open(filename_pickle))
        bias_moments_list = pickle_dict['moments']
        bias_hsmcorr_list = pickle_dict['hsmcorr']

    else:
        logging.info('file %s not found, analysing results' % filename_pickle)   

        # get number of shears, angles and galaxies, useful later
        n_shears = config['reconvolution_validation_settings']['n_shears']
        n_angles = config['reconvolution_validation_settings']['n_angles']
        n_gals = config['reconvolution_validation_settings']['n_gals']

        # initialise lists for results
        bias_moments_list = []
        bias_hsmcorr_list = []

        # load results
        results_direct = numpy.loadtxt(filename_results_direct)
        results_reconv = numpy.loadtxt(filename_results_reconv)

        # check if ring test is complete, we should have n_angles results for each galaxy and each shear
        for gi in range(n_gals):
        
            # initialise lists for results and truth
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
        
            # this will count how many shear we are using
            n_used_shears = 0

            # loop over shears
            for si in range(n_shears):

                # calculate indices of galaxies which belong to this ring test
                start_id = gi*si
                end_id = gi*si + n_angles

                # select galaxies from this ring
                select_reconv = numpy.logical_and(results_reconv[:,0] >= start_id,
                    results_reconv[:,0] < end_id)
                select_direct = numpy.logical_and(results_direct[:,0] >= start_id,
                    results_direct[:,0] < end_id)

                # count how many galaxies we got
                n_found_angles_reconv = sum(select_reconv)
                n_found_angles_direct = sum(select_direct)

                # initialise the variable which will tell us if to skip this shear
                skip_shear = False

                # do not include shear which has missing data           
                if (n_found_angles_reconv != n_angles) or (n_found_angles_direct != n_angles):
                    skip_shear = True

                # do not include the shear which has an error in one of the angles
                for col in range(1,7):
                    if any(results_reconv[select_reconv,col].astype(int) == HSM_ERROR_VALUE) or any(
                        results_direct[select_direct,col].astype(int) == HSM_ERROR_VALUE): 
                        skip_shear = True
                
                # continue with loop if bad ring
                if skip_shear:
                    logging.warning('gal %d shear %d has HSM errors or missing data- skipping' % (gi,si))
                    continue

                # increment the number of used shears
                n_used_shears += 1

                # get the shear from the ring
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

            # convert to numpy
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

            # get the shear bias for moments
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
            
            # get the shear bias for hsmcorr
            c1,m1,cov1 = _getLineFit(true_G1,hsmcorr_direct_G1-hsmcorr_reconv_G1,
                numpy.ones(hsmcorr_direct_G1.shape))
            c2,m2,cov2 = _getLineFit(true_G2,hsmcorr_direct_G2-hsmcorr_reconv_G2,
                numpy.ones(hsmcorr_direct_G2.shape))
            
            # create result dict
            bias_hsmcorr = {'c1' : c1, 'm1': m1,  'c2' : c2, 'm2': m2, 
                            'c1_std' : 0. ,
                            'c2_std' : 0. ,
                            'm1_std' : 0. ,
                            'm2_std' : 0. }
            


            if config['debug']:
                name1 = os.path.basename(filename_results_reconv).replace('results','').replace('yaml',
                    '').replace('cat','').replace('reconvolution_validation','')
                name1= name1.strip('.')
                name2 = os.path.basename(filename_results_direct).replace('results','').replace('yaml',
                    '').replace('cat','').replace('reconvolution_validation','').replace('..','.')
                name2= name2.strip('.')

                filename_fig = 'fig.linefit.%s.%s.%03d.png' % (name1,name2,gi)
                import pylab
                pylab.figure(figsize=(10,5))           
                pylab.plot(true_G1,moments_direct_G1-moments_reconv_G1,'bx');
                pylab.plot(true_G2,moments_direct_G2-moments_reconv_G2,'rx');
                pylab.plot(true_G1,true_G1*bias_moments['m1'] + bias_moments['c1'],'b-')
                pylab.plot(true_G2,true_G2*bias_moments['m2'] + bias_moments['c2'],'r-')
                x1,x2,y1,y2 = pylab.axis()
                pylab.axis((min(true_G1)*1.1,max(true_G1)*1.1,y1,y2))
                pylab.xlabel('true_Gi')
                pylab.ylabel('moments_direct_G1-moments_reconv_G1')
                pylab.legend(['G1','G2'])
                pylab.savefig(filename_fig)
                pylab.close()
                logging.info('saved figure %s' % filename_fig)


            logging.info( 'gal %3d used %3d shears, m1 = % 2.3e, m2=% 2.3e ' % (gi,n_used_shears,bias_moments['m1'],bias_moments['m2']))

            # append the results list
            bias_moments_list.append(bias_moments)
            bias_hsmcorr_list.append(bias_hsmcorr)

        # may want to scatter plot the m1,m2 of all galaxies in the results file
        if config['debug']:
            name1 = os.path.basename(filename_results_reconv).replace('results','').replace('yaml',
                '').replace('cat','').replace('reconvolution_validation','')
            name1= name1.strip('.')
            name2 = os.path.basename(filename_results_direct).replace('results','').replace('yaml',
                '').replace('cat','').replace('reconvolution_validation','').replace('..','.')
            name2= name2.strip('.')
            filename_fig = 'fig.mscatter.%s.%s.png' % (name1,name2)
            m1_list = numpy.asarray([b['m1'] for b in bias_moments_list])
            m2_list = numpy.asarray([b['m2'] for b in bias_moments_list])

            pylab.figure()
            pylab.scatter(m1,m2)
            pylab.savefig(filename_fig)
            pylab.close()

        pickle_dict = {'moments' : bias_moments_list, 'hsmcorr' : bias_hsmcorr_list}
        pickle.dump(pickle_dict,open(filename_pickle,'w'),protocol=2)
        logging.info('saved %s' % filename_pickle)

    return bias_moments_list,bias_hsmcorr_list
    
if __name__ == "__main__":


    description = 'Use data produced by reconvolution_validation.py to produce various plots.'

    # parse arguments
    parser = argparse.ArgumentParser(description=description, add_help=True)
    parser.add_argument('filename_config', type=str, 
        help='yaml config file, see reconvolution_validation.yaml for example.')
    parser.add_argument('--debug', action="store_true", 
        help='run with debug verbosity', default=False)
    parser.add_argument('-rd','--results_dir',type=str, action='store', default='.', 
        help="dir where the results files are, default '.'")    
    args = parser.parse_args()

    # set up logger
    if args.debug: logger_level = 'logging.DEBUG'
    else:  logger_level = 'logging.INFO'
    logging.basicConfig(format="%(message)s", level=eval(logger_level), stream=sys.stdout)

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
    logging.info('getting stats for default parameters')
    GetBias(config,filepath_results_direct,filepath_results_reconv)
    
    # save a plot for each of the varied parameters
    for param_name in config['vary_params'].keys():
        PlotStatsForParam(config,param_name)


