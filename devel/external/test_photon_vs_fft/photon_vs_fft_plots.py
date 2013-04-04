"""@file photon_vs_fft_plots.py 
Create various plot from data produced by photon_vs_fft.py.
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

hsm_error_value = -1
no_psf_value = -1
dirname_figs = 'figures'


def plotEllipticityBiases():
    """
    This function is a prototype for a plotting function. 
    I am not sure what is the best plots/subplots combination to show what we want to see, so for now 
    let's use this form.
    """

    data = numpy.loadtxt(config['filename_output'],ndmin=2)

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

def plotEllipticityBiasesHistogram():

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

    data = numpy.loadtxt(config['filename_output'])
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

def plotStatsGSParams():


    fiducial_value = eval('galsim.GSParams().%s' % config['gsparams']['vary_param'])
    # logging.info('changing gsparam %s from fiducial %f to %f',config['gsparams']['vary_param'],fiducial_value,new_value)


    # config['gsparams']['vary_param']
    vary_params = config['gsparams']

    nrows_subplot = len(vary_params)
    ncols_subplot = len(config['image']['observe_from'])

    for no,observe_from in enumerate(config['image']['observe_from']):

        for np,param in enumerate(vary_params):

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

            for nn,new_value in enumerate(param['grid']):

                filename_out = '%s.%s.%s.%d.cat' % (config['filename_output'],observe_from,param['name'],nn)
                results = numpy.loadtxt(filename_out)
                n_res = results.shape[0]

                logging.info('opened file %s with %d measurements for %s=%e',filename_out,n_res,config['gsparams']['vary_param'],nn)

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

            index_subplot += 1
            pylab.subplot(nrows_subplot,ncols_subplot,index_subplot)
            pylab.xscale('log')
            pylab.errorbar(vary_params,moments_E1_diff_mean,yerr=moments_E1_diff_stdm,fmt='b+-')
            pylab.errorbar(vary_params,moments_E2_diff_mean,yerr=moments_E2_diff_stdm,fmt='rx-')
            pylab.plot(vary_params,moments_E1_diff_medn,'b+--')
            pylab.plot(vary_params,moments_E2_diff_medn,'rx--')
            # pylab.savefig(os.path.join(dirname_figs,'gsparams_test.pdf'))
            # pylab.title('moments')
            pylab.ylabel('Ei_fft - Ei_photon')
            pylab.xlabel(config['gsparams']['vary_param'])
            pylab.xlim([min(vary_params)*0.5, max(vary_params)*1.5])
            pylab.show()




if __name__ == "__main__":


    description = 'Use data produced by photon_vs_fft.py to produce various plots.'

    # parse arguments
    parser = argparse.ArgumentParser(description=description, add_help=True)
    parser.add_argument('filename_config', type=str, help='yaml config file, see photon_vs_fft.yaml for example.')
    global args
    args = parser.parse_args()

    # set up logger
    logging.basicConfig(format="%(message)s", level=logging.DEBUG, stream=sys.stdout)
    logger = logging.getLogger("photon_vs_fft") 

    # load the configuration file
    filename_config = args.filename_config
    global config
    config = yaml.load(open(filename_config,'r'))

    # save a figure showing scatter on ellipticity fractional difference between shoot and fft
    # plotEllipticityBiases()
    # plotEllipticityBiasesHistogram()
    plotStatsGSParams()


