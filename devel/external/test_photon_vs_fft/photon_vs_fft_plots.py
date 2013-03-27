"""@file photon_vs_fft_plots.py 
Create various plot from data produced by photon_vs_fft.py.
"""

import logging
import galsim
import sys
import os
import numpy
import argparse
import yaml
import pdb
import pylab
import math
import matplotlib.pyplot as plt

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
    # E1_hsm_obs_fft
    # E2_hsm_obs_fft
    # E1_hsm_obs_photon
    # E2_hsm_obs_photon
    # E1_hsm_corr_fft
    # E2_hsm_corr_fft
    # E1_hsm_corr_photon
    # E2_hsm_corr_photon
    # moments_fft_sigma
    # moments_shoot_sigma
    # hsm_fft_sigma
    # hsm_shoot_sigma

    data = numpy.loadtxt(config['filename_output'],ndmin=2)
    n_test_gals = data.shape[0]
    n_bins = 25
    n_ticks= 3

    E1_fft=data[:,2]
    E2_fft=data[:,3]
    E1_photon=data[:,4]
    E2_photon=data[:,5]
    pylab.subplot(241)
    # pylab.hist((E1_fft-E1_photon)/E1_fft,label='E1',bins=n_bins)
    pylab.hist((E1_fft-E1_photon),label='E1',bins=n_bins)
    pylab.title('adaptive moments')
    pylab.xlabel('g1_fft - g1_photon')
    pylab.ylabel('count')
    pylab.xticks(numpy.linspace(min(pylab.xticks()[0]),max(pylab.xticks()[0]),n_ticks))

    
    pylab.subplot(245)
    # pylab.hist((E2_fft-E2_photon)/E2_fft,label='E1',bins=n_bins)
    pylab.hist((E2_fft-E2_photon),label='E2',bins=n_bins)
    pylab.xlabel('g2_fft - g2_photon')
    pylab.ylabel('count')
    pylab.xticks(numpy.linspace(min(pylab.xticks()[0]),max(pylab.xticks()[0]),n_ticks))

    E1_fft=data[:,6]
    E2_fft=data[:,7]
    E1_photon=data[:,8]
    E2_photon=data[:,9]
    pylab.subplot(242)
    # pylab.hist((E1_fft-E1_photon)/E1_fft,label='E1',bins=n_bins)
    pylab.hist((E1_fft-E1_photon),label='E1',bins=n_bins)
    pylab.title('HSM observed')
    pylab.xlabel('E1_fft - E1_photon')
    pylab.ylabel('count')
    pylab.xticks(numpy.linspace(min(pylab.xticks()[0]),max(pylab.xticks()[0]),n_ticks))

    
    pylab.subplot(246)
    # pylab.hist((E2_fft-E2_photon)/E2_fft,label='E1',bins=n_bins)
    pylab.hist((E2_fft-E2_photon),label='E2',bins=n_bins)
    pylab.xlabel('E2_fft - E2_photon')
    pylab.ylabel('count')
    pylab.xticks(numpy.linspace(min(pylab.xticks()[0]),max(pylab.xticks()[0]),n_ticks))



    E1_fft=data[:,10]
    E2_fft=data[:,11]
    E1_photon=data[:,12]
    E2_photon=data[:,13]
    pylab.subplot(243)
    # pylab.hist((E1_fft-E1_photon)/E1_fft,label='E1',bins=n_bins)
    pylab.hist((E1_fft-E1_photon),label='E1',bins=n_bins)
    pylab.title('HSM corrected')
    pylab.xlabel('E1_fft - E1_photon')
    pylab.ylabel('count')
    pylab.xticks(numpy.linspace(min(pylab.xticks()[0]),max(pylab.xticks()[0]),n_ticks))


    pylab.subplot(247)
    # pylab.hist((E2_fft-E2_photon)/E2_fft,label='E1',bins=n_bins)
    pylab.hist((E2_fft-E2_photon),label='E2',bins=n_bins)           
    pylab.xlabel('E2_fft - E2_photon')
    pylab.ylabel('count')
    pylab.xticks(numpy.linspace(min(pylab.xticks()[0]),max(pylab.xticks()[0]),n_ticks))


    # plot moments differnces

    sigma_moments_fft=data[:,14]
    sigma_moments_photon=data[:,15]
    sigma_hsm_fft=data[:,16]
    sigma_hsm_photon=data[:,17]

    pylab.subplot(244)
    pylab.hist((sigma_moments_fft-sigma_moments_photon),label='sigma',bins=n_bins)
    pylab.title('moments')
    pylab.xlabel('sigma_fft - sigma_photon')
    pylab.ylabel('count')
    pylab.xticks(numpy.linspace(min(pylab.xticks()[0]),max(pylab.xticks()[0]),n_ticks))

    pylab.subplot(248)
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
    plotEllipticityBiasesHistogram()



